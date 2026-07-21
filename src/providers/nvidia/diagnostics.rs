use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result};

use crate::{
    model::{
        command::CommandSpec,
        device::GpuVendor,
        environment::{
            Confidence, DiagnosticCause, DiagnosticCheck, DiagnosticId, DiagnosticSection,
            DiagnosticStatus, Diagnostics, DriverInstallation, DriverRuntimeState, Fix, FixId,
            FixPlan,
        },
        system::OsInfo,
    },
    platform::{os, package_manager},
};

use super::{
    compatibility::{self, Compatibility},
    driver,
    gpu::{self, NvidiaGpu},
    policy, recipe, repository, runtime, state,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoctorProfile {
    ModelTraining,
    CudaDevelopment,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandEvidence {
    pub exists: bool,
    pub succeeded: bool,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaSymlinkState {
    Missing,
    Valid(PathBuf),
    Broken(PathBuf),
    NotSymlink,
    Unavailable(String),
}

#[derive(Clone, Debug)]
pub struct NvidiaEvidence {
    pub os: OsInfo,
    pub gpus: Vec<NvidiaGpu>,
    pub driver: DriverInstallation,
    pub nvidia_module_loaded: bool,
    pub nvidia_smi: CommandEvidence,
    pub kernel_release: String,
    pub matching_kernel_headers: bool,
    pub secure_boot_enabled: Option<bool>,
    pub dkms_status: Option<String>,
    pub driver_version: Option<String>,
    pub driver_module_signed: bool,
    pub toolkit_packages: Vec<String>,
    pub managed_toolkit_version: Option<String>,
    pub managed_nvcc: CommandEvidence,
    pub nvcc: CommandEvidence,
    pub nvcc_path: Option<String>,
    pub nvcc_version: Option<String>,
    pub cuda_symlink: CudaSymlinkState,
    pub installed_cuda_versions: Vec<String>,
}

impl NvidiaEvidence {
    fn toolkit_installed(&self) -> bool {
        !self.toolkit_packages.is_empty()
    }
}

pub fn collect_evidence() -> Result<NvidiaEvidence> {
    let os = os::detect()?;
    let kernel_release = command_stdout("uname", &["-r"])
        .context("could not determine the running kernel release")?;
    let status = state::inspect(&os)?;
    let nvidia_smi = command_evidence(
        "nvidia-smi",
        &["--query-gpu=driver_version", "--format=csv,noheader"],
    );
    let mut nvcc = command_evidence("nvcc", &["--version"]);
    if !nvcc.exists {
        nvcc = command_evidence("/usr/local/cuda/bin/nvcc", &["--version"]);
    }
    let installed_cuda_versions = installed_cuda_versions();
    let managed = status.toolkits.first();
    let managed_nvcc = managed
        .and_then(|toolkit| toolkit.executable_path.as_deref())
        .map(|path| command_evidence(path, &["--version"]))
        .unwrap_or_default();
    Ok(NvidiaEvidence {
        gpus: gpu::detect()?,
        driver: status.driver,
        nvidia_module_loaded: Path::new("/sys/module/nvidia").exists(),
        matching_kernel_headers: Path::new("/lib/modules")
            .join(&kernel_release)
            .join("build")
            .exists(),
        secure_boot_enabled: driver::secure_boot_enabled(),
        dkms_status: command_optional_stdout("dkms", &["status"]),
        driver_version: status.driver_version.clone().or_else(|| {
            status
                .driver_module
                .as_ref()
                .and_then(|module| module.version.clone())
        }),
        driver_module_signed: status
            .driver_module
            .as_ref()
            .is_some_and(|module| module.signer.is_some() || module.signature_id.is_some()),
        toolkit_packages: managed
            .map(|toolkit| toolkit.packages.clone())
            .unwrap_or_default(),
        managed_toolkit_version: managed.and_then(|toolkit| toolkit.version.clone()),
        managed_nvcc,
        nvcc_path: status
            .active_toolkit
            .as_ref()
            .and_then(|toolkit| toolkit.executable_path.clone()),
        nvcc_version: parse_nvcc_version(&nvcc.stdout).map(str::to_owned),
        nvcc,
        cuda_symlink: cuda_symlink_state(Path::new("/usr/local/cuda")),
        installed_cuda_versions,
        kernel_release,
        nvidia_smi,
        os,
    })
}

pub fn detect(profile: DoctorProfile) -> Result<Diagnostics> {
    diagnose(collect_evidence()?, profile)
}

pub fn diagnose(e: NvidiaEvidence, profile: DoctorProfile) -> Result<Diagnostics> {
    let checks = checks(&e, profile);
    let fix_plan = fix_plan(&e, &checks, profile)?;
    Ok(Diagnostics {
        vendor: GpuVendor::Nvidia,
        checks,
        fix_plan,
    })
}

pub fn checks(e: &NvidiaEvidence, profile: DoctorProfile) -> Vec<DiagnosticCheck> {
    let runtime_state = runtime_state(e);
    let gpu_ok = !e.gpus.is_empty();
    let repository_state = repository::resolve(&e.os);
    let os_resolution = repository::resolve(&e.os)
        .and_then(|_| recipe::validate_release(&e.os))
        .and_then(|_| e.os.ensure_driver_installable("NVIDIA"))
        .and_then(|_| {
            policy::resolve(
                &e.os,
                &e.gpus,
                super::driver::DriverPreference::Auto,
                e.managed_toolkit_version.as_deref(),
                profile == DoctorProfile::CudaDevelopment,
            )
            .map(|_| ())
        });
    let mut result = vec![check(
        DiagnosticId::NvidiaGpu,
        DiagnosticSection::Hardware,
        "NVIDIA GPU detected",
        if gpu_ok {
            DiagnosticStatus::Pass
        } else {
            DiagnosticStatus::Error
        },
        vec![if gpu_ok {
            format!("{} NVIDIA GPU(s) detected", e.gpus.len())
        } else {
            "No NVIDIA PCI device found".into()
        }],
        (!gpu_ok).then(|| "No NVIDIA GPU was detected by lspci or sysfs.".into()),
        vec![],
        vec![FixId::InspectHardware],
    )];
    let release_warning = repository_state.as_ref().ok().and_then(|repository| {
        (!repository.nvidia_validated).then(|| {
            format!(
                "Repository-compatible via {}, but this exact release is not NVIDIA-validated.",
                repository.distro
            )
        })
    });
    let os_problem = os_resolution
        .as_ref()
        .err()
        .map(ToString::to_string)
        .or_else(|| release_warning.clone());
    result.push(check(
        DiagnosticId::OperatingSystem,
        DiagnosticSection::OperatingSystem,
        "Supported OS/GPU policy",
        if os_resolution.is_err() {
            DiagnosticStatus::Error
        } else if release_warning.is_some() {
            DiagnosticStatus::Warning
        } else {
            DiagnosticStatus::Pass
        },
        vec![
            format!("{} ({})", e.os.display_name(), e.os.architecture),
            repository_state.as_ref().map_or_else(
                |_| "repository target: unavailable".into(),
                |repository| {
                    format!(
                        "repository target: {}; family: {}; NVIDIA validated: {}",
                        repository.distro,
                        repository.family,
                        yes_no(repository.nvidia_validated)
                    )
                },
            ),
        ],
        os_problem,
        vec![DiagnosticId::NvidiaGpu],
        vec![],
    ));
    result.push(check(DiagnosticId::KernelHeaders, DiagnosticSection::OperatingSystem, "Headers for the running kernel", if e.matching_kernel_headers { DiagnosticStatus::Pass } else { DiagnosticStatus::Warning }, vec![format!("kernel {}: headers {}", e.kernel_release, yes_no(e.matching_kernel_headers))], (!e.matching_kernel_headers).then(|| "Matching kernel development packages must be installed before DKMS builds the driver.".into()), vec![], vec![FixId::InstallKernelHeaders]));
    result.push(check(
        DiagnosticId::SecureBoot,
        DiagnosticSection::OperatingSystem,
        "Secure Boot state",
        match e.secure_boot_enabled {
            Some(false) => DiagnosticStatus::Pass,
            _ => DiagnosticStatus::Warning,
        },
        vec![format!(
            "Secure Boot: {}",
            match e.secure_boot_enabled {
                Some(true) => "enabled",
                Some(false) => "disabled",
                None => "unknown",
            }
        )],
        e.secure_boot_enabled
            .is_none()
            .then(|| "Secure Boot state could not be determined.".into()),
        vec![],
        vec![],
    ));
    let (driver_status, driver_problem) = match &e.driver {
        DriverInstallation::Managed { .. } => (DiagnosticStatus::Pass, None),
        DriverInstallation::Missing => (
            DiagnosticStatus::Error,
            Some("No managed NVIDIA driver package installation was detected.".into()),
        ),
        DriverInstallation::BrokenManaged { .. } => (
            DiagnosticStatus::Error,
            Some("NVIDIA packages are installed, but the driver runtime is broken.".into()),
        ),
        DriverInstallation::Unmanaged {
            working: true,
            evidence,
        } => (
            DiagnosticStatus::Warning,
            Some(format!(
                "A working unmanaged driver is present (evidence: {}); arc will not overwrite it with repository packages.",
                evidence
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )),
        ),
        DriverInstallation::Unmanaged {
            working: false,
            evidence,
        } => (
            DiagnosticStatus::Error,
            Some(format!(
                "An unmanaged driver installation appears broken (evidence: {}) and must be removed with its original installer.",
                evidence
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )),
        ),
    };
    result.push(check(
        DiagnosticId::DriverPackage,
        DiagnosticSection::Driver,
        "NVIDIA driver installation method",
        driver_status,
        vec![e.driver.description()],
        driver_problem,
        vec![DiagnosticId::NvidiaGpu],
        match e.driver {
            DriverInstallation::BrokenManaged { .. } => vec![FixId::RepairManagedDriver],
            _ => vec![FixId::InstallDriver],
        },
    ));
    push_dependent(
        &mut result,
        check(
            DiagnosticId::DriverModule,
            DiagnosticSection::Driver,
            "NVIDIA kernel module loaded",
            if e.nvidia_module_loaded {
                DiagnosticStatus::Pass
            } else {
                DiagnosticStatus::Error
            },
            vec![format!(
                "/sys/module/nvidia: {}; DKMS: {}",
                if e.nvidia_module_loaded {
                    "present"
                } else {
                    "missing"
                },
                e.dkms_status.as_deref().unwrap_or("unavailable")
            )],
            runtime::module_problem(runtime_state).map(str::to_owned),
            vec![DiagnosticId::DriverPackage],
            match runtime_state {
                DriverRuntimeState::Operational => vec![],
                DriverRuntimeState::RebootLikelyRequired => vec![FixId::RebootThenRecheck],
                DriverRuntimeState::DkmsModuleMissing => {
                    vec![FixId::RebuildDkms]
                }
                DriverRuntimeState::SecureBootBlocked => vec![FixId::ResolveSecureBoot],
                DriverRuntimeState::Failed => vec![FixId::DebugDriver],
            },
        ),
    );
    push_dependent(
        &mut result,
        check(
            DiagnosticId::NvidiaSmi,
            DiagnosticSection::Driver,
            "nvidia-smi operational",
            if e.nvidia_smi.exists && e.nvidia_smi.succeeded {
                DiagnosticStatus::Pass
            } else {
                DiagnosticStatus::Error
            },
            command_evidence_lines("nvidia-smi", &e.nvidia_smi),
            (!e.nvidia_smi.succeeded)
                .then(|| "nvidia-smi cannot communicate with the driver.".into()),
            vec![DiagnosticId::DriverModule],
            vec![FixId::DebugDriver],
        ),
    );
    let mismatch = version_mismatch(e);
    push_dependent(
        &mut result,
        check(
            DiagnosticId::DriverLibrary,
            DiagnosticSection::Driver,
            "Driver and userspace libraries match",
            if mismatch {
                DiagnosticStatus::Error
            } else {
                DiagnosticStatus::Pass
            },
            e.driver_version
                .as_ref()
                .map(|v| format!("driver version: {v}"))
                .into_iter()
                .collect(),
            mismatch.then(|| "NVML reports a driver/library version mismatch.".into()),
            vec![DiagnosticId::DriverPackage],
            vec![FixId::ReinstallDriverLibraries, FixId::Reboot],
        ),
    );

    let toolkit_present = e.toolkit_installed();
    let missing_status = if profile == DoctorProfile::CudaDevelopment {
        DiagnosticStatus::Error
    } else {
        DiagnosticStatus::Warning
    };
    result.push(check(
        DiagnosticId::ToolkitInstall,
        DiagnosticSection::CudaToolkit,
        "CUDA Toolkit installation",
        if toolkit_present {
            DiagnosticStatus::Pass
        } else {
            missing_status
        },
        vec![format!(
            "system package(s): {}; /usr/local versions: {}",
            list_or_none(&e.toolkit_packages),
            list_or_none(&e.installed_cuda_versions)
        )],
        (!toolkit_present).then(|| {
            if profile == DoctorProfile::CudaDevelopment {
                "CUDA development requires a Toolkit.".into()
            } else {
                "No Toolkit detected; this is normal for frameworks that bundle a CUDA runtime."
                    .into()
            }
        }),
        vec![],
        if profile == DoctorProfile::CudaDevelopment {
            vec![FixId::InstallToolkit]
        } else {
            vec![]
        },
    ));
    push_dependent(
        &mut result,
        check(
            DiagnosticId::Nvcc,
            DiagnosticSection::CudaToolkit,
            "nvcc available",
            if e.managed_nvcc.exists
                && e.managed_nvcc.succeeded
                && e.managed_toolkit_version.is_some()
            {
                DiagnosticStatus::Pass
            } else {
                DiagnosticStatus::Error
            },
            {
                let mut lines = command_evidence_lines("system-managed nvcc", &e.managed_nvcc);
                lines.push(format!(
                    "active nvcc: {}{}",
                    e.nvcc_path.as_deref().unwrap_or("not found on PATH"),
                    e.nvcc_version
                        .as_deref()
                        .map(|version| format!(" (CUDA {version})"))
                        .unwrap_or_default()
                ));
                lines.extend(command_evidence_lines("active nvcc", &e.nvcc));
                lines
            },
            (!(e.managed_nvcc.exists
                && e.managed_nvcc.succeeded
                && e.managed_toolkit_version.is_some()))
            .then(|| "System Toolkit packages are present, but their nvcc is missing or broken; an unrelated active nvcc does not satisfy this check.".into()),
            vec![DiagnosticId::ToolkitInstall],
            vec![FixId::InstallToolkit, FixId::DebugToolkit],
        ),
    );
    push_dependent(
        &mut result,
        check(
            DiagnosticId::CudaSymlink,
            DiagnosticSection::CudaToolkit,
            "/usr/local/cuda configuration",
            match e.cuda_symlink {
                CudaSymlinkState::Valid(_) => DiagnosticStatus::Pass,
                CudaSymlinkState::Missing if e.installed_cuda_versions.len() <= 1 => {
                    DiagnosticStatus::Warning
                }
                _ => DiagnosticStatus::Error,
            },
            vec![cuda_symlink_description(&e.cuda_symlink)],
            (!matches!(e.cuda_symlink, CudaSymlinkState::Valid(_)))
                .then(|| "/usr/local/cuda does not point to a valid Toolkit.".into()),
            vec![DiagnosticId::ToolkitInstall],
            vec![FixId::RepairCudaSymlink],
        ),
    );
    let compatibility = e
        .driver_version
        .as_deref()
        .zip(e.managed_toolkit_version.as_deref())
        .and_then(|(driver, toolkit)| compatibility::evaluate(driver, toolkit));
    push_dependent(&mut result, check(DiagnosticId::DriverToolkitCompatibility, DiagnosticSection::CudaToolkit, "Driver supports CUDA Toolkit", match compatibility { Some(Compatibility::Incompatible) => DiagnosticStatus::Error, Some(Compatibility::MinorVersionCompatible) => DiagnosticStatus::Warning, _ => DiagnosticStatus::Pass }, vec![format!("driver: {}; system Toolkit: {}; compatibility: {:?}", e.driver_version.as_deref().unwrap_or("unknown"), e.managed_toolkit_version.as_deref().unwrap_or("unknown"), compatibility)], (compatibility == Some(Compatibility::Incompatible)).then(|| "The complete driver version is below the Toolkit's minimum compatibility version.".into()), vec![DiagnosticId::NvidiaSmi, DiagnosticId::Nvcc], vec![FixId::UpgradeDriver, FixId::Reboot]));
    result
}

fn push_dependent(result: &mut Vec<DiagnosticCheck>, mut check: DiagnosticCheck) {
    if check.dependencies.iter().any(|id| {
        result
            .iter()
            .any(|prior| prior.id == *id && prior.status != DiagnosticStatus::Pass)
    }) {
        check.status = DiagnosticStatus::Skipped;
        check.problem = Some("Skipped because a prerequisite check did not pass.".into());
        check.recommended_fixes.clear();
    }
    result.push(check);
}
#[allow(clippy::too_many_arguments)]
fn check(
    id: DiagnosticId,
    section: DiagnosticSection,
    name: &str,
    status: DiagnosticStatus,
    evidence: Vec<String>,
    problem: Option<String>,
    dependencies: Vec<DiagnosticId>,
    fixes: Vec<FixId>,
) -> DiagnosticCheck {
    DiagnosticCheck {
        id,
        section,
        name: name.into(),
        status,
        evidence,
        problem,
        dependencies,
        recommended_fixes: fixes,
    }
}

pub fn fix_plan(
    e: &NvidiaEvidence,
    checks: &[DiagnosticCheck],
    profile: DoctorProfile,
) -> Result<FixPlan> {
    let failed = |id| {
        checks
            .iter()
            .any(|c| c.id == id && c.status == DiagnosticStatus::Error)
    };
    let mut causes = Vec::new();
    let runtime_state = runtime_state(e);
    if failed(DiagnosticId::NvidiaGpu) {
        causes.push(cause(
            "The NVIDIA GPU is not visible",
            vec![FixId::InspectHardware],
        ));
    }
    if failed(DiagnosticId::OperatingSystem) {
        causes.push(cause("The OS/GPU combination is unsupported", vec![]));
    }
    if failed(DiagnosticId::DriverPackage) {
        let (title, fixes) = match e.driver {
            DriverInstallation::BrokenManaged { .. } => (
                "The managed NVIDIA driver installation is broken",
                vec![FixId::RepairManagedDriver, FixId::Reboot],
            ),
            DriverInstallation::Missing => (
                "The NVIDIA driver installation is missing",
                vec![FixId::InstallDriver],
            ),
            DriverInstallation::Unmanaged { .. } => (
                "The unmanaged NVIDIA driver needs its original maintenance method",
                vec![FixId::InstallDriver],
            ),
            DriverInstallation::Managed { .. } => unreachable!(),
        };
        causes.push(cause(title, fixes));
    }
    if matches!(e.driver, DriverInstallation::Managed { .. })
        && !e.nvidia_module_loaded
        && !e.matching_kernel_headers
    {
        causes.push(cause(
            "Headers for the running kernel are missing",
            vec![
                FixId::InstallKernelHeaders,
                FixId::RebuildDkms,
                FixId::Reboot,
            ],
        ));
    }
    if matches!(e.driver, DriverInstallation::Managed { .. }) && !e.nvidia_module_loaded {
        match runtime_state {
            DriverRuntimeState::RebootLikelyRequired => causes.push(cause(
                "The NVIDIA driver was installed successfully and a reboot is likely required",
                vec![FixId::RebootThenRecheck],
            )),
            DriverRuntimeState::DkmsModuleMissing if e.matching_kernel_headers => {
                causes.push(cause(
                    "The matching NVIDIA DKMS module is missing for the running kernel",
                    vec![FixId::RebuildDkms],
                ))
            }
            DriverRuntimeState::SecureBootBlocked => causes.push(cause(
                "Secure Boot is blocking the NVIDIA kernel module",
                vec![FixId::ResolveSecureBoot],
            )),
            DriverRuntimeState::Failed => causes.push(cause(
                "The NVIDIA module still fails to load",
                vec![FixId::DebugDriver],
            )),
            _ => {}
        }
    }
    if version_mismatch(e) && matches!(e.driver, DriverInstallation::Managed { .. }) {
        causes.push(cause(
            "NVIDIA driver and userspace libraries do not match",
            vec![FixId::ReinstallDriverLibraries, FixId::Reboot],
        ));
    }
    if e.toolkit_installed() && failed(DiagnosticId::Nvcc) {
        causes.push(cause(
            "The CUDA Toolkit is partially installed or broken",
            vec![FixId::InstallToolkit, FixId::DebugToolkit],
        ));
    }
    if profile == DoctorProfile::CudaDevelopment && !e.toolkit_installed() {
        causes.push(cause(
            "CUDA development was requested but the Toolkit is missing",
            vec![FixId::InstallToolkit],
        ));
    }
    if failed(DiagnosticId::DriverToolkitCompatibility) {
        causes.push(cause(
            "The NVIDIA driver is incompatible with the CUDA Toolkit",
            vec![FixId::UpgradeDriver, FixId::Reboot],
        ));
    }
    if failed(DiagnosticId::CudaSymlink) && e.toolkit_installed() {
        causes.push(cause(
            "The system CUDA Toolkit symlink is invalid",
            vec![FixId::RepairCudaSymlink],
        ));
    }
    Ok(FixPlan::new(causes, available_fixes(e)?))
}
fn cause(title: &str, fixes: Vec<FixId>) -> DiagnosticCause {
    DiagnosticCause {
        title: title.into(),
        confidence: Confidence::High,
        evidence: vec![],
        fixes,
    }
}
fn available_fixes(e: &NvidiaEvidence) -> Result<Vec<Fix>> {
    let prerequisites = recipe::prerequisites(&e.os, &e.kernel_release).unwrap_or_default();
    let (install_driver_commands, install_driver_steps) = match &e.driver {
        DriverInstallation::Unmanaged { evidence, .. } => (
            vec![],
            vec![format!(
                "Do not use arc to overwrite this installation (evidence: {}). Repair or remove it with the original installation method.",
                evidence
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )],
        ),
        _ => (vec![arc_install_command()], vec![]),
    };
    let repair_commands = managed_driver_repair_commands(e, &prerequisites);
    let (symlink_commands, symlink_steps) = cuda_symlink_repair(e);
    Ok(vec![
        fix_with_manual(
            FixId::RebootThenRecheck,
            "Reboot, then verify the driver again",
            vec![],
            vec![
                "Run `sudo reboot`.".into(),
                "If the module still does not load after rebooting, run `sudo modprobe nvidia` and inspect the kernel logs reported by `arc doctor`.".into(),
            ],
            1,
        ),
        fix(
            FixId::InspectHardware,
            "Verify that the NVIDIA GPU is visible",
            vec![CommandSpec::new("lspci", ["-nnk", "-d", "10de:"])],
            5,
        ),
        fix(
            FixId::InstallKernelHeaders,
            "Install exact prerequisites for the running kernel",
            prerequisites,
            10,
        ),
        fix_with_manual(
            FixId::InstallDriver,
            "Install a missing managed driver or handle an unmanaged driver manually",
            install_driver_commands,
            install_driver_steps,
            20,
        ),
        fix(
            FixId::RepairManagedDriver,
            "Reinstall the exact managed NVIDIA packages and rebuild the running-kernel module",
            repair_commands,
            20,
        ),
        fix(
            FixId::UpgradeDriver,
            "Upgrade the managed NVIDIA driver",
            vec![arc_install_command()],
            20,
        ),
        fix(
            FixId::ReinstallDriverLibraries,
            "Reinstall the managed NVIDIA packages",
            managed_package_reinstall_commands(&e.os, driver_packages(&e.driver)),
            30,
        ),
        fix_with_manual(
            FixId::RebuildDkms,
            "Rebuild NVIDIA DKMS modules",
            vec![CommandSpec::sudo(
                "dkms",
                ["autoinstall", "-k", &e.kernel_release],
            )],
            vec!["If DKMS cannot build the matching module, reinstall the managed NVIDIA driver packages.".into()],
            40,
        ),
        fix(
            FixId::InstallToolkit,
            "Install or repair the CUDA Toolkit",
            vec![arc_install_command()],
            50,
        ),
        fix_with_manual(
            FixId::RepairCudaSymlink,
            "Repair /usr/local/cuda",
            symlink_commands,
            symlink_steps,
            60,
        ),
        fix(
            FixId::DebugDriver,
            "Try loading the module and inspect the kernel logs",
            vec![
                CommandSpec::sudo("modprobe", ["nvidia"]),
                CommandSpec::new("journalctl", ["-k", "-b", "-g", "NVRM|nvidia|nouveau"]),
                CommandSpec::new("dkms", ["status"]),
            ],
            80,
        ),
        fix_with_manual(
            FixId::ResolveSecureBoot,
            "Enroll the NVIDIA module signing key or disable Secure Boot",
            vec![],
            vec!["Enroll the distribution/NVIDIA module signing key (MOK), or disable Secure Boot in firmware, then reboot.".into()],
            85,
        ),
        fix(
            FixId::DebugToolkit,
            "Inspect Toolkit paths",
            vec![
                CommandSpec::new("readlink", ["-f", "/usr/local/cuda"]),
                CommandSpec::new("nvcc", ["--version"]),
            ],
            80,
        ),
        fix(
            FixId::Reboot,
            "Reboot to load the repaired driver",
            vec![CommandSpec::sudo("systemctl", ["reboot"])],
            90,
        ),
    ])
}

fn arc_install_command() -> CommandSpec {
    CommandSpec::new("arc", ["install"])
}

fn driver_packages(driver: &DriverInstallation) -> &[String] {
    match driver {
        DriverInstallation::Managed { packages, .. }
        | DriverInstallation::BrokenManaged { packages, .. } => packages,
        DriverInstallation::Missing | DriverInstallation::Unmanaged { .. } => &[],
    }
}

fn managed_driver_repair_commands(
    e: &NvidiaEvidence,
    prerequisites: &[CommandSpec],
) -> Vec<CommandSpec> {
    let DriverInstallation::BrokenManaged { packages, .. } = &e.driver else {
        return vec![];
    };
    let mut commands = prerequisites.to_vec();
    commands.push(package_manager::refresh_command(e.os.package_manager()));
    commands.extend(managed_package_reinstall_commands(&e.os, packages));
    if e.dkms_status.is_some() || packages.iter().any(|package| package.contains("dkms")) {
        commands.push(CommandSpec::sudo(
            "dkms",
            ["autoinstall", "-k", &e.kernel_release],
        ));
    }
    commands.push(CommandSpec::new("modinfo", ["nvidia"]));
    commands
}

fn managed_package_reinstall_commands(os: &OsInfo, packages: &[String]) -> Vec<CommandSpec> {
    package_manager::reinstall_command(os.package_manager(), packages)
        .into_iter()
        .collect()
}

fn cuda_symlink_repair(e: &NvidiaEvidence) -> (Vec<CommandSpec>, Vec<String>) {
    let Some(version) = e.installed_cuda_versions.last() else {
        return (
            vec![],
            vec!["Select a directory installed by the system CUDA packages under /usr/local/cuda-VERSION, then create /usr/local/cuda as a symlink to that exact directory.".into()],
        );
    };
    let target = format!("/usr/local/cuda-{version}");
    (
        vec![
            CommandSpec::sudo("ln", ["-sfn", &target, "/usr/local/cuda"]),
            CommandSpec::new("test", ["-x", "/usr/local/cuda/bin/nvcc"]),
        ],
        vec![],
    )
}

fn fix(id: FixId, title: &str, commands: Vec<CommandSpec>, order: u16) -> Fix {
    fix_with_manual(id, title, commands, vec![], order)
}

fn fix_with_manual(
    id: FixId,
    title: &str,
    commands: Vec<CommandSpec>,
    manual_steps: Vec<String>,
    order: u16,
) -> Fix {
    Fix {
        id,
        title: title.into(),
        commands,
        manual_steps,
        order,
    }
}

fn command_evidence(program: &str, args: &[&str]) -> CommandEvidence {
    match Command::new(program).args(args).output() {
        Ok(output) => CommandEvidence {
            exists: true,
            succeeded: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).trim().into(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().into(),
        },
        Err(error) => CommandEvidence {
            stderr: error.to_string(),
            ..Default::default()
        },
    }
}
fn command_stdout(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    anyhow::ensure!(output.status.success(), "{program} failed");
    Ok(String::from_utf8_lossy(&output.stdout).trim().into())
}
fn command_optional_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().into())
}
fn installed_cuda_versions() -> Vec<String> {
    let Ok(entries) = fs::read_dir("/usr/local") else {
        return vec![];
    };
    let mut values = entries
        .flatten()
        .filter_map(|e| {
            e.file_name()
                .to_str()
                .and_then(|n| n.strip_prefix("cuda-"))
                .filter(|v| v.starts_with(|c: char| c.is_ascii_digit()))
                .map(Into::into)
        })
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}
fn cuda_symlink_state(path: &Path) -> CudaSymlinkState {
    match fs::symlink_metadata(path) {
        Ok(m) if !m.file_type().is_symlink() => CudaSymlinkState::NotSymlink,
        Ok(_) => match fs::read_link(path) {
            Ok(target) => {
                let resolved = if target.is_absolute() {
                    target.clone()
                } else {
                    path.parent().unwrap_or(Path::new("/")).join(&target)
                };
                if resolved.exists() {
                    CudaSymlinkState::Valid(target)
                } else {
                    CudaSymlinkState::Broken(target)
                }
            }
            Err(e) => CudaSymlinkState::Unavailable(e.to_string()),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => CudaSymlinkState::Missing,
        Err(e) => CudaSymlinkState::Unavailable(e.to_string()),
    }
}
fn parse_nvcc_version(output: &str) -> Option<&str> {
    let (_, rest) = output.split_once("release ")?;
    rest.split(|c: char| c == ',' || c.is_whitespace())
        .find(|p| !p.is_empty())
}
fn version_mismatch(e: &NvidiaEvidence) -> bool {
    format!("{}\n{}", e.nvidia_smi.stdout, e.nvidia_smi.stderr)
        .to_ascii_lowercase()
        .contains("driver/library version mismatch")
}
fn runtime_state(e: &NvidiaEvidence) -> DriverRuntimeState {
    runtime::classify(runtime::RuntimeEvidence {
        driver: &e.driver,
        driver_version: e.driver_version.as_deref(),
        module_loaded: e.nvidia_module_loaded,
        runtime_operational: e.nvidia_smi.exists && e.nvidia_smi.succeeded,
        kernel_release: Some(&e.kernel_release),
        dkms_status: e.dkms_status.as_deref(),
        secure_boot_enabled: e.secure_boot_enabled,
        module_signed: e.driver_module_signed,
    })
}
fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}
fn list_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "none".into()
    } else {
        values.join(", ")
    }
}
fn command_evidence_lines(name: &str, value: &CommandEvidence) -> Vec<String> {
    vec![
        format!(
            "{name}: {}",
            if !value.exists {
                "not found"
            } else if value.succeeded {
                "succeeded"
            } else {
                "failed"
            }
        ),
        format!("stderr: {}", value.stderr),
    ]
}
fn cuda_symlink_description(state: &CudaSymlinkState) -> String {
    match state {
        CudaSymlinkState::Missing => "/usr/local/cuda: missing".into(),
        CudaSymlinkState::Valid(p) => format!("/usr/local/cuda -> {} (valid)", p.display()),
        CudaSymlinkState::Broken(p) => format!("/usr/local/cuda -> {} (broken)", p.display()),
        CudaSymlinkState::NotSymlink => "/usr/local/cuda is not a symlink".into(),
        CudaSymlinkState::Unavailable(e) => format!("/usr/local/cuda unavailable: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::gpu::Generation;
    use super::*;
    use crate::model::{
        environment::{DriverFlavorState, DriverPackageScope},
        system::Distribution,
    };
    fn evidence() -> NvidiaEvidence {
        NvidiaEvidence {
            os: OsInfo {
                distribution: Distribution::Ubuntu,
                name: "Ubuntu".into(),
                version_id: "24.04".into(),
                architecture: "x86_64".into(),
                is_wsl: false,
            },
            gpus: vec![NvidiaGpu {
                name: "GPU".into(),
                pci_device_id: None,
                generation: Generation::TuringOrNewer,
            }],
            driver: DriverInstallation::Managed {
                flavor: DriverFlavorState::Open,
                scope: DriverPackageScope::ComputeOnly,
                branch: None,
                packages: vec![],
            },
            nvidia_module_loaded: true,
            nvidia_smi: CommandEvidence {
                exists: true,
                succeeded: true,
                stdout: "570.26".into(),
                stderr: "".into(),
            },
            kernel_release: "6.8.0-generic".into(),
            matching_kernel_headers: true,
            secure_boot_enabled: Some(false),
            dkms_status: None,
            driver_version: Some("570.26".into()),
            driver_module_signed: true,
            toolkit_packages: vec![],
            managed_toolkit_version: None,
            managed_nvcc: CommandEvidence::default(),
            nvcc: CommandEvidence::default(),
            nvcc_path: None,
            nvcc_version: None,
            cuda_symlink: CudaSymlinkState::Missing,
            installed_cuda_versions: vec![],
        }
    }

    #[test]
    fn nvidia_validated_release_passes_without_arc_coverage_evidence() {
        let mut e = evidence();
        e.os.version_id = "22.04".into();
        let checks = checks(&e, DoctorProfile::ModelTraining);
        let os = checks
            .iter()
            .find(|check| check.id == DiagnosticId::OperatingSystem)
            .unwrap();

        assert_eq!(os.status, DiagnosticStatus::Pass);
        assert!(os.problem.is_none());
        assert!(os.evidence.iter().all(|line| !line.contains("arc")));
    }

    #[test]
    fn repository_compatible_but_unvalidated_release_still_warns() {
        let mut e = evidence();
        e.os.distribution = Distribution::Rhel;
        e.os.name = "RHEL".into();
        e.os.version_id = "9.9".into();
        let checks = checks(&e, DoctorProfile::ModelTraining);
        let os = checks
            .iter()
            .find(|check| check.id == DiagnosticId::OperatingSystem)
            .unwrap();

        assert_eq!(os.status, DiagnosticStatus::Warning);
        assert_eq!(
            os.problem.as_deref(),
            Some(
                "Repository-compatible via rhel9, but this exact release is not NVIDIA-validated."
            )
        );
    }
    #[test]
    fn missing_toolkit_is_profile_aware() {
        let e = evidence();
        assert!(
            !diagnose(e.clone(), DoctorProfile::ModelTraining)
                .unwrap()
                .has_errors()
        );
        assert!(
            diagnose(e, DoctorProfile::CudaDevelopment)
                .unwrap()
                .has_errors()
        );
    }
    #[test]
    fn compatibility_warning_is_not_an_error() {
        let mut e = evidence();
        e.toolkit_packages = vec!["cuda-toolkit-12-8".into()];
        e.managed_toolkit_version = Some("12.8".into());
        e.managed_nvcc = CommandEvidence {
            exists: true,
            succeeded: true,
            stdout: "release 12.8,".into(),
            stderr: "".into(),
        };
        e.installed_cuda_versions = vec!["12.8".into()];
        e.nvcc = CommandEvidence {
            exists: true,
            succeeded: true,
            stdout: "release 12.8,".into(),
            stderr: "".into(),
        };
        e.nvcc_version = Some("12.8".into());
        e.driver_version = Some("525.60.13".into());
        assert_eq!(
            checks(&e, DoctorProfile::CudaDevelopment)
                .iter()
                .find(|c| c.id == DiagnosticId::DriverToolkitCompatibility)
                .unwrap()
                .status,
            DiagnosticStatus::Warning
        );
    }

    #[test]
    fn active_conda_nvcc_is_not_a_system_toolkit() {
        let mut e = evidence();
        e.nvcc = CommandEvidence {
            exists: true,
            succeeded: true,
            stdout: "Cuda compilation tools, release 12.8,".into(),
            stderr: "".into(),
        };
        e.nvcc_path = Some("/opt/conda/envs/ml/bin/nvcc".into());
        e.nvcc_version = Some("12.8".into());
        let diagnostics = diagnose(e, DoctorProfile::CudaDevelopment).unwrap();
        let toolkit = diagnostics
            .checks
            .iter()
            .find(|check| check.id == DiagnosticId::ToolkitInstall)
            .unwrap();
        assert_eq!(toolkit.status, DiagnosticStatus::Error);
        assert!(toolkit.evidence[0].contains("system package(s): none"));
    }

    #[test]
    fn freshly_installed_driver_keeps_module_check_failed_and_reboots_first() {
        let mut e = evidence();
        e.nvidia_module_loaded = false;
        e.nvidia_smi.succeeded = false;
        e.dkms_status = Some("nvidia/570.26, 6.8.0-generic, x86_64: installed".into());
        let diagnostics = diagnose(e, DoctorProfile::ModelTraining).unwrap();
        let module = diagnostics
            .checks
            .iter()
            .find(|check| check.id == DiagnosticId::DriverModule)
            .unwrap();
        assert_eq!(module.status, DiagnosticStatus::Error);
        assert!(
            module
                .problem
                .as_deref()
                .unwrap()
                .contains("reboot is likely required")
        );
        assert_eq!(
            diagnostics.fix_plan.fixes.first().map(|fix| fix.id),
            Some(FixId::RebootThenRecheck)
        );
        let reboot = diagnostics.fix_plan.fixes.first().unwrap();
        assert!(reboot.manual_steps[0].contains("sudo reboot"));
        assert!(reboot.manual_steps[1].contains("arc doctor"));
    }

    #[test]
    fn missing_running_kernel_dkms_module_requests_rebuild_or_reinstall() {
        let mut e = evidence();
        e.nvidia_module_loaded = false;
        e.nvidia_smi.succeeded = false;
        e.dkms_status = Some("nvidia/570.26, 6.5.0-old, x86_64: installed".into());
        let diagnostics = diagnose(e, DoctorProfile::ModelTraining).unwrap();
        let module = diagnostics
            .checks
            .iter()
            .find(|check| check.id == DiagnosticId::DriverModule)
            .unwrap();
        assert!(
            module
                .problem
                .as_deref()
                .unwrap()
                .contains("Rebuild the module or reinstall")
        );
        let rebuild = diagnostics
            .fix_plan
            .fixes
            .iter()
            .find(|fix| fix.id == FixId::RebuildDkms)
            .unwrap();
        assert!(rebuild.commands[0].display().contains("dkms autoinstall"));
        assert!(rebuild.manual_steps[0].contains("reinstall"));
    }

    #[test]
    fn secure_boot_blockage_has_specific_manual_fix() {
        let mut e = evidence();
        e.nvidia_module_loaded = false;
        e.nvidia_smi.succeeded = false;
        e.dkms_status = Some("nvidia/570.26, 6.8.0-generic, x86_64: installed".into());
        e.secure_boot_enabled = Some(true);
        e.driver_module_signed = false;
        let diagnostics = diagnose(e, DoctorProfile::ModelTraining).unwrap();
        let fix = diagnostics
            .fix_plan
            .fixes
            .iter()
            .find(|fix| fix.id == FixId::ResolveSecureBoot)
            .unwrap();
        assert!(fix.manual_steps[0].contains("MOK"));
        assert!(
            !diagnostics
                .fix_plan
                .fixes
                .iter()
                .any(|fix| fix.id == FixId::RebootThenRecheck)
        );
    }

    #[test]
    fn operational_driver_has_no_driver_runtime_error() {
        let diagnostics = diagnose(evidence(), DoctorProfile::ModelTraining).unwrap();
        assert_eq!(
            diagnostics
                .checks
                .iter()
                .find(|check| check.id == DiagnosticId::DriverModule)
                .unwrap()
                .status,
            DiagnosticStatus::Pass
        );
        assert!(!diagnostics.has_errors());
    }

    #[test]
    fn broken_managed_driver_gets_exact_repair_instead_of_install_loop() {
        let mut e = evidence();
        e.driver = DriverInstallation::BrokenManaged {
            flavor: DriverFlavorState::Open,
            packages: vec!["nvidia-open".into(), "kmod-nvidia-open-dkms".into()],
        };
        e.nvidia_module_loaded = false;
        e.nvidia_smi.succeeded = false;
        e.driver_version = None;
        e.dkms_status = Some("nvidia/580: added".into());
        let diagnostics = diagnose(e, DoctorProfile::ModelTraining).unwrap();
        let repair = diagnostics
            .fix_plan
            .fixes
            .iter()
            .find(|fix| fix.id == FixId::RepairManagedDriver)
            .expect("managed repair fix");
        let commands = repair
            .commands
            .iter()
            .map(CommandSpec::display)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            commands.contains("apt-get install --reinstall -y nvidia-open kmod-nvidia-open-dkms")
        );
        assert!(commands.contains("linux-headers-6.8.0-generic"));
        assert!(commands.contains("dkms autoinstall -k 6.8.0-generic"));
        assert!(!commands.contains("arc install"));
        assert!(
            diagnostics
                .fix_plan
                .fixes
                .iter()
                .any(|fix| fix.id == FixId::Reboot)
        );
        assert!(
            diagnostics
                .fix_plan
                .fixes
                .iter()
                .all(|fix| { !fix.commands.is_empty() || !fix.manual_steps.is_empty() })
        );
    }

    #[test]
    fn unmanaged_driver_repair_is_manual() {
        let mut e = evidence();
        e.driver = DriverInstallation::Unmanaged {
            working: false,
            evidence: vec![crate::model::environment::UnmanagedDriverEvidence::RunfileUninstaller],
        };
        let diagnostics = diagnose(e, DoctorProfile::ModelTraining).unwrap();
        let fix = diagnostics
            .fix_plan
            .fixes
            .iter()
            .find(|fix| fix.id == FixId::InstallDriver)
            .unwrap();
        assert!(fix.commands.is_empty());
        assert!(
            fix.manual_steps
                .iter()
                .any(|step| step.contains("nvidia-uninstall"))
        );
    }

    #[test]
    fn managed_reinstall_commands_are_package_scoped_for_every_manager() {
        let packages = vec!["nvidia-open".into(), "kmod-nvidia-open-dkms".into()];
        for (distribution, version, expected) in [
            (
                Distribution::Ubuntu,
                "24.04",
                "apt-get install --reinstall -y",
            ),
            (Distribution::Rhel, "9.8", "dnf reinstall -y"),
            (Distribution::AzureLinux, "3.1", "tdnf reinstall -y"),
            (
                Distribution::OpenSuse,
                "15.7",
                "zypper --non-interactive install --force",
            ),
        ] {
            let system = OsInfo {
                distribution,
                name: "Test".into(),
                version_id: version.into(),
                architecture: "x86_64".into(),
                is_wsl: false,
            };
            let rendered = managed_package_reinstall_commands(&system, &packages)[0].display();
            assert!(rendered.contains(expected), "{rendered}");
            assert!(rendered.contains("nvidia-open kmod-nvidia-open-dkms"));
        }
    }

    #[test]
    fn cuda_symlink_repair_is_never_empty() {
        let mut e = evidence();
        e.toolkit_packages = vec!["cuda-toolkit-12-8".into()];
        e.managed_toolkit_version = Some("12.8".into());
        e.cuda_symlink = CudaSymlinkState::Broken("/missing/cuda-12.8".into());
        let diagnostics = diagnose(e, DoctorProfile::CudaDevelopment).unwrap();
        let repair = diagnostics
            .fix_plan
            .fixes
            .iter()
            .find(|fix| fix.id == FixId::RepairCudaSymlink)
            .unwrap();
        assert!(!repair.commands.is_empty() || !repair.manual_steps.is_empty());
    }

    #[test]
    fn arc_install_recommendations_never_include_options() {
        let fixes = available_fixes(&evidence()).unwrap();
        let recommendations = fixes
            .iter()
            .flat_map(|fix| &fix.commands)
            .filter(|command| {
                command.program == "arc" && command.args.first().is_some_and(|arg| arg == "install")
            })
            .collect::<Vec<_>>();

        assert!(!recommendations.is_empty());
        assert!(
            recommendations
                .iter()
                .all(|command| command.display() == "arc install")
        );
    }
}
