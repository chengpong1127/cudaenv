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
            DiagnosticStatus, Diagnostics, Fix, FixId, FixPlan,
        },
        system::OsInfo,
    },
    platform::{os, package_manager},
};

use super::{
    driver,
    gpu::{self, NvidiaGpu},
    toolkit,
};

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
    pub nvidia_open_installed: bool,
    pub cuda_drivers_installed: bool,
    pub nvidia_module_loaded: bool,
    pub nvidia_smi: CommandEvidence,
    pub kernel_release: String,
    pub matching_kernel_headers: bool,
    pub secure_boot_enabled: Option<bool>,
    pub dkms_status: Option<String>,
    pub driver_version: Option<String>,
    pub toolkit_package_installed: bool,
    pub nvcc: CommandEvidence,
    pub nvcc_version: Option<String>,
    pub cuda_symlink: CudaSymlinkState,
    pub installed_cuda_versions: Vec<String>,
}

impl NvidiaEvidence {
    fn driver_installed(&self) -> bool {
        self.nvidia_open_installed || self.cuda_drivers_installed
    }

    fn toolkit_installed(&self) -> bool {
        self.toolkit_package_installed || !self.installed_cuda_versions.is_empty()
    }
}

/// Collect facts only. Interpretation belongs in `checks` and remediation belongs in `fix_plan`.
pub fn collect_evidence() -> Result<NvidiaEvidence> {
    let os = os::detect()?;
    let manager = os.package_manager();
    let kernel_release = command_stdout("uname", &["-r"])
        .context("could not determine the running kernel release")?;
    let nvidia_smi = command_evidence(
        "nvidia-smi",
        &["--query-gpu=driver_version", "--format=csv,noheader"],
    );
    let mut nvcc = command_evidence("nvcc", &["--version"]);
    if !nvcc.exists {
        nvcc = command_evidence("/usr/local/cuda/bin/nvcc", &["--version"]);
    }
    let nvcc_version = parse_nvcc_version(&nvcc.stdout).map(str::to_owned);
    let installed_cuda_versions = installed_cuda_versions();

    Ok(NvidiaEvidence {
        gpus: gpu::detect()?,
        nvidia_open_installed: package_manager::is_installed(manager, "nvidia-open")?,
        cuda_drivers_installed: package_manager::is_installed(manager, "cuda-drivers")?,
        nvidia_module_loaded: Path::new("/sys/module/nvidia").exists(),
        matching_kernel_headers: Path::new("/lib/modules")
            .join(&kernel_release)
            .join("build")
            .exists(),
        secure_boot_enabled: driver::secure_boot_enabled(),
        dkms_status: command_optional_stdout("dkms", &["status"]),
        driver_version: driver_version(&nvidia_smi)?,
        toolkit_package_installed: package_manager::is_installed(manager, "cuda-toolkit")?,
        nvcc,
        nvcc_version,
        cuda_symlink: cuda_symlink_state(Path::new("/usr/local/cuda")),
        installed_cuda_versions,
        kernel_release,
        nvidia_smi,
        os,
    })
}

pub fn detect() -> Result<Diagnostics> {
    diagnose(collect_evidence()?)
}

pub fn diagnose(evidence: NvidiaEvidence) -> Result<Diagnostics> {
    let checks = checks(&evidence);
    let fix_plan = fix_plan(&evidence, &checks)?;
    Ok(Diagnostics {
        vendor: GpuVendor::Nvidia,
        checks,
        fix_plan,
    })
}

/// Evaluate health only. A failed dependency turns a downstream check into `Skipped`.
pub fn checks(e: &NvidiaEvidence) -> Vec<DiagnosticCheck> {
    let mut result = Vec::new();
    push(
        &mut result,
        check(
            DiagnosticId::NvidiaGpu,
            DiagnosticSection::Hardware,
            "NVIDIA GPU detected",
            if e.gpus.is_empty() {
                DiagnosticStatus::Error
            } else {
                DiagnosticStatus::Pass
            },
            vec![if e.gpus.is_empty() {
                "No NVIDIA PCI device found".into()
            } else {
                format!("{} NVIDIA GPU(s) detected", e.gpus.len())
            }],
            e.gpus
                .is_empty()
                .then(|| "No NVIDIA GPU was detected by lspci or sysfs.".into()),
            vec![],
            vec![],
        ),
    );
    push(
        &mut result,
        check(
            DiagnosticId::OperatingSystem,
            DiagnosticSection::OperatingSystem,
            "Supported operating system",
            DiagnosticStatus::Pass,
            vec![format!("{} ({})", e.os.display_name(), e.os.architecture)],
            None,
            vec![],
            vec![],
        ),
    );
    push(&mut result, check(
        DiagnosticId::KernelHeaders, DiagnosticSection::OperatingSystem, "Headers for the running kernel",
        if e.matching_kernel_headers { DiagnosticStatus::Pass } else { DiagnosticStatus::Warning },
        vec![format!("kernel {}: headers {}", e.kernel_release, yes_no(e.matching_kernel_headers))],
        (!e.matching_kernel_headers).then(|| "Matching kernel headers are unavailable; DKMS cannot build a module for this kernel.".into()),
        vec![], vec![FixId::InstallKernelHeaders],
    ));
    push(
        &mut result,
        check(
            DiagnosticId::SecureBoot,
            DiagnosticSection::OperatingSystem,
            "Secure Boot state",
            match e.secure_boot_enabled {
                Some(true) => DiagnosticStatus::Warning,
                Some(false) => DiagnosticStatus::Pass,
                None => DiagnosticStatus::Warning,
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
        ),
    );
    let driver_package_status = if e.driver_installed() {
        DiagnosticStatus::Pass
    } else {
        DiagnosticStatus::Error
    };
    push_dependent(
        &mut result,
        check(
            DiagnosticId::DriverPackage,
            DiagnosticSection::Driver,
            "NVIDIA driver package",
            driver_package_status,
            vec![format!(
                "nvidia-open: {}; cuda-drivers: {}",
                yes_no(e.nvidia_open_installed),
                yes_no(e.cuda_drivers_installed)
            )],
            (!e.driver_installed())
                .then(|| "Neither nvidia-open nor cuda-drivers is installed.".into()),
            vec![DiagnosticId::NvidiaGpu],
            vec![FixId::InstallDriver],
        ),
    );
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
            vec![
                format!(
                    "/sys/module/nvidia: {}",
                    if e.nvidia_module_loaded {
                        "present"
                    } else {
                        "missing"
                    }
                ),
                format!(
                    "DKMS: {}",
                    e.dkms_status.as_deref().unwrap_or("unavailable")
                ),
            ],
            (!e.nvidia_module_loaded)
                .then(|| "The installed NVIDIA driver has not loaded a kernel module.".into()),
            vec![DiagnosticId::DriverPackage],
            vec![FixId::RebuildDkms, FixId::Reboot],
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
            (!e.nvidia_smi.succeeded).then(|| {
                if e.nvidia_smi.exists {
                    "nvidia-smi ran but could not communicate with the driver."
                } else {
                    "nvidia-smi is not installed or not in PATH."
                }
                .into()
            }),
            vec![DiagnosticId::DriverModule],
            vec![FixId::DebugDriver],
        ),
    );
    push_dependent(
        &mut result,
        check(
            DiagnosticId::DriverLibrary,
            DiagnosticSection::Driver,
            "Driver and userspace libraries match",
            if version_mismatch(e) {
                DiagnosticStatus::Error
            } else {
                DiagnosticStatus::Pass
            },
            e.driver_version
                .as_ref()
                .map(|v| format!("driver version: {v}"))
                .into_iter()
                .collect(),
            version_mismatch(e).then(|| "NVML reports a driver/library version mismatch.".into()),
            vec![DiagnosticId::DriverPackage],
            vec![FixId::ReinstallDriverLibraries, FixId::Reboot],
        ),
    );
    push(&mut result, check(
        DiagnosticId::ToolkitInstall, DiagnosticSection::CudaToolkit, "CUDA Toolkit installation",
        if e.toolkit_installed() { DiagnosticStatus::Pass } else { DiagnosticStatus::Warning },
        vec![format!("cuda-toolkit package: {}; installed versions: {}", yes_no(e.toolkit_package_installed), list_or_none(&e.installed_cuda_versions))],
        (!e.toolkit_installed()).then(|| "No CUDA Toolkit installation was detected (optional for framework-only workloads).".into()),
        vec![], vec![FixId::InstallToolkit],
    ));
    push_dependent(
        &mut result,
        check(
            DiagnosticId::Nvcc,
            DiagnosticSection::CudaToolkit,
            "nvcc available",
            if e.nvcc.exists && e.nvcc.succeeded && e.nvcc_version.is_some() {
                DiagnosticStatus::Pass
            } else {
                DiagnosticStatus::Error
            },
            command_evidence_lines("nvcc", &e.nvcc),
            (!(e.nvcc.exists && e.nvcc.succeeded && e.nvcc_version.is_some())).then(|| {
                "A Toolkit is installed, but nvcc is missing, broken, or not in PATH.".into()
            }),
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
            vec![
                cuda_symlink_description(&e.cuda_symlink),
                format!(
                    "installed CUDA versions: {}",
                    list_or_none(&e.installed_cuda_versions)
                ),
            ],
            (!matches!(e.cuda_symlink, CudaSymlinkState::Valid(_))).then(|| {
                "/usr/local/cuda does not point to a valid CUDA Toolkit installation.".into()
            }),
            vec![DiagnosticId::ToolkitInstall],
            vec![FixId::RepairCudaSymlink],
        ),
    );
    let incompatible = compatibility(e).is_some_and(|compatible| !compatible);
    push_dependent(
        &mut result,
        check(
            DiagnosticId::DriverToolkitCompatibility,
            DiagnosticSection::CudaToolkit,
            "Driver supports CUDA Toolkit",
            if incompatible {
                DiagnosticStatus::Error
            } else {
                DiagnosticStatus::Pass
            },
            vec![format!(
                "driver: {}; toolkit: {}",
                e.driver_version.as_deref().unwrap_or("unknown"),
                e.nvcc_version.as_deref().unwrap_or("unknown")
            )],
            incompatible
                .then(|| "The loaded driver is too old for the installed CUDA Toolkit.".into()),
            vec![DiagnosticId::NvidiaSmi, DiagnosticId::Nvcc],
            vec![FixId::UpgradeDriver, FixId::Reboot],
        ),
    );
    result
}

fn push(result: &mut Vec<DiagnosticCheck>, check: DiagnosticCheck) {
    result.push(check);
}

fn push_dependent(result: &mut Vec<DiagnosticCheck>, mut check: DiagnosticCheck) {
    if check.dependencies.iter().any(|dependency| {
        result
            .iter()
            .any(|prior| prior.id == *dependency && prior.status != DiagnosticStatus::Pass)
    }) {
        check.status = DiagnosticStatus::Skipped;
        check.problem = Some("Skipped because a prerequisite check failed.".into());
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
    recommended_fixes: Vec<FixId>,
) -> DiagnosticCheck {
    DiagnosticCheck {
        id,
        section,
        name: name.into(),
        status,
        evidence,
        problem,
        dependencies,
        recommended_fixes,
    }
}

/// Match causes and create actions only. No evidence is collected or commands executed here.
pub fn fix_plan(e: &NvidiaEvidence, checks: &[DiagnosticCheck]) -> Result<FixPlan> {
    let mut causes = Vec::new();
    let failed = |id| {
        checks
            .iter()
            .any(|check| check.id == id && check.status == DiagnosticStatus::Error)
    };
    if failed(DiagnosticId::NvidiaGpu) {
        causes.push(cause(
            "The NVIDIA GPU is not visible to the operating system",
            Confidence::High,
            vec!["No NVIDIA PCI device was detected".into()],
            vec![FixId::InspectHardware],
        ));
    }
    if failed(DiagnosticId::DriverPackage) {
        causes.push(cause(
            "NVIDIA driver package is missing",
            Confidence::High,
            vec!["Neither nvidia-open nor cuda-drivers is installed".into()],
            vec![FixId::InstallDriver],
        ));
    }
    if e.driver_installed() && !e.nvidia_module_loaded && !e.matching_kernel_headers {
        causes.push(cause(
            "Kernel headers are missing, so the NVIDIA module cannot be built",
            Confidence::High,
            vec![
                format!("No headers for {}", e.kernel_release),
                "/sys/module/nvidia is missing".into(),
            ],
            vec![
                FixId::InstallKernelHeaders,
                FixId::RebuildDkms,
                FixId::Reboot,
            ],
        ));
    } else if e.driver_installed() && !e.nvidia_module_loaded && e.secure_boot_enabled == Some(true)
    {
        causes.push(cause(
            "Secure Boot is likely blocking the NVIDIA module",
            Confidence::High,
            vec![
                "Secure Boot is enabled".into(),
                "/sys/module/nvidia is missing".into(),
            ],
            vec![FixId::SecureBootEnrollment, FixId::Reboot],
        ));
    } else if e.driver_installed() && !e.nvidia_module_loaded && e.matching_kernel_headers {
        causes.push(cause(
            "The NVIDIA driver is installed but a reboot is likely required",
            Confidence::High,
            vec![
                "Driver package is installed".into(),
                "Matching headers exist, but the module is not loaded".into(),
            ],
            vec![FixId::Reboot, FixId::DebugDriver],
        ));
    }
    if version_mismatch(e) {
        causes.push(cause(
            "NVIDIA driver and userspace library versions do not match",
            Confidence::High,
            vec![trimmed(&e.nvidia_smi.stderr)],
            vec![FixId::ReinstallDriverLibraries, FixId::Reboot],
        ));
    }
    if e.toolkit_installed() && (!e.nvcc.exists || !e.nvcc.succeeded || e.nvcc_version.is_none()) {
        causes.push(cause(
            "CUDA Toolkit configuration is incomplete or broken",
            Confidence::High,
            command_evidence_lines("nvcc", &e.nvcc),
            vec![FixId::InstallToolkit, FixId::DebugToolkit],
        ));
    }
    if !e.toolkit_installed() && !failed(DiagnosticId::NvidiaGpu) {
        causes.push(cause(
            "CUDA Toolkit is not installed",
            Confidence::High,
            vec![
                "No cuda-toolkit package or /usr/local/cuda-<version> installation was detected"
                    .into(),
            ],
            vec![FixId::InstallToolkit],
        ));
    }
    if e.toolkit_installed()
        && matches!(
            e.cuda_symlink,
            CudaSymlinkState::Broken(_) | CudaSymlinkState::NotSymlink
        )
    {
        causes.push(cause(
            "/usr/local/cuda is broken",
            Confidence::High,
            vec![cuda_symlink_description(&e.cuda_symlink)],
            vec![FixId::RepairCudaSymlink],
        ));
    }
    if compatibility(e) == Some(false) {
        causes.push(cause(
            "The NVIDIA driver is incompatible with the CUDA Toolkit",
            Confidence::High,
            vec![format!(
                "driver {} does not meet the minimum for CUDA {}",
                e.driver_version.as_deref().unwrap_or("unknown"),
                e.nvcc_version.as_deref().unwrap_or("unknown")
            )],
            vec![FixId::UpgradeDriver, FixId::Reboot],
        ));
    }
    if causes.is_empty()
        && checks
            .iter()
            .any(|check| check.status == DiagnosticStatus::Error)
    {
        causes.push(cause(
            "The available evidence does not identify a single root cause",
            Confidence::Low,
            vec![
                "Inspect the failing checks and collect kernel logs before changing packages"
                    .into(),
            ],
            vec![FixId::DebugDriver, FixId::DebugToolkit],
        ));
    }
    Ok(FixPlan::new(causes, available_fixes(e)?))
}

fn available_fixes(e: &NvidiaEvidence) -> Result<Vec<Fix>> {
    let manager = e.os.package_manager();
    let driver_package = if e.nvidia_open_installed {
        "nvidia-open"
    } else if e.cuda_drivers_installed {
        "cuda-drivers"
    } else {
        "nvidia-open"
    };
    let target = e
        .installed_cuda_versions
        .last()
        .map(|version| format!("/usr/local/cuda-{version}"));
    Ok(vec![
        fix(FixId::InspectHardware, "Verify that the NVIDIA GPU is visible to Linux", vec![CommandSpec::new("lspci", ["-nn"]), CommandSpec::new("lspci", ["-nnk", "-d", "10de:"])], vec!["For a VM or container, verify GPU passthrough/device exposure on the host before changing guest packages.".into()], 5),
        fix(FixId::InstallKernelHeaders, "Install headers for the running kernel", vec![package_manager::kernel_headers_install_command(&e.os, &e.kernel_release)], vec![], 10),
        fix(FixId::InstallDriver, "Install the NVIDIA driver using the normal cudaenv install flow", vec![CommandSpec::new("cudaenv", ["install", "--profile", "model-training", "--dry-run"])], vec!["Review the generated installation plan, then rerun it without --dry-run.".into()], 20),
        fix(FixId::UpgradeDriver, "Upgrade the NVIDIA driver", vec![package_manager::refresh_command(manager), package_manager::install_command(manager, driver_package)], vec![], 20),
        fix(FixId::ReinstallDriverLibraries, "Reinstall the NVIDIA driver package and libraries", vec![package_manager::reinstall_command(manager, driver_package)], vec![], 30),
        fix(FixId::RebuildDkms, "Rebuild NVIDIA DKMS modules", vec![CommandSpec::sudo("dkms", ["autoinstall", "-k", &e.kernel_release])], vec![], 40),
        fix(FixId::InstallToolkit, "Install or repair the CUDA Toolkit", vec![package_manager::install_command(manager, &toolkit::package(None)?)], vec!["Use `cudaenv install --profile cuda-development` to select and review a specific Toolkit version.".into()], 50),
        fix(FixId::RepairCudaSymlink, "Repair /usr/local/cuda", target.map(|target| CommandSpec::sudo("ln", ["-sfn", &target, "/usr/local/cuda"])).into_iter().collect(), if e.installed_cuda_versions.is_empty() { vec!["Choose an installed /usr/local/cuda-<version> directory, then update /usr/local/cuda to point to it.".into()] } else { vec![] }, 60),
        fix(FixId::SecureBootEnrollment, "Authorize the NVIDIA kernel module under Secure Boot", vec![CommandSpec::new("mokutil", ["--sb-state"]), CommandSpec::new("modinfo", ["-F", "signer", "nvidia"])], vec!["Use your distribution's NVIDIA DKMS module-signing or MOK-enrollment procedure; key locations differ by distribution. Follow the firmware enrollment screen at the next reboot if a key is enrolled.".into()], 70),
        fix(FixId::DebugDriver, "Collect driver debugging evidence", vec![CommandSpec::new("journalctl", ["-k", "-b", "-g", "NVRM|nvidia|nouveau"]), CommandSpec::new("modinfo", ["nvidia"]), CommandSpec::new("dkms", ["status"])], vec!["Inspect /var/lib/dkms/nvidia/*/build/make.log when present.".into()], 80),
        fix(FixId::DebugToolkit, "Inspect CUDA Toolkit paths", vec![CommandSpec::new("which", ["nvcc"]), CommandSpec::new("readlink", ["-f", "/usr/local/cuda"]), CommandSpec::new("/usr/local/cuda/bin/nvcc", ["--version"])], vec![], 80),
        fix(FixId::Reboot, "Reboot to load the repaired driver", vec![CommandSpec::sudo("systemctl", ["reboot"])], vec!["Save your work before rebooting.".into()], 90),
    ])
}

fn fix(
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
fn cause(
    title: &str,
    confidence: Confidence,
    evidence: Vec<String>,
    fixes: Vec<FixId>,
) -> DiagnosticCause {
    DiagnosticCause {
        title: title.into(),
        confidence,
        evidence,
        fixes,
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
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => CommandEvidence::default(),
        Err(error) => CommandEvidence {
            stderr: error.to_string(),
            ..CommandEvidence::default()
        },
    }
}

fn command_stdout(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    anyhow::ensure!(output.status.success(), "{program} exited unsuccessfully");
    Ok(String::from_utf8_lossy(&output.stdout).trim().into())
}

fn command_optional_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().into())
}

fn driver_version(smi: &CommandEvidence) -> Result<Option<String>> {
    if smi.succeeded
        && let Some(version) = smi
            .stdout
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
    {
        return Ok(Some(version.into()));
    }
    driver::detect_version()
}

fn installed_cuda_versions() -> Vec<String> {
    let Ok(entries) = fs::read_dir("/usr/local") else {
        return Vec::new();
    };
    let mut versions = entries
        .flatten()
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.strip_prefix("cuda-"))
                .filter(|version| version.chars().next().is_some_and(|c| c.is_ascii_digit()))
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    versions.sort_by_key(|version| version_parts(version));
    versions.dedup();
    versions
}

fn cuda_symlink_state(path: &Path) -> CudaSymlinkState {
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_symlink() => CudaSymlinkState::NotSymlink,
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
            Err(error) => CudaSymlinkState::Unavailable(error.to_string()),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => CudaSymlinkState::Missing,
        Err(error) => CudaSymlinkState::Unavailable(error.to_string()),
    }
}

fn parse_nvcc_version(output: &str) -> Option<&str> {
    let (_, rest) = output.split_once("release ")?;
    rest.split(|c: char| c == ',' || c.is_whitespace())
        .find(|part| !part.is_empty())
}

fn compatibility(e: &NvidiaEvidence) -> Option<bool> {
    let driver = version_parts(e.driver_version.as_deref()?);
    let toolkit = version_parts(e.nvcc_version.as_deref()?);
    let minimum = match toolkit.as_slice() {
        [major, ..] if *major >= 13 => vec![580],
        [12, minor, ..] if *minor >= 8 => vec![570],
        [12, minor, ..] if *minor >= 5 => vec![555],
        [12, ..] => vec![525],
        [11, minor, ..] if *minor >= 8 => vec![450],
        [11, ..] => vec![450],
        _ => return None,
    };
    Some(driver >= minimum)
}

fn version_parts(version: &str) -> Vec<u32> {
    version
        .split(['.', '-'])
        .map_while(|part| part.parse().ok())
        .collect()
}
fn version_mismatch(e: &NvidiaEvidence) -> bool {
    format!("{}\n{}", e.nvidia_smi.stdout, e.nvidia_smi.stderr)
        .to_ascii_lowercase()
        .contains("driver/library version mismatch")
}
fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}
fn trimmed(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        "No command output was captured".into()
    } else {
        value.into()
    }
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
        format!("stdout: {}", trimmed(&value.stdout)),
        format!("stderr: {}", trimmed(&value.stderr)),
    ]
}
fn cuda_symlink_description(state: &CudaSymlinkState) -> String {
    match state {
        CudaSymlinkState::Missing => "/usr/local/cuda: missing".into(),
        CudaSymlinkState::Valid(target) => {
            format!("/usr/local/cuda -> {} (valid)", target.display())
        }
        CudaSymlinkState::Broken(target) => {
            format!("/usr/local/cuda -> {} (broken)", target.display())
        }
        CudaSymlinkState::NotSymlink => "/usr/local/cuda exists but is not a symlink".into(),
        CudaSymlinkState::Unavailable(error) => {
            format!("/usr/local/cuda could not be inspected: {error}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::system::Distribution;

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
                generation: gpu::Generation::TuringOrNewer,
            }],
            nvidia_open_installed: true,
            cuda_drivers_installed: false,
            nvidia_module_loaded: true,
            nvidia_smi: CommandEvidence {
                exists: true,
                succeeded: true,
                stdout: "570.86.15".into(),
                stderr: String::new(),
            },
            kernel_release: "6.8.0".into(),
            matching_kernel_headers: true,
            secure_boot_enabled: Some(false),
            dkms_status: Some("nvidia/570: installed".into()),
            driver_version: Some("570.86.15".into()),
            toolkit_package_installed: true,
            nvcc: CommandEvidence {
                exists: true,
                succeeded: true,
                stdout: "Cuda compilation tools, release 12.8, V12.8".into(),
                stderr: String::new(),
            },
            nvcc_version: Some("12.8".into()),
            cuda_symlink: CudaSymlinkState::Valid("cuda-12.8".into()),
            installed_cuda_versions: vec!["12.8".into()],
        }
    }

    #[test]
    fn skips_downstream_driver_checks_when_package_is_missing() {
        let mut e = evidence();
        e.nvidia_open_installed = false;
        let checks = checks(&e);
        assert_eq!(
            checks
                .iter()
                .find(|c| c.id == DiagnosticId::DriverPackage)
                .unwrap()
                .status,
            DiagnosticStatus::Error
        );
        assert_eq!(
            checks
                .iter()
                .find(|c| c.id == DiagnosticId::DriverModule)
                .unwrap()
                .status,
            DiagnosticStatus::Skipped
        );
        assert_eq!(
            checks
                .iter()
                .find(|c| c.id == DiagnosticId::NvidiaSmi)
                .unwrap()
                .status,
            DiagnosticStatus::Skipped
        );
    }

    #[test]
    fn matches_missing_headers_before_rebuild_and_reboot() {
        let mut e = evidence();
        e.nvidia_module_loaded = false;
        e.matching_kernel_headers = false;
        let checks = checks(&e);
        let plan = fix_plan(&e, &checks).unwrap();
        assert!(
            plan.causes
                .iter()
                .any(|cause| cause.title.contains("headers"))
        );
        assert_eq!(
            plan.fixes.iter().map(|fix| fix.id).collect::<Vec<_>>(),
            vec![
                FixId::InstallKernelHeaders,
                FixId::RebuildDkms,
                FixId::Reboot
            ]
        );
    }

    #[test]
    fn matches_secure_boot_and_version_mismatch_rules() {
        let mut e = evidence();
        e.nvidia_module_loaded = false;
        e.secure_boot_enabled = Some(true);
        e.nvidia_smi.succeeded = false;
        e.nvidia_smi.stderr = "Failed to initialize NVML: Driver/library version mismatch".into();
        let plan = fix_plan(&e, &checks(&e)).unwrap();
        assert!(
            plan.causes
                .iter()
                .any(|cause| cause.title.contains("Secure Boot"))
        );
        assert!(
            plan.causes
                .iter()
                .any(|cause| cause.title.contains("userspace"))
        );
    }

    #[test]
    fn matches_broken_toolkit_and_incompatible_versions() {
        let mut e = evidence();
        e.cuda_symlink = CudaSymlinkState::Broken("cuda-11.0".into());
        e.driver_version = Some("525.1".into());
        e.nvidia_smi.stdout = "525.1".into();
        e.nvcc_version = Some("13.0".into());
        let plan = fix_plan(&e, &checks(&e)).unwrap();
        assert!(
            plan.causes
                .iter()
                .any(|cause| cause.title.contains("/usr/local/cuda"))
        );
        assert!(
            plan.causes
                .iter()
                .any(|cause| cause.title.contains("incompatible"))
        );
    }

    #[test]
    fn matches_missing_driver_reboot_and_missing_toolkit_rules() {
        let mut missing_driver = evidence();
        missing_driver.nvidia_open_installed = false;
        assert!(
            fix_plan(&missing_driver, &checks(&missing_driver))
                .unwrap()
                .causes
                .iter()
                .any(|cause| cause.title.contains("driver package"))
        );

        let mut reboot = evidence();
        reboot.nvidia_module_loaded = false;
        assert!(
            fix_plan(&reboot, &checks(&reboot))
                .unwrap()
                .causes
                .iter()
                .any(|cause| cause.title.contains("reboot"))
        );

        let mut toolkit = evidence();
        toolkit.toolkit_package_installed = false;
        toolkit.installed_cuda_versions.clear();
        toolkit.nvcc = CommandEvidence::default();
        toolkit.nvcc_version = None;
        toolkit.cuda_symlink = CudaSymlinkState::Missing;
        assert!(
            fix_plan(&toolkit, &checks(&toolkit))
                .unwrap()
                .causes
                .iter()
                .any(|cause| cause.title == "CUDA Toolkit is not installed")
        );
    }
}
