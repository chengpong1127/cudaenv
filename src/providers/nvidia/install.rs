use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::{
    model::{
        environment::{DriverFlavorState, DriverInstallation, ProviderStatus},
        operation::{OperationPlan, PlanDetail, PlanStep},
        system::OsInfo,
    },
    platform::package_manager,
    providers::nvidia::{
        compatibility::{self, Compatibility},
        driver::{DriverFlavor, DriverPreference},
        gpu::{self, NvidiaGpu},
        policy, recipe,
        repository::{self, NvidiaRepository},
        state, toolkit,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallProfile {
    ModelTraining,
    CudaDevelopment,
}
impl InstallProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::ModelTraining => "Model training (PyTorch, TensorFlow, JAX)",
            Self::CudaDevelopment => "CUDA development",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstallOptions {
    pub profile: InstallProfile,
    pub toolkit_version: Option<String>,
    pub driver: DriverPreference,
}

pub fn plan(os: &OsInfo, options: &InstallOptions) -> Result<OperationPlan> {
    os.ensure_driver_installable("NVIDIA")?;
    let kernel = kernel_release()?;
    let gpus = gpu::detect()?;
    let status = state::inspect(os)?;
    let repository = repository::resolve(os)?;
    let repository_configured = repository::is_configured(os, &repository)?;
    let requested_toolkit = requested_toolkit(options)?;
    let toolkit_installed = requested_toolkit
        .as_deref()
        .is_some_and(|p| package_manager::is_installed(os.package_manager(), p).unwrap_or(false));
    build_plan(
        os,
        options,
        &kernel,
        &gpus,
        &status,
        &repository,
        repository_configured,
        toolkit_installed,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_plan(
    os: &OsInfo,
    options: &InstallOptions,
    kernel: &str,
    gpus: &[NvidiaGpu],
    status: &ProviderStatus,
    repository: &NvidiaRepository,
    repository_configured: bool,
    toolkit_package_installed: bool,
) -> Result<OperationPlan> {
    let cuda_development = options.profile == InstallProfile::CudaDevelopment;
    let policy = policy::resolve(
        os,
        gpus,
        options.driver,
        options.toolkit_version.as_deref(),
        cuda_development,
    )?;
    let recipe = recipe::resolve(os, kernel, policy)?;
    if let DriverInstallation::Unmanaged {
        working,
        runfile_likely,
    } = status.driver
    {
        bail!(
            "A{} unmanaged NVIDIA driver installation was detected{}. arc will not install repository packages over it. Remove it with its original installer or migrate it to distribution packages first.",
            if working { " working" } else { " broken" },
            if runfile_likely {
                " (likely installed by the NVIDIA runfile)"
            } else {
                ""
            }
        );
    }
    if status.driver.flavor() == Some(DriverFlavorState::Mixed) {
        bail!(
            "Conflicting open and proprietary NVIDIA packages are installed. Repair or remove the mixed package installation before using arc install."
        );
    }
    let broken_managed_packages = match &status.driver {
        DriverInstallation::BrokenManaged { packages, .. } => Some(packages.as_slice()),
        _ => None,
    };
    let target_state = match policy.flavor {
        DriverFlavor::Open => DriverFlavorState::Open,
        DriverFlavor::Proprietary => DriverFlavorState::Proprietary,
    };
    let current_flavor = status.driver.flavor().and_then(|value| match value {
        DriverFlavorState::Open => Some(DriverFlavor::Open),
        DriverFlavorState::Proprietary => Some(DriverFlavor::Proprietary),
        DriverFlavorState::Mixed => None,
    });
    let branch_matches = policy.branch.is_none()
        || matches!(&status.driver, DriverInstallation::Managed { branch: Some(branch), .. } if Some(*branch) == policy.branch);
    let requested_toolkit_version = if cuda_development {
        Some(
            options
                .toolkit_version
                .as_deref()
                .unwrap_or(toolkit::LATEST_TOOLKIT_VERSION),
        )
    } else {
        None
    };
    let driver_compatible = status
        .driver_version
        .as_deref()
        .zip(requested_toolkit_version)
        .and_then(|(driver, toolkit)| compatibility::evaluate(driver, toolkit))
        != Some(Compatibility::Incompatible);
    let driver_correct = matches!(status.driver, DriverInstallation::Managed { .. })
        && status.driver.flavor() == Some(target_state)
        && branch_matches
        && driver_compatible;
    let driver_pending_activation = driver_correct && status.driver_version.is_none();
    let branch_transition = policy.branch.is_some()
        && matches!(&status.driver, DriverInstallation::Managed { branch, .. } if *branch != policy.branch);
    let transition = current_flavor.is_some_and(|from| from != policy.flavor) || branch_transition;
    let toolkit_package = requested_toolkit(options)?;
    let current_toolkit = status
        .toolkits
        .first()
        .and_then(|value| value.version.as_deref());
    let install_toolkit = toolkit_package.as_deref().is_some_and(|package| {
        toolkit_install_needed(package, current_toolkit, toolkit_package_installed)
    });
    let install_driver = !driver_correct;
    let mut steps = Vec::new();
    if install_driver || install_toolkit {
        if !repository_configured {
            if !repository::downloader_available() {
                bail!(
                    "Configuring the NVIDIA repository requires curl or wget, but neither command is available."
                );
            }
            steps.extend(
                repository::setup_commands(os.package_manager(), repository)
                    .into_iter()
                    .map(|command| PlanStep::new("Configure the NVIDIA CUDA repository", command)),
            );
        }
        steps.push(PlanStep::new(
            "Refresh package metadata",
            package_manager::refresh_command(os.package_manager()),
        ));
    }
    if install_driver {
        steps.extend(
            recipe
                .prerequisites
                .into_iter()
                .map(|command| PlanStep::new("Ensure NVIDIA driver prerequisites", command)),
        );
        steps.push(PlanStep::new(
            "Refresh package metadata after ensuring prerequisites",
            package_manager::refresh_command(os.package_manager()),
        ));
        if let Some(packages) = broken_managed_packages {
            if let Some(command) = package_manager::reinstall_command(os.package_manager(), packages)
            {
                steps.push(PlanStep::new(
                    "Reinstall the detected NVIDIA driver packages",
                    command,
                ));
            }
            if packages.iter().any(|package| package.contains("dkms")) {
                steps.push(PlanStep::new(
                    "Rebuild the NVIDIA module for the running kernel",
                    crate::model::command::CommandSpec::sudo(
                        "dkms",
                        ["autoinstall", "-k", kernel],
                    ),
                ));
            }
        }
        if let Some(from) = current_flavor {
            steps.extend(
                recipe::transition_commands(os, policy, from)
                    .into_iter()
                    .map(|command| {
                        PlanStep::new("Transition the NVIDIA driver package stream", command)
                    }),
            );
        } else {
            steps.extend(
                recipe.driver_preparation.into_iter().map(|command| {
                    PlanStep::new("Select the NVIDIA driver package stream", command)
                }),
            );
            steps.push(PlanStep::new(
                format!(
                    "Verify NVIDIA driver package {} is available",
                    policy.flavor.package()
                ),
                package_manager::query_command(os.package_manager(), policy.flavor.package()),
            ));
            steps.push(PlanStep::new(
                "Install the NVIDIA driver",
                recipe.driver_install,
            ));
        }
        steps.push(PlanStep::new(
            "Verify that NVIDIA kernel module metadata is installed",
            recipe.driver_verification,
        ));
    }
    if install_toolkit && let Some(package) = toolkit_package.as_deref() {
        steps.push(PlanStep::new(
            format!("Verify CUDA Toolkit package {package} is available"),
            package_manager::query_command(os.package_manager(), package),
        ));
        steps.push(PlanStep::new(
            format!("Install {package}"),
            package_manager::install_command(os.package_manager(), package),
        ));
        steps.push(PlanStep::new(
            "Verify the CUDA Toolkit with nvcc",
            toolkit::verification_command(),
        ));
    }
    let driver_detail = if driver_pending_activation {
        format!(
            "{} installed; kernel module is ready but not loaded — reboot required",
            policy.flavor.package()
        )
    } else if driver_correct {
        format!(
            "{} already managed correctly — skipped",
            policy.flavor.package()
        )
    } else if broken_managed_packages.is_some() {
        format!(
            "repair detected packages and ensure {} is installed",
            policy.flavor.package()
        )
    } else if transition {
        format!(
            "transition to {}{}",
            policy.flavor.package(),
            policy
                .branch
                .map(|b| format!(" pinned to R{b}"))
                .unwrap_or_default()
        )
    } else {
        format!(
            "install {}{}",
            policy.flavor.package(),
            policy
                .branch
                .map(|b| format!(" pinned to R{b}"))
                .unwrap_or_default()
        )
    };
    Ok(OperationPlan {
        title: "NVIDIA Installation Plan".into(),
        details: vec![
            PlanDetail::new("OS", os.display_name()),
            PlanDetail::new("Kernel", kernel),
            PlanDetail::new("Package manager", os.package_manager().to_string()),
            PlanDetail::new("Repository", repository.base_url.clone()),
            PlanDetail::new(
                "Release validation",
                format!(
                    "repository-compatible {}; NVIDIA validated: {}; arc tested: {}",
                    repository.family,
                    if repository.nvidia_validated {
                        "yes"
                    } else {
                        "no"
                    },
                    if repository.arc_tested {
                        "yes"
                    } else {
                        "no"
                    }
                ),
            ),
            PlanDetail::new("Profile", options.profile.label()),
            PlanDetail::new("Existing driver", status.driver.description()),
            PlanDetail::new("Driver", driver_detail),
            PlanDetail::new(
                "CUDA Toolkit",
                toolkit_package
                    .as_deref()
                    .map_or("not requested".into(), |p| {
                        if install_toolkit {
                            format!("install {p}")
                        } else {
                            "requested version already installed — skipped".into()
                        }
                    }),
            ),
            PlanDetail::new(
                "Kernel headers",
                if super::driver::kernel_headers_available() {
                    "detected for running kernel"
                } else {
                    "install exact matching prerequisites before driver"
                },
            ),
        ],
        devices: status.devices.clone(),
        steps,
        confirmation_warning: "No system changes will be made until you confirm.".into(),
        completion_message: "Installation completed.".into(),
        reboot_message: if driver_pending_activation {
            Some("The NVIDIA kernel module is installed but not loaded. Reboot, then run `arc doctor` if the driver is still unavailable.".into())
        } else {
            install_driver.then(|| "Reboot to load the NVIDIA driver.".into())
        },
    })
}

fn requested_toolkit(options: &InstallOptions) -> Result<Option<String>> {
    match options.profile {
        InstallProfile::ModelTraining => Ok(None),
        InstallProfile::CudaDevelopment => {
            toolkit::package(options.toolkit_version.as_deref()).map(Some)
        }
    }
}
fn kernel_release() -> Result<String> {
    let output = Command::new("uname")
        .arg("-r")
        .output()
        .context("could not determine running kernel")?;
    anyhow::ensure!(output.status.success(), "uname -r failed");
    Ok(String::from_utf8_lossy(&output.stdout).trim().into())
}
fn toolkit_install_needed(package: &str, current_version: Option<&str>, installed: bool) -> bool {
    if installed {
        return false;
    }
    if package == "cuda-toolkit" {
        return true;
    }
    current_version
        .and_then(|version| toolkit::versioned_package(version).ok())
        .is_none_or(|current| current != package)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        device::GpuVendor,
        environment::{DriverPackageScope, ToolkitSource, ToolkitStatus},
        system::Distribution,
    };
    use crate::providers::nvidia::gpu::Generation;
    fn os() -> OsInfo {
        OsInfo {
            distribution: Distribution::Ubuntu,
            name: "Ubuntu".into(),
            version_id: "24.04".into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        }
    }
    fn gpu(generation: Generation) -> NvidiaGpu {
        NvidiaGpu {
            name: "GPU".into(),
            pci_device_id: None,
            generation,
        }
    }
    fn status(driver: DriverInstallation, toolkit_version: Option<&str>) -> ProviderStatus {
        ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![],
            driver,
            driver_version: Some("580.65.06".into()),
            toolkits: toolkit_version
                .map(|v| ToolkitStatus {
                    name: "System-managed CUDA Toolkit".into(),
                    version: Some(v.into()),
                    executable_path: Some(format!("/usr/local/cuda-{v}/bin/nvcc")),
                    source: ToolkitSource::SystemPackageManager,
                    packages: vec![format!("cuda-toolkit-{}", v.replace('.', "-"))],
                    manageable: true,
                })
                .into_iter()
                .collect(),
            active_toolkit: None,
        }
    }
    fn managed(flavor: DriverFlavorState, branch: Option<u32>) -> DriverInstallation {
        DriverInstallation::Managed {
            flavor,
            scope: DriverPackageScope::Full,
            branch,
            packages: vec![],
        }
    }
    fn options(profile: InstallProfile, toolkit: Option<&str>) -> InstallOptions {
        InstallOptions {
            profile,
            toolkit_version: toolkit.map(Into::into),
            driver: DriverPreference::Auto,
        }
    }

    #[test]
    fn matching_modern_install_is_noop() {
        let plan = build_plan(
            &os(),
            &options(InstallProfile::CudaDevelopment, Some("13.1")),
            "6.8.0-generic",
            &[gpu(Generation::TuringOrNewer)],
            &status(managed(DriverFlavorState::Open, None), Some("13.1")),
            &repository::resolve(&os()).unwrap(),
            true,
            true,
        )
        .unwrap();
        assert!(plan.is_noop());
    }

    #[test]
    fn installed_module_waiting_for_reboot_is_a_noop_with_guidance() {
        let mut current = status(managed(DriverFlavorState::Open, None), None);
        current.driver_version = None;
        let plan = build_plan(
            &os(),
            &options(InstallProfile::ModelTraining, None),
            "6.8.0-generic",
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &repository::resolve(&os()).unwrap(),
            true,
            false,
        )
        .unwrap();

        assert!(plan.is_noop());
        assert!(
            plan.reboot_message
                .as_deref()
                .unwrap()
                .contains("installed but not loaded")
        );
    }

    #[test]
    fn broken_managed_driver_gets_an_executable_repair_plan() {
        let mut current = status(
            DriverInstallation::BrokenManaged {
                flavor: DriverFlavorState::Open,
                packages: vec!["nvidia-open".into(), "nvidia-dkms-610-open".into()],
            },
            None,
        );
        current.driver_version = None;
        let plan = build_plan(
            &os(),
            &options(InstallProfile::ModelTraining, None),
            "6.8.0-generic",
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &repository::resolve(&os()).unwrap(),
            true,
            false,
        )
        .unwrap();
        let commands = plan
            .steps
            .iter()
            .map(|step| step.command.display())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(commands.contains(
            "apt-get install --reinstall -y nvidia-open nvidia-dkms-610-open"
        ));
        assert!(commands.contains("dkms autoinstall -k 6.8.0-generic"));
        assert!(commands.contains("apt-get install -y nvidia-open"));
        assert!(commands.contains("modinfo nvidia"));
        assert!(plan.reboot_message.is_some());
    }

    #[test]
    fn custom_active_nvcc_does_not_suppress_system_toolkit_install() {
        let mut current = status(managed(DriverFlavorState::Open, None), None);
        current.active_toolkit = Some(ToolkitStatus {
            name: "Active nvcc".into(),
            version: Some("13.1".into()),
            executable_path: Some("/opt/conda/envs/cuda/bin/nvcc".into()),
            source: ToolkitSource::ActivePath,
            packages: vec![],
            manageable: false,
        });
        let plan = build_plan(
            &os(),
            &options(InstallProfile::CudaDevelopment, Some("13.1")),
            "6.8.0-generic",
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &repository::resolve(&os()).unwrap(),
            true,
            false,
        )
        .unwrap();
        assert!(plan.steps.iter().any(|step| {
            step.command
                .display()
                .contains("apt-get install -y cuda-toolkit-13-1")
        }));
    }
    #[test]
    fn legacy_plan_pins_r580_and_rejects_cuda_13() {
        let plan = build_plan(
            &os(),
            &options(InstallProfile::CudaDevelopment, Some("12.8")),
            "6.8.0-generic",
            &[gpu(Generation::MaxwellPascalVolta)],
            &status(DriverInstallation::Missing, None),
            &repository::resolve(&os()).unwrap(),
            true,
            false,
        )
        .unwrap();
        assert!(
            plan.steps
                .iter()
                .any(|s| s.command.display().contains("nvidia-driver-pinning-580"))
        );
        assert!(
            build_plan(
                &os(),
                &options(InstallProfile::CudaDevelopment, Some("13.1")),
                "6.8.0-generic",
                &[gpu(Generation::MaxwellPascalVolta)],
                &status(DriverInstallation::Missing, None),
                &repository::resolve(&os()).unwrap(),
                true,
                false
            )
            .is_err()
        );
    }
    #[test]
    fn refuses_working_unmanaged_driver() {
        let result = build_plan(
            &os(),
            &options(InstallProfile::ModelTraining, None),
            "6.8.0-generic",
            &[gpu(Generation::TuringOrNewer)],
            &status(
                DriverInstallation::Unmanaged {
                    working: true,
                    runfile_likely: true,
                },
                None,
            ),
            &repository::resolve(&os()).unwrap(),
            true,
            false,
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("will not install repository packages")
        );
    }

    #[test]
    fn incompatible_managed_driver_is_upgraded_before_toolkit_install() {
        let mut current = status(managed(DriverFlavorState::Open, None), None);
        current.driver_version = Some("570.26".into());
        let plan = build_plan(
            &os(),
            &options(InstallProfile::CudaDevelopment, Some("13.3")),
            "6.8.0-generic",
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &repository::resolve(&os()).unwrap(),
            true,
            false,
        )
        .unwrap();
        let commands = plan
            .steps
            .iter()
            .map(|s| s.command.display())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(commands.contains("apt-get install -y nvidia-open"));
        assert!(
            commands.find("nvidia-open").unwrap() < commands.find("cuda-toolkit-13-3").unwrap()
        );
    }
}
