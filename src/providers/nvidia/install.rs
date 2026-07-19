use anyhow::{Result, bail};

use crate::{
    model::{
        environment::ProviderStatus,
        operation::{OperationPlan, PlanDetail, PlanStep},
        system::{Distribution, OsInfo},
    },
    platform::package_manager,
    providers::nvidia::{
        driver::{self, DriverFlavor, DriverPreference},
        gpu::{self, NvidiaGpu},
        repository::{self, NvidiaRepository},
        toolkit,
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

#[derive(Clone, Copy, Debug, Default)]
struct PackageState {
    open_driver: bool,
    proprietary_driver: bool,
    toolkit: bool,
}

pub fn plan(os: &OsInfo, options: &InstallOptions) -> Result<OperationPlan> {
    os.ensure_driver_installable("NVIDIA")?;
    let gpus = gpu::detect()?;
    if gpus.is_empty() {
        bail!("No NVIDIA GPU was detected. Check that the GPU is visible and try again.");
    }
    let repository = repository::resolve(os)?;
    let repository_configured = repository::is_configured(os, &repository)?;
    let status = ProviderStatus {
        vendor: crate::model::device::GpuVendor::Nvidia,
        devices: gpus.clone().into_iter().map(Into::into).collect(),
        driver_installed: false,
        driver_version: driver::detect_version()?,
        toolkits: toolkit::detect_version()?
            .map(|version| crate::model::environment::ToolkitStatus {
                name: "CUDA Toolkit".into(),
                version,
            })
            .into_iter()
            .collect(),
    };
    let manager = os.package_manager();
    let requested_toolkit = match options.profile {
        InstallProfile::ModelTraining => None,
        InstallProfile::CudaDevelopment => {
            Some(toolkit::package(options.toolkit_version.as_deref())?)
        }
    };
    let packages = PackageState {
        open_driver: package_manager::is_installed(manager, DriverFlavor::Open.package())?,
        proprietary_driver: package_manager::is_installed(
            manager,
            DriverFlavor::Proprietary.package(),
        )?,
        toolkit: match requested_toolkit.as_deref() {
            Some(package) => package_manager::is_installed(manager, package)?,
            None => false,
        },
    };
    let mut status = status;
    status.driver_installed = packages.open_driver || packages.proprietary_driver;
    build_plan(
        os,
        options,
        &gpus,
        &status,
        &repository,
        repository_configured,
        packages,
    )
}

fn build_plan(
    os: &OsInfo,
    options: &InstallOptions,
    gpus: &[NvidiaGpu],
    status: &ProviderStatus,
    repository: &NvidiaRepository,
    repository_configured: bool,
    packages: PackageState,
) -> Result<OperationPlan> {
    let manager = os.package_manager();
    let flavor = match (
        options.driver,
        packages.open_driver,
        packages.proprietary_driver,
    ) {
        (DriverPreference::Auto, true, false) => DriverFlavor::Open,
        (DriverPreference::Auto, false, true) => DriverFlavor::Proprietary,
        _ => driver::select(options.driver, gpus)?,
    };
    if os.distribution == Distribution::AzureLinux && flavor == DriverFlavor::Proprietary {
        bail!("Azure Linux supports only NVIDIA open kernel modules; use --driver open.");
    }
    let toolkit_package = match options.profile {
        InstallProfile::ModelTraining => None,
        InstallProfile::CudaDevelopment => {
            Some(toolkit::package(options.toolkit_version.as_deref())?)
        }
    };
    let current_toolkit = status
        .toolkits
        .first()
        .map(|toolkit| toolkit.version.as_str());
    let selected_driver_installed = match flavor {
        DriverFlavor::Open => packages.open_driver,
        DriverFlavor::Proprietary => packages.proprietary_driver,
    };
    let other_driver_installed = match flavor {
        DriverFlavor::Open => packages.proprietary_driver,
        DriverFlavor::Proprietary => packages.open_driver,
    };
    let install_driver = !selected_driver_installed;
    let install_toolkit = toolkit_package
        .as_deref()
        .is_some_and(|package| toolkit_install_needed(package, current_toolkit, packages.toolkit));
    let mut steps = Vec::new();
    if install_driver || install_toolkit {
        if !repository_configured {
            steps.extend(
                repository::setup_commands(manager, repository)
                    .into_iter()
                    .map(|command| PlanStep::new("Configure the NVIDIA CUDA repository", command)),
            );
        }
        steps.push(PlanStep::new(
            "Refresh package metadata",
            package_manager::refresh_command(manager),
        ));
        if install_toolkit && let Some(package) = toolkit_package.as_deref() {
            steps.push(PlanStep::new(
                format!("Verify CUDA Toolkit package {package} is available"),
                package_manager::query_command(manager, package),
            ));
        }
        if install_driver {
            steps.extend(
                driver::preparation_commands(os, flavor)
                    .into_iter()
                    .map(|command| {
                        PlanStep::new("Select the NVIDIA driver package stream", command)
                    }),
            );
            steps.push(PlanStep::new(
                format!(
                    "Verify NVIDIA driver package {} is available",
                    flavor.package()
                ),
                package_manager::query_command(manager, flavor.package()),
            ));
            steps.push(PlanStep::new(
                "Install the NVIDIA driver",
                package_manager::install_command_with_options(
                    manager,
                    flavor.package(),
                    other_driver_installed,
                ),
            ));
        }
        if install_toolkit && let Some(package) = toolkit_package.as_deref() {
            steps.push(PlanStep::new(
                format!("Install {package}"),
                package_manager::install_command(manager, package),
            ));
            steps.push(PlanStep::new(
                "Verify the CUDA Toolkit with nvcc",
                toolkit::verification_command(),
            ));
        }
    }
    let driver_detail = if install_driver && other_driver_installed {
        format!("switch to {}", flavor.package())
    } else if install_driver {
        format!("install {}", flavor.package())
    } else {
        format!(
            "{} already installed{} — skipped",
            flavor.package(),
            status
                .driver_version
                .as_deref()
                .map(|version| format!(" and loaded ({version})"))
                .unwrap_or_default()
        )
    };
    let toolkit_detail = match (toolkit_package.as_deref(), install_toolkit) {
        (Some(package), true) => format!("install {package}"),
        (Some(_), false) => "requested version already installed — skipped".into(),
        (None, _) => "not requested".into(),
    };
    Ok(OperationPlan {
        title: "NVIDIA Installation Plan".into(),
        details: vec![
            PlanDetail::new("OS", os.display_name()),
            PlanDetail::new("Package manager", manager.to_string()),
            PlanDetail::new("Repository", repository.base_url.clone()),
            PlanDetail::new("Profile", options.profile.label()),
            PlanDetail::new("Driver", driver_detail),
            PlanDetail::new("CUDA Toolkit", toolkit_detail),
            PlanDetail::new(
                "Kernel headers",
                if driver::kernel_headers_available() {
                    "detected for the running kernel"
                } else {
                    "not detected; the package manager must install matching headers"
                },
            ),
            PlanDetail::new(
                "Secure Boot",
                match driver::secure_boot_enabled() {
                    Some(true) => "enabled; module signing or key enrollment may be required",
                    Some(false) => "disabled",
                    None => "state unavailable",
                },
            ),
        ],
        devices: status.devices.clone(),
        steps,
        confirmation_warning: "No system changes will be made until you confirm.".into(),
        completion_message: "Installation completed.".into(),
        reboot_message: install_driver.then(|| "Reboot to load the NVIDIA driver.".into()),
    })
}

fn toolkit_install_needed(
    package: &str,
    current_version: Option<&str>,
    requested_package_installed: bool,
) -> bool {
    if requested_package_installed {
        return false;
    }
    if package == "cuda-toolkit" {
        return true;
    }
    let Some(current_version) = current_version else {
        return true;
    };
    toolkit::versioned_package(current_version)
        .map(|current_package| current_package != package)
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{device::GpuVendor, system::PackageManager};
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
    fn gpu() -> NvidiaGpu {
        NvidiaGpu {
            name: "NVIDIA GPU".into(),
            pci_device_id: None,
            generation: Generation::TuringOrNewer,
        }
    }
    fn status(driver: Option<&str>, toolkit: Option<&str>) -> ProviderStatus {
        ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![gpu().into()],
            driver_installed: driver.is_some(),
            driver_version: driver.map(str::to_owned),
            toolkits: toolkit
                .map(|version| crate::model::environment::ToolkitStatus {
                    name: "CUDA Toolkit".into(),
                    version: version.into(),
                })
                .into_iter()
                .collect(),
        }
    }
    fn repository() -> NvidiaRepository {
        repository::resolve(&os()).unwrap()
    }

    #[test]
    fn model_training_plan_has_driver_only() {
        let plan = build_plan(
            &os(),
            &InstallOptions {
                profile: InstallProfile::ModelTraining,
                toolkit_version: None,
                driver: DriverPreference::Auto,
            },
            &[gpu()],
            &status(None, None),
            &repository(),
            true,
            PackageState::default(),
        )
        .unwrap();
        let commands = plan
            .steps
            .iter()
            .map(|step| step.command.display())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(commands.contains("nvidia-open"));
        assert!(!commands.contains("cuda-toolkit"));
        assert_eq!(os().package_manager(), PackageManager::AptGet);
    }

    #[test]
    fn matching_installation_is_a_noop() {
        let plan = build_plan(
            &os(),
            &InstallOptions {
                profile: InstallProfile::CudaDevelopment,
                toolkit_version: Some("13.1".into()),
                driver: DriverPreference::Auto,
            },
            &[gpu()],
            &status(Some("570"), Some("13.1")),
            &repository(),
            true,
            PackageState {
                open_driver: true,
                toolkit: true,
                ..PackageState::default()
            },
        )
        .unwrap();
        assert!(plan.is_noop());
    }

    #[test]
    fn cuda_development_plan_installs_and_verifies_toolkit() {
        let plan = build_plan(
            &os(),
            &InstallOptions {
                profile: InstallProfile::CudaDevelopment,
                toolkit_version: Some("13.3".into()),
                driver: DriverPreference::Auto,
            },
            &[gpu()],
            &status(None, None),
            &repository(),
            true,
            PackageState::default(),
        )
        .unwrap();
        let commands = plan
            .steps
            .iter()
            .map(|step| step.command.display())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(commands.contains("cuda-toolkit-13-3"));
        assert!(commands.contains("nvidia-open"));
        assert!(commands.contains("nvcc --version"));
    }

    #[test]
    fn unversioned_toolkit_meta_package_is_idempotent() {
        let plan = build_plan(
            &os(),
            &InstallOptions {
                profile: InstallProfile::CudaDevelopment,
                toolkit_version: None,
                driver: DriverPreference::Auto,
            },
            &[gpu()],
            &status(Some("610"), Some("13.3")),
            &repository(),
            true,
            PackageState {
                open_driver: true,
                toolkit: true,
                ..PackageState::default()
            },
        )
        .unwrap();
        assert!(plan.is_noop());
    }
}
