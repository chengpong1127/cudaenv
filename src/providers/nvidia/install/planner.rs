use anyhow::{Result, bail};

use crate::{
    model::operation::{NextStep, OperationPlan, PlanDetail, PlanStage, PlanStep},
    platform::package_manager,
    providers::nvidia::{driver::DriverFlavor, recipe, repository, toolkit},
};

use super::{InstallContext, InstallDecision, InstallOptions};

pub(super) fn plan(
    context: &InstallContext,
    decision: &InstallDecision,
    options: &InstallOptions,
) -> Result<OperationPlan> {
    let os = &context.os;
    let kernel = &context.kernel;
    let status = &context.status;
    let repository = &context.repository;
    let policy = decision.policy;
    let recipe = recipe::resolve(os, kernel, policy)?;
    let broken_managed_packages = decision.broken_managed_packages.as_deref();
    let current_flavor = decision.current_flavor;
    let toolkit_package = decision.toolkit_package.as_deref();
    let install_toolkit = decision.install_toolkit;
    let install_driver = decision.install_driver;
    let driver_pending_activation = decision.driver_pending_activation;
    let transition = decision.transition_driver;
    let driver_name = match policy.flavor {
        DriverFlavor::Open => "NVIDIA Open driver",
        DriverFlavor::Proprietary => "NVIDIA proprietary driver",
    };
    let repository_stage = PlanStage::new(
        "Configure the NVIDIA CUDA repository",
        "Configuring the NVIDIA repository...",
        "Configured the NVIDIA repository",
        "Could not configure the NVIDIA repository",
    );
    let refresh_stage = PlanStage::new(
        "Refresh package metadata",
        "Refreshing package metadata...",
        "Refreshed package metadata",
        "Could not refresh package metadata",
    );
    let prerequisites_stage = PlanStage::new(
        "Install driver prerequisites",
        "Installing driver prerequisites...",
        "Installed driver prerequisites",
        "Could not install driver prerequisites",
    );
    let driver_stage = PlanStage::new(
        format!("Install the {driver_name}"),
        format!("Installing the {driver_name}..."),
        format!("Installed the {driver_name}"),
        format!("Could not install the {driver_name}"),
    );
    let driver_verification_stage = PlanStage::new(
        "Verify the installation",
        "Verifying the installation...",
        "Verified the installation",
        "Could not verify the installation",
    );
    let toolkit_stage = PlanStage::new(
        "Install the CUDA Toolkit",
        "Installing the CUDA Toolkit...",
        "Installed the CUDA Toolkit",
        "Could not install the CUDA Toolkit",
    );
    let toolkit_verification_stage = PlanStage::new(
        "Verify the CUDA Toolkit",
        "Verifying the CUDA Toolkit...",
        "Verified the CUDA Toolkit",
        "Could not verify the CUDA Toolkit",
    );
    let mut steps = Vec::new();
    if install_driver || install_toolkit {
        if !context.repository_configured {
            if !context.repository_downloader_available {
                bail!(
                    "Configuring the NVIDIA repository requires curl or wget, but neither command is available."
                );
            }
            steps.extend(
                repository::setup_commands(os.package_manager(), repository)
                    .into_iter()
                    .map(|command| {
                        PlanStep::new("Configure the NVIDIA CUDA repository", command)
                            .in_stage(&repository_stage)
                    }),
            );
        }
        steps.push(
            PlanStep::new(
                "Refresh package metadata",
                package_manager::refresh_command(os.package_manager()),
            )
            .in_stage(&refresh_stage),
        );
    }
    if install_driver {
        steps.extend(recipe.prerequisites.into_iter().map(|command| {
            PlanStep::new("Ensure NVIDIA driver prerequisites", command)
                .in_stage(&prerequisites_stage)
        }));
        steps.push(
            PlanStep::new(
                "Refresh package metadata after ensuring prerequisites",
                package_manager::refresh_command(os.package_manager()),
            )
            .in_stage(&prerequisites_stage),
        );
        if let Some(packages) = broken_managed_packages {
            if let Some(command) =
                package_manager::reinstall_command(os.package_manager(), packages)
            {
                steps.push(
                    PlanStep::new("Reinstall the detected NVIDIA driver packages", command)
                        .in_stage(&driver_stage),
                );
            }
            if packages.iter().any(|package| package.contains("dkms")) {
                steps.push(
                    PlanStep::new(
                        "Rebuild the NVIDIA module for the running kernel",
                        crate::model::command::CommandSpec::sudo(
                            "dkms",
                            ["autoinstall", "-k", kernel],
                        ),
                    )
                    .in_stage(&driver_stage),
                );
            }
        }
        if let Some(from) = current_flavor {
            steps.extend(
                recipe::transition_commands(os, policy, from)
                    .into_iter()
                    .map(|command| {
                        PlanStep::new("Transition the NVIDIA driver package stream", command)
                            .in_stage(&driver_stage)
                    }),
            );
        } else {
            steps.extend(recipe.driver_preparation.into_iter().map(|command| {
                PlanStep::new("Select the NVIDIA driver package stream", command)
                    .in_stage(&driver_stage)
            }));
            steps.push(
                PlanStep::new(
                    format!(
                        "Verify NVIDIA driver package {} is available",
                        policy.flavor.package()
                    ),
                    package_manager::query_command(os.package_manager(), policy.flavor.package()),
                )
                .in_stage(&driver_stage),
            );
            steps.push(
                PlanStep::new("Install the NVIDIA driver", recipe.driver_install)
                    .in_stage(&driver_stage),
            );
        }
        steps.push(
            PlanStep::new(
                "Verify that NVIDIA kernel module metadata is installed",
                recipe.driver_verification,
            )
            .in_stage(&driver_verification_stage),
        );
    }
    if install_toolkit && let Some(package) = toolkit_package {
        steps.push(
            PlanStep::new(
                format!("Verify CUDA Toolkit package {package} is available"),
                package_manager::query_command(os.package_manager(), package),
            )
            .in_stage(&toolkit_stage),
        );
        steps.push(
            PlanStep::new(
                format!("Install {package}"),
                package_manager::install_command(os.package_manager(), package),
            )
            .in_stage(&toolkit_stage),
        );
        steps.push(
            PlanStep::new(
                "Verify the CUDA Toolkit with nvcc",
                toolkit::verification_command(),
            )
            .in_stage(&toolkit_verification_stage),
        );
    }
    let driver_detail = if driver_pending_activation {
        format!(
            "{} installed; kernel module is ready but not loaded — reboot required",
            policy.flavor.package()
        )
    } else if !install_driver {
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
                    "repository-compatible {}; NVIDIA validated: {}",
                    repository.family,
                    if repository.nvidia_validated {
                        "yes"
                    } else {
                        "no"
                    }
                ),
            ),
            PlanDetail::new("Profile", options.profile.plan_label()),
            PlanDetail::new("Existing driver", status.driver.description()),
            PlanDetail::new("Driver", driver_detail),
            PlanDetail::new(
                "CUDA Toolkit",
                toolkit_package.map_or("not requested".into(), |p| {
                    if install_toolkit {
                        format!("install {p}")
                    } else {
                        "requested version already installed — skipped".into()
                    }
                }),
            ),
            PlanDetail::new(
                "Kernel headers",
                if context.kernel_headers_available {
                    "detected for running kernel"
                } else {
                    "install exact matching prerequisites before driver"
                },
            ),
        ],
        devices: status.devices.clone(),
        steps,
        confirmation_warning: String::new(),
        completion_message: match (install_driver, install_toolkit) {
            (true, true) => format!("{driver_name} and CUDA Toolkit installed and verified."),
            (true, false) => format!("{driver_name} installed and verified."),
            (false, true) => "CUDA Toolkit installed and verified.".into(),
            (false, false) => "Requested NVIDIA components are already installed.".into(),
        },
        next_step: (driver_pending_activation || install_driver)
            .then_some(NextStep::LoadNvidiaDriver),
    })
}
