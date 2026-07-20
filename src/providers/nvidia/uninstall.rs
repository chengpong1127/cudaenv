use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::{
    model::{
        environment::{DriverInstallation, ProviderStatus},
        operation::{NextStep, OperationPlan, PlanDetail, PlanStage, PlanStep},
        system::{Distribution, OsInfo},
    },
    platform::package_manager,
};

const NVIDIA_CUDA_PATTERNS: &[&str] = &[
    "cuda-*",
    "nvidia-*",
    "libnvidia-*",
    "xserver-xorg-video-nvidia*",
];

pub fn plan(os: &OsInfo, status: &ProviderStatus) -> Result<OperationPlan> {
    if os.distribution != Distribution::Ubuntu {
        bail!(
            "arc uninstall supports NVIDIA packages on Ubuntu only (detected {}).",
            os.display_name()
        );
    }
    if let DriverInstallation::Unmanaged { evidence, .. } = &status.driver {
        bail!(
            "An unmanaged NVIDIA driver was detected (evidence: {}). arc cannot safely uninstall it; use the original installation method or migrate it to Ubuntu packages first.",
            evidence
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    let packages = installed_apt_packages(NVIDIA_CUDA_PATTERNS)?;
    Ok(build_plan(status, packages))
}

fn build_plan(status: &ProviderStatus, packages: Vec<String>) -> OperationPlan {
    let package_count = packages.len();
    let toolkit_installed = packages
        .iter()
        .any(|p| p.starts_with("cuda-") && !p.starts_with("cuda-drivers"));
    let driver_installed = packages.iter().any(|p| {
        p.starts_with("nvidia-")
            || p.starts_with("libnvidia-")
            || p.starts_with("cuda-drivers")
            || p.starts_with("xserver-xorg-video-nvidia")
    });
    let mut steps = Vec::new();
    if !packages.is_empty() {
        let removal_stage = PlanStage::new(
            format!("Remove {package_count} NVIDIA and CUDA packages"),
            format!("Removing {package_count} NVIDIA and CUDA packages..."),
            format!("Removed {package_count} NVIDIA and CUDA packages"),
            format!("Could not remove {package_count} NVIDIA and CUDA packages"),
        );
        let dependencies_stage = PlanStage::new(
            "Remove unused dependencies",
            "Removing unused dependencies...",
            "Removed unused dependencies",
            "Could not remove unused dependencies",
        );
        steps.push(
            PlanStep::new(
                "Purge every listed NVIDIA and CUDA package",
                package_manager::apt_remove_command(
                    &["purge", "--yes"],
                    &packages.iter().map(String::as_str).collect::<Vec<_>>(),
                ),
            )
            .in_stage(&removal_stage),
        );
        steps.push(
            PlanStep::new(
                "Remove dependencies made unnecessary by the purge",
                crate::model::command::CommandSpec::sudo(
                    "apt-get",
                    ["autoremove", "--purge", "--yes"],
                ),
            )
            .in_stage(&dependencies_stage),
        );
    }
    OperationPlan {
        title: "NVIDIA Uninstall Plan".into(),
        details: vec![
            PlanDetail::new(
                "Driver",
                if driver_installed {
                    "remove"
                } else {
                    "not detected"
                },
            ),
            PlanDetail::new(
                "CUDA Toolkit",
                if toolkit_installed {
                    "remove"
                } else {
                    "not detected"
                },
            ),
            PlanDetail::new("Packages", package_summary(&packages)),
        ],
        devices: status.devices.clone(),
        steps,
        confirmation_warning: "This will remove the installed NVIDIA driver and CUDA components."
            .into(),
        completion_message: format!("Removed {package_count} NVIDIA and CUDA packages."),
        next_step: driver_installed.then_some(NextStep::RebootBeforeNvidiaInstall),
    }
}

fn package_summary(packages: &[String]) -> String {
    let count = packages.len();
    let noun = if count == 1 { "package" } else { "packages" };
    let mut preview = packages
        .iter()
        .take(3)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if count > 3 {
        preview.push_str(&format!(", +{} more", count - 3));
    }
    if preview.is_empty() {
        format!("{count} installed {noun}")
    } else {
        format!("{count} installed {noun}\n{preview}")
    }
}

fn installed_apt_packages(patterns: &[&str]) -> Result<Vec<String>> {
    let output = Command::new("dpkg-query")
        .args(["-W", "-f=${db:Status-Abbrev}\t${binary:Package}\\n"])
        .args(patterns)
        .output()
        .context("could not inspect installed CUDA/NVIDIA packages")?;
    let mut packages = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let (status, package) = line.split_once('\t')?;
            (status.starts_with("ii ")
                && patterns
                    .iter()
                    .any(|pattern| package_matches(pattern, package)))
            .then(|| package.to_owned())
        })
        .collect::<Vec<_>>();
    packages.sort();
    packages.dedup();
    Ok(packages)
}

fn package_matches(pattern: &str, package: &str) -> bool {
    let package = package.split(':').next().unwrap_or(package);
    match pattern {
        "cuda-*" => package.starts_with("cuda-"),
        "nvidia-*" => package.starts_with("nvidia-"),
        "libnvidia-*" => package.starts_with("libnvidia-"),
        "xserver-xorg-video-nvidia*" => package.starts_with("xserver-xorg-video-nvidia"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        device::GpuVendor,
        environment::{
            DriverFlavorState, DriverInstallation, DriverPackageScope, ToolkitSource, ToolkitStatus,
        },
    };

    fn managed() -> DriverInstallation {
        DriverInstallation::Managed {
            flavor: DriverFlavorState::Open,
            scope: DriverPackageScope::Full,
            branch: None,
            packages: vec!["nvidia-open".into()],
        }
    }

    #[test]
    fn plan_uses_same_typed_commands_that_will_be_executed() {
        let status = ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![],
            driver: managed(),
            driver_version: Some("570".into()),
            driver_runtime_operational: true,
            driver_runtime_state: crate::model::environment::DriverRuntimeState::Operational,
            dkms_status: None,
            driver_module: None,
            kernel_version: None,
            secure_boot_enabled: None,
            toolkits: vec![ToolkitStatus {
                name: "System-managed CUDA Toolkit".into(),
                version: Some("13.1".into()),
                executable_path: Some("/usr/local/cuda-13.1/bin/nvcc".into()),
                source: ToolkitSource::SystemPackageManager,
                packages: vec!["cuda-toolkit-13-1".into()],
                manageable: true,
            }],
            active_toolkit: None,
        };
        let plan = build_plan(
            &status,
            vec!["cuda-toolkit-13-1".into(), "nvidia-open".into()],
        );
        assert_eq!(plan.steps.len(), 2);
        assert!(
            plan.steps[0]
                .command
                .display()
                .contains("cuda-toolkit-13-1")
        );
        assert!(plan.steps[0].command.display().contains("nvidia-open"));
    }

    #[test]
    fn plan_only_removes_detected_components() {
        let status = ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![],
            driver: DriverInstallation::Missing,
            driver_version: None,
            driver_runtime_operational: false,
            driver_runtime_state: crate::model::environment::DriverRuntimeState::Failed,
            dkms_status: None,
            driver_module: None,
            kernel_version: None,
            secure_boot_enabled: None,
            toolkits: vec![ToolkitStatus {
                name: "System-managed CUDA Toolkit".into(),
                version: Some("13.1".into()),
                executable_path: Some("/usr/local/cuda-13.1/bin/nvcc".into()),
                source: ToolkitSource::SystemPackageManager,
                packages: vec!["cuda-toolkit-13-1".into()],
                manageable: true,
            }],
            active_toolkit: None,
        };
        let plan = build_plan(&status, vec!["cuda-toolkit-13-1".into()]);
        let commands = plan
            .steps
            .iter()
            .map(|step| step.command.display())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(commands.contains("cuda-toolkit-13-1"));
        assert!(!commands.contains("nvidia-open"));
    }

    #[test]
    fn removal_patterns_cover_complete_nvidia_and_cuda_families() {
        assert!(package_matches("cuda-*", "cuda-toolkit-13-1"));
        assert!(package_matches("nvidia-*", "nvidia-open-dkms"));
        assert!(package_matches("libnvidia-*", "libnvidia-compute-580"));
        assert!(!package_matches("cuda-*", "arc"));
    }

    #[test]
    fn package_summary_uses_count_and_short_preview() {
        let packages = [
            "cuda-keyring".into(),
            "nvidia-open".into(),
            "nvidia-driver-open".into(),
            "libnvidia-compute".into(),
            "libnvidia-gl".into(),
        ];
        assert_eq!(
            package_summary(&packages),
            "5 installed packages\ncuda-keyring, nvidia-open, nvidia-driver-open, +2 more"
        );
    }

    #[test]
    fn uninstall_summary_and_labels_are_deterministic_and_distribution_neutral() {
        let status = ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![],
            driver: managed(),
            driver_version: Some("610.43.02".into()),
            driver_runtime_operational: true,
            driver_runtime_state: crate::model::environment::DriverRuntimeState::Operational,
            dkms_status: None,
            driver_module: None,
            kernel_version: None,
            secure_boot_enabled: None,
            toolkits: vec![],
            active_toolkit: None,
        };
        let plan = build_plan(&status, vec!["cuda-keyring".into(), "nvidia-open".into()]);
        assert_eq!(
            plan.completion_message,
            "Removed 2 NVIDIA and CUDA packages."
        );
        assert_eq!(
            plan.steps[0].stage.success,
            "Removed 2 NVIDIA and CUDA packages"
        );
        assert_eq!(plan.next_step, Some(NextStep::RebootBeforeNvidiaInstall));
        assert!(!plan.completion_message.contains("Ubuntu"));
    }
}
