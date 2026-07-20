use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::{
    model::{
        environment::{DriverInstallation, ProviderStatus},
        operation::{OperationPlan, PlanDetail, PlanStep},
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
        steps.push(PlanStep::new(
            "Purge every listed NVIDIA and CUDA package",
            package_manager::apt_remove_command(
                &["purge", "--yes"],
                &packages.iter().map(String::as_str).collect::<Vec<_>>(),
            ),
        ));
        steps.push(PlanStep::new(
            "Remove dependencies made unnecessary by the purge",
            crate::model::command::CommandSpec::sudo("apt-get", ["autoremove", "--purge", "--yes"]),
        ));
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
            PlanDetail::new(
                "Exact packages",
                packages.join(", "),
            ),
        ],
        devices: status.devices.clone(),
        steps,
        confirmation_warning:
            "Every package above will be purged; apt autoremove will then remove dependencies that are no longer required."
                .into(),
        completion_message: "Detected CUDA/NVIDIA components were removed.".into(),
        reboot_message: driver_installed
            .then(|| "Reboot Ubuntu before installing another driver.".into()),
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
}
