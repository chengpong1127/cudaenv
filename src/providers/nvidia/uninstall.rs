use anyhow::{Result, bail};

use crate::{
    model::{
        environment::ProviderStatus,
        operation::{OperationPlan, PlanDetail, PlanStep},
        system::{Distribution, OsInfo},
    },
    platform::package_manager,
};

const CUDA_TOOLKIT_PACKAGES: &[&str] = &[
    "cuda-toolkit*",
    "cuda-compiler*",
    "cuda-command-line-tools*",
    "cuda-demo-suite*",
    "cuda-documentation*",
    "cuda-libraries*",
    "cuda-nsight*",
    "cuda-nvcc*",
    "cuda-nvdisasm*",
    "cuda-nvml-dev*",
    "cuda-nvprof*",
    "cuda-nvprune*",
    "cuda-nvrtc*",
    "cuda-nvtx*",
    "cuda-opencl*",
    "cuda-profiler-api*",
    "cuda-sanitizer*",
    "cuda-tools*",
    "*cublas*",
    "*cufft*",
    "*cufile*",
    "*curand*",
    "*cusolver*",
    "*cusparse*",
    "*gds-tools*",
    "*npp*",
    "*nvjpeg*",
    "nsight*",
    "*nvvm*",
];

const NVIDIA_DRIVER_PACKAGES: &[&str] = &[
    "cuda-compat*",
    "cuda-drivers*",
    "libnvidia-cfg1*",
    "libnvidia-compute*",
    "libnvidia-decode*",
    "libnvidia-encode*",
    "libnvidia-extra*",
    "libnvidia-fbc1*",
    "libnvidia-gl*",
    "libnvidia-gpucomp*",
    "libnvidia-nscq*",
    "libnvsdm*",
    "libxnvctrl*",
    "nvidia-dkms*",
    "nvidia-driver*",
    "nvidia-fabricmanager*",
    "nvidia-firmware*",
    "nvidia-headless*",
    "nvidia-imex*",
    "nvidia-kernel*",
    "nvidia-modprobe*",
    "nvidia-open*",
    "nvidia-persistenced*",
    "nvidia-settings*",
    "nvidia-xconfig*",
    "xserver-xorg-video-nvidia*",
];

pub fn plan(os: &OsInfo, status: &ProviderStatus) -> Result<OperationPlan> {
    if os.distribution != Distribution::Ubuntu {
        bail!(
            "cudaenv uninstall supports NVIDIA packages on Ubuntu only (detected {}).",
            os.display_name()
        );
    }
    let toolkit_installed = !status.toolkits.is_empty();
    let driver_installed = status.driver_version.is_some();
    let mut steps = Vec::new();
    if toolkit_installed {
        steps.push(PlanStep::new(
            "could not remove CUDA Toolkit packages",
            package_manager::apt_remove_command(
                &["remove", "--purge", "--yes"],
                CUDA_TOOLKIT_PACKAGES,
            ),
        ));
    }
    if driver_installed {
        steps.push(PlanStep::new(
            "could not remove NVIDIA driver packages",
            package_manager::apt_remove_command(
                &["remove", "--autoremove", "--purge", "-V", "--yes"],
                NVIDIA_DRIVER_PACKAGES,
            ),
        ));
    }
    if toolkit_installed || driver_installed {
        steps.push(PlanStep::new(
            "could not clean up unused CUDA and NVIDIA dependencies",
            package_manager::apt_remove_command(&["autoremove", "--purge", "--yes"], &[]),
        ));
    }
    Ok(OperationPlan {
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
        ],
        devices: status.devices.clone(),
        steps,
        confirmation_warning:
            "This operation changes system packages and cannot be automatically undone.".into(),
        completion_message: "Detected CUDA/NVIDIA components were removed.".into(),
        reboot_message: driver_installed
            .then(|| "Reboot Ubuntu before installing another driver.".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{device::GpuVendor, environment::ToolkitStatus};

    fn os() -> OsInfo {
        OsInfo {
            distribution: Distribution::Ubuntu,
            name: "Ubuntu".into(),
            version_id: "24.04".into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        }
    }

    #[test]
    fn plan_uses_same_typed_commands_that_will_be_executed() {
        let status = ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![],
            driver_version: Some("570".into()),
            toolkits: vec![ToolkitStatus {
                name: "CUDA Toolkit".into(),
                version: "13.1".into(),
            }],
        };
        let plan = plan(&os(), &status).unwrap();
        assert_eq!(plan.steps.len(), 3);
        assert!(plan.steps[0].command.display().contains("cuda-toolkit*"));
        assert!(plan.steps[1].command.display().contains("nvidia-driver*"));
    }

    #[test]
    fn plan_only_removes_detected_components() {
        let status = ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![],
            driver_version: None,
            toolkits: vec![ToolkitStatus {
                name: "CUDA Toolkit".into(),
                version: "13.1".into(),
            }],
        };
        let plan = plan(&os(), &status).unwrap();
        let commands = plan
            .steps
            .iter()
            .map(|step| step.command.display())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(commands.contains("cuda-toolkit*"));
        assert!(!commands.contains("nvidia-driver*"));
    }
}
