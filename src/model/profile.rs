use clap::ValueEnum;

use crate::model::{
    device::GpuVendor,
    environment::{DriverInstallation, DriverRuntimeState, ProviderStatus},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum InstallProfile {
    /// Set up model training with frameworks such as PyTorch, TensorFlow, or JAX.
    ModelTraining,
    /// Set up native CUDA development.
    CudaDevelopment,
}

impl InstallProfile {
    /// Backwards-compatible label used by the interactive profile selector.
    pub fn label(self) -> &'static str {
        self.selection_label()
    }

    pub fn plan_label(self) -> &'static str {
        match self {
            Self::ModelTraining => "Model training (PyTorch, TensorFlow, JAX)",
            Self::CudaDevelopment => "CUDA development",
        }
    }

    pub fn selection_label(self) -> &'static str {
        match self {
            Self::ModelTraining => "Model training     PyTorch, TensorFlow, or JAX",
            Self::CudaDevelopment => "CUDA development   Native CUDA apps and custom kernels",
        }
    }

    pub fn readiness(self, providers: &[ProviderStatus]) -> ProfileReadiness {
        let status = providers
            .iter()
            .find(|status| status.vendor == GpuVendor::Nvidia);
        let driver_available = status.is_some_and(|status| {
            matches!(
                status.driver,
                DriverInstallation::Managed { .. }
                    | DriverInstallation::Unmanaged { working: true, .. }
            )
        });
        let runtime_operational = status.is_some_and(|status| {
            status.driver_runtime_operational
                && status.driver_runtime_state == DriverRuntimeState::Operational
        });
        let toolkit_available = status.is_some_and(|status| !status.toolkits.is_empty());
        let nvcc_available = status.is_some_and(|status| {
            status
                .active_toolkit
                .as_ref()
                .is_some_and(|toolkit| toolkit.version.is_some())
        });

        let mut missing = Vec::new();
        if !driver_available {
            missing.push("NVIDIA driver");
        }
        if !runtime_operational {
            missing.push("operational NVIDIA driver runtime");
        }
        if self == Self::CudaDevelopment {
            if !toolkit_available {
                missing.push("CUDA Toolkit");
            }
            if !nvcc_available {
                missing.push("nvcc");
            }
        }

        let components_missing =
            !driver_available || (self == Self::CudaDevelopment && !toolkit_available);
        let next_action = if missing.is_empty() {
            None
        } else if !components_missing {
            Some(match self {
                Self::ModelTraining => "arc doctor --profile model-training",
                Self::CudaDevelopment => "arc doctor --profile cuda-development",
            })
        } else {
            Some("arc install")
        };

        ProfileReadiness {
            profile: self,
            missing,
            next_action,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileReadiness {
    pub profile: InstallProfile,
    pub missing: Vec<&'static str>,
    pub next_action: Option<&'static str>,
}

impl ProfileReadiness {
    pub fn ready(&self) -> bool {
        self.missing.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::environment::{
        DriverFlavorState, DriverPackageScope, ToolkitSource, ToolkitStatus,
    };

    fn operational_status() -> ProviderStatus {
        ProviderStatus {
            vendor: crate::model::device::GpuVendor::Nvidia,
            devices: vec![],
            driver: DriverInstallation::Managed {
                flavor: DriverFlavorState::Open,
                scope: DriverPackageScope::Full,
                branch: Some(610),
                packages: vec!["nvidia-open".into()],
            },
            driver_version: Some("610.1".into()),
            driver_runtime_operational: true,
            driver_runtime_state: DriverRuntimeState::Operational,
            dkms_status: None,
            driver_module: None,
            kernel_version: None,
            secure_boot_enabled: None,
            toolkits: vec![],
            active_toolkit: None,
        }
    }

    #[test]
    fn model_training_only_requires_an_operational_driver() {
        let readiness = InstallProfile::ModelTraining.readiness(&[operational_status()]);
        assert!(readiness.ready());
    }

    #[test]
    fn cuda_development_requires_toolkit_packages_and_active_nvcc() {
        let mut status = operational_status();
        let readiness = InstallProfile::CudaDevelopment.readiness(&[status.clone()]);
        assert_eq!(readiness.missing, ["CUDA Toolkit", "nvcc"]);

        let toolkit = ToolkitStatus {
            name: "CUDA Toolkit".into(),
            version: Some("13.3".into()),
            executable_path: Some("/usr/local/cuda/bin/nvcc".into()),
            source: ToolkitSource::SystemPackageManager,
            packages: vec!["cuda-toolkit-13-3".into()],
            manageable: true,
        };
        status.toolkits.push(toolkit.clone());
        status.active_toolkit = Some(ToolkitStatus {
            source: ToolkitSource::ActivePath,
            ..toolkit
        });
        assert!(InstallProfile::CudaDevelopment.readiness(&[status]).ready());
    }

    #[test]
    fn broken_runtime_recommends_doctor() {
        let mut status = operational_status();
        status.driver_runtime_operational = false;
        status.driver_runtime_state = DriverRuntimeState::SecureBootBlocked;
        let readiness = InstallProfile::ModelTraining.readiness(&[status]);
        assert_eq!(readiness.missing, ["operational NVIDIA driver runtime"]);
        assert_eq!(
            readiness.next_action,
            Some("arc doctor --profile model-training")
        );
    }

    #[test]
    fn missing_components_only_recommend_plain_arc_install() {
        for profile in [
            InstallProfile::ModelTraining,
            InstallProfile::CudaDevelopment,
        ] {
            let readiness = profile.readiness(&[]);
            assert_eq!(readiness.next_action, Some("arc install"));
        }
    }
}
