use crate::model::environment::{DriverInstallation, DriverRuntimeState};

pub fn status_warning(state: DriverRuntimeState) -> &'static str {
    match state {
        DriverRuntimeState::RebootLikelyRequired => {
            "The NVIDIA driver is installed, but the matching module for the running kernel is not loaded. This commonly happens immediately after installing or upgrading the driver. Run `sudo reboot`, then rerun `arc status`."
        }
        DriverRuntimeState::DkmsModuleMissing => {
            "The NVIDIA driver is installed, but DKMS has not installed the matching module for the running kernel. Rebuild the module or reinstall the driver, then run `arc doctor`."
        }
        DriverRuntimeState::SecureBootBlocked => {
            "Secure Boot appears to be blocking the NVIDIA kernel module. Run `arc doctor` for the Secure Boot-specific fix."
        }
        DriverRuntimeState::Failed => {
            "The NVIDIA driver is installed but not currently operational. Run `sudo modprobe nvidia`, inspect the kernel logs, and rerun `arc doctor`."
        }
        DriverRuntimeState::Operational => "",
    }
}

pub fn module_problem(state: DriverRuntimeState) -> Option<&'static str> {
    match state {
        DriverRuntimeState::Operational => None,
        DriverRuntimeState::RebootLikelyRequired => Some(
            "The matching NVIDIA DKMS module is installed for the running kernel but is not loaded. This commonly happens immediately after a driver installation or upgrade; a reboot is likely required.",
        ),
        DriverRuntimeState::DkmsModuleMissing => Some(
            "DKMS has not installed the matching NVIDIA module for the running kernel. Rebuild the module or reinstall the driver.",
        ),
        DriverRuntimeState::SecureBootBlocked => {
            Some("Secure Boot appears to be blocking the unsigned NVIDIA kernel module.")
        }
        DriverRuntimeState::Failed => Some(
            "The NVIDIA kernel module is not loaded. If the machine has already been rebooted, try `sudo modprobe nvidia` and inspect the kernel logs.",
        ),
    }
}

pub struct RuntimeEvidence<'a> {
    pub driver: &'a DriverInstallation,
    pub driver_version: Option<&'a str>,
    pub module_loaded: bool,
    pub runtime_operational: bool,
    pub kernel_release: Option<&'a str>,
    pub dkms_status: Option<&'a str>,
    pub secure_boot_enabled: Option<bool>,
    pub module_signed: bool,
}

pub fn classify(evidence: RuntimeEvidence<'_>) -> DriverRuntimeState {
    if evidence.module_loaded && evidence.runtime_operational {
        return DriverRuntimeState::Operational;
    }

    let managed = matches!(evidence.driver, DriverInstallation::Managed { .. });
    let dkms_installed = evidence
        .driver_version
        .zip(evidence.kernel_release)
        .is_some_and(|(version, kernel)| {
            dkms_module_installed(evidence.dkms_status, version, kernel)
        });

    if managed && !evidence.module_loaded && dkms_installed {
        if evidence.secure_boot_enabled == Some(true) && !evidence.module_signed {
            DriverRuntimeState::SecureBootBlocked
        } else {
            DriverRuntimeState::RebootLikelyRequired
        }
    } else if managed && !evidence.module_loaded {
        DriverRuntimeState::DkmsModuleMissing
    } else {
        DriverRuntimeState::Failed
    }
}

fn dkms_module_installed(status: Option<&str>, driver_version: &str, kernel: &str) -> bool {
    status.is_some_and(|status| {
        status.lines().any(|line| {
            let lower = line.to_ascii_lowercase();
            let nvidia = lower
                .split_once('/')
                .is_some_and(|(module, _)| module.trim().contains("nvidia"));
            nvidia
                && line.contains(driver_version)
                && line.contains(kernel)
                && lower
                    .rsplit_once(':')
                    .is_some_and(|(_, state)| state.split_whitespace().next() == Some("installed"))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::environment::{DriverFlavorState, DriverPackageScope};

    fn managed() -> DriverInstallation {
        DriverInstallation::Managed {
            flavor: DriverFlavorState::Open,
            scope: DriverPackageScope::Full,
            branch: Some(580),
            packages: vec!["nvidia-open".into()],
        }
    }

    fn evidence<'a>(driver: &'a DriverInstallation) -> RuntimeEvidence<'a> {
        RuntimeEvidence {
            driver,
            driver_version: Some("580.65.06"),
            module_loaded: false,
            runtime_operational: false,
            kernel_release: Some("6.8.0-generic"),
            dkms_status: Some("nvidia/580.65.06, 6.8.0-generic, x86_64: installed"),
            secure_boot_enabled: Some(false),
            module_signed: true,
        }
    }

    #[test]
    fn freshly_installed_managed_driver_likely_needs_reboot() {
        let driver = managed();
        assert_eq!(
            classify(evidence(&driver)),
            DriverRuntimeState::RebootLikelyRequired
        );
    }

    #[test]
    fn missing_matching_dkms_module_is_a_real_failure() {
        let driver = managed();
        let mut evidence = evidence(&driver);
        evidence.dkms_status = Some("nvidia/580.65.06, 6.5.0-old, x86_64: installed");
        assert_eq!(classify(evidence), DriverRuntimeState::DkmsModuleMissing);
    }

    #[test]
    fn unsigned_module_with_secure_boot_is_blocked() {
        let driver = managed();
        let mut evidence = evidence(&driver);
        evidence.secure_boot_enabled = Some(true);
        evidence.module_signed = false;
        assert_eq!(classify(evidence), DriverRuntimeState::SecureBootBlocked);
    }

    #[test]
    fn loaded_working_driver_is_operational() {
        let driver = managed();
        let mut evidence = evidence(&driver);
        evidence.module_loaded = true;
        evidence.runtime_operational = true;
        assert_eq!(classify(evidence), DriverRuntimeState::Operational);
    }
}
