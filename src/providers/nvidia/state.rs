use std::{path::Path, process::Command};

use anyhow::{Context, Result};

use crate::model::{
    environment::{
        DriverFlavorState, DriverInstallation, DriverPackageScope, ProviderStatus,
        UnmanagedDriverEvidence,
    },
    system::OsInfo,
};

use super::{driver, gpu, runtime, toolkit};

pub fn inspect(os: &OsInfo) -> Result<ProviderStatus> {
    let devices = gpu::detect()?;
    let driver_inspection = driver::inspect()?;
    let driver_version = driver_inspection.runtime_version.clone();
    let packages = installed_packages(os)?;
    let module_loaded = driver_inspection.module_loaded;
    let module_metadata_available = driver_inspection.module_info.is_some();
    let runfile_uninstaller = Path::new("/usr/bin/nvidia-uninstall").exists();
    let installer_log = Path::new("/var/log/nvidia-installer.log").exists();
    let driver = classify_driver(
        &packages,
        driver_version.is_some(),
        module_loaded,
        module_metadata_available,
        runfile_uninstaller,
        installer_log,
    );
    let dkms_status = command_optional_stdout("dkms", &["status"]);
    let detected_driver_version = driver_inspection
        .runtime_version
        .as_deref()
        .or_else(|| driver_inspection.module_info.as_ref()?.version.as_deref());
    let driver_runtime_state = runtime::classify(runtime::RuntimeEvidence {
        driver: &driver,
        driver_version: detected_driver_version,
        module_loaded,
        runtime_operational: driver_inspection.runtime_operational,
        kernel_release: driver_inspection.kernel_version.as_deref(),
        dkms_status: dkms_status.as_deref(),
        secure_boot_enabled: driver_inspection.secure_boot_enabled,
        module_signed: driver_inspection
            .module_info
            .as_ref()
            .is_some_and(|module| module.signer.is_some() || module.signature_id.is_some()),
    });
    let active_toolkit = toolkit::detect_active()?;
    let toolkits = toolkit::managed_status(&packages, active_toolkit.as_ref());
    Ok(ProviderStatus {
        vendor: crate::model::device::GpuVendor::Nvidia,
        devices: devices.into_iter().map(Into::into).collect(),
        driver,
        driver_version,
        driver_runtime_operational: driver_inspection.runtime_operational,
        driver_runtime_state,
        dkms_status,
        driver_module: driver_inspection.module_info,
        kernel_version: driver_inspection.kernel_version,
        secure_boot_enabled: driver_inspection.secure_boot_enabled,
        toolkits,
        active_toolkit,
    })
}

fn command_optional_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().into())
}

pub fn classify_driver(
    installed: &[String],
    driver_version_detected: bool,
    module_loaded: bool,
    module_metadata_available: bool,
    runfile_uninstaller: bool,
    installer_log: bool,
) -> DriverInstallation {
    let runtime_working = driver_version_detected || module_loaded;
    let packages = installed
        .iter()
        .filter(|package| is_nvidia_driver_package(package))
        .cloned()
        .collect::<Vec<_>>();
    if packages.is_empty() {
        return if runtime_working || module_metadata_available || runfile_uninstaller {
            let mut evidence = Vec::new();
            if runfile_uninstaller {
                evidence.push(UnmanagedDriverEvidence::RunfileUninstaller);
            }
            if driver_version_detected {
                evidence.push(UnmanagedDriverEvidence::DriverVersion);
            }
            if module_loaded {
                evidence.push(UnmanagedDriverEvidence::LoadedModule);
            }
            if module_metadata_available {
                evidence.push(UnmanagedDriverEvidence::ModuleMetadata);
            }
            // The log supports other evidence, but is never sufficient by itself.
            if installer_log {
                evidence.push(UnmanagedDriverEvidence::InstallerLog);
            }
            DriverInstallation::Unmanaged {
                working: runtime_working,
                evidence,
            }
        } else {
            DriverInstallation::Missing
        };
    }
    let open = packages.iter().any(|p| {
        p.starts_with("nvidia-open")
            || p.starts_with("nvidia-kernel-open")
            || p.starts_with("kmod-nvidia-open")
            || p.starts_with("nvidia-open-driver")
            || p.ends_with("-open")
    });
    let proprietary_marker = packages.iter().any(|p| {
        p.starts_with("cuda-drivers")
            || p == "nvidia-kernel-dkms"
            || p.starts_with("kmod-nvidia-latest")
    });
    let proprietary = proprietary_marker
        || (!open
            && packages.iter().any(|p| {
                p.starts_with("nvidia-driver")
                    || p.starts_with("nvidia-compute-")
                    || p.starts_with("nvidia-video-")
            }));
    let flavor = match (open, proprietary) {
        (true, false) => DriverFlavorState::Open,
        (false, true) => DriverFlavorState::Proprietary,
        _ => DriverFlavorState::Mixed,
    };
    if !runtime_working && !module_metadata_available {
        return DriverInstallation::BrokenManaged { flavor, packages };
    }
    let compute = packages
        .iter()
        .any(|p| p == "nvidia-driver-cuda" || p.starts_with("nvidia-compute-"));
    let desktop = packages
        .iter()
        .any(|p| p == "nvidia-driver" || p.starts_with("nvidia-video-"));
    let scope = match (compute, desktop) {
        (true, false) => DriverPackageScope::ComputeOnly,
        (false, true) => DriverPackageScope::DesktopOnly,
        _ => DriverPackageScope::Full,
    };
    let branch = packages.iter().find_map(|p| branch_from_package(p));
    DriverInstallation::Managed {
        flavor,
        scope,
        branch,
        packages,
    }
}

pub fn installed_packages(os: &OsInfo) -> Result<Vec<String>> {
    let output = match os.package_manager() {
        crate::model::system::PackageManager::AptGet => Command::new("dpkg-query")
            .args(["-W", "-f=${db:Status-Abbrev}\t${binary:Package}\\n"])
            .output(),
        _ => Command::new("rpm")
            .args(["-qa", "--qf", "%{NAME}\\n"])
            .output(),
    }
    .context("could not inspect installed packages")?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    let mut result = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            if os.package_manager() == crate::model::system::PackageManager::AptGet {
                let (status, package) = line.split_once('\t')?;
                status
                    .starts_with("ii ")
                    .then(|| package.split(':').next().unwrap_or(package).to_owned())
            } else {
                Some(line.trim().to_owned())
            }
        })
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    result.sort();
    result.dedup();
    Ok(result)
}

pub fn is_nvidia_driver_package(package: &str) -> bool {
    [
        "nvidia-open",
        "cuda-drivers",
        "nvidia-driver",
        "nvidia-dkms",
        "nvidia-kernel",
        "kmod-nvidia",
        "nvidia-compute-",
        "nvidia-video-",
        "nvidia-open-driver",
    ]
    .iter()
    .any(|prefix| package == *prefix || package.starts_with(&format!("{prefix}-")))
}

fn branch_from_package(package: &str) -> Option<u32> {
    if let Some(value) = package.strip_prefix("nvidia-driver-pinning-") {
        return value.split('.').next()?.parse().ok();
    }
    package
        .split(['-', '.'])
        .find_map(|part| (part.len() == 3).then(|| part.parse().ok()).flatten())
        .filter(|branch: &u32| (400..700).contains(branch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinguishes_missing_unmanaged_scoped_broken_and_pinned_installs() {
        assert_eq!(
            classify_driver(&[], false, false, false, false, false),
            DriverInstallation::Missing
        );
        assert!(matches!(
            classify_driver(&[], true, false, false, true, false),
            DriverInstallation::Unmanaged { working: true, .. }
        ));
        assert!(matches!(
            classify_driver(
                &["nvidia-driver-cuda".into(), "kmod-nvidia-open-dkms".into()],
                true,
                false,
                true,
                false,
                false
            ),
            DriverInstallation::Managed {
                scope: DriverPackageScope::ComputeOnly,
                flavor: DriverFlavorState::Open,
                ..
            }
        ));
        assert!(matches!(
            classify_driver(&["cuda-drivers".into()], false, false, false, false, false),
            DriverInstallation::BrokenManaged {
                flavor: DriverFlavorState::Proprietary,
                ..
            }
        ));
        assert!(matches!(
            classify_driver(
                &["cuda-drivers".into(), "nvidia-driver-pinning-580".into()],
                true,
                false,
                true,
                false,
                false
            ),
            DriverInstallation::Managed {
                branch: Some(580),
                ..
            }
        ));
    }

    #[test]
    fn installed_open_module_waiting_for_reboot_is_managed_not_broken() {
        let installation = classify_driver(
            &[
                "nvidia-driver-610-open".into(),
                "nvidia-dkms-610-open".into(),
                "nvidia-kernel-source-610-open".into(),
            ],
            false,
            false,
            true,
            false,
            false,
        );

        assert!(matches!(
            installation,
            DriverInstallation::Managed {
                flavor: DriverFlavorState::Open,
                ..
            }
        ));
        assert!(is_nvidia_driver_package("nvidia-dkms-610-open"));
    }

    #[test]
    fn stale_installer_log_alone_is_missing() {
        assert_eq!(
            classify_driver(&[], false, false, false, false, true),
            DriverInstallation::Missing
        );
    }

    #[test]
    fn runfile_uninstaller_is_strong_unmanaged_evidence() {
        assert_eq!(
            classify_driver(&[], false, false, false, true, true),
            DriverInstallation::Unmanaged {
                working: false,
                evidence: vec![
                    UnmanagedDriverEvidence::RunfileUninstaller,
                    UnmanagedDriverEvidence::InstallerLog,
                ],
            }
        );
    }

    #[test]
    fn loaded_module_without_managed_packages_is_unmanaged() {
        assert_eq!(
            classify_driver(&[], false, true, false, false, false),
            DriverInstallation::Unmanaged {
                working: true,
                evidence: vec![UnmanagedDriverEvidence::LoadedModule],
            }
        );
    }

    #[test]
    fn packages_without_runtime_or_module_metadata_are_broken_managed() {
        assert_eq!(
            classify_driver(&["cuda-drivers".into()], false, false, false, false, true,),
            DriverInstallation::BrokenManaged {
                flavor: DriverFlavorState::Proprietary,
                packages: vec!["cuda-drivers".into()],
            }
        );
    }
}
