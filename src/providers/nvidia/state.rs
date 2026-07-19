use std::{path::Path, process::Command};

use anyhow::{Context, Result};

use crate::model::{
    environment::{
        DriverFlavorState, DriverInstallation, DriverPackageScope, ProviderStatus, ToolkitStatus,
    },
    system::OsInfo,
};

use super::{driver, gpu, toolkit};

pub fn inspect(os: &OsInfo) -> Result<ProviderStatus> {
    let devices = gpu::detect()?;
    let driver_version = driver::detect_version()?;
    let packages = installed_packages(os)?;
    let module_loaded = Path::new("/sys/module/nvidia").exists();
    let runfile_likely = Path::new("/usr/bin/nvidia-uninstall").exists()
        || Path::new("/var/log/nvidia-installer.log").exists();
    let driver = classify_driver(
        &packages,
        driver_version.is_some() || module_loaded,
        runfile_likely,
    );
    let toolkits = toolkit::detect_version()?
        .map(|version| ToolkitStatus {
            name: "CUDA Toolkit".to_owned(),
            version,
        })
        .into_iter()
        .collect();
    Ok(ProviderStatus {
        vendor: crate::model::device::GpuVendor::Nvidia,
        devices: devices.into_iter().map(Into::into).collect(),
        driver,
        driver_version,
        toolkits,
    })
}

pub fn classify_driver(
    installed: &[String],
    runtime_working: bool,
    runfile_likely: bool,
) -> DriverInstallation {
    let packages = installed
        .iter()
        .filter(|package| is_nvidia_driver_package(package))
        .cloned()
        .collect::<Vec<_>>();
    if packages.is_empty() {
        return if runtime_working || runfile_likely {
            DriverInstallation::Unmanaged {
                working: runtime_working,
                runfile_likely,
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
    if !runtime_working {
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
            classify_driver(&[], false, false),
            DriverInstallation::Missing
        );
        assert!(matches!(
            classify_driver(&[], true, true),
            DriverInstallation::Unmanaged { working: true, .. }
        ));
        assert!(matches!(
            classify_driver(
                &["nvidia-driver-cuda".into(), "kmod-nvidia-open-dkms".into()],
                true,
                false
            ),
            DriverInstallation::Managed {
                scope: DriverPackageScope::ComputeOnly,
                flavor: DriverFlavorState::Open,
                ..
            }
        ));
        assert!(matches!(
            classify_driver(&["cuda-drivers".into()], false, false),
            DriverInstallation::BrokenManaged {
                flavor: DriverFlavorState::Proprietary,
                ..
            }
        ));
        assert!(matches!(
            classify_driver(
                &["cuda-drivers".into(), "nvidia-driver-pinning-580".into()],
                true,
                false
            ),
            DriverInstallation::Managed {
                branch: Some(580),
                ..
            }
        ));
    }
}
