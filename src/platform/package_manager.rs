use std::process::Command;

use anyhow::{Context, Result};

use crate::model::{
    command::CommandSpec,
    system::{OsInfo, PackageManager},
};

pub fn refresh_command(manager: PackageManager) -> CommandSpec {
    match manager {
        PackageManager::AptGet => CommandSpec::sudo("apt-get", ["update"]),
        PackageManager::Dnf => CommandSpec::sudo("dnf", ["makecache"]),
        PackageManager::Tdnf => CommandSpec::sudo("tdnf", ["makecache"]),
        PackageManager::Zypper => CommandSpec::sudo("zypper", ["--non-interactive", "refresh"]),
    }
}

pub fn query_command(manager: PackageManager, package: &str) -> CommandSpec {
    match manager {
        PackageManager::AptGet => {
            CommandSpec::new("apt-cache", ["show", "--no-all-versions", package])
        }
        PackageManager::Dnf => CommandSpec::new("dnf", ["--quiet", "list", "--available", package]),
        PackageManager::Tdnf => CommandSpec::new("tdnf", ["list", "available", package]),
        PackageManager::Zypper => {
            CommandSpec::new("zypper", ["--non-interactive", "info", package])
        }
    }
}

pub fn install_command(manager: PackageManager, package: &str) -> CommandSpec {
    install_command_with_options(manager, package, false)
}

pub fn kernel_headers_install_command(os: &OsInfo, kernel_release: &str) -> CommandSpec {
    let package = match os.package_manager() {
        PackageManager::AptGet => format!("linux-headers-{kernel_release}"),
        PackageManager::Dnf | PackageManager::Tdnf => format!("kernel-devel-{kernel_release}"),
        PackageManager::Zypper => "kernel-devel".to_owned(),
    };
    install_command(os.package_manager(), &package)
}

pub fn install_command_with_options(
    manager: PackageManager,
    package: &str,
    allow_erasing: bool,
) -> CommandSpec {
    match manager {
        PackageManager::AptGet => CommandSpec::sudo("apt-get", ["install", "-y", package]),
        PackageManager::Dnf if allow_erasing => {
            CommandSpec::sudo("dnf", ["install", "-y", "--allowerasing", package])
        }
        PackageManager::Dnf => CommandSpec::sudo("dnf", ["install", "-y", package]),
        PackageManager::Tdnf => CommandSpec::sudo("tdnf", ["install", "-y", package]),
        PackageManager::Zypper => {
            CommandSpec::sudo("zypper", ["--non-interactive", "install", package])
        }
    }
}

pub fn reinstall_command(manager: PackageManager, package: &str) -> CommandSpec {
    match manager {
        PackageManager::AptGet => {
            CommandSpec::sudo("apt-get", ["install", "--reinstall", "-y", package])
        }
        PackageManager::Dnf => CommandSpec::sudo("dnf", ["reinstall", "-y", package]),
        PackageManager::Tdnf => CommandSpec::sudo("tdnf", ["reinstall", "-y", package]),
        PackageManager::Zypper => CommandSpec::sudo(
            "zypper",
            ["--non-interactive", "install", "--force", package],
        ),
    }
}

pub fn is_installed(manager: PackageManager, package: &str) -> Result<bool> {
    let (program, args): (&str, Vec<&str>) = match manager {
        PackageManager::AptGet => ("dpkg-query", vec!["-W", "-f=${Status}", package]),
        PackageManager::Dnf | PackageManager::Tdnf | PackageManager::Zypper => {
            ("rpm", vec!["-q", package])
        }
    };
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("could not query whether {package} is installed"))?;
    if !output.status.success() {
        return Ok(false);
    }
    Ok(manager != PackageManager::AptGet
        || String::from_utf8_lossy(&output.stdout).trim() == "install ok installed")
}

pub fn apt_remove_command(options: &[&str], packages: &[&str]) -> CommandSpec {
    CommandSpec::sudo(
        "apt",
        options.iter().copied().chain(packages.iter().copied()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_each_package_manager_pipeline() {
        for manager in [
            PackageManager::AptGet,
            PackageManager::Dnf,
            PackageManager::Tdnf,
            PackageManager::Zypper,
        ] {
            assert!(!refresh_command(manager).args.is_empty());
            assert!(
                query_command(manager, "gpu-sdk")
                    .display()
                    .contains("gpu-sdk")
            );
            assert!(
                install_command(manager, "gpu-sdk")
                    .display()
                    .contains("gpu-sdk")
            );
        }
        assert!(
            install_command_with_options(PackageManager::Dnf, "nvidia-open", true)
                .display()
                .contains("--allowerasing")
        );
    }
}
