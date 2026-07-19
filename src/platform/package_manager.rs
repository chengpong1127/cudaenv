use crate::model::{command::CommandSpec, system::PackageManager};

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
    match manager {
        PackageManager::AptGet => CommandSpec::sudo("apt-get", ["install", "-y", package]),
        PackageManager::Dnf => CommandSpec::sudo("dnf", ["install", "-y", package]),
        PackageManager::Tdnf => CommandSpec::sudo("tdnf", ["install", "-y", package]),
        PackageManager::Zypper => {
            CommandSpec::sudo("zypper", ["--non-interactive", "install", package])
        }
    }
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
    }
}
