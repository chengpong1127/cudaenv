use std::{fs, path::Path, process::Command};

use anyhow::{Context, Result, bail};

use crate::system::{
    command::CommandSpec,
    os::{OsInfo, PackageManager},
    repository::Repository,
};

const CUDA_KEYRING_PACKAGE: &str = "cuda-keyring_1.1-1_all.deb";
const CUDA_KEYRING_PATH: &str = "/tmp/cuda-keyring_1.1-1_all.deb";

pub fn versioned_package(version: &str) -> Result<String> {
    let normalized = version.trim().replace('.', "-");
    let mut parts = normalized.split('-');
    let (Some(major), Some(minor), None) = (parts.next(), parts.next(), parts.next()) else {
        bail!("invalid CUDA Toolkit version {version:?}; expected MAJOR.MINOR, for example 13.3");
    };
    if major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || !minor.bytes().all(|byte| byte.is_ascii_digit())
    {
        bail!("invalid CUDA Toolkit version {version:?}; expected MAJOR.MINOR, for example 13.3");
    }
    Ok(format!("cuda-toolkit-{major}-{minor}"))
}

/// Detect the exact NVIDIA repository URL in the package manager's configuration.
pub fn repository_is_configured(os: &OsInfo, repository: &Repository) -> Result<bool> {
    let roots: &[&str] = match os.package_manager() {
        PackageManager::AptGet => &["/etc/apt/sources.list", "/etc/apt/sources.list.d"],
        PackageManager::Dnf | PackageManager::Tdnf => &["/etc/yum.repos.d"],
        PackageManager::Zypper => &["/etc/zypp/repos.d"],
    };
    roots.iter().try_fold(false, |found, root| {
        Ok(found || path_contains_repository(Path::new(root), &repository.base_url)?)
    })
}

fn path_contains_repository(path: &Path, base_url: &str) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    if path.is_file() {
        let contents = fs::read_to_string(path).with_context(|| {
            format!(
                "could not inspect repository configuration at {}",
                path.display()
            )
        })?;
        return Ok(contents.contains(base_url) || contents.contains(base_url.trim_end_matches('/')));
    }
    for entry in fs::read_dir(path)
        .with_context(|| format!("could not inspect repository directory {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("could not inspect {}", path.display()))?;
        if entry
            .file_type()
            .map(|kind| kind.is_file())
            .unwrap_or(false)
            && path_contains_repository(&entry.path(), base_url)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn repository_setup_commands(
    manager: PackageManager,
    repository: &Repository,
) -> Vec<CommandSpec> {
    let repo_url = format!("{}cuda-{}.repo", repository.base_url, repository.distro);
    match manager {
        PackageManager::AptGet => vec![
            CommandSpec::new(
                "curl",
                [
                    "--fail",
                    "--location",
                    "--output",
                    CUDA_KEYRING_PATH,
                    &format!("{}{CUDA_KEYRING_PACKAGE}", repository.base_url),
                ],
            ),
            CommandSpec::sudo("dpkg", ["-i", CUDA_KEYRING_PATH]),
        ],
        PackageManager::Dnf | PackageManager::Tdnf => vec![CommandSpec::sudo(
            "curl",
            [
                "--fail",
                "--location",
                "--output",
                &format!("/etc/yum.repos.d/cuda-{}.repo", repository.distro),
                &repo_url,
            ],
        )],
        PackageManager::Zypper => vec![CommandSpec::sudo(
            "zypper",
            ["--non-interactive", "addrepo", &repo_url, "cuda-nvidia"],
        )],
    }
}

pub fn setup_repository(os: &OsInfo, repository: &Repository) -> Result<()> {
    if repository_is_configured(os, repository)? {
        return Ok(());
    }
    for command in repository_setup_commands(os.package_manager(), repository) {
        command
            .execute()
            .context("could not configure the NVIDIA CUDA network repository")?;
    }
    Ok(())
}

pub fn metadata_refresh_command(manager: PackageManager) -> CommandSpec {
    match manager {
        PackageManager::AptGet => CommandSpec::sudo("apt-get", ["update"]),
        PackageManager::Dnf => CommandSpec::sudo("dnf", ["makecache"]),
        PackageManager::Tdnf => CommandSpec::sudo("tdnf", ["makecache"]),
        PackageManager::Zypper => CommandSpec::sudo("zypper", ["--non-interactive", "refresh"]),
    }
}

pub fn refresh_metadata(manager: PackageManager) -> Result<()> {
    metadata_refresh_command(manager)
        .execute()
        .context("could not refresh package metadata")
}

pub fn package_availability_command(manager: PackageManager, package: &str) -> CommandSpec {
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

pub fn ensure_package_available(manager: PackageManager, package: &str) -> Result<()> {
    let command = package_availability_command(manager, package);
    let output = Command::new(&command.program)
        .args(&command.args)
        .output()
        .with_context(|| format!("could not query package metadata with {}", command.program))?;
    if !output.status.success() {
        bail!(
            "requested CUDA Toolkit package {package} is not available from the configured NVIDIA repository"
        );
    }
    Ok(())
}

pub fn package_install_command(manager: PackageManager, package: &str) -> CommandSpec {
    match manager {
        PackageManager::AptGet => CommandSpec::sudo("apt-get", ["install", "-y", package]),
        PackageManager::Dnf => CommandSpec::sudo("dnf", ["install", "-y", package]),
        PackageManager::Tdnf => CommandSpec::sudo("tdnf", ["install", "-y", package]),
        PackageManager::Zypper => {
            CommandSpec::sudo("zypper", ["--non-interactive", "install", package])
        }
    }
}

pub fn install_package(manager: PackageManager, package: &str) -> Result<()> {
    package_install_command(manager, package)
        .execute()
        .with_context(|| format!("could not install {package}"))
}

pub fn verification_command() -> CommandSpec {
    CommandSpec::new("nvcc", ["--version"])
}

pub fn verify_installation() -> Result<()> {
    verification_command().execute().context(
        "CUDA Toolkit package installation completed, but `nvcc --version` failed; ensure the toolkit bin directory is on PATH",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repository() -> Repository {
        Repository {
            distro: "ubuntu2404".into(),
            architecture: "x86_64".into(),
            base_url: "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
                .into(),
        }
    }

    #[test]
    fn generates_versioned_package_names() {
        assert_eq!(versioned_package("13.3").unwrap(), "cuda-toolkit-13-3");
        assert_eq!(versioned_package("12-8").unwrap(), "cuda-toolkit-12-8");
        for invalid in ["13", "13.3.0", "latest", "13.x", ""] {
            assert!(versioned_package(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn generates_network_repository_setup_for_each_manager() {
        let repository = repository();
        let apt = repository_setup_commands(PackageManager::AptGet, &repository);
        assert!(apt[0].display().contains("cuda-keyring_1.1-1_all.deb"));
        assert_eq!(apt.len(), 2);

        let dnf = repository_setup_commands(PackageManager::Dnf, &repository);
        assert!(dnf[0].display().contains("/etc/yum.repos.d/"));

        let tdnf = repository_setup_commands(PackageManager::Tdnf, &repository);
        assert!(tdnf[0].display().contains("/etc/yum.repos.d/"));

        let zypper = repository_setup_commands(PackageManager::Zypper, &repository);
        assert!(zypper[0].display().contains("addrepo"));
    }

    #[test]
    fn generates_each_pipeline_stage_for_all_managers() {
        for manager in [
            PackageManager::AptGet,
            PackageManager::Dnf,
            PackageManager::Tdnf,
            PackageManager::Zypper,
        ] {
            let package = "cuda-toolkit-13-3";
            assert!(!metadata_refresh_command(manager).args.is_empty());
            assert!(
                package_availability_command(manager, package)
                    .display()
                    .contains(package)
            );
            assert!(
                package_install_command(manager, package)
                    .display()
                    .contains(package)
            );
        }
        assert_eq!(verification_command().display(), "nvcc --version");
    }
}
