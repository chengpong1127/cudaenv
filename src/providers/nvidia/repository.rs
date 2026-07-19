use std::{fs, path::Path};

use anyhow::{Context, Result, bail};

use crate::model::{
    command::CommandSpec,
    system::{Distribution, OsInfo, PackageManager},
};

const CUDA_KEYRING_PACKAGE: &str = "cuda-keyring_1.1-1_all.deb";
const CUDA_KEYRING_PATH: &str = "/tmp/cuda-keyring_1.1-1_all.deb";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaRepository {
    pub distro: String,
    pub architecture: String,
    pub base_url: String,
}

pub fn resolve(os: &OsInfo) -> Result<NvidiaRepository> {
    let major = version_major(&os.version_id);
    let distro = match os.distribution {
        Distribution::Ubuntu if matches!(release(&os.version_id), "22.04" | "24.04" | "26.04") => {
            format!("ubuntu{}", release(&os.version_id).replace('.', ""))
        }
        Distribution::Debian if matches!(major, Some(12 | 13)) => {
            format!("debian{}", major.unwrap())
        }
        Distribution::Rhel | Distribution::AlmaLinux | Distribution::RockyLinux
            if matches!(major, Some(8..=10)) =>
        {
            format!("rhel{}", major.unwrap())
        }
        Distribution::OracleLinux if matches!(major, Some(8 | 9)) => {
            format!("rhel{}", major.unwrap())
        }
        Distribution::Fedora if major == Some(44) => "fedora44".to_owned(),
        Distribution::AmazonLinux if major == Some(2023) => "amzn2023".to_owned(),
        Distribution::AzureLinux if major == Some(3) => "azl3".to_owned(),
        Distribution::OpenSuse if major == Some(15) => "opensuse15".to_owned(),
        Distribution::OpenSuse if major == Some(16) => "suse16".to_owned(),
        Distribution::Sles if major == Some(15) => "sles15".to_owned(),
        Distribution::Sles if major == Some(16) => "suse16".to_owned(),
        Distribution::KylinOs
            if major == Some(11)
                || os.version_id.to_ascii_uppercase().starts_with('V')
                    && version_major(&os.version_id[1..]) == Some(11) =>
        {
            "kylin11".to_owned()
        }
        _ => bail!(
            "NVIDIA does not publish an exact repository target for {}. Refusing to substitute another distribution or release.",
            os.display_name()
        ),
    };
    let architecture = match os.architecture.as_str() {
        "x86_64" | "amd64" => "x86_64",
        "aarch64" | "arm64" => "sbsa",
        architecture => bail!(
            "NVIDIA does not publish a supported CUDA repository for architecture {architecture} on {}",
            os.display_name()
        ),
    };
    let base_url = format!(
        "https://developer.download.nvidia.com/compute/cuda/repos/{distro}/{architecture}/"
    );
    Ok(NvidiaRepository {
        distro,
        architecture: architecture.to_owned(),
        base_url,
    })
}

pub fn is_configured(os: &OsInfo, repository: &NvidiaRepository) -> Result<bool> {
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

pub fn setup_commands(manager: PackageManager, repository: &NvidiaRepository) -> Vec<CommandSpec> {
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

fn release(version: &str) -> &str {
    let end = version
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit() || *c == '.')
        .map(|(i, c)| i + c.len_utf8())
        .last()
        .unwrap_or(0);
    &version[..end]
}

fn version_major(version: &str) -> Option<u32> {
    version
        .split(['.', ' ', '-'])
        .next()?
        .trim_start_matches(['v', 'V'])
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(distribution: Distribution, version: &str) -> OsInfo {
        OsInfo {
            distribution,
            name: "Test".into(),
            version_id: version.into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        }
    }

    #[test]
    fn resolves_official_repository_targets() {
        for (distribution, version, expected) in [
            (Distribution::Ubuntu, "24.04", "ubuntu2404"),
            (Distribution::Debian, "13", "debian13"),
            (Distribution::AlmaLinux, "10.1", "rhel10"),
            (Distribution::AzureLinux, "3.0", "azl3"),
            (Distribution::KylinOs, "V11", "kylin11"),
        ] {
            assert_eq!(
                resolve(&os(distribution, version)).unwrap().distro,
                expected
            );
        }
    }

    #[test]
    fn rejects_unpublished_release_instead_of_substituting() {
        assert!(resolve(&os(Distribution::Ubuntu, "25.10")).is_err());
    }

    #[test]
    fn generates_repository_setup_for_each_manager_family() {
        let repository = resolve(&os(Distribution::Ubuntu, "24.04")).unwrap();
        assert_eq!(setup_commands(PackageManager::AptGet, &repository).len(), 2);
        assert!(
            setup_commands(PackageManager::Dnf, &repository)[0]
                .display()
                .contains("/etc/yum.repos.d/")
        );
        assert!(
            setup_commands(PackageManager::Zypper, &repository)[0]
                .display()
                .contains("addrepo")
        );
    }
}
