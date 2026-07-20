use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};

use crate::model::{
    command::CommandSpec,
    system::{Distribution, OsInfo, PackageManager},
};

const CUDA_KEYRING_PACKAGE: &str = "cuda-keyring_1.1-1_all.deb";

#[derive(Clone, Copy)]
enum ReleaseTargets {
    Exact(&'static [(&'static str, &'static str)]),
    Major(&'static [(u32, &'static str)]),
    MajorMinorAtLeast(&'static [(u32, u32, &'static str)]),
}

#[derive(Clone, Copy)]
struct SupportPolicy {
    label: &'static str,
    distributions: &'static [Distribution],
    releases: ReleaseTargets,
    architectures: &'static [&'static str],
    nvidia_validated: &'static [&'static str],
}

const X86: &[&str] = &["x86_64"];
const X86_SBSA: &[&str] = &["x86_64", "sbsa"];

/// Repository compatibility and NVIDIA validation are represented separately.
/// Only repository compatibility controls target resolution. Keep this matrix
/// aligned with Tables 1 and 2 of NVIDIA's current CUDA Linux installation guide:
/// https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements
const SUPPORT_POLICIES: &[SupportPolicy] = &[
    SupportPolicy {
        label: "Ubuntu",
        distributions: &[Distribution::Ubuntu],
        releases: ReleaseTargets::Exact(&[
            ("22.04", "ubuntu2204"),
            ("24.04", "ubuntu2404"),
            ("26.04", "ubuntu2604"),
        ]),
        architectures: X86_SBSA,
        nvidia_validated: &["22.04", "24.04", "26.04"],
    },
    SupportPolicy {
        label: "Debian",
        distributions: &[Distribution::Debian],
        releases: ReleaseTargets::Major(&[(12, "debian12"), (13, "debian13")]),
        architectures: X86,
        nvidia_validated: &["12", "13"],
    },
    SupportPolicy {
        label: "Red Hat Enterprise Linux",
        distributions: &[Distribution::Rhel],
        releases: ReleaseTargets::Major(&[(8, "rhel8"), (9, "rhel9"), (10, "rhel10")]),
        architectures: X86_SBSA,
        nvidia_validated: &["8.10", "9.8", "10.2"],
    },
    SupportPolicy {
        label: "AlmaLinux",
        distributions: &[Distribution::AlmaLinux],
        releases: ReleaseTargets::Major(&[(8, "rhel8"), (9, "rhel9"), (10, "rhel10")]),
        architectures: X86,
        nvidia_validated: &["8.10", "9.8", "10.2"],
    },
    SupportPolicy {
        label: "Rocky Linux",
        distributions: &[Distribution::RockyLinux],
        releases: ReleaseTargets::Major(&[(8, "rhel8"), (9, "rhel9"), (10, "rhel10")]),
        architectures: X86,
        nvidia_validated: &["8.10", "9.8", "10.2"],
    },
    SupportPolicy {
        label: "Oracle Linux",
        distributions: &[Distribution::OracleLinux],
        releases: ReleaseTargets::Major(&[(8, "rhel8"), (9, "rhel9")]),
        architectures: X86,
        nvidia_validated: &["8", "9"],
    },
    SupportPolicy {
        label: "Fedora",
        distributions: &[Distribution::Fedora],
        releases: ReleaseTargets::Exact(&[("44", "fedora44")]),
        architectures: X86,
        nvidia_validated: &["44"],
    },
    SupportPolicy {
        label: "Amazon Linux",
        distributions: &[Distribution::AmazonLinux],
        releases: ReleaseTargets::Major(&[(2023, "amzn2023")]),
        architectures: X86_SBSA,
        nvidia_validated: &["2023"],
    },
    SupportPolicy {
        label: "Azure Linux",
        distributions: &[Distribution::AzureLinux],
        releases: ReleaseTargets::Major(&[(3, "azl3")]),
        architectures: X86_SBSA,
        nvidia_validated: &["3.0"],
    },
    SupportPolicy {
        label: "openSUSE Leap",
        distributions: &[Distribution::OpenSuse],
        releases: ReleaseTargets::Exact(&[("15.6", "opensuse15"), ("16.0", "suse16")]),
        architectures: X86,
        nvidia_validated: &["15.6"],
    },
    SupportPolicy {
        label: "SLES",
        distributions: &[Distribution::Sles],
        releases: ReleaseTargets::MajorMinorAtLeast(&[(15, 6, "sles15"), (16, 0, "suse16")]),
        architectures: X86_SBSA,
        nvidia_validated: &["15.6", "15.7", "16.0"],
    },
    SupportPolicy {
        label: "KylinOS",
        distributions: &[Distribution::KylinOs],
        releases: ReleaseTargets::Exact(&[("V11", "kylin11"), ("V11 2503", "kylin11")]),
        architectures: X86_SBSA,
        nvidia_validated: &["V11", "V11 2503"],
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaRepository {
    pub distro: String,
    pub base_url: String,
    pub family: String,
    pub nvidia_validated: bool,
}

pub fn resolve(os: &OsInfo) -> Result<NvidiaRepository> {
    let policy = SUPPORT_POLICIES
        .iter()
        .find(|policy| policy.distributions.contains(&os.distribution))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no NVIDIA repository policy exists for {}",
                os.display_name()
            )
        })?;
    let distro = resolve_release_target(policy.releases, &os.version_id).ok_or_else(|| {
        anyhow::anyhow!(
            "NVIDIA does not publish a compatible {} repository target for {}. NVIDIA-validated releases: {}. Refusing to substitute another distribution family or release.",
            policy.label,
            os.display_name(),
            policy.nvidia_validated.join(", ")
        )
    })?;
    let architecture = match os.architecture.as_str() {
        "x86_64" | "amd64" => "x86_64",
        "aarch64" | "arm64" => "sbsa",
        architecture => architecture,
    };
    if !policy.architectures.contains(&architecture) {
        bail!(
            "NVIDIA does not publish a supported CUDA repository for architecture {} on {}",
            os.architecture,
            os.display_name()
        );
    }
    let base_url = format!(
        "https://developer.download.nvidia.com/compute/cuda/repos/{distro}/{architecture}/"
    );
    Ok(NvidiaRepository {
        distro,
        base_url,
        family: policy.label.into(),
        nvidia_validated: contains_release(policy.nvidia_validated, &os.version_id),
    })
}

fn contains_release(releases: &[&str], version: &str) -> bool {
    releases
        .iter()
        .any(|release| release.eq_ignore_ascii_case(version))
}

fn resolve_release_target(targets: ReleaseTargets, version: &str) -> Option<String> {
    match targets {
        ReleaseTargets::Exact(values) => values
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(version))
            .map(|(_, target)| (*target).to_owned()),
        ReleaseTargets::Major(values) => {
            let major = version_major(version)?;
            values
                .iter()
                .find(|(candidate, _)| *candidate == major)
                .map(|(_, target)| (*target).to_owned())
        }
        ReleaseTargets::MajorMinorAtLeast(values) => {
            let (major, minor) = version_major_minor(version)?;
            values
                .iter()
                .find(|(candidate, minimum_minor, _)| {
                    *candidate == major && minor >= *minimum_minor
                })
                .map(|(_, _, target)| (*target).to_owned())
        }
    }
}

fn version_major_minor(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

#[cfg(test)]
fn readme_support_table() -> String {
    let mut table = String::from(
        "| Distribution family | Compatible repository releases | NVIDIA validated | Architectures |\n| --- | --- | --- | --- |\n",
    );
    for policy in SUPPORT_POLICIES {
        let compatible = match policy.releases {
            ReleaseTargets::Exact(values) => values
                .iter()
                .map(|(release, _)| *release)
                .collect::<Vec<_>>()
                .join(", "),
            ReleaseTargets::Major(values) => values
                .iter()
                .map(|(major, _)| format!("{major}.x"))
                .collect::<Vec<_>>()
                .join(", "),
            ReleaseTargets::MajorMinorAtLeast(values) => values
                .iter()
                .map(|(major, minor, _)| format!("{major}.{minor}+"))
                .collect::<Vec<_>>()
                .join(", "),
        };
        table.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            policy.label,
            compatible,
            policy.nvidia_validated.join(", "),
            policy.architectures.join(", ")
        ));
    }
    table
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
        let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
            return Ok(false);
        };
        if !matches!(extension, "list" | "sources" | "repo")
            && path.file_name().and_then(|value| value.to_str()) != Some("sources.list")
        {
            return Ok(false);
        }
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
    let downloader = if command_available("curl") {
        "curl"
    } else {
        "wget"
    };
    setup_commands_with_downloader(manager, repository, downloader)
}

pub fn downloader_available() -> bool {
    command_available("curl") || command_available("wget")
}

fn setup_commands_with_downloader(
    manager: PackageManager,
    repository: &NvidiaRepository,
    downloader: &str,
) -> Vec<CommandSpec> {
    let repo_url = format!("{}cuda-{}.repo", repository.base_url, repository.distro);
    let temporary_path = temporary_download_path(match manager {
        PackageManager::AptGet => CUDA_KEYRING_PACKAGE,
        _ => "cuda.repo",
    });
    let temporary_path = temporary_path.to_string_lossy().into_owned();
    let download = |url: &str| match downloader {
        "wget" => CommandSpec::new(
            "wget",
            ["--https-only", "--output-document", &temporary_path, url],
        ),
        _ => CommandSpec::new(
            "curl",
            [
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--tlsv1.2",
                "--output",
                &temporary_path,
                url,
            ],
        ),
    };
    match manager {
        PackageManager::AptGet => vec![
            download(&format!("{}{CUDA_KEYRING_PACKAGE}", repository.base_url)),
            CommandSpec::sudo("dpkg", ["-i", &temporary_path]),
            CommandSpec::new("rm", ["-f", &temporary_path]),
        ],
        PackageManager::Dnf | PackageManager::Tdnf => vec![
            download(&repo_url),
            CommandSpec::sudo(
                "install",
                [
                    "-m",
                    "0644",
                    &temporary_path,
                    &format!("/etc/yum.repos.d/cuda-{}.repo", repository.distro),
                ],
            ),
            CommandSpec::new("rm", ["-f", &temporary_path]),
        ],
        PackageManager::Zypper => vec![
            CommandSpec::sudo(
                "zypper",
                ["--non-interactive", "addrepo", &repo_url, "cuda-nvidia"],
            ),
            CommandSpec::sudo(
                "zypper",
                ["--gpg-auto-import-keys", "refresh", "cuda-nvidia"],
            ),
        ],
    }
}

fn command_available(program: &str) -> bool {
    std::process::Command::new(program)
        .arg("--version")
        .output()
        .is_ok()
}

fn temporary_download_path(file_name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join(format!("arc-{}-{nonce}-{file_name}", std::process::id()))
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
            (Distribution::Ubuntu, "22.04", "ubuntu2204"),
            (Distribution::Ubuntu, "24.04", "ubuntu2404"),
            (Distribution::Ubuntu, "26.04", "ubuntu2604"),
            (Distribution::Debian, "12", "debian12"),
            (Distribution::Debian, "13", "debian13"),
            (Distribution::Rhel, "8.10", "rhel8"),
            (Distribution::Rhel, "9.8", "rhel9"),
            (Distribution::Rhel, "10.2", "rhel10"),
            (Distribution::RockyLinux, "10.2", "rhel10"),
            (Distribution::RockyLinux, "9.8", "rhel9"),
            (Distribution::AlmaLinux, "10.2", "rhel10"),
            (Distribution::OracleLinux, "9", "rhel9"),
            (Distribution::Fedora, "44", "fedora44"),
            (Distribution::AmazonLinux, "2023", "amzn2023"),
            (Distribution::AzureLinux, "3.0", "azl3"),
            (Distribution::OpenSuse, "15.6", "opensuse15"),
            (Distribution::OpenSuse, "16.0", "suse16"),
            (Distribution::Sles, "15.7", "sles15"),
            (Distribution::Sles, "16", "suse16"),
            (Distribution::KylinOs, "V11 2503", "kylin11"),
        ] {
            assert_eq!(
                resolve(&os(distribution, version)).unwrap().distro,
                expected
            );
        }
    }

    #[test]
    fn distinguishes_compatible_and_nvidia_validated_releases() {
        let validated = resolve(&os(Distribution::Rhel, "9.8")).unwrap();
        assert!(validated.nvidia_validated);

        let newer_minor = resolve(&os(Distribution::Rhel, "9.9")).unwrap();
        assert_eq!(newer_minor.distro, "rhel9");
        assert!(!newer_minor.nvidia_validated);
    }

    #[test]
    fn rejects_unpublished_release_instead_of_substituting() {
        assert!(resolve(&os(Distribution::Ubuntu, "25.10")).is_err());
        assert!(resolve(&os(Distribution::Rhel, "7.9")).is_err());
        assert!(resolve(&os(Distribution::Rhel, "11.0")).is_err());
        assert!(resolve(&os(Distribution::AmazonLinux, "2")).is_err());
        assert!(resolve(&os(Distribution::AzureLinux, "4.0")).is_err());
        assert!(resolve(&os(Distribution::Fedora, "45")).is_err());
        assert!(resolve(&os(Distribution::OpenSuse, "15.5")).is_err());
        assert!(resolve(&os(Distribution::OpenSuse, "15.7")).is_err());
        assert!(resolve(&os(Distribution::Sles, "15.5")).is_err());
    }

    #[test]
    fn rejects_unsupported_architecture_distribution_combinations() {
        let mut debian_arm = os(Distribution::Debian, "13");
        debian_arm.architecture = "aarch64".into();
        assert!(resolve(&debian_arm).is_err());

        for distribution in [Distribution::AlmaLinux, Distribution::RockyLinux] {
            let mut derivative_arm = os(distribution, "9.8");
            derivative_arm.architecture = "aarch64".into();
            assert!(resolve(&derivative_arm).is_err());
        }

        let mut rhel_arm = os(Distribution::Rhel, "9.8");
        rhel_arm.architecture = "aarch64".into();
        assert_eq!(resolve(&rhel_arm).unwrap().distro, "rhel9");

        let mut ubuntu_arm = os(Distribution::Ubuntu, "24.04");
        ubuntu_arm.architecture = "aarch64".into();
        assert_eq!(
            resolve(&ubuntu_arm).unwrap().base_url,
            "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/"
        );
    }

    #[test]
    fn covers_every_documented_distribution_architecture_family() {
        let supported = [
            (Distribution::Ubuntu, "24.04", "x86_64"),
            (Distribution::Ubuntu, "24.04", "aarch64"),
            (Distribution::Debian, "13", "x86_64"),
            (Distribution::Rhel, "9.8", "x86_64"),
            (Distribution::Rhel, "9.8", "aarch64"),
            (Distribution::AlmaLinux, "9.8", "x86_64"),
            (Distribution::RockyLinux, "9.8", "x86_64"),
            (Distribution::OracleLinux, "9", "x86_64"),
            (Distribution::Fedora, "44", "x86_64"),
            (Distribution::AmazonLinux, "2023", "x86_64"),
            (Distribution::AmazonLinux, "2023", "aarch64"),
            (Distribution::AzureLinux, "3.0", "x86_64"),
            (Distribution::AzureLinux, "3.0", "aarch64"),
            (Distribution::OpenSuse, "15.6", "x86_64"),
            (Distribution::Sles, "15.7", "x86_64"),
            (Distribution::Sles, "15.7", "aarch64"),
            (Distribution::KylinOs, "V11 2503", "x86_64"),
            (Distribution::KylinOs, "V11 2503", "aarch64"),
        ];
        for (distribution, version, architecture) in supported {
            let mut value = os(distribution, version);
            value.architecture = architecture.into();
            assert!(
                resolve(&value).is_ok(),
                "{} {architecture}",
                value.display_name()
            );
        }
    }

    #[test]
    fn generates_repository_setup_for_each_manager_family() {
        let repository = resolve(&os(Distribution::Ubuntu, "24.04")).unwrap();
        assert_eq!(
            setup_commands_with_downloader(PackageManager::AptGet, &repository, "curl").len(),
            3
        );
        assert!(
            setup_commands_with_downloader(PackageManager::Dnf, &repository, "wget")[0]
                .display()
                .contains("wget")
        );
        assert!(
            setup_commands_with_downloader(PackageManager::Dnf, &repository, "curl")[1]
                .display()
                .contains("/etc/yum.repos.d/")
        );
        assert!(
            setup_commands_with_downloader(PackageManager::Zypper, &repository, "curl")[1]
                .display()
                .contains("--gpg-auto-import-keys")
        );
    }

    #[test]
    fn readme_support_table_is_generated_from_resolver_metadata() {
        let readme = include_str!("../../../README.md");
        let table = readme_support_table();
        assert!(
            readme.contains(&table),
            "README support table must match centralized repository metadata:\n{table}"
        );
    }
}
