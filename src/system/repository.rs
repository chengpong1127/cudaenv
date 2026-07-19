use anyhow::{Result, bail};

use crate::system::os::{Distribution, OsInfo};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repository {
    pub distro: String,
    pub architecture: String,
    pub base_url: String,
}

pub fn resolve(os: &OsInfo) -> Result<Repository> {
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
    Ok(Repository {
        distro,
        architecture: architecture.to_owned(),
        base_url,
    })
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

    fn os(distribution: Distribution, version: &str, architecture: &str) -> OsInfo {
        OsInfo {
            distribution,
            name: "Test".into(),
            version_id: version.into(),
            architecture: architecture.into(),
            is_wsl: false,
        }
    }

    #[test]
    fn resolves_official_repository_targets() {
        let cases = [
            (Distribution::Ubuntu, "24.04", "ubuntu2404"),
            (Distribution::Debian, "13", "debian13"),
            (Distribution::AlmaLinux, "10.1", "rhel10"),
            (Distribution::OracleLinux, "9", "rhel9"),
            (Distribution::AmazonLinux, "2023", "amzn2023"),
            (Distribution::AzureLinux, "3.0", "azl3"),
            (Distribution::OpenSuse, "15.6", "opensuse15"),
            (Distribution::Sles, "16.0", "suse16"),
            (Distribution::KylinOs, "V11", "kylin11"),
        ];
        for (distribution, version, expected) in cases {
            assert_eq!(
                resolve(&os(distribution, version, "x86_64"))
                    .unwrap()
                    .distro,
                expected
            );
        }
    }

    #[test]
    fn rejects_unpublished_release_instead_of_substituting() {
        assert!(resolve(&os(Distribution::Ubuntu, "25.10", "x86_64")).is_err());
    }
}
