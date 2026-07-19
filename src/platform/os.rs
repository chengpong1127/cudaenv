use std::fs;

use anyhow::{Result, bail};
use os_info::Type;

use crate::model::system::{Distribution, OsInfo};

pub fn detect() -> Result<OsInfo> {
    let info = os_info::get();
    let os_release_id = if matches!(info.os_type(), Type::Linux | Type::Unknown) {
        os_release_id()
    } else {
        None
    };
    let distribution = map_distribution(info.os_type(), os_release_id.as_deref())?;
    let architecture = info
        .architecture()
        .map(str::to_owned)
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned());

    Ok(OsInfo {
        distribution,
        name: distribution_name(distribution).to_owned(),
        version_id: info.version().to_string(),
        architecture,
        is_wsl: detect_wsl(),
    })
}

fn map_distribution(os_type: Type, release_id: Option<&str>) -> Result<Distribution> {
    let distribution = match os_type {
        Type::Ubuntu => Distribution::Ubuntu,
        Type::Debian => Distribution::Debian,
        Type::RedHatEnterprise | Type::Redhat => Distribution::Rhel,
        Type::AlmaLinux => Distribution::AlmaLinux,
        Type::RockyLinux => Distribution::RockyLinux,
        Type::OracleLinux => Distribution::OracleLinux,
        Type::Fedora => Distribution::Fedora,
        Type::Amazon => Distribution::AmazonLinux,
        Type::Mariner => Distribution::AzureLinux,
        Type::openSUSE => Distribution::OpenSuse,
        Type::SUSE => Distribution::Sles,
        Type::Linux | Type::Unknown if matches!(release_id, Some("kylin" | "kylinos")) => {
            Distribution::KylinOs
        }
        Type::Linux | Type::Unknown if matches!(release_id, Some("azurelinux" | "azl")) => {
            Distribution::AzureLinux
        }
        _ => bail!("GPU package installation is not supported on {os_type}"),
    };
    Ok(distribution)
}

fn distribution_name(distribution: Distribution) -> &'static str {
    match distribution {
        Distribution::Ubuntu => "Ubuntu",
        Distribution::Debian => "Debian",
        Distribution::Rhel => "Red Hat Enterprise Linux",
        Distribution::AlmaLinux => "AlmaLinux",
        Distribution::RockyLinux => "Rocky Linux",
        Distribution::OracleLinux => "Oracle Linux",
        Distribution::Fedora => "Fedora",
        Distribution::AmazonLinux => "Amazon Linux",
        Distribution::AzureLinux => "Azure Linux",
        Distribution::OpenSuse => "openSUSE",
        Distribution::Sles => "SUSE Linux Enterprise Server",
        Distribution::KylinOs => "KylinOS",
    }
}

fn os_release_id() -> Option<String> {
    let contents = fs::read_to_string("/etc/os-release").ok()?;
    contents.lines().find_map(|line| {
        line.strip_prefix("ID=")
            .map(|value| value.trim_matches(['\'', '"']).to_ascii_lowercase())
    })
}

fn detect_wsl() -> bool {
    std::env::var_os("WSL_INTEROP").is_some()
        || std::env::var_os("WSL_DISTRO_NAME").is_some()
        || ["/proc/sys/kernel/osrelease", "/proc/version"]
            .iter()
            .filter_map(|path| fs::read_to_string(path).ok())
            .any(|value| value.to_ascii_lowercase().contains("microsoft"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::system::PackageManager;

    #[test]
    fn maps_os_info_results() {
        let cases = [
            (Type::Ubuntu, None, Distribution::Ubuntu),
            (Type::Debian, None, Distribution::Debian),
            (Type::RedHatEnterprise, None, Distribution::Rhel),
            (Type::AlmaLinux, None, Distribution::AlmaLinux),
            (Type::RockyLinux, None, Distribution::RockyLinux),
            (Type::OracleLinux, None, Distribution::OracleLinux),
            (Type::Fedora, None, Distribution::Fedora),
            (Type::Amazon, None, Distribution::AmazonLinux),
            (Type::Mariner, None, Distribution::AzureLinux),
            (Type::Linux, Some("azurelinux"), Distribution::AzureLinux),
            (Type::openSUSE, None, Distribution::OpenSuse),
            (Type::SUSE, None, Distribution::Sles),
            (Type::Linux, Some("kylin"), Distribution::KylinOs),
        ];
        for (kind, id, expected) in cases {
            assert_eq!(map_distribution(kind, id).unwrap(), expected);
        }
    }

    #[test]
    fn selects_package_manager_family() {
        for (distribution, expected) in [
            (Distribution::Ubuntu, PackageManager::AptGet),
            (Distribution::KylinOs, PackageManager::AptGet),
            (Distribution::OracleLinux, PackageManager::Dnf),
            (Distribution::AzureLinux, PackageManager::Tdnf),
            (Distribution::Sles, PackageManager::Zypper),
        ] {
            assert_eq!(sample(distribution, false).package_manager(), expected);
        }
    }

    #[test]
    fn rejects_wsl_with_vendor_neutral_host_explanation() {
        let error = sample(Distribution::Ubuntu, true)
            .ensure_driver_installable("NVIDIA")
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("Windows host"));
        assert!(message.contains("NVIDIA"));
    }

    fn sample(distribution: Distribution, is_wsl: bool) -> OsInfo {
        OsInfo {
            distribution,
            name: "test".into(),
            version_id: "1".into(),
            architecture: "x86_64".into(),
            is_wsl,
        }
    }
}
