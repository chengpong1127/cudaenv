use std::fmt;

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distribution {
    Ubuntu,
    Debian,
    Rhel,
    AlmaLinux,
    RockyLinux,
    OracleLinux,
    Fedora,
    AmazonLinux,
    AzureLinux,
    OpenSuse,
    Sles,
    KylinOs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageManager {
    AptGet,
    Dnf,
    Tdnf,
    Zypper,
}

impl fmt::Display for PackageManager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AptGet => "apt-get",
            Self::Dnf => "dnf",
            Self::Tdnf => "tdnf",
            Self::Zypper => "zypper",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OsInfo {
    pub distribution: Distribution,
    pub name: String,
    pub version_id: String,
    pub architecture: String,
    pub is_wsl: bool,
}

impl OsInfo {
    pub fn package_manager(&self) -> PackageManager {
        match self.distribution {
            Distribution::Ubuntu | Distribution::Debian | Distribution::KylinOs => {
                PackageManager::AptGet
            }
            Distribution::Rhel
            | Distribution::AlmaLinux
            | Distribution::RockyLinux
            | Distribution::OracleLinux
            | Distribution::Fedora
            | Distribution::AmazonLinux => PackageManager::Dnf,
            Distribution::AzureLinux => PackageManager::Tdnf,
            Distribution::OpenSuse | Distribution::Sles => PackageManager::Zypper,
        }
    }

    pub fn display_name(&self) -> String {
        format!("{} {}", self.name, self.version_id)
    }

    pub fn ensure_driver_installable(&self, vendor: &str) -> Result<()> {
        if self.is_wsl {
            bail!(
                "{vendor} driver installation is not supported inside WSL. Install the GPU driver on the Windows host; WSL uses the host driver."
            );
        }
        Ok(())
    }
}
