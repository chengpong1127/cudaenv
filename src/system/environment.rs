use anyhow::Result;

use crate::system::{driver, toolkit};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InstallationStatus {
    pub driver_version: Option<String>,
    pub toolkit_version: Option<String>,
}

pub fn detect() -> Result<InstallationStatus> {
    Ok(InstallationStatus {
        driver_version: driver::detect_version()?,
        toolkit_version: toolkit::detect_version()?,
    })
}
