use crate::model::device::{GpuDevice, GpuVendor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolkitStatus {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderStatus {
    pub vendor: GpuVendor,
    pub devices: Vec<GpuDevice>,
    pub driver_version: Option<String>,
    pub toolkits: Vec<ToolkitStatus>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticCheck {
    pub name: String,
    pub passed: bool,
    pub problem: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostics {
    pub vendor: GpuVendor,
    pub checks: Vec<DiagnosticCheck>,
}

impl Diagnostics {
    pub fn healthy(&self) -> bool {
        self.checks.iter().all(|check| check.passed)
    }
}
