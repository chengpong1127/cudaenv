use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuDevice {
    pub vendor: GpuVendor,
    pub name: String,
    pub pci_device_id: Option<u16>,
}
