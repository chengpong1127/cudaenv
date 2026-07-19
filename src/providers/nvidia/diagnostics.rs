use anyhow::Result;

use crate::model::{
    device::GpuVendor,
    environment::{DiagnosticCheck, Diagnostics},
};

use super::{driver, gpu};

pub fn detect() -> Result<Diagnostics> {
    let gpu_detected = !gpu::detect()?.is_empty();
    let driver_installed = driver::detect_version()?.is_some();
    let nvidia_smi_available = driver::nvidia_smi_available();
    Ok(Diagnostics {
        vendor: GpuVendor::Nvidia,
        checks: vec![
            DiagnosticCheck {
                name: "NVIDIA GPU detected".into(),
                passed: gpu_detected,
                problem: (!gpu_detected)
                    .then(|| "No NVIDIA GPU was detected by lspci or sysfs.".into()),
            },
            DiagnosticCheck {
                name: "NVIDIA driver installed".into(),
                passed: driver_installed,
                problem: (!driver_installed)
                    .then(|| "The NVIDIA driver does not appear to be installed or loaded.".into()),
            },
            DiagnosticCheck {
                name: "nvidia-smi available".into(),
                passed: nvidia_smi_available,
                problem: (!nvidia_smi_available)
                    .then(|| "nvidia-smi is not available in PATH.".into()),
            },
        ],
    })
}
