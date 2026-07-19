use anyhow::Result;

use crate::{
    model::{
        device::GpuVendor,
        environment::{Diagnostics, ProviderStatus, ToolkitStatus},
    },
    providers::AcceleratorProvider,
};

pub mod diagnostics;
pub mod driver;
pub mod gpu;
pub mod install;
pub mod repository;
pub mod toolkit;
pub mod uninstall;

#[derive(Clone, Copy, Debug, Default)]
pub struct NvidiaProvider;

impl AcceleratorProvider for NvidiaProvider {
    fn vendor(&self) -> GpuVendor {
        GpuVendor::Nvidia
    }

    fn inspect(&self) -> Result<ProviderStatus> {
        let devices = gpu::detect()?;
        let driver_version = driver::detect_version()?;
        let toolkits = toolkit::detect_version()?
            .map(|version| ToolkitStatus {
                name: "CUDA Toolkit".to_owned(),
                version,
            })
            .into_iter()
            .collect();

        Ok(ProviderStatus {
            vendor: self.vendor(),
            devices: devices.into_iter().map(Into::into).collect(),
            driver_version,
            toolkits,
        })
    }

    fn diagnose(&self) -> Result<Diagnostics> {
        diagnostics::detect()
    }
}
