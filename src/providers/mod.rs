use anyhow::Result;

use crate::model::{
    device::GpuVendor,
    environment::{Diagnostics, ProviderStatus},
};

pub mod nvidia;

/// Shared inspection contract implemented by each GPU vendor integration.
pub trait AcceleratorProvider {
    fn vendor(&self) -> GpuVendor;
    fn inspect(&self) -> Result<ProviderStatus>;
    fn diagnose(&self) -> Result<Diagnostics>;
}

/// Providers registered here automatically participate in shared inspection commands.
pub fn registered() -> Vec<Box<dyn AcceleratorProvider>> {
    vec![Box::new(nvidia::NvidiaProvider)]
}
