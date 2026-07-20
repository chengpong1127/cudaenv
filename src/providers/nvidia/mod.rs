use anyhow::Result;

use crate::{model::environment::ProviderStatus, platform::os, providers::AcceleratorProvider};

pub mod compatibility;
pub mod diagnostics;
pub mod driver;
pub mod gpu;
pub mod install;
pub mod policy;
pub mod recipe;
pub mod repository;
pub mod runtime;
pub mod state;
pub mod toolkit;
pub mod uninstall;
pub mod upgrade;

pub struct NvidiaProvider;

impl AcceleratorProvider for NvidiaProvider {
    fn inspect(&self) -> Result<ProviderStatus> {
        state::inspect(&os::detect()?)
    }
}
