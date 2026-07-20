use anyhow::Result;

use crate::{
    model::{environment::ProviderStatus, system::OsInfo},
    platform::{command::SystemCommandRunner, kernel, package_manager},
};

use super::InstallOptions;
use crate::providers::nvidia::{gpu, repository, state, toolkit};

pub trait InstallSystem {
    fn kernel_release(&self) -> Result<String>;
    fn gpus(&self) -> Result<Vec<gpu::NvidiaGpu>>;
    fn installed_packages(&self, os: &OsInfo) -> Result<Vec<String>>;
    fn provider_status(
        &self,
        gpus: Vec<gpu::NvidiaGpu>,
        packages: Vec<String>,
    ) -> Result<ProviderStatus>;
    fn repository_configured(
        &self,
        os: &OsInfo,
        repository: &repository::NvidiaRepository,
    ) -> Result<bool>;
    fn repository_downloader_available(&self) -> bool;
    fn kernel_headers_available(&self) -> bool;
}

pub struct RealInstallSystem;

impl InstallSystem for RealInstallSystem {
    fn kernel_release(&self) -> Result<String> {
        kernel::release_with(&SystemCommandRunner)
    }

    fn gpus(&self) -> Result<Vec<gpu::NvidiaGpu>> {
        gpu::detect()
    }

    fn installed_packages(&self, os: &OsInfo) -> Result<Vec<String>> {
        package_manager::installed_packages_with(&SystemCommandRunner, os.package_manager())
    }

    fn provider_status(
        &self,
        gpus: Vec<gpu::NvidiaGpu>,
        packages: Vec<String>,
    ) -> Result<ProviderStatus> {
        state::inspect_with(&SystemCommandRunner, gpus, packages)
    }

    fn repository_configured(
        &self,
        os: &OsInfo,
        repository: &repository::NvidiaRepository,
    ) -> Result<bool> {
        repository::is_configured(os, repository)
    }

    fn repository_downloader_available(&self) -> bool {
        repository::downloader_available()
    }

    fn kernel_headers_available(&self) -> bool {
        crate::providers::nvidia::driver::kernel_headers_available()
    }
}

#[derive(Clone, Debug)]
pub struct InstallContext {
    pub os: OsInfo,
    pub kernel: String,
    pub gpus: Vec<gpu::NvidiaGpu>,
    pub repository: repository::NvidiaRepository,
    pub repository_configured: bool,
    pub repository_downloader_available: bool,
    pub installed_packages: Vec<String>,
    pub status: ProviderStatus,
    pub kernel_headers_available: bool,
}

impl InstallContext {
    pub fn inspect(os: &OsInfo) -> Result<Self> {
        Self::inspect_with(os, &RealInstallSystem)
    }

    pub fn inspect_with(os: &OsInfo, system: &impl InstallSystem) -> Result<Self> {
        os.ensure_driver_installable("NVIDIA")?;
        let kernel = system.kernel_release()?;
        let gpus = system.gpus()?;
        let installed_packages = system.installed_packages(os)?;
        let status = system.provider_status(gpus.clone(), installed_packages.clone())?;
        let repository = repository::resolve(os)?;
        let repository_configured = system.repository_configured(os, &repository)?;
        Ok(Self {
            os: os.clone(),
            kernel,
            gpus,
            repository,
            repository_configured,
            repository_downloader_available: system.repository_downloader_available(),
            installed_packages,
            status,
            kernel_headers_available: system.kernel_headers_available(),
        })
    }
}

pub(super) fn requested_toolkit(options: &InstallOptions) -> Result<Option<String>> {
    match options.profile {
        super::InstallProfile::ModelTraining => Ok(None),
        super::InstallProfile::CudaDevelopment => {
            toolkit::package(options.toolkit_version.as_deref()).map(Some)
        }
    }
}
