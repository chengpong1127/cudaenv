#![allow(dead_code)]

use std::{cell::RefCell, collections::VecDeque, path::PathBuf};

use anyhow::{Context, Result};
use arc::{
    model::{
        command::CommandSpec,
        environment::{
            DriverFlavorState, DriverInstallation, DriverPackageScope, DriverRuntimeState,
            ProviderStatus, ToolkitSource, ToolkitStatus, UnmanagedDriverEvidence,
        },
        operation::{NextStep, OperationPlan},
        profile::InstallProfile,
        system::{Distribution, OsInfo},
    },
    platform::command::{CommandInvocation, CommandResult, CommandRunner, OutputMode},
    providers::nvidia::{
        driver::DriverPreference,
        gpu::{Generation, NvidiaGpu},
        install::{InstallContext, InstallOptions, InstallSystem},
        repository,
        state::DriverEvidence,
    },
};

pub struct TestOs;

impl TestOs {
    pub fn ubuntu(version: &str) -> OsInfo {
        OsInfo {
            distribution: Distribution::Ubuntu,
            name: "Ubuntu".into(),
            version_id: version.into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        }
    }
}

pub struct DriverEvidenceBuilder {
    evidence: DriverEvidence,
}

impl Default for DriverEvidenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DriverEvidenceBuilder {
    pub fn new() -> Self {
        Self {
            evidence: DriverEvidence {
                installed_packages: vec![],
                driver_version_detected: false,
                module_loaded: false,
                module_metadata_available: false,
                runfile_uninstaller: false,
                installer_log: false,
            },
        }
    }

    pub fn packages(mut self, packages: &[&str]) -> Self {
        self.evidence.installed_packages = strings(packages);
        self
    }

    pub fn runtime_version(mut self) -> Self {
        self.evidence.driver_version_detected = true;
        self
    }

    pub fn loaded_module(mut self) -> Self {
        self.evidence.module_loaded = true;
        self
    }

    pub fn module_metadata(mut self) -> Self {
        self.evidence.module_metadata_available = true;
        self
    }

    pub fn runfile_uninstaller(mut self) -> Self {
        self.evidence.runfile_uninstaller = true;
        self
    }

    pub fn installer_log(mut self) -> Self {
        self.evidence.installer_log = true;
        self
    }

    pub fn build(self) -> DriverEvidence {
        self.evidence
    }
}

pub struct TestGpu;

impl TestGpu {
    pub fn modern() -> NvidiaGpu {
        Self::with_generation(Generation::TuringOrNewer)
    }

    pub fn legacy() -> NvidiaGpu {
        Self::with_generation(Generation::MaxwellPascalVolta)
    }

    fn with_generation(generation: Generation) -> NvidiaGpu {
        NvidiaGpu {
            name: "Test NVIDIA GPU".into(),
            pci_device_id: None,
            generation,
        }
    }
}

#[derive(Clone)]
pub struct ProviderStatusBuilder {
    status: ProviderStatus,
}

impl ProviderStatusBuilder {
    pub fn new() -> Self {
        Self {
            status: ProviderStatus {
                vendor: arc::model::device::GpuVendor::Nvidia,
                devices: vec![],
                driver: DriverInstallation::Missing,
                driver_version: None,
                driver_runtime_operational: false,
                driver_runtime_state: DriverRuntimeState::Failed,
                dkms_status: None,
                driver_module: None,
                kernel_version: Some("6.8.0-generic".into()),
                secure_boot_enabled: Some(false),
                toolkits: vec![],
                active_toolkit: None,
            },
        }
    }

    pub fn missing_driver(mut self) -> Self {
        self.status.driver = DriverInstallation::Missing;
        self.status.driver_version = None;
        self
    }

    pub fn managed_open_driver(self) -> Self {
        self.managed_driver(DriverFlavorState::Open, None)
    }

    pub fn managed_proprietary_driver(self) -> Self {
        self.managed_driver(DriverFlavorState::Proprietary, None)
    }

    pub fn managed_driver(mut self, flavor: DriverFlavorState, branch: Option<u32>) -> Self {
        self.status.driver = DriverInstallation::Managed {
            flavor,
            scope: DriverPackageScope::Full,
            branch,
            packages: vec![],
        };
        self.status.driver_runtime_operational = true;
        self.status.driver_runtime_state = DriverRuntimeState::Operational;
        if self.status.driver_version.is_none() {
            self.status.driver_version = Some("610.43.02".into());
        }
        self
    }

    pub fn broken_managed_open(mut self, packages: &[&str]) -> Self {
        self.status.driver = DriverInstallation::BrokenManaged {
            flavor: DriverFlavorState::Open,
            packages: strings(packages),
        };
        self.status.driver_version = None;
        self
    }

    pub fn mixed_driver(mut self) -> Self {
        self.status.driver = DriverInstallation::Managed {
            flavor: DriverFlavorState::Mixed,
            scope: DriverPackageScope::Full,
            branch: None,
            packages: vec!["nvidia-open".into(), "cuda-drivers".into()],
        };
        self
    }

    pub fn unmanaged_driver(mut self, working: bool) -> Self {
        self.status.driver = DriverInstallation::Unmanaged {
            working,
            evidence: vec![UnmanagedDriverEvidence::RunfileUninstaller],
        };
        self.status.driver_version = working.then(|| "610.43.02".into());
        self
    }

    pub fn driver_version(mut self, version: &str) -> Self {
        self.status.driver_version = Some(version.into());
        self
    }

    pub fn waiting_for_reboot(mut self) -> Self {
        self.status.driver_version = None;
        self.status.driver_runtime_operational = false;
        self.status.driver_runtime_state = DriverRuntimeState::RebootLikelyRequired;
        self
    }

    pub fn toolkit(mut self, version: &str) -> Self {
        self.status.toolkits = vec![ToolkitStatus {
            name: "System-managed CUDA Toolkit".into(),
            version: Some(version.into()),
            executable_path: Some(format!("/usr/local/cuda-{version}/bin/nvcc")),
            source: ToolkitSource::SystemPackageManager,
            packages: vec![format!("cuda-toolkit-{}", version.replace('.', "-"))],
            manageable: true,
        }];
        self
    }

    pub fn active_unmanaged_toolkit(mut self, version: &str) -> Self {
        self.status.active_toolkit = Some(ToolkitStatus {
            name: "Active nvcc".into(),
            version: Some(version.into()),
            executable_path: Some("/opt/conda/envs/cuda/bin/nvcc".into()),
            source: ToolkitSource::ActivePath,
            packages: vec![],
            manageable: false,
        });
        self
    }

    pub fn build(self) -> ProviderStatus {
        self.status
    }
}

impl Default for ProviderStatusBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct InstallOptionsBuilder {
    options: InstallOptions,
}

impl InstallOptionsBuilder {
    pub fn model_training() -> Self {
        Self {
            options: InstallOptions {
                profile: InstallProfile::ModelTraining,
                toolkit_version: None,
                driver: DriverPreference::Auto,
            },
        }
    }

    pub fn cuda(version: &str) -> Self {
        Self {
            options: InstallOptions {
                profile: InstallProfile::CudaDevelopment,
                toolkit_version: Some(version.into()),
                driver: DriverPreference::Auto,
            },
        }
    }

    pub fn driver(mut self, preference: DriverPreference) -> Self {
        self.options.driver = preference;
        self
    }

    pub fn build(self) -> InstallOptions {
        self.options
    }
}

pub struct InstallContextBuilder {
    context: InstallContext,
}

pub struct FakeSystem {
    pub kernel: String,
    pub gpus: Vec<NvidiaGpu>,
    pub installed_packages: Vec<String>,
    pub status: ProviderStatus,
    pub repository_configured: bool,
    pub repository_downloader_available: bool,
    pub kernel_headers_available: bool,
}

impl FakeSystem {
    pub fn modern_ubuntu(status: ProviderStatus) -> Self {
        Self {
            kernel: "6.8.0-generic".into(),
            gpus: vec![TestGpu::modern()],
            installed_packages: vec![],
            status,
            repository_configured: true,
            repository_downloader_available: true,
            kernel_headers_available: true,
        }
    }
}

impl InstallSystem for FakeSystem {
    fn kernel_release(&self) -> Result<String> {
        Ok(self.kernel.clone())
    }

    fn gpus(&self) -> Result<Vec<NvidiaGpu>> {
        Ok(self.gpus.clone())
    }

    fn installed_packages(&self, _os: &OsInfo) -> Result<Vec<String>> {
        Ok(self.installed_packages.clone())
    }

    fn provider_status(
        &self,
        _gpus: Vec<NvidiaGpu>,
        _packages: Vec<String>,
    ) -> Result<ProviderStatus> {
        Ok(self.status.clone())
    }

    fn repository_configured(
        &self,
        _os: &OsInfo,
        _repository: &repository::NvidiaRepository,
    ) -> Result<bool> {
        Ok(self.repository_configured)
    }

    fn repository_downloader_available(&self) -> bool {
        self.repository_downloader_available
    }

    fn kernel_headers_available(&self) -> bool {
        self.kernel_headers_available
    }
}

impl InstallContextBuilder {
    pub fn ubuntu(status: ProviderStatus) -> Self {
        let os = TestOs::ubuntu("24.04");
        Self {
            context: InstallContext {
                repository: repository::resolve(&os).expect("Ubuntu test repository"),
                os,
                kernel: "6.8.0-generic".into(),
                gpus: vec![TestGpu::modern()],
                repository_configured: true,
                repository_downloader_available: true,
                installed_packages: vec![],
                status,
                kernel_headers_available: true,
            },
        }
    }

    pub fn gpus(mut self, gpus: Vec<NvidiaGpu>) -> Self {
        self.context.gpus = gpus;
        self
    }

    pub fn repository_configured(mut self, configured: bool) -> Self {
        self.context.repository_configured = configured;
        self
    }

    pub fn downloader_available(mut self, available: bool) -> Self {
        self.context.repository_downloader_available = available;
        self
    }

    pub fn installed_packages(mut self, packages: &[&str]) -> Self {
        self.context.installed_packages = strings(packages);
        self
    }

    pub fn build(self) -> InstallContext {
        self.context
    }
}

#[derive(Default)]
pub struct FakeCommandRunner {
    calls: RefCell<Vec<(CommandInvocation, OutputMode)>>,
    results: RefCell<VecDeque<Result<CommandResult>>>,
}

impl FakeCommandRunner {
    pub fn with_results(results: Vec<Result<CommandResult>>) -> Self {
        Self {
            calls: RefCell::new(vec![]),
            results: RefCell::new(results.into()),
        }
    }

    pub fn success() -> Result<CommandResult> {
        Ok(CommandResult {
            success: true,
            exit_code: Some(0),
            stdout: vec![],
            stderr: vec![],
        })
    }

    pub fn failure(code: i32, stderr: &str) -> Result<CommandResult> {
        Ok(CommandResult {
            success: false,
            exit_code: Some(code),
            stdout: vec![],
            stderr: stderr.as_bytes().to_vec(),
        })
    }

    pub fn calls(&self) -> Vec<(CommandInvocation, OutputMode)> {
        self.calls.borrow().clone()
    }
}

impl CommandRunner for FakeCommandRunner {
    fn run(&self, invocation: &CommandInvocation, mode: OutputMode) -> Result<CommandResult> {
        self.calls.borrow_mut().push((invocation.clone(), mode));
        self.results
            .borrow_mut()
            .pop_front()
            .context("fake command runner has no configured result")?
    }
}

pub fn operation_plan(commands: Vec<CommandSpec>) -> OperationPlan {
    use arc::model::operation::PlanStep;
    OperationPlan {
        title: "Test plan".into(),
        details: vec![],
        devices: vec![],
        steps: commands
            .into_iter()
            .enumerate()
            .map(|(index, command)| PlanStep::new(format!("step {}", index + 1), command))
            .collect(),
        confirmation_warning: String::new(),
        completion_message: String::new(),
        next_step: None,
    }
}

pub fn assert_plan_commands(plan: &OperationPlan, expected: &[(&str, &[&str])]) {
    let actual = plan
        .steps
        .iter()
        .map(|step| {
            (
                step.command.program.as_str(),
                step.command
                    .args
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let expected = expected
        .iter()
        .map(|(program, args)| (*program, args.to_vec()))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

pub fn assert_stage_titles(plan: &OperationPlan, expected: &[&str]) {
    let actual = plan
        .steps
        .iter()
        .enumerate()
        .filter_map(|(index, step)| {
            plan.stage_position(index)
                .1
                .then_some(step.stage.title.as_str())
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

pub fn assert_command_before(plan: &OperationPlan, first: &str, second: &str) {
    let index = |program: &str| {
        plan.steps
            .iter()
            .position(|step| step.command.args.iter().any(|arg| arg == program))
            .unwrap_or_else(|| panic!("plan does not contain argument {program:?}"))
    };
    assert!(
        index(first) < index(second),
        "{first:?} must precede {second:?}"
    );
}

pub fn assert_noop_with_next_step(plan: &OperationPlan, next_step: Option<NextStep>) {
    assert!(plan.is_noop(), "expected an empty operation plan");
    assert_eq!(plan.next_step, next_step);
}

pub fn temp_log_dir(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("arc-{test_name}-{}", std::process::id()))
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
}
