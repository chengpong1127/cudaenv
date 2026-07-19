use clap::{Args, Parser, Subcommand, ValueEnum};

/// An NVIDIA GPU environment manager for Linux.
#[derive(Debug, Parser)]
#[command(name = "cudaenv", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Install for model training or CUDA development.
    Install(InstallArgs),
    /// Display the current GPU environment.
    Status,
    /// Diagnose common NVIDIA driver problems.
    Doctor,
    /// Plan and remove CUDA Toolkit and NVIDIA driver packages on Ubuntu.
    Uninstall(UninstallArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum UsageProfile {
    /// Install only the NVIDIA driver.
    ModelTraining,
    /// Install the NVIDIA driver and CUDA Toolkit.
    CudaDevelopment,
}

impl UsageProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::ModelTraining => "Model training (driver only)",
            Self::CudaDevelopment => "CUDA development (driver + toolkit)",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum DriverMode {
    #[default]
    Auto,
    Open,
    Proprietary,
}

#[derive(Args, Debug)]
pub struct InstallArgs {
    /// Installation profile; prompted when omitted.
    #[arg(long, value_enum)]
    pub profile: Option<UsageProfile>,
    /// CUDA Toolkit version used by the CUDA development profile.
    #[arg(long, default_value = "13.3")]
    pub toolkit_version: String,
    /// Kernel module flavor to install.
    #[arg(long, value_enum, default_value_t)]
    pub driver: DriverMode,
    /// Print the plan without changing the system.
    #[arg(long)]
    pub dry_run: bool,
    /// Do not ask for final confirmation.
    #[arg(long, short = 'y')]
    pub yes: bool,
}

#[derive(Args, Debug)]
pub struct UninstallArgs {
    /// Do not ask for final confirmation.
    #[arg(long, short = 'y')]
    pub yes: bool,
}
