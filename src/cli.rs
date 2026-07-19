use clap::{Args, Parser, Subcommand, ValueEnum};

/// A GPU environment manager for Linux.
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
    /// Diagnose common GPU driver problems.
    Doctor,
    /// Plan and remove CUDA Toolkit and NVIDIA driver packages on Ubuntu.
    Uninstall(UninstallArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum UsageProfile {
    /// Set up model training with frameworks such as PyTorch, TensorFlow, or JAX.
    ModelTraining,
    /// Set up native CUDA development.
    CudaDevelopment,
}

impl UsageProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::ModelTraining => "Train machine learning models with PyTorch, TensorFlow, or JAX",
            Self::CudaDevelopment => {
                "Develop CUDA applications or custom GPU kernels (only if you write CUDA code directly)"
            }
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
    /// Install and pin a CUDA Toolkit version, for example 13.1.
    #[arg(long)]
    pub toolkit: Option<String>,
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
