use clap::{Args, Parser, Subcommand, ValueEnum};

/// A GPU environment manager for Linux.
#[derive(Debug, Parser)]
#[command(name = "arc", version, about)]
pub struct Cli {
    /// Stream command output directly instead of using compact progress output.
    #[arg(long, short = 'v', global = true)]
    pub verbose: bool,
    /// Show exact commands in operation plans without streaming runtime logs.
    #[arg(long, global = true)]
    pub show_commands: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Install for model training or CUDA development.
    Install(InstallArgs),
    /// Display the current GPU environment.
    Status(StatusArgs),
    /// Upgrade installed NVIDIA components to their latest compatible versions.
    Upgrade(UpgradeArgs),
    /// Diagnose common GPU driver problems.
    Doctor(DoctorArgs),
    /// Plan and remove CUDA Toolkit and NVIDIA driver packages on Ubuntu.
    Uninstall(UninstallArgs),
}

#[derive(Args, Debug)]
pub struct StatusArgs {}

#[derive(Args, Debug)]
pub struct UpgradeArgs {
    /// Upgrade the installed NVIDIA driver.
    #[arg(long)]
    pub driver: bool,
    /// Upgrade installed CUDA Toolkits.
    #[arg(long)]
    pub toolkit: bool,
    /// Print the plan without changing the system.
    #[arg(long)]
    pub dry_run: bool,
    /// Do not ask for final confirmation.
    #[arg(long, short = 'y')]
    pub yes: bool,
}

#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Expected workload; Toolkit absence is normal for model training.
    #[arg(long, value_enum, default_value_t = UsageProfile::ModelTraining)]
    pub profile: UsageProfile,
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
            Self::ModelTraining => "Model training     PyTorch, TensorFlow, or JAX",
            Self::CudaDevelopment => "CUDA development   Native CUDA apps and custom kernels",
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn status_accepts_short_and_long_verbose_flags() {
        for flag in ["-v", "--verbose"] {
            let cli = Cli::try_parse_from(["arc", "status", flag]).unwrap();
            assert!(cli.verbose);
            assert!(matches!(cli.command, Command::Status(StatusArgs {})));
        }
    }

    #[test]
    fn verbose_is_global_for_mutating_commands() {
        assert!(
            Cli::try_parse_from(["arc", "-v", "install", "--dry-run"])
                .unwrap()
                .verbose
        );
        assert!(
            Cli::try_parse_from(["arc", "uninstall", "-v", "-y"])
                .unwrap()
                .verbose
        );
    }

    #[test]
    fn show_commands_is_global_and_does_not_enable_verbose_output() {
        let cli = Cli::try_parse_from(["arc", "install", "--dry-run", "--show-commands"]).unwrap();
        assert!(cli.show_commands);
        assert!(!cli.verbose);
    }
}
