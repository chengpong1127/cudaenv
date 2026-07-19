pub mod cli;
mod commands;
mod model;
mod platform;
mod providers;
mod ui;

use anyhow::Result;
use cli::{Cli, Command};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitStatus {
    Success,
    DiagnosticErrors,
    UpgradeUnavailable,
}

impl ExitStatus {
    pub fn code(self) -> u8 {
        match self {
            Self::Success => 0,
            Self::DiagnosticErrors | Self::UpgradeUnavailable => 1,
        }
    }
}

pub const EXECUTION_FAILURE_EXIT_CODE: u8 = 2;

pub fn run(cli: Cli) -> Result<ExitStatus> {
    match cli.command {
        Command::Install(args) => commands::install::run(args).map(|_| ExitStatus::Success),
        Command::Status => commands::status::run().map(|_| ExitStatus::Success),
        Command::Upgrade(args) => commands::upgrade::run(args).map(|outcome| match outcome {
            commands::upgrade::UpgradeOutcome::Success => ExitStatus::Success,
            commands::upgrade::UpgradeOutcome::Unavailable => ExitStatus::UpgradeUnavailable,
        }),
        Command::Doctor(args) => commands::doctor::run(args).map(|outcome| match outcome {
            commands::doctor::DoctorOutcome::Healthy => ExitStatus::Success,
            commands::doctor::DoctorOutcome::ErrorsFound => ExitStatus::DiagnosticErrors,
        }),
        Command::Uninstall(args) => commands::uninstall::run(args).map(|_| ExitStatus::Success),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_distinguish_health_errors_and_execution_failure() {
        assert_eq!(ExitStatus::Success.code(), 0);
        assert_eq!(ExitStatus::DiagnosticErrors.code(), 1);
        assert_eq!(ExitStatus::UpgradeUnavailable.code(), 1);
        assert_eq!(EXECUTION_FAILURE_EXIT_CODE, 2);
    }
}
