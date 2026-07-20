use arc::{
    EXECUTION_FAILURE_EXIT_CODE, ExitStatus,
    cli::{Cli, Command},
};
use clap::Parser;

#[test]
fn cli_parses_each_user_facing_command_without_hardware_access() {
    for name in ["install", "status", "upgrade", "doctor", "uninstall"] {
        let cli = Cli::try_parse_from(["arc", name]).unwrap();
        assert_eq!(
            match cli.command {
                Command::Install(_) => "install",
                Command::Status(_) => "status",
                Command::Upgrade(_) => "upgrade",
                Command::Doctor(_) => "doctor",
                Command::Uninstall(_) => "uninstall",
            },
            name
        );
    }
}

#[test]
fn global_output_flags_parse_before_commands() {
    let cli = Cli::try_parse_from(["arc", "--verbose", "--show-commands", "install"]).unwrap();

    assert!(cli.verbose);
    assert!(cli.show_commands);
    assert!(matches!(cli.command, Command::Install(_)));
}

#[test]
fn exit_codes_distinguish_success_domain_outcomes_and_execution_failure() {
    assert_eq!(ExitStatus::Success.code(), 0);
    assert_eq!(ExitStatus::DiagnosticErrors.code(), 1);
    assert_eq!(ExitStatus::UpgradeUnavailable.code(), 1);
    assert_eq!(EXECUTION_FAILURE_EXIT_CODE, 2);
}
