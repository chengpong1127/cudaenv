use arc::cli::Cli;
use clap::Parser;
use console::style;
use std::process::ExitCode;

fn main() -> ExitCode {
    match arc::run(Cli::parse()) {
        Ok(status) => ExitCode::from(status.code()),
        Err(error) => {
            eprintln!(
                "\n  {}  {}\n",
                style("✗").red().bold(),
                style(format!("arc could not complete: {error:#}"))
                    .red()
                    .bold()
            );
            ExitCode::from(arc::EXECUTION_FAILURE_EXIT_CODE)
        }
    }
}
