use clap::Parser;
use cudaenv::cli::Cli;
use std::process::ExitCode;

fn main() -> ExitCode {
    match cudaenv::run(Cli::parse()) {
        Ok(status) => ExitCode::from(status.code()),
        Err(error) => {
            eprintln!("cudaenv could not complete: {error:#}");
            ExitCode::from(cudaenv::EXECUTION_FAILURE_EXIT_CODE)
        }
    }
}
