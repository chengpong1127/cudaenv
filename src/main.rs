use anyhow::Result;
use clap::Parser;
use cudaenv::cli::Cli;

fn main() -> Result<()> {
    cudaenv::run(Cli::parse())
}
