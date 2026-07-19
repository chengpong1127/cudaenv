pub mod cli;
pub mod commands;
pub mod model;
pub mod platform;
pub mod providers;
pub mod ui;

use anyhow::Result;
use cli::{Cli, Command};

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Install(args) => commands::install::run(args),
        Command::Status => commands::status::run(),
        Command::Doctor => commands::doctor::run(),
        Command::Uninstall(args) => commands::uninstall::run(args),
    }
}
