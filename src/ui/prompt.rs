use anyhow::{Context, Result};
use dialoguer::{Confirm, Select, theme::ColorfulTheme};

use crate::cli::UsageProfile;

pub fn confirm_install() -> Result<bool> {
    Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Continue with this installation plan?")
        .default(false)
        .interact()
        .context("could not read installation confirmation")
}

pub fn select_usage_profile() -> Result<UsageProfile> {
    let profiles = [UsageProfile::ModelTraining, UsageProfile::CudaDevelopment];
    let options = profiles.map(UsageProfile::label);
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("What will you use this machine for?")
        .items(&options)
        .default(0)
        .interact()
        .context("could not read the usage profile")?;
    Ok(profiles[selection])
}

pub fn confirm_uninstall() -> Result<bool> {
    Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Continue with this uninstall plan?")
        .default(false)
        .interact()
        .context("could not read uninstall confirmation")
}
