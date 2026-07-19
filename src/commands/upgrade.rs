use anyhow::Result;

use crate::{
    cli::UpgradeArgs,
    platform::{command, os},
    providers::nvidia::upgrade::{self, UpgradeOptions},
    ui::{output, prompt},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpgradeOutcome {
    Success,
    Unavailable,
}

pub fn run(args: UpgradeArgs) -> Result<UpgradeOutcome> {
    let options = UpgradeOptions::from_component_flags(args.driver, args.toolkit);
    let mut plan = match upgrade::plan(&os::detect()?, &options) {
        Ok(plan) => plan,
        Err(error) if upgrade::is_actionable(&error) => {
            eprintln!(
                "Upgrade unavailable: {}",
                upgrade::actionable_message(&error)
            );
            return Ok(UpgradeOutcome::Unavailable);
        }
        Err(error) => return Err(error),
    };
    command::normalize_for_current_user(&mut plan);
    output::operation_plan(&plan);

    if args.dry_run {
        println!("\nDry run complete. No changes were made.");
        return Ok(UpgradeOutcome::Success);
    }
    if plan.is_noop() {
        println!(
            "\nNo selected installed component has a compatible upgrade. No changes were made."
        );
        return Ok(UpgradeOutcome::Success);
    }
    if !args.yes && !prompt::confirm_upgrade()? {
        println!("\nUpgrade cancelled. No changes were made.");
        return Ok(UpgradeOutcome::Success);
    }
    command::ensure_execution_privileges(&plan)?;
    command::execute_plan(&command::SystemCommandRunner, &plan)?;
    output::operation_completed(&plan);
    Ok(UpgradeOutcome::Success)
}
