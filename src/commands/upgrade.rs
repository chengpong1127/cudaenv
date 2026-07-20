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

pub fn run(args: UpgradeArgs, verbose: bool, show_commands: bool) -> Result<UpgradeOutcome> {
    let options = UpgradeOptions::from_component_flags(args.driver, args.toolkit);
    let mut plan = match upgrade::plan(&os::detect()?, &options) {
        Ok(plan) => plan,
        Err(error) if upgrade::is_actionable(&error) => {
            output::unavailable(&format!(
                "Upgrade unavailable: {}",
                upgrade::actionable_message(&error)
            ));
            return Ok(UpgradeOutcome::Unavailable);
        }
        Err(error) => return Err(error),
    };
    command::normalize_for_current_user(&mut plan);
    output::operation_plan(&plan, show_commands);

    if args.dry_run {
        output::notice("Dry run complete. No changes were made.");
        return Ok(UpgradeOutcome::Success);
    }
    if plan.is_noop() {
        output::notice(
            "No selected installed component has a compatible upgrade. No changes were made.",
        );
        return Ok(UpgradeOutcome::Success);
    }
    if !args.yes && !prompt::confirm_upgrade()? {
        output::cancelled("Upgrade");
        return Ok(UpgradeOutcome::Success);
    }
    let mut reporter = output::ExecutionReporter::new(&plan, verbose);
    let execution =
        command::execute_plan(&command::SystemCommandRunner, &plan, verbose, |event| {
            reporter.report(event)
        })?;
    output::operation_completed(&plan);
    output::execution_log(execution.log_path.as_deref());
    Ok(UpgradeOutcome::Success)
}
