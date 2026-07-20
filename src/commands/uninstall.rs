use anyhow::Result;

use crate::{
    cli::UninstallArgs,
    platform::{command, os},
    providers::{
        AcceleratorProvider,
        nvidia::{NvidiaProvider, uninstall},
    },
    ui::{output, prompt},
};

pub fn run(args: UninstallArgs, verbose: bool, show_commands: bool) -> Result<()> {
    let system = os::detect()?;
    let status = NvidiaProvider.inspect()?;
    let mut plan = uninstall::plan(&system, &status)?;
    command::normalize_for_current_user(&mut plan);
    if plan.is_noop() {
        output::notice("No installed CUDA Toolkit or NVIDIA driver was detected.");
        return Ok(());
    }
    output::operation_plan(&plan, show_commands);
    if !args.yes && !prompt::confirm_uninstall()? {
        output::cancelled("Uninstall");
        return Ok(());
    }
    let mut reporter = output::ExecutionReporter::new(&plan, verbose);
    let execution =
        command::execute_plan(&command::SystemCommandRunner, &plan, verbose, |event| {
            reporter.report(event)
        })?;
    output::operation_completed(&plan);
    output::execution_log(execution.log_path.as_deref());
    Ok(())
}
