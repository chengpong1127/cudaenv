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

pub fn run(args: UninstallArgs) -> Result<()> {
    let system = os::detect()?;
    let status = NvidiaProvider.inspect()?;
    let plan = uninstall::plan(&system, &status)?;
    if plan.is_noop() {
        println!("No installed CUDA Toolkit or NVIDIA driver was detected.");
        return Ok(());
    }
    output::operation_plan(&plan);
    if !args.yes && !prompt::confirm_uninstall()? {
        println!("\nUninstall cancelled. No changes were made.");
        return Ok(());
    }
    command::execute_plan(&command::SystemCommandRunner, &plan)?;
    output::operation_completed(&plan);
    Ok(())
}
