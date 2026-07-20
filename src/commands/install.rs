use anyhow::{Result, bail};

use crate::{
    cli::{DriverMode, InstallArgs, UsageProfile},
    platform::{command, os},
    providers::nvidia::{
        driver::DriverPreference,
        install::{self, InstallOptions, InstallProfile},
    },
    ui::{output, prompt},
};

pub fn run(args: InstallArgs) -> Result<()> {
    let profile = resolve_profile(args.profile, args.toolkit.as_deref())?;
    let options = InstallOptions {
        profile: match profile {
            UsageProfile::ModelTraining => InstallProfile::ModelTraining,
            UsageProfile::CudaDevelopment => InstallProfile::CudaDevelopment,
        },
        toolkit_version: args.toolkit.clone(),
        driver: match args.driver {
            DriverMode::Auto => DriverPreference::Auto,
            DriverMode::Open => DriverPreference::Open,
            DriverMode::Proprietary => DriverPreference::Proprietary,
        },
    };
    let mut plan = install::plan(&os::detect()?, &options)?;
    command::normalize_for_current_user(&mut plan);
    output::operation_plan(&plan);

    if args.dry_run {
        output::notice("Dry run complete. No changes were made.");
        return Ok(());
    }
    if plan.is_noop() {
        output::notice(plan.reboot_message.as_deref().unwrap_or(
            "Requested components are already installed. No changes were made.",
        ));
        return Ok(());
    }
    if !args.yes && !prompt::confirm_install()? {
        output::cancelled("Installation");
        return Ok(());
    }
    command::ensure_execution_privileges(&plan)?;
    command::execute_plan_with_reporter(
        &command::SystemCommandRunner,
        &plan,
        output::execution_event,
    )?;
    output::operation_completed(&plan);
    Ok(())
}

fn resolve_profile(profile: Option<UsageProfile>, toolkit: Option<&str>) -> Result<UsageProfile> {
    match (profile, toolkit) {
        (Some(UsageProfile::ModelTraining), Some(_)) => {
            bail!("--toolkit cannot be used with --profile model-training; choose cuda-development")
        }
        (Some(profile), _) => Ok(profile),
        (None, Some(_)) => Ok(UsageProfile::CudaDevelopment),
        (None, None) => prompt::select_usage_profile(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolkit_option_selects_cuda_development() {
        assert_eq!(
            resolve_profile(None, Some("13.1")).unwrap(),
            UsageProfile::CudaDevelopment
        );
        assert!(resolve_profile(Some(UsageProfile::ModelTraining), Some("13.1")).is_err());
    }
}
