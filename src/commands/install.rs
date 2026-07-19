use anyhow::{Result, bail};

use crate::{
    cli::{InstallArgs, UsageProfile},
    system::{
        command::CommandSpec,
        driver::{self, DriverFlavor},
        environment, gpu, os, repository, toolkit,
    },
    ui::{output, prompt},
};

pub fn run(args: InstallArgs) -> Result<()> {
    let os = os::detect()?;
    os.ensure_driver_installable()?;
    let gpus = gpu::detect()?;
    if gpus.is_empty() {
        bail!("No NVIDIA GPU was detected. Check that the GPU is visible and try again.");
    }

    let profile = resolve_profile(args.profile, args.toolkit.as_deref())?;
    let flavor = driver::select(args.driver, &gpus);
    if os.distribution == os::Distribution::AzureLinux && flavor == DriverFlavor::Proprietary {
        bail!("Azure Linux supports only NVIDIA open kernel modules; use --driver open.");
    }

    let repository = repository::resolve(&os)?;
    let repository_configured = toolkit::repository_is_configured(&os, &repository)?;
    let toolkit_package = match profile {
        UsageProfile::ModelTraining => None,
        UsageProfile::CudaDevelopment => Some(toolkit::package(args.toolkit.as_deref())?),
    };
    let current = environment::detect()?;
    let install_driver = current.driver_version.is_none();
    let install_toolkit = toolkit_package
        .as_deref()
        .is_some_and(|package| toolkit_install_needed(package, current.toolkit_version.as_deref()));
    let repository_setup_commands = if (install_driver || install_toolkit) && !repository_configured
    {
        toolkit::repository_setup_commands(os.package_manager(), &repository)
    } else {
        Vec::new()
    };
    let commands = operation_commands(
        os.package_manager(),
        flavor,
        toolkit_package.as_deref(),
        install_driver,
        install_toolkit,
    );
    output::installation_plan(&output::InstallationPlan {
        os: &os,
        gpus: &gpus,
        profile: profile.label(),
        driver_package: flavor.package(),
        repository: &repository,
        repository_configured,
        toolkit_package: toolkit_package.as_deref(),
        current: &current,
        install_driver,
        install_toolkit,
        repository_setup_commands: &repository_setup_commands,
        commands: &commands,
    });

    if args.dry_run {
        println!("\nDry run complete. No changes were made.");
        return Ok(());
    }
    if !install_driver && !install_toolkit {
        println!("\nRequested components are already installed. No changes were made.");
        return Ok(());
    }
    if !args.yes && !prompt::confirm_install()? {
        println!("\nInstallation cancelled. No changes were made.");
        return Ok(());
    }
    toolkit::setup_repository(&os, &repository)?;
    toolkit::refresh_metadata(os.package_manager())?;

    // Check before installing anything so an unavailable toolkit cannot leave a partial plan.
    if install_toolkit && let Some(package) = toolkit_package.as_deref() {
        toolkit::ensure_package_available(os.package_manager(), package)?;
    }
    if install_driver {
        install_command(os.package_manager(), flavor).execute()?;
    }
    if install_toolkit && let Some(package) = toolkit_package.as_deref() {
        toolkit::install_package(os.package_manager(), package)?;
        toolkit::verify_installation()?;
    }

    println!("\nInstallation completed.");
    if install_driver {
        println!("Reboot to load the NVIDIA driver.");
    }
    Ok(())
}

fn toolkit_install_needed(package: &str, current_version: Option<&str>) -> bool {
    let Some(current_version) = current_version else {
        return true;
    };
    toolkit::versioned_package(current_version)
        .map(|current_package| current_package != package)
        .unwrap_or(true)
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

fn install_command(manager: os::PackageManager, flavor: DriverFlavor) -> CommandSpec {
    let package = flavor.package();
    match manager {
        os::PackageManager::AptGet => CommandSpec::sudo("apt-get", ["install", "-y", package]),
        os::PackageManager::Dnf => CommandSpec::sudo("dnf", ["install", "-y", package]),
        os::PackageManager::Tdnf => CommandSpec::sudo("tdnf", ["install", "-y", package]),
        os::PackageManager::Zypper => {
            CommandSpec::sudo("zypper", ["--non-interactive", "install", package])
        }
    }
}

fn operation_commands(
    manager: os::PackageManager,
    flavor: DriverFlavor,
    toolkit_package: Option<&str>,
    install_driver: bool,
    install_toolkit: bool,
) -> Vec<CommandSpec> {
    if !install_driver && !install_toolkit {
        return Vec::new();
    }
    let mut commands = vec![toolkit::metadata_refresh_command(manager)];
    if install_toolkit && let Some(package) = toolkit_package {
        commands.push(toolkit::package_availability_command(manager, package));
    }
    if install_driver {
        commands.push(install_command(manager, flavor));
    }
    if install_toolkit && let Some(package) = toolkit_package {
        commands.push(toolkit::package_install_command(manager, package));
        commands.push(toolkit::verification_command());
    }
    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_install_commands_for_each_manager() {
        assert_eq!(
            install_command(os::PackageManager::AptGet, DriverFlavor::Open).display(),
            "sudo apt-get install -y nvidia-open"
        );
        assert_eq!(
            install_command(os::PackageManager::Dnf, DriverFlavor::Proprietary).display(),
            "sudo dnf install -y cuda-drivers"
        );
        assert_eq!(
            install_command(os::PackageManager::Tdnf, DriverFlavor::Open).display(),
            "sudo tdnf install -y nvidia-open"
        );
        assert_eq!(
            install_command(os::PackageManager::Zypper, DriverFlavor::Proprietary).display(),
            "sudo zypper --non-interactive install cuda-drivers"
        );
    }

    #[test]
    fn model_training_plan_has_driver_only() {
        let plan = operation_commands(
            os::PackageManager::AptGet,
            DriverFlavor::Open,
            None,
            true,
            false,
        )
        .iter()
        .map(CommandSpec::display)
        .collect::<Vec<_>>()
        .join("\n");
        assert!(plan.contains("nvidia-open"));
        assert!(!plan.contains("cuda-toolkit"));
        assert!(!plan.contains("nvcc"));
    }

    #[test]
    fn cuda_development_plan_has_driver_and_toolkit() {
        let plan = operation_commands(
            os::PackageManager::Dnf,
            DriverFlavor::Open,
            Some("cuda-toolkit-13-3"),
            true,
            true,
        )
        .iter()
        .map(CommandSpec::display)
        .collect::<Vec<_>>()
        .join("\n");
        assert!(plan.contains("nvidia-open"));
        assert!(plan.contains("cuda-toolkit-13-3"));
        assert!(plan.contains("nvcc --version"));
    }

    #[test]
    fn toolkit_option_selects_cuda_development() {
        assert_eq!(
            resolve_profile(None, Some("13.1")).unwrap(),
            UsageProfile::CudaDevelopment
        );
        assert!(resolve_profile(Some(UsageProfile::ModelTraining), Some("13.1")).is_err());
    }

    #[test]
    fn installed_driver_is_omitted_from_toolkit_plan() {
        let plan = operation_commands(
            os::PackageManager::AptGet,
            DriverFlavor::Open,
            Some("cuda-toolkit-13-1"),
            false,
            true,
        )
        .iter()
        .map(CommandSpec::display)
        .collect::<Vec<_>>()
        .join("\n");
        assert!(!plan.contains("nvidia-open"));
        assert!(plan.contains("cuda-toolkit-13-1"));
    }

    #[test]
    fn matching_pinned_toolkit_is_already_installed() {
        assert!(!toolkit_install_needed("cuda-toolkit-13-1", Some("13.1")));
        assert!(toolkit_install_needed("cuda-toolkit-13-2", Some("13.1")));
        assert!(toolkit_install_needed("cuda-toolkit", Some("13.1")));
    }
}
