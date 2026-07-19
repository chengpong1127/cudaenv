use anyhow::{Result, bail};

use crate::{
    cli::{InstallArgs, UsageProfile},
    system::{
        command::CommandSpec,
        driver::{self, DriverFlavor},
        gpu, os, repository, toolkit,
    },
    ui::prompt,
};

pub fn run(args: InstallArgs) -> Result<()> {
    let os = os::detect()?;
    os.ensure_driver_installable()?;
    let gpus = gpu::detect()?;
    if gpus.is_empty() {
        bail!("No NVIDIA GPU was detected. Check that the GPU is visible and try again.");
    }

    let profile = match args.profile {
        Some(profile) => profile,
        None => prompt::select_usage_profile()?,
    };
    let flavor = driver::select(args.driver, &gpus);
    if os.distribution == os::Distribution::AzureLinux && flavor == DriverFlavor::Proprietary {
        bail!("Azure Linux supports only NVIDIA open kernel modules; use --driver open.");
    }

    let repository = repository::resolve(&os)?;
    let repository_configured = toolkit::repository_is_configured(&os, &repository)?;
    let toolkit_package = match profile {
        UsageProfile::ModelTraining => None,
        UsageProfile::CudaDevelopment => Some(toolkit::versioned_package(&args.toolkit_version)?),
    };
    print_plan(
        &os,
        &gpus,
        profile,
        flavor,
        &repository,
        repository_configured,
        toolkit_package.as_deref(),
    );

    if args.dry_run {
        println!("\nDry run complete. No changes were made.");
        return Ok(());
    }
    if !args.yes && !prompt::confirm_install()? {
        println!("\nInstallation cancelled. No changes were made.");
        return Ok(());
    }
    toolkit::setup_repository(&os, &repository)?;
    toolkit::refresh_metadata(os.package_manager())?;

    // Check before installing anything so an unavailable toolkit cannot leave a partial plan.
    if let Some(package) = toolkit_package.as_deref() {
        toolkit::ensure_package_available(os.package_manager(), package)?;
    }
    install_command(os.package_manager(), flavor).execute()?;
    if let Some(package) = toolkit_package.as_deref() {
        toolkit::install_package(os.package_manager(), package)?;
        toolkit::verify_installation()?;
    }

    match profile {
        UsageProfile::ModelTraining => println!("\nNVIDIA driver installation completed."),
        UsageProfile::CudaDevelopment => {
            println!("\nNVIDIA driver and CUDA Toolkit installation completed.")
        }
    }
    println!("Reboot to load the NVIDIA driver.");
    Ok(())
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

fn print_plan(
    os: &os::OsInfo,
    gpus: &[gpu::Gpu],
    profile: UsageProfile,
    flavor: DriverFlavor,
    repository: &repository::Repository,
    repository_configured: bool,
    toolkit_package: Option<&str>,
) {
    println!("Installation Plan\n");
    println!("OS: {}", os.display_name());
    println!("Package manager: {:?}", os.package_manager());
    println!("Repository: {}", repository.base_url);
    println!("Profile: {}", profile.label());
    println!("Driver: {}", flavor.package());
    println!(
        "CUDA Toolkit: {}",
        toolkit_package.unwrap_or("not installed")
    );
    println!("GPU(s):");
    for gpu in gpus {
        println!("  - {} ({:?})", gpu.name, gpu.generation);
    }
    println!("\nCommands:");
    if repository_configured {
        println!("  # NVIDIA CUDA repository is already configured");
    } else {
        for command in toolkit::repository_setup_commands(os.package_manager(), repository) {
            println!("  $ {}", command.display());
        }
    }
    let commands = operation_commands(os.package_manager(), flavor, toolkit_package);
    for command in &commands {
        println!("  $ {}", command.display());
    }
    println!("No system changes will be made until you confirm.");
}

fn operation_commands(
    manager: os::PackageManager,
    flavor: DriverFlavor,
    toolkit_package: Option<&str>,
) -> Vec<CommandSpec> {
    let mut commands = vec![toolkit::metadata_refresh_command(manager)];
    if let Some(package) = toolkit_package {
        commands.push(toolkit::package_availability_command(manager, package));
    }
    commands.push(install_command(manager, flavor));
    if let Some(package) = toolkit_package {
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
        let plan = operation_commands(os::PackageManager::AptGet, DriverFlavor::Open, None)
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
        )
        .iter()
        .map(CommandSpec::display)
        .collect::<Vec<_>>()
        .join("\n");
        assert!(plan.contains("nvidia-open"));
        assert!(plan.contains("cuda-toolkit-13-3"));
        assert!(plan.contains("nvcc --version"));
    }
}
