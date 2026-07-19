use crate::system::{
    command::CommandSpec, environment::InstallationStatus, gpu::Gpu, os::OsInfo,
    repository::Repository,
};

fn print_commands(commands: &[String]) {
    for command in commands {
        println!("  $ {command}");
    }
}

pub struct InstallationPlan<'a> {
    pub os: &'a OsInfo,
    pub gpus: &'a [Gpu],
    pub profile: &'a str,
    pub driver_package: &'a str,
    pub repository: &'a Repository,
    pub repository_configured: bool,
    pub toolkit_package: Option<&'a str>,
    pub current: &'a InstallationStatus,
    pub install_driver: bool,
    pub install_toolkit: bool,
    pub repository_setup_commands: &'a [CommandSpec],
    pub commands: &'a [CommandSpec],
}

pub fn installation_plan(plan: &InstallationPlan<'_>) {
    println!("Installation Plan\n");
    println!("OS: {}", plan.os.display_name());
    println!("Package manager: {:?}", plan.os.package_manager());
    println!("Repository: {}", plan.repository.base_url);
    println!("Profile: {}", plan.profile);
    println!(
        "Driver: {}",
        if plan.install_driver {
            format!("install {}", plan.driver_package)
        } else {
            format!(
                "already installed ({}) — skipped",
                plan.current.driver_version.as_deref().unwrap_or("unknown")
            )
        }
    );
    println!(
        "Current CUDA Toolkit: {}",
        plan.current
            .toolkit_version
            .as_deref()
            .unwrap_or("not installed")
    );
    println!(
        "CUDA Toolkit: {}",
        match (plan.toolkit_package, plan.install_toolkit) {
            (Some(package), true) => format!("install {package}"),
            (Some(_), false) => "requested version already installed — skipped".to_owned(),
            (None, _) => "not requested".to_owned(),
        }
    );
    println!("GPU(s):");
    for gpu in plan.gpus {
        println!("  - {} ({:?})", gpu.name, gpu.generation);
    }
    println!("\nCommands:");
    if (plan.install_driver || plan.install_toolkit) && plan.repository_configured {
        println!("  # NVIDIA CUDA repository is already configured");
    }
    for command in plan.repository_setup_commands {
        println!("  $ {}", command.display());
    }
    for command in plan.commands {
        println!("  $ {}", command.display());
    }
    println!("No system changes will be made until you confirm.");
}

pub fn system_status(os: &OsInfo, gpu: Option<&str>, driver: Option<&str>, toolkit: Option<&str>) {
    println!("GPU Environment");
    println!("\nOS:\n{}", os.display_name());
    println!("\nGPU:\n{}", gpu.unwrap_or("Not detected"));
    println!("\nDriver:\n{}", driver.unwrap_or("Not installed"));
    println!("\nCUDA Toolkit:\n{}", toolkit.unwrap_or("Not installed"));
}

pub fn diagnostics(gpu_detected: bool, driver_installed: bool, nvidia_smi: bool) {
    let healthy = gpu_detected && driver_installed && nvidia_smi;
    println!("NVIDIA Diagnostics\n");
    println!("{} NVIDIA GPU detected", mark(gpu_detected));
    println!("{} NVIDIA driver installed", mark(driver_installed));
    println!("{} nvidia-smi available", mark(nvidia_smi));

    if healthy {
        println!("\nHealthy");
        return;
    }

    println!("\nProblems found");
    if !gpu_detected {
        println!("- No NVIDIA GPU was detected by lspci or nvidia-smi.");
    }
    if !driver_installed {
        println!("- The NVIDIA driver does not appear to be installed or loaded.");
    }
    if !nvidia_smi {
        println!("- nvidia-smi is not available in PATH.");
    }
}

pub fn uninstall_plan(commands: &[String]) {
    println!("Uninstall Plan\n");
    println!("The following detected CUDA/NVIDIA packages will be removed:");
    print_commands(commands);
    println!("\nThis operation changes system packages and cannot be automatically undone.");
}

fn mark(ok: bool) -> &'static str {
    if ok { "✓" } else { "✗" }
}
