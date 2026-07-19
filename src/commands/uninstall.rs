use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::{
    cli::UninstallArgs,
    system::{environment, os},
    ui::{output, prompt},
};

const CUDA_TOOLKIT_PACKAGES: &[&str] = &[
    "cuda-toolkit*",
    "cuda-compiler*",
    "cuda-command-line-tools*",
    "cuda-demo-suite*",
    "cuda-documentation*",
    "cuda-libraries*",
    "cuda-nsight*",
    "cuda-nvcc*",
    "cuda-nvdisasm*",
    "cuda-nvml-dev*",
    "cuda-nvprof*",
    "cuda-nvprune*",
    "cuda-nvrtc*",
    "cuda-nvtx*",
    "cuda-opencl*",
    "cuda-profiler-api*",
    "cuda-sanitizer*",
    "cuda-tools*",
    "*cublas*",
    "*cufft*",
    "*cufile*",
    "*curand*",
    "*cusolver*",
    "*cusparse*",
    "*gds-tools*",
    "*npp*",
    "*nvjpeg*",
    "nsight*",
    "*nvvm*",
];

// Package families from NVIDIA's Ubuntu driver removal guide.
const NVIDIA_DRIVER_PACKAGES: &[&str] = &[
    "cuda-compat*",
    "cuda-drivers*",
    "libnvidia-cfg1*",
    "libnvidia-compute*",
    "libnvidia-decode*",
    "libnvidia-encode*",
    "libnvidia-extra*",
    "libnvidia-fbc1*",
    "libnvidia-gl*",
    "libnvidia-gpucomp*",
    "libnvidia-nscq*",
    "libnvsdm*",
    "libxnvctrl*",
    "nvidia-dkms*",
    "nvidia-driver*",
    "nvidia-fabricmanager*",
    "nvidia-firmware*",
    "nvidia-headless*",
    "nvidia-imex*",
    "nvidia-kernel*",
    "nvidia-modprobe*",
    "nvidia-open*",
    "nvidia-persistenced*",
    "nvidia-settings*",
    "nvidia-xconfig*",
    "xserver-xorg-video-nvidia*",
];

pub fn run(args: UninstallArgs) -> Result<()> {
    let os = os::detect()?;
    if !os.is_supported() {
        bail!(
            "cudaenv uninstall supports Ubuntu only (detected {}).",
            os.display_name()
        );
    }
    let current = environment::detect()?;
    let toolkit_installed = current.toolkit_version.is_some();
    let driver_installed = current.driver_version.is_some();
    if !toolkit_installed && !driver_installed {
        println!("No installed CUDA Toolkit or NVIDIA driver was detected.");
        return Ok(());
    }

    output::uninstall_plan(&uninstall_commands(toolkit_installed, driver_installed));

    if !args.yes && !prompt::confirm_uninstall()? {
        println!("\nUninstall cancelled. No changes were made.");
        return Ok(());
    }

    if toolkit_installed {
        uninstall_cuda_toolkit()?;
    }
    if driver_installed {
        uninstall_driver()?;
    }
    autoremove_packages()?;

    println!("\nDetected CUDA/NVIDIA components were removed.");
    if driver_installed {
        println!("Reboot Ubuntu before installing another driver.");
    }
    Ok(())
}

fn uninstall_cuda_toolkit() -> Result<()> {
    run_apt(
        &["remove", "--purge", "--yes"],
        CUDA_TOOLKIT_PACKAGES,
        "remove CUDA Toolkit packages",
    )
}

fn uninstall_driver() -> Result<()> {
    run_apt(
        &["remove", "--autoremove", "--purge", "-V", "--yes"],
        NVIDIA_DRIVER_PACKAGES,
        "remove NVIDIA driver packages",
    )
}

fn autoremove_packages() -> Result<()> {
    run_apt(
        &["autoremove", "--purge", "--yes"],
        &[],
        "clean up unused CUDA and NVIDIA dependencies",
    )
}

fn run_apt(options: &[&str], packages: &[&str], action: &str) -> Result<()> {
    let status = Command::new("sudo")
        .arg("apt")
        .args(options)
        .args(packages)
        .status()
        .with_context(|| format!("could not start apt to {action}"))?;

    if !status.success() {
        bail!("apt failed to {action} (exit status: {status})");
    }

    Ok(())
}

fn uninstall_commands(toolkit_installed: bool, driver_installed: bool) -> Vec<String> {
    let mut commands = Vec::new();
    if toolkit_installed {
        commands.push(display_apt_command(
            &["remove", "--purge", "--yes"],
            CUDA_TOOLKIT_PACKAGES,
        ));
    }
    if driver_installed {
        commands.push(display_apt_command(
            &["remove", "--autoremove", "--purge", "-V", "--yes"],
            NVIDIA_DRIVER_PACKAGES,
        ));
    }
    if toolkit_installed || driver_installed {
        commands.push(display_apt_command(
            &["autoremove", "--purge", "--yes"],
            &[],
        ));
    }
    commands
}

fn display_apt_command(options: &[&str], packages: &[&str]) -> String {
    let mut parts = vec!["sudo", "apt"];
    parts.extend_from_slice(options);

    let mut command = parts.join(" ");
    for package in packages {
        command.push_str(&format!(" \"{package}\""));
    }
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_contains_toolkit_driver_and_cleanup_commands() {
        let commands = uninstall_commands(true, true);

        assert_eq!(commands.len(), 3);
        assert!(commands[0].contains("\"cuda-toolkit*\""));
        assert!(commands[1].contains("\"nvidia-driver*\""));
        assert_eq!(commands[2], "sudo apt autoremove --purge --yes");
    }

    #[test]
    fn plan_only_removes_detected_components() {
        let toolkit = uninstall_commands(true, false);
        assert_eq!(toolkit.len(), 2);
        assert!(toolkit[0].contains("cuda-toolkit"));
        assert!(!toolkit.join("\n").contains("nvidia-driver"));

        let driver = uninstall_commands(false, true);
        assert_eq!(driver.len(), 2);
        assert!(driver[0].contains("nvidia-driver"));
        assert!(!driver.join("\n").contains("cuda-toolkit"));

        assert!(uninstall_commands(false, false).is_empty());
    }
}
