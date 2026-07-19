use std::{fs, process::Command};

use anyhow::{Context, Result, bail};

use crate::{
    model::{
        command::CommandSpec,
        system::{Distribution, OsInfo},
    },
    providers::nvidia::gpu::{Generation, NvidiaGpu},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DriverPreference {
    #[default]
    Auto,
    Open,
    Proprietary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverFlavor {
    Open,
    Proprietary,
}

impl DriverFlavor {
    pub fn package(self) -> &'static str {
        match self {
            Self::Open => "nvidia-open",
            Self::Proprietary => "cuda-drivers",
        }
    }
}

pub fn select(preference: DriverPreference, gpus: &[NvidiaGpu]) -> Result<DriverFlavor> {
    let flavor = match preference {
        DriverPreference::Open => DriverFlavor::Open,
        DriverPreference::Proprietary => DriverFlavor::Proprietary,
        DriverPreference::Auto
            if gpus
                .iter()
                .any(|gpu| gpu.generation == Generation::MaxwellPascalVolta) =>
        {
            DriverFlavor::Proprietary
        }
        DriverPreference::Auto if gpus.iter().any(|gpu| gpu.generation == Generation::Unknown) => {
            bail!(
                "Could not determine whether every NVIDIA GPU supports open kernel modules. Re-run with --driver open or --driver proprietary after checking the GPU generation."
            )
        }
        DriverPreference::Auto => DriverFlavor::Open,
    };
    Ok(flavor)
}

pub fn preparation_commands(os: &OsInfo, flavor: DriverFlavor) -> Vec<CommandSpec> {
    let Some(major) = os
        .version_id
        .split('.')
        .next()
        .and_then(|part| part.trim_start_matches(['v', 'V']).parse::<u32>().ok())
    else {
        return Vec::new();
    };
    let modular = matches!(
        os.distribution,
        Distribution::Rhel
            | Distribution::AlmaLinux
            | Distribution::RockyLinux
            | Distribution::OracleLinux
    ) && matches!(major, 8 | 9)
        || os.distribution == Distribution::AmazonLinux && major == 2023
        || os.distribution == Distribution::KylinOs && major == 11;
    if !modular {
        return Vec::new();
    }
    let stream = match flavor {
        DriverFlavor::Open => "nvidia-driver:open-dkms",
        DriverFlavor::Proprietary => "nvidia-driver:latest-dkms",
    };
    vec![CommandSpec::sudo("dnf", ["module", "enable", "-y", stream])]
}

pub fn kernel_headers_available() -> bool {
    let release = match Command::new("uname").arg("-r").output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        _ => return false,
    };
    std::path::Path::new("/lib/modules")
        .join(release)
        .join("build")
        .exists()
}

pub fn secure_boot_enabled() -> Option<bool> {
    if let Ok(output) = Command::new("mokutil").arg("--sb-state").output()
        && output.status.success()
    {
        let state = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
        if state.contains("secureboot enabled") || state.contains("secure boot enabled") {
            return Some(true);
        }
        if state.contains("secureboot disabled") || state.contains("secure boot disabled") {
            return Some(false);
        }
    }
    let entries = fs::read_dir("/sys/firmware/efi/efivars").ok()?;
    for entry in entries.flatten() {
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with("SecureBoot-")
        {
            return fs::read(entry.path())
                .ok()?
                .last()
                .copied()
                .map(|value| value == 1);
        }
    }
    None
}

pub fn detect_version() -> Result<Option<String>> {
    if let Some(version) = version_from_nvidia_smi()? {
        return Ok(Some(version));
    }
    version_from_proc()
}

fn version_from_nvidia_smi() -> Result<Option<String>> {
    let output = match Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("failed to run nvidia-smi"),
    };
    if !output.status.success() {
        return Ok(None);
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string))
}

fn version_from_proc() -> Result<Option<String>> {
    let contents = match fs::read_to_string("/proc/driver/nvidia/version") {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("failed to read the NVIDIA driver version"),
    };
    Ok(parse_proc_version(&contents))
}

fn parse_proc_version(contents: &str) -> Option<String> {
    let marker = "Kernel Module  ";
    contents
        .lines()
        .find_map(|line| line.split_once(marker).map(|(_, rest)| rest))
        .and_then(|rest| rest.split_whitespace().next())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(generation: Generation) -> NvidiaGpu {
        NvidiaGpu {
            name: "GPU".into(),
            pci_device_id: None,
            generation,
        }
    }

    #[test]
    fn parses_proc_driver_version() {
        let input = "NVRM version: NVIDIA UNIX Open Kernel Module  570.86.15  Release Build\n";
        assert_eq!(parse_proc_version(input).as_deref(), Some("570.86.15"));
    }

    #[test]
    fn selects_flavor_for_gpu_generation() {
        assert_eq!(
            select(DriverPreference::Auto, &[gpu(Generation::TuringOrNewer)]).unwrap(),
            DriverFlavor::Open
        );
        assert_eq!(
            select(
                DriverPreference::Auto,
                &[gpu(Generation::MaxwellPascalVolta)]
            )
            .unwrap(),
            DriverFlavor::Proprietary
        );
        assert!(
            select(DriverPreference::Auto, &[gpu(Generation::Unknown)])
                .unwrap_err()
                .to_string()
                .contains("--driver")
        );
        assert_eq!(
            select(
                DriverPreference::Auto,
                &[
                    gpu(Generation::TuringOrNewer),
                    gpu(Generation::MaxwellPascalVolta),
                ],
            )
            .unwrap(),
            DriverFlavor::Proprietary
        );
    }

    #[test]
    fn prepares_modular_dnf_distributions() {
        let os = OsInfo {
            distribution: Distribution::Rhel,
            name: "RHEL".into(),
            version_id: "9.7".into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        };
        assert!(
            preparation_commands(&os, DriverFlavor::Open)[0]
                .display()
                .contains("nvidia-driver:open-dkms")
        );
    }
}
