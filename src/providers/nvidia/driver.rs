use std::{fs, process::Command};

use anyhow::{Context, Result};

use crate::providers::nvidia::gpu::{Generation, NvidiaGpu};

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

pub fn select(preference: DriverPreference, gpus: &[NvidiaGpu]) -> DriverFlavor {
    match preference {
        DriverPreference::Open => DriverFlavor::Open,
        DriverPreference::Proprietary => DriverFlavor::Proprietary,
        DriverPreference::Auto
            if gpus
                .iter()
                .any(|gpu| gpu.generation == Generation::MaxwellPascalVolta) =>
        {
            DriverFlavor::Proprietary
        }
        DriverPreference::Auto => DriverFlavor::Open,
    }
}

pub fn detect_version() -> Result<Option<String>> {
    if let Some(version) = version_from_nvidia_smi()? {
        return Ok(Some(version));
    }
    version_from_proc()
}

pub fn nvidia_smi_available() -> bool {
    Command::new("nvidia-smi").arg("--help").output().is_ok()
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
            select(DriverPreference::Auto, &[gpu(Generation::TuringOrNewer)]),
            DriverFlavor::Open
        );
        assert_eq!(
            select(
                DriverPreference::Auto,
                &[gpu(Generation::MaxwellPascalVolta)]
            ),
            DriverFlavor::Proprietary
        );
        assert_eq!(
            select(DriverPreference::Auto, &[gpu(Generation::Unknown)]),
            DriverFlavor::Open
        );
        assert_eq!(
            select(
                DriverPreference::Auto,
                &[
                    gpu(Generation::TuringOrNewer),
                    gpu(Generation::MaxwellPascalVolta),
                ],
            ),
            DriverFlavor::Proprietary
        );
    }
}
