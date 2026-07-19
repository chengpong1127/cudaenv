use std::{fs, path::Path, process::Command};

use anyhow::{Context, Result};

use crate::model::device::{GpuDevice, GpuVendor};

const PCI_VENDOR_ID: u32 = 0x10de;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Generation {
    MaxwellPascalVolta,
    TuringOrNewer,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaGpu {
    pub name: String,
    pub pci_device_id: Option<u16>,
    pub generation: Generation,
}

impl From<NvidiaGpu> for GpuDevice {
    fn from(gpu: NvidiaGpu) -> Self {
        Self {
            vendor: GpuVendor::Nvidia,
            name: gpu.name,
            pci_device_id: gpu.pci_device_id,
        }
    }
}

pub fn detect() -> Result<Vec<NvidiaGpu>> {
    let sysfs_devices = detect_sysfs()?;
    let pci_devices = detect_with_lspci()?;
    if !pci_devices.is_empty() {
        return Ok(pci_devices);
    }
    Ok(sysfs_devices)
}

fn detect_sysfs() -> Result<Vec<NvidiaGpu>> {
    let root = Path::new("/sys/bus/pci/devices");
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).context("could not inspect PCI devices in sysfs"),
    };
    let mut devices = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if read_hex(&path.join("vendor")) != Some(PCI_VENDOR_ID) {
            continue;
        }
        let class = read_hex(&path.join("class")).unwrap_or(0);
        if class >> 16 != 0x03 {
            continue;
        }
        let id = read_hex(&path.join("device")).map(|value| value as u16);
        let name = id.map_or_else(
            || "NVIDIA GPU".to_owned(),
            |id| format!("NVIDIA GPU [10de:{id:04x}]"),
        );
        devices.push(NvidiaGpu {
            generation: classify_device_id(id),
            name,
            pci_device_id: id,
        });
    }
    Ok(devices)
}

fn read_hex(path: &Path) -> Option<u32> {
    let value = fs::read_to_string(path).ok()?;
    u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok()
}

fn detect_with_lspci() -> Result<Vec<NvidiaGpu>> {
    let output = match Command::new("lspci").args(["-Dnn"]).output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).context("failed to run lspci"),
    };
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(parse_nvidia_lspci_line)
        .collect())
}

fn parse_nvidia_lspci_line(line: &str) -> Option<NvidiaGpu> {
    let lowercase = line.to_ascii_lowercase();
    if !lowercase.contains("[10de:")
        || !(lowercase.contains("vga compatible controller")
            || lowercase.contains("3d controller")
            || lowercase.contains("display controller"))
    {
        return None;
    }
    let id = lowercase
        .split("[10de:")
        .nth(1)?
        .get(..4)
        .and_then(|id| u16::from_str_radix(id, 16).ok());
    let description = line.split_once(": ").map_or(line, |(_, value)| value);
    let name = description
        .split(" (rev ")
        .next()
        .unwrap_or(description)
        .trim()
        .to_owned();
    let by_name = classify_name(&name);
    Some(NvidiaGpu {
        name,
        pci_device_id: id,
        generation: if by_name == Generation::Unknown {
            classify_device_id(id)
        } else {
            by_name
        },
    })
}

fn classify_name(name: &str) -> Generation {
    let upper = name.to_ascii_uppercase();
    if [" TU", " GA", " GH", " AD", " GB", " GN"]
        .iter()
        .any(|marker| upper.contains(marker))
        || [
            "RTX 20",
            "RTX 30",
            "RTX 40",
            "RTX 50",
            "TITAN RTX",
            "TESLA T4",
            "A100",
            "A10",
            "A30",
            "A40",
            "H100",
            "H200",
            "L4",
            "L40",
            "B100",
            "B200",
        ]
        .iter()
        .any(|marker| upper.contains(marker))
    {
        Generation::TuringOrNewer
    } else if [" GM", " GP", " GV"]
        .iter()
        .any(|marker| upper.contains(marker))
        || [
            "GTX 9",
            "GTX 10",
            "TITAN V",
            "TESLA M",
            "TESLA P",
            "TESLA V",
            "QUADRO M",
            "QUADRO P",
            "QUADRO GV",
        ]
        .iter()
        .any(|marker| upper.contains(marker))
    {
        Generation::MaxwellPascalVolta
    } else {
        Generation::Unknown
    }
}

fn classify_device_id(id: Option<u16>) -> Generation {
    let Some(id) = id else {
        return Generation::Unknown;
    };
    match id {
        0x1340..=0x13ff | 0x15f0..=0x15ff | 0x17c0..=0x17ff | 0x1b00..=0x1dff => {
            Generation::MaxwellPascalVolta
        }
        0x1e00..=0x1fff | 0x20b0..=0x28ff | 0x2b00..=0x2fff => Generation::TuringOrNewer,
        _ => Generation::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_classifies_turing_gpu() {
        let gpu = parse_nvidia_lspci_line("0000:01:00.0 VGA compatible controller: NVIDIA Corporation TU104 [GeForce RTX 2080] [10de:1e87] (rev a1)").unwrap();
        assert_eq!(gpu.generation, Generation::TuringOrNewer);
    }

    #[test]
    fn parses_old_gpu() {
        let gpu = parse_nvidia_lspci_line(
            "0000:01:00.0 3D controller: NVIDIA Corporation GP100 [Tesla P100] [10de:15f8]",
        )
        .unwrap();
        assert_eq!(gpu.generation, Generation::MaxwellPascalVolta);
    }

    #[test]
    fn exposes_vendor_neutral_device() {
        let device: GpuDevice = NvidiaGpu {
            name: "GPU".into(),
            pci_device_id: Some(1),
            generation: Generation::Unknown,
        }
        .into();
        assert_eq!(device.vendor, GpuVendor::Nvidia);
    }

    #[test]
    fn leaves_unrecognized_gpu_generation_unknown() {
        let gpu = parse_nvidia_lspci_line(
            "0000:01:00.0 VGA compatible controller: NVIDIA Corporation Device [10de:0001]",
        )
        .unwrap();
        assert_eq!(gpu.generation, Generation::Unknown);
    }
}
