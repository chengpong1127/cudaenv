use std::{fs, process::Command};

use anyhow::{Context, Result};

use crate::model::environment::DriverModuleInfo;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DriverInspection {
    pub runtime_version: Option<String>,
    pub runtime_operational: bool,
    pub module_loaded: bool,
    pub module_info: Option<DriverModuleInfo>,
    pub kernel_version: Option<String>,
    pub secure_boot_enabled: Option<bool>,
}

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

pub fn inspect() -> Result<DriverInspection> {
    let runtime_version = detect_runtime_version()?;
    Ok(DriverInspection {
        runtime_operational: runtime_version.is_some(),
        runtime_version,
        module_loaded: std::path::Path::new("/sys/module/nvidia").exists(),
        module_info: inspect_module(),
        kernel_version: command_stdout("uname", &["-r"]),
        secure_boot_enabled: secure_boot_enabled(),
    })
}

fn inspect_module() -> Option<DriverModuleInfo> {
    inspect_module_command("modinfo", &["nvidia"])
}

fn inspect_module_command(program: &str, args: &[&str]) -> Option<DriverModuleInfo> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| parse_modinfo(&String::from_utf8_lossy(&output.stdout)))
}

fn parse_modinfo(contents: &str) -> DriverModuleInfo {
    let field = |name: &str| {
        contents.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == name)
                .then(|| value.trim())
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
    };
    DriverModuleInfo {
        path: field("filename"),
        version: field("version"),
        signer: field("signer"),
        signature_id: field("sig_id"),
    }
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
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

fn detect_runtime_version() -> Result<Option<String>> {
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

    #[test]
    fn parses_proc_driver_version() {
        let input = "NVRM version: NVIDIA UNIX Open Kernel Module  570.86.15  Release Build\n";
        assert_eq!(parse_proc_version(input).as_deref(), Some("570.86.15"));
    }

    #[test]
    fn parses_modinfo_fields_without_retaining_raw_output() {
        let input = "filename: /lib/modules/test/nvidia.ko\nversion: 610.43.02\nsigner: Test key\nsig_id: PKCS#7\nalias: lots of raw data\n";
        let info = parse_modinfo(input);
        assert_eq!(info.path.as_deref(), Some("/lib/modules/test/nvidia.ko"));
        assert_eq!(info.version.as_deref(), Some("610.43.02"));
        assert_eq!(info.signer.as_deref(), Some("Test key"));
        assert_eq!(info.signature_id.as_deref(), Some("PKCS#7"));
    }

    #[test]
    fn module_subprocess_stdout_and_stderr_are_captured() {
        let info = inspect_module_command(
            "sh",
            &["-c", "printf 'filename: /test/nvidia.ko\\nversion: 610.43.02\\n'; printf 'diagnostic' >&2"],
        )
        .unwrap();
        assert_eq!(info.version.as_deref(), Some("610.43.02"));
    }
}
