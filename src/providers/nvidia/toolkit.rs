use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::model::command::CommandSpec;

const LATEST_TOOLKIT_PACKAGE: &str = "cuda-toolkit";

pub fn package(version: Option<&str>) -> Result<String> {
    match version {
        Some(version) => versioned_package(version),
        None => Ok(LATEST_TOOLKIT_PACKAGE.to_owned()),
    }
}

pub fn detect_version() -> Result<Option<String>> {
    for nvcc in ["nvcc", "/usr/local/cuda/bin/nvcc"] {
        let output = match Command::new(nvcc).arg("--version").output() {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error).with_context(|| format!("failed to run {nvcc}")),
        };
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(version) = parse_nvcc_version(&stdout) {
                return Ok(Some(version.to_owned()));
            }
        }
    }
    Ok(None)
}

fn parse_nvcc_version(output: &str) -> Option<&str> {
    let (_, after_release) = output.split_once("release ")?;
    after_release
        .split(|character: char| character == ',' || character.is_whitespace())
        .find(|part| !part.is_empty())
}

pub fn versioned_package(version: &str) -> Result<String> {
    let normalized = version.trim().replace('.', "-");
    let mut parts = normalized.split('-');
    let (Some(major), Some(minor), None) = (parts.next(), parts.next(), parts.next()) else {
        bail!("invalid CUDA Toolkit version {version:?}; expected MAJOR.MINOR, for example 13.3");
    };
    if major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || !minor.bytes().all(|byte| byte.is_ascii_digit())
    {
        bail!("invalid CUDA Toolkit version {version:?}; expected MAJOR.MINOR, for example 13.3");
    }
    Ok(format!("cuda-toolkit-{major}-{minor}"))
}

pub fn verification_command() -> CommandSpec {
    CommandSpec::new("nvcc", ["--version"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_versioned_package_names() {
        assert_eq!(versioned_package("13.3").unwrap(), "cuda-toolkit-13-3");
        assert_eq!(versioned_package("12-8").unwrap(), "cuda-toolkit-12-8");
        for invalid in ["13", "13.3.0", "latest", "13.x", ""] {
            assert!(versioned_package(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn parses_nvcc_toolkit_version() {
        let output = "Cuda compilation tools, release 13.1, V13.1.80\n";
        assert_eq!(parse_nvcc_version(output), Some("13.1"));
    }

    #[test]
    fn uses_latest_meta_package_when_version_is_not_pinned() {
        assert_eq!(package(None).unwrap(), "cuda-toolkit");
        assert_eq!(package(Some("13.3")).unwrap(), "cuda-toolkit-13-3");
    }
}
