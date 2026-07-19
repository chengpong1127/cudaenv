use anyhow::{Result, bail};

use crate::{
    model::{
        command::CommandSpec,
        system::{Distribution, OsInfo},
    },
    platform::package_manager,
};

use super::{driver::DriverFlavor, policy::DriverPolicy, repository};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstallRecipe {
    pub prerequisites: Vec<CommandSpec>,
    pub driver_preparation: Vec<CommandSpec>,
    pub driver_install: CommandSpec,
    pub driver_verification: CommandSpec,
}

pub fn validate_release(os: &OsInfo) -> Result<()> {
    repository::resolve(os).map(|_| ())
}

pub fn resolve(os: &OsInfo, kernel_release: &str, policy: DriverPolicy) -> Result<InstallRecipe> {
    validate_release(os)?;
    let prerequisites = prerequisites(os, kernel_release)?;
    let driver_preparation = branch_preparation(os, policy);
    let driver_install =
        package_manager::install_command(os.package_manager(), policy.flavor.package());
    Ok(InstallRecipe {
        prerequisites,
        driver_preparation,
        driver_install,
        driver_verification: CommandSpec::new("modinfo", ["nvidia"]),
    })
}

pub fn prerequisites(os: &OsInfo, kernel: &str) -> Result<Vec<CommandSpec>> {
    validate_release(os)?;
    let commands = match os.distribution {
        Distribution::Ubuntu => vec![CommandSpec::sudo(
            "apt-get",
            ["install", "-y", &format!("linux-headers-{kernel}")],
        )],
        Distribution::Debian => vec![
            CommandSpec::sudo("apt-get", ["install", "-y", "software-properties-common"]),
            CommandSpec::sudo("add-apt-repository", ["-y", "contrib"]),
            CommandSpec::sudo("apt-get", ["update"]),
            CommandSpec::sudo(
                "apt-get",
                ["install", "-y", &format!("linux-headers-{kernel}")],
            ),
        ],
        Distribution::Rhel => rhel_prerequisites(os, kernel, true),
        Distribution::AlmaLinux | Distribution::RockyLinux => rhel_prerequisites(os, kernel, false),
        Distribution::OracleLinux => oracle_prerequisites(os, kernel),
        Distribution::Fedora => vec![CommandSpec::sudo(
            "dnf",
            [
                "install",
                "-y",
                &format!("kernel-devel-{kernel}"),
                "kernel-headers",
            ],
        )],
        Distribution::AmazonLinux => vec![CommandSpec::sudo(
            "dnf",
            [
                "install",
                "-y",
                &format!("kernel-devel-{kernel}"),
                &format!("kernel-headers-{kernel}"),
            ],
        )],
        Distribution::AzureLinux => vec![
            CommandSpec::sudo("tdnf", ["install", "-y", "azurelinux-repos-extended"]),
            CommandSpec::sudo(
                "tdnf",
                [
                    "install",
                    "-y",
                    &format!("kernel-devel-{kernel}"),
                    &format!("kernel-headers-{kernel}"),
                    &format!("kernel-modules-extra-{kernel}"),
                ],
            ),
        ],
        Distribution::OpenSuse | Distribution::Sles => suse_prerequisites(os, kernel)?,
        Distribution::KylinOs => vec![CommandSpec::sudo(
            "dnf",
            [
                "install",
                "-y",
                &format!("kernel-devel-{kernel}"),
                "kernel-headers",
            ],
        )],
    };
    Ok(commands)
}

fn rhel_prerequisites(os: &OsInfo, kernel: &str, subscription: bool) -> Vec<CommandSpec> {
    let release = major(os).unwrap_or_default();
    let mut commands = Vec::new();
    if release <= 9 {
        if subscription {
            commands.push(CommandSpec::sudo(
                "subscription-manager",
                [
                    "repos",
                    &format!(
                        "--enable=codeready-builder-for-rhel-{release}-{}-rpms",
                        repo_arch(os)
                    ),
                ],
            ));
            commands.push(CommandSpec::sudo("dnf", ["install", "-y", &format!("https://dl.fedoraproject.org/pub/epel/epel-release-latest-{release}.noarch.rpm")]));
        } else {
            commands.push(CommandSpec::sudo(
                "dnf",
                [
                    "config-manager",
                    "--set-enabled",
                    if release == 8 { "powertools" } else { "crb" },
                ],
            ));
            commands.push(CommandSpec::sudo("dnf", ["install", "-y", "epel-release"]));
        }
    }
    if release == 8 {
        commands.push(CommandSpec::sudo(
            "dnf",
            [
                "install",
                "-y",
                &format!("kernel-devel-{kernel}"),
                "kernel-headers",
            ],
        ));
    } else if os.architecture == "aarch64" && kernel.contains("64k") {
        commands.push(CommandSpec::sudo(
            "dnf",
            [
                "install",
                "-y",
                "kernel-64k-devel-matched",
                "kernel-headers",
            ],
        ));
    } else {
        commands.push(CommandSpec::sudo(
            "dnf",
            ["install", "-y", "kernel-devel-matched", "kernel-headers"],
        ));
    }
    commands
}

fn oracle_prerequisites(os: &OsInfo, kernel: &str) -> Vec<CommandSpec> {
    let release = major(os).unwrap_or_default();
    let mut commands = vec![
        CommandSpec::sudo(
            "dnf",
            [
                "config-manager",
                "--set-enabled",
                &format!("ol{release}_codeready_builder"),
            ],
        ),
        CommandSpec::sudo(
            "dnf",
            ["install", "-y", &format!("oracle-epel-release-el{release}")],
        ),
    ];
    let headers = if kernel.contains("uek") {
        format!("kernel-uek-devel-{kernel}")
    } else if release == 8 {
        format!("kernel-devel-{kernel}")
    } else if os.architecture == "aarch64" && kernel.contains("64k") {
        "kernel-64k-devel-matched".into()
    } else {
        "kernel-devel-matched".into()
    };
    commands.push(CommandSpec::sudo(
        "dnf",
        ["install", "-y", &headers, "kernel-headers"],
    ));
    commands
}

fn suse_prerequisites(os: &OsInfo, kernel: &str) -> Result<Vec<CommandSpec>> {
    let (version, variant) = kernel
        .rsplit_once('-')
        .ok_or_else(|| anyhow::anyhow!("could not determine SUSE kernel variant from {kernel}"))?;
    if !matches!(variant, "default" | "64k" | "azure") {
        bail!("unsupported SUSE kernel variant {variant:?}; expected default, 64k, or azure");
    }
    let mut commands = Vec::new();
    if os.distribution == Distribution::Sles && os.version_id.starts_with("15.") {
        commands.push(CommandSpec::sudo(
            "SUSEConnect",
            ["--product", &format!("PackageHub/15/{}", repo_arch(os))],
        ));
    }
    commands.push(CommandSpec::sudo(
        "zypper",
        [
            "--non-interactive",
            "install",
            "-y",
            &format!("kernel-{variant}-devel={version}"),
        ],
    ));
    Ok(commands)
}

fn branch_preparation(os: &OsInfo, policy: DriverPolicy) -> Vec<CommandSpec> {
    let Some(branch) = policy.branch else {
        if modular_dnf(os) {
            let stream: String = match policy.flavor {
                DriverFlavor::Open => "nvidia-driver:open-dkms".into(),
                DriverFlavor::Proprietary => "nvidia-driver:latest-dkms".into(),
            };
            return vec![CommandSpec::sudo(
                "dnf",
                ["module", "enable", "-y", &stream],
            )];
        }
        return Vec::new();
    };
    match os.distribution {
        Distribution::Ubuntu | Distribution::Debian => vec![CommandSpec::sudo(
            "apt-get",
            ["install", "-y", &format!("nvidia-driver-pinning-{branch}")],
        )],
        Distribution::Rhel
        | Distribution::AlmaLinux
        | Distribution::RockyLinux
        | Distribution::OracleLinux
        | Distribution::AmazonLinux
        | Distribution::KylinOs
            if modular_dnf(os) =>
        {
            vec![CommandSpec::sudo(
                "dnf",
                [
                    "module",
                    "enable",
                    "-y",
                    &format!("nvidia-driver:{branch}-dkms"),
                ],
            )]
        }
        Distribution::Fedora => vec![CommandSpec::sudo(
            "dnf",
            ["versionlock", "add", &format!("*nvidia*{branch}*")],
        )],
        Distribution::Rhel | Distribution::AlmaLinux | Distribution::RockyLinux => {
            vec![CommandSpec::sudo(
                "dnf",
                ["versionlock", &format!("*nvidia*{branch}*")],
            )]
        }
        Distribution::OpenSuse | Distribution::Sles => vec![CommandSpec::sudo(
            "zypper",
            ["addlock", &format!("*nvidia* >= {}", branch + 10)],
        )],
        _ => Vec::new(),
    }
}

pub fn transition_commands(
    os: &OsInfo,
    policy: DriverPolicy,
    from: DriverFlavor,
) -> Vec<CommandSpec> {
    let target = policy.flavor.package();
    let old = from.package();
    let flavor_change = from != policy.flavor;
    match os.distribution {
        Distribution::Ubuntu | Distribution::Debian => {
            let mut commands = if flavor_change {
                vec![CommandSpec::sudo(
                    "apt-get",
                    ["remove", "--purge", "--autoremove", "-y", old],
                )]
            } else {
                Vec::new()
            };
            commands.extend(branch_preparation(os, policy));
            commands.push(CommandSpec::sudo("apt-get", ["install", "-y", target]));
            commands
        }
        Distribution::Rhel
        | Distribution::AlmaLinux
        | Distribution::RockyLinux
        | Distribution::OracleLinux
        | Distribution::AmazonLinux
        | Distribution::KylinOs
            if modular_dnf(os) =>
        {
            let stream = if let Some(branch) = policy.branch {
                format!("nvidia-driver:{branch}-dkms")
            } else {
                format!(
                    "nvidia-driver:{}",
                    if policy.flavor == DriverFlavor::Open {
                        "open-dkms"
                    } else {
                        "latest-dkms"
                    }
                )
            };
            vec![CommandSpec::sudo(
                "dnf",
                ["-y", "module", "switch-to", &stream, "--allowerasing"],
            )]
        }
        Distribution::AzureLinux => vec![CommandSpec::sudo(
            "tdnf",
            ["install", "-y", "--allowerasing", target],
        )],
        Distribution::OpenSuse | Distribution::Sles => {
            let mut commands = branch_preparation(os, policy);
            commands.push(CommandSpec::sudo(
                "zypper",
                ["--non-interactive", "install", "--force-resolution", target],
            ));
            commands
        }
        _ => vec![CommandSpec::sudo(
            "dnf",
            ["install", "-y", "--allowerasing", target],
        )],
    }
}

pub(crate) fn is_modular_dnf(os: &OsInfo) -> bool {
    matches!(
        os.distribution,
        Distribution::Rhel
            | Distribution::AlmaLinux
            | Distribution::RockyLinux
            | Distribution::OracleLinux
    ) && matches!(major(os), Some(8 | 9))
        || matches!(
            os.distribution,
            Distribution::AmazonLinux | Distribution::KylinOs
        )
}

fn modular_dnf(os: &OsInfo) -> bool {
    is_modular_dnf(os)
}

fn major(os: &OsInfo) -> Option<u32> {
    os.version_id
        .trim_start_matches(['v', 'V'])
        .split(['.', ' ', '-'])
        .next()?
        .parse()
        .ok()
}
fn repo_arch(os: &OsInfo) -> &'static str {
    if matches!(os.architecture.as_str(), "aarch64" | "arm64") {
        "aarch64"
    } else {
        "x86_64"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::nvidia::policy::LEGACY_DRIVER_BRANCH;

    fn os(distribution: Distribution, version: &str) -> OsInfo {
        OsInfo {
            distribution,
            name: "Test".into(),
            version_id: version.into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        }
    }
    fn policy(flavor: DriverFlavor, legacy: bool) -> DriverPolicy {
        DriverPolicy {
            flavor,
            branch: legacy.then_some(LEGACY_DRIVER_BRANCH),
            legacy_gpu: legacy,
        }
    }
    fn displays(commands: Vec<CommandSpec>) -> String {
        commands
            .into_iter()
            .map(|c| c.display())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn snapshots_distro_specific_prerequisites() {
        let cases = [
            (
                os(Distribution::Ubuntu, "24.04"),
                "6.8.0-generic",
                "linux-headers-6.8.0-generic",
            ),
            (
                os(Distribution::Debian, "13"),
                "6.12.0-amd64",
                "add-apt-repository -y contrib",
            ),
            (
                os(Distribution::Rhel, "9.7"),
                "5.14.0",
                "codeready-builder-for-rhel-9-x86_64-rpms",
            ),
            (
                os(Distribution::AlmaLinux, "10.1"),
                "6.12.0",
                "kernel-devel-matched",
            ),
            (os(Distribution::RockyLinux, "8.10"), "4.18.0", "powertools"),
            (
                os(Distribution::OracleLinux, "9"),
                "5.15.0-uek",
                "kernel-uek-devel-5.15.0-uek",
            ),
            (
                os(Distribution::Fedora, "44"),
                "6.15.0",
                "kernel-devel-6.15.0",
            ),
            (
                os(Distribution::AmazonLinux, "2023"),
                "6.1.0",
                "kernel-headers-6.1.0",
            ),
            (
                os(Distribution::AzureLinux, "3.0"),
                "6.6.0",
                "kernel-modules-extra-6.6.0",
            ),
            (
                os(Distribution::OpenSuse, "15.6"),
                "6.4.0-default",
                "kernel-default-devel=6.4.0",
            ),
            (
                os(Distribution::Sles, "15.7"),
                "6.4.0-azure",
                "PackageHub/15/x86_64",
            ),
            (
                os(Distribution::KylinOs, "V11 2503"),
                "6.6.0",
                "kernel-devel-6.6.0",
            ),
        ];
        for (os, kernel, expected) in cases {
            assert!(
                displays(prerequisites(&os, kernel).unwrap()).contains(expected),
                "{}",
                os.display_name()
            );
        }
    }

    #[test]
    fn snapshots_each_transition_family() {
        assert!(
            displays(transition_commands(
                &os(Distribution::Ubuntu, "24.04"),
                policy(DriverFlavor::Proprietary, true),
                DriverFlavor::Open
            ))
            .contains("--autoremove")
        );
        assert!(
            displays(transition_commands(
                &os(Distribution::Rhel, "9.7"),
                policy(DriverFlavor::Open, false),
                DriverFlavor::Proprietary
            ))
            .contains("module switch-to nvidia-driver:open-dkms --allowerasing")
        );
        assert!(
            displays(transition_commands(
                &os(Distribution::Fedora, "44"),
                policy(DriverFlavor::Open, false),
                DriverFlavor::Proprietary
            ))
            .contains("dnf install -y --allowerasing nvidia-open")
        );
        assert!(
            displays(transition_commands(
                &os(Distribution::AzureLinux, "3.0"),
                policy(DriverFlavor::Open, false),
                DriverFlavor::Proprietary
            ))
            .contains("tdnf install -y --allowerasing")
        );
        assert!(
            displays(transition_commands(
                &os(Distribution::OpenSuse, "15.6"),
                policy(DriverFlavor::Open, false),
                DriverFlavor::Proprietary
            ))
            .contains("--force-resolution")
        );
    }

    #[test]
    fn validates_exact_service_packs_and_releases() {
        assert!(validate_release(&os(Distribution::OpenSuse, "15.5")).is_err());
        assert!(validate_release(&os(Distribution::Sles, "15.5")).is_err());
        assert!(validate_release(&os(Distribution::KylinOs, "V11 2403")).is_err());
    }
}
