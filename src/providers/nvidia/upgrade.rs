//! Upgrade planning is deliberately separate from installation planning: it may
//! update a component only after proving that component is already managed.

use std::{cmp::Ordering, fs, path::Path, process::Command};

use anyhow::{Context, Result, anyhow, bail};

use crate::{
    model::{
        command::CommandSpec,
        environment::{DriverFlavorState, DriverInstallation, ProviderStatus},
        operation::{OperationPlan, PlanDetail, PlanStep},
        system::{Distribution, OsInfo, PackageManager},
    },
    platform::package_manager,
};

use super::{
    compatibility, driver,
    gpu::{self, Generation, NvidiaGpu},
    policy, recipe, repository, state,
};

const ACTIONABLE: &str = "upgrade-state:";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UpgradeOptions {
    pub driver: bool,
    pub toolkit: bool,
}

impl UpgradeOptions {
    pub fn from_component_flags(driver: bool, toolkit: bool) -> Self {
        if driver || toolkit {
            Self { driver, toolkit }
        } else {
            Self {
                driver: true,
                toolkit: true,
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageRecord {
    pub name: String,
    pub installed: String,
    pub candidate: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolkitTracking {
    Unversioned,
    Major(u32),
    Exact(u32, u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolkitInstall {
    pub package: PackageRecord,
    pub tracking: ToolkitTracking,
    pub toolkit_version: Option<String>,
}

#[derive(Clone, Debug)]
pub struct UpgradeState {
    pub status: ProviderStatus,
    pub driver_package: Option<PackageRecord>,
    pub toolkits: Vec<ToolkitInstall>,
    pub restrictions: Vec<String>,
    pub repository_configured: bool,
    pub kernel: String,
    pub kernel_headers: bool,
    pub secure_boot: Option<bool>,
    pub cuda_link: CudaLink,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaLink {
    Missing,
    Managed(String),
    UserManaged(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolkitCandidate {
    pub package: String,
    pub version: String,
    pub package_version: String,
}

/// Boundary used by both the real planner and deterministic tests. Implementors
/// return package-manager versions verbatim; the planner compares them safely.
pub trait PackageQuery {
    fn installed_version(&self, manager: PackageManager, package: &str) -> Result<Option<String>>;
    fn candidate_version(&self, manager: PackageManager, package: &str) -> Result<Option<String>>;
    fn restrictions(&self, manager: PackageManager) -> Result<Vec<String>>;
}

pub struct SystemPackageQuery;

impl PackageQuery for SystemPackageQuery {
    fn installed_version(&self, manager: PackageManager, package: &str) -> Result<Option<String>> {
        let output = match manager {
            PackageManager::AptGet => Command::new("dpkg-query")
                .args(["-W", "-f=${Version}", package])
                .output(),
            _ => Command::new("rpm")
                .args(["-q", "--qf", "%{EVR}", package])
                .output(),
        }
        .with_context(|| format!("could not query installed version of {package}"))?;
        Ok(output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .filter(|v| !v.is_empty()))
    }

    fn candidate_version(&self, manager: PackageManager, package: &str) -> Result<Option<String>> {
        let mut command = match manager {
            PackageManager::AptGet => {
                let mut value = Command::new("apt-cache");
                value.args(["policy", package]);
                value
            }
            PackageManager::Dnf => {
                let mut value = Command::new("dnf");
                value.args([
                    "--quiet",
                    "repoquery",
                    "--latest-limit",
                    "1",
                    "--qf",
                    "%{EVR}",
                    package,
                ]);
                value
            }
            PackageManager::Tdnf => {
                let mut value = Command::new("tdnf");
                value.args(["info", package]);
                value
            }
            PackageManager::Zypper => {
                let mut value = Command::new("zypper");
                value.args(["--non-interactive", "info", package]);
                value
            }
        };
        let output = command
            .output()
            .with_context(|| format!("could not query upgrade candidate for {package}"))?;
        if !output.status.success() {
            return Ok(None);
        }
        Ok(parse_candidate(
            manager,
            &String::from_utf8_lossy(&output.stdout),
        ))
    }

    fn restrictions(&self, manager: PackageManager) -> Result<Vec<String>> {
        let commands: &[(&str, &[&str], &str)] = match manager {
            PackageManager::AptGet => &[("apt-mark", &["showhold"], "APT hold")],
            PackageManager::Dnf => &[
                ("dnf", &["versionlock", "list"], "DNF version lock"),
                (
                    "dnf",
                    &["module", "list", "nvidia-driver", "--enabled"],
                    "DNF module stream",
                ),
            ],
            PackageManager::Tdnf => &[],
            PackageManager::Zypper => {
                &[("zypper", &["locks", "--solvable-name-only"], "Zypper lock")]
            }
        };
        let mut result = Vec::new();
        for (program, args, kind) in commands {
            if let Ok(output) = Command::new(program).args(*args).output()
                && output.status.success()
            {
                result.extend(
                    String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .map(str::trim)
                        .filter(|v| !v.is_empty())
                        .map(|line| format!("{kind}: {line}")),
                );
            }
        }
        if manager == PackageManager::AptGet {
            for root in ["/etc/apt/preferences", "/etc/apt/preferences.d"] {
                collect_text_files(Path::new(root), "APT preference", &mut result)?;
            }
        }
        Ok(result)
    }
}

pub fn is_actionable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().contains(ACTIONABLE))
}

pub fn actionable_message(error: &anyhow::Error) -> String {
    format!("{error:#}").replace(&format!("{ACTIONABLE} "), "")
}

fn blocked(message: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{ACTIONABLE} {message}")
}

pub fn plan(os: &OsInfo, options: &UpgradeOptions) -> Result<OperationPlan> {
    os.ensure_driver_installable("NVIDIA")?;
    let gpus = gpu::detect()?;
    let packages = state::installed_packages(os)?;
    let query = SystemPackageQuery;
    let current = detect_state(os, &packages, &query)?;
    build_plan(os, options, &gpus, &current, &query)
}

pub fn detect_state(
    os: &OsInfo,
    packages: &[String],
    query: &impl PackageQuery,
) -> Result<UpgradeState> {
    let status = state::inspect(os)?;
    let manager = os.package_manager();
    let driver_package = primary_driver_package(&status.driver)
        .map(|name| package_record(query, manager, name))
        .transpose()?;
    let mut toolkits = Vec::new();
    for name in packages
        .iter()
        .filter(|name| toolkit_tracking(name).is_some())
    {
        if let Some(installed) = query.installed_version(manager, name)? {
            toolkits.push(ToolkitInstall {
                package: PackageRecord {
                    name: name.clone(),
                    installed,
                    candidate: query.candidate_version(manager, name)?,
                },
                tracking: toolkit_tracking(name).unwrap(),
                toolkit_version: toolkit_version_from_name(name).or_else(|| {
                    status
                        .toolkits
                        .first()
                        .and_then(|toolkit| toolkit.version.clone())
                }),
            });
        }
    }
    toolkits.sort_by(|a, b| a.package.name.cmp(&b.package.name));
    let repo = repository::resolve(os)?;
    Ok(UpgradeState {
        status,
        driver_package,
        toolkits,
        restrictions: query.restrictions(manager)?,
        repository_configured: repository::is_configured(os, &repo)?,
        kernel: kernel_release()?,
        kernel_headers: driver::kernel_headers_available(),
        secure_boot: driver::secure_boot_enabled(),
        cuda_link: detect_cuda_link(),
    })
}

pub fn build_plan(
    os: &OsInfo,
    options: &UpgradeOptions,
    gpus: &[NvidiaGpu],
    current: &UpgradeState,
    query: &impl PackageQuery,
) -> Result<OperationPlan> {
    validate_gpus(os, gpus)?;
    let repository = repository::resolve(os)?;
    let legacy = gpus
        .iter()
        .any(|gpu| gpu.generation == Generation::MaxwellPascalVolta);
    let driver_selected = options.driver;
    let toolkit_selected = options.toolkit;

    if driver_selected {
        validate_driver_state(os, current, legacy)?;
    }
    if toolkit_selected && !current.toolkits.is_empty() {
        validate_driver_manageability(current)?;
    }

    let driver_target = if driver_selected {
        if let Some(package) = current.driver_package.as_ref() {
            let candidate = if legacy
                && current.status.driver.branch().is_none()
                && !has_r580_restriction(&current.restrictions)
            {
                query.candidate_version(os.package_manager(), "cuda-drivers-580")?
            } else {
                package.candidate.clone()
            };
            candidate
                .filter(|candidate| version_cmp(candidate, &package.installed) == Ordering::Greater)
        } else {
            None
        }
    } else {
        None
    };
    let resulting_driver = driver_target
        .as_deref()
        .and_then(upstream_driver_version)
        .or(current.status.driver_version.as_deref());

    let toolkit_candidate = if toolkit_selected && !current.toolkits.is_empty() {
        resolve_toolkit_candidate(os.package_manager(), &current.toolkits, legacy, query)?
    } else {
        None
    };
    if let Some(candidate) = &toolkit_candidate {
        let Some(driver_version) = resulting_driver else {
            return Err(blocked(
                "a CUDA Toolkit upgrade cannot be validated because no working installed or planned driver version was detected",
            ));
        };
        if compatibility::evaluate(driver_version, &candidate.version)
            != Some(compatibility::Compatibility::Incompatible)
        {
            // compatible, including minor-version compatibility
        } else {
            return Err(blocked(format!(
                "CUDA Toolkit {} requires a newer driver than the compatible target {driver_version}",
                candidate.version
            )));
        }
    }

    let driver_changes = driver_target.is_some();
    let toolkit_changes = toolkit_candidate.is_some();
    let mut steps = Vec::new();
    if driver_changes || toolkit_changes {
        if !current.repository_configured {
            return Err(blocked(
                "the NVIDIA repository is not configured; use `arc install` to configure a supported repository before upgrading",
            ));
        }
        steps.push(PlanStep::new(
            "Refresh package metadata",
            package_manager::refresh_command(os.package_manager()),
        ));
    }
    if driver_changes {
        steps.extend(
            recipe::prerequisites(os, &current.kernel)?
                .into_iter()
                .map(|command| {
                    PlanStep::new(
                        "Ensure matching running-kernel development packages",
                        command,
                    )
                }),
        );
        if legacy
            && !has_r580_restriction(&current.restrictions)
            && current.status.driver.branch().is_none()
        {
            steps.extend(
                legacy_lock_commands(os).into_iter().map(|command| {
                    PlanStep::new("Pin the legacy driver to the R580 branch", command)
                }),
            );
        }
        let package = current.driver_package.as_ref().unwrap();
        steps.push(PlanStep::new(
            format!(
                "Upgrade {} without changing kernel-module flavor",
                package.name
            ),
            driver_upgrade_command(os, &package.name, current.status.driver.branch()),
        ));
        steps.push(PlanStep::new(
            "Verify installed NVIDIA package version",
            installed_version_command(os.package_manager(), &package.name),
        ));
        steps.push(PlanStep::new(
            "Verify NVIDIA kernel module metadata",
            CommandSpec::new("modinfo", ["nvidia"]),
        ));
    }
    if let Some(candidate) = &toolkit_candidate {
        steps.push(PlanStep::new(
            format!(
                "Verify CUDA Toolkit package {} is available",
                candidate.package
            ),
            package_manager::query_command(os.package_manager(), &candidate.package),
        ));
        steps.push(PlanStep::new(
            format!("Install CUDA Toolkit {} side by side", candidate.version),
            package_manager::install_command(os.package_manager(), &candidate.package),
        ));
        let target_dir = format!("/usr/local/cuda-{}", candidate.version);
        if should_update_link(&current.cuda_link, &current.toolkits) {
            steps.push(PlanStep::new(
                "Update the arc-managed /usr/local/cuda symlink",
                CommandSpec::sudo("ln", ["-sfn", &target_dir, "/usr/local/cuda"]),
            ));
        }
        steps.push(PlanStep::new(
            "Verify the target CUDA Toolkit with nvcc",
            CommandSpec::new(&format!("{target_dir}/bin/nvcc"), ["--version"]),
        ));
        steps.push(PlanStep::new(
            "Verify the target CUDA Toolkit directory",
            CommandSpec::new("test", ["-d", &target_dir]),
        ));
        if should_update_link(&current.cuda_link, &current.toolkits) {
            steps.push(PlanStep::new(
                "Verify the managed /usr/local/cuda symlink",
                CommandSpec::new("test", ["-L", "/usr/local/cuda"]),
            ));
        }
    }

    let flavor = current
        .status
        .driver
        .flavor()
        .map_or("not installed".into(), |f| format!("{f:?}"));
    let branch = current
        .status
        .driver
        .branch()
        .map_or("repository/default".into(), |b| format!("R{b}"));
    let driver_current = current
        .driver_package
        .as_ref()
        .map(|package| package.installed.as_str())
        .or(current.status.driver_version.as_deref())
        .unwrap_or("not installed");
    let driver_detail = if !driver_selected {
        "not selected".into()
    } else if current.driver_package.is_none() {
        "absent — skipped".into()
    } else if let Some(target) = &driver_target {
        format!(
            "{driver_current} -> {} ({flavor}, {branch})",
            upstream_driver_version(target).unwrap_or(target)
        )
    } else {
        format!("{driver_current} — already current")
    };
    let toolkit_detail = if !toolkit_selected {
        "not selected".into()
    } else if current.toolkits.is_empty() {
        "absent — skipped".into()
    } else if let Some(target) = &toolkit_candidate {
        format!(
            "{} -> {} (older installations retained)",
            toolkit_versions(&current.toolkits),
            target.version
        )
    } else {
        format!(
            "{} — already current within its compatibility boundary",
            toolkit_versions(&current.toolkits)
        )
    };

    Ok(OperationPlan {
        title: "NVIDIA Upgrade Plan".into(),
        details: vec![
            PlanDetail::new("OS / architecture", format!("{} / {}", os.display_name(), os.architecture)),
            PlanDetail::new("Kernel", &current.kernel),
            PlanDetail::new("Package manager", os.package_manager().to_string()),
            PlanDetail::new("Repository", repository.base_url.clone()),
            PlanDetail::new("Release validation", format!("repository-compatible {}; NVIDIA validated: {}; arc tested: {}", repository.family, if repository.nvidia_validated { "yes" } else { "no" }, if repository.arc_tested { "yes" } else { "no" })),
            PlanDetail::new("GPU policy", if legacy { "Maxwell/Pascal/Volta: proprietary R580, CUDA 12.x maximum" } else { "Turing or newer: preserve installed flavor, latest compatible branch" }),
            PlanDetail::new("Current driver", format!("{}; {flavor}; {branch}; package version {driver_current}; loaded version {}", current.status.driver.description(), current.status.driver_version.as_deref().unwrap_or("not loaded"))),
            PlanDetail::new("Target driver", driver_detail),
            PlanDetail::new("Current CUDA Toolkits", if current.toolkits.is_empty() { "none".into() } else { toolkit_versions(&current.toolkits) }),
            PlanDetail::new("Target CUDA Toolkit", toolkit_detail),
            PlanDetail::new("Package restrictions", if current.restrictions.is_empty() { "none detected".into() } else { current.restrictions.join(" | ") }),
            PlanDetail::new("Kernel development packages", if current.kernel_headers { "matching running kernel detected" } else { "will be installed before driver upgrade" }),
            PlanDetail::new("Secure Boot", current.secure_boot.map_or("unknown", |v| if v { "enabled" } else { "disabled" })),
            PlanDetail::new("/usr/local/cuda", cuda_link_description(&current.cuda_link)),
        ],
        devices: current.status.devices.clone(),
        steps,
        confirmation_warning: "No system changes will be made until you confirm.".into(),
        completion_message: "Upgrade completed and package versions were verified.".into(),
        reboot_message: driver_changes.then(|| "Reboot required: the running NVIDIA module may remain at the previous version until reboot.".into()),
    })
}

fn validate_gpus(os: &OsInfo, gpus: &[NvidiaGpu]) -> Result<()> {
    if gpus.is_empty() {
        return Err(blocked("no NVIDIA GPU was detected"));
    }
    if gpus.iter().any(|gpu| gpu.generation == Generation::Unknown) {
        return Err(blocked(
            "an NVIDIA GPU generation is unknown; refusing to select the newest packages automatically",
        ));
    }
    if gpus
        .iter()
        .any(|gpu| gpu.generation == Generation::MaxwellPascalVolta)
        && os.distribution == Distribution::AzureLinux
    {
        return Err(blocked(
            "Maxwell, Pascal, and Volta GPUs require proprietary R580, but Azure Linux is open-kernel-module-only",
        ));
    }
    Ok(())
}

fn validate_driver_state(os: &OsInfo, current: &UpgradeState, legacy: bool) -> Result<()> {
    match &current.status.driver {
        DriverInstallation::Missing => return Ok(()),
        DriverInstallation::Unmanaged { evidence, .. } => {
            return Err(blocked(format!(
                "unmanaged driver installation detected (evidence: {}); migrate it to a supported package-manager installation before `arc upgrade`",
                evidence
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }
        DriverInstallation::BrokenManaged { .. } => {
            return Err(blocked(
                "the managed driver installation is partial or broken; run `arc doctor` before upgrading",
            ));
        }
        DriverInstallation::Managed {
            flavor: DriverFlavorState::Mixed,
            ..
        } => {
            return Err(blocked(
                "mixed open and proprietary driver packages are installed; use an explicit migration flow",
            ));
        }
        DriverInstallation::Managed { flavor, branch, .. } => {
            if legacy
                && (*flavor != DriverFlavorState::Proprietary
                    || branch.is_some_and(|b| b != policy::LEGACY_DRIVER_BRANCH))
            {
                return Err(blocked(
                    "the installed driver flavor or branch is incompatible with this legacy GPU; use the explicit installation/migration flow to move to proprietary R580",
                ));
            }
            if os.distribution == Distribution::AzureLinux && *flavor != DriverFlavorState::Open {
                return Err(blocked(
                    "Azure Linux supports only the open NVIDIA kernel-module flavor",
                ));
            }
        }
    }
    if current.driver_package.is_none() {
        return Err(blocked(
            "the driver packages could not be mapped to an arc-manageable package",
        ));
    }
    if current
        .restrictions
        .iter()
        .any(|line| line.starts_with("APT hold:") && line.to_ascii_lowercase().contains("nvidia"))
    {
        return Err(blocked(format!(
            "an NVIDIA package hold conflicts with the requested upgrade: {}",
            current.restrictions.join(" | ")
        )));
    }
    if legacy
        && current.restrictions.iter().any(|line| {
            line.to_ascii_lowercase().contains("nvidia")
                && branch_numbers(line)
                    .into_iter()
                    .any(|branch| branch != policy::LEGACY_DRIVER_BRANCH)
        })
    {
        return Err(blocked(format!(
            "an existing package restriction conflicts with required R580: {}",
            current.restrictions.join(" | ")
        )));
    }
    Ok(())
}

fn validate_driver_manageability(current: &UpgradeState) -> Result<()> {
    match &current.status.driver {
        DriverInstallation::Unmanaged { evidence, .. } => Err(blocked(format!(
            "unmanaged driver installation detected (evidence: {}); migrate it to a supported package-manager installation before upgrading CUDA repository packages",
            evidence
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        ))),
        DriverInstallation::BrokenManaged { .. } => Err(blocked(
            "the managed driver installation is partial or broken; run `arc doctor` before upgrading",
        )),
        DriverInstallation::Managed {
            flavor: DriverFlavorState::Mixed,
            ..
        } => Err(blocked(
            "mixed open and proprietary driver packages are installed; use an explicit migration flow",
        )),
        _ => Ok(()),
    }
}

fn resolve_toolkit_candidate(
    manager: PackageManager,
    installed: &[ToolkitInstall],
    legacy: bool,
    query: &impl PackageQuery,
) -> Result<Option<ToolkitCandidate>> {
    let boundary = strongest_boundary(installed);
    let mut releases = compatibility::TABLE
        .iter()
        .filter(|release| !legacy || release.toolkit.starts_with("12."))
        .collect::<Vec<_>>();
    releases.sort_by(|a, b| version_cmp(b.toolkit, a.toolkit));
    for release in releases {
        if matches!(boundary, Some(ToolkitTracking::Major(major)) if !release.toolkit.starts_with(&format!("{major}.")))
        {
            continue;
        }
        let package = format!("cuda-toolkit-{}", release.toolkit.replace('.', "-"));
        if let Some(package_version) = query.candidate_version(manager, &package)? {
            let newest_installed = installed
                .iter()
                .filter_map(|value| value.toolkit_version.clone())
                .max_by(|a, b| version_cmp(a, b));
            if newest_installed.is_none_or(|installed_version| {
                version_cmp(release.toolkit, &installed_version) == Ordering::Greater
            }) {
                let install_package = match boundary {
                    Some(ToolkitTracking::Unversioned) => "cuda-toolkit".into(),
                    Some(ToolkitTracking::Major(major)) => format!("cuda-toolkit-{major}"),
                    _ => package,
                };
                return Ok(Some(ToolkitCandidate {
                    package: install_package,
                    version: release.toolkit.into(),
                    package_version,
                }));
            }
            return Ok(None);
        }
    }
    Ok(None)
}

fn strongest_boundary(installed: &[ToolkitInstall]) -> Option<ToolkitTracking> {
    if installed
        .iter()
        .any(|v| v.tracking == ToolkitTracking::Unversioned)
    {
        return Some(ToolkitTracking::Unversioned);
    }
    installed.iter().find_map(|v| {
        if let ToolkitTracking::Major(m) = v.tracking {
            Some(ToolkitTracking::Major(m))
        } else {
            None
        }
    })
}

fn package_record(
    query: &impl PackageQuery,
    manager: PackageManager,
    name: &str,
) -> Result<PackageRecord> {
    let installed = query
        .installed_version(manager, name)?
        .ok_or_else(|| anyhow!("installed package {name} has no queryable version"))?;
    Ok(PackageRecord {
        name: name.into(),
        installed,
        candidate: query.candidate_version(manager, name)?,
    })
}

fn primary_driver_package(driver: &DriverInstallation) -> Option<&str> {
    let packages = match driver {
        DriverInstallation::Managed { packages, .. }
        | DriverInstallation::BrokenManaged { packages, .. } => packages,
        _ => return None,
    };
    [
        "nvidia-open",
        "cuda-drivers",
        "nvidia-driver",
        "nvidia-driver-cuda",
    ]
    .into_iter()
    .find(|candidate| packages.iter().any(|p| p == candidate))
    .or_else(|| {
        packages
            .iter()
            .find(|p| !p.starts_with("nvidia-driver-pinning-"))
            .map(String::as_str)
    })
}

fn toolkit_tracking(name: &str) -> Option<ToolkitTracking> {
    if name == "cuda-toolkit" {
        return Some(ToolkitTracking::Unversioned);
    }
    let rest = name.strip_prefix("cuda-toolkit-")?;
    let parts = rest
        .split('-')
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()?;
    match parts.as_slice() {
        [major] => Some(ToolkitTracking::Major(*major)),
        [major, minor] => Some(ToolkitTracking::Exact(*major, *minor)),
        _ => None,
    }
}

fn toolkit_version_from_name(name: &str) -> Option<String> {
    match toolkit_tracking(name)? {
        ToolkitTracking::Exact(a, b) => Some(format!("{a}.{b}")),
        ToolkitTracking::Major(a) => Some(format!("{a}.0")),
        ToolkitTracking::Unversioned => None,
    }
}

fn toolkit_versions(toolkits: &[ToolkitInstall]) -> String {
    toolkits
        .iter()
        .map(|v| {
            format!(
                "{} via {} ({:?})",
                v.toolkit_version
                    .clone()
                    .unwrap_or_else(|| v.package.installed.clone()),
                v.package.name,
                v.tracking
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn driver_upgrade_command(os: &OsInfo, package: &str, _branch: Option<u32>) -> CommandSpec {
    match os.package_manager() {
        PackageManager::AptGet => {
            CommandSpec::sudo("apt-get", ["install", "--only-upgrade", "-y", package])
        }
        PackageManager::Dnf => CommandSpec::sudo("dnf", ["upgrade", "-y", package]),
        PackageManager::Tdnf => CommandSpec::sudo("tdnf", ["update", "-y", package]),
        PackageManager::Zypper => CommandSpec::sudo(
            "zypper",
            ["--non-interactive", "update", "--details", package],
        ),
    }
}

fn installed_version_command(manager: PackageManager, package: &str) -> CommandSpec {
    match manager {
        PackageManager::AptGet => CommandSpec::new("dpkg-query", ["-W", "-f=${Version}", package]),
        _ => CommandSpec::new("rpm", ["-q", "--qf", "%{EVR}\\n", package]),
    }
}

fn legacy_lock_commands(os: &OsInfo) -> Vec<CommandSpec> {
    let branch = policy::LEGACY_DRIVER_BRANCH.to_string();
    match os.package_manager() {
        PackageManager::AptGet => vec![CommandSpec::sudo(
            "apt-get",
            ["install", "-y", &format!("nvidia-driver-pinning-{branch}")],
        )],
        PackageManager::Dnf if recipe::is_modular_dnf(os) => vec![CommandSpec::sudo(
            "dnf",
            [
                "module",
                "enable",
                "-y",
                &format!("nvidia-driver:{branch}-dkms"),
            ],
        )],
        PackageManager::Dnf => vec![CommandSpec::sudo(
            "dnf",
            ["versionlock", "add", &format!("*nvidia*{branch}*")],
        )],
        PackageManager::Zypper => vec![CommandSpec::sudo(
            "zypper",
            [
                "addlock",
                &format!("*nvidia* >= {}", policy::LEGACY_DRIVER_BRANCH + 10),
            ],
        )],
        PackageManager::Tdnf => vec![],
    }
}

fn has_r580_restriction(values: &[String]) -> bool {
    values.iter().any(|v| v.contains("580"))
}

fn should_update_link(link: &CudaLink, toolkits: &[ToolkitInstall]) -> bool {
    match link {
        CudaLink::Managed(target) => toolkits
            .iter()
            .filter_map(|v| toolkit_version_from_name(&v.package.name))
            .any(|v| target.contains(&format!("cuda-{v}"))),
        _ => false,
    }
}

fn detect_cuda_link() -> CudaLink {
    let path = Path::new("/usr/local/cuda");
    match fs::symlink_metadata(path) {
        Err(_) => CudaLink::Missing,
        Ok(metadata) if metadata.file_type().is_symlink() => fs::read_link(path)
            .map(|p| CudaLink::Managed(p.display().to_string()))
            .unwrap_or_else(|_| CudaLink::UserManaged("unreadable symlink".into())),
        Ok(_) => CudaLink::UserManaged("non-symlink path (will not be changed)".into()),
    }
}

fn cuda_link_description(link: &CudaLink) -> String {
    match link {
        CudaLink::Missing => "absent; no link will be created".into(),
        CudaLink::Managed(v) => format!("managed symlink -> {v}"),
        CudaLink::UserManaged(v) => format!("user-managed: {v}; unchanged"),
    }
}

fn kernel_release() -> Result<String> {
    let output = Command::new("uname")
        .arg("-r")
        .output()
        .context("could not determine running kernel")?;
    if !output.status.success() {
        bail!("uname -r failed");
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().into())
}

fn upstream_driver_version(version: &str) -> Option<&str> {
    version
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .find(|part| {
            part.matches('.').count() >= 1
                && part
                    .split('.')
                    .next()
                    .and_then(|value| value.parse::<u32>().ok())
                    .is_some_and(|value| value >= 400)
        })
}

fn version_cmp(left: &str, right: &str) -> Ordering {
    let parse = |value: &str| {
        value
            .split(|c: char| !c.is_ascii_digit())
            .filter(|v| !v.is_empty())
            .map(|v| v.parse::<u64>().unwrap_or(0))
            .collect::<Vec<_>>()
    };
    let (left, right) = (parse(left), parse(right));
    for i in 0..left.len().max(right.len()) {
        match left
            .get(i)
            .copied()
            .unwrap_or(0)
            .cmp(&right.get(i).copied().unwrap_or(0))
        {
            Ordering::Equal => {}
            other => return other,
        }
    }
    Ordering::Equal
}

fn parse_candidate(manager: PackageManager, output: &str) -> Option<String> {
    match manager {
        PackageManager::AptGet => output
            .lines()
            .map(str::trim)
            .find_map(|line| line.strip_prefix("Candidate:").map(str::trim))
            .filter(|v| *v != "(none)")
            .map(str::to_owned),
        PackageManager::Dnf => output
            .lines()
            .map(str::trim)
            .rfind(|v| !v.is_empty() && !v.starts_with("Last metadata"))
            .map(str::to_owned),
        PackageManager::Tdnf | PackageManager::Zypper => {
            output.lines().map(str::trim).find_map(|line| {
                line.strip_prefix("Version")
                    .and_then(|v| v.split_once(':').map(|(_, value)| value.trim().to_owned()))
            })
        }
    }
}

fn collect_text_files(path: &Path, kind: &str, output: &mut Vec<String>) -> Result<()> {
    if path.is_file() {
        output.push(format!(
            "{kind}: {}",
            fs::read_to_string(path)
                .unwrap_or_default()
                .lines()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        ));
    } else if path.is_dir() {
        for entry in fs::read_dir(path)? {
            collect_text_files(&entry?.path(), kind, output)?;
        }
    }
    Ok(())
}

fn branch_numbers(value: &str) -> Vec<u32> {
    value
        .split(|character: char| !character.is_ascii_digit())
        .filter(|part| part.len() == 3)
        .filter_map(|part| part.parse().ok())
        .filter(|branch| (400..700).contains(branch))
        .collect()
}

trait DriverBranch {
    fn branch(&self) -> Option<u32>;
}
impl DriverBranch for DriverInstallation {
    fn branch(&self) -> Option<u32> {
        match self {
            DriverInstallation::Managed { branch, .. } => *branch,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{device::GpuVendor, environment::DriverPackageScope};
    use std::collections::HashMap;

    #[derive(Default)]
    struct MockQuery {
        candidates: HashMap<String, String>,
    }
    impl PackageQuery for MockQuery {
        fn installed_version(&self, _: PackageManager, _: &str) -> Result<Option<String>> {
            Ok(None)
        }
        fn candidate_version(&self, _: PackageManager, package: &str) -> Result<Option<String>> {
            Ok(self.candidates.get(package).cloned())
        }
        fn restrictions(&self, _: PackageManager) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }
    fn os(distribution: Distribution) -> OsInfo {
        OsInfo {
            distribution,
            name: "Test".into(),
            version_id: if distribution == Distribution::AzureLinux {
                "3.0"
            } else {
                "24.04"
            }
            .into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        }
    }
    fn gpu(generation: Generation) -> NvidiaGpu {
        NvidiaGpu {
            name: "GPU".into(),
            pci_device_id: None,
            generation,
        }
    }
    fn state(
        flavor: DriverFlavorState,
        driver: bool,
        toolkits: Vec<ToolkitInstall>,
    ) -> UpgradeState {
        let installation = if driver {
            DriverInstallation::Managed {
                flavor,
                scope: DriverPackageScope::Full,
                branch: None,
                packages: vec![
                    if flavor == DriverFlavorState::Open {
                        "nvidia-open"
                    } else {
                        "cuda-drivers"
                    }
                    .into(),
                ],
            }
        } else {
            DriverInstallation::Missing
        };
        UpgradeState {
            status: ProviderStatus {
                vendor: GpuVendor::Nvidia,
                devices: vec![gpu(Generation::TuringOrNewer).into()],
                driver: installation,
                driver_version: driver.then(|| "580.65.06".into()),
                driver_runtime_operational: driver,
                driver_module: None,
                kernel_version: None,
                secure_boot_enabled: None,
                toolkits: vec![],
                active_toolkit: None,
            },
            driver_package: driver.then(|| PackageRecord {
                name: if flavor == DriverFlavorState::Open {
                    "nvidia-open"
                } else {
                    "cuda-drivers"
                }
                .into(),
                installed: "580.65.06".into(),
                candidate: Some("590.44.01".into()),
            }),
            toolkits,
            restrictions: vec![],
            repository_configured: true,
            kernel: "6.8.0-test".into(),
            kernel_headers: true,
            secure_boot: Some(false),
            cuda_link: CudaLink::Missing,
        }
    }
    fn tk(name: &str) -> ToolkitInstall {
        ToolkitInstall {
            package: PackageRecord {
                name: name.into(),
                installed: "1".into(),
                candidate: Some("2".into()),
            },
            tracking: toolkit_tracking(name).unwrap(),
            toolkit_version: toolkit_version_from_name(name),
        }
    }

    #[test]
    fn flags_default_to_both_and_remain_independent() {
        assert_eq!(
            UpgradeOptions::from_component_flags(false, false),
            UpgradeOptions {
                driver: true,
                toolkit: true
            }
        );
        assert_eq!(
            UpgradeOptions::from_component_flags(true, false),
            UpgradeOptions {
                driver: true,
                toolkit: false
            }
        );
        assert_eq!(
            UpgradeOptions::from_component_flags(false, true),
            UpgradeOptions {
                driver: false,
                toolkit: true
            }
        );
        assert_eq!(
            UpgradeOptions::from_component_flags(true, true),
            UpgradeOptions {
                driver: true,
                toolkit: true
            }
        );
    }

    #[test]
    fn upgrades_both_and_preserves_open_or_proprietary_flavor() {
        let query = MockQuery {
            candidates: [("cuda-toolkit-13-1".into(), "13.1.1".into())].into(),
        };
        for flavor in [DriverFlavorState::Open, DriverFlavorState::Proprietary] {
            let plan = build_plan(
                &os(Distribution::Ubuntu),
                &UpgradeOptions {
                    driver: true,
                    toolkit: true,
                },
                &[gpu(Generation::TuringOrNewer)],
                &state(flavor, true, vec![tk("cuda-toolkit-12-8")]),
                &query,
            )
            .unwrap();
            let rendered = plan
                .steps
                .iter()
                .map(|s| s.command.display())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(rendered.contains(if flavor == DriverFlavorState::Open {
                "nvidia-open"
            } else {
                "cuda-drivers"
            }));
            assert!(rendered.contains("cuda-toolkit-13-1"));
        }
    }

    #[test]
    fn missing_components_are_successful_noops() {
        let query = MockQuery::default();
        let plan = build_plan(
            &os(Distribution::Ubuntu),
            &UpgradeOptions {
                driver: true,
                toolkit: true,
            },
            &[gpu(Generation::TuringOrNewer)],
            &state(DriverFlavorState::Open, false, vec![]),
            &query,
        )
        .unwrap();
        assert!(plan.is_noop());
        assert!(
            plan.details
                .iter()
                .any(|d| d.value.contains("absent — skipped"))
        );
    }

    #[test]
    fn active_nvcc_without_inventory_is_not_upgraded() {
        let query = MockQuery::default();
        let mut current = state(DriverFlavorState::Open, true, vec![]);
        current.status.active_toolkit = Some(crate::model::environment::ToolkitStatus {
            name: "Active nvcc".into(),
            version: Some("12.8".into()),
            executable_path: Some("/custom/cuda/bin/nvcc".into()),
            source: crate::model::environment::ToolkitSource::ActivePath,
            packages: vec![],
            manageable: false,
        });
        let plan = build_plan(
            &os(Distribution::Ubuntu),
            &UpgradeOptions {
                driver: false,
                toolkit: true,
            },
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &query,
        )
        .unwrap();
        assert!(plan.is_noop());
        assert!(
            plan.details
                .iter()
                .any(|detail| detail.label == "Target CUDA Toolkit"
                    && detail.value.contains("absent"))
        );
    }

    #[test]
    fn driver_only_and_toolkit_only_do_not_touch_the_other_component() {
        let query = MockQuery {
            candidates: [("cuda-toolkit-13-1".into(), "13.1.1".into())].into(),
        };
        let current = state(DriverFlavorState::Open, true, vec![tk("cuda-toolkit-12-8")]);
        let driver = build_plan(
            &os(Distribution::Ubuntu),
            &UpgradeOptions {
                driver: true,
                toolkit: false,
            },
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &query,
        )
        .unwrap();
        assert!(
            !driver
                .steps
                .iter()
                .any(|s| s.command.display().contains("cuda-toolkit"))
        );
        let toolkit = build_plan(
            &os(Distribution::Ubuntu),
            &UpgradeOptions {
                driver: false,
                toolkit: true,
            },
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &query,
        )
        .unwrap();
        assert!(
            !toolkit
                .steps
                .iter()
                .any(|s| s.command.display().contains("nvidia-open"))
        );
    }

    #[test]
    fn legacy_is_limited_to_cuda_12_and_azure_or_unknown_are_rejected() {
        let query = MockQuery {
            candidates: [
                ("cuda-toolkit-13-3".into(), "13".into()),
                ("cuda-toolkit-12-9".into(), "12".into()),
            ]
            .into(),
        };
        let mut current = state(
            DriverFlavorState::Proprietary,
            true,
            vec![tk("cuda-toolkit-12-8")],
        );
        if let DriverInstallation::Managed { branch, .. } = &mut current.status.driver {
            *branch = Some(580);
        }
        let plan = build_plan(
            &os(Distribution::Ubuntu),
            &UpgradeOptions {
                driver: false,
                toolkit: true,
            },
            &[gpu(Generation::MaxwellPascalVolta)],
            &current,
            &query,
        )
        .unwrap();
        assert!(
            plan.steps
                .iter()
                .any(|s| s.command.display().contains("cuda-toolkit-12-9"))
        );
        assert!(
            build_plan(
                &os(Distribution::AzureLinux),
                &UpgradeOptions {
                    driver: true,
                    toolkit: true
                },
                &[gpu(Generation::MaxwellPascalVolta)],
                &current,
                &query
            )
            .is_err()
        );
        assert!(
            build_plan(
                &os(Distribution::Ubuntu),
                &UpgradeOptions {
                    driver: true,
                    toolkit: true
                },
                &[gpu(Generation::Unknown)],
                &current,
                &query
            )
            .is_err()
        );
    }

    #[test]
    fn refuses_unmanaged_broken_and_hidden_flavor_migration() {
        let query = MockQuery::default();
        for installation in [
            DriverInstallation::Unmanaged {
                working: true,
                evidence: vec![
                    crate::model::environment::UnmanagedDriverEvidence::RunfileUninstaller,
                ],
            },
            DriverInstallation::BrokenManaged {
                flavor: DriverFlavorState::Open,
                packages: vec!["nvidia-open".into()],
            },
        ] {
            let mut current = state(DriverFlavorState::Open, false, vec![]);
            current.status.driver = installation;
            assert!(
                build_plan(
                    &os(Distribution::Ubuntu),
                    &UpgradeOptions {
                        driver: true,
                        toolkit: false
                    },
                    &[gpu(Generation::TuringOrNewer)],
                    &current,
                    &query
                )
                .is_err()
            );
        }
    }

    #[test]
    fn exact_and_unversioned_toolkits_upgrade_side_by_side_and_managed_link_only() {
        let query = MockQuery {
            candidates: [("cuda-toolkit-13-1".into(), "13.1".into())].into(),
        };
        let mut current = state(DriverFlavorState::Open, true, vec![tk("cuda-toolkit-12-8")]);
        current.cuda_link = CudaLink::Managed("/usr/local/cuda-12.8".into());
        let plan = build_plan(
            &os(Distribution::Ubuntu),
            &UpgradeOptions {
                driver: false,
                toolkit: true,
            },
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &query,
        )
        .unwrap();
        let commands = plan
            .steps
            .iter()
            .map(|s| s.command.display())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(commands.contains("ln -sfn /usr/local/cuda-13.1 /usr/local/cuda"));
        assert!(!commands.contains("remove"));
        current.cuda_link = CudaLink::UserManaged("directory".into());
        let plan = build_plan(
            &os(Distribution::Ubuntu),
            &UpgradeOptions {
                driver: false,
                toolkit: true,
            },
            &[gpu(Generation::TuringOrNewer)],
            &current,
            &query,
        )
        .unwrap();
        assert!(
            !plan
                .steps
                .iter()
                .any(|s| s.command.program == "sudo" && s.command.args.contains(&"ln".into()))
        );
    }

    #[test]
    fn parses_candidates_and_version_order() {
        assert_eq!(
            parse_candidate(
                PackageManager::AptGet,
                "Installed: 1\n Candidate: 580.135.02\n"
            ),
            Some("580.135.02".into())
        );
        assert_eq!(version_cmp("580.135.02", "580.99.9"), Ordering::Greater);
    }

    #[test]
    fn driver_updates_are_always_scoped_to_the_detected_package() {
        let cases = [
            (
                OsInfo {
                    version_id: "24.04".into(),
                    ..os(Distribution::Ubuntu)
                },
                "sudo apt-get install --only-upgrade -y nvidia-open",
            ),
            (
                OsInfo {
                    version_id: "9.7".into(),
                    ..os(Distribution::Rhel)
                },
                "sudo dnf upgrade -y nvidia-open",
            ),
            (
                OsInfo {
                    version_id: "44".into(),
                    ..os(Distribution::Fedora)
                },
                "sudo dnf upgrade -y nvidia-open",
            ),
            (
                os(Distribution::AzureLinux),
                "sudo tdnf update -y nvidia-open",
            ),
            (
                OsInfo {
                    version_id: "15.6".into(),
                    ..os(Distribution::OpenSuse)
                },
                "sudo zypper --non-interactive update --details nvidia-open",
            ),
        ];
        for (system, expected) in cases {
            let command = driver_upgrade_command(&system, "nvidia-open", None).display();
            assert_eq!(command, expected);
            assert!(command.contains("nvidia-open"));
            for forbidden in [
                "apt-get dist-upgrade",
                "dnf update -y\n",
                "dnf update -y$",
                "tdnf update -y\n",
                "zypper --non-interactive update --details$",
            ] {
                assert!(!command.contains(forbidden), "unscoped command: {command}");
            }
        }
    }

    #[test]
    fn toolkit_meta_packages_keep_their_tracking_boundary() {
        let query = MockQuery {
            candidates: [
                ("cuda-toolkit-13-1".into(), "13.1".into()),
                ("cuda-toolkit-12-9".into(), "12.9".into()),
            ]
            .into(),
        };
        let mut unversioned = tk("cuda-toolkit");
        unversioned.toolkit_version = Some("12.8".into());
        assert_eq!(
            resolve_toolkit_candidate(PackageManager::AptGet, &[unversioned], false, &query)
                .unwrap()
                .unwrap()
                .package,
            "cuda-toolkit"
        );
        let mut major = tk("cuda-toolkit-12");
        major.toolkit_version = Some("12.8".into());
        let candidate = resolve_toolkit_candidate(PackageManager::AptGet, &[major], false, &query)
            .unwrap()
            .unwrap();
        assert_eq!(
            (candidate.package.as_str(), candidate.version.as_str()),
            ("cuda-toolkit-12", "12.9")
        );
    }

    #[test]
    fn legacy_adds_r580_lock_and_rejects_conflicting_lock() {
        let query = MockQuery {
            candidates: [("cuda-drivers-580".into(), "580.135.02".into())].into(),
        };
        let current = state(DriverFlavorState::Proprietary, true, vec![]);
        let plan = build_plan(
            &os(Distribution::Ubuntu),
            &UpgradeOptions {
                driver: true,
                toolkit: false,
            },
            &[gpu(Generation::MaxwellPascalVolta)],
            &current,
            &query,
        )
        .unwrap();
        assert!(
            plan.steps
                .iter()
                .any(|step| step.command.display().contains("nvidia-driver-pinning-580"))
        );
        let mut conflicting = current;
        conflicting.restrictions =
            vec!["APT preference: Package: nvidia-* Pin: version 570.*".into()];
        assert!(
            build_plan(
                &os(Distribution::Ubuntu),
                &UpgradeOptions {
                    driver: true,
                    toolkit: false
                },
                &[gpu(Generation::MaxwellPascalVolta)],
                &conflicting,
                &query,
            )
            .is_err()
        );
    }
}
