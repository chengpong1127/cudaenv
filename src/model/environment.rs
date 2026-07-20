use std::{collections::HashSet, fmt};

use crate::model::{
    command::CommandSpec,
    device::{GpuDevice, GpuVendor},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolkitStatus {
    pub name: String,
    pub version: Option<String>,
    pub executable_path: Option<String>,
    pub source: ToolkitSource,
    pub packages: Vec<String>,
    pub manageable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolkitSource {
    SystemPackageManager,
    ActivePath,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderStatus {
    pub vendor: GpuVendor,
    pub devices: Vec<GpuDevice>,
    pub driver: DriverInstallation,
    pub driver_version: Option<String>,
    pub driver_runtime_operational: bool,
    pub driver_runtime_state: DriverRuntimeState,
    pub dkms_status: Option<String>,
    pub driver_module: Option<DriverModuleInfo>,
    pub kernel_version: Option<String>,
    pub secure_boot_enabled: Option<bool>,
    /// Toolkit installations proven by the system package-manager inventory.
    pub toolkits: Vec<ToolkitStatus>,
    /// The nvcc selected by PATH, which may be Conda-, module-, or user-managed.
    pub active_toolkit: Option<ToolkitStatus>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DriverModuleInfo {
    pub path: Option<String>,
    pub version: Option<String>,
    pub signer: Option<String>,
    pub signature_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverRuntimeState {
    Operational,
    RebootLikelyRequired,
    DkmsModuleMissing,
    SecureBootBlocked,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverFlavorState {
    Open,
    Proprietary,
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverPackageScope {
    Full,
    ComputeOnly,
    DesktopOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnmanagedDriverEvidence {
    RunfileUninstaller,
    DriverVersion,
    LoadedModule,
    ModuleMetadata,
    InstallerLog,
}

impl fmt::Display for UnmanagedDriverEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RunfileUninstaller => "`/usr/bin/nvidia-uninstall` exists",
            Self::DriverVersion => "an NVIDIA driver version was detected",
            Self::LoadedModule => "the NVIDIA kernel module is loaded",
            Self::ModuleMetadata => "NVIDIA kernel module metadata is present",
            Self::InstallerLog => {
                "`/var/log/nvidia-installer.log` exists (supporting evidence only)"
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DriverInstallation {
    Missing,
    Managed {
        flavor: DriverFlavorState,
        scope: DriverPackageScope,
        branch: Option<u32>,
        packages: Vec<String>,
    },
    BrokenManaged {
        flavor: DriverFlavorState,
        packages: Vec<String>,
    },
    Unmanaged {
        working: bool,
        evidence: Vec<UnmanagedDriverEvidence>,
    },
}

impl DriverInstallation {
    pub fn flavor(&self) -> Option<DriverFlavorState> {
        match self {
            Self::Managed { flavor, .. } | Self::BrokenManaged { flavor, .. } => Some(*flavor),
            Self::Missing | Self::Unmanaged { .. } => None,
        }
    }

    pub fn description(&self) -> String {
        match self {
            Self::Missing => "not installed".into(),
            Self::Managed {
                flavor,
                scope,
                branch,
                ..
            } => format!(
                "managed {scope:?} {flavor:?} installation{}",
                branch
                    .map(|b| format!(" pinned to R{b}"))
                    .unwrap_or_default()
            ),
            Self::BrokenManaged { flavor, .. } => {
                format!("broken managed {flavor:?} installation")
            }
            Self::Unmanaged { working, evidence } => format!(
                "{} unmanaged installation (evidence: {})",
                if *working { "working" } else { "broken" },
                evidence
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticStatus {
    Pass,
    Warning,
    Error,
    Skipped,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum DiagnosticId {
    NvidiaGpu,
    OperatingSystem,
    KernelHeaders,
    SecureBoot,
    DriverPackage,
    DriverModule,
    NvidiaSmi,
    DriverLibrary,
    ToolkitInstall,
    Nvcc,
    CudaSymlink,
    DriverToolkitCompatibility,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSection {
    Hardware,
    OperatingSystem,
    Driver,
    CudaToolkit,
}

impl fmt::Display for DiagnosticSection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Hardware => "Hardware",
            Self::OperatingSystem => "Operating System",
            Self::Driver => "Driver",
            Self::CudaToolkit => "CUDA Toolkit",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticCheck {
    pub id: DiagnosticId,
    pub section: DiagnosticSection,
    pub name: String,
    pub status: DiagnosticStatus,
    pub evidence: Vec<String>,
    pub problem: Option<String>,
    pub dependencies: Vec<DiagnosticId>,
    pub recommended_fixes: Vec<FixId>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    Low,
    Medium,
    High,
}

impl fmt::Display for Confidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticCause {
    pub title: String,
    pub confidence: Confidence,
    pub evidence: Vec<String>,
    pub fixes: Vec<FixId>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum FixId {
    InspectHardware,
    InstallKernelHeaders,
    InstallDriver,
    RepairManagedDriver,
    RebuildDkms,
    ReinstallDriverLibraries,
    InstallToolkit,
    RepairCudaSymlink,
    UpgradeDriver,
    Reboot,
    RebootThenRecheck,
    ResolveSecureBoot,
    DebugDriver,
    DebugToolkit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fix {
    pub id: FixId,
    pub title: String,
    pub commands: Vec<CommandSpec>,
    pub manual_steps: Vec<String>,
    pub order: u16,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FixPlan {
    pub causes: Vec<DiagnosticCause>,
    pub fixes: Vec<Fix>,
}

impl FixPlan {
    pub fn new(causes: Vec<DiagnosticCause>, mut fixes: Vec<Fix>) -> Self {
        let requested = causes
            .iter()
            .flat_map(|cause| cause.fixes.iter().copied())
            .collect::<HashSet<_>>();
        let mut seen_fixes = HashSet::new();
        let mut seen_commands = HashSet::new();
        fixes.retain_mut(|fix| {
            if !requested.contains(&fix.id) || !seen_fixes.insert(fix.id) {
                return false;
            }
            fix.commands
                .retain(|command| seen_commands.insert(command.display()));
            true
        });
        fixes.sort_by_key(|fix| fix.order);
        Self { causes, fixes }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostics {
    pub vendor: GpuVendor,
    pub checks: Vec<DiagnosticCheck>,
    pub fix_plan: FixPlan,
}

impl Diagnostics {
    pub fn healthy(&self) -> bool {
        !self.has_errors()
    }

    pub fn has_errors(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == DiagnosticStatus::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_plan_orders_and_deduplicates_fixes_and_commands() {
        let command = CommandSpec::new("same", ["command"]);
        let causes = vec![DiagnosticCause {
            title: "cause".into(),
            confidence: Confidence::High,
            evidence: vec![],
            fixes: vec![FixId::Reboot, FixId::InstallKernelHeaders, FixId::Reboot],
        }];
        let fixes = vec![
            Fix {
                id: FixId::Reboot,
                title: "reboot".into(),
                commands: vec![command.clone()],
                manual_steps: vec![],
                order: 90,
            },
            Fix {
                id: FixId::InstallKernelHeaders,
                title: "headers".into(),
                commands: vec![command],
                manual_steps: vec![],
                order: 10,
            },
            Fix {
                id: FixId::Reboot,
                title: "duplicate".into(),
                commands: vec![],
                manual_steps: vec![],
                order: 90,
            },
        ];
        let plan = FixPlan::new(causes, fixes);
        assert_eq!(
            plan.fixes.iter().map(|fix| fix.id).collect::<Vec<_>>(),
            vec![FixId::InstallKernelHeaders, FixId::Reboot]
        );
        assert_eq!(
            plan.fixes
                .iter()
                .map(|fix| fix.commands.len())
                .sum::<usize>(),
            1
        );
    }
}
