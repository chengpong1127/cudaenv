use std::fmt::Write;

use console::style;

use crate::model::{
    environment::{
        DiagnosticSection, DiagnosticStatus, Diagnostics, DriverFlavorState, DriverInstallation,
        DriverPackageScope, DriverRuntimeState, ProviderStatus,
    },
    operation::OperationPlan,
    system::OsInfo,
};
use crate::platform::command::ExecutionEvent;
use crate::providers::nvidia::runtime;

pub fn operation_plan(plan: &OperationPlan) {
    print!("{}", format_operation_plan(plan));
}

pub fn format_operation_plan(plan: &OperationPlan) -> String {
    let mut rendered = String::new();
    let label_width = plan
        .details
        .iter()
        .map(|detail| detail.label.chars().count())
        .chain((!plan.devices.is_empty()).then_some("GPU(s)".len()))
        .max()
        .unwrap_or(0);

    writeln!(rendered).unwrap();
    writeln!(
        rendered,
        "  {}  {}",
        style("arc").cyan().bold(),
        style(&plan.title).bold()
    )
    .unwrap();
    writeln!(rendered, "  {}", style("─".repeat(52)).dim()).unwrap();

    if !plan.details.is_empty() || !plan.devices.is_empty() {
        writeln!(rendered, "\n  {}", section_label("Environment")).unwrap();
    }
    for detail in &plan.details {
        let padded_label = format!("{:<label_width$}", detail.label);
        writeln!(
            rendered,
            "  {}  {}",
            style(padded_label).dim(),
            detail.value
        )
        .unwrap();
    }
    if !plan.devices.is_empty() {
        let padded_label = format!("{:<label_width$}", "GPU(s)");
        let empty_label = " ".repeat(label_width);
        for (index, device) in plan.devices.iter().enumerate() {
            let label = if index == 0 {
                padded_label.as_str()
            } else {
                empty_label.as_str()
            };
            writeln!(
                rendered,
                "  {}  {} {}",
                style(label).dim(),
                style("◆").cyan(),
                format_args!("{} ({})", device.name, device.vendor)
            )
            .unwrap();
        }
    }

    let step_count = match plan.steps.len() {
        1 => "1 step".to_owned(),
        count => format!("{count} steps"),
    };
    writeln!(
        rendered,
        "\n  {}  {}",
        section_label("Changes"),
        style(step_count).dim()
    )
    .unwrap();
    if plan.steps.is_empty() {
        writeln!(
            rendered,
            "  {}  {}",
            style("✓").green().bold(),
            style("No changes required").green()
        )
        .unwrap();
    } else {
        for (index, step) in plan.steps.iter().enumerate() {
            writeln!(
                rendered,
                "  {}  {}",
                style(format!("{:02}", index + 1)).cyan().bold(),
                style(&step.description).bold()
            )
            .unwrap();
            writeln!(
                rendered,
                "      {} {}",
                style("$").dim(),
                style(step.command.display()).dim()
            )
            .unwrap();
        }
    }
    if !plan.confirmation_warning.is_empty() {
        writeln!(
            rendered,
            "\n  {}  {}\n",
            style("!").yellow().bold(),
            style(&plan.confirmation_warning).yellow()
        )
        .unwrap();
    } else {
        writeln!(rendered).unwrap();
    }

    rendered
}

pub fn operation_completed(plan: &OperationPlan) {
    println!(
        "\n  {}  {}",
        style("✓").green().bold(),
        style(&plan.completion_message).green().bold()
    );
    if let Some(message) = &plan.reboot_message {
        println!(
            "  {}  {}",
            style("↻").yellow().bold(),
            style(message).yellow()
        );
    }
    println!();
}

pub fn execution_event(event: ExecutionEvent<'_>) {
    match event {
        ExecutionEvent::Started { index, total, step } => {
            if index == 0 {
                println!("\n  {}\n", section_label("Applying changes"));
            }
            println!(
                "  {}  {}  {}",
                style("◆").cyan(),
                style(format!("{}/{}", index + 1, total)).cyan().bold(),
                style(&step.description).bold()
            );
        }
        ExecutionEvent::Completed { index, total, step } => println!(
            "  {}  {}  {}\n",
            style("✓").green().bold(),
            style(format!("{}/{}", index + 1, total)).dim(),
            style(format!("{} complete", step.description)).green()
        ),
        ExecutionEvent::Failed { index, total, step } => println!(
            "  {}  {}  {}\n",
            style("✗").red().bold(),
            style(format!("{}/{}", index + 1, total)).dim(),
            style(format!("{} failed", step.description)).red().bold()
        ),
    }
}

pub fn notice(message: &str) {
    println!("\n  {}  {}\n", style("•").cyan().bold(), message);
}

pub fn cancelled(action: &str) {
    println!(
        "\n  {}  {} cancelled. No changes were made.\n",
        style("○").yellow(),
        action
    );
}

fn section_label(label: &str) -> String {
    style(label.to_uppercase()).cyan().bold().to_string()
}

pub fn system_status(os: &OsInfo, providers: &[ProviderStatus], verbose: bool) {
    print!("{}", format_system_status(os, providers, verbose));
}

pub fn format_system_status(os: &OsInfo, providers: &[ProviderStatus], verbose: bool) -> String {
    let mut rendered = String::from("GPU Environment\n\n");
    status_row(&mut rendered, "OS", &os.display_name());
    for status in providers {
        status_row(&mut rendered, "GPU", &grouped_devices(status));
        status_row(
            &mut rendered,
            "Driver installation",
            &driver_installation_label(&status.driver),
        );
        status_row(
            &mut rendered,
            "Driver version",
            status
                .driver_version
                .as_deref()
                .or_else(|| status.driver_module.as_ref()?.version.as_deref())
                .unwrap_or("Not detected"),
        );
        status_row(
            &mut rendered,
            "Driver runtime",
            match status.driver_runtime_state {
                DriverRuntimeState::Operational => "Operational",
                DriverRuntimeState::RebootLikelyRequired => "Reboot likely required",
                DriverRuntimeState::DkmsModuleMissing
                | DriverRuntimeState::SecureBootBlocked
                | DriverRuntimeState::Failed => "Not operational",
            },
        );
        status_row(
            &mut rendered,
            "CUDA Toolkit",
            status
                .toolkits
                .first()
                .and_then(|toolkit| toolkit.version.as_deref())
                .or_else(|| (!status.toolkits.is_empty()).then_some("Installed"))
                .unwrap_or("Not installed"),
        );
        status_row(
            &mut rendered,
            "nvcc",
            status
                .active_toolkit
                .as_ref()
                .and_then(|toolkit| toolkit.version.as_deref())
                .unwrap_or("Not found"),
        );

        if verbose {
            writeln!(rendered, "\nTechnical details").unwrap();
            technical_row(
                &mut rendered,
                "Kernel version",
                status.kernel_version.as_deref().unwrap_or("Unknown"),
            );
            technical_row(
                &mut rendered,
                "Module path",
                status
                    .driver_module
                    .as_ref()
                    .and_then(|module| module.path.as_deref())
                    .unwrap_or("Not found"),
            );
            technical_row(
                &mut rendered,
                "Kernel module version",
                status
                    .driver_module
                    .as_ref()
                    .and_then(|module| module.version.as_deref())
                    .unwrap_or("Not detected"),
            );
            technical_row(
                &mut rendered,
                "Driver package scope",
                driver_package_scope(&status.driver),
            );
            technical_row(
                &mut rendered,
                "Secure Boot",
                status.secure_boot_enabled.map_or("Unknown", |enabled| {
                    if enabled { "Enabled" } else { "Disabled" }
                }),
            );
            technical_row(
                &mut rendered,
                "Secure Boot signature",
                &module_signature(status),
            );
        }

        if !matches!(status.driver, DriverInstallation::Missing)
            && status.driver_runtime_state != DriverRuntimeState::Operational
        {
            writeln!(
                rendered,
                "\n⚠ {}",
                runtime::status_warning(status.driver_runtime_state)
            )
            .unwrap();
        }
    }
    rendered
}

fn status_row(rendered: &mut String, label: &str, value: &str) {
    writeln!(rendered, "{label:<20}{value}").unwrap();
}

fn technical_row(rendered: &mut String, label: &str, value: &str) {
    writeln!(rendered, "{label:<24}{value}").unwrap();
}

fn grouped_devices(status: &ProviderStatus) -> String {
    if status.devices.is_empty() {
        return "Not detected".into();
    }
    let mut groups: Vec<(String, usize)> = Vec::new();
    for device in &status.devices {
        let name = friendly_gpu_name(&device.name);
        if let Some((_, count)) = groups.iter_mut().find(|(existing, _)| *existing == name) {
            *count += 1;
        } else {
            groups.push((name, 1));
        }
    }
    groups
        .into_iter()
        .map(|(name, count)| {
            if count == 1 {
                name
            } else {
                format!("{count} × {name}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn friendly_gpu_name(name: &str) -> String {
    let model = name
        .split('[')
        .skip(1)
        .filter_map(|suffix| suffix.split_once(']').map(|(value, _)| value))
        .filter(|value| !value.contains(':'))
        .last()
        .unwrap_or(name)
        .trim();
    if model.starts_with("NVIDIA ") {
        model.to_owned()
    } else {
        format!("NVIDIA {model}")
    }
}

fn driver_installation_label(driver: &DriverInstallation) -> String {
    let flavor = |value| match value {
        DriverFlavorState::Open => "Open",
        DriverFlavorState::Proprietary => "Proprietary",
        DriverFlavorState::Mixed => "Mixed",
    };
    match driver {
        DriverInstallation::Missing => "Not installed".into(),
        DriverInstallation::Managed { flavor: value, .. } => {
            format!("Managed · {}", flavor(*value))
        }
        DriverInstallation::BrokenManaged { flavor: value, .. } => {
            format!("Managed · {} · Needs repair", flavor(*value))
        }
        DriverInstallation::Unmanaged { .. } => "Unmanaged".into(),
    }
}

fn driver_package_scope(driver: &DriverInstallation) -> &'static str {
    match driver {
        DriverInstallation::Managed { scope, .. } => match scope {
            DriverPackageScope::Full => "Full",
            DriverPackageScope::ComputeOnly => "Compute only",
            DriverPackageScope::DesktopOnly => "Desktop only",
        },
        DriverInstallation::BrokenManaged { .. } => "Incomplete",
        DriverInstallation::Unmanaged { .. } => "Unmanaged",
        DriverInstallation::Missing => "Not installed",
    }
}

fn module_signature(status: &ProviderStatus) -> String {
    let Some(module) = &status.driver_module else {
        return "Unknown".into();
    };
    match (&module.signature_id, &module.signer) {
        (Some(id), Some(signer)) => format!("Signed · {id} · {signer}"),
        (Some(id), None) => format!("Signed · {id}"),
        (None, Some(signer)) => format!("Signed · {signer}"),
        (None, None) => "Unsigned or unknown".into(),
    }
}

pub fn diagnostics(diagnostics: &Diagnostics) {
    print!("{}", format_diagnostics(diagnostics));
}

pub fn format_diagnostics(diagnostics: &Diagnostics) -> String {
    let mut rendered = String::new();
    writeln!(rendered, "{} Diagnostics", diagnostics.vendor).unwrap();
    for section in [
        DiagnosticSection::Hardware,
        DiagnosticSection::OperatingSystem,
        DiagnosticSection::Driver,
        DiagnosticSection::CudaToolkit,
    ] {
        writeln!(rendered, "\n{section}").unwrap();
        for check in diagnostics
            .checks
            .iter()
            .filter(|check| check.section == section)
        {
            writeln!(rendered, "{} {}", mark(check.status), check.name).unwrap();
            if check.status != DiagnosticStatus::Pass {
                if let Some(problem) = &check.problem {
                    writeln!(rendered, "  {problem}").unwrap();
                }
                for evidence in &check.evidence {
                    writeln!(rendered, "  Evidence: {evidence}").unwrap();
                }
            }
        }
    }
    if diagnostics.healthy() && diagnostics.fix_plan.causes.is_empty() {
        let has_warnings = diagnostics
            .checks
            .iter()
            .any(|check| check.status == DiagnosticStatus::Warning);
        writeln!(
            rendered,
            "\n{}",
            if has_warnings {
                "Completed with warnings"
            } else {
                "Healthy"
            }
        )
        .unwrap();
        return rendered;
    }
    writeln!(rendered, "\nActionable fix plan").unwrap();
    for (index, cause) in diagnostics.fix_plan.causes.iter().enumerate() {
        writeln!(
            rendered,
            "{}. Likely root cause: {} ({} confidence)",
            index + 1,
            cause.title,
            cause.confidence
        )
        .unwrap();
        for evidence in &cause.evidence {
            writeln!(rendered, "   Evidence: {evidence}").unwrap();
        }
    }
    for (index, fix) in diagnostics.fix_plan.fixes.iter().enumerate() {
        writeln!(rendered, "\n{}. {}", index + 1, fix.title).unwrap();
        for command in &fix.commands {
            writeln!(rendered, "   $ {}", command.display()).unwrap();
        }
        for step in &fix.manual_steps {
            writeln!(rendered, "   - {step}").unwrap();
        }
    }
    if diagnostics
        .fix_plan
        .fixes
        .iter()
        .any(|fix| !fix.manual_steps.is_empty())
    {
        writeln!(rendered, "\nComplete the recommended manual actions above.").unwrap();
    } else {
        writeln!(
            rendered,
            "\nNo fixes were executed. After completing the plan, rerun `arc doctor`."
        )
        .unwrap();
    }
    rendered
}

fn mark(status: DiagnosticStatus) -> &'static str {
    match status {
        DiagnosticStatus::Pass => "✓",
        DiagnosticStatus::Warning => "⚠",
        DiagnosticStatus::Error => "✗",
        DiagnosticStatus::Skipped => "↷ skipped",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        command::CommandSpec,
        device::{GpuDevice, GpuVendor},
        environment::{
            DiagnosticCheck, DiagnosticId, DiagnosticSection, DiagnosticStatus, DriverFlavorState,
            DriverInstallation, DriverModuleInfo, DriverPackageScope, FixPlan,
        },
        operation::{OperationPlan, PlanDetail, PlanStep},
        system::{Distribution, OsInfo},
    };

    #[test]
    fn operation_plan_output_has_a_scannable_hierarchy() {
        let plan = OperationPlan {
            title: "NVIDIA Installation Plan".into(),
            details: vec![PlanDetail::new("OS", "Ubuntu 24.04")],
            devices: vec![],
            steps: vec![PlanStep::new(
                "Install the NVIDIA driver",
                CommandSpec::sudo("apt-get", ["install", "nvidia-driver"]),
            )],
            confirmation_warning: "No changes until confirmation.".into(),
            completion_message: "Installation completed.".into(),
            reboot_message: None,
        };

        let rendered = format_operation_plan(&plan);
        let output = console::strip_ansi_codes(&rendered);

        for expected in [
            "arc",
            "NVIDIA Installation Plan",
            "ENVIRONMENT",
            "Ubuntu 24.04",
            "CHANGES",
            "1 step",
            "01",
            "Install the NVIDIA driver",
            "$ sudo apt-get install nvidia-driver",
            "No changes until confirmation.",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output}"
            );
        }
    }

    #[test]
    fn diagnostic_output_has_sections_marks_and_rerun_instruction() {
        let diagnostics = Diagnostics {
            vendor: GpuVendor::Nvidia,
            checks: vec![
                sample(DiagnosticSection::Hardware, DiagnosticStatus::Pass),
                sample(
                    DiagnosticSection::OperatingSystem,
                    DiagnosticStatus::Warning,
                ),
                sample(DiagnosticSection::Driver, DiagnosticStatus::Error),
                sample(DiagnosticSection::CudaToolkit, DiagnosticStatus::Skipped),
            ],
            fix_plan: FixPlan::default(),
        };
        let output = format_diagnostics(&diagnostics);
        for expected in [
            "Hardware",
            "Operating System",
            "Driver",
            "CUDA Toolkit",
            "✓",
            "⚠",
            "✗",
            "↷ skipped",
            "rerun `arc doctor`",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output}"
            );
        }
    }

    #[test]
    fn manual_fix_plan_does_not_claim_no_fixes_were_executed() {
        let diagnostics = Diagnostics {
            vendor: GpuVendor::Nvidia,
            checks: vec![sample(DiagnosticSection::Driver, DiagnosticStatus::Error)],
            fix_plan: FixPlan {
                causes: vec![],
                fixes: vec![crate::model::environment::Fix {
                    id: crate::model::environment::FixId::RebootThenRecheck,
                    title: "Reboot, then verify".into(),
                    commands: vec![],
                    manual_steps: vec!["Run `sudo reboot`.".into()],
                    order: 1,
                }],
            },
        };
        let output = format_diagnostics(&diagnostics);
        assert!(output.contains("Complete the recommended manual actions"));
        assert!(!output.contains("No fixes were executed"));
    }

    #[test]
    fn status_is_compact_groups_duplicate_gpus_and_warns_on_inactive_driver() {
        let os = OsInfo {
            distribution: Distribution::Ubuntu,
            name: "Ubuntu".into(),
            version_id: "22.04".into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        };
        let status = sample_status();
        let output = format_system_status(&os, &[status], false);

        for expected in [
            "GPU Environment",
            "OS                  Ubuntu 22.04",
            "GPU                 2 × NVIDIA GeForce RTX 2080",
            "Driver installation Managed · Open",
            "Driver version      610.43.02",
            "Driver runtime      Reboot likely required",
            "CUDA Toolkit        Not installed",
            "nvcc                Not found",
            "Run `sudo reboot`, then rerun `arc status`.",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output}"
            );
        }
        assert_eq!(output.matches("NVIDIA GeForce RTX 2080").count(), 1);
        assert!(!output.contains("Driver package scope"));
        assert!(!output.contains("/lib/modules"));
    }

    #[test]
    fn verbose_status_adds_selected_technical_details_not_raw_modinfo() {
        let os = OsInfo {
            distribution: Distribution::Ubuntu,
            name: "Ubuntu".into(),
            version_id: "22.04".into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        };
        let output = format_system_status(&os, &[sample_status()], true);

        for expected in [
            "Technical details",
            "Kernel version          6.8.0-test",
            "Module path             /lib/modules/test/nvidia.ko",
            "Kernel module version   610.43.02",
            "Driver package scope    Full",
            "Secure Boot             Enabled",
            "Secure Boot signature   Signed · PKCS#7 · Test key",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output}"
            );
        }
        assert!(!output.contains("alias:"));
        assert!(!output.contains("depends:"));
    }

    fn sample_status() -> ProviderStatus {
        let device = || GpuDevice {
            vendor: GpuVendor::Nvidia,
            name: "NVIDIA Corporation TU104 [GeForce RTX 2080] [10de:1e87]".into(),
            pci_device_id: Some(0x1e87),
        };
        ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![device(), device()],
            driver: DriverInstallation::Managed {
                flavor: DriverFlavorState::Open,
                scope: DriverPackageScope::Full,
                branch: Some(610),
                packages: vec!["nvidia-driver-610-open".into()],
            },
            driver_version: None,
            driver_runtime_operational: false,
            driver_runtime_state: DriverRuntimeState::RebootLikelyRequired,
            dkms_status: Some("nvidia/610.43.02, 6.8.0-test, x86_64: installed".into()),
            driver_module: Some(DriverModuleInfo {
                path: Some("/lib/modules/test/nvidia.ko".into()),
                version: Some("610.43.02".into()),
                signer: Some("Test key".into()),
                signature_id: Some("PKCS#7".into()),
            }),
            kernel_version: Some("6.8.0-test".into()),
            secure_boot_enabled: Some(true),
            toolkits: vec![],
            active_toolkit: None,
        }
    }

    fn sample(section: DiagnosticSection, status: DiagnosticStatus) -> DiagnosticCheck {
        DiagnosticCheck {
            id: DiagnosticId::NvidiaGpu,
            section,
            name: "check".into(),
            status,
            evidence: vec!["fact".into()],
            problem: Some("problem".into()),
            dependencies: vec![],
            recommended_fixes: vec![],
        }
    }
}
