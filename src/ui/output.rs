use std::fmt::Write;

use console::style;

use crate::model::{
    environment::{
        DiagnosticSection, DiagnosticStatus, Diagnostics, DriverFlavorState,
        DriverInstallation, DriverPackageScope, ProviderStatus,
    },
    operation::OperationPlan,
    system::OsInfo,
};
use crate::platform::command::ExecutionEvent;
use crate::providers::nvidia::upgrade::AvailableUpgrades;

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

pub fn system_status(
    os: &OsInfo,
    providers: &[ProviderStatus],
    upgrades: Option<&AvailableUpgrades>,
) {
    print!("{}", format_system_status(os, providers, upgrades));
}

pub fn format_system_status(
    os: &OsInfo,
    providers: &[ProviderStatus],
    upgrades: Option<&AvailableUpgrades>,
) -> String {
    let mut rendered = String::new();
    write_header(&mut rendered, "GPU Environment");

    writeln!(rendered, "\n  {}", section_label("System")).unwrap();
    write_metadata(&mut rendered, "OS", &os.display_name());
    write_metadata(&mut rendered, "Architecture", &os.architecture);

    for status in providers {
        writeln!(
            rendered,
            "\n  {}",
            section_label(&format!("{} GPU", status.vendor))
        )
        .unwrap();
        if status.devices.is_empty() {
            writeln!(
                rendered,
                "  {}  {}",
                style("○").yellow(),
                style("No compatible GPU detected").yellow()
            )
            .unwrap();
        } else {
            for device in &status.devices {
                writeln!(
                    rendered,
                    "  {}  {}",
                    style("◆").cyan(),
                    style(&device.name).bold()
                )
                .unwrap();
            }
        }

        writeln!(rendered, "\n  {}", section_label("Driver")).unwrap();
        let package_mark = match &status.driver {
            DriverInstallation::Managed { .. } => style("✓").green().bold().to_string(),
            DriverInstallation::Unmanaged { working: true, .. } => {
                style("!").yellow().bold().to_string()
            }
            DriverInstallation::Missing => style("○").yellow().to_string(),
            DriverInstallation::BrokenManaged { .. }
            | DriverInstallation::Unmanaged { working: false, .. } => {
                style("✗").red().bold().to_string()
            }
        };
        let package_value = driver_status_description(&status.driver);
        write_state(&mut rendered, &package_mark, "Package", &package_value);

        let (runtime_mark, runtime_value) = status.driver_version.as_deref().map_or_else(
            || {
                (
                    style("✗").red().bold().to_string(),
                    "Not loaded or not operational",
                )
            },
            |version| (style("✓").green().bold().to_string(), version),
        );
        write_state(&mut rendered, &runtime_mark, "Runtime", runtime_value);
        if let Some(version) = upgrades.and_then(|value| value.driver.as_deref()) {
            write_state(
                &mut rendered,
                &style("↑").cyan().bold().to_string(),
                "Update",
                &format!("{version} available"),
            );
        } else {
            write_state(
                &mut rendered,
                &style("·").dim().to_string(),
                "Update",
                "No compatible update reported",
            );
        }

        writeln!(rendered, "\n  {}", section_label("CUDA Toolkit")).unwrap();
        for toolkit in &status.toolkits {
            let version = toolkit.version.as_deref().unwrap_or("version unknown");
            write_state(
                &mut rendered,
                &style("✓").green().bold().to_string(),
                &toolkit.name,
                version,
            );
            write_nested_metadata(&mut rendered, "Packages", &toolkit.packages.join(", "));
            write_nested_metadata(
                &mut rendered,
                "Ownership",
                if toolkit.manageable {
                    "Managed by arc"
                } else {
                    "External"
                },
            );
            if let Some(version) = upgrades.and_then(|value| value.toolkit.as_deref()) {
                write_state(
                    &mut rendered,
                    &style("↑").cyan().bold().to_string(),
                    "Update",
                    &format!("compatible {version} available"),
                );
            }
        }
        if status.toolkits.is_empty() {
            write_state(
                &mut rendered,
                &style("○").dim().to_string(),
                "System",
                "Not installed",
            );
        }
        if let Some(active) = &status.active_toolkit {
            write_state(
                &mut rendered,
                &style("•").cyan().to_string(),
                "Active nvcc",
                active.version.as_deref().unwrap_or("version unknown"),
            );
            write_nested_metadata(
                &mut rendered,
                "Path",
                active.executable_path.as_deref().unwrap_or("unknown"),
            );
            write_nested_metadata(&mut rendered, "Ownership", "External / informational");
        } else {
            write_state(
                &mut rendered,
                &style("○").dim().to_string(),
                "Active nvcc",
                "Not found on PATH",
            );
        }
    }

    writeln!(rendered).unwrap();
    rendered
}

pub fn diagnostics(diagnostics: &Diagnostics) {
    print!("{}", format_diagnostics(diagnostics));
}

pub fn format_diagnostics(diagnostics: &Diagnostics) -> String {
    let mut rendered = String::new();
    write_header(&mut rendered, &format!("{} Diagnostics", diagnostics.vendor));

    let passed = diagnostic_count(diagnostics, DiagnosticStatus::Pass);
    let warnings = diagnostic_count(diagnostics, DiagnosticStatus::Warning);
    let errors = diagnostic_count(diagnostics, DiagnosticStatus::Error);
    let skipped = diagnostic_count(diagnostics, DiagnosticStatus::Skipped);
    let (result_label, result_style) = if errors > 0 {
        ("Action required", style("Action required").red().bold())
    } else if warnings > 0 {
        ("Completed with warnings", style("Completed with warnings").yellow().bold())
    } else {
        ("Healthy", style("Healthy").green().bold())
    };
    writeln!(
        rendered,
        "\n  {}  {}",
        result_style,
        style(format!(
            "{}  ·  {}  ·  {}  ·  {}",
            count_phrase(passed, "passed", "passed"),
            count_phrase(warnings, "warning", "warnings"),
            count_phrase(errors, "error", "errors"),
            count_phrase(skipped, "skipped", "skipped"),
        ))
        .dim()
    )
    .unwrap();

    for section in [
        DiagnosticSection::Hardware,
        DiagnosticSection::OperatingSystem,
        DiagnosticSection::Driver,
        DiagnosticSection::CudaToolkit,
    ] {
        writeln!(rendered, "\n  {}", section_label(&section.to_string())).unwrap();
        for check in diagnostics
            .checks
            .iter()
            .filter(|check| check.section == section)
        {
            writeln!(
                rendered,
                "  {}  {}",
                styled_diagnostic_mark(check.status),
                if check.status == DiagnosticStatus::Pass {
                    style(&check.name).to_string()
                } else {
                    style(&check.name).bold().to_string()
                }
            )
            .unwrap();
            if check.status != DiagnosticStatus::Pass {
                if let Some(problem) = &check.problem {
                    writeln!(rendered, "     {problem}").unwrap();
                }
                for evidence in check
                    .evidence
                    .iter()
                    .filter(|evidence| !empty_evidence(evidence))
                {
                    writeln!(rendered, "     {} {}", style("›").dim(), style(evidence).dim())
                        .unwrap();
                }
            }
        }
    }
    if diagnostics.healthy() && diagnostics.fix_plan.causes.is_empty() {
        writeln!(
            rendered,
            "\n  {}  {}\n",
            if warnings > 0 {
                style("!").yellow().bold().to_string()
            } else {
                style("✓").green().bold().to_string()
            },
            result_label
        )
        .unwrap();
        return rendered;
    }

    writeln!(rendered, "\n  {}", section_label("Next steps")).unwrap();
    for (index, cause) in diagnostics.fix_plan.causes.iter().enumerate() {
        writeln!(
            rendered,
            "  {}  {}",
            style(if index == 0 { "!" } else { "·" }).red().bold(),
            style(&cause.title).bold()
        )
        .unwrap();
        writeln!(
            rendered,
            "     {} confidence",
            style(cause.confidence).dim()
        )
        .unwrap();
        for evidence in &cause.evidence {
            writeln!(rendered, "     {} {}", style("›").dim(), style(evidence).dim()).unwrap();
        }
    }
    for (index, fix) in diagnostics.fix_plan.fixes.iter().enumerate() {
        writeln!(
            rendered,
            "\n  {}  {}",
            style(format!("{:02}", index + 1)).cyan().bold(),
            style(&fix.title).bold()
        )
        .unwrap();
        for command in &fix.commands {
            writeln!(
                rendered,
                "      {} {}",
                style("$").dim(),
                style(command.display()).cyan()
            )
            .unwrap();
        }
        for step in &fix.manual_steps {
            writeln!(rendered, "      {} {step}", style("→").dim()).unwrap();
        }
    }
    if diagnostics.fix_plan.causes.is_empty() && diagnostics.fix_plan.fixes.is_empty() {
        writeln!(
            rendered,
            "  {}  No remediation plan was generated; review the failed checks above.",
            style("!").yellow().bold()
        )
        .unwrap();
    }
    writeln!(
        rendered,
        "\n  {}  No changes were made. Complete the steps, then rerun `arc doctor`.\n",
        style("i").cyan().bold()
    )
    .unwrap();
    rendered
}

fn write_header(rendered: &mut String, title: &str) {
    writeln!(rendered).unwrap();
    writeln!(
        rendered,
        "  {}  {}",
        style("arc").cyan().bold(),
        style(title).bold()
    )
    .unwrap();
    writeln!(rendered, "  {}", style("─".repeat(52)).dim()).unwrap();
}

fn write_metadata(rendered: &mut String, label: &str, value: &str) {
    let padded_label = format!("{label:<12}");
    writeln!(rendered, "  {}  {}", style(padded_label).dim(), value).unwrap();
}

fn write_nested_metadata(rendered: &mut String, label: &str, value: &str) {
    let padded_label = format!("{label:<10}");
    writeln!(rendered, "     {}  {}", style(padded_label).dim(), value).unwrap();
}

fn write_state(rendered: &mut String, mark: &str, label: &str, value: &str) {
    let padded_label = format!("{label:<10}");
    writeln!(rendered, "  {mark}  {}  {value}", style(padded_label).dim()).unwrap();
}

fn driver_status_description(driver: &DriverInstallation) -> String {
    match driver {
        DriverInstallation::Missing => "Not installed".into(),
        DriverInstallation::Managed {
            flavor,
            scope,
            branch,
            ..
        } => {
            let flavor = match flavor {
                DriverFlavorState::Open => "open kernel modules",
                DriverFlavorState::Proprietary => "proprietary kernel modules",
                DriverFlavorState::Mixed => "mixed module packages",
            };
            let scope = match scope {
                DriverPackageScope::Full => "full",
                DriverPackageScope::ComputeOnly => "compute-only",
                DriverPackageScope::DesktopOnly => "desktop-only",
            };
            format!(
                "Managed · {flavor} · {scope}{}",
                branch
                    .map(|branch| format!(" · pinned to R{branch}"))
                    .unwrap_or_default()
            )
        }
        DriverInstallation::BrokenManaged { flavor, .. } => format!(
            "Broken managed {} installation",
            match flavor {
                DriverFlavorState::Open => "open-module",
                DriverFlavorState::Proprietary => "proprietary-module",
                DriverFlavorState::Mixed => "mixed-module",
            }
        ),
        DriverInstallation::Unmanaged {
            working,
            runfile_likely,
        } => format!(
            "{} unmanaged installation{}",
            if *working { "Working" } else { "Broken" },
            if *runfile_likely {
                " · runfile likely"
            } else {
                ""
            }
        ),
    }
}

fn diagnostic_count(diagnostics: &Diagnostics, status: DiagnosticStatus) -> usize {
    diagnostics
        .checks
        .iter()
        .filter(|check| check.status == status)
        .count()
}

fn count_phrase(count: usize, singular: &str, plural: &str) -> String {
    format!("{count} {}", if count == 1 { singular } else { plural })
}

fn empty_evidence(evidence: &str) -> bool {
    evidence
        .split_once(':')
        .is_some_and(|(_, value)| value.trim().is_empty())
}

fn styled_diagnostic_mark(status: DiagnosticStatus) -> String {
    match status {
        DiagnosticStatus::Pass => style("✓").green().bold().to_string(),
        DiagnosticStatus::Warning => style("!").yellow().bold().to_string(),
        DiagnosticStatus::Error => style("✗").red().bold().to_string(),
        DiagnosticStatus::Skipped => style("↷").dim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        command::CommandSpec,
        device::{GpuDevice, GpuVendor},
        environment::{
            DiagnosticCheck, DiagnosticId, DiagnosticSection, DiagnosticStatus,
            DriverFlavorState, DriverInstallation, DriverPackageScope, FixPlan, ProviderStatus,
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
        let rendered = format_diagnostics(&diagnostics);
        let output = console::strip_ansi_codes(&rendered);
        for expected in [
            "NVIDIA Diagnostics",
            "Action required",
            "1 passed  ·  1 warning  ·  1 error  ·  1 skipped",
            "HARDWARE",
            "OPERATING SYSTEM",
            "DRIVER",
            "CUDA TOOLKIT",
            "✓",
            "!",
            "✗",
            "↷",
            "› fact",
            "NEXT STEPS",
            "rerun `arc doctor`",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output}"
            );
        }
    }

    #[test]
    fn status_output_has_a_compact_environment_hierarchy() {
        let os = OsInfo {
            distribution: Distribution::Ubuntu,
            name: "Ubuntu".into(),
            version_id: "26.04".into(),
            architecture: "x86_64".into(),
            is_wsl: false,
        };
        let provider = ProviderStatus {
            vendor: GpuVendor::Nvidia,
            devices: vec![GpuDevice {
                vendor: GpuVendor::Nvidia,
                name: "GeForce RTX 3080 Ti".into(),
                pci_device_id: Some(0x2420),
            }],
            driver: DriverInstallation::Managed {
                flavor: DriverFlavorState::Open,
                scope: DriverPackageScope::Full,
                branch: None,
                packages: vec!["nvidia-open".into()],
            },
            driver_version: Some("610.43.02".into()),
            toolkits: vec![],
            active_toolkit: None,
        };

        let rendered = format_system_status(&os, &[provider], None);
        let output = console::strip_ansi_codes(&rendered);
        for expected in [
            "arc  GPU Environment",
            "SYSTEM",
            "Ubuntu 26.04",
            "NVIDIA GPU",
            "GeForce RTX 3080 Ti",
            "DRIVER",
            "Package",
            "Runtime",
            "CUDA TOOLKIT",
            "Not installed",
            "Not found on PATH",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output}"
            );
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
