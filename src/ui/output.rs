use crate::model::{
    environment::{DiagnosticSection, DiagnosticStatus, Diagnostics, ProviderStatus},
    operation::OperationPlan,
    system::OsInfo,
};

pub fn operation_plan(plan: &OperationPlan) {
    println!("{}\n", plan.title);
    for detail in &plan.details {
        println!("{}: {}", detail.label, detail.value);
    }
    if !plan.devices.is_empty() {
        println!("GPU(s):");
        for device in &plan.devices {
            println!("  - {} ({})", device.name, device.vendor);
        }
    }
    println!("\nCommands:");
    if plan.steps.is_empty() {
        println!("  # no changes required");
    } else {
        for step in &plan.steps {
            println!("  # {}", step.description);
            println!("  $ {}", step.command.display());
        }
    }
    println!("{}", plan.confirmation_warning);
}

pub fn operation_completed(plan: &OperationPlan) {
    println!("\n{}", plan.completion_message);
    if let Some(message) = &plan.reboot_message {
        println!("{message}");
    }
}

pub fn system_status(os: &OsInfo, providers: &[ProviderStatus]) {
    println!("GPU Environment\n");
    println!("OS:\n{}", os.display_name());
    for status in providers {
        println!("\n{} GPU(s):", status.vendor);
        if status.devices.is_empty() {
            println!("Not detected");
        } else {
            for device in &status.devices {
                println!("{}", device.name);
            }
        }
        println!(
            "\n{} Driver package:\n{}",
            status.vendor,
            if status.driver_installed {
                "Installed"
            } else {
                "Not installed"
            }
        );
        println!(
            "\n{} Driver runtime:\n{}",
            status.vendor,
            status
                .driver_version
                .as_deref()
                .unwrap_or("Not loaded or not operational")
        );
        for toolkit in &status.toolkits {
            println!("\n{}:\n{}", toolkit.name, toolkit.version);
        }
        if status.toolkits.is_empty() {
            println!("\nDevelopment Toolkit:\nNot installed");
        }
    }
}

pub fn diagnostics(diagnostics: &Diagnostics) {
    print!("{}", format_diagnostics(diagnostics));
}

pub fn format_diagnostics(diagnostics: &Diagnostics) -> String {
    use std::fmt::Write;

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
    writeln!(
        rendered,
        "\nNo fixes were executed. After completing the plan, rerun `cudaenv doctor`."
    )
    .unwrap();
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
        device::GpuVendor,
        environment::{
            DiagnosticCheck, DiagnosticId, DiagnosticSection, DiagnosticStatus, FixPlan,
        },
    };

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
            "rerun `cudaenv doctor`",
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
