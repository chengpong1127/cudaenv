use crate::model::{
    environment::{Diagnostics, ProviderStatus},
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
            "\n{} Driver:\n{}",
            status.vendor,
            status.driver_version.as_deref().unwrap_or("Not installed")
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
    println!("{} Diagnostics\n", diagnostics.vendor);
    for check in &diagnostics.checks {
        println!("{} {}", mark(check.passed), check.name);
    }
    if diagnostics.healthy() {
        println!("\nHealthy");
        return;
    }
    println!("\nProblems found");
    for problem in diagnostics.checks.iter().filter_map(|check| {
        (!check.passed)
            .then_some(check.problem.as_deref())
            .flatten()
    }) {
        println!("- {problem}");
    }
}

fn mark(ok: bool) -> &'static str {
    if ok { "✓" } else { "✗" }
}
