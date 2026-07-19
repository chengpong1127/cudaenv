use anyhow::Result;

use crate::{providers, ui::output};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoctorOutcome {
    Healthy,
    ErrorsFound,
}

pub fn run() -> Result<DoctorOutcome> {
    let mut errors = false;
    for provider in providers::registered() {
        let diagnostics = provider.diagnose()?;
        errors |= diagnostics.has_errors();
        output::diagnostics(&diagnostics);
    }
    Ok(if errors {
        DoctorOutcome::ErrorsFound
    } else {
        DoctorOutcome::Healthy
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        device::GpuVendor,
        environment::{
            DiagnosticCheck, DiagnosticId, DiagnosticSection, DiagnosticStatus, Diagnostics,
            FixPlan,
        },
    };

    #[test]
    fn error_status_is_distinct_from_success() {
        let diagnostics = Diagnostics {
            vendor: GpuVendor::Nvidia,
            checks: vec![DiagnosticCheck {
                id: DiagnosticId::NvidiaGpu,
                section: DiagnosticSection::Hardware,
                name: "GPU".into(),
                status: DiagnosticStatus::Error,
                evidence: vec![],
                problem: None,
                dependencies: vec![],
                recommended_fixes: vec![],
            }],
            fix_plan: FixPlan::default(),
        };
        assert!(diagnostics.has_errors());
        assert_eq!(outcome_for(&[diagnostics]), DoctorOutcome::ErrorsFound);
        assert_eq!(outcome_for(&[]), DoctorOutcome::Healthy);
    }

    fn outcome_for(diagnostics: &[Diagnostics]) -> DoctorOutcome {
        if diagnostics.iter().any(Diagnostics::has_errors) {
            DoctorOutcome::ErrorsFound
        } else {
            DoctorOutcome::Healthy
        }
    }
}
