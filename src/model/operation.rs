use crate::model::{command::CommandSpec, device::GpuDevice};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanDetail {
    pub label: String,
    pub value: String,
}

impl PlanDetail {
    pub fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanStep {
    pub description: String,
    pub command: CommandSpec,
}

impl PlanStep {
    pub fn new(description: impl Into<String>, command: CommandSpec) -> Self {
        Self {
            description: description.into(),
            command,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationPlan {
    pub title: String,
    pub details: Vec<PlanDetail>,
    pub devices: Vec<GpuDevice>,
    pub steps: Vec<PlanStep>,
    pub confirmation_warning: String,
    pub completion_message: String,
    pub reboot_message: Option<String>,
}

impl OperationPlan {
    pub fn is_noop(&self) -> bool {
        self.steps.is_empty()
    }
}
