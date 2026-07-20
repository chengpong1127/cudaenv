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
pub struct PlanStage {
    pub title: String,
    pub running: String,
    pub success: String,
    pub failure: String,
}

impl PlanStage {
    pub fn new(
        title: impl Into<String>,
        running: impl Into<String>,
        success: impl Into<String>,
        failure: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            running: running.into(),
            success: success.into(),
            failure: failure.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanStep {
    pub description: String,
    pub command: CommandSpec,
    pub stage: PlanStage,
}

impl PlanStep {
    pub fn new(description: impl Into<String>, command: CommandSpec) -> Self {
        let description = description.into();
        Self {
            stage: PlanStage::new(
                &description,
                format!("{description}..."),
                &description,
                format!("{description} failed"),
            ),
            description,
            command,
        }
    }

    pub fn in_stage(mut self, stage: &PlanStage) -> Self {
        self.stage = stage.clone();
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextStep {
    LoadNvidiaDriver,
    RebootBeforeNvidiaInstall,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationPlan {
    pub title: String,
    pub details: Vec<PlanDetail>,
    pub devices: Vec<GpuDevice>,
    pub steps: Vec<PlanStep>,
    pub confirmation_warning: String,
    pub completion_message: String,
    pub next_step: Option<NextStep>,
}

impl OperationPlan {
    pub fn is_noop(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn stage_count(&self) -> usize {
        self.steps
            .iter()
            .enumerate()
            .filter(|(index, step)| *index == 0 || self.steps[*index - 1].stage != step.stage)
            .count()
    }

    pub fn stage_position(&self, command_index: usize) -> (usize, bool, bool) {
        let mut stage_index = 0;
        for index in 1..=command_index {
            if self.steps[index - 1].stage != self.steps[index].stage {
                stage_index += 1;
            }
        }
        let first = command_index == 0
            || self.steps[command_index - 1].stage != self.steps[command_index].stage;
        let last = command_index + 1 == self.steps.len()
            || self.steps[command_index + 1].stage != self.steps[command_index].stage;
        (stage_index, first, last)
    }
}
