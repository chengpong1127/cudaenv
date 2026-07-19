use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::model::{command::CommandSpec, operation::OperationPlan};

pub trait CommandRunner {
    fn run(&self, command: &CommandSpec) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    fn run(&self, command: &CommandSpec) -> Result<()> {
        let status = Command::new(&command.program)
            .args(&command.args)
            .status()
            .with_context(|| format!("could not start {}", command.program))?;
        if !status.success() {
            bail!(
                "command failed (exit status {status}): {}",
                command.display()
            );
        }
        Ok(())
    }
}

pub fn execute_plan(runner: &impl CommandRunner, plan: &OperationPlan) -> Result<()> {
    for step in &plan.steps {
        runner
            .run(&step.command)
            .with_context(|| step.description.clone())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use crate::model::operation::{OperationPlan, PlanStep};

    use super::*;

    #[derive(Default)]
    struct RecordingRunner {
        commands: RefCell<Vec<CommandSpec>>,
    }

    impl CommandRunner for RecordingRunner {
        fn run(&self, command: &CommandSpec) -> Result<()> {
            self.commands.borrow_mut().push(command.clone());
            Ok(())
        }
    }

    #[test]
    fn executes_the_exact_commands_stored_in_the_plan() {
        let expected = CommandSpec::new("gpu-check", ["--version"]);
        let plan = OperationPlan {
            title: "Test".into(),
            details: vec![],
            devices: vec![],
            steps: vec![PlanStep::new("check GPU", expected.clone())],
            confirmation_warning: String::new(),
            completion_message: String::new(),
            reboot_message: None,
        };
        let runner = RecordingRunner::default();

        execute_plan(&runner, &plan).unwrap();

        assert_eq!(*runner.commands.borrow(), vec![expected]);
    }
}
