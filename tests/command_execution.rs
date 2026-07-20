mod support;

use std::cell::RefCell;

use arc::{
    model::command::CommandSpec,
    platform::command::{ExecutionEvent, OutputMode, execute_plan, normalize_for_user},
};
use support::{FakeCommandRunner, operation_plan};

#[test]
fn root_normalization_removes_sudo_without_touching_arguments() {
    let mut plan = operation_plan(vec![CommandSpec::sudo("apt-get", ["update"])]);

    normalize_for_user(&mut plan, true);

    assert_eq!(plan.steps[0].command.program, "apt-get");
    assert_eq!(plan.steps[0].command.args, ["update"]);
}

#[test]
fn non_root_normalization_preserves_sudo() {
    let mut plan = operation_plan(vec![CommandSpec::sudo("apt-get", ["update"])]);

    normalize_for_user(&mut plan, false);

    assert_eq!(plan.steps[0].command.program, "sudo");
    assert_eq!(plan.steps[0].command.args, ["apt-get", "update"]);
}

#[test]
fn execution_stops_at_the_first_failed_command() {
    let runner = FakeCommandRunner::with_results(vec![
        FakeCommandRunner::success(),
        FakeCommandRunner::failure(23, "package transaction failed"),
    ]);
    let plan = operation_plan(vec![
        CommandSpec::new("first", ["ok"]),
        CommandSpec::new("second", ["fails"]),
        CommandSpec::new("never", ["runs"]),
    ]);
    let events = RefCell::new(Vec::new());

    let error = execute_plan(&runner, &plan, true, |event| match event {
        ExecutionEvent::Started { index, .. } => events.borrow_mut().push(("start", index)),
        ExecutionEvent::Completed { index, .. } => events.borrow_mut().push(("complete", index)),
        ExecutionEvent::Failed { index, .. } => events.borrow_mut().push(("failed", index)),
    })
    .unwrap_err();

    assert_eq!(runner.calls().len(), 2);
    assert_eq!(
        events.into_inner(),
        [("start", 0), ("complete", 0), ("start", 1), ("failed", 1)]
    );
    assert!(error.to_string().contains("exit status 23"));
}

#[test]
fn sudo_authentication_failure_prevents_plan_execution() {
    let runner = FakeCommandRunner::with_results(vec![FakeCommandRunner::failure(1, "denied")]);
    let plan = operation_plan(vec![CommandSpec::sudo("apt-get", ["update"])]);
    let events = RefCell::new(0);

    let error = execute_plan(&runner, &plan, true, |_| *events.borrow_mut() += 1).unwrap_err();
    let calls = runner.calls();

    assert_eq!(*events.borrow(), 0);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0.command.program, "sudo");
    assert_eq!(calls[0].0.command.args, ["-v"]);
    assert_eq!(calls[0].1, OutputMode::Inherit);
    assert!(error.to_string().contains("sudo authentication failed"));
}

#[test]
fn verbose_execution_inherits_process_streams() {
    let runner = FakeCommandRunner::with_results(vec![FakeCommandRunner::success()]);
    let plan = operation_plan(vec![CommandSpec::new("tool", ["run"])]);

    let summary = execute_plan(&runner, &plan, true, |_| {}).unwrap();

    assert_eq!(runner.calls()[0].1, OutputMode::Inherit);
    assert!(summary.log_path.is_none());
}
