use std::{
    env,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};

use crate::model::{
    command::CommandSpec,
    operation::{OperationPlan, PlanStep},
};

pub fn normalize_for_current_user(plan: &mut OperationPlan) {
    normalize_for_user(plan, running_as_root());
}

/// Normalize privileged commands for an explicitly described execution user.
///
/// Keeping this transformation separate from user detection makes the command
/// policy deterministic and directly testable.
pub fn normalize_for_user(plan: &mut OperationPlan, running_as_root: bool) {
    if !running_as_root {
        return;
    }
    for step in &mut plan.steps {
        if step.command.program == "sudo" && !step.command.args.is_empty() {
            step.command.program = step.command.args.remove(0);
        }
    }
}

fn running_as_root() -> bool {
    Command::new("id").arg("-u").output().is_ok_and(|output| {
        output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == "0"
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    Capture,
    Inherit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandInvocation {
    pub command: CommandSpec,
    pub env: Vec<(String, String)>,
}

impl CommandInvocation {
    pub fn new(command: CommandSpec) -> Self {
        Self {
            command,
            env: Vec::new(),
        }
    }
}

pub fn capture(runner: &impl CommandRunner, command: CommandSpec) -> Result<CommandResult> {
    runner.run(&CommandInvocation::new(command), OutputMode::Capture)
}

pub fn capture_stdout(runner: &impl CommandRunner, command: CommandSpec) -> Result<Option<String>> {
    let result = capture(runner, command)?;
    Ok(result
        .success
        .then(|| String::from_utf8_lossy(&result.stdout).trim().to_owned()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandResult {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl CommandResult {
    #[cfg(test)]
    fn success(stdout: impl Into<Vec<u8>>, stderr: impl Into<Vec<u8>>) -> Self {
        Self {
            success: true,
            exit_code: Some(0),
            stdout: stdout.into(),
            stderr: stderr.into(),
        }
    }
}

pub trait CommandRunner {
    fn run(&self, invocation: &CommandInvocation, mode: OutputMode) -> Result<CommandResult>;
}

pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    fn run(&self, invocation: &CommandInvocation, mode: OutputMode) -> Result<CommandResult> {
        let mut command = Command::new(&invocation.command.program);
        command
            .args(&invocation.command.args)
            .envs(invocation.env.iter().map(|(k, v)| (k, v)));
        match mode {
            OutputMode::Inherit => {
                let status = command
                    .stdin(Stdio::inherit())
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .status()
                    .with_context(|| format!("could not start {}", invocation.command.program))?;
                Ok(CommandResult {
                    success: status.success(),
                    exit_code: status.code(),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                })
            }
            OutputMode::Capture => {
                let mut child = command
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .with_context(|| format!("could not start {}", invocation.command.program))?;
                let stdout = child
                    .stdout
                    .take()
                    .context("could not capture command stdout")?;
                let stderr = child
                    .stderr
                    .take()
                    .context("could not capture command stderr")?;
                // Both pipes must be drained while the child is running: reading one fully before
                // the other can deadlock when the other pipe fills.
                let stdout_reader = thread::spawn(move || read_all(stdout));
                let stderr_reader = thread::spawn(move || read_all(stderr));
                let status = child.wait().context("could not wait for command")?;
                let stdout = stdout_reader
                    .join()
                    .map_err(|_| anyhow!("stdout capture thread panicked"))??;
                let stderr = stderr_reader
                    .join()
                    .map_err(|_| anyhow!("stderr capture thread panicked"))??;
                Ok(CommandResult {
                    success: status.success(),
                    exit_code: status.code(),
                    stdout,
                    stderr,
                })
            }
        }
    }
}

fn read_all(mut reader: impl Read) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[derive(Clone, Copy, Debug)]
pub enum ExecutionEvent<'a> {
    Started {
        index: usize,
        total: usize,
        step: &'a PlanStep,
    },
    Completed {
        index: usize,
        total: usize,
        step: &'a PlanStep,
    },
    Failed {
        index: usize,
        total: usize,
        step: &'a PlanStep,
    },
}

#[derive(Debug)]
pub struct ExecutionSummary {
    pub log_path: Option<PathBuf>,
}

pub fn execute_plan<'a>(
    runner: &impl CommandRunner,
    plan: &'a OperationPlan,
    verbose: bool,
    report: impl FnMut(ExecutionEvent<'a>),
) -> Result<ExecutionSummary> {
    execute_plan_in(runner, plan, verbose, None, report)
}

#[doc(hidden)]
pub fn execute_plan_in<'a>(
    runner: &impl CommandRunner,
    plan: &'a OperationPlan,
    verbose: bool,
    log_dir: Option<&Path>,
    mut report: impl FnMut(ExecutionEvent<'a>),
) -> Result<ExecutionSummary> {
    authenticate_sudo(runner, plan)?;

    let mut log = if verbose {
        None
    } else {
        let path = create_log_path(log_dir)?;
        Some((
            File::create(&path)
                .with_context(|| format!("could not create log {}", path.display()))?,
            path,
        ))
    };
    let total = plan.steps.len();
    for (index, step) in plan.steps.iter().enumerate() {
        report(ExecutionEvent::Started { index, total, step });
        let invocation = prepare_invocation(&step.command, !verbose);
        if let Some((file, _)) = &mut log {
            writeln!(
                file,
                "\n===== {}/{}: {} =====",
                index + 1,
                total,
                step.description
            )?;
            writeln!(file, "$ {}", invocation.command.display())?;
        }
        let result = match runner.run(
            &invocation,
            if verbose {
                OutputMode::Inherit
            } else {
                OutputMode::Capture
            },
        ) {
            Ok(result) => result,
            Err(error) => {
                report(ExecutionEvent::Failed { index, total, step });
                if let Some((file, path)) = &mut log {
                    writeln!(file, "--- execution error ---")?;
                    writeln!(file, "{error:#}")?;
                    file.flush()?;
                    let tail = tail_lines(path, 40)?;
                    bail!(
                        "could not execute command: {}\n\nLast log lines:\n{}\n\nFull log: {}",
                        invocation.command.display(),
                        tail,
                        path.display()
                    );
                }
                return Err(error)
                    .with_context(|| format!("Failed command: {}", invocation.command.display()));
            }
        };
        if let Some((file, _)) = &mut log {
            write_stream(file, "stdout", &result.stdout)?;
            write_stream(file, "stderr", &result.stderr)?;
            writeln!(file, "[exit status: {}]", exit_label(&result))?;
            file.flush()?;
        }
        if !result.success {
            report(ExecutionEvent::Failed { index, total, step });
            let detail = if let Some((_, path)) = &log {
                let tail = tail_lines(path, 40)?;
                format!(
                    "command failed (exit status {}): {}\n\nLast log lines:\n{}\n\nFull log: {}",
                    exit_label(&result),
                    invocation.command.display(),
                    tail,
                    path.display()
                )
            } else {
                format!(
                    "command failed (exit status {}): {}",
                    exit_label(&result),
                    invocation.command.display()
                )
            };
            bail!(detail);
        }
        report(ExecutionEvent::Completed { index, total, step });
    }
    Ok(ExecutionSummary {
        log_path: log.map(|(_, path)| path),
    })
}

fn authenticate_sudo(runner: &impl CommandRunner, plan: &OperationPlan) -> Result<()> {
    if !plan.steps.iter().any(|step| step.command.program == "sudo") {
        return Ok(());
    }
    let invocation = CommandInvocation::new(CommandSpec::new("sudo", ["-v"]));
    let result = runner
        .run(&invocation, OutputMode::Inherit)
        .context("could not authenticate with sudo")?;
    if !result.success {
        bail!(
            "sudo authentication failed (exit status {}). No changes were made.",
            exit_label(&result)
        );
    }
    Ok(())
}

fn prepare_invocation(original: &CommandSpec, quiet: bool) -> CommandInvocation {
    let (sudo, program, args) = if original.program == "sudo" && !original.args.is_empty() {
        (true, original.args[0].clone(), original.args[1..].to_vec())
    } else {
        (false, original.program.clone(), original.args.clone())
    };
    let package_manager = matches!(
        program.as_str(),
        "apt" | "apt-get" | "dnf" | "yum" | "tdnf" | "zypper"
    );
    let mut args = if quiet && package_manager {
        quiet_package_args(&program, args)
    } else {
        args
    };
    let apt = matches!(program.as_str(), "apt" | "apt-get");
    if sudo {
        let mut sudo_args = vec!["-n".to_owned()];
        if apt {
            sudo_args.extend([
                "env".to_owned(),
                "DEBIAN_FRONTEND=noninteractive".to_owned(),
            ]);
        }
        sudo_args.push(program);
        sudo_args.append(&mut args);
        CommandInvocation::new(CommandSpec::new("sudo", sudo_args))
    } else {
        let mut invocation = CommandInvocation::new(CommandSpec::new(&program, args));
        if apt {
            invocation
                .env
                .push(("DEBIAN_FRONTEND".into(), "noninteractive".into()));
        }
        invocation
    }
}

fn quiet_package_args(program: &str, mut args: Vec<String>) -> Vec<String> {
    let options: &[&str] = match program {
        "apt" | "apt-get" => &["-q", "-o", "Dpkg::Use-Pty=0"],
        "dnf" | "yum" | "tdnf" | "zypper" => &["--quiet"],
        _ => &[],
    };
    if !args.iter().any(|arg| arg == "-q" || arg == "--quiet") {
        args.splice(0..0, options.iter().map(|value| (*value).to_owned()));
    } else if matches!(program, "apt" | "apt-get")
        && !args.iter().any(|arg| arg == "Dpkg::Use-Pty=0")
    {
        args.splice(0..0, ["-o".to_owned(), "Dpkg::Use-Pty=0".to_owned()]);
    }
    args
}

fn create_log_path(override_dir: Option<&Path>) -> Result<PathBuf> {
    let directory = override_dir.map(Path::to_path_buf).or_else(default_log_dir).context(
        "could not determine a cache directory for command logs (HOME and XDG_CACHE_HOME are unset)",
    )?;
    fs::create_dir_all(&directory)
        .with_context(|| format!("could not create log directory {}", directory.display()))?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(directory.join(format!("arc-{timestamp}-{}.log", std::process::id())))
}

fn default_log_dir() -> Option<PathBuf> {
    env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .map(|cache| cache.join("arc").join("logs"))
}

fn write_stream(file: &mut File, label: &str, bytes: &[u8]) -> Result<()> {
    writeln!(file, "--- {label} ---")?;
    file.write_all(bytes)?;
    if !bytes.ends_with(b"\n") {
        writeln!(file)?;
    }
    Ok(())
}

fn exit_label(result: &CommandResult) -> String {
    result
        .exit_code
        .map_or_else(|| "terminated by signal".into(), |code| code.to_string())
}

fn tail_lines(path: &Path, count: usize) -> Result<String> {
    let contents =
        fs::read(path).with_context(|| format!("could not read log {}", path.display()))?;
    let text = String::from_utf8_lossy(&contents);
    Ok(text
        .lines()
        .rev()
        .take(count)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::VecDeque};

    use super::*;
    use crate::model::operation::OperationPlan;

    #[derive(Default)]
    struct FakeRunner {
        calls: RefCell<Vec<(CommandInvocation, OutputMode)>>,
        results: RefCell<VecDeque<CommandResult>>,
    }

    impl FakeRunner {
        fn with_results(results: Vec<CommandResult>) -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                results: RefCell::new(results.into()),
            }
        }
    }

    impl CommandRunner for FakeRunner {
        fn run(&self, invocation: &CommandInvocation, mode: OutputMode) -> Result<CommandResult> {
            self.calls.borrow_mut().push((invocation.clone(), mode));
            self.results
                .borrow_mut()
                .pop_front()
                .context("fake runner has no result")
        }
    }

    fn plan(commands: Vec<CommandSpec>) -> OperationPlan {
        OperationPlan {
            title: "Test".into(),
            details: vec![],
            devices: vec![],
            steps: commands
                .into_iter()
                .enumerate()
                .map(|(i, command)| PlanStep::new(format!("step {}", i + 1), command))
                .collect(),
            confirmation_warning: String::new(),
            completion_message: String::new(),
            next_step: None,
        }
    }

    fn temp_log_dir(name: &str) -> PathBuf {
        env::temp_dir().join(format!("arc-command-test-{name}-{}", std::process::id()))
    }

    #[test]
    fn successful_execution_captures_both_streams_and_creates_log() {
        let dir = temp_log_dir("success");
        let runner = FakeRunner::with_results(vec![CommandResult::success(b"out\n", b"warning\n")]);
        let summary = execute_plan_in(
            &runner,
            &plan(vec![CommandSpec::new("tool", ["go"])]),
            false,
            Some(&dir),
            |_| {},
        )
        .unwrap();
        let log = fs::read_to_string(summary.log_path.unwrap()).unwrap();
        assert!(log.contains("out\n"));
        assert!(log.contains("warning\n"));
        assert_eq!(runner.calls.borrow()[0].1, OutputMode::Capture);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn failure_stops_later_steps_and_reports_tail_and_log_path() {
        let dir = temp_log_dir("failure");
        let failed = CommandResult {
            success: false,
            exit_code: Some(23),
            stdout: b"partial\n".to_vec(),
            stderr: b"important error\n".to_vec(),
        };
        let runner = FakeRunner::with_results(vec![failed]);
        let error = execute_plan_in(
            &runner,
            &plan(vec![
                CommandSpec::new("bad", ["arg with spaces"]),
                CommandSpec::new("never", ["run"]),
            ]),
            false,
            Some(&dir),
            |_| {},
        )
        .unwrap_err()
        .to_string();
        assert_eq!(runner.calls.borrow().len(), 1);
        assert!(error.contains("bad 'arg with spaces'"));
        assert!(error.contains("important error"));
        assert!(error.contains("Full log:"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn failure_inside_grouped_stage_reports_parent_and_stops_following_commands() {
        let dir = temp_log_dir("grouped-failure");
        let stage = crate::model::operation::PlanStage::new(
            "Configure repository",
            "Configuring repository...",
            "Configured repository",
            "Could not configure repository",
        );
        let mut grouped = plan(vec![
            CommandSpec::new("first", ["ok"]),
            CommandSpec::new("second", ["fails"]),
            CommandSpec::new("never", ["run"]),
        ]);
        grouped.steps[0].stage = stage.clone();
        grouped.steps[1].stage = stage;
        let failed = CommandResult {
            success: false,
            exit_code: Some(17),
            stdout: vec![],
            stderr: b"group setup failed\n".to_vec(),
        };
        let runner = FakeRunner::with_results(vec![CommandResult::success([], []), failed]);
        let events = RefCell::new(Vec::new());
        let error = execute_plan_in(&runner, &grouped, false, Some(&dir), |event| match event {
            ExecutionEvent::Started { index, step, .. } => events
                .borrow_mut()
                .push(format!("start:{index}:{}", step.stage.title)),
            ExecutionEvent::Completed { index, .. } => {
                events.borrow_mut().push(format!("complete:{index}"))
            }
            ExecutionEvent::Failed { index, step, .. } => events
                .borrow_mut()
                .push(format!("fail:{index}:{}", step.stage.failure)),
        })
        .unwrap_err()
        .to_string();

        assert_eq!(runner.calls.borrow().len(), 2);
        assert_eq!(
            events.into_inner(),
            [
                "start:0:Configure repository",
                "complete:0",
                "start:1:Configure repository",
                "fail:1:Could not configure repository",
            ]
        );
        assert!(error.contains("second fails"));
        assert!(error.contains("group setup failed"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn verbose_selects_inherited_streaming_and_does_not_create_a_log() {
        let dir = temp_log_dir("verbose");
        let runner = FakeRunner::with_results(vec![CommandResult::success([], [])]);
        let summary = execute_plan_in(
            &runner,
            &plan(vec![CommandSpec::new("tool", ["go"])]),
            true,
            Some(&dir),
            |_| {},
        )
        .unwrap();
        assert_eq!(runner.calls.borrow()[0].1, OutputMode::Inherit);
        assert!(summary.log_path.is_none());
        assert!(!dir.exists());
    }

    #[test]
    fn sudo_authentication_failure_happens_before_progress_or_commands() {
        let failed = CommandResult {
            success: false,
            exit_code: Some(1),
            stdout: vec![],
            stderr: vec![],
        };
        let runner = FakeRunner::with_results(vec![failed]);
        let events = RefCell::new(0);
        let error = execute_plan_in(
            &runner,
            &plan(vec![CommandSpec::sudo("apt-get", ["update"])]),
            false,
            Some(&temp_log_dir("sudo")),
            |_| *events.borrow_mut() += 1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("sudo authentication failed"));
        assert_eq!(*events.borrow(), 0);
        assert_eq!(runner.calls.borrow()[0].0.command.display(), "sudo -v");
        assert_eq!(runner.calls.borrow()[0].1, OutputMode::Inherit);
    }

    #[test]
    fn quiet_package_commands_use_noninteractive_machine_friendly_options() {
        let apt = prepare_invocation(
            &CommandSpec::sudo("apt-get", ["install", "-y", "pkg"]),
            true,
        );
        assert_eq!(
            apt.command.display(),
            "sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -q -o Dpkg::Use-Pty=0 install -y pkg"
        );
        let dnf = prepare_invocation(&CommandSpec::sudo("dnf", ["install", "-y", "pkg"]), true);
        assert_eq!(dnf.command.display(), "sudo -n dnf --quiet install -y pkg");
        let zypper = prepare_invocation(
            &CommandSpec::sudo("zypper", ["--non-interactive", "install", "pkg"]),
            true,
        );
        assert!(zypper.command.args.contains(&"--quiet".into()));
    }
}
