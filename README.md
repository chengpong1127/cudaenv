# Arc

**One command for a working NVIDIA GPU environment.**

Setting up NVIDIA GPUs on Linux should not require becoming an expert in
drivers, CUDA Toolkit releases, package repositories, GPU compatibility, and
kernel modules. Arc automatically detects the system, selects compatible
NVIDIA components, and safely configures the environment for model training or
CUDA development.

Arc is a Linux command-line tool. The product is **Arc**; the executable and
all commands use lowercase `arc`.

## Why Arc?

NVIDIA's Linux software stack has several layers that are easy to confuse. A
working driver is required to use the GPU, while the system CUDA Toolkit and
`nvcc` are needed only to compile native CUDA code. The correct packages also
depend on the Linux distribution, CPU architecture, GPU generation, kernel,
and existing installation state.

Arc turns those decisions into a guided workflow:

- Detect NVIDIA GPUs, Linux distribution, architecture, kernel, driver state,
  CUDA Toolkit installations, and active `nvcc`.
- Choose compatible packages from official NVIDIA repository targets.
- Configure for either model training or native CUDA development.
- Show the complete installation plan before making changes.
- Diagnose common failures with specific next steps.

## Installation

Install the latest verified release:

```bash
curl -LsSf https://raw.githubusercontent.com/chengpong1127/arc/main/install.sh | sh
```

The installer verifies the release checksum and places `arc` in
`~/.local/bin`. If needed, follow its prompt to add that directory to `PATH`.
The installer can start GPU setup immediately, or you can run `arc install`
later.

Run Arc as your normal user. It uses `sudo` for privileged plan steps when
available and stops before execution if the required privileges are missing.

## Quick Start

Start with a read-only view of the machine:

```bash
arc status
arc doctor
```

Then launch the guided installer:

```bash
arc install
```

Arc asks which environment you need:

| Profile | Use it for | What Arc requires |
| --- | --- | --- |
| `model-training` | PyTorch, TensorFlow, or JAX | An operational NVIDIA driver. Frameworks provide their own CUDA runtime and are installed separately. |
| `cuda-development` | Native CUDA applications and custom kernels | An operational NVIDIA driver, a system CUDA Toolkit, and working `nvcc`. |

Preview an installation without changing the system, or select a profile for
an unattended workflow:

```bash
arc install --dry-run
arc install --profile model-training --yes
arc install --profile cuda-development --toolkit 13.1 --yes
```

After installing or upgrading a driver, reboot if Arc requests it. Then verify
the result:

```bash
nvidia-smi
arc status
```

## How It Works

1. **Inspect:** Arc detects hardware, the operating system, installed packages,
   driver runtime health, repositories, the system CUDA Toolkit, and `nvcc`.
2. **Decide:** It applies GPU-generation, operating-system, and
   driver/Toolkit compatibility rules to select a safe configuration.
3. **Plan:** Arc displays the repository, packages, and operations it intends
   to use. Already-satisfied components are skipped.
4. **Confirm and execute:** Nothing is installed until you approve the plan.
5. **Verify:** `arc status` summarizes readiness, while `arc doctor` checks
   failures and recommends concrete repairs.

Driver selection is automatic for recognized GPUs. Turing and newer GPUs
default to the latest compatible NVIDIA open kernel modules where supported.
Maxwell, Pascal, and Volta use the proprietary R580 branch and CUDA 12.x; mixed
modern and legacy systems follow the legacy policy so every GPU remains
supported. If Arc cannot identify a GPU generation safely, it stops and asks
for an explicit choice:

```bash
arc install --driver open
arc install --driver proprietary
```

Arc distinguishes package-managed installations from unmanaged or NVIDIA
runfile installations. It does not install distribution packages over a
working unmanaged driver; migrate or remove that driver with its original
installation method first.

## Built with Codex and GPT-5.6

Arc was built with Codex and GPT-5.6 as active development collaborators throughout the project.

### How Codex was used

Codex helped turn the project design into a working Rust CLI. It was used to:

- Design and refactor Arc's module architecture.
- Implement GPU, driver, operating-system, repository, and CUDA Toolkit detection.
- Build installation, upgrade, uninstall, status, and diagnostic workflows.
- Generate and improve unit and integration tests.
- Review error handling, unsafe installation states, and edge cases.
- Refine CLI output, prompts, operation plans, and user-facing diagnostics.

For example, Codex helped restructure Arc's driver detection into explicit states for missing, package-managed, broken, and unmanaged installations. This allows Arc to avoid installing distribution packages over an NVIDIA runfile installation.

### How GPT-5.6 was used

GPT-5.6 was used primarily for technical reasoning, research synthesis, and product design. It helped:

- Analyze NVIDIA's Linux installation documentation and repository structure.
- Distinguish driver-only machine-learning environments from full CUDA development environments.
- Design compatibility policies for GPU generations, driver branches, CUDA Toolkit versions, Linux distributions, and CPU architectures.
- Identify failure cases involving Secure Boot, kernel headers, mixed GPU generations, unmanaged drivers, and incompatible Toolkit versions.
- Review the CLI workflow from a user's perspective and simplify decisions that normally require NVIDIA-specific knowledge.

### Human validation

AI-generated suggestions were not accepted blindly. The architecture, compatibility rules, commands, and diagnostics were reviewed against official NVIDIA documentation and tested on real Linux systems.

Arc does not call Codex or GPT-5.6 at runtime. All hardware detection, compatibility decisions, installation planning, and command execution happen locally in the Rust CLI.

## Supported Systems

Arc uses official NVIDIA repository targets. When NVIDIA publishes one target
for a distribution family, such as `rhel9`, compatible minor releases in that
family resolve to it. Arc never substitutes a target from another distribution
family.

| Distribution family | Compatible repository releases | NVIDIA validated | Architectures |
| --- | --- | --- | --- |
| Ubuntu | 22.04, 24.04, 26.04 | 22.04, 24.04, 26.04 | x86_64, sbsa |
| Debian | 12.x, 13.x | 12, 13 | x86_64 |
| Red Hat Enterprise Linux | 8.x, 9.x, 10.x | 8.10, 9.8, 10.2 | x86_64, sbsa |
| AlmaLinux | 8.x, 9.x, 10.x | 8.10, 9.8, 10.2 | x86_64 |
| Rocky Linux | 8.x, 9.x, 10.x | 8.10, 9.8, 10.2 | x86_64 |
| Oracle Linux | 8.x, 9.x | 8, 9 | x86_64 |
| Fedora | 44 | 44 | x86_64 |
| Amazon Linux | 2023.x | 2023 | x86_64, sbsa |
| Azure Linux | 3.x | 3.0 | x86_64, sbsa |
| openSUSE Leap | 15.6, 16.0 | 15.6 | x86_64 |
| SLES | 15.6+, 16.0+ | 15.6, 15.7, 16.0 | x86_64, sbsa |
| KylinOS | V11, V11 2503 | V11, V11 2503 | x86_64, sbsa |

Unsupported distribution families, repository targets, and CPU architectures
are rejected. WSL is detected, but driver installation inside WSL is blocked;
install the NVIDIA driver on the Windows host instead.

## Commands

### `arc status`

Reports detected GPUs, driver installation and runtime state, system-managed
CUDA Toolkits, the active `nvcc`, and readiness for the selected profile.

```bash
arc status
arc status --profile cuda-development
```

### `arc doctor`

Checks GPU detection, driver and `nvidia-smi` health, OS/GPU policy, repository
support, running-kernel development packages, Secure Boot, and driver/Toolkit
compatibility. It provides repair steps but does not automatically repair
unmanaged or runfile installations.

```bash
arc doctor
arc doctor --profile cuda-development
```

### `arc install`

Builds and executes a safety-first installation plan. Use `--dry-run` to
preview it, `--profile` to skip the workload prompt, `--toolkit` to select a
CUDA Toolkit version, and `--yes` for unattended confirmation.

```bash
arc install
arc install --dry-run
arc install --profile model-training
arc install --profile cuda-development --toolkit 12.8
```

### `arc upgrade`

Upgrades installed, supported components to their latest compatible versions.
It does not add an absent Toolkit, change the installed driver flavor, or run
an unrestricted system upgrade.

```bash
arc upgrade
arc upgrade --driver
arc upgrade --toolkit
arc upgrade --dry-run
```

### `arc uninstall`

Plans removal of NVIDIA driver and CUDA Toolkit packages, shows the exact
package list, and asks for confirmation. Uninstall is currently supported only
on Ubuntu and refuses to remove unmanaged or runfile installations.

```bash
arc uninstall
arc uninstall --yes
```

Global flags include `--verbose` (`-v`) for streamed command output and
`--show-commands` for exact commands in operation plans.

## Safety

- Installation, upgrade, and removal plans are displayed before confirmation.
- `--dry-run` never changes the system.
- Commands run directly without shell interpolation.
- Repository and release downloads require HTTPS, and release archives are
  verified with published SHA-256 checksums.
- Unsupported systems, ambiguous GPUs, incompatible mixed generations, and
  unsafe unmanaged installations fail safely.
- Arc installs prerequisites for the running kernel before the driver package.
- Driver flavor and branch transitions are explicit and shown separately in
  the plan.
- System package inventory—not a Conda environment, custom `PATH`, or
  user-installed `nvcc`—determines installation and upgrade decisions.
