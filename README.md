# arc

`arc` makes it easier to set up an NVIDIA GPU environment on Linux. It
detects your GPU and operating system, chooses the appropriate NVIDIA packages,
shows every planned change, and asks for confirmation before installation.

## Install arc

```bash
curl -LsSf https://raw.githubusercontent.com/chengpong1127/arc/main/install.sh | sh
```

The installer downloads a verified release and places `arc` in
`~/.local/bin`. If that directory is not already in your `PATH`, follow the
instruction printed by the installer.

After installing the command, the script asks whether you want to configure
your GPU environment immediately. You can answer `n` and run `arc install`
later.

## Set up your GPU environment

```bash
arc install
```

The guided setup asks what you want to use the machine for:

```text
AI Model Training       PyTorch, TensorFlow, or JAX
CUDA Development        Native CUDA apps and custom kernels
```

Model training configures the system for frameworks that provide their own CUDA
runtime. CUDA development also installs the tools needed to compile CUDA code.

Before changing the system, `arc` displays the detected GPU, operating
system, repository, packages, and commands it plans to use. Nothing is installed
until you confirm.

Useful installation options:

```bash
# Preview the installation without changing the system
arc install --dry-run

# Select model training without showing the usage prompt
arc install --profile model-training

# Select CUDA development without showing the usage prompt
arc install --profile cuda-development

# Install a specific CUDA version
arc install --toolkit 13.1

# Skip the final confirmation for an unattended installation
arc install --profile model-training --yes
```

Driver selection is automatic for recognized GPUs. If the GPU generation cannot
be identified safely, `arc` asks you to check the GPU and rerun with an
explicit choice:

```bash
arc install --driver open
arc install --driver proprietary
```

The support policy is generation-aware:

- Turing and newer GPUs default to the latest NVIDIA open kernel modules.
- Maxwell, Pascal, and Volta GPUs use proprietary modules pinned to the R580
  branch. They cannot use CUDA 13.x; CUDA development requires an explicit
  CUDA 12.x selection such as `--toolkit 12.8`.
- A machine containing both modern and legacy GPUs follows the legacy policy so
  every installed GPU remains supported.
- `--driver open` is rejected when any detected GPU is Maxwell, Pascal, or
  Volta. Unknown GPUs require an explicit choice, and OS-specific restrictions
  are still enforced.

`arc` detects full, compute-only, desktop-only, branch-pinned, broken, and
unmanaged driver installations. It never installs distribution packages over
a working runfile or otherwise unmanaged driver; migrate or remove that driver
with its original installation method first.

## Check the current environment

```bash
arc status
```

`arc status` reports:

- Detected NVIDIA GPUs
- Whether an NVIDIA driver package is installed
- Whether the driver is loaded and operational
- System package-manager CUDA Toolkit installations and the packages that
  prove they are manageable by arc
- The active `nvcc` version and executable path separately, as informational
  PATH state

A Conda environment, environment module, custom `PATH`, or user-installed
`nvcc` is never treated as proof that an APT, RPM, or Zypper Toolkit package is
installed. Install and upgrade decisions use system package inventory; the
active compiler can therefore be reported while the system-managed Toolkit is
reported as absent.

Already-installed components are skipped when you run `arc install` again.

When repository metadata can be queried reliably, status also reports a newer
compatible driver or Toolkit version. If metadata is unavailable, it reports
only installed state and does not claim the system is current.

## Upgrade installed components

```bash
# Upgrade every supported component that is already installed
arc upgrade

# Select either component, or provide both flags together
arc upgrade --driver
arc upgrade --toolkit
arc upgrade --driver --toolkit

# Preview the exact plan, or skip confirmation
arc upgrade --dry-run
arc upgrade --yes
```

“Latest compatible” means the newest candidate published by the configured
NVIDIA repository that satisfies the detected GPU generation, operating system,
architecture, installed driver flavor and branch restrictions, Toolkit tracking
boundary, and driver/Toolkit compatibility policy. It does not mean the package
with the highest version number regardless of those constraints.

Upgrade changes only components already present. A machine without a system
CUDA Toolkit will not gain one, and selecting only absent components is a
successful no-op. Driver upgrades preserve the installed open or proprietary
kernel-module flavor; changing flavor is an explicit install/migration operation.
Maxwell, Pascal, and Volta remain on proprietary R580 and CUDA 12.x. Unknown GPU
generations and incompatible mixed-generation systems fail safely.

`--driver` is strictly package-scoped: APT uses `install --only-upgrade` for the
detected driver package, while DNF, TDNF, and Zypper receive that exact package
as the update target. Dependencies required by the package manager may change,
but arc never plans a distribution upgrade or an unrestricted system update
as part of a driver-only upgrade.

Exact-version Toolkit installations move to the newest compatible exact Toolkit
package side by side. Older Toolkit packages and directories are retained.
`/usr/local/cuda` is updated only when it is a symlink clearly pointing at the
Toolkit being upgraded; a user-managed file or directory is never overwritten.
Unmanaged and NVIDIA runfile driver installations are refused until migrated to
a supported package-manager installation. A driver upgrade requires a reboot;
the loaded driver may continue reporting its pre-upgrade version until then.

## Diagnose problems

```bash
arc doctor

# Require the CUDA development Toolkit checks
arc doctor --profile cuda-development
```

The doctor checks GPU detection, driver installation, driver runtime health,
`nvidia-smi`, exact OS/GPU policy, repository support, matching kernel
development packages, Secure Boot, and complete driver/Toolkit version
compatibility. The default `model-training` profile treats a missing Toolkit as
normal; `cuda-development` treats it as an error. If a Toolkit is present but
partial or broken, both profiles report the fault.

When Secure Boot is enabled and a managed DKMS driver is present but not loaded,
doctor checks whether the local module-signing key is enrolled. If it is not,
the fix plan shows the exact `mokutil` import, forced DKMS rebuild, MOK enrollment,
and reboot sequence instead of treating the driver as a generic load failure.

For a broken package-managed driver, doctor provides a concrete repair plan:
install the development packages matching the running kernel, reinstall the
exact detected NVIDIA packages with the native package manager, rebuild DKMS
when applicable, verify module metadata, and reboot. Doctor does not
automatically repair unmanaged or NVIDIA runfile installations; it instead
prints clear steps for using the original installation method. CUDA symlink
advice always includes executable commands or an explicit manual target-selection
step.

## Uninstall

```bash
arc uninstall
```

Uninstall is currently supported only on Ubuntu. It enumerates every installed
`cuda-*`, `nvidia-*`, `libnvidia-*`, and NVIDIA X driver package, displays the
exact list and purge/autoremove commands, and asks for confirmation. It refuses
to remove unmanaged or runfile installations automatically.

To skip the final confirmation:

```bash
arc uninstall --yes
```

## Supported systems

`arc` resolves only official NVIDIA repository targets. Repository
compatibility is separate from NVIDIA's current validation matrix and from the
releases exercised by arc's maintainers. When NVIDIA publishes one target
for a major family (for example `rhel9` or `rhel10`), compatible newer minor
releases in that same family resolve to it. arc never substitutes a target
from another distribution family.

| Distribution family | Compatible repository releases | NVIDIA validated | Tested by arc | Architectures |
| --- | --- | --- | --- | --- |
| Ubuntu | 22.04, 24.04, 26.04 | 22.04, 24.04, 26.04 | 24.04 | x86_64, sbsa |
| Debian | 12.x, 13.x | 12, 13 | 12, 13 | x86_64 |
| RHEL / AlmaLinux / Rocky Linux | 8.x, 9.x, 10.x | 8.10, 9.7, 10.1 | 8.10, 9.7, 10.1 | x86_64, sbsa |
| Oracle Linux | 8.x, 9.x | 8, 9 | 8, 9 | x86_64 |
| Fedora | 44 | 44 | 44 | x86_64 |
| Amazon Linux | 2023.x | 2023 | 2023 | x86_64, sbsa |
| Azure Linux | 3.x | 3.0 | 3.0 | x86_64, sbsa |
| openSUSE Leap | 15.x, 16.x | 15.6, 16.0 | 15.6, 16.0 | x86_64 |
| SLES | 15.x, 16.x | 15.6, 15.7, 16.0 | 15.6, 15.7, 16.0 | x86_64, sbsa |
| KylinOS | V11, V11 2503 | V11, V11 2503 | V11 2503 | x86_64, sbsa |

Unsupported major families, distribution-specific targets that NVIDIA does not
publish, and unsupported CPU architectures are rejected.

WSL is detected but driver installation inside WSL is intentionally blocked.
Install the NVIDIA driver on the Windows host; WSL uses the host driver.

Run `arc install` as your normal user. Privileged plan steps use `sudo` when
available. When already running as root, the displayed and executed commands
run directly without `sudo`. If neither root privileges nor `sudo` is
available, arc stops before executing the plan. Status and evidence
collection remain unprivileged wherever the operating system permits it.

## Safety

- Installation and removal plans are displayed before confirmation.
- `--dry-run` never changes the system.
- Commands are executed directly without shell interpolation.
- Repository and release downloads require HTTPS.
- Release archives are verified using published SHA-256 checksums.
- Unsupported systems and ambiguous GPU generations fail safely.
- Matching running-kernel prerequisites and distro dependency repositories are
  installed before the driver package.
- Driver flavor or branch changes use explicit package-manager transition
  operations and are shown separately in the plan.
