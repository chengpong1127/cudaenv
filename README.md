# cudaenv

`cudaenv` makes it easier to set up an NVIDIA GPU environment on Linux. It
detects your GPU and operating system, chooses the appropriate NVIDIA packages,
shows every planned change, and asks for confirmation before installation.

## Install cudaenv

```bash
curl -LsSf https://raw.githubusercontent.com/chengpong1127/cudaenv/main/install.sh | sh
```

The installer downloads a verified release and places `cudaenv` in
`~/.local/bin`. If that directory is not already in your `PATH`, follow the
instruction printed by the installer.

After installing the command, the script asks whether you want to configure
your GPU environment immediately. You can answer `n` and run `cudaenv install`
later.

## Set up your GPU environment

```bash
cudaenv install
```

The guided setup asks what you want to use the machine for:

```text
Model training (PyTorch, TensorFlow, JAX)
CUDA development
```

Model training configures the system for frameworks that provide their own CUDA
runtime. CUDA development also installs the tools needed to compile CUDA code.

Before changing the system, `cudaenv` displays the detected GPU, operating
system, repository, packages, and commands it plans to use. Nothing is installed
until you confirm.

Useful installation options:

```bash
# Preview the installation without changing the system
cudaenv install --dry-run

# Select model training without showing the usage prompt
cudaenv install --profile model-training

# Select CUDA development without showing the usage prompt
cudaenv install --profile cuda-development

# Install a specific CUDA version
cudaenv install --toolkit 13.1

# Skip the final confirmation for an unattended installation
cudaenv install --profile model-training --yes
```

Driver selection is automatic for recognized GPUs. If the GPU generation cannot
be identified safely, `cudaenv` asks you to check the GPU and rerun with an
explicit choice:

```bash
cudaenv install --driver open
cudaenv install --driver proprietary
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

`cudaenv` detects full, compute-only, desktop-only, branch-pinned, broken, and
unmanaged driver installations. It never installs distribution packages over
a working runfile or otherwise unmanaged driver; migrate or remove that driver
with its original installation method first.

## Check the current environment

```bash
cudaenv status
```

`cudaenv status` reports:

- Detected NVIDIA GPUs
- Whether an NVIDIA driver package is installed
- Whether the driver is loaded and operational
- The installed CUDA development tools version, when available

Already-installed components are skipped when you run `cudaenv install` again.

When repository metadata can be queried reliably, status also reports a newer
compatible driver or Toolkit version. If metadata is unavailable, it reports
only installed state and does not claim the system is current.

## Upgrade installed components

```bash
# Upgrade every supported component that is already installed
cudaenv upgrade

# Select either component, or provide both flags together
cudaenv upgrade --driver
cudaenv upgrade --toolkit
cudaenv upgrade --driver --toolkit

# Preview the exact plan, or skip confirmation
cudaenv upgrade --dry-run
cudaenv upgrade --yes
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

Exact-version Toolkit installations move to the newest compatible exact Toolkit
package side by side. Older Toolkit packages and directories are retained.
`/usr/local/cuda` is updated only when it is a symlink clearly pointing at the
Toolkit being upgraded; a user-managed file or directory is never overwritten.
Unmanaged and NVIDIA runfile driver installations are refused until migrated to
a supported package-manager installation. A driver upgrade requires a reboot;
the loaded driver may continue reporting its pre-upgrade version until then.

## Diagnose problems

```bash
cudaenv doctor

# Require the CUDA development Toolkit checks
cudaenv doctor --profile cuda-development
```

The doctor checks GPU detection, driver installation, driver runtime health,
`nvidia-smi`, exact OS/GPU policy, repository support, matching kernel
development packages, Secure Boot, and complete driver/Toolkit version
compatibility. The default `model-training` profile treats a missing Toolkit as
normal; `cuda-development` treats it as an error. If a Toolkit is present but
partial or broken, both profiles report the fault.

## Uninstall

```bash
cudaenv uninstall
```

Uninstall is currently supported only on Ubuntu. It enumerates every installed
`cuda-*`, `nvidia-*`, `libnvidia-*`, and NVIDIA X driver package, displays the
exact list and purge/autoremove commands, and asks for confirmation. It refuses
to remove unmanaged or runfile installations automatically.

To skip the final confirmation:

```bash
cudaenv uninstall --yes
```

## Supported systems

`cudaenv` supports NVIDIA's official repositories for these Linux families:

- Ubuntu 22.04, 24.04, and 26.04
- Debian 12 and 13
- RHEL, AlmaLinux, and Rocky Linux 8.10, 9.7, and 10.1
- Oracle Linux 8 and 9
- Fedora 44
- Amazon Linux 2023
- Azure Linux 3
- openSUSE Leap 15 SP6 and 16
- SUSE Linux Enterprise Server 15 SP6/SP7 and 16
- KylinOS V11 / V11 2503

Support depends on NVIDIA publishing a repository for the exact operating
system, release, and CPU architecture. `cudaenv` stops instead of substituting a
repository from another distribution.

WSL is detected but driver installation inside WSL is intentionally blocked.
Install the NVIDIA driver on the Windows host; WSL uses the host driver.

Run `cudaenv install` as your normal user. Privileged plan steps use `sudo` when
available. When already running as root, the displayed and executed commands
run directly without `sudo`. If neither root privileges nor `sudo` is
available, cudaenv stops before executing the plan. Status and evidence
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
