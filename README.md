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

## Diagnose problems

```bash
cudaenv doctor
```

The doctor checks GPU detection, driver installation, driver runtime health,
`nvidia-smi`, kernel headers, and Secure Boot compatibility. If it finds a
problem, it prints the likely cause and suggested next action.

## Uninstall

```bash
cudaenv uninstall
```

Uninstall is currently supported on Ubuntu. It displays the exact CUDA and
NVIDIA meta-packages it found and asks for confirmation before removing them.
Dependencies are retained to avoid unexpectedly removing unrelated software.

To skip the final confirmation:

```bash
cudaenv uninstall --yes
```

## Supported systems

`cudaenv` supports NVIDIA's official repositories for these Linux families:

- Ubuntu 22.04, 24.04, and 26.04
- Debian 12 and 13
- RHEL, AlmaLinux, and Rocky Linux 8–10
- Oracle Linux 8 and 9
- Fedora 44
- Amazon Linux 2023
- Azure Linux 3
- openSUSE Leap 15 and 16
- SUSE Linux Enterprise Server 15 and 16
- KylinOS V11

Support depends on NVIDIA publishing a repository for the exact operating
system, release, and CPU architecture. `cudaenv` stops instead of substituting a
repository from another distribution.

WSL is detected but driver installation inside WSL is intentionally blocked.
Install the NVIDIA driver on the Windows host; WSL uses the host driver.

## Safety

- Installation and removal plans are displayed before confirmation.
- `--dry-run` never changes the system.
- Commands are executed directly without shell interpolation.
- Repository and release downloads require HTTPS.
- Release archives are verified using published SHA-256 checksums.
- Unsupported systems and ambiguous GPU generations fail safely.
