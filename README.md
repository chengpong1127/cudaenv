# cudaenv

`cudaenv` inspects NVIDIA GPU environments and installs NVIDIA drivers from
official NVIDIA repositories. Driver installation supports Ubuntu, Debian,
RHEL, AlmaLinux, Rocky Linux, Oracle Linux, Fedora, Amazon Linux, Azure Linux,
openSUSE, SLES, and KylinOS. WSL is intentionally rejected because its NVIDIA
driver must be installed on the Windows host.

Repository targets are resolved from the exact distribution, release, and CPU
architecture. If NVIDIA does not publish that exact target, `cudaenv` stops
instead of borrowing another distribution's repository.

## Build and test

```bash
cargo build
cargo test
```

## Guided installation

```bash
cargo run -- install
cargo run -- install --profile model-training
cargo run -- install --profile cuda-development
cargo run -- install --toolkit 13.1
cargo run -- install --profile cuda-development --dry-run
```

With no `--profile`, `install` asks whether the machine is for model training or
CUDA development. Model training installs only the NVIDIA driver. CUDA
development installs the driver and NVIDIA's latest stable CUDA Toolkit.

Driver flavor selection is automatic. Unidentified GPUs default to open kernel
modules without another prompt. The optional `--driver` flag can still provide
an explicit override.

Every install prints the full repository and package command plan first. It asks
for confirmation unless `--yes` is supplied; `--dry-run` never changes the
system. For CUDA development, the unversioned `cuda-toolkit` meta-package tracks
the latest stable toolkit in NVIDIA's repository. An optional pin such as
`--toolkit 13.1` selects `cuda-toolkit-13-1` and implies the CUDA development
profile. The network repository is configured only when needed, package
availability is checked before installation, and `nvcc --version` verifies the
result.

Other inspection commands remain available:

```bash
cargo run -- status
cargo run -- doctor
cargo run -- uninstall
cargo run -- uninstall --yes
```

`status` reports both the loaded NVIDIA driver version and active CUDA Toolkit
version. Install and uninstall plans use that status: already-installed
components are skipped, and uninstall only removes components detected as
present.
