use anyhow::Result;

use crate::{
    system::{environment, gpu, os},
    ui::output,
};

pub fn run() -> Result<()> {
    let os = os::detect()?;
    let gpus = gpu::detect()?;
    let status = environment::detect()?;
    let gpu_summary = (!gpus.is_empty()).then(|| {
        gpus.iter()
            .map(|gpu| gpu.name.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    });

    output::system_status(
        &os,
        gpu_summary.as_deref(),
        status.driver_version.as_deref(),
        status.toolkit_version.as_deref(),
    );
    Ok(())
}
