use anyhow::Result;

use crate::{platform::os, providers, ui::output};

pub fn run() -> Result<()> {
    let system = os::detect()?;
    let statuses = providers::registered()
        .into_iter()
        .map(|provider| provider.inspect())
        .collect::<Result<Vec<_>>>()?;
    output::system_status(&system, &statuses);
    Ok(())
}
