use anyhow::Result;

use crate::{platform::os, providers, providers::nvidia::upgrade, ui::output};

pub fn run() -> Result<()> {
    let system = os::detect()?;
    let statuses = providers::registered()
        .into_iter()
        .map(|provider| provider.inspect())
        .collect::<Result<Vec<_>>>()?;
    // Availability is best effort: stale, missing, or inaccessible repository
    // metadata must not turn an otherwise useful status report into a failure.
    let upgrades = upgrade::availability(&system).ok();
    output::system_status(&system, &statuses, upgrades.as_ref());
    Ok(())
}
