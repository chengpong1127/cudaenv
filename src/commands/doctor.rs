use anyhow::Result;

use crate::{providers, ui::output};

pub fn run() -> Result<()> {
    for provider in providers::registered() {
        output::diagnostics(&provider.diagnose()?);
    }
    Ok(())
}
