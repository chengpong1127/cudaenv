use anyhow::Result;

use crate::{model::operation::OperationPlan, model::system::OsInfo};

pub use crate::model::profile::InstallProfile;

use super::driver::DriverPreference;

mod decision;
mod inspect;
mod planner;

pub use decision::InstallDecision;
pub use inspect::{InstallContext, InstallSystem, RealInstallSystem};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstallOptions {
    pub profile: InstallProfile,
    pub toolkit_version: Option<String>,
    pub driver: DriverPreference,
}

/// Build an installation plan through the explicit Inspect → Decide → Plan pipeline.
pub fn plan(os: &OsInfo, options: &InstallOptions) -> Result<OperationPlan> {
    let context = InstallContext::inspect(os)?;
    let decision = InstallDecision::decide(&context, options)?;
    planner::plan(&context, &decision, options)
}

/// Plan from already-collected evidence.
///
/// This is the deterministic Decide → Plan seam used by tests and other
/// callers that collect inspection evidence outside this crate.
pub fn plan_from_context(
    context: &InstallContext,
    options: &InstallOptions,
) -> Result<OperationPlan> {
    let decision = InstallDecision::decide(context, options)?;
    planner::plan(context, &decision, options)
}
