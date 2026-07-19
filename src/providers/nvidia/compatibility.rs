use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compatibility {
    Incompatible,
    MinorVersionCompatible,
    BundledDriverOrNewer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolkitCompatibility {
    pub toolkit: &'static str,
    pub minimum_driver: &'static str,
    pub bundled_driver: &'static str,
}

// NVIDIA CUDA release-note data. `minimum_driver` is the Linux minimum for
// minor-version compatibility; `bundled_driver` is the release's development
// driver and is a recommendation, not a hard runtime requirement.
pub const TABLE: &[ToolkitCompatibility] = &[
    ToolkitCompatibility {
        toolkit: "13.3",
        minimum_driver: "580.65.06",
        bundled_driver: "610.43.02",
    },
    ToolkitCompatibility {
        toolkit: "13.2",
        minimum_driver: "580.65.06",
        bundled_driver: "595.45.04",
    },
    ToolkitCompatibility {
        toolkit: "13.1",
        minimum_driver: "580.65.06",
        bundled_driver: "590.44.01",
    },
    ToolkitCompatibility {
        toolkit: "13.0",
        minimum_driver: "580.65.06",
        bundled_driver: "580.65.06",
    },
    ToolkitCompatibility {
        toolkit: "12.9",
        minimum_driver: "525.60.13",
        bundled_driver: "575.51.03",
    },
    ToolkitCompatibility {
        toolkit: "12.8",
        minimum_driver: "525.60.13",
        bundled_driver: "570.26",
    },
    ToolkitCompatibility {
        toolkit: "12.6",
        minimum_driver: "525.60.13",
        bundled_driver: "560.28.03",
    },
    ToolkitCompatibility {
        toolkit: "12.5",
        minimum_driver: "525.60.13",
        bundled_driver: "555.42.02",
    },
    ToolkitCompatibility {
        toolkit: "12.4",
        minimum_driver: "525.60.13",
        bundled_driver: "550.54.14",
    },
    ToolkitCompatibility {
        toolkit: "12.3",
        minimum_driver: "525.60.13",
        bundled_driver: "545.23.06",
    },
    ToolkitCompatibility {
        toolkit: "12.2",
        minimum_driver: "525.60.13",
        bundled_driver: "535.54.03",
    },
    ToolkitCompatibility {
        toolkit: "12.1",
        minimum_driver: "525.60.13",
        bundled_driver: "530.30.02",
    },
    ToolkitCompatibility {
        toolkit: "12.0",
        minimum_driver: "525.60.13",
        bundled_driver: "525.60.13",
    },
    ToolkitCompatibility {
        toolkit: "11.8",
        minimum_driver: "450.80.02",
        bundled_driver: "520.61.05",
    },
    ToolkitCompatibility {
        toolkit: "11.7",
        minimum_driver: "450.80.02",
        bundled_driver: "515.43.04",
    },
    ToolkitCompatibility {
        toolkit: "11.6",
        minimum_driver: "450.80.02",
        bundled_driver: "510.39.01",
    },
    ToolkitCompatibility {
        toolkit: "11.5",
        minimum_driver: "450.80.02",
        bundled_driver: "495.29.05",
    },
    ToolkitCompatibility {
        toolkit: "11.4",
        minimum_driver: "450.80.02",
        bundled_driver: "470.42.01",
    },
    ToolkitCompatibility {
        toolkit: "11.3",
        minimum_driver: "450.80.02",
        bundled_driver: "465.19.01",
    },
    ToolkitCompatibility {
        toolkit: "11.2",
        minimum_driver: "450.80.02",
        bundled_driver: "460.27.04",
    },
    ToolkitCompatibility {
        toolkit: "11.1",
        minimum_driver: "450.80.02",
        bundled_driver: "455.23",
    },
];

pub fn evaluate(driver: &str, toolkit: &str) -> Option<Compatibility> {
    let toolkit = major_minor(toolkit)?;
    let release = TABLE.iter().find(|entry| entry.toolkit == toolkit)?;
    if compare_versions(driver, release.minimum_driver) == Ordering::Less {
        Some(Compatibility::Incompatible)
    } else if compare_versions(driver, release.bundled_driver) == Ordering::Less {
        Some(Compatibility::MinorVersionCompatible)
    } else {
        Some(Compatibility::BundledDriverOrNewer)
    }
}

pub fn major_minor(version: &str) -> Option<String> {
    let mut parts = version.split(['.', '-']);
    Some(format!(
        "{}.{}",
        parts.next()?.parse::<u32>().ok()?,
        parts.next()?.parse::<u32>().ok()?
    ))
}

pub fn compare_versions(left: &str, right: &str) -> Ordering {
    let parse = |version: &str| {
        version
            .split(['.', '-'])
            .map(|part| part.parse::<u32>().unwrap_or(0))
            .collect::<Vec<_>>()
    };
    let left = parse(left);
    let right = parse(right);
    for index in 0..left.len().max(right.len()) {
        match left
            .get(index)
            .copied()
            .unwrap_or(0)
            .cmp(&right.get(index).copied().unwrap_or(0))
        {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_complete_versions_at_compatibility_boundaries() {
        assert_eq!(
            evaluate("525.60.12", "12.8"),
            Some(Compatibility::Incompatible)
        );
        assert_eq!(
            evaluate("525.60.13", "12.8"),
            Some(Compatibility::MinorVersionCompatible)
        );
        assert_eq!(
            evaluate("570.25.99", "12.8"),
            Some(Compatibility::MinorVersionCompatible)
        );
        assert_eq!(
            evaluate("570.26", "12.8"),
            Some(Compatibility::BundledDriverOrNewer)
        );
        assert_eq!(
            evaluate("580.65.05", "13.0"),
            Some(Compatibility::Incompatible)
        );
        assert_eq!(
            evaluate("580.65.06", "13.0"),
            Some(Compatibility::BundledDriverOrNewer)
        );
    }
}
