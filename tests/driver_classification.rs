mod support;

use arc::{
    model::environment::{
        DriverFlavorState, DriverInstallation, DriverPackageScope, UnmanagedDriverEvidence,
    },
    providers::nvidia::state::classify_driver,
};
use support::DriverEvidenceBuilder;

#[test]
fn no_packages_or_runtime_evidence_is_missing() {
    assert_eq!(
        classify_driver(&DriverEvidenceBuilder::new().build()),
        DriverInstallation::Missing
    );
}

#[test]
fn stale_installer_log_alone_is_missing() {
    assert_eq!(
        classify_driver(&DriverEvidenceBuilder::new().installer_log().build()),
        DriverInstallation::Missing
    );
}

#[test]
fn runfile_uninstaller_is_broken_unmanaged_evidence() {
    assert_eq!(
        classify_driver(
            &DriverEvidenceBuilder::new()
                .runfile_uninstaller()
                .installer_log()
                .build()
        ),
        DriverInstallation::Unmanaged {
            working: false,
            evidence: vec![
                UnmanagedDriverEvidence::RunfileUninstaller,
                UnmanagedDriverEvidence::InstallerLog,
            ],
        }
    );
}

#[test]
fn loaded_module_without_packages_is_working_unmanaged() {
    assert_eq!(
        classify_driver(&DriverEvidenceBuilder::new().loaded_module().build()),
        DriverInstallation::Unmanaged {
            working: true,
            evidence: vec![UnmanagedDriverEvidence::LoadedModule],
        }
    );
}

#[test]
fn managed_packages_without_runtime_or_metadata_are_broken() {
    assert_eq!(
        classify_driver(
            &DriverEvidenceBuilder::new()
                .packages(&["cuda-drivers"])
                .build()
        ),
        DriverInstallation::BrokenManaged {
            flavor: DriverFlavorState::Proprietary,
            packages: vec!["cuda-drivers".into()],
        }
    );
}

#[test]
fn compute_only_open_packages_are_classified_semantically() {
    let installation = classify_driver(
        &DriverEvidenceBuilder::new()
            .packages(&["nvidia-driver-cuda", "kmod-nvidia-open-dkms"])
            .runtime_version()
            .module_metadata()
            .build(),
    );

    assert!(matches!(
        installation,
        DriverInstallation::Managed {
            flavor: DriverFlavorState::Open,
            scope: DriverPackageScope::ComputeOnly,
            ..
        }
    ));
}

#[test]
fn desktop_only_proprietary_packages_are_classified_semantically() {
    let installation = classify_driver(
        &DriverEvidenceBuilder::new()
            .packages(&["nvidia-driver"])
            .module_metadata()
            .build(),
    );

    assert!(matches!(
        installation,
        DriverInstallation::Managed {
            flavor: DriverFlavorState::Proprietary,
            scope: DriverPackageScope::DesktopOnly,
            ..
        }
    ));
}

#[test]
fn open_and_proprietary_markers_are_mixed() {
    let installation = classify_driver(
        &DriverEvidenceBuilder::new()
            .packages(&["nvidia-open", "cuda-drivers"])
            .module_metadata()
            .build(),
    );

    assert_eq!(installation.flavor(), Some(DriverFlavorState::Mixed));
}

#[test]
fn driver_pinning_package_records_branch() {
    let installation = classify_driver(
        &DriverEvidenceBuilder::new()
            .packages(&["cuda-drivers", "nvidia-driver-pinning-580"])
            .runtime_version()
            .module_metadata()
            .build(),
    );

    assert!(matches!(
        installation,
        DriverInstallation::Managed {
            branch: Some(580),
            ..
        }
    ));
}

#[test]
fn installed_open_module_waiting_for_reboot_is_managed() {
    let installation = classify_driver(
        &DriverEvidenceBuilder::new()
            .packages(&[
                "nvidia-driver-610-open",
                "nvidia-dkms-610-open",
                "nvidia-kernel-source-610-open",
            ])
            .module_metadata()
            .build(),
    );

    assert!(matches!(
        installation,
        DriverInstallation::Managed {
            flavor: DriverFlavorState::Open,
            ..
        }
    ));
}
