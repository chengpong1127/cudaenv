mod support;

use arc::{
    model::{environment::DriverFlavorState, operation::NextStep},
    providers::nvidia::{driver::DriverPreference, install::plan_from_context},
};
use support::{
    InstallContextBuilder, InstallOptionsBuilder, ProviderStatusBuilder, TestGpu,
    assert_command_before, assert_noop_with_next_step, assert_stage_titles,
};

#[test]
fn clean_install_has_ordered_driver_stages() {
    let context =
        InstallContextBuilder::ubuntu(ProviderStatusBuilder::new().missing_driver().build())
            .build();
    let plan =
        plan_from_context(&context, &InstallOptionsBuilder::model_training().build()).unwrap();

    assert_stage_titles(
        &plan,
        &[
            "Refresh package metadata",
            "Install driver prerequisites",
            "Install the NVIDIA Open driver",
            "Verify the installation",
        ],
    );
    assert_eq!(plan.next_step, Some(NextStep::LoadNvidiaDriver));
}

#[test]
fn already_correct_install_is_a_noop() {
    let status = ProviderStatusBuilder::new()
        .managed_open_driver()
        .toolkit("13.1")
        .build();
    let context = InstallContextBuilder::ubuntu(status)
        .installed_packages(&["cuda-toolkit-13-1"])
        .build();
    let plan = plan_from_context(&context, &InstallOptionsBuilder::cuda("13.1").build()).unwrap();

    assert_noop_with_next_step(&plan, None);
}

#[test]
fn toolkit_only_install_does_not_schedule_driver_commands() {
    let context =
        InstallContextBuilder::ubuntu(ProviderStatusBuilder::new().managed_open_driver().build())
            .build();
    let plan = plan_from_context(&context, &InstallOptionsBuilder::cuda("13.1").build()).unwrap();

    assert_stage_titles(
        &plan,
        &[
            "Refresh package metadata",
            "Install the CUDA Toolkit",
            "Verify the CUDA Toolkit",
        ],
    );
    assert!(plan.steps.iter().all(|step| {
        !step
            .command
            .args
            .iter()
            .any(|arg| arg == "nvidia-open" || arg == "cuda-drivers")
    }));
}

#[test]
fn driver_flavor_transition_uses_the_requested_package_stream() {
    let context = InstallContextBuilder::ubuntu(
        ProviderStatusBuilder::new()
            .managed_proprietary_driver()
            .build(),
    )
    .build();
    let options = InstallOptionsBuilder::model_training()
        .driver(DriverPreference::Open)
        .build();
    let plan = plan_from_context(&context, &options).unwrap();

    assert!(plan.steps.iter().any(|step| {
        step.command.program == "sudo" && step.command.args.iter().any(|arg| arg == "nvidia-open")
    }));
}

#[test]
fn driver_branch_transition_replaces_the_existing_pin() {
    let context = InstallContextBuilder::ubuntu(
        ProviderStatusBuilder::new()
            .managed_driver(DriverFlavorState::Open, Some(570))
            .build(),
    )
    .gpus(vec![TestGpu::legacy()])
    .build();
    let plan =
        plan_from_context(&context, &InstallOptionsBuilder::model_training().build()).unwrap();

    assert!(plan.steps.iter().any(|step| {
        step.command
            .args
            .iter()
            .any(|arg| arg == "nvidia-driver-pinning-580")
    }));
}

#[test]
fn installed_driver_waiting_for_reboot_is_noop_with_guidance() {
    let context = InstallContextBuilder::ubuntu(
        ProviderStatusBuilder::new()
            .managed_open_driver()
            .waiting_for_reboot()
            .build(),
    )
    .build();
    let plan =
        plan_from_context(&context, &InstallOptionsBuilder::model_training().build()).unwrap();

    assert_noop_with_next_step(&plan, Some(NextStep::LoadNvidiaDriver));
}

#[test]
fn broken_managed_driver_is_reinstalled_and_dkms_is_rebuilt() {
    let context = InstallContextBuilder::ubuntu(
        ProviderStatusBuilder::new()
            .broken_managed_open(&["nvidia-open", "nvidia-dkms-610-open"])
            .build(),
    )
    .build();
    let plan =
        plan_from_context(&context, &InstallOptionsBuilder::model_training().build()).unwrap();

    assert!(plan.steps.iter().any(|step| {
        step.command.args.windows(3).any(|args| {
            args == ["--reinstall", "-y", "nvidia-open"]
                || args == ["-y", "nvidia-open", "nvidia-dkms-610-open"]
        })
    }));
    assert!(plan.steps.iter().any(|step| {
        step.command
            .args
            .windows(3)
            .any(|args| args == ["autoinstall", "-k", "6.8.0-generic"])
    }));
}

#[test]
fn mixed_package_installation_is_refused() {
    let context =
        InstallContextBuilder::ubuntu(ProviderStatusBuilder::new().mixed_driver().build()).build();
    let error =
        plan_from_context(&context, &InstallOptionsBuilder::model_training().build()).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("Conflicting open and proprietary")
    );
}

#[test]
fn working_and_broken_unmanaged_drivers_are_refused() {
    for working in [true, false] {
        let context = InstallContextBuilder::ubuntu(
            ProviderStatusBuilder::new()
                .unmanaged_driver(working)
                .build(),
        )
        .build();
        let error = plan_from_context(&context, &InstallOptionsBuilder::model_training().build())
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unmanaged NVIDIA driver installation")
        );
    }
}

#[test]
fn legacy_gpu_pins_r580_and_rejects_cuda_13() {
    let status = ProviderStatusBuilder::new().missing_driver().build();
    let context = InstallContextBuilder::ubuntu(status)
        .gpus(vec![TestGpu::legacy()])
        .build();
    let cuda_12 =
        plan_from_context(&context, &InstallOptionsBuilder::cuda("12.8").build()).unwrap();

    assert!(cuda_12.steps.iter().any(|step| {
        step.command
            .args
            .iter()
            .any(|arg| arg == "nvidia-driver-pinning-580")
    }));
    assert!(plan_from_context(&context, &InstallOptionsBuilder::cuda("13.1").build()).is_err());
}

#[test]
fn incompatible_driver_is_updated_before_toolkit() {
    let context = InstallContextBuilder::ubuntu(
        ProviderStatusBuilder::new()
            .managed_open_driver()
            .driver_version("570.26")
            .build(),
    )
    .build();
    let plan = plan_from_context(&context, &InstallOptionsBuilder::cuda("13.3").build()).unwrap();

    assert_command_before(&plan, "nvidia-open", "cuda-toolkit-13-3");
}

#[test]
fn configured_repository_is_not_configured_again() {
    let context =
        InstallContextBuilder::ubuntu(ProviderStatusBuilder::new().missing_driver().build())
            .repository_configured(true)
            .build();
    let plan =
        plan_from_context(&context, &InstallOptionsBuilder::model_training().build()).unwrap();

    assert!(
        plan.steps
            .iter()
            .all(|step| step.stage.title != "Configure the NVIDIA CUDA repository")
    );
}

#[test]
fn missing_repository_is_configured_before_metadata_refresh() {
    let context =
        InstallContextBuilder::ubuntu(ProviderStatusBuilder::new().missing_driver().build())
            .repository_configured(false)
            .build();
    let plan =
        plan_from_context(&context, &InstallOptionsBuilder::model_training().build()).unwrap();

    let titles = plan
        .steps
        .iter()
        .map(|step| step.stage.title.as_str())
        .collect::<Vec<_>>();
    let repository = titles
        .iter()
        .position(|title| *title == "Configure the NVIDIA CUDA repository")
        .unwrap();
    let refresh = titles
        .iter()
        .position(|title| *title == "Refresh package metadata")
        .unwrap();
    assert!(repository < refresh);
}

#[test]
fn unavailable_repository_downloader_stops_planning() {
    let context =
        InstallContextBuilder::ubuntu(ProviderStatusBuilder::new().missing_driver().build())
            .repository_configured(false)
            .downloader_available(false)
            .build();
    let error =
        plan_from_context(&context, &InstallOptionsBuilder::model_training().build()).unwrap_err();

    assert!(error.to_string().contains("requires curl or wget"));
}

#[test]
fn unmanaged_active_nvcc_does_not_suppress_system_toolkit_install() {
    let context = InstallContextBuilder::ubuntu(
        ProviderStatusBuilder::new()
            .managed_open_driver()
            .active_unmanaged_toolkit("13.1")
            .build(),
    )
    .build();
    let plan = plan_from_context(&context, &InstallOptionsBuilder::cuda("13.1").build()).unwrap();

    assert!(plan.steps.iter().any(|step| {
        step.command
            .args
            .iter()
            .any(|arg| arg == "cuda-toolkit-13-1")
    }));
}
