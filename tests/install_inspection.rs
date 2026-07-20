mod support;

use arc::providers::nvidia::install::InstallContext;
use support::{FakeSystem, ProviderStatusBuilder, TestOs};

#[test]
fn explicit_system_evidence_builds_install_context_without_host_probes() {
    let status = ProviderStatusBuilder::new()
        .managed_open_driver()
        .toolkit("13.1")
        .build();
    let mut system = FakeSystem::modern_ubuntu(status.clone());
    system.kernel = "6.12.7-test".into();
    system.installed_packages = vec!["nvidia-open".into(), "cuda-toolkit-13-1".into()];
    system.repository_configured = false;
    system.repository_downloader_available = false;
    system.kernel_headers_available = false;

    let context = InstallContext::inspect_with(&TestOs::ubuntu("24.04"), &system).unwrap();

    assert_eq!(context.kernel, "6.12.7-test");
    assert_eq!(context.gpus, system.gpus);
    assert_eq!(context.installed_packages, system.installed_packages);
    assert_eq!(context.status, status);
    assert!(!context.repository_configured);
    assert!(!context.repository_downloader_available);
    assert!(!context.kernel_headers_available);
}
