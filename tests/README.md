# Test architecture

Tests are organized by the boundary they verify:

- Keep small unit tests beside private, pure functions such as parsers, version comparisons,
  classifiers, and command constructors.
- Put public policy, planning, execution, and CLI contracts in behavior-oriented files under
  `tests/`. Do not mirror the source tree mechanically.
- Build domain evidence with the focused builders in `tests/support`; defaults represent a
  modern Ubuntu host and each test overrides only evidence relevant to its scenario.
- Use `FakeCommandRunner` for execution tests. Configure every result explicitly and assert its
  recorded `CommandInvocation` values. Tests must not invoke package managers, inspect host GPU
  hardware, require root, or mutate global environment variables.
- Prefer semantic assertions against command programs and arguments, plan stages, domain enums,
  ordering, and next steps. Assert rendered prose only when wording is itself a user contract.

Before submitting changes, run:

```console
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```
