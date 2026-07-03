## Summary

<!-- What does this PR change and why? Reference related issues (e.g. Closes #123). -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Model/numerical correction (cite paper + equation)
- [ ] Documentation
- [ ] Refactor / chore / CI

## Checklist

- [ ] `make lint` and `make typecheck` pass
- [ ] `make test` passes (new behaviour is covered by tests)
- [ ] Bio-validation remains 6/6 (`bash scripts/run_validation.sh --quick`) if simulation code changed
- [ ] `configs/wagenaar_calibrated.yaml` / `configs/wagenaar_burst.yaml` were **not** modified (or full validation was re-run and results reported)
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for user-visible changes
- [ ] Rust changes pass `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`

## Scientific justification (if applicable)

<!-- For model/parameter changes: which paper, which equation, and what validation confirms it. -->
