# Contributing to BL-1

Thanks for your interest in improving BL-1! This document explains how to set up a
development environment, the quality bar changes must clear, and the conventions we follow.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

BL-1 targets Python 3.10+ (developed and CI-tested on 3.12).

```bash
git clone https://github.com/lau-sam/bl1.git
cd bl1
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,quality]"
```

For GPU acceleration, install the JAX variant matching your CUDA version — see the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quality bar

All of the following must pass before a pull request is merged:

```bash
make lint        # ruff check + format check
make typecheck   # mypy
make test        # pytest (excludes @slow tests)
make quality     # docstring coverage, dead-code, complexity
```

`make all` runs the full suite. Additional project-specific guardrails:

- **Bio-validation must remain 6/6** on the Wagenaar (2006) metrics:
  `bash scripts/run_validation.sh --quick`.
- **Do not modify** `configs/wagenaar_calibrated.yaml` or `configs/wagenaar_burst.yaml`
  without re-running the full validation suite and reporting the results.
- New behaviour needs tests. Slow or GPU-only tests are marked `@slow` and excluded from `make test`.

## Rust workspace

The `rust/` directory hosts a Cargo workspace that ports the forward simulator and provides
a terminal UI. It requires a recent stable Rust toolchain (`rustup update stable`).

```bash
cd rust
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo run -p bl1-tui      # launch the TUI
```

## Coding style

- **Python**: [Ruff](https://docs.astral.sh/ruff/) for lint + format (line length 100),
  [mypy](https://mypy-lang.org/) for types. Run `make lint-fix` to auto-fix.
- **Rust**: `rustfmt` + `clippy` with warnings denied.

## Commit and pull request conventions

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `type(scope): summary` (e.g. `fix(core): correct NMDA Mg²⁺ block constant`).
  Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`.
- Keep each pull request focused on a single concern; reference related issues.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

## Reporting bugs and requesting features

Please use the GitHub issue templates. For scientific/model-accuracy issues, cite the
relevant paper and the equation or metric affected so the discrepancy can be verified.
