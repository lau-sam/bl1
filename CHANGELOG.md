# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LICENSE` file (MIT), previously only declared in `pyproject.toml` and linked from the README.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- GitHub issue templates (bug report, feature request), pull request template, and
  `dependabot.yml` (pip, GitHub Actions, and Cargo ecosystems).
- Release workflow (`.github/workflows/release.yml`): tagging `v*` publishes a GitHub Release
  with notes extracted from this changelog.

### Fixed

- README clone URLs now point to the canonical repository.

## [0.1.0] - 2026-07-03

Initial release of the JAX-based in-silico cortical culture simulator.

### Added

- **Neuron models**: Izhikevich (2003, five cortical cell types), Adaptive Exponential
  integrate-and-fire (Brette & Gerstner 2005), and hybrid Izhikevich/AdEx populations.
- **Conductance-based synapses**: AMPA, NMDA (with Mg²⁺ voltage-dependent block),
  GABA_A, and GABA_B (dual-exponential kinetics).
- **Plasticity across four timescales**: short-term (Tsodyks-Markram), spike-timing-dependent
  (trace-based), homeostatic scaling (Turrigiano), and structural plasticity.
- **Virtual MEA**: CL1 64-channel (8×8, 200 µm) and MaxOne HD-MEA (26,400 electrodes,
  17.5 µm) with spike detection, LFP approximation, and stimulation.
- **Simulation core**: fully JIT-compiled loop via `jax.lax.scan`, differentiable through
  surrogate gradients (SuperSpike, sigmoid, arctan).
- **Analysis toolkit**: criticality (branching ratio, avalanche distributions), burst detection
  (Wagenaar 2006), functional connectivity, information theory, and pharmacology.
- **Closed-loop games**: Pong (DishBrain replication, Kagan et al. 2022) and Doom (ViZDoom).
- **Training pipeline**: differentiable weight optimization against recorded firing- and
  burst-rate targets, with multi-GPU neuron-parallel sharding scaffold.
- **doom-neuron integration**: virtual CL1 UDP server, live monitor, and dashboard.

[Unreleased]: https://github.com/lau-sam/bl1/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lau-sam/bl1/releases/tag/v0.1.0
