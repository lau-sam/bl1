# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LICENSE` file (MIT), previously only declared in `pyproject.toml` and linked from the README.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- GitHub issue templates (bug report, feature request) and a pull request template.
- Rust Cargo workspace under `rust/` with the `bl1-core` crate: a forward spiking-network
  simulator (Izhikevich/AdEx neurons, AMPA/NMDA/GABA_A/GABA_B synapses, Tsodyks-Markram
  short-term plasticity, trace-based STDP, CSR connectivity) whose per-step ordering matches
  the reference JAX implementation, with a configurable AMPA/NMDA and GABA_A/GABA_B split.
- `bl1-analysis` crate: burst detection (Wagenaar) and criticality metrics (branching ratio,
  neuronal avalanches, MLE power-law exponents).
- `bl1-mea` crate: CL1 64-channel and MaxOne HD-MEA layouts, neuron→electrode mapping, spike
  detection, and a point-source LFP approximation.
- `bl1-sim` crate: neuron placement, distance-dependent connectivity, a reproducible `Culture`
  factory, and a YAML config loader compatible with `configs/*.yaml`.
- `bl1-tui` crate: a lazygit-style terminal UI (`bl1` binary) to browse configs, run capped
  preview simulations, and inspect the spike raster and culture statistics live. A `--headless`
  mode prints statistics without a terminal for smoke testing.

### Fixed

- README clone URLs now point to the canonical repository.
- Avalanche size/duration exponents are now estimated by maximum likelihood with
  KS-based `xmin` selection (Clauset et al. 2009) instead of a biased log-log CCDF
  regression. The estimator now returns `-alpha`, directly comparable to the Beggs &
  Plenz reference exponents (size `-1.5`, duration `-2.0`).

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
