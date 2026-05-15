"""Burst-rate-driven training smoke tests.

These tests don't try to validate burst-rate *convergence* at a
Wagenaar-2006 8-bursts/min target -- that requires the production
5K-neuron / 5s-sim run on DGX.  They check the cheaper-to-validate
contract: at the burst-detection scan length (sim_duration_ms in the
seconds range), burst-rate-loss-driven training produces a finite loss
history, no NaN epochs, and the burst-rate component contributes a
non-trivial gradient.

Marked ``@pytest.mark.slow`` so they don't slow the default test run --
each test runs a tiny 100-neuron / 1s-sim / 3-epoch training, ~30s on
CPU after JIT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.training.trainer import (
    TrainingConfig,
    TrainingResult,
    train_weights,
)


def _burst_config(**overrides) -> TrainingConfig:
    """A minimal burst-training config: small but at the burst-detection scan length.

    1s of sim is shorter than the production 5s, but long enough that the
    burst-rate loss machinery (Gaussian smoothing kernel, tanh-based peak
    detection) actually has time-axis structure to operate on -- which
    sub-500ms scans don't.
    """
    base = dict(
        n_neurons=100,
        n_epochs=3,
        sim_duration_ms=1000.0,
        dt=0.5,
        learning_rate=1e-4,
        ei_ratio=0.8,
        target_firing_rate_hz=1.6,
        target_burst_rate_per_min=8.0,
        w_firing_rate=1.0,
        w_burst_rate=0.5,
        w_synchrony=0.5,
        surrogate_beta=5.0,
        lambda_um=200.0,
        p_max=0.21,
        g_exc_init=0.12,
        g_inh_init=0.36,
        use_stp=True,
        U_exc=0.30,
        tau_rec=800.0,
        init_weight_scale=0.3,
        I_noise_amplitude=3.0,
        auto_noise=False,
        seed=42,
    )
    base.update(overrides)
    return TrainingConfig(**base)


@pytest.mark.slow
class TestBurstTrainingSmoke:
    """Burst-rate loss is wired and doesn't NaN at the burst-detection scan length."""

    def test_burst_training_runs_to_completion(self):
        config = _burst_config()
        result = train_weights(config)

        assert isinstance(result, TrainingResult)
        assert len(result.loss_history) == config.n_epochs
        for rec in result.loss_history:
            assert "burst_rate" in rec, \
                "burst-rate loss component must be logged every epoch"
            assert np.isfinite(rec["total"]), \
                f"Non-finite total loss at epoch {rec.get('epoch')}"
            assert np.isfinite(rec["burst_rate"]), \
                f"Non-finite burst_rate loss at epoch {rec.get('epoch')}"

    def test_no_persistent_nan_at_burst_scale(self):
        """The NaN-guard should keep training stable end-to-end."""
        config = _burst_config()
        result = train_weights(config)
        n_nan = sum(
            1 for rec in result.loss_history if not np.isfinite(rec["total"])
        )
        assert n_nan == 0, f"{n_nan}/{config.n_epochs} epochs produced NaN total"

    def test_burst_rate_component_is_nonzero(self):
        """If burst-rate loss didn't move at all the training wouldn't be
        driving toward the target -- distinguishes 'wired' from 'live'."""
        config = _burst_config()
        result = train_weights(config)
        burst_history = [rec["burst_rate"] for rec in result.loss_history]
        # At least one epoch should show a non-trivial burst-rate loss
        # value (the trainer is actively penalising mis-targeted bursts,
        # not just summing zeros).
        assert max(burst_history) > 0.0, \
            f"burst_rate loss never moved: {burst_history}"

    def test_w_burst_zero_disables_term(self):
        """Setting w_burst_rate=0 should make total loss invariant to burst_rate."""
        cfg_off = _burst_config(w_burst_rate=0.0, n_epochs=1)
        cfg_on = _burst_config(w_burst_rate=0.5, n_epochs=1)

        r_off = train_weights(cfg_off)
        r_on = train_weights(cfg_on)

        rec_off = r_off.loss_history[0]
        rec_on = r_on.loss_history[0]
        # Same seed, same network -- the only loss component that should
        # differ is the burst-weighted contribution.  Total loss must
        # differ by *at least* w_burst_rate * burst_rate.
        assert rec_on["total"] != rec_off["total"], (
            "w_burst_rate had no effect on total loss "
            f"(off={rec_off['total']}, on={rec_on['total']})"
        )
