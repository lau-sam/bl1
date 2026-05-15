"""Tests for the multi-GPU sharding scaffold (bl1.training.sharding).

JAX picks up its device count from XLA_FLAGS *before* the first jax
import.  We set the flag at module load time and import jax inside the
test functions so pytest, which imports many tests in one process,
doesn't lose the flag (the JAX backend is also process-global -- once
initialized it can't be re-initialized with a new device count).

For this reason these tests should run BEFORE any other test that has
already touched jax.  ``pytest -p no:cacheprovider tests/test_sharding.py``
in isolation always works; in a full ``pytest tests/`` run the fake-device
mesh assertions are skipped if jax has already locked in 1 device.
"""

from __future__ import annotations

import os

# Must be set BEFORE the first `import jax` anywhere in the process.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_force_host_platform_device_count=4",
)

import jax
import jax.numpy as jnp
import pytest

from bl1.training.sharding import (
    NEURON_AXIS,
    ShardedNetwork,
    external_current_spec,
    is_sharded,
    make_neuron_mesh,
    neuron_spec_1d,
    neuron_spec_2d,
    replicated_spec,
    shard_network,
)


_FAKE_DEVICE_COUNT_OK = len(jax.devices()) >= 2


@pytest.fixture(scope="module")
def mesh():
    if not _FAKE_DEVICE_COUNT_OK:
        pytest.skip(
            "JAX backend already initialized with <2 devices "
            "(XLA_FLAGS=--xla_force_host_platform_device_count=4 must be "
            "set before any jax import; this run picked up a single-device "
            "backend, run test_sharding.py in isolation to exercise the "
            "multi-device mesh assertions)."
        )
    return make_neuron_mesh()


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def test_neuron_axis_name_is_stable():
    assert NEURON_AXIS == "neuron"


def test_make_neuron_mesh_uses_all_devices(mesh):
    assert mesh.axis_names == (NEURON_AXIS,)
    assert mesh.size == len(jax.devices())
    assert mesh.size >= 2


def test_make_neuron_mesh_rejects_empty_device_list():
    with pytest.raises(ValueError, match="no JAX devices available"):
        make_neuron_mesh(devices=[])


# ---------------------------------------------------------------------------
# Partition specs
# ---------------------------------------------------------------------------


def test_partition_specs_shape():
    # Weight matrices: shard row (post-synaptic), replicate column (pre).
    assert tuple(neuron_spec_2d()) == (NEURON_AXIS, None)
    # Per-neuron 1-D state vectors.
    assert tuple(neuron_spec_1d()) == (NEURON_AXIS,)
    # External current: time replicated, neurons sharded.
    assert tuple(external_current_spec()) == (None, NEURON_AXIS)
    # Replicated (PartitionSpec() prints as PartitionSpec()).
    assert len(replicated_spec()) == 0


# ---------------------------------------------------------------------------
# shard_network
# ---------------------------------------------------------------------------


def test_shard_network_places_weights(mesh):
    N = 16
    W_exc = jnp.ones((N, N), dtype=jnp.float32)
    W_inh = jnp.ones((N, N), dtype=jnp.float32)

    out = shard_network(mesh, W_exc=W_exc, W_inh=W_inh)
    assert isinstance(out, ShardedNetwork)
    assert is_sharded(out.W_exc, mesh, neuron_spec_2d())
    assert is_sharded(out.W_inh, mesh, neuron_spec_2d())


def test_shard_network_places_external_current(mesh):
    T, N = 32, 16
    I = jnp.zeros((T, N), dtype=jnp.float32)
    out = shard_network(mesh, I_external=I)
    assert is_sharded(out.I_external, mesh, external_current_spec())


def test_shard_network_places_neuron_state(mesh):
    from bl1.core.izhikevich import NeuronState

    N = 16
    state = NeuronState(
        v=-65.0 * jnp.ones((N,), dtype=jnp.float32),
        u=jnp.zeros((N,), dtype=jnp.float32),
        spikes=jnp.zeros((N,), dtype=jnp.float32),
    )
    out = shard_network(mesh, init_state=state)
    assert is_sharded(out.v, mesh, neuron_spec_1d())
    assert is_sharded(out.u, mesh, neuron_spec_1d())
    assert is_sharded(out.spikes, mesh, neuron_spec_1d())


def test_shard_network_passes_through_none(mesh):
    out = shard_network(mesh)
    assert out.W_exc is None
    assert out.W_inh is None
    assert out.I_external is None
    assert out.v is None


# ---------------------------------------------------------------------------
# Sharded matmul actually distributes
# ---------------------------------------------------------------------------


def test_sharded_matvec_produces_correct_result(mesh):
    """A row-sharded matmul has to agree numerically with the replicated one."""
    N = 32
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    W = jax.random.normal(k1, (N, N))
    x = jax.random.normal(k2, (N,))

    expected = W @ x

    out = shard_network(mesh, W_exc=W)
    got = out.W_exc @ x

    # The sharded result is itself a sharded array; pull it back for the
    # numerical comparison.  Using rtol=1e-5 since the reduction order in
    # the sharded matmul differs from the replicated one.
    assert jnp.allclose(jax.device_get(got), jax.device_get(expected), rtol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end: trainer accepts a mesh and runs a step without NaN
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_weights_with_mesh(mesh):
    """One epoch of train_weights under a fake-multi-device mesh stays finite.

    Doesn't assert anything about *performance* -- a fake CPU mesh isn't
    faster than single-device -- only that the wiring through
    TrainingConfig.mesh actually flows the sharded arrays through
    simulate() without crashing or producing NaN.
    """
    from bl1.training.trainer import TrainingConfig, train_weights
    import numpy as np

    config = TrainingConfig(
        n_neurons=64,
        n_epochs=1,
        sim_duration_ms=200.0,
        dt=0.5,
        learning_rate=1e-4,
        target_firing_rate_hz=1.6,
        target_burst_rate_per_min=0.0,        # FR-only for speed
        w_burst_rate=0.0,
        surrogate_beta=5.0,
        p_max=0.21,
        init_weight_scale=0.3,
        auto_noise=False,
        seed=0,
        mesh=mesh,
    )

    result = train_weights(config)
    assert len(result.loss_history) == 1
    assert np.isfinite(result.loss_history[0]["total"]), \
        f"NaN under sharded mesh: {result.loss_history[0]}"
