//! Distance-dependent connectivity.
//!
//! Connection probability decays exponentially with distance,
//! `p(d) = p_max · exp(-d / lambda)`, and weights are drawn around the group
//! conductance with ±10% jitter. Excitatory and inhibitory presynaptic
//! connections are returned as separate CSR matrices with the `W[post, pre]`
//! convention.

use crate::placement::{Position, distance};
use bl1_core::CsrMatrix;
use rand::Rng;

/// Build excitatory and inhibitory weight matrices.
///
/// `g_exc` / `g_inh` are the mean conductances; each realised weight is
/// `g · (1 + 0.1·U(-1, 1))`. Self-connections are excluded.
pub fn build_connectivity<R: Rng>(
    rng: &mut R,
    positions: &[Position],
    is_excitatory: &[bool],
    lambda_um: f32,
    p_max: f32,
    g_exc: f32,
    g_inh: f32,
) -> (CsrMatrix, CsrMatrix) {
    let n = positions.len();
    let mut exc = Vec::new();
    let mut inh = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if i == j {
                continue;
            }
            let d = distance(positions[i], positions[j]);
            let prob = p_max * (-d / lambda_um).exp();
            if rng.random::<f32>() < prob {
                let jitter = 1.0 + 0.1 * (rng.random::<f32>() * 2.0 - 1.0);
                if is_excitatory[i] {
                    exc.push((j, i, g_exc * jitter));
                } else {
                    inh.push((j, i, g_inh * jitter));
                }
            }
        }
    }
    (
        CsrMatrix::from_triplets(n, n, exc),
        CsrMatrix::from_triplets(n, n, inh),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn closer_neurons_connect_more_often() {
        // With lambda small, only nearby pairs connect. Place neurons on a line.
        let mut rng = Pcg64::seed_from_u64(1);
        let positions: Vec<Position> = (0..200).map(|k| [k as f32 * 10.0, 0.0]).collect();
        let is_exc = vec![true; 200];
        let (exc, _inh) = build_connectivity(&mut rng, &positions, &is_exc, 50.0, 0.9, 0.1, 0.3);
        assert!(exc.nnz() > 0);
        // No self-connections.
        for j in 0..exc.n_rows {
            for k in exc.indptr[j]..exc.indptr[j + 1] {
                assert_ne!(exc.indices[k], j);
            }
        }
    }

    #[test]
    fn inhibitory_presynaptic_go_to_inh_matrix() {
        let mut rng = Pcg64::seed_from_u64(2);
        let positions = vec![[0.0, 0.0], [5.0, 0.0]];
        // Neuron 1 is inhibitory.
        let is_exc = vec![true, false];
        let (exc, inh) = build_connectivity(&mut rng, &positions, &is_exc, 1000.0, 1.0, 0.1, 0.3);
        // Column index 1 (the inhibitory presynaptic neuron) never appears in exc.
        assert!(exc.indices.iter().all(|&c| c != 1));
        // And neuron 1's outgoing connection lands in inh (col == 1).
        assert!(inh.indices.contains(&1));
    }
}
