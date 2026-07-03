//! Build Izhikevich populations with biologically motivated cell-type mixes.
//!
//! Excitatory types (Regular Spiking, Intrinsically Bursting, Chattering) and
//! inhibitory types (Fast Spiking, Low-Threshold Spiking) follow the fractions
//! from the reference model. Neurons are laid out in contiguous blocks with all
//! excitatory neurons first, matching the Python convention.

use crate::neuron::IzhParams;

/// `(a, b, c, d, fraction)` for one Izhikevich cell type.
struct CellType {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    frac: f32,
}

const EXCITATORY: [CellType; 3] = [
    CellType {
        a: 0.02,
        b: 0.2,
        c: -65.0,
        d: 8.0,
        frac: 0.64,
    }, // RS
    CellType {
        a: 0.02,
        b: 0.2,
        c: -55.0,
        d: 4.0,
        frac: 0.08,
    }, // IB
    CellType {
        a: 0.02,
        b: 0.2,
        c: -50.0,
        d: 2.0,
        frac: 0.08,
    }, // CH
];

const INHIBITORY: [CellType; 2] = [
    CellType {
        a: 0.1,
        b: 0.2,
        c: -65.0,
        d: 2.0,
        frac: 0.16,
    }, // FS
    CellType {
        a: 0.02,
        b: 0.25,
        c: -65.0,
        d: 2.0,
        frac: 0.04,
    }, // LTS
];

/// Split `total` neurons across `types` proportionally to their fractions,
/// assigning any rounding remainder to the first (most common) type.
fn counts_for(total: usize, types: &[CellType], group_frac_sum: f32) -> Vec<usize> {
    let mut counts: Vec<usize> = types
        .iter()
        .map(|t| ((t.frac / group_frac_sum) * total as f32).round() as usize)
        .collect();
    let assigned: usize = counts.iter().sum();
    if !counts.is_empty() {
        // Correct rounding drift on the dominant type (never underflow).
        let diff = total as isize - assigned as isize;
        let first = counts[0] as isize + diff;
        counts[0] = first.max(0) as usize;
    }
    counts
}

/// A built population: per-neuron Izhikevich parameters, an excitatory mask,
/// and the excitatory neuron count.
pub struct Population {
    pub params: IzhParams,
    pub is_excitatory: Vec<bool>,
    pub n_exc: usize,
}

/// Build `n` neurons with the given excitatory ratio (e.g. `0.8`).
pub fn build_population(n: usize, ei_ratio: f32) -> Population {
    let n_exc = ((n as f32) * ei_ratio).round() as usize;
    let n_exc = n_exc.min(n);
    let n_inh = n - n_exc;

    let exc_counts = counts_for(n_exc, &EXCITATORY, 0.8);
    let inh_counts = counts_for(n_inh, &INHIBITORY, 0.2);

    let mut params = IzhParams {
        a: Vec::with_capacity(n),
        b: Vec::with_capacity(n),
        c: Vec::with_capacity(n),
        d: Vec::with_capacity(n),
    };
    let mut is_excitatory = Vec::with_capacity(n);

    let push = |ct: &CellType, count: usize, exc: bool, p: &mut IzhParams, mask: &mut Vec<bool>| {
        for _ in 0..count {
            p.a.push(ct.a);
            p.b.push(ct.b);
            p.c.push(ct.c);
            p.d.push(ct.d);
            mask.push(exc);
        }
    };

    for (ct, &count) in EXCITATORY.iter().zip(&exc_counts) {
        push(ct, count, true, &mut params, &mut is_excitatory);
    }
    for (ct, &count) in INHIBITORY.iter().zip(&inh_counts) {
        push(ct, count, false, &mut params, &mut is_excitatory);
    }

    // Guard against rounding leaving us short/over by padding/truncating on RS.
    while params.a.len() < n {
        push(&EXCITATORY[0], 1, true, &mut params, &mut is_excitatory);
    }
    params.a.truncate(n);
    params.b.truncate(n);
    params.c.truncate(n);
    params.d.truncate(n);
    is_excitatory.truncate(n);

    let n_exc_actual = is_excitatory.iter().filter(|&&e| e).count();
    Population {
        params,
        is_excitatory,
        n_exc: n_exc_actual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn population_has_exact_neuron_count() {
        for &n in &[10usize, 200, 5000, 1] {
            let pop = build_population(n, 0.8);
            assert_eq!(pop.params.a.len(), n);
            assert_eq!(pop.is_excitatory.len(), n);
        }
    }

    #[test]
    fn ei_ratio_is_approximately_respected() {
        let pop = build_population(1000, 0.8);
        assert!((pop.n_exc as i32 - 800).abs() <= 2, "n_exc = {}", pop.n_exc);
    }

    #[test]
    fn excitatory_neurons_come_first() {
        let pop = build_population(100, 0.8);
        let first_inh = pop.is_excitatory.iter().position(|&e| !e).unwrap();
        // Everything before the first inhibitory neuron is excitatory.
        assert!(pop.is_excitatory[..first_inh].iter().all(|&e| e));
    }
}
