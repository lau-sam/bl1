//! Neuron placement on a 2-D substrate.

use rand::Rng;

/// A 2-D position in micrometres.
pub type Position = [f32; 2];

/// Place `n` neurons uniformly at random on a `substrate` (µm) rectangle.
pub fn place_neurons<R: Rng>(rng: &mut R, n: usize, substrate: [f32; 2]) -> Vec<Position> {
    (0..n)
        .map(|_| {
            [
                rng.random::<f32>() * substrate[0],
                rng.random::<f32>() * substrate[1],
            ]
        })
        .collect()
}

/// Euclidean distance between two positions (µm).
pub fn distance(a: Position, b: Position) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}
