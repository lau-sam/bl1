//! Sparse connectivity in CSR form.
//!
//! The weight convention matches the Python implementation: `W[post, pre]`,
//! so a matrix-vector product with a presynaptic spike/drive vector yields the
//! postsynaptic input, `input[post] = sum_pre W[post, pre] * drive[pre]`.

/// Compressed sparse row matrix of `f32` weights indexed by postsynaptic row.
#[derive(Debug, Clone, Default)]
pub struct CsrMatrix {
    /// Number of rows (postsynaptic neurons).
    pub n_rows: usize,
    /// Number of columns (presynaptic neurons).
    pub n_cols: usize,
    /// Row offsets, length `n_rows + 1`.
    pub indptr: Vec<usize>,
    /// Column (presynaptic) index of each stored value.
    pub indices: Vec<usize>,
    /// Stored weights, parallel to `indices`.
    pub data: Vec<f32>,
}

impl CsrMatrix {
    /// Build a CSR matrix from `(row, col, weight)` triplets.
    pub fn from_triplets(
        n_rows: usize,
        n_cols: usize,
        mut triplets: Vec<(usize, usize, f32)>,
    ) -> Self {
        triplets.sort_by_key(|&(r, c, _)| (r, c));
        let mut indptr = vec![0usize; n_rows + 1];
        let mut indices = Vec::with_capacity(triplets.len());
        let mut data = Vec::with_capacity(triplets.len());
        for &(r, c, w) in &triplets {
            debug_assert!(r < n_rows && c < n_cols, "triplet index out of bounds");
            indptr[r + 1] += 1;
            indices.push(c);
            data.push(w);
        }
        for i in 0..n_rows {
            indptr[i + 1] += indptr[i];
        }
        Self {
            n_rows,
            n_cols,
            indptr,
            indices,
            data,
        }
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Compute `out = W @ drive` where `drive` has length `n_cols`.
    ///
    /// Writes into `out` (length `n_rows`), overwriting its contents.
    pub fn matvec_into(&self, drive: &[f32], out: &mut [f32]) {
        debug_assert_eq!(drive.len(), self.n_cols);
        debug_assert_eq!(out.len(), self.n_rows);
        for (row, out_val) in out.iter_mut().enumerate() {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let mut acc = 0.0f32;
            for k in start..end {
                acc += self.data[k] * drive[self.indices[k]];
            }
            *out_val = acc;
        }
    }

    /// Allocating variant of [`Self::matvec_into`].
    pub fn matvec(&self, drive: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_rows];
        self.matvec_into(drive, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_dense() {
        // W[post, pre]: row 0 gets 2*drive[1]; row 1 gets 3*drive[0] + 4*drive[2].
        let w = CsrMatrix::from_triplets(2, 3, vec![(0, 1, 2.0), (1, 0, 3.0), (1, 2, 4.0)]);
        let drive = [1.0, 5.0, 2.0];
        assert_eq!(w.matvec(&drive), vec![10.0, 11.0]);
        assert_eq!(w.nnz(), 3);
    }

    #[test]
    fn empty_rows_are_zero() {
        let w = CsrMatrix::from_triplets(3, 2, vec![(1, 0, 1.5)]);
        assert_eq!(w.matvec(&[2.0, 0.0]), vec![0.0, 3.0, 0.0]);
    }
}
