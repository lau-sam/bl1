//! `bl1-analysis` — burst detection and criticality metrics for spike rasters,
//! ported from the NumPy analysis modules of BL-1.

pub mod bursts;
pub mod criticality;

pub use bursts::{Burst, BurstStatistics, burst_statistics, detect_bursts};
pub use criticality::{avalanche_distributions, branching_ratio, estimate_power_law_exponent};
