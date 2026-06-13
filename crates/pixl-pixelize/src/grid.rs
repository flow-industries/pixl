//! Period + phase detection via modular folding (coverage scoring), and grid
//! reconstruction.
//!
//! Key idea: real grid boundaries always sit at multiples of the true cell size,
//! so folding the change-signal modulo the true period concentrates ~all edge
//! energy into a single phase bin. Harmonics (2x, 3x, ...) scatter that energy
//! across several bins, so the *largest* period whose fold concentrates the
//! energy is the fundamental.

use crate::signal::detrend;

#[derive(Clone, Copy, Debug)]
pub struct AxisGrid {
    pub period: f32,
    /// Phase in change-signal index space (edge between column k and k+1).
    pub phase: f32,
    pub confidence: f32,
}

const COVERAGE_THRESHOLD: f32 = 0.78;
const WINDOW_TOL: usize = 1; // phase-bin half-width counted as "covered"

/// Detect the grid for one axis. Returns the largest period whose modular fold
/// gathers at least `COVERAGE_THRESHOLD` of the total edge energy into one phase.
pub fn detect_axis(
    signal: &[f32],
    detrend_window: usize,
    _phase_step: f32,
    _unused: usize,
) -> Option<AxisGrid> {
    if signal.len() < 8 {
        return None;
    }
    let det = detrend(signal, detrend_window);
    let len = det.len();
    let total: f32 = det.iter().sum();
    if total <= 1e-6 {
        return None;
    }
    let min_p = 2usize;
    let max_p = (len / 4).max(min_p + 1);

    let mut best_any: Option<(usize, usize, f32)> = None; // period, phase, coverage
    for t in (min_p..=max_p).rev() {
        let mut hist = vec![0.0f32; t];
        for (k, &v) in det.iter().enumerate() {
            hist[k % t] += v;
        }
        let (peak, covered) = fold_peak(&hist, WINDOW_TOL);
        let coverage = covered / total;
        if best_any.is_none_or(|b| coverage > b.2) {
            best_any = Some((t, peak, coverage));
        }
        if coverage >= COVERAGE_THRESHOLD {
            return Some(grid_from(t, peak, coverage));
        }
    }
    best_any.map(|(t, p, cov)| grid_from(t, p, cov))
}

fn grid_from(period: usize, phase: usize, coverage: f32) -> AxisGrid {
    AxisGrid {
        period: period as f32,
        phase: phase as f32,
        confidence: coverage / (1.0 - coverage).max(0.02),
    }
}

/// Peak phase bin (argmax) and the maximum circular-window energy (coverage) of
/// a folded histogram. Phase is the single most energetic bin; coverage sums a
/// `2*tol+1` window so a boundary that straddles two bins still counts.
fn fold_peak(hist: &[f32], tol: usize) -> (usize, f32) {
    let t = hist.len();
    let w = (2 * tol + 1).min(t);
    let mut s: f32 = (0..w).map(|j| hist[j]).sum();
    let mut best_sum = f32::MIN;
    for start in 0..t {
        if s > best_sum {
            best_sum = s;
        }
        s -= hist[start];
        s += hist[(start + w) % t];
    }
    let mut peak = 0usize;
    let mut pv = f32::MIN;
    for (i, &v) in hist.iter().enumerate() {
        if v > pv {
            pv = v;
            peak = i;
        }
    }
    (peak, best_sum)
}

/// Cell boundary columns `[0, ..., len]`. `phase` is a column position (the first
/// interior boundary); successive boundaries are `phase + m*period`. Head/tail
/// slivers shorter than 0.4 * period are merged so non-integer periods and phase
/// offsets do not spawn a spurious thin cell.
pub fn boundaries(len: usize, period: f32, phase: f32) -> Vec<usize> {
    let p = period.max(1.0);
    let mut b = vec![0usize];
    let mut m = phase;
    while m < len as f32 {
        let col = m.round() as usize;
        if col > *b.last().unwrap() && col < len {
            b.push(col);
        }
        m += p;
    }
    if *b.last().unwrap() != len {
        b.push(len);
    }
    let thr = ((0.4 * p) as usize).max(1);
    if b.len() >= 3 && b[1] - b[0] < thr {
        b.remove(1);
    }
    let n = b.len();
    if n >= 3 && b[n - 1] - b[n - 2] < thr {
        b.remove(n - 2);
    }
    b
}
