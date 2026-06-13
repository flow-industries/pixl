//! Per-axis perceptual change signals.

use crate::color::{delta_e76, Lab};

#[derive(Clone, Copy)]
pub enum Axis {
    X,
    Y,
}

/// Precomputed Lab values + opacity mask for a region.
pub struct LabField {
    pub w: usize,
    pub h: usize,
    pub lab: Vec<Lab>,
    pub opaque: Vec<bool>,
}

impl LabField {
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> Lab {
        self.lab[y * self.w + x]
    }
    #[inline]
    pub fn is_opaque(&self, x: usize, y: usize) -> bool {
        self.opaque[y * self.w + x]
    }
}

/// Sum of perceptual change across the axis orthogonal to `axis`, per boundary.
/// `Axis::X` returns length `w-1` (vertical edges); `Axis::Y` returns length `h-1`.
/// `s[k]` is the change between index `k` and `k+1`, so a real cell boundary at
/// column `c` shows up as energy at `s[c-1]`.
pub fn change_signal(f: &LabField, axis: Axis) -> Vec<f32> {
    match axis {
        Axis::X => {
            let n = f.w.saturating_sub(1);
            let mut s = vec![0.0f32; n];
            for (k, sk) in s.iter_mut().enumerate() {
                let mut acc = 0.0;
                for y in 0..f.h {
                    if f.is_opaque(k, y) && f.is_opaque(k + 1, y) {
                        acc += delta_e76(f.at(k, y), f.at(k + 1, y));
                    }
                }
                *sk = acc;
            }
            s
        }
        Axis::Y => {
            let n = f.h.saturating_sub(1);
            let mut s = vec![0.0f32; n];
            for (k, sk) in s.iter_mut().enumerate() {
                let mut acc = 0.0;
                for x in 0..f.w {
                    if f.is_opaque(x, k) && f.is_opaque(x, k + 1) {
                        acc += delta_e76(f.at(x, k), f.at(x, k + 1));
                    }
                }
                *sk = acc;
            }
            s
        }
    }
}

/// Subtract a sliding median to flatten gradient plateaus without erasing sharp edges.
pub fn detrend(signal: &[f32], window: usize) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let half = window / 2;
    let mut out = vec![0.0f32; n];
    let mut buf: Vec<f32> = Vec::with_capacity(window);
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        buf.clear();
        buf.extend_from_slice(&signal[lo..hi]);
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = buf[buf.len() / 2];
        out[i] = (signal[i] - med).max(0.0);
    }
    out
}
