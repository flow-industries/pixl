//! Lab k-means palette quantization (hand-rolled, deterministic for a fixed seed).

use crate::color::{delta_e76, lab_to_rgb, rgb_to_lab, Lab, Rgb};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// k-means++ over Lab; returns up to `k` RGB centroids. Deterministic for a fixed seed.
pub fn kmeans_palette(colors: &[Rgb], k: usize, iters: usize, seed: u64) -> Vec<Rgb> {
    if colors.is_empty() {
        return Vec::new();
    }
    let k = k.min(crate::color::distinct(colors)).max(1);
    let pts: Vec<Lab> = colors.iter().map(|&c| rgb_to_lab(c)).collect();
    let mut rng = Lcg::new(seed);

    let mut centers: Vec<Lab> = Vec::with_capacity(k);
    let first = ((rng.next_f32() * pts.len() as f32) as usize).min(pts.len() - 1);
    centers.push(pts[first]);
    while centers.len() < k {
        let d2: Vec<f32> = pts
            .iter()
            .map(|p| {
                centers
                    .iter()
                    .map(|c| {
                        let e = delta_e76(*p, *c);
                        e * e
                    })
                    .fold(f32::MAX, f32::min)
            })
            .collect();
        let sum: f32 = d2.iter().sum();
        if sum <= 0.0 {
            break;
        }
        let mut t = rng.next_f32() * sum;
        let mut idx = d2.len() - 1;
        for (i, &w) in d2.iter().enumerate() {
            t -= w;
            if t <= 0.0 {
                idx = i;
                break;
            }
        }
        centers.push(pts[idx]);
    }

    for _ in 0..iters {
        let mut acc = vec![(0.0f32, 0.0f32, 0.0f32, 0u32); centers.len()];
        for p in &pts {
            let mut bi = 0;
            let mut bd = f32::MAX;
            for (i, c) in centers.iter().enumerate() {
                let e = delta_e76(*p, *c);
                if e < bd {
                    bd = e;
                    bi = i;
                }
            }
            acc[bi].0 += p.l;
            acc[bi].1 += p.a;
            acc[bi].2 += p.b;
            acc[bi].3 += 1;
        }
        for (c, a) in centers.iter_mut().zip(acc.iter()) {
            if a.3 > 0 {
                *c = Lab {
                    l: a.0 / a.3 as f32,
                    a: a.1 / a.3 as f32,
                    b: a.2 / a.3 as f32,
                };
            }
        }
    }
    centers.into_iter().map(lab_to_rgb).collect()
}

/// Nearest palette color by Lab CIE76.
pub fn nearest(palette: &[Rgb], labs: &[Lab], c: Rgb) -> Rgb {
    let lc = rgb_to_lab(c);
    let mut best = 0;
    let mut bd = f32::MAX;
    for (i, l) in labs.iter().enumerate() {
        let e = delta_e76(lc, *l);
        if e < bd {
            bd = e;
            best = i;
        }
    }
    palette[best]
}
