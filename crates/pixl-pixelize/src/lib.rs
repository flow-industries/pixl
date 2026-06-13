//! Detect the true pixel grid of an AI-generated "pixel-art-style" image and
//! snap it to a clean, limited-palette sprite.
//!
//! Pipeline: per-axis perceptual change signal -> comb-score period & phase
//! (edge energy minus mid-cell energy, so harmonics lose) -> dominant-color
//! cell collapse -> Lab k-means palette. Pure CPU, deterministic, GPU-free.

mod collapse;
mod color;
mod grid;
mod palette;
mod signal;

pub use color::Rgb;

use color::rgb_to_lab;
use image::{RgbImage, RgbaImage};
use rayon::prelude::*;
use signal::{change_signal, Axis, LabField};

#[derive(Clone, Debug)]
pub struct PixelizeParams {
    /// Force the logical cell size in source pixels, bypassing detection entirely.
    pub pixel_size: Option<u32>,
    /// Fallback logical long-edge (in cells) used only when detection fails.
    pub target_cells: Option<u32>,
    /// Palette size. `0` keeps every distinct cell color (no quantization).
    pub max_colors: u16,
    /// Border pixels trimmed before detection (AI borders are often noisy).
    pub trim_border: u32,
    /// Alpha at or above this is opaque; below is treated as transparent.
    pub alpha_threshold: u8,
    /// Seed for the deterministic k-means init.
    pub kmeans_seed: u64,
}

impl Default for PixelizeParams {
    fn default() -> Self {
        Self {
            pixel_size: None,
            target_cells: Some(128),
            max_colors: 16,
            trim_border: 2,
            alpha_threshold: 128,
            kmeans_seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PixelizeReport {
    /// Detected (or forced) cell size in source pixels, per axis.
    pub detected_cell_px: (f32, f32),
    /// Output logical dimensions in cells.
    pub out_cells: (u32, u32),
    /// Number of palette colors in the result.
    pub palette_len: u16,
    /// True when grid detection was weak and a fallback was used.
    pub low_confidence: bool,
}

#[derive(thiserror::Error, Debug)]
pub enum PixelizeError {
    #[error("image too small for grid detection")]
    TooSmall,
}

const LOW_CONF: f32 = 3.0;
const FAIL_CONF: f32 = 1.5;
const DETREND_WINDOW: usize = 9;
const PHASE_STEP: f32 = 0.5;
const TOP_CANDIDATES: usize = 5;
const COLLAPSE_BIN: f32 = 12.0;
const KMEANS_ITERS: usize = 16;

/// Snap an AI pixel-art image to a true-grid, limited-palette sprite at logical resolution.
pub fn pixelize(
    img: &RgbaImage,
    params: &PixelizeParams,
) -> Result<(RgbImage, PixelizeReport), PixelizeError> {
    let (w0, h0) = img.dimensions();
    let t = params.trim_border.min(w0 / 4).min(h0 / 4);
    let cropped = image::imageops::crop_imm(
        img,
        t,
        t,
        w0.saturating_sub(2 * t),
        h0.saturating_sub(2 * t),
    )
    .to_image();
    let (cw, ch) = cropped.dimensions();
    if cw < 4 || ch < 4 {
        return Err(PixelizeError::TooSmall);
    }
    let (w, h) = (cw as usize, ch as usize);

    let mut lab = Vec::with_capacity(w * h);
    let mut opaque = Vec::with_capacity(w * h);
    for p in cropped.pixels() {
        let [r, g, b, a] = p.0;
        lab.push(rgb_to_lab(Rgb::new(r, g, b)));
        opaque.push(a >= params.alpha_threshold);
    }
    let field = LabField { w, h, lab, opaque };

    let (gx, gy, low_conf) = decide_grid(&field, params);
    let bx = grid::boundaries(w, gx.0, gx.1);
    let by = grid::boundaries(h, gy.0, gy.1);
    let nx = bx.len() - 1;
    let ny = by.len() - 1;

    let cells: Vec<Option<Rgb>> = (0..nx * ny)
        .into_par_iter()
        .map(|idx| {
            let cx = idx % nx;
            let cy = idx / nx;
            let x1 = bx[cx + 1].max(bx[cx] + 1);
            let y1 = by[cy + 1].max(by[cy] + 1);
            collapse::collapse_cell(&field, bx[cx], x1, by[cy], y1, COLLAPSE_BIN)
        })
        .collect();

    let opaque_colors: Vec<Rgb> = cells.iter().flatten().copied().collect();
    let palette = if params.max_colors == 0 || opaque_colors.is_empty() {
        Vec::new()
    } else {
        palette::kmeans_palette(
            &opaque_colors,
            params.max_colors as usize,
            KMEANS_ITERS,
            params.kmeans_seed,
        )
    };
    let palette_labs: Vec<color::Lab> = palette.iter().map(|&c| rgb_to_lab(c)).collect();

    let mut out = RgbImage::new(nx as u32, ny as u32);
    for cy in 0..ny {
        for cx in 0..nx {
            let rgb = match cells[cy * nx + cx] {
                Some(col) if !palette.is_empty() => palette::nearest(&palette, &palette_labs, col),
                Some(col) => col,
                None => Rgb::new(0, 0, 0),
            };
            out.put_pixel(cx as u32, cy as u32, image::Rgb([rgb.r, rgb.g, rgb.b]));
        }
    }

    let palette_len = if palette.is_empty() {
        distinct(&opaque_colors) as u16
    } else {
        palette.len() as u16
    };

    Ok((
        out,
        PixelizeReport {
            detected_cell_px: (gx.0, gy.0),
            out_cells: (nx as u32, ny as u32),
            palette_len,
            low_confidence: low_conf,
        },
    ))
}

fn distinct(colors: &[Rgb]) -> usize {
    let mut v: Vec<u32> = colors
        .iter()
        .map(|c| (c.r as u32) << 16 | (c.g as u32) << 8 | c.b as u32)
        .collect();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// Returns `((period_x, phase_x), (period_y, phase_y), low_confidence)` with
/// phases already in column-boundary space (detected edge-phase + 1).
fn decide_grid(field: &LabField, params: &PixelizeParams) -> ((f32, f32), (f32, f32), bool) {
    if let Some(p) = params.pixel_size {
        let p = p.max(1) as f32;
        return ((p, 0.0), (p, 0.0), false);
    }
    let sx = change_signal(field, Axis::X);
    let sy = change_signal(field, Axis::Y);
    let dx = grid::detect_axis(&sx, DETREND_WINDOW, PHASE_STEP, TOP_CANDIDATES);
    let dy = grid::detect_axis(&sy, DETREND_WINDOW, PHASE_STEP, TOP_CANDIDATES);

    match (dx, dy) {
        (Some(a), Some(b)) => {
            if a.confidence < FAIL_CONF && b.confidence < FAIL_CONF {
                return fallback_grid(field, params);
            }
            let low = a.confidence < LOW_CONF || b.confidence < LOW_CONF;
            let (ax, ay) = (a.phase + 1.0, b.phase + 1.0);
            // AI pixel art is near-square: if one axis is weak, borrow the strong period.
            if a.confidence >= LOW_CONF && b.confidence < LOW_CONF {
                ((a.period, ax), (a.period, ay), low)
            } else if b.confidence >= LOW_CONF && a.confidence < LOW_CONF {
                ((b.period, ax), (b.period, ay), low)
            } else {
                ((a.period, ax), (b.period, ay), low)
            }
        }
        (Some(a), None) => (
            (a.period, a.phase + 1.0),
            (a.period, a.phase + 1.0),
            a.confidence < LOW_CONF,
        ),
        (None, Some(b)) => (
            (b.period, b.phase + 1.0),
            (b.period, b.phase + 1.0),
            b.confidence < LOW_CONF,
        ),
        (None, None) => fallback_grid(field, params),
    }
}

fn fallback_grid(field: &LabField, params: &PixelizeParams) -> ((f32, f32), (f32, f32), bool) {
    let target = params.target_cells.unwrap_or(128).max(1) as f32;
    let long = field.w.max(field.h) as f32;
    let period = (long / target).max(1.0);
    ((period, 0.0), (period, 0.0), true)
}
