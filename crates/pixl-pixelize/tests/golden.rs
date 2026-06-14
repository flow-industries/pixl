//! Synthetic golden tests: build a KNOWN low-res sprite, upscale it the way an
//! AI model would render blocky pixels, degrade it (noise / non-integer scale /
//! phase offset), and assert `pixelize` recovers the original grid and palette.
//! No GPU, fully deterministic -> runs on CI.

use image::{RgbImage, Rgba, RgbaImage};
use pixl_pixelize::{pixelize, PixelizeParams};

struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Lcg((s ^ 0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn range(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
    fn jitter(&mut self, mag: i32) -> i32 {
        (self.next_u32() % (2 * mag as u32 + 1)) as i32 - mag
    }
}

const PALETTE8: [[u8; 3]; 8] = [
    [20, 20, 30],
    [200, 60, 60],
    [60, 160, 80],
    [70, 110, 200],
    [230, 200, 90],
    [240, 240, 240],
    [150, 90, 180],
    [120, 80, 40],
];

fn make_source(n: u32, palette: &[[u8; 3]], seed: u64) -> RgbImage {
    let mut rng = Lcg::new(seed);
    let mut img = RgbImage::new(n, n);
    // bias toward spatial structure: pick a color, sometimes repeat the left neighbor
    let mut prev = palette[0];
    for y in 0..n {
        for x in 0..n {
            let c = if x > 0 && rng.range(100) < 55 {
                prev
            } else {
                palette[rng.range(palette.len() as u32) as usize]
            };
            prev = c;
            img.put_pixel(x, y, image::Rgb(c));
        }
    }
    img
}

fn upscale_nearest(src: &RgbImage, factor: u32) -> RgbaImage {
    let (w, h) = src.dimensions();
    let mut out = RgbaImage::new(w * factor, h * factor);
    for y in 0..h * factor {
        for x in 0..w * factor {
            let p = src.get_pixel(x / factor, y / factor).0;
            out.put_pixel(x, y, Rgba([p[0], p[1], p[2], 255]));
        }
    }
    out
}

fn add_noise(img: &RgbaImage, mag: i32, seed: u64) -> RgbaImage {
    let mut rng = Lcg::new(seed);
    let mut out = img.clone();
    for p in out.pixels_mut() {
        for c in 0..3 {
            p.0[c] = (p.0[c] as i32 + rng.jitter(mag)).clamp(0, 255) as u8;
        }
    }
    out
}

fn to_rgba(img: &RgbImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut out = RgbaImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels() {
        out.put_pixel(x, y, Rgba([p.0[0], p.0[1], p.0[2], 255]));
    }
    out
}

fn mean_rgb_delta(a: &RgbImage, b: &RgbImage) -> f32 {
    assert_eq!(a.dimensions(), b.dimensions());
    let mut acc = 0.0;
    let mut n = 0.0;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let d: f32 = (0..3)
            .map(|i| {
                let x = pa.0[i] as f32 - pb.0[i] as f32;
                x * x
            })
            .sum::<f32>()
            .sqrt();
        acc += d;
        n += 1.0;
    }
    acc / n
}

fn params(colors: u16, target: u32) -> PixelizeParams {
    PixelizeParams {
        max_colors: colors,
        trim_border: 0,
        target_cells: Some(target),
        ..Default::default()
    }
}

#[test]
fn recovers_clean_upscaled_grid() {
    let src = make_source(16, &PALETTE8, 1);
    let big = upscale_nearest(&src, 32); // 512x512, integer cells
    let noisy = add_noise(&big, 6, 2);
    let (out, rep) = pixelize(&noisy, &params(8, 16)).unwrap();
    assert_eq!(
        rep.out_cells,
        (16, 16),
        "cell px {:?}",
        rep.detected_cell_px
    );
    assert!(rep.palette_len <= 8, "palette {}", rep.palette_len);
    let d = mean_rgb_delta(&out, &src);
    assert!(d < 14.0, "mean rgb delta {d}");
}

#[test]
fn recovers_16px_cells_32_colors() {
    let src = make_source(24, &PALETTE8, 7);
    let big = upscale_nearest(&src, 16); // 384x384
    let noisy = add_noise(&big, 5, 9);
    let (_out, rep) = pixelize(&noisy, &params(16, 24)).unwrap();
    assert_eq!(
        rep.out_cells,
        (24, 24),
        "cell px {:?}",
        rep.detected_cell_px
    );
}

#[test]
fn handles_phase_offset() {
    let src = make_source(16, &PALETTE8, 3);
    let big = upscale_nearest(&src, 32);
    // crop an off-grid offset so the first boundary is not at 0
    let shifted = image::imageops::crop_imm(&big, 5, 11, 512 - 5, 512 - 11).to_image();
    let (_out, rep) = pixelize(&shifted, &params(8, 16)).unwrap();
    assert!(
        (15..=17).contains(&rep.out_cells.0) && (15..=17).contains(&rep.out_cells.1),
        "out_cells {:?} cell {:?}",
        rep.out_cells,
        rep.detected_cell_px
    );
}

#[test]
fn handles_non_integer_scale() {
    let src = make_source(16, &PALETTE8, 5);
    let big = to_rgba(&src);
    // nearest resize to a non-integer factor (500/16 = 31.25 px per cell)
    let scaled = image::imageops::resize(&big, 500, 500, image::imageops::FilterType::Nearest);
    let (_out, rep) = pixelize(&scaled, &params(8, 16)).unwrap();
    assert!(
        (15..=17).contains(&rep.out_cells.0),
        "out_cells {:?} cell {:?}",
        rep.out_cells,
        rep.detected_cell_px
    );
}

#[test]
fn pixel_size_override_bypasses_detection() {
    let src = make_source(20, &PALETTE8, 11);
    let big = upscale_nearest(&src, 10); // 200x200
    let p = PixelizeParams {
        pixel_size: Some(10),
        trim_border: 0,
        max_colors: 8,
        ..Default::default()
    };
    let (_out, rep) = pixelize(&big, &p).unwrap();
    assert_eq!(rep.out_cells, (20, 20));
    assert!(!rep.low_confidence);
}

#[test]
fn idempotent_on_clean_sprite() {
    let src = make_source(16, &PALETTE8, 2);
    let big = upscale_nearest(&src, 20);
    let (first, rep1) = pixelize(&big, &params(8, 16)).unwrap();
    let again = upscale_nearest(&first, 12);
    let (second, rep2) = pixelize(&again, &params(8, 16)).unwrap();
    assert_eq!(rep1.out_cells, rep2.out_cells, "dims drifted");
    assert_eq!(first.dimensions(), second.dimensions());
    let d = mean_rgb_delta(&first, &second);
    assert!(d < 6.0, "idempotence delta {d}");
}

#[test]
fn tiny_lowdetail_falls_back_not_bogus_grid() {
    // a small, near-uniform image has no real grid: detection must fail into the
    // fallback (low_confidence) instead of fabricating a 2-3px grid.
    let mut img = RgbaImage::new(40, 40);
    for p in img.pixels_mut() {
        *p = Rgba([90, 120, 70, 255]);
    }
    let noisy = add_noise(&img, 3, 4);
    let (_out, rep) = pixelize(&noisy, &params(8, 16)).unwrap();
    assert!(
        rep.low_confidence,
        "near-uniform image should fall back, got confident cell {:?}",
        rep.detected_cell_px
    );
}
