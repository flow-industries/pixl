//! Fabricate a synthetic "AI pixel-art-style" image to eyeball `pixl pixelize`:
//! a known low-res sprite upscaled with per-pixel noise (no clean grid lines).
//! Usage: cargo run --example demo_fixture -- <out.png>

use image::{Rgba, RgbaImage};

const PAL: [[u8; 3]; 16] = [
    [34, 32, 52],
    [69, 40, 60],
    [102, 57, 49],
    [143, 86, 59],
    [223, 113, 38],
    [217, 160, 102],
    [238, 195, 154],
    [251, 242, 54],
    [153, 229, 80],
    [106, 190, 48],
    [55, 148, 110],
    [75, 105, 47],
    [63, 63, 116],
    [48, 96, 130],
    [91, 110, 225],
    [99, 155, 255],
];

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "demo.png".into());
    let n: u32 = 24;
    let factor: u32 = 27; // 24 -> 648px, 27px "fat pixels"

    let mut s: u64 = 0x1234_5678;
    let mut rng = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };

    let mut sprite = vec![[0u8; 3]; (n * n) as usize];
    for i in 0..sprite.len() {
        sprite[i] = if i > 0 && rng() % 100 < 60 {
            sprite[i - 1]
        } else {
            PAL[(rng() % PAL.len() as u32) as usize]
        };
    }

    let big = n * factor;
    let mut img = RgbaImage::new(big, big);
    for y in 0..big {
        for x in 0..big {
            let c = sprite[((y / factor) * n + x / factor) as usize];
            let mut px = [0u8, 0, 0, 255];
            for k in 0..3 {
                let nz = (rng() % 9) as i32 - 4;
                px[k] = (c[k] as i32 + nz).clamp(0, 255) as u8;
            }
            img.put_pixel(x, y, Rgba(px));
        }
    }
    img.save(&out).unwrap();
    eprintln!("wrote {out}  ({big}x{big}, true sprite {n}x{n}, {factor}px cells)");
}
