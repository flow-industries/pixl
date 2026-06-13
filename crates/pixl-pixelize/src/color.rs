//! Minimal sRGB <-> CIELAB (D65) plus CIE76 color-difference.
//! Hand-rolled so `pixl-pixelize` stays dependency-light and fully deterministic.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

const XN: f32 = 0.950_47;
const YN: f32 = 1.0;
const ZN: f32 = 1.088_83;

fn f_fwd(t: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if t > D * D * D {
        t.cbrt()
    } else {
        t / (3.0 * D * D) + 4.0 / 29.0
    }
}

fn f_inv(t: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if t > D {
        t * t * t
    } else {
        3.0 * D * D * (t - 4.0 / 29.0)
    }
}

pub fn rgb_to_lab(c: Rgb) -> Lab {
    let r = srgb_to_linear(c.r as f32 / 255.0);
    let g = srgb_to_linear(c.g as f32 / 255.0);
    let b = srgb_to_linear(c.b as f32 / 255.0);
    let x = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b) / XN;
    let y = (0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b) / YN;
    let z = (0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b) / ZN;
    let (fx, fy, fz) = (f_fwd(x), f_fwd(y), f_fwd(z));
    Lab {
        l: 116.0 * fy - 16.0,
        a: 500.0 * (fx - fy),
        b: 200.0 * (fy - fz),
    }
}

pub fn lab_to_rgb(c: Lab) -> Rgb {
    let fy = (c.l + 16.0) / 116.0;
    let fx = fy + c.a / 500.0;
    let fz = fy - c.b / 200.0;
    let x = XN * f_inv(fx);
    let y = YN * f_inv(fy);
    let z = ZN * f_inv(fz);
    let r = 3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z;
    let g = -0.969_266 * x + 1.876_010_8 * y + 0.041_556 * z;
    let b = 0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z;
    let enc = |v: f32| {
        (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Rgb::new(enc(r), enc(g), enc(b))
}

#[inline]
pub fn delta_e76(p: Lab, q: Lab) -> f32 {
    let dl = p.l - q.l;
    let da = p.a - q.a;
    let db = p.b - q.b;
    (dl * dl + da * da + db * db).sqrt()
}
