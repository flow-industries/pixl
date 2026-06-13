//! Collapse each grid cell to one representative color via a coarse Lab dominant-bin.

use crate::color::{lab_to_rgb, Lab, Rgb};
use crate::signal::LabField;
use std::collections::HashMap;

/// Running (sum_l, sum_a, sum_b, count) for one coarse Lab bin.
type BinAccum = (f32, f32, f32, u32);

/// Densest coarse-Lab-bin mean for a cell, ignoring transparent pixels.
/// Returns `None` when the cell has no opaque pixels.
pub fn collapse_cell(
    f: &LabField,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    bin: f32,
) -> Option<Rgb> {
    let mut bins: HashMap<(i32, i32, i32), BinAccum> = HashMap::new();
    for y in y0..y1 {
        for x in x0..x1 {
            if !f.is_opaque(x, y) {
                continue;
            }
            let c = f.at(x, y);
            let key = (
                (c.l / bin).round() as i32,
                (c.a / bin).round() as i32,
                (c.b / bin).round() as i32,
            );
            let e = bins.entry(key).or_insert((0.0, 0.0, 0.0, 0));
            e.0 += c.l;
            e.1 += c.a;
            e.2 += c.b;
            e.3 += 1;
        }
    }
    let best = bins.values().max_by_key(|e| e.3)?;
    let n = best.3 as f32;
    Some(lab_to_rgb(Lab {
        l: best.0 / n,
        a: best.1 / n,
        b: best.2 / n,
    }))
}
