//! Device selection. Prefers the Apple Silicon GPU (Metal); falls back to CPU.

use candle_core::{Device, Result};

/// Open the Metal GPU at `ordinal`, or fall back to CPU if Metal is unavailable.
pub fn select(ordinal: usize) -> Result<Device> {
    match Device::new_metal(ordinal) {
        Ok(d) => Ok(d),
        Err(_) => Ok(Device::Cpu),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

    // Ignored on CI (no Metal); run locally with: cargo test -p pixl-gen -- --ignored
    #[test]
    #[ignore]
    fn metal_matmul_roundtrips() {
        let dev = Device::new_metal(0).expect("metal device");
        let a = Tensor::randn(0f32, 1f32, (64, 64), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (64, 64), &dev).unwrap();
        let c = a.matmul(&b).unwrap().to_dtype(DType::F32).unwrap();
        let host = c.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(host.len(), 64 * 64);
        assert!(host.iter().any(|v| *v != 0.0), "matmul produced all zeros");
    }
}
