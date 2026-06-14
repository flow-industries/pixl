//! Device selection. Prefers a GPU backend appropriate for the build — CUDA when
//! the `cuda` feature is on, otherwise Metal on macOS — and falls back to CPU.

use candle_core::{Device, Result};

/// Open the best available device for this build, falling back to CPU.
pub fn select(ordinal: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(ordinal) {
        return Ok(d);
    }
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    if let Ok(d) = Device::new_metal(ordinal) {
        return Ok(d);
    }
    let _ = ordinal;
    Ok(Device::Cpu)
}

#[cfg(all(test, target_os = "macos", not(feature = "cuda")))]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

    // Ignored on CI (no GPU); run locally with: cargo test -p pixl-gen --features gen -- --ignored
    #[test]
    #[ignore]
    fn metal_matmul_roundtrips() {
        let dev = Device::new_metal(0).expect("metal device");
        let a = Tensor::randn(0f32, 1f32, (64, 64), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (64, 64), &dev).unwrap();
        let c = a.matmul(&b).unwrap().to_dtype(DType::F32).unwrap();
        let host = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(host.len(), 64 * 64);
        assert!(host.iter().any(|v| *v != 0.0), "matmul produced all zeros");
    }
}
