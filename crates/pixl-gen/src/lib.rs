//! Generation backend abstraction.
//!
//! The candle/Metal SDXL implementation lands in M2; this crate defines the
//! `Generator` trait and its request/response types now so the CLI and the
//! overlapped pipeline can be built and tested against the seam. Keeping the
//! heavy ML tree behind this boundary is what lets `pixl-pixelize` stay GPU-free.

use std::path::PathBuf;

use image::RgbImage;

/// Sampler / output parameters shared by all backends.
#[derive(Clone, Debug)]
pub struct GenParams {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    /// Per-image seed is `base_seed + index`.
    pub base_seed: u64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 8,
            guidance: 1.0,
            base_seed: 0,
        }
    }
}

/// A LoRA to merge into the base UNet at load time (runtime merge, see M3).
#[derive(Clone, Debug)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub scale: f32,
}

/// One generation request; the backend renders `index` to vary the seed.
#[derive(Clone, Debug)]
pub struct GenRequest {
    pub prompt: String,
    pub params: GenParams,
    pub loras: Vec<LoraSpec>,
}

/// A rendered image plus the seed that produced it (for reproducibility).
pub struct GenImage {
    pub image: RgbImage,
    pub seed: u64,
}

/// Per-step progress hook (denoise step, total steps), driven from the backend's
/// own sampling loop so the CLI can show a live spinner.
pub type StepCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

#[derive(thiserror::Error, Debug)]
pub enum GenError {
    #[error("generation backend not yet implemented (lands in M2: candle/Metal SDXL)")]
    NotImplemented,
    #[error("weights unavailable: {0}")]
    Weights(String),
    #[error("device init failed: {0}")]
    Device(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A local diffusion backend. `CandleSdxlGenerator` will implement this in M2;
/// a remote/subprocess backend could implement the same signature unchanged.
pub trait Generator: Send + Sync {
    fn generate(&self, req: &GenRequest, index: usize) -> Result<GenImage, GenError>;
    /// Install a per-step progress callback. Default no-op.
    fn set_step_callback(&mut self, _cb: StepCallback) {}
}

/// Placeholder backend so the CLI and pipeline compile and run before M2.
pub struct PendingGenerator;

impl Generator for PendingGenerator {
    fn generate(&self, _req: &GenRequest, _index: usize) -> Result<GenImage, GenError> {
        Err(GenError::NotImplemented)
    }
}
