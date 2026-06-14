//! Generation backend abstraction.
//!
//! The `Generator` trait + request/response types compile everywhere. The
//! candle/Metal SDXL backend (`CandleSdxlGenerator`) is behind the `metal`
//! feature so the trait, the CLI, and `pixl-pixelize` stay buildable on CI and
//! Linux without the GPU stack.

use std::path::PathBuf;

use image::RgbImage;

#[cfg(feature = "metal")]
pub mod device;
#[cfg(feature = "metal")]
mod lora;
#[cfg(feature = "metal")]
mod sdxl;
#[cfg(feature = "metal")]
pub use sdxl::{BaseModel, CandleSdxlGenerator, LoraRef};

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

/// Per-step progress hook (denoise step, total steps).
pub type StepCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

#[derive(thiserror::Error, Debug)]
pub enum GenError {
    #[error("generation backend not in this build (rebuild with --features metal)")]
    NotImplemented,
    #[error("weights unavailable: {0}")]
    Weights(String),
    #[error("device init failed: {0}")]
    Device(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A local diffusion backend. `CandleSdxlGenerator` implements this under the
/// `metal` feature; a remote/subprocess backend could implement it unchanged.
pub trait Generator: Send + Sync {
    fn generate(&self, req: &GenRequest, index: usize) -> Result<GenImage, GenError>;
    fn set_step_callback(&mut self, _cb: StepCallback) {}
}

/// Placeholder backend for builds without a generation feature.
pub struct PendingGenerator;

impl Generator for PendingGenerator {
    fn generate(&self, _req: &GenRequest, _index: usize) -> Result<GenImage, GenError> {
        Err(GenError::NotImplemented)
    }
}
