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

/// Cache root (`$XDG_CACHE_HOME` or `$HOME/.cache`, else a temp dir — never
/// CWD-relative). `with_xdg` lets callers that must follow a tool's own contract
/// (e.g. hf-hub, which ignores `XDG_CACHE_HOME`) opt out of the XDG override.
fn cache_root(with_xdg: bool) -> PathBuf {
    if with_xdg {
        if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
            return PathBuf::from(x);
        }
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".cache")
}

/// Directory holding cached merged-UNet safetensors (`~/.cache/pixl/merged`).
pub fn merged_cache_dir() -> PathBuf {
    cache_root(true).join("pixl").join("merged")
}

/// Directory where hf-hub stores downloaded weights (`$HF_HOME/hub` or
/// `~/.cache/huggingface/hub`). hf-hub does not honor `XDG_CACHE_HOME`.
pub fn hf_cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    cache_root(false).join("huggingface").join("hub")
}

/// Whether the merged-UNet cache was reused or freshly built.
#[derive(Clone, Copy, Debug)]
pub enum MergeState {
    None,
    Cached,
    Merged(usize),
}

/// What `CandleSdxlGenerator::load` actually did, for one-line status reporting.
pub struct LoadReport {
    pub model: &'static str,
    pub weights_cached: bool,
    pub lora: Option<(String, f32)>,
    pub merge: MergeState,
}

/// True if the given HF repo already has downloaded snapshots in the local cache.
pub fn hf_model_cached(repo: &str) -> bool {
    let dir = hf_cache_dir()
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(dir)
        .map(|mut it| it.next().is_some())
        .unwrap_or(false)
}

/// Sampler / output parameters shared by all backends.
#[derive(Clone, Debug)]
pub struct GenParams {
    pub steps: u32,
    pub guidance: f32,
    /// Per-image seed is `base_seed + index`.
    pub base_seed: u64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            steps: 8,
            guidance: 1.0,
            base_seed: 0,
        }
    }
}

/// One generation request; the backend renders `index` to vary the seed.
#[derive(Clone, Debug)]
pub struct GenRequest {
    pub prompt: String,
    pub params: GenParams,
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
