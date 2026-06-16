//! SDXL / SDXL-Turbo text-to-image on candle's Metal backend.
//!
//! Models are built once (UNet + VAE in f16, the two CLIP text encoders in f32)
//! and reused across a batch. Mirrors candle's own stable-diffusion example for
//! the dual-CLIP embedding and the denoise loop; the UNet forward is 3-arg (no
//! SDXL micro-conditioning — verified to render fine). Pixel-art LoRAs are
//! merged into the UNet at load time (see `crate::lora`).

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::Module;
use candle_transformers::models::stable_diffusion::{
    self as sd, clip, vae::AutoEncoderKL, StableDiffusionConfig,
};
use hf_hub::api::sync::ApiBuilder;
use image::RgbImage;
use tokenizers::Tokenizer;

use crate::{GenError, GenImage, GenRequest, Generator};

#[derive(Clone, Copy, Debug)]
pub enum BaseModel {
    /// Few-step, CFG-free; candle-native. Best for fast iteration / M2.
    SdxlTurbo,
    /// Full SDXL base (needs more steps without a Lightning LoRA).
    Sdxl,
}

impl BaseModel {
    fn repo(&self) -> &'static str {
        match self {
            BaseModel::SdxlTurbo => "stabilityai/sdxl-turbo",
            BaseModel::Sdxl => "stabilityai/stable-diffusion-xl-base-1.0",
        }
    }
    fn vae_scale(&self) -> f64 {
        match self {
            BaseModel::SdxlTurbo => 0.13025,
            BaseModel::Sdxl => 0.18215,
        }
    }
    fn config(&self, w: usize, h: usize) -> StableDiffusionConfig {
        match self {
            BaseModel::SdxlTurbo => StableDiffusionConfig::sdxl_turbo(None, Some(h), Some(w)),
            BaseModel::Sdxl => StableDiffusionConfig::sdxl(None, Some(h), Some(w)),
        }
    }
}

/// A LoRA to fetch from the HF hub and merge into the UNet at load time.
#[derive(Clone, Debug)]
pub struct LoraRef {
    pub repo: String,
    pub file: String,
    pub scale: f32,
}

pub struct CandleSdxlGenerator {
    device: Device,
    dtype: DType,
    cfg: StableDiffusionConfig,
    vae: AutoEncoderKL,
    unet: sd::unet_2d::UNet2DConditionModel,
    clip1: clip::ClipTextTransformer,
    clip2: clip::ClipTextTransformer,
    tok1: Tokenizer,
    tok2: Tokenizer,
    vae_scale: f64,
    width: usize,
    height: usize,
    step_cb: Option<crate::StepCallback>,
    preview_cb: Option<crate::PreviewCallback>,
}

impl CandleSdxlGenerator {
    pub fn load(
        model: BaseModel,
        width: u32,
        height: u32,
        loras: &[LoraRef],
        progress: Option<crate::ProgressFn>,
    ) -> Result<(Self, crate::LoadReport)> {
        let device = crate::device::select(0).map_err(|e| anyhow!("device: {e}"))?;
        let dtype = DType::F16;
        let (w, h) = (width as usize, height as usize);
        let cfg = model.config(w, h);
        let repo = model.repo();
        let weights_cached = crate::hf_model_cached(repo);

        // With a progress sink, suppress hf-hub's own terminal bars (they would
        // corrupt a TUI) and forward bytes to the sink instead. The cache lookup
        // mirrors `ApiRepo::get` so cached files are never re-downloaded; it must
        // use `Cache::default()` to match `ApiBuilder::new`'s cache dir exactly.
        let api = ApiBuilder::new()
            .with_progress(progress.is_none())
            .build()
            .context("hf-hub api")?;
        let cache = hf_hub::Cache::default();
        let progress = progress.map(|cb| std::sync::Arc::new(std::sync::Mutex::new(cb)));
        let pull = |r: &str, f: &str| -> Result<std::path::PathBuf> {
            let cb = match &progress {
                None => {
                    return api
                        .model(r.to_string())
                        .get(f)
                        .with_context(|| format!("download {r}/{f}"));
                }
                Some(cb) => cb,
            };
            if let Some(p) = cache.model(r.to_string()).get(f) {
                return Ok(p);
            }
            api.model(r.to_string())
                .download_with_progress(f, HubProgress::new(cb.clone(), f))
                .with_context(|| format!("download {r}/{f}"))
        };

        let unet_w = pull(repo, "unet/diffusion_pytorch_model.fp16.safetensors")?;
        // f16 SDXL VAE renders black images (candle #1060) -> fp16-fix VAE is mandatory.
        let vae_w = pull(
            "madebyollin/sdxl-vae-fp16-fix",
            "diffusion_pytorch_model.safetensors",
        )?;
        let clip1_w = pull(repo, "text_encoder/model.fp16.safetensors")?;
        let clip2_w = pull(repo, "text_encoder_2/model.fp16.safetensors")?;
        let tok1_p = pull("openai/clip-vit-large-patch14", "tokenizer.json")?;
        let tok2_p = pull("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")?;

        let vae = cfg.build_vae(&vae_w, &device, dtype).context("build vae")?;
        let (unet_file, merge) = if loras.is_empty() {
            (unet_w, crate::MergeState::None)
        } else {
            let mut specs = Vec::with_capacity(loras.len());
            for l in loras {
                specs.push((pull(&l.repo, &l.file)?, l.scale));
            }
            crate::lora::merged_unet_path(&unet_w, &specs, &crate::merged_cache_dir())
                .map_err(|e| anyhow!("lora merge: {e}"))?
        };
        let unet = cfg
            .build_unet(&unet_file, &device, 4, false, dtype)
            .context("build unet")?;
        let clip1 = sd::build_clip_transformer(&cfg.clip, &clip1_w, &device, DType::F32)
            .context("build clip1")?;
        let clip2 = sd::build_clip_transformer(
            cfg.clip2.as_ref().context("sdxl clip2 config")?,
            &clip2_w,
            &device,
            DType::F32,
        )
        .context("build clip2")?;
        let tok1 = Tokenizer::from_file(&tok1_p).map_err(|e| anyhow!("tokenizer1: {e}"))?;
        let tok2 = Tokenizer::from_file(&tok2_p).map_err(|e| anyhow!("tokenizer2: {e}"))?;

        let report = crate::LoadReport {
            model: match model {
                BaseModel::Sdxl => "SDXL",
                BaseModel::SdxlTurbo => "SDXL-Turbo",
            },
            weights_cached,
            lora: loras.first().map(|l| {
                (
                    l.repo.rsplit('/').next().unwrap_or(&l.repo).to_string(),
                    l.scale,
                )
            }),
            merge,
        };
        Ok((
            Self {
                device,
                dtype,
                vae_scale: model.vae_scale(),
                cfg,
                vae,
                unet,
                clip1,
                clip2,
                tok1,
                tok2,
                width: w,
                height: h,
                step_cb: None,
                preview_cb: None,
            },
            report,
        ))
    }

    fn pad_id(&self, tok: &Tokenizer) -> u32 {
        let key = self.cfg.clip.pad_with.as_deref().unwrap_or("<|endoftext|>");
        tok.get_vocab(true).get(key).copied().unwrap_or(0)
    }

    fn encode(
        &self,
        tok: &Tokenizer,
        clip: &clip::ClipTextTransformer,
        prompt: &str,
        negative: &str,
        use_guide: bool,
    ) -> Result<Tensor> {
        let max = self.cfg.clip.max_position_embeddings;
        let pad = self.pad_id(tok);
        let tokens = |text: &str| -> Result<Tensor> {
            let mut ids = tok
                .encode(text, true)
                .map_err(|e| anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec();
            if ids.len() > max {
                ids.truncate(max);
                // keep a terminator in the final slot rather than a bare content token
                if let Some(last) = ids.last_mut() {
                    *last = pad;
                }
            }
            while ids.len() < max {
                ids.push(pad);
            }
            Ok(Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?)
        };
        let cond = clip.forward(&tokens(prompt)?)?;
        if use_guide {
            let uncond = clip.forward(&tokens(negative)?)?;
            Ok(Tensor::cat(&[uncond, cond], 0)?.to_dtype(self.dtype)?)
        } else {
            Ok(cond.to_dtype(self.dtype)?)
        }
    }

    fn text_embeddings(&self, prompt: &str, negative: &str, use_guide: bool) -> Result<Tensor> {
        let e1 = self.encode(&self.tok1, &self.clip1, prompt, negative, use_guide)?;
        let e2 = self.encode(&self.tok2, &self.clip2, prompt, negative, use_guide)?;
        Ok(Tensor::cat(&[e1, e2], D::Minus1)?)
    }

    fn render(
        &self,
        prompt: &str,
        negative: &str,
        steps: usize,
        guidance: f64,
        seed: u64,
    ) -> Result<RgbImage> {
        self.device.set_seed(seed)?;
        let use_guide = guidance > 1.0;
        let text = self.text_embeddings(prompt, negative, use_guide)?;

        let mut scheduler = self.cfg.build_scheduler(steps)?;
        let (lh, lw) = (self.height / 8, self.width / 8);
        let mut latents = (Tensor::randn(0f32, 1f32, (1, 4, lh, lw), &self.device)?
            * scheduler.init_noise_sigma())?
        .to_dtype(self.dtype)?;

        let timesteps = scheduler.timesteps().to_vec();
        let total = timesteps.len();
        for (si, &t) in timesteps.iter().enumerate() {
            let input = if use_guide {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };
            let input = scheduler.scale_model_input(input, t)?;
            let noise = self.unet.forward(&input, t as f64, &text)?;
            let noise = if use_guide {
                let c = noise.chunk(2, 0)?;
                (&c[0] + ((&c[1] - &c[0])? * guidance)?)?
            } else {
                noise
            };
            latents = scheduler.step(&noise, t, &latents)?;
            if let Some(cb) = &self.step_cb {
                cb(si + 1, total);
            }
            if let Some(cb) = &self.preview_cb {
                if let Ok(img) = self.latent_preview(&latents) {
                    cb(img);
                }
            }
        }

        let img = self.vae.decode(&(latents / self.vae_scale)?)?;
        let img = ((img / 2.0)? + 0.5)?
            .clamp(0f32, 1f32)?
            .to_device(&Device::Cpu)?;
        let img = (img * 255.0)?.to_dtype(DType::U8)?.i(0)?;
        tensor_to_rgb(&img)
    }
}

// SDXL latent -> approximate RGB factors (ComfyUI), row-major 4 channels x 3 rgb,
// calibrated with the bias so `latent . factors + bias` is already ~[0,1].
const PREVIEW_FACTORS: [f32; 12] = [
    0.3651, 0.4232, 0.4341, //
    -0.2533, -0.0042, 0.1068, //
    0.1076, 0.1111, -0.0362, //
    -0.3165, -0.2492, -0.2188,
];
const PREVIEW_BIAS: [f32; 3] = [0.1084, -0.0175, -0.0011];

impl CandleSdxlGenerator {
    /// Cheap linear decode of the current latent to a rough RGB preview (no VAE),
    /// at latent resolution. Used for the in-flight gallery preview.
    fn latent_preview(&self, latents: &Tensor) -> Result<RgbImage> {
        let lat = latents.squeeze(0)?.to_dtype(DType::F32)?; // (4, lh, lw)
        let (c, h, w) = lat.dims3()?;
        let flat = lat.reshape((c, h * w))?.t()?.contiguous()?; // (lh*lw, 4)
        let factors = Tensor::from_slice(&PREVIEW_FACTORS, (4, 3), &self.device)?;
        let bias = Tensor::from_slice(&PREVIEW_BIAS, (1, 3), &self.device)?;
        let rgb = flat
            .matmul(&factors)?
            .broadcast_add(&bias)?
            .clamp(0f32, 1f32)?;
        let rgb = (rgb * 255.0)?
            .to_dtype(DType::U8)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        RgbImage::from_raw(w as u32, h as u32, rgb)
            .ok_or_else(|| anyhow!("preview buffer mismatch"))
    }
}

fn tensor_to_rgb(chw: &Tensor) -> Result<RgbImage> {
    let (c, h, w) = chw.dims3()?;
    anyhow::ensure!(c == 3, "expected 3 channels, got {c}");
    let data = chw.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    RgbImage::from_raw(w as u32, h as u32, data).ok_or_else(|| anyhow!("rgb buffer size mismatch"))
}

/// Forwards hf-hub download bytes to a [`crate::ProgressFn`], coalescing updates
/// to ~4 MB so the UI isn't flooded.
struct HubProgress {
    cb: std::sync::Arc<std::sync::Mutex<crate::ProgressFn>>,
    file: String,
    done: u64,
    total: u64,
    last: u64,
}

fn short_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

impl HubProgress {
    fn new(cb: std::sync::Arc<std::sync::Mutex<crate::ProgressFn>>, file: &str) -> Self {
        Self {
            cb,
            file: short_name(file),
            done: 0,
            total: 0,
            last: 0,
        }
    }
    fn emit(&self) {
        if let Ok(mut cb) = self.cb.lock() {
            (cb)(crate::DownloadProgress {
                file: self.file.clone(),
                done: self.done,
                total: self.total,
            });
        }
    }
}

impl hf_hub::api::Progress for HubProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.total = size as u64;
        self.done = 0;
        self.last = 0;
        self.file = short_name(filename);
        self.emit();
    }
    fn update(&mut self, size: usize) {
        self.done += size as u64;
        if self.done.saturating_sub(self.last) >= 4_000_000 || self.done >= self.total {
            self.last = self.done;
            self.emit();
        }
    }
    fn finish(&mut self) {
        self.done = self.total;
        self.emit();
    }
}

impl Generator for CandleSdxlGenerator {
    fn generate(&self, req: &GenRequest, index: usize) -> Result<GenImage, GenError> {
        let seed = req.params.base_seed + index as u64;
        let image = self
            .render(
                &req.prompt,
                &req.negative,
                req.params.steps as usize,
                req.params.guidance as f64,
                seed,
            )
            .map_err(|e| GenError::Backend(e.to_string()))?;
        Ok(GenImage { image, seed })
    }

    fn set_step_callback(&mut self, cb: crate::StepCallback) {
        self.step_cb = Some(cb);
    }

    fn set_preview_callback(&mut self, cb: crate::PreviewCallback) {
        self.preview_cb = Some(cb);
    }
}
