//! Command-line surface. `pixl 100 "stardew valley style house" ./` is the
//! default (generate) form; subcommands cover post-processing existing images
//! and managing the local cache.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "pixl",
    version,
    about = "Local pixel-art generator for Apple Silicon: SDXL + pixel-art LoRA, snapped to true pixel art.",
    args_conflicts_with_subcommands = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Default form (no subcommand): generate `count` images for `prompt` into `out_dir`.
    #[command(flatten)]
    pub generate: GenerateArgs,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Generate images from a prompt (same as the default form).
    Gen(GenerateArgs),
    /// Post-process existing images into true pixel art (no GPU, no model).
    Pixelize(PixelizeArgs),
    /// Inspect or clear the local model / merge cache.
    Models {
        #[command(subcommand)]
        action: ModelsCmd,
    },
    /// Browse a finished run directory in the interactive gallery (no generation).
    View(ViewArgs),
}

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// List cached merged-UNet files and show where weights live.
    Ls,
    /// Delete pixl's merged-UNet cache (not the shared HF weights).
    Clear {
        /// Skip the confirmation prompt.
        #[arg(long, default_value_t = false)]
        yes: bool,
    },
    /// Print the cache directories.
    Path,
}

/// Base diffusion model.
#[derive(ValueEnum, Clone, Copy, Debug, Default)]
pub enum ModelArg {
    /// Few-step, CFG-free, fast.
    #[default]
    Turbo,
    /// Full SDXL base (needs more steps).
    Sdxl,
}

#[derive(Args, Debug, Clone)]
pub struct GenerateArgs {
    /// `[COUNT] PROMPT [OUT_DIR]`: COUNT defaults to 4 and may be omitted
    /// (`pixl "a prompt"`); OUT_DIR defaults to ~/.pixl/<timestamp>-<prompt>/.
    #[arg(value_name = "ARGS", num_args = 0..)]
    pub positional: Vec<String>,

    /// Base model.
    #[arg(long, value_enum, default_value_t = ModelArg::Turbo)]
    pub model: ModelArg,
    /// Palette size for the post-process pass (0 = keep all distinct cell colors).
    #[arg(short = 'c', long)]
    pub colors: Option<u16>,
    /// Force the logical cell size in source pixels (bypass grid detection).
    #[arg(long)]
    pub pixel_size: Option<u32>,
    /// Diffusion steps (default: 8 for Turbo, 25 for SDXL).
    #[arg(long)]
    pub steps: Option<u32>,
    /// Base seed; per-image seed is base + index.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Classifier-free guidance (default: 1.0 = off for Turbo, 7.0 for SDXL).
    #[arg(long)]
    pub cfg: Option<f32>,
    /// Negative prompt (what to avoid). Only takes effect at cfg > 1, i.e. on
    /// `--model sdxl`, not the CFG-free Turbo path.
    #[arg(long, default_value = "")]
    pub negative: String,
    /// Generation resolution, WxH.
    #[arg(long, default_value = "512x512")]
    pub size: String,
    /// Skip the true-pixel-art post-process and save raw generations.
    #[arg(long, default_value_t = false)]
    pub no_postprocess: bool,
    /// Disable the default pixel-art LoRA.
    #[arg(long, default_value_t = false)]
    pub no_lora: bool,
    /// Pixel-art LoRA strength.
    #[arg(long, default_value_t = 1.0)]
    pub lora_weight: f32,
    /// Pixelize/save worker threads (0 = auto).
    #[arg(short = 'j', long, default_value_t = 0)]
    pub jobs: usize,
    /// Emit one JSON line per finished image on stdout.
    #[arg(long, default_value_t = false)]
    pub json: bool,
    /// Suppress progress output.
    #[arg(long, default_value_t = false)]
    pub quiet: bool,
    /// Abort on the first failed image.
    #[arg(long, default_value_t = false)]
    pub fail_fast: bool,
    /// Run at low priority so the batch doesn't bog down the rest of the machine:
    /// macOS background QoS (efficiency cores + I/O throttle) + single-threaded pixelize.
    #[arg(long = "low-prio", visible_aliases = ["bg", "background"], default_value_t = false)]
    pub low_prio: bool,
    /// Force the interactive image gallery on, even on terminals without inline
    /// graphics (falls back to unicode half-blocks).
    #[arg(long, default_value_t = false)]
    pub view: bool,
    /// Never launch the gallery; use the headless batch output.
    #[arg(long, default_value_t = false)]
    pub no_view: bool,
    /// Folder that saved favorites are copied into (default ~/.pixl/saved).
    #[arg(long)]
    pub saved_dir: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct PixelizeArgs {
    /// Input image(s) to snap to true pixel art.
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,
    /// Output file (single input) or directory (multiple inputs). Defaults next to each input.
    #[arg(short = 'o', long)]
    pub out: Option<PathBuf>,
    /// Palette size (0 = keep all distinct cell colors).
    #[arg(short = 'c', long, default_value_t = 16)]
    pub colors: u16,
    /// Force the logical cell size in source pixels (bypass detection).
    #[arg(long)]
    pub pixel_size: Option<u32>,
    /// Fallback logical long-edge (cells) used only when detection fails.
    #[arg(long, default_value_t = 128)]
    pub target_cells: u32,
    /// Upscale the result by this integer factor (nearest) for easy viewing.
    #[arg(long, default_value_t = 1)]
    pub scale: u32,
}

#[derive(Args, Debug, Clone)]
pub struct ViewArgs {
    /// Directory of images to browse.
    pub dir: PathBuf,
    /// Folder that saved favorites are copied into (default ~/.pixl/saved).
    #[arg(long)]
    pub saved_dir: Option<PathBuf>,
}

/// Parse a `WxH` (or `N`) size string into (w, h).
pub fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let s = s.trim().to_lowercase();
    let (w, h) = if let Some((w, h)) = s.split_once('x') {
        (
            w.trim()
                .parse()
                .map_err(|_| format!("bad width in {s:?}"))?,
            h.trim()
                .parse()
                .map_err(|_| format!("bad height in {s:?}"))?,
        )
    } else {
        let n = s.parse().map_err(|_| format!("bad size {s:?}"))?;
        (n, n)
    };
    // SDXL requires positive, multiple-of-8 dimensions; reject early for a clean error.
    if w == 0 || h == 0 || w % 8 != 0 || h % 8 != 0 {
        return Err(format!(
            "size must be positive and divisible by 8, got {w}x{h}"
        ));
    }
    Ok((w, h))
}
