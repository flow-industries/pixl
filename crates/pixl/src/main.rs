mod cli;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command, GenerateArgs, PixelizeArgs};
use image::imageops::FilterType;
use pixl_pixelize::{pixelize, PixelizeParams};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Pixelize(args)) => run_pixelize(args),
        Some(Command::Gen(args)) => run_generate(args),
        None => run_generate(cli.generate),
    }
}

fn run_pixelize(args: PixelizeArgs) -> Result<()> {
    let multi = args.inputs.len() > 1;
    if multi {
        if let Some(out) = &args.out {
            std::fs::create_dir_all(out)
                .with_context(|| format!("creating output dir {}", out.display()))?;
        }
    }
    for input in &args.inputs {
        let img = image::open(input)
            .with_context(|| format!("opening {}", input.display()))?
            .to_rgba8();
        let params = PixelizeParams {
            pixel_size: args.pixel_size,
            target_cells: Some(args.target_cells),
            max_colors: args.colors,
            ..Default::default()
        };
        let (small, report) =
            pixelize(&img, &params).with_context(|| format!("pixelizing {}", input.display()))?;
        let out_img = if args.scale > 1 {
            image::imageops::resize(
                &small,
                small.width() * args.scale,
                small.height() * args.scale,
                FilterType::Nearest,
            )
        } else {
            small
        };
        let out_path = resolve_out(input, &args.out, multi);
        out_img
            .save(&out_path)
            .with_context(|| format!("saving {}", out_path.display()))?;
        println!(
            "{} -> {}  [{}x{} cells, {} colors, cell {:.1}px{}]",
            input.display(),
            out_path.display(),
            report.out_cells.0,
            report.out_cells.1,
            report.palette_len,
            report.detected_cell_px.0,
            if report.low_confidence {
                ", low-confidence grid"
            } else {
                ""
            },
        );
    }
    Ok(())
}

fn resolve_out(input: &Path, out: &Option<PathBuf>, multi: bool) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    match out {
        Some(p) if multi => p.join(format!("{stem}.png")),
        Some(p) => p.clone(),
        None => input.with_file_name(format!("{stem}.pixl.png")),
    }
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let count = args
        .count
        .context("missing COUNT (usage: pixl <COUNT> <PROMPT> <OUT_DIR>)")?;
    let prompt = args.prompt.clone().context("missing PROMPT")?;
    let out_dir = args.out_dir.clone().context("missing OUT_DIR")?;
    let (w, h) = cli::parse_size(&args.size).map_err(anyhow::Error::msg)?;
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating output dir {}", out_dir.display()))?;

    println!("pixl: generating {count} image(s) at {w}x{h}");
    println!("  prompt : {prompt:?}");
    println!("  out_dir: {}", out_dir.display());
    println!(
        "  sampler: {} steps, cfg {}, seed {}..{}",
        args.steps,
        args.cfg,
        args.seed,
        args.seed + count as u64 - 1
    );

    #[cfg(feature = "metal")]
    {
        generate_metal(&prompt, count, &out_dir, w, h, &args)
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (w, h);
        anyhow::bail!(
            "this build has no generation backend; rebuild with --features metal (macOS). `pixl pixelize <img>` works without it."
        )
    }
}

#[cfg(feature = "metal")]
fn generate_metal(
    prompt: &str,
    count: u32,
    out_dir: &Path,
    w: u32,
    h: u32,
    args: &GenerateArgs,
) -> Result<()> {
    use pixl_gen::{BaseModel, CandleSdxlGenerator, GenParams, GenRequest, Generator, LoraRef};
    use std::time::Instant;

    let loras = if args.no_lora {
        Vec::new()
    } else {
        vec![LoraRef {
            repo: "nerijs/pixel-art-xl".into(),
            file: "pixel-art-xl.safetensors".into(),
            scale: args.lora_weight,
        }]
    };
    eprintln!("loading SDXL-Turbo (first run downloads weights, ~7 GB, one time)…");
    if !loras.is_empty() {
        eprintln!("  + pixel-art LoRA (weight {})", args.lora_weight);
    }
    let load = Instant::now();
    let generator = CandleSdxlGenerator::load(BaseModel::SdxlTurbo, w, h, &loras)
        .map_err(|e| anyhow::anyhow!("loading generator: {e}"))?;
    eprintln!("ready in {:.1}s", load.elapsed().as_secs_f32());

    let req = GenRequest {
        prompt: prompt.to_string(),
        params: GenParams {
            width: w,
            height: h,
            steps: args.steps,
            guidance: args.cfg,
            base_seed: args.seed,
        },
        loras: vec![],
    };
    let slug = slugify(prompt);

    for i in 0..count {
        let t = Instant::now();
        let gi = generator
            .generate(&req, i as usize)
            .map_err(|e| anyhow::anyhow!("generating image {i}: {e}"))?;
        let out = if args.no_postprocess {
            gi.image
        } else {
            postprocess(&gi.image, args)?
        };
        let path = out_dir.join(format!("{slug}_{i:03}.png"));
        out.save(&path)
            .with_context(|| format!("saving {}", path.display()))?;
        println!(
            "[{}/{count}] seed {} {:.1}s -> {}",
            i + 1,
            gi.seed,
            t.elapsed().as_secs_f32(),
            path.display()
        );
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn postprocess(img: &image::RgbImage, args: &GenerateArgs) -> Result<image::RgbImage> {
    let (w, h) = img.dimensions();
    let mut rgba = image::RgbaImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels() {
        rgba.put_pixel(x, y, image::Rgba([p.0[0], p.0[1], p.0[2], 255]));
    }
    let params = PixelizeParams {
        pixel_size: args.pixel_size,
        max_colors: args.colors,
        ..Default::default()
    };
    let (small, _report) = pixelize(&rgba, &params)?;
    // upscale nearest so the saved file is comfortably viewable
    let scale = (512 / small.width().max(1)).max(1);
    Ok(image::imageops::resize(
        &small,
        small.width() * scale,
        small.height() * scale,
        FilterType::Nearest,
    ))
}

#[cfg(feature = "metal")]
fn slugify(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(40);
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "img".into()
    } else {
        trimmed
    }
}
