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
    let prompt = args.prompt.context("missing PROMPT")?;
    let out_dir = args.out_dir.context("missing OUT_DIR")?;
    let (w, h) = cli::parse_size(&args.size).map_err(anyhow::Error::msg)?;
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating output dir {}", out_dir.display()))?;

    // The generation backend (candle/Metal SDXL + runtime-merged pixel-art &
    // Lightning LoRAs) lands in M2; the overlapped pipeline that drives it and the
    // pixelize pass is M4. The Generator seam already exists in `pixl-gen`.
    println!("pixl: would generate {count} image(s) at {w}x{h}");
    println!("  prompt : {prompt:?}");
    println!("  out_dir: {}", out_dir.display());
    println!(
        "  sampler: {} steps, cfg {}, seed {}..{}",
        args.steps,
        args.cfg,
        args.seed,
        args.seed + count as u64 - 1
    );
    println!(
        "  post   : {}",
        if args.no_postprocess {
            "raw (no pixel-snap)".to_string()
        } else {
            format!(
                "{} colors{}",
                args.colors,
                args.pixel_size
                    .map(|p| format!(", {p}px cells"))
                    .unwrap_or_default()
            )
        }
    );
    anyhow::bail!(
        "generation backend not yet wired (M2). Use `pixl pixelize <img>` to post-process existing images today."
    );
}
