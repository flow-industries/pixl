mod cli;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command, GenerateArgs, ModelsCmd, PixelizeArgs};
use image::imageops::FilterType;
use pixl_pixelize::{pixelize, PixelizeParams};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Pixelize(args)) => run_pixelize(args),
        Some(Command::Models { action }) => run_models(action),
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

    #[cfg(feature = "metal")]
    {
        generate_metal(&prompt, count, &out_dir, w, h, &args)
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (w, h, count, &prompt);
        anyhow::bail!(
            "this build has no generation backend; rebuild with --features metal (macOS). `pixl pixelize <img>` works without it."
        )
    }
}

#[cfg(feature = "metal")]
fn push_fail(failures: &std::sync::Mutex<Vec<(usize, String)>>, i: usize, msg: String) {
    failures
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((i, msg));
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
    use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
    use pixl_gen::{
        BaseModel, CandleSdxlGenerator, GenImage, GenParams, GenRequest, Generator, LoraRef,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    let loras = if args.no_lora {
        Vec::new()
    } else {
        vec![LoraRef {
            repo: "nerijs/pixel-art-xl".into(),
            file: "pixel-art-xl.safetensors".into(),
            scale: args.lora_weight,
        }]
    };

    if !args.quiet {
        eprintln!("loading SDXL-Turbo (first run downloads weights, ~7 GB, one time)…");
        if !loras.is_empty() {
            eprintln!("  + pixel-art LoRA (weight {})", args.lora_weight);
        }
    }
    if !args.lora_weight.is_finite() {
        anyhow::bail!("--lora-weight must be a finite number");
    }
    let model = match args.model {
        cli::ModelArg::Sdxl => BaseModel::Sdxl,
        cli::ModelArg::Turbo => BaseModel::SdxlTurbo,
    };
    if args.cfg > 1.0 && matches!(model, BaseModel::SdxlTurbo) && !args.quiet {
        eprintln!("note: --cfg > 1 has no effect on SDXL-Turbo (CFG-distilled); use --model sdxl for guidance");
    }
    let mut generator = CandleSdxlGenerator::load(model, w, h, &loras)
        .map_err(|e| anyhow::anyhow!("loading generator: {e}"))?;

    let draw = if args.quiet {
        ProgressDrawTarget::hidden()
    } else {
        ProgressDrawTarget::stderr()
    };
    let mp = MultiProgress::with_draw_target(draw);
    let overall = mp.add(ProgressBar::new(count as u64));
    overall.set_style(
        ProgressStyle::with_template(
            "{prefix:>8.cyan.bold} [{bar:28.cyan/blue}] {pos}/{len} · {elapsed_precise} · eta {eta}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    overall.set_prefix("pixl");
    let gen_spin = mp.add(ProgressBar::new_spinner());
    gen_spin.enable_steady_tick(Duration::from_millis(100));
    gen_spin.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());

    let cur = Arc::new(AtomicUsize::new(0));
    {
        let spin = gen_spin.clone();
        let cur = cur.clone();
        generator.set_step_callback(Box::new(move |step, steps| {
            spin.set_message(format!(
                "image {}/{count} · diffusing {step}/{steps}",
                cur.load(Ordering::Relaxed) + 1
            ));
        }));
    }

    let cancel = Arc::new(AtomicBool::new(false));
    {
        let cancel = cancel.clone();
        let _ = ctrlc::set_handler(move || cancel.store(true, Ordering::Relaxed));
    }

    let req = GenRequest {
        prompt: prompt.to_string(),
        params: GenParams {
            steps: args.steps,
            guidance: args.cfg,
            base_seed: args.seed,
        },
    };
    let slug = slugify(prompt);
    let jobs = if args.jobs > 0 {
        args.jobs
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get().min(2))
            .unwrap_or(1)
    };

    let (tx, rx) = crossbeam_channel::bounded::<(usize, GenImage)>(jobs);
    let failures: Arc<Mutex<Vec<(usize, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let saved = Arc::new(AtomicUsize::new(0));
    let abort = Arc::new(AtomicBool::new(false)); // fail-fast (distinct from Ctrl-C cancel)
    let started = Instant::now();

    std::thread::scope(|scope| {
        for _ in 0..jobs {
            let rx = rx.clone();
            let overall = overall.clone();
            let failures = failures.clone();
            let out_dir = out_dir.to_path_buf();
            let slug = slug.clone();
            let args = args.clone();
            let saved = saved.clone();
            scope.spawn(move || {
                for (i, gi) in rx.iter() {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        pixelize_and_save(gi, i, &out_dir, &slug, &args)
                    }));
                    match r {
                        Ok(Ok(json)) => {
                            saved.fetch_add(1, Ordering::Relaxed);
                            if args.json {
                                println!("{json}");
                            }
                        }
                        Ok(Err(e)) => push_fail(&failures, i, e.to_string()),
                        Err(_) => push_fail(&failures, i, "panicked during pixelize/save".into()),
                    }
                    overall.inc(1);
                }
            });
        }
        drop(rx);

        for i in 0..count as usize {
            if cancel.load(Ordering::Relaxed) || abort.load(Ordering::Relaxed) {
                break;
            }
            cur.store(i, Ordering::Relaxed);
            match generator.generate(&req, i) {
                Ok(gi) => {
                    if tx.send((i, gi)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    push_fail(&failures, i, e.to_string());
                    if args.fail_fast {
                        abort.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }
        drop(tx);
    });

    gen_spin.finish_and_clear();
    overall.finish_and_clear();

    let fails = failures.lock().unwrap_or_else(|e| e.into_inner());
    let ok = saved.load(Ordering::Relaxed);
    let cancelled = cancel.load(Ordering::Relaxed);
    if !args.quiet {
        let secs = started.elapsed().as_secs_f32();
        eprintln!(
            "{} {ok}/{count} saved · {secs:.1}s · {:.1}s/img{}{} -> {}",
            if fails.is_empty() && !cancelled {
                "✓"
            } else {
                "•"
            },
            if ok > 0 { secs / ok as f32 } else { 0.0 },
            if cancelled { " · cancelled" } else { "" },
            if fails.is_empty() {
                String::new()
            } else {
                format!(" · {} failed", fails.len())
            },
            out_dir.display(),
        );
        for (i, e) in fails.iter() {
            eprintln!("  image {i}: {e}");
        }
    }
    if cancelled {
        std::process::exit(130);
    }
    if !fails.is_empty() {
        anyhow::bail!("{} image(s) failed", fails.len());
    }
    Ok(())
}

/// Pixelize (unless disabled) + save one generated image. Returns a JSON summary.
#[cfg(feature = "metal")]
fn pixelize_and_save(
    gi: pixl_gen::GenImage,
    i: usize,
    out_dir: &Path,
    slug: &str,
    args: &GenerateArgs,
) -> Result<String> {
    let path = out_dir.join(format!("{slug}_{i:03}.png"));
    let (cells, colors) = if args.no_postprocess {
        gi.image
            .save(&path)
            .with_context(|| format!("saving {}", path.display()))?;
        ((gi.image.width(), gi.image.height()), 0u16)
    } else {
        let (out, report) = postprocess(&gi.image, args)?;
        out.save(&path)
            .with_context(|| format!("saving {}", path.display()))?;
        (report.out_cells, report.palette_len)
    };
    Ok(serde_json::json!({
        "index": i,
        "seed": gi.seed,
        "path": path.to_string_lossy(),
        "cells": [cells.0, cells.1],
        "colors": colors,
    })
    .to_string())
}

#[cfg(feature = "metal")]
fn postprocess(
    img: &image::RgbImage,
    args: &GenerateArgs,
) -> Result<(image::RgbImage, pixl_pixelize::PixelizeReport)> {
    let (w, h) = img.dimensions();
    let rgba = image::RgbaImage::from_fn(w, h, |x, y| {
        let p = img.get_pixel(x, y).0;
        image::Rgba([p[0], p[1], p[2], 255])
    });
    let params = PixelizeParams {
        pixel_size: args.pixel_size,
        max_colors: args.colors,
        ..Default::default()
    };
    let (small, report) = pixelize(&rgba, &params)?;
    let scale = (512 / small.width().max(1)).max(1);
    let up = image::imageops::resize(
        &small,
        small.width() * scale,
        small.height() * scale,
        FilterType::Nearest,
    );
    Ok((up, report))
}

#[cfg(feature = "metal")]
fn slugify(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = out[..out.len().min(40)].trim_matches('_').to_string();
    if trimmed.is_empty() {
        "img".into()
    } else {
        trimmed
    }
}

fn run_models(action: ModelsCmd) -> Result<()> {
    let merged = pixl_gen::merged_cache_dir();
    match action {
        ModelsCmd::Path => {
            println!("merged cache: {}", merged.display());
            println!("hf weights  : {}", pixl_gen::hf_cache_dir().display());
        }
        ModelsCmd::Ls => {
            let (files, total) = list_merged(&merged);
            if files.is_empty() {
                println!("no merged-UNet cache yet ({})", merged.display());
            } else {
                for (name, sz) in &files {
                    println!("  {:>9}  {name}", human(*sz));
                }
                println!("  total {} in {}", human(total), merged.display());
            }
            println!("hf weights cache: {}", pixl_gen::hf_cache_dir().display());
        }
        ModelsCmd::Clear { yes } => {
            let (files, total) = list_merged(&merged);
            if files.is_empty() {
                println!("nothing to clear ({})", merged.display());
                return Ok(());
            }
            // structural guard: the path must end in exactly .../pixl/merged and not be a symlink
            let tail: Vec<String> = merged
                .components()
                .rev()
                .take(2)
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            if tail != ["merged", "pixl"] {
                anyhow::bail!(
                    "refusing to clear unexpected cache path {}",
                    merged.display()
                );
            }
            if std::fs::symlink_metadata(&merged)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false)
            {
                anyhow::bail!(
                    "cache path is a symlink, refusing to clear {}",
                    merged.display()
                );
            }
            if !yes {
                if !std::io::stdin().is_terminal() {
                    anyhow::bail!(
                        "not a terminal; pass --yes to clear {} ({})",
                        merged.display(),
                        human(total)
                    );
                }
                eprint!(
                    "delete {} merged file(s) ({}) from {}? [y/N] ",
                    files.len(),
                    human(total),
                    merged.display()
                );
                std::io::Write::flush(&mut std::io::stderr())?;
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line).is_err()
                    || !line.trim().eq_ignore_ascii_case("y")
                {
                    println!("aborted");
                    return Ok(());
                }
            }
            // remove only the known cache files, then the now-empty dir
            for (name, _) in &files {
                let f = merged.join(name);
                std::fs::remove_file(&f).with_context(|| format!("removing {}", f.display()))?;
            }
            let _ = std::fs::remove_dir(&merged);
            println!("cleared {} ({})", human(total), merged.display());
        }
    }
    Ok(())
}

fn list_merged(dir: &Path) -> (Vec<(String, u64)>, u64) {
    let mut out = Vec::new();
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                if md.is_file() {
                    total += md.len();
                    out.push((e.file_name().to_string_lossy().into_owned(), md.len()));
                }
            }
        }
    }
    out.sort();
    (out, total)
}

fn human(bytes: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut b = bytes as f64;
    let mut i = 0;
    while b >= 1024.0 && i < U.len() - 1 {
        b /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{bytes} B")
    } else {
        format!("{b:.1} {}", U[i])
    }
}
