# pixl — Design & Implementation Plan

> **Status:** Active. **All milestones (M0–M5) are implemented and green.**
> `pixl N "prompt" ./out` generates on Metal (SDXL + runtime-merged pixel-art
> LoRA), pixelizes via an overlapped pipeline with per-image progress, saves true
> pixel art, and ships cache management + packaging.
>
> This plan was produced by a multi-agent research+design pass and hardened by
> two adversarial verifiers (candle/LoRA feasibility; pixelize-algorithm
> soundness). Their corrections are folded in.

**pixl is a single self-contained Rust CLI that generates pixel-art images
locally on Apple Silicon (SDXL + a pixel-art LoRA on candle's Metal backend) and
post-processes each into TRUE pixel art (grid-detect → cell-collapse →
limited-palette quantize) — `pixl 100 "stardew valley style house" ./`.**

---

## Implementation status

| Milestone | State |
|---|---|
| M0 — workspace scaffold, CLI surface, `Generator` seam | **done** |
| M1 — `pixl-pixelize` algorithm + synthetic golden tests (no GPU) | **done** |
| M2 — candle SDXL, one image on Metal | **done** |
| M3 — runtime LoRA merge (sgm→diffusers key map) | **done** |
| M4 — overlapped generate→pixelize→save pipeline + progress UX | planned |
| M5 — weights UX, `pixl models` cache mgmt, packaging | **done** |

### Post-M5 adversarial review — fixes applied

A 5-reviewer + verifier pass (0 critical, 1 high, several mediums confirmed) ran
before commit; the real findings are fixed:

- **[high] degenerate grid** — tiny / near-uniform images fabricated a confident
  2–3px grid because the coverage window spans the whole fold at small periods.
  Fixed by a `MIN_PERIOD` floor **and** normalizing coverage against the uniform
  baseline `w/t`, so only genuine concentration scores high; near-uniform inputs
  now fall back. New golden test covers it.
- **pixelize** — borrowed-axis phase is recomputed under the borrowed period (was
  carried from a different modular space); NaN-safe `total_cmp` in detrend.
- **cache safety** — `models clear` writes its prompt to stderr, refuses on a
  non-TTY without `--yes`, structurally verifies the path is `…/pixl/merged` and
  not a symlink, and removes known files then the dir (not a blind `remove_dir_all`);
  cache dirs use XDG/HOME/temp (never CWD-relative when HOME is unset).
- **LoRA merge** — per-process temp file (no concurrent-merge corruption), guards
  on rank/finite-scale and non-2D weights (skip, don't poison/​panic), honest
  cache-key doc (path+scale, not content).
- **pipeline** — summary counts images actually `saved` (correct under Ctrl-C);
  fail-fast uses a distinct flag (real failures exit 1 + `bail`, not the Ctrl-C
  130/"cancelled"); worker panics are caught per-image and Mutex poisoning is
  tolerated, so one bad image can't abort the batch's accounting.
- **generation defaults** — `--size` defaults to **512²** (Turbo-native; 1024 ran
  Turbo at 4× cost) and is validated (positive, ÷8) for a clean error not a candle
  panic; over-long prompts keep a terminator token; `--cfg>1` on Turbo warns;
  `--lora-weight` must be finite.

### M5 results — UX, cache management, packaging

- **First-run download progress** via `hf-hub` `ApiBuilder::with_progress(true)`.
- **`--model turbo|sdxl`** selects the base model (Turbo default).
- **`pixl models ls | clear | path`** — list the merged-UNet cache with sizes,
  clear it (confirmation prompt + a refuse-if-path-unexpected guard; never
  touches the shared HF weights), and print cache locations. Un-gated (works in
  the GPU-free build). Verified without a GPU.
- **Packaging:** `cargo install --path crates/pixl [--features metal]`, a
  `justfile` (build / build-metal / test / lint / install / demo), and a rewritten
  README (install, usage, resource throttling).

### M4 results — overlapped pipeline + progress

- `generate → pixelize → save` is a bounded `crossbeam-channel` (cap = `--jobs`)
  with `std::thread::scope` workers: generation runs serially on the single Metal
  queue (producer) while pixelize+save of image *i* overlaps diffusion of *i+1*.
- `indicatif` MultiProgress (stderr): overall `[k/N]` bar + a spinner driven by a
  **denoise-step callback** on the `Generator` (`image i/N · diffusing s/S`).
- Flags: `--jobs` (default 2), `--json` (one JSONL event per finished image on
  stdout, bars on stderr), `--quiet`, `--fail-fast`. Ctrl-C cancels between
  images and exits 130; per-image failures are collected (exit 1 if any).
- Verified end-to-end (2 imgs, JSONL + summary, no deadlock).
- **Resource note:** generation is GPU-bound and can saturate an integrated GPU.
  Default `--jobs` lowered to 2; README documents a `taskpolicy -b` / `nice` /
  `RAYON_NUM_THREADS` throttle. The `autoreleasepool` Metal-leak guard remains
  deferred (M2's mmap path showed flat memory; revisit on long real batches).

### M3 results — runtime LoRA merge

- Pixel-art SDXL LoRAs ship in **sgm/A1111 naming** (`lora_unet_input_blocks_4_1_…`)
  while candle's UNet uses **diffusers naming** — these are different schemes, not
  a dots-vs-underscores difference (the verifier's flagged risk, sharpened).
- Since these LoRAs only touch attention blocks, the mapper is just the
  **attention-block envelope** (11 entries) + inner pass-through, not mold's full
  checkpoint converter. **Verified: `nerijs/pixel-art-xl` maps 722/722 modules,
  0 skipped** against the real base keys. Output is visibly pixel-art-styled.
- **Merge strategy:** merging in-memory via `VarBuilder::from_tensors` pinned ~5 GB
  and made generation 3× slower and *escalating* (22→100→119 s — memory pressure).
  Fixed by merging on CPU and writing a **content-addressed merged-UNet cache**,
  then loading it through candle's normal **mmap path** — generation is back to the
  flat ~7.5 s/image baseline. First merge ~one-time (4.8 GB cache per LoRA combo);
  subsequent runs hit the cache (load ~11 s).
- Default: SDXL-Turbo + `nerijs/pixel-art-xl` (`--lora-weight`, `--no-lora`). The
  mapper is generic kohya-SDXL, so SDXL-Lightning would slot in via the same path
  (not wired — Turbo already provides few-step speed).

### M2 results (measured on the M4 Pro)

- candle 0.10.2 + Metal **builds and runs** here (GPU matmul smoke test passes).
- SDXL-Turbo @ 4 steps, 512², dev build: **~7.5 s/image steady-state**; one-time
  weight download ~7 GB (~33 min on this network), cached load ~11 s thereafter.
- Output is coherent (not black) — the **fp16-fix VAE is mandatory** and applied.
- Full `generate → pixelize` chain produces a clean true-pixel-art sprite.
- **Memory:** RSS flat (~90 MB anonymous; weights mmap'd + in Metal unified
  memory) and per-image time steady across an 8-image loop — no leak observed,
  so the `objc2::autoreleasepool` guard is deferred to M4's long-run pipeline.
- The candle/Metal stack is behind an optional **`metal` feature** (default off)
  so default builds + CI stay GPU-free; `cargo build --features metal` (macOS)
  builds the real generator.

### Deviations adopted during M1 (intentional, documented)

- **Color + k-means are hand-rolled** (`color.rs`, `palette.rs`) instead of the
  `palette` / `kmeans_colors` / `deltae` crates the plan named. Reason:
  dependency-light, fully deterministic, zero C deps — `pixl-pixelize` depends
  only on `image`, `rayon`, `thiserror`. CIEDE2000 fixed-palette snapping is
  deferred (the default path uses Lab k-means / CIE76).
- **Grid detection uses a modular-fold coverage score**, not the Canny→Hough
  pipeline. After the verifier's harmonic analysis this proved both simpler and
  more robust: real grid boundaries always land at multiples of the true cell
  size, so folding the per-axis change-signal modulo a candidate period
  concentrates ~all edge energy into one phase bin for the fundamental, while
  harmonics scatter it. The largest period clearing a coverage threshold wins.
  (imageproc Canny/Hough remains a documented future fallback, unshipped.)

---

## 1. Guiding principles

- **Local-only, single clean Rust binary, zero Python at runtime.** The only
  tolerated non-Rust step is a *documented fallback* (Python LoRA bake, see
  `bake/`) that the default path does not use — LoRAs merge at runtime in Rust.
- **Two-stage pipeline.** Stage 1 = diffusion (GPU, serial — one Metal queue).
  Stage 2 = pixelize (CPU, parallel via rayon). The pixelize stage is what makes
  output *true* pixel art, so the generator only needs blocky structure, not
  perfect fidelity — which justifies aggressive few-step sampling.
- **The dependency cut IS the architecture.** `pixl-pixelize` (pure CPU, no
  candle, golden-testable on CI) is split from `pixl-gen` (the ML/Metal tree
  behind a `Generator` trait). Single most valuable cleanliness property.
- **Decisive defaults, escape hatches everywhere.** `--pixel-size`, `--colors`,
  `--steps`, `--lora` override; detection degrades gracefully to a never-crash
  fixed-size downscale.
- **Reproducibility.** `seed_i = base_seed + i`; fixed k-means seed; deterministic
  pixelize. Same inputs ⇒ identical output.

## 2. Tech stack

- **Workspace, 3 crates:** `pixl-pixelize` (lib, pure CPU), `pixl-gen` (lib,
  candle/Metal), `pixl` (bin). Keeps candle out of the testable core.
- **candle `0.10.x` upstream** (no fork). Metal via `Device::new_metal(0)`,
  `DType::F16` for UNet/VAE, `DType::F32` for the CLIP text encoders.
- **LoRA: runtime `SimpleBackend` merge** wrapping the mmap'd UNet safetensors
  (§3). Build the UNet from a self-constructed `UNet2DConditionModelConfig` — do
  NOT call `sd_config.unet()` (that accessor exists only on mold's fork).
- **Palette:** hand-rolled Lab k-means (M1). `kmeans_colors` rejected to avoid
  the dep; `imagequant` rejected (GPL).
- **No async runtime.** `hf-hub` blocking API; GPU serial; pixelize via rayon.

## 3. Generation pipeline (M2–M3)

### Models (HF repo IDs — diffusers component layout, not single-file)

| Component | Repo | File (f16) |
|---|---|---|
| UNet | `stabilityai/stable-diffusion-xl-base-1.0` | `unet/diffusion_pytorch_model.fp16.safetensors` |
| Text enc 1 | same | `text_encoder/model.fp16.safetensors` |
| Text enc 2 | same | `text_encoder_2/model.fp16.safetensors` |
| **VAE (mandatory fp16-fix)** | `madebyollin/sdxl-vae-fp16-fix` | `diffusion_pytorch_model.safetensors` |
| Tokenizer 1 | `openai/clip-vit-large-patch14` | `tokenizer.json` |
| Tokenizer 2 | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | `tokenizer.json` |
| Pixel-art LoRA | `nerijs/pixel-art-xl` | `pixel-art-xl.safetensors` (kohya `lora_unet_*`, scale 1.2, trigger "pixel art") |
| Speed LoRA | `ByteDance/SDXL-Lightning` | `sdxl_lightning_8step_lora.safetensors` (scale 1.0) |

> The stock f16 SDXL VAE renders **black images** (candle #1060). The fp16-fix
> VAE is **mandatory**.

### Default sampler / resolution

Primary: SDXL base + pixel-art LoRA (1.2) + SDXL-Lightning 8-step LoRA (1.0),
both runtime-merged into one UNet. **8 steps, EulerAncestralDiscrete,
`TimestepSpacing::Trailing`, `PredictionType::Epsilon`, CFG 1.0** (Lightning is
CFG-distilled → single forward pass per step). **1024×1024** → pixelize target
**128 logical px** (8px cells). Fallback fast path: **SDXL-Turbo** (`stabilityai/
sdxl-turbo`, vae_scale 0.13025, 4 steps, CFG 0.0) — candle ships Turbo turnkey.

Verifier note: candle's EulerAncestral is *ancestral*; Lightning was tuned on
non-ancestral Euler. Validate visually at M3; if 8-step ancestral noise hurts,
switch to `DDIMScheduler` (also Trailing+Epsilon).

### candle denoise loop (key facts)

`unet.forward(&sample, t, &encoder_hidden_states)` is **3-arg** — no SDXL
micro-conditioning (`add_time_ids` / pooled `add_text_embeds`). Verified to
render fine; accept it. Build CLIP encoders in **F32** (candle forces this).
Build the scheduler independently (the `scheduler` field is private). Seed via
`device.set_seed(base_seed + i)` before latent init.

### LoRA approach — runtime merge via `SimpleBackend`

Wrap the mmap'd base-UNet backend so each weight `get()` returns `W + Σ
scale·(B@A)`; `UNet2DConditionModel::new` never knows LoRA exists. All merge math
in **F32**. `effective_scale = user_scale · alpha/rank` when the file carries
`.alpha` (kohya), else `user_scale`. Conv-1×1 LoRAs need reshape-matmul-reshape;
SDXL attention has no fused QKV, so every target is a plain `Direct` add.

Key mapping verified against candle `attention.rs` (`to_q/to_k/to_v/to_out.0`,
`ff.net.0.proj`, `ff.net.2`, `proj_in/proj_out`): parse the kohya stem and
reverse-map leaves via an explicit kohya→candle table. Skip `lora_te*`
(text-encoder LoRA out of scope for v1; nerijs and Lightning are UNet-only).

**Critical:** build the UNet via a self-constructed `UNet2DConditionModelConfig`
(copy the literal SDXL config from candle's `sdxl_()` body); `use_flash_attn =
false` always (CUDA-only). **First integration test:** load the LoRA on CPU,
assert every UNet stem maps and only `lora_te*` are skipped.

Fallback (documented, not default): one-time Python `fuse_lora` bake → cached
merged unet loaded via plain `build_unet`. See `bake/`.

### Weights / cache

`hf-hub` sync `Api` → standard HF cache, content-addressed. First run prints a
bytes bar (`Downloading SDXL ~6.9 GB, one time…`); subsequent runs silent. Build
all models once (~7 GB f16, mmap'd), loop generation. `pixl models pull/ls`.

## 4. Pixelize post-processing (M1 — implemented)

Pure CPU, deterministic. Stages and the implementing module:

| Stage | Module | Default |
|---|---|---|
| Trim border | `lib.rs` | `trim_border=2` |
| Per-axis ΔE change signal (CIE76, alpha-masked) | `signal.rs` | — |
| Detrend (sliding median, rejects gradients) | `signal.rs` | window 9 |
| Period+phase via modular-fold coverage (harmonic-proof) | `grid.rs` | threshold 0.78 |
| Grid reconstruct + head/tail sliver merge | `grid.rs` | merge < 0.4·period |
| Per-cell dominant color (coarse Lab bin), rayon-parallel | `collapse.rs` | bin ΔE 12 |
| Lab k-means palette (deterministic, fixed seed) | `palette.rs` | colors 16 |

**Degradation ladder:** `--pixel-size P` forces `T=P, φ=0` (always wins). If one
axis is weak, borrow the strong axis's period (AI pixel art is near-square). If
both fail, fixed downscale to `target_cells` long-edge, flagged
`low_confidence`. Never crashes.

**Verified by golden tests** (`tests/golden.rs`, no GPU): clean upscale recovery,
16px/32-color, phase offset, non-integer scale (×31.25), `--pixel-size` override,
and idempotence — all green.

Future (not in M1): ±5° skew pre-search, JPEG-8 false-grid guard, dither-vs-noise
bimodal guard, CIEDE2000 fixed-palette snapping.

## 5. CLI & progress UX

`pixl <count> <prompt> <out_dir>` is the default form; subcommands: `gen`,
`pixelize` (works today, no GPU), `models` (M5). Flags: `--colors/-c`,
`--pixel-size`, `--steps`, `--seed`, `--cfg`, `--size`, `--no-postprocess`,
`--scale` (pixelize viewing), and (M4) `--lora`, `--jobs`, `--json`, `--quiet`,
`--fail-fast`.

**Overlapped pipeline (M4):** generation is serial on the single Metal queue; a
bounded `crossbeam-channel` (cap = jobs) lets pixelize+save of image *i* overlap
diffusion of *i+1* via a rayon pool. Pixelize (~tens of ms) hides under a gen
step. `indicatif` MultiProgress: overall `[k/N]` bar + a per-in-flight spinner
driven by our own denoise-step callback (`diffusing s/S → pixelizing → saved`).
Ctrl-C finishes/cancels in-flight then summarizes (exit 130); per-image error =
log + continue (exit 1 if any failed), `--fail-fast` aborts.

## 6. Architecture

```
pixl/
├── Cargo.toml                      # [workspace]
├── rust-toolchain.toml  pixl.example.toml  README.md  DESIGN.md
├── bake/                           # fallback-only Python LoRA bake (offline)
├── crates/
│   ├── pixl-pixelize/  src/{lib,grid,signal,collapse,palette,color}.rs
│   │                   tests/golden.rs
│   ├── pixl-gen/       src/lib.rs   # Generator trait + types (candle impl = M2)
│   └── pixl/           src/{main,cli}.rs  examples/demo_fixture.rs
└── .github/workflows/ci.yml        # fmt + clippy -D warnings + test (no GPU)
```

**Seams.** `pixl-gen::Generator` decouples candle (one impl now; a remote/
subprocess backend could implement the same trait). `pixl-pixelize::pixelize` is
a pure function (`&RgbaImage → (RgbImage, PixelizeReport)`), usable standalone
and as `pixl pixelize`. Errors: `thiserror` enums in libs (`GenError`,
`PixelizeError`); `anyhow` only in the binary. Config layering: defaults →
`pixl.toml` → CLI flags.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Metal memory leak in the 100-image loop** (candle #2271/#3464, open) | Wrap each `generate` in `objc2::rc::autoreleasepool`; log RSS across the loop in the first M2 integration test; recreate `Device` every N images if it climbs. |
| candle slower than hoped | "Seconds/image" holds for the few-step path (Turbo 4 / Lightning 8). Measure on the real M4 Pro at M2 before promising UX timings. |
| LoRA key mapper misses a convention | First M3 test dumps `pixel-art-xl` keys + asserts full mapping. Fallback = Python `fuse_lora` bake (still no Python at runtime). |
| Lightning ancestral-noise mismatch | Validate visually at M3; fall back to `DDIMScheduler`. |
| f16 VAE black/NaN | Mandatory fp16-fix VAE; if artifacts persist, F32 VAE decode only. |
| Hard image fools auto-detect | `--pixel-size` always wins; never-crash fixed fallback with a visible warning. |
| candle disappoints entirely | `Generator` trait is the swap point — a remote/subprocess backend implements the same signature; pixelize is wholly independent. |

## 8. Milestones

- **M0 — Scaffold.** *(done)* Workspace + 3 crates compile; clap parses `pixl
  100 "x" ./` and `pixl pixelize`; `Generator` seam exists. `pixl pixelize`
  works end-to-end.
- **M1 — Pixelize lib + golden tests.** *(done)* Full algorithm + degradation
  ladder; golden tests green on CI, no GPU.
- **M2 — candle SDXL, one image.** Load base SDXL on Metal, fp16-fix VAE, u64
  seed, Turbo@4 or DDIM@30; save a non-black PNG. Log RSS across 100 gens with
  `autoreleasepool`; measure per-step timing on the real M4 Pro.
- **M3 — LoRA.** Port the `SimpleBackend` mapper; self-built UNet config; key-dump
  test; pixel-art LoRA, then stack Lightning 8-step; validate quality.
- **M4 — Pipeline + progress.** Bounded-channel generate→pixelize→save, rayon
  consumers, indicatif per-step spinner, Ctrl-C, error policy, `--json`/`--quiet`.
  `pixl 100 "stardew valley style house" ./` runs end-to-end.
- **M5 — Polish.** First-run weight download UX + bytes bar; `pixl models`;
  `pixl.toml`; Turbo fast-path; fixed-palette mode; release single binary.
