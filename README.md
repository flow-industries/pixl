# pixl

**Local pixel-art generator for Apple Silicon.** Generate with SDXL + a pixel-art
LoRA on the GPU (candle/Metal), then snap each image to *true* pixel art — a
clean, uniform grid and a limited palette — entirely on-device, no Python at
runtime.

```bash
pixl 100 "stardew valley style house" ./out
```

Generate 100 variations and post-process every one into a game-ready sprite,
with per-image progress.

## Status

All milestones (M0–M5) are implemented and green. Works today:

- **Generation** — SDXL-Turbo (or `--model sdxl`) + a runtime-merged pixel-art
  LoRA on Metal, ~7.5 s/image (Turbo @ 4 steps, 512²) on an M4 Pro.
- **True-pixel-art post-processing** — grid detection, cell collapse, Lab k-means
  palette. Pure CPU, deterministic, golden-tested.
- **Overlapped pipeline** — generate → pixelize → save with per-image progress,
  `--json`, `--quiet`, `--fail-fast`, Ctrl-C handling.
- **`pixl pixelize`** — post-process existing images (no GPU).
- **`pixl models`** — inspect / clear the local cache.

## Install

```bash
# pixelize-only (no GPU) — also what CI builds
cargo build --release
cargo install --path crates/pixl

# full build with the candle/Metal generation backend (macOS / Apple Silicon)
cargo build --release --features metal
cargo install --path crates/pixl --features metal
```

Needs a stable Rust toolchain (see `rust-toolchain.toml`). The first generate
downloads ~7 GB of SDXL weights (one time, cached under `~/.cache/huggingface`).

## Usage

```bash
# generate + snap to true pixel art
pixl 8 "stardew valley style house" ./out

# knobs
pixl 8 "..." ./out --colors 24 --pixel-size 8 --steps 4 --seed 1000 --model turbo
pixl 8 "..." ./out --no-lora          # skip the pixel-art LoRA (and its merge)
pixl 8 "..." ./out --no-postprocess   # save raw generations
pixl 8 "..." ./out --json             # one JSONL event per image on stdout

# post-process existing images (no GPU)
pixl pixelize sprite.png -o sprite.pixl.png --colors 16 --scale 8
pixl pixelize art/*.png -o out/ --colors 32

# cache management
pixl models ls          # list cached merged UNets + where weights live
pixl models clear       # delete pixl's merged-UNet cache (asks first)
pixl models path
```

## How it works

Two decoupled stages — the dependency cut is the architecture:

| Crate | Role | Heavy deps |
|---|---|---|
| `pixl-pixelize` | True-pixel-art post-processing. Pure CPU, deterministic, no GPU. | `image`, `rayon` |
| `pixl-gen` | SDXL + LoRA generation behind a `Generator` trait (candle/Metal). | candle (`metal` feature) |
| `pixl` | CLI + the overlapped generate→pixelize→save pipeline. | `clap`, `indicatif` |

Generation only needs to produce blocky structure; the pixelize stage is what
makes the output *true* pixel art (detects the real cell size by folding the
per-axis color-change signal modulo each candidate period — the fundamental wins
because harmonics scatter the edge energy). LoRAs ship in sgm/A1111 naming and are
translated to candle's diffusers naming, merged on CPU, and cached as a merged
UNet that loads via the fast mmap path. See `DESIGN.md` for the full design.

## Resource usage

Generation runs SDXL on the GPU — it is **GPU-heavy** and will make the machine
warm/loud during a batch. The biggest lever is doing less: fewer `--steps`,
smaller `--size`, fewer images. To keep the UI responsive, run it throttled:

```bash
# macOS: background QoS + nice + cap the pixelize threads
RAYON_NUM_THREADS=2 /usr/sbin/taskpolicy -b nice -n 19 \
  pixl 100 "stardew valley style house" ./out --jobs 1
```

- `--jobs` controls pixelize/save worker threads (default 2; `--jobs 1` is plenty).
- Generation is serial on the single GPU queue regardless of `--jobs`.
- The merged-LoRA cache (`pixl models ls`) is ~4.8 GB per LoRA combo; `--no-lora`
  skips it, `pixl models clear` reclaims it.

## License

MIT OR Apache-2.0.
