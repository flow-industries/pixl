# pixl

[![crates.io](https://img.shields.io/crates/v/flow-pixl.svg)](https://crates.io/crates/flow-pixl)

A local pixel-art generator for Apple Silicon — generate with SDXL + a pixel-art LoRA on the GPU, then snap the result to *true* pixel art (a clean, uniform grid and a limited palette). Fully on-device, no Python at runtime.

```bash
pixl 100 "stardew valley style house" ./out
```

## Installation

With the generation backend (auto-selected by platform — Metal on macOS, CPU elsewhere):

```bash
cargo install flow-pixl --features gen
```

NVIDIA (Linux/Windows; needs the CUDA toolkit at build time):

```bash
cargo install flow-pixl --features cuda
```

Pixelize-only (no GPU/ML — post-process existing images; builds fast on any platform):

```bash
cargo install flow-pixl
```

The installed binary is `pixl`. To build from source instead:

```bash
git clone https://github.com/flow-industries/pixl.git
cd pixl
cargo install --path crates/pixl --features gen
```

The first generate downloads ~7 GB of SDXL weights (one time, cached under `~/.cache/huggingface`).

## Usage

```bash
# generate N images and snap each to true pixel art
pixl 100 "stardew valley style house"           # -> ~/.pixl/<timestamp>-<prompt>/
pixl 100 "stardew valley style house" ./out     # -> ./out (override)

# knobs
pixl 8 "..." ./out --colors 24 --pixel-size 8 --steps 4 --seed 1000 --model turbo
pixl 8 "..." ./out --no-lora            # skip the pixel-art LoRA (and its merge)
pixl 8 "..." ./out --no-postprocess     # save raw generations
pixl 8 "..." ./out --json               # one JSONL event per image on stdout

# post-process existing images into true pixel art (no GPU)
pixl pixelize sprite.png -o sprite.pixl.png --colors 16 --scale 8
pixl pixelize art/*.png -o out/ --colors 32

# cache management
pixl models ls          # cached merged UNets + where weights live
pixl models clear       # reclaim the merged-LoRA cache (asks first)
```

## How it works

Two decoupled stages. **Generation** (candle/Metal SDXL + a runtime-merged pixel-art LoRA) only needs to produce blocky structure; the **pixelize** stage is what makes the output *true* pixel art — it detects the real cell size by folding the per-axis color-change signal modulo each candidate period (the fundamental wins because harmonics scatter the edge energy), collapses each cell to one color, and quantizes to a limited Lab palette. The pixelize core is pure CPU, deterministic, and dependency-light; the generation backend lives behind the `metal` feature so the post-processor builds and tests anywhere.

| Crate | Role |
|---|---|
| `pixl-pixelize` | True-pixel-art post-processing (pure CPU, golden-tested) |
| `pixl-gen` | SDXL + LoRA generation behind a `Generator` trait (candle/Metal) |
| `pixl` | CLI + the overlapped generate → pixelize → save pipeline |

See [`DESIGN.md`](DESIGN.md) for the full design and roadmap.

## Resource usage

Generation runs SDXL on the GPU — it is GPU-heavy. The biggest levers are fewer `--steps`, smaller `--size`, and fewer images. Pass `--low-prio` (alias `--bg`) to keep the machine responsive — it drops the process to macOS background QoS (the same as `taskpolicy -b`: efficiency cores + I/O throttle) and single-threads the pixelize pass:

```bash
pixl 100 "stardew valley style house" --low-prio
```

## License

MIT OR Apache-2.0.
