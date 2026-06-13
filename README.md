# pixl

**Local pixel-art generator for Apple Silicon.** Generate images with SDXL + a
pixel-art LoRA on the GPU (candle/Metal), then snap each one to *true* pixel art
— a clean, uniform grid and a limited palette — entirely on-device, no Python at
runtime.

```
pixl 100 "stardew valley style house" ./out
```

Generate 100 variations and post-process every one into a game-ready sprite,
with per-image status.

## Status

Early. Built and tested today:

- **`pixl-pixelize`** — the true-pixel-art post-processor (grid detection,
  cell collapse, Lab k-means palette). Pure CPU, deterministic, golden-tested.
- **`pixl pixelize <img>`** — fully working CLI subcommand (no GPU needed).
- **`pixl-gen`** — the `Generator` trait/seam; the candle/Metal SDXL backend
  lands next (see `DESIGN.md`, milestones M2–M5).

`pixl <count> <prompt> <out_dir>` parses and is wired end-to-end except for the
generation backend itself.

## Usage today

```bash
# snap an existing AI "pixel-art-style" image to a true grid + 16-color palette
pixl pixelize sprite.png -o sprite.pixl.png --colors 16

# force the cell size instead of detecting it, and upscale x8 for viewing
pixl pixelize sprite.png --pixel-size 24 --scale 8

# batch a folder
pixl pixelize art/*.png -o out/ --colors 32
```

## Build

```bash
cargo build --release      # single self-contained binary at target/release/pixl
cargo test                 # GPU-free golden tests
cargo run --example demo_fixture -- /tmp/demo.png   # fabricate a test image
```

Requires a stable Rust toolchain (see `rust-toolchain.toml`).

## Architecture

A three-crate workspace; the dependency cut is the point:

| Crate | Role | Heavy deps |
|---|---|---|
| `pixl-pixelize` | True-pixel-art post-processing. Pure CPU, deterministic, no GPU. | `image`, `rayon` |
| `pixl-gen` | Generation backend behind a `Generator` trait (candle/Metal SDXL). | candle (M2) |
| `pixl` | CLI + the overlapped generate→pixelize→save pipeline. | `clap` |

Keeping `pixl-pixelize` free of candle is what lets the core be unit-tested on
CI with no GPU. See `DESIGN.md` for the full design and roadmap.

## License

MIT OR Apache-2.0.
