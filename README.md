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
- **`pixl-gen`** — candle SDXL on Metal **works** (behind the `metal` feature):
  `pixl "a prompt"` generates and snaps to true pixel art. ~7.5 s/image
  (SDXL-Turbo @ 4 steps, 512²) on an M4 Pro.

Next: M3 (runtime pixel-art + Lightning LoRA merge), then the overlapped
pipeline + progress UX (M4).

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
# pixelize-only build (no GPU); CI builds this
cargo build --release
cargo test                                   # GPU-free golden tests

# full build with the candle/Metal generation backend (macOS / Apple Silicon)
cargo build --release --features metal
cargo run --features metal -- 4 "stardew valley style house" ./out
```

Requires a stable Rust toolchain (see `rust-toolchain.toml`). The first generate
downloads ~7 GB of SDXL weights (one time, cached).

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
