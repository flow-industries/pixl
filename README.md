# pixl

[![crates.io](https://img.shields.io/crates/v/flow-pixl.svg)](https://crates.io/crates/flow-pixl)

A local pixel-art generator. It generates with SDXL + a pixel-art LoRA, then snaps the result to true pixel art — a clean grid and a limited palette. Runs on Apple Silicon (Metal), NVIDIA (CUDA), or CPU.

## Installation

```bash
cargo install flow-pixl
```

Generation is included by default (Metal on macOS, CPU elsewhere). Use `--features cuda` for NVIDIA, or `--no-default-features` for a pixelize-only build (no GPU/ML, builds fast anywhere).

Build from source:

```bash
git clone https://github.com/flow-industries/pixl.git
cd pixl
cargo install --path crates/pixl
```

## Usage

```bash
pixl "a cozy tavern"                         # 4 images -> ~/.pixl/<timestamp>-<prompt>/
pixl 100 "stardew valley style house" ./out  # count + output dir
pixl 8 "a cozy tavern" --colors 24 --low-prio
pixl pixelize sprite.png --scale 8           # snap an existing image (no GPU)
pixl models ls                               # inspect the model cache
```

Run `pixl --help` for all options.

## License

MIT OR Apache-2.0
