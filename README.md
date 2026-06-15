# pixl

[![crates.io](https://img.shields.io/crates/v/flow-pixl.svg)](https://crates.io/crates/flow-pixl)

A local pixel-art generator. It generates with SDXL + a pixel-art LoRA, then snaps the result to true pixel art — a clean grid and a limited palette. Runs on Apple Silicon (Metal), NVIDIA (CUDA), or CPU.

## Installation

```bash
cargo install flow-pixl --features gen
```

`--features gen` builds the generation backend (Metal on macOS, CPU elsewhere); use `--features cuda` for NVIDIA. Without it, `pixl pixelize` still works with no GPU.

Build from source:

```bash
git clone https://github.com/flow-industries/pixl.git
cd pixl
cargo install --path crates/pixl --features gen
```

## Usage

```bash
pixl 100 "stardew valley style house"        # -> ~/.pixl/<timestamp>-<prompt>/
pixl 8 "a cozy tavern" ./out --colors 24     # output dir + palette size
pixl 8 "a cozy tavern" --low-prio            # run at low priority
pixl pixelize sprite.png --scale 8           # snap an existing image (no GPU)
pixl models ls                               # inspect the model cache
```

Run `pixl --help` for all options.

## License

MIT OR Apache-2.0
