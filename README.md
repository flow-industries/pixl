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

## Gallery

In a graphics-capable terminal (Ghostty, Kitty, iTerm2) `pixl` opens an interactive gallery.
The whole batch shows up front — queued images as a spinner placeholder, the one generating
as a live denoise preview, and finished ones as the final pixel art — all navigable with the
arrow keys. Flip through them with the arrow keys, mark the
ones you like (copied to `~/.pixl/saved/`), and rerun or edit the prompt without reloading
the model:

```
left/right  navigate   s  save   t  modifiers   r  rerun   e  edit prompt   c  cancel   q  quit
```

Elsewhere — or when piping, with `--json`, or `--no-view` — it falls back to the headless
batch output. Re-browse a finished run any time:

```bash
pixl view ~/.pixl/<run>
```

The gallery ships in the default build. For a no-GPU build that still includes it (and
`pixl view`), use `--no-default-features --features view`.

## Sprite assets

By default pixl uses SDXL-Turbo (fast, but CFG-free, so it ignores negative prompts) plus the
pixel-art LoRA, which leans toward busy *scenes*. For an isolated single sprite, switch to full
SDXL so CFG and negative prompts take effect:

```bash
pixl --model sdxl "a wooden treasure chest" \
  --negative "scene, multiple objects, background, shadow"
```

`--model sdxl` defaults to cfg 7 / 25 steps (negatives only bite at cfg > 1). Higher cfg means
tighter prompt adherence but less variety between seeds — for a constrained subject the results
can look near-identical; drop `--cfg` to ~4-5 for more varied takes. In the gallery,
press `t` for a checklist of one-key modifiers — single subject, plain background, item icon,
no shadow, keyable background — that fold isolation fragments onto your prompt (and the matching
negatives). Toggle, press Enter, and it regenerates with the already-loaded model.

## License

MIT OR Apache-2.0
