# pixl

[![crates.io](https://img.shields.io/crates/v/flow-pixl.svg)](https://crates.io/crates/flow-pixl)

A local pixel-art generator. It renders with SDXL + a pixel-art LoRA, then snaps the result to
*true* pixel art — a clean, uniform grid and a limited palette. Runs on Apple Silicon (Metal),
NVIDIA (CUDA), or CPU, with an interactive terminal gallery for browsing and curating results.

## Install

```bash
cargo install flow-pixl
```

Generation is included by default (Metal on macOS, CPU elsewhere). Build variants:

- `--features cuda` — NVIDIA GPUs (needs the CUDA toolkit).
- `--no-default-features` — pixelize-only, no GPU/ML, builds fast anywhere.
- `--no-default-features --features view` — pixelize + the interactive viewer, still no GPU.

From source:

```bash
git clone https://github.com/flow-industries/pixl.git
cd pixl
cargo install --path crates/pixl
```

## Quick start

```bash
pixl                                  # open the gallery and configure everything in-app
pixl "a cozy tavern"                  # 4 images -> ~/.pixl/<timestamp>-<prompt>/
pixl 12 "stardew valley house"        # a count
pixl --model sdxl --cfg 5 "a sword"   # full SDXL with classifier-free guidance
pixl pixelize photo.png --scale 8     # snap an existing image to pixel art (no GPU)
```

In a graphics-capable terminal this opens the interactive gallery; otherwise it prints a
headless batch and a clickable link to the output folder.

## The gallery

In Ghostty, Kitty, or iTerm2, pixl opens an interactive gallery that shows the whole batch up
front and updates live as images render:

- **Queued** slots show a spinner placeholder.
- The **generating** slot shows a live denoise preview.
- **Finished** slots show the final pixel art.

Navigate with the arrow keys; the cursor follows the image currently rendering and holds your
position when you step back to inspect earlier ones.

| key | action |
|-----|--------|
| `←` / `→` | previous / next image |
| `space` | save the current image to `~/.pixl/saved/` |
| `s` | open the settings panel |
| `x` | discard the current slot (cancels it if queued or in-flight) |
| `r` | rerun — generate more with the current prompt |
| `e` | edit the prompt |
| `c` | cancel the in-flight generation |
| `q` | quit |

### Settings panel (`s`)

The panel tweaks generation parameters and prompt modifiers without touching the command line:

- **Parameters** — count, cfg, steps, colors, seed. `↑`/`↓` selects a row, `←`/`→` adjusts.
- **Modifiers** — one-key toggles that fold isolation fragments onto your prompt: *single
  subject*, *plain background*, *item icon*, *no shadow*. `space` toggles.

`Enter` regenerates, `Esc` closes. Settings persist to `~/.pixl/config.json`, so the next run —
including a bare `pixl` with no arguments — starts where you left off.

### No-args mode

Run `pixl` with no arguments to open the gallery idle: `e` sets a prompt, `s` configures
parameters and modifiers, then generate. Everything is configurable in-app.

### Browse a finished run

```bash
pixl view ~/.pixl/<run>
```

## Generating game sprites

By default pixl uses SDXL-Turbo (fast, few-step, CFG-free) with the pixel-art LoRA. Turbo is
quick but ignores negative prompts and leans toward busy *scenes*. For isolated single sprites,
switch to full SDXL so classifier-free guidance and negative prompts take effect:

```bash
pixl --model sdxl --cfg 5 "a wooden treasure chest" \
  --negative "scene, multiple objects, background, shadow"
```

- **Model** — `--model sdxl` (quality, negatives) vs the default `turbo` (speed).
- **Guidance (`--cfg`)** — higher sticks closely to the prompt but reduces variety between
  seeds; ~4-5 gives more varied takes, ~7-8 adheres tightly. Negative prompts only take effect
  at cfg > 1 (so on SDXL, not Turbo).
- **Isolation** — the settings-panel modifiers compose the right positive and negative
  fragments for single-subject, plain-background sprites.
- **Seed** — random by default (each run differs); pin a value in the settings panel for
  reproducible results.

## Commands & flags

`pixl [COUNT] [PROMPT] [OUT_DIR]` is the default (generate) form; `pixl gen` is the explicit
alias.

| flag | default | description |
|------|---------|-------------|
| `--model turbo\|sdxl` | turbo | base model |
| `--cfg <f>` | 1 (turbo) / 7 (sdxl) | classifier-free guidance |
| `--steps <n>` | 8 (turbo) / 25 (sdxl) | diffusion steps |
| `--seed <n>` | random | base seed (omit for random) |
| `-c, --colors <n>` | 16 | palette size (0 = keep all distinct cell colors) |
| `--size <WxH>` | 512x512 | generation resolution (multiple of 8) |
| `--negative <text>` | — | negative prompt (SDXL / cfg > 1) |
| `--no-lora` | off | disable the pixel-art LoRA |
| `--no-postprocess` | off | skip the pixelize pass, save the raw render |
| `--no-view` | off | force the headless batch output |
| `--saved-dir <path>` | `~/.pixl/saved` | where saved favorites are copied |
| `-j, --jobs <n>` | auto | pixelize/save worker threads |
| `--json` | off | emit one JSON line per finished image |
| `--low-prio` | off | run at low priority (macOS background QoS) |

Other subcommands:

- `pixl pixelize <img>... [-o out] [--scale n] [-c colors]` — snap existing images to true
  pixel art. No GPU or model needed (works in every build).
- `pixl view <dir>` — browse a directory of images in the gallery.
- `pixl models ls | path | clear` — inspect or clear the local model / merge cache.

## How it works

pixl renders with SDXL (via [candle](https://github.com/huggingface/candle)) plus the
`nerijs/pixel-art-xl` LoRA merged into the UNet, then runs a GPU-free pixelize pass: it detects
the pixel grid (per-axis edge-energy folding, with square cells), collapses each cell to its
dominant color, and quantizes to a limited Lab-space palette. Model weights download once from
Hugging Face and cache locally — `pixl models ls` shows where.

## License

MIT OR Apache-2.0
