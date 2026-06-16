# pixl

Local pixel-art generator: SDXL + a pixel-art LoRA on the GPU, snapped to *true*
pixel art (a clean, uniform grid and a limited palette). Runs entirely on-device —
**Metal** on Apple Silicon, **CUDA** on NVIDIA, or **CPU** anywhere.

```bash
cargo install flow-pixl                         # generation included (Metal on macOS, CPU elsewhere)
cargo install flow-pixl --features cuda         # NVIDIA (needs the CUDA toolkit)
cargo install flow-pixl --no-default-features   # pixelize-only, no GPU/ML, builds anywhere

pixl 100 "stardew valley style house"
```

In a graphics-capable terminal (Ghostty, Kitty, iTerm2) it opens an interactive gallery —
inline images as they generate, arrow-key navigation, save/rerun/edit — and `pixl view <dir>`
re-browses a finished run; elsewhere it prints a headless batch.

The installed binary is `pixl`. Full documentation, usage, and design notes:
**https://github.com/flow-industries/pixl**

## License

MIT OR Apache-2.0
