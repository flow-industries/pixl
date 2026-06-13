# LoRA bake (fallback only — not used by the default path)

`pixl` merges LoRAs into the base UNet **at runtime in pure Rust** (see
`DESIGN.md` §3 and milestone M3). No Python is required to run `pixl`.

This directory exists only as the documented escape hatch: if a LoRA ships with
a key convention the runtime mapper can't bridge, bake it once with diffusers
and point `pixl` at the merged checkpoint.

```python
# one-time, offline; produces a merged unet the Rust loader treats as plain weights
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe.load_lora_weights("nerijs/pixel-art-xl")
pipe.fuse_lora(lora_scale=1.2)
pipe.unet.save_pretrained("baked/pixel-art-xl-fused")
```

If you reach for this, note it in an issue — the goal is for the runtime merger
to cover every LoRA so this stays unused.
