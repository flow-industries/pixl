# AGENTS.md

This file provides guidance to agents working with code in this repository. `CLAUDE.md` is a symlink to it.

## What this is

`pixl` — a local pixel-art generator in Rust: SDXL + a pixel-art LoRA, then the result is snapped to
*true* pixel art (uniform grid, limited palette). Published on crates.io as **`flow-pixl`** (the name
`pixl` was taken); the binary is still `pixl`. Three workspace crates, in dependency order:
`pixl-pixelize` (the GPU-free core) → `pixl-gen` (the generation backend) → `pixl` (the CLI + gallery).

The generation backend selects itself: Metal on macOS, CUDA with `--features cuda`, CPU otherwise.
`README.md` covers usage; `DESIGN.md` covers why the pixelize pipeline works the way it does.

## Commands

```bash
just build          # release build, generation included (Metal on macOS, CPU elsewhere)
just build-lite     # pixelize-only, no GPU/ML, builds anywhere
just build-cuda     # NVIDIA
just test           # GPU-free golden tests over the pixelize core
just lint           # cargo fmt --check + clippy -D warnings, exactly as CI
just lint-gen       # clippy over the generation backend
```

Generation is expensive: a batch pins the GPU and can take minutes. Say what a run will cost before
starting one, and prefer `pixl pixelize <file> --scale 8` (no GPU) when the change is in the pixelize
core — the golden tests cover it without a model download.

## Publishing

**Any change ships as a republish of all three crates**, in dependency order
(`pixl-pixelize` → `pixl-gen` → `flow-pixl`), with the workspace version bumped first — the path
dependencies carry an exact `version =`, so publishing one crate against an unpublished sibling fails.
A change that is only in one crate still needs the others' versions moved, or the published graph
no longer matches this tree.

## Conventions

- No WHAT comments. Doc comments (`///`) for public APIs and sparse WHY comments for non-obvious
  constraints only; write self-documenting code with descriptive names instead.
- No emojis or `[SECTION]` prefixes in log messages.
- `cargo` is the package manager; the pinned toolchain lives in `rust-toolchain.toml`.
- Timings on a loaded machine swing wildly — interleave both sides and take minima before claiming a
  performance delta.
