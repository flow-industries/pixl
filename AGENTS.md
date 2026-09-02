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

## The board: Plane pages (`page_*`)

Work here is tracked in Plane project **PIXL** (`plane.flow.industries`, workspace `flow`), driven
through the `flow` MCP. Tickets (`plane_*`) say *what to do*; **pages (`page_*`) are this repo's message
board** — design records, findings, runbooks, and the running thread of a piece of work. Design docs
live on pages, not in the repo (they were migrated out of `docs/internal/` on 2026-09-02).

Leave messages before, during and after the work:

1. **Before you touch code** — `page_list(project="PIXL")` (or `page_recent()` when you don't know
   which project holds the thread) and `page_get` anything whose title touches your area. A previous
   session may have mapped the subsystem, hit the dead end you are about to walk into, or left the exact
   repro. Skipping this is how the same investigation gets paid for twice.
2. **Before implementing** — `page_append` the plan: what you are about to do, the approach you picked
   and the ones you rejected. That message is how a parallel session finds out you are in this code.
3. **While working** — append what a future session would need: the non-obvious constraint, the thing
   that looked right and wasn't, the command that finally reproduced it.
4. **After** — append the outcome: what shipped, the PR, what you deliberately left undone. A one-line
   "done" is noise; a paragraph that saves an hour is the whole return.

Rules that bite:

- `page_append` is the **only** safe way to add to a thread — it read-modify-writes server-side and
  returns a `marker` that is both your receipt and your cursor for `page_get(after=…)`.
  `page_update(content=)` **replaces the whole body**; that is how one session deletes another's messages.
- Pass `author` (`claude-code/<workspace>`) or the whole board reads as one voice.
- Content is **HTML, not markdown** — emit raw `<p>`, `<ul><li>`, `<code>`, `<strong>`. Markdown renders
  literally.
- Title by convention, because the title is the address: `board: <topic>` for a running thread,
  `design: <topic>` for an architecture record, `research: <topic>` for a finding, `runbook: <topic>` for
  a procedure. `page_list` the project before creating — advancing an existing thread beats a near-duplicate.
- A long page comes back across calls: keep calling `page_get(from=<next_from>)` until `next_from` is absent.

Ticket rules (when to file, dedup, states, cross-ticket edges) live in the mono `AGENTS.md`.
