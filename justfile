# pixl developer tasks (https://just.systems)

default:
    @just --list

# release build (generation included by default: Metal on macOS, CPU elsewhere)
build:
    cargo build --release

# pixelize-only release build (no GPU/ML, builds anywhere)
build-lite:
    cargo build --release --no-default-features

# generation backend for NVIDIA GPUs (needs the CUDA toolkit)
build-cuda:
    cargo build --release --features cuda

# GPU-free tests (the pixelize core)
test:
    cargo test --workspace --no-default-features

# lint the pixelize core exactly as CI does
lint:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets --no-default-features -- -D warnings

# lint the generation backend
lint-gen:
    cargo clippy -p flow-pixl --features gen --all-targets -- -D warnings

# install the full binary (generation included)
install:
    cargo install --path crates/pixl

# fabricate a demo "AI pixel-art" image to try `pixl pixelize`
demo out="/tmp/demo.png":
    cargo run --example demo_fixture -- {{out}}
