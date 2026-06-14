# pixl developer tasks (https://just.systems)

# default: list recipes
default:
    @just --list

# pixelize-only build (no GPU)
build:
    cargo build --release

# full build with the generation backend (Metal on macOS, CPU elsewhere)
build-gen:
    cargo build --release --features gen

# generation backend for NVIDIA GPUs (needs the CUDA toolkit)
build-cuda:
    cargo build --release --features cuda

# GPU-free tests (the pixelize golden tests)
test:
    cargo test --workspace

# lint exactly as CI does
lint:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings

# lint the generation backend too
lint-gen:
    cargo clippy -p pixl -p pixl-gen --features gen --all-targets -- -D warnings

# install the full binary (Metal on macOS, CPU elsewhere)
install:
    cargo install --path crates/pixl --features gen

# fabricate a demo "AI pixel-art" image to try `pixl pixelize`
demo out="/tmp/demo.png":
    cargo run --example demo_fixture -- {{out}}
