# pixl developer tasks (https://just.systems)

# default: list recipes
default:
    @just --list

# pixelize-only build (no GPU)
build:
    cargo build --release

# full build with the candle/Metal generation backend (macOS)
build-metal:
    cargo build --release --features metal

# GPU-free tests (the pixelize golden tests)
test:
    cargo test --workspace

# lint exactly as CI does
lint:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings

# lint the metal backend too
lint-metal:
    cargo clippy -p pixl -p pixl-gen --features metal --all-targets -- -D warnings

# install the full binary
install:
    cargo install --path crates/pixl --features metal

# fabricate a demo "AI pixel-art" image to try `pixl pixelize`
demo out="/tmp/demo.png":
    cargo run --example demo_fixture -- {{out}}
