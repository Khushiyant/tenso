# tenso fuzz targets (cargo-fuzz)

Coverage-guided fuzz harnesses for the Tenso Rust packet parser. Fuzzing
is opt-in; this crate is **not** part of the top-level Cargo workspace
and is not built by `maturin develop`.

## One-time setup

```sh
rustup toolchain install nightly
cargo install cargo-fuzz
```

## Run a target

```sh
# from the repo root
cargo +nightly fuzz run parse_header     # header parser
cargo +nightly fuzz run loads_dense      # header + shape table parser
```

Add libFuzzer flags after `--`, e.g. `cargo +nightly fuzz run parse_header -- -max_total_time=60`.

## Corpus seeds

```sh
mkdir -p fuzz/corpus/parse_header
cp ../pyfuzz/seeds/*.bin fuzz/corpus/parse_header/   # seed from python side
```

The Python seed generator (`../pyfuzz/_make_seeds.py`) emits valid v4
packets that double as Rust header-parser seeds.

## Crash artifacts

Failing inputs land in `fuzz/artifacts/<target>/`. Reproduce with:

```sh
cargo +nightly fuzz run parse_header fuzz/artifacts/parse_header/crash-XXXX
```
