# Contributing

```bash
uv pip install -e ".[dev]"
maturin develop --release
pytest tests/
```

Python style: `ruff check .` and `ruff format .`.
Rust style: `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings`.

Any change to a dtype or flag must be updated in **both** `src/tenso/config.py`
**and** `src/lib.rs`; the two implementations must agree on the wire format.

For security issues, see [SECURITY.md](SECURITY.md).
