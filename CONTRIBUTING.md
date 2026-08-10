# Contributing

```bash
uv pip install -e ".[dev]"
maturin develop --release
pytest tests/
```

Python style: `ruff check .` and `ruff format .`.
Rust style: `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings`.

## Where the wire format lives

`crates/tenso/src/lib.rs` is the single source of truth for encoding and
decoding. `src/lib.rs` is a thin PyO3 binding over it, and `crates/tenso-ffi`
exposes the same core through the C ABI. Do not add a second implementation of a
format rule in Python or in C.

Any change to a dtype code or flag bit must be updated in **both**
`crates/tenso/src/lib.rs` and `src/tenso/config.py`, and needs a conformance
fixture (`tests/test_conformance.py`) so the Rust core, the Python binding, and
the C ABI are asserted to agree byte-for-byte.

## Things that must not regress

- **`loads()` is zero-copy.** The returned array must be a view onto the caller's
  buffer. `tests/test_fixes.py::test_loads_is_zero_copy_by_default` asserts this
  with `np.shares_memory`. Alignment guarantees are opt-in via
  `loads(..., align=True)` precisely because guaranteeing an absolute address
  costs a full copy of the body; do not make that the default path again.
- **Decoding untrusted bytes never panics.** Size a buffer from the bytes you
  actually have, not from a field the packet declares. Use checked arithmetic on
  anything read off a header.
- **No silently wrong data.** If input cannot be represented faithfully, raise.
  Do not convert on the caller's behalf inside a documented zero-copy path.

For security issues, see [SECURITY.md](SECURITY.md).
