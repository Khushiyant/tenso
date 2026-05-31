#![no_main]

//! Fuzz target for the Tenso dense-packet pre-body decode path.
//!
//! This covers the header parse PLUS the per-dim shape decode. The body
//! decode (LZ4 decompression, integrity hashing, numpy hand-off) lives
//! inside `deserialize_impl`, which is `pyo3`-bound and cannot be invoked
//! without an initialized Python interpreter — see the TODO in the
//! `fuzz_api` module.
//!
//! Contract under test: the pure-Rust validation path MUST NOT panic on
//! arbitrary input. Out-of-range ndim values, truncated shape tables,
//! garbage flags, and unsupported versions should all return a typed
//! error rather than aborting.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(parsed) = tenso_rs::fuzz_api::fuzz_parse_header_and_shape(data) {
        // Cheap sanity invariants: if the parser said it succeeded, the
        // shape vector length must match ndim and base_size must be the
        // documented v3 or v4 constant. A regression that violated either
        // would indicate a parser bug worth surfacing.
        assert_eq!(parsed.shape.len(), parsed.ndim);
        assert!(parsed.base_size == 8 || parsed.base_size == 10);
        // Touch every shape entry to make sure we didn't read uninitialized
        // memory (in debug builds with sanitizers).
        let _: usize = parsed.shape.iter().copied().fold(0usize, usize::saturating_add);
    }
});
