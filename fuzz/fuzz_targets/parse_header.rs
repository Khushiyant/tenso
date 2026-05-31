#![no_main]

//! Fuzz target for the Tenso packet header parser.
//!
//! Contract under test: `parse_packet_header_raw` MUST NOT panic on any
//! input. Every malformed packet should be reported as a typed
//! `HeaderError`, never as a Rust panic / overflow / OOB index.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Discard the result; we only care that this call returns rather than
    // panicking. The `#[cfg(fuzzing)]`-guarded shim lives in
    // `tenso-rs/src/lib.rs` and re-exports the otherwise-private parser.
    let _ = tenso_rs::fuzz_api::fuzz_parse_header(data);
});
