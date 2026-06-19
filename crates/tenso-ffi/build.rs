//! Regenerate `include/tenso.h` from the C ABI in `src/lib.rs` via cbindgen.
//!
//! Best-effort: if generation fails (e.g. a transient parse issue while the ffi
//! agent is mid-edit), we warn but DO NOT fail the build, so the workspace stays
//! green. The committed `include/tenso.h` remains the source of truth in that
//! case.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // Workspace root is two levels up from crates/tenso-ffi.
    let header_path = crate_dir
        .join("..")
        .join("..")
        .join("include")
        .join("tenso.h");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let config = match cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")) {
        Ok(c) => c,
        Err(e) => {
            println!("cargo:warning=tenso-ffi: cbindgen config load failed: {e}");
            return;
        }
    };

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header_path);
        }
        Err(e) => {
            println!("cargo:warning=tenso-ffi: cbindgen generation skipped: {e}");
        }
    }
}
