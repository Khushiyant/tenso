//! Rust-side smoke test that drives the C ABI exactly as a C caller would:
//! raw pointers, status codes, Mode A (caller-allocates) encode, Mode B
//! (core-allocates opaque view) decode, and the thread-local error channel.
//!
//! This exercises the FFI surface end-to-end via `tenso-ffi`'s rlib face. The
//! C equivalent (`examples/smoke.c`) compiles against `include/tenso.h` and the
//! staticlib/cdylib; this Rust test lets `cargo test -p tenso-ffi` validate the
//! same behaviour without a C toolchain.

use std::ffi::CStr;

use tenso_ffi::*;

/// Read the thread-local last error as a String (empty if none).
fn last_error() -> String {
    let p = tenso_last_error();
    assert!(!p.is_null(), "tenso_last_error must never return NULL");
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

#[test]
fn dense_roundtrip_through_abi() {
    // f32 vector [1.0, 2.0, 3.0, 4.0], shape [4].
    let elems: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let data: Vec<u8> = elems.iter().flat_map(|f| f.to_le_bytes()).collect();
    let shape: [u32; 1] = [4];
    let dtype_code: u8 = 1; // f32

    // --- Mode A: required size, then encode into caller buffer ---
    let mut needed: usize = 0;
    let rc = unsafe {
        tenso_dense_required_size(
            data.as_ptr(),
            data.len(),
            dtype_code,
            shape.as_ptr(),
            shape.len(),
            false, // check_integrity
            false, // compress
            0,     // alignment (0 -> default 64)
            &mut needed,
        )
    };
    assert_eq!(rc, TENSO_OK, "required_size failed: {}", last_error());
    assert!(needed > 0);

    let mut packet = vec![0u8; needed];
    let mut written: usize = 0;
    let rc = unsafe {
        tenso_encode_dense_into(
            data.as_ptr(),
            data.len(),
            dtype_code,
            shape.as_ptr(),
            shape.len(),
            false,
            false,
            0,
            packet.as_mut_ptr(),
            packet.len(),
            &mut written,
        )
    };
    assert_eq!(rc, TENSO_OK, "encode failed: {}", last_error());
    assert!(written <= needed && written > 0);
    packet.truncate(written);

    // --- parse_header on the freshly encoded packet ---
    let mut hdr = std::mem::MaybeUninit::<TensoHeader>::uninit();
    let rc = unsafe { tenso_parse_header(packet.as_ptr(), packet.len(), hdr.as_mut_ptr()) };
    assert_eq!(rc, TENSO_OK, "parse_header failed: {}", last_error());
    let hdr = unsafe { hdr.assume_init() };
    assert_eq!(hdr.version, 4);
    assert_eq!(hdr.dtype_code, dtype_code);
    assert_eq!(hdr.ndim, 1);

    // --- Mode B: decode into opaque view ---
    let view = unsafe { tenso_decode(packet.as_ptr(), packet.len()) };
    assert!(!view.is_null(), "decode failed: {}", last_error());

    unsafe {
        assert_eq!(tenso_view_dtype(view), dtype_code);
        assert_eq!(tenso_view_ndim(view), 1);

        let shp = tenso_view_shape(view);
        assert!(!shp.is_null());
        assert_eq!(*shp, 4);

        let blen = tenso_view_body_len(view);
        assert_eq!(blen, data.len(), "body len mismatch");

        let bptr = tenso_view_body_ptr(view);
        assert!(!bptr.is_null());
        let body = std::slice::from_raw_parts(bptr, blen);
        assert_eq!(body, &data[..], "round-tripped body differs");

        tenso_view_free(view);
    }
}

#[test]
fn null_guards_return_errors() {
    // null out for parse_header.
    let rc = unsafe { tenso_parse_header(std::ptr::null(), 0, std::ptr::null_mut()) };
    assert_eq!(rc, TENSO_ERR_NULL);

    // null data with non-zero len.
    let mut hdr = std::mem::MaybeUninit::<TensoHeader>::uninit();
    let rc = unsafe { tenso_parse_header(std::ptr::null(), 8, hdr.as_mut_ptr()) };
    assert_eq!(rc, TENSO_ERR_NULL);

    // decode of garbage -> NULL + message.
    let junk = [0u8; 4];
    let v = unsafe { tenso_decode(junk.as_ptr(), junk.len()) };
    assert!(v.is_null());
    assert!(!last_error().is_empty());

    // null view accessors are total (no UB, defined sentinels).
    unsafe {
        assert_eq!(tenso_view_dtype(std::ptr::null()), 0);
        assert_eq!(tenso_view_ndim(std::ptr::null()), 0);
        assert!(tenso_view_shape(std::ptr::null()).is_null());
        assert!(tenso_view_body_ptr(std::ptr::null()).is_null());
        assert_eq!(tenso_view_body_len(std::ptr::null()), 0);
        tenso_view_free(std::ptr::null_mut()); // no-op
    }
}

#[test]
fn bad_dtype_is_reported() {
    let data = [0u8; 4];
    let shape = [1u32];
    let mut needed = 0usize;
    let rc = unsafe {
        tenso_dense_required_size(
            data.as_ptr(),
            data.len(),
            200, // invalid dtype code
            shape.as_ptr(),
            shape.len(),
            false,
            false,
            0,
            &mut needed,
        )
    };
    assert_eq!(rc, TENSO_ERR_BAD_DTYPE);
    assert!(last_error().contains("dtype"));
}

#[test]
fn buffer_too_small_is_reported() {
    let elems: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let data: Vec<u8> = elems.iter().flat_map(|f| f.to_le_bytes()).collect();
    let shape = [4u32];
    let mut tiny = [0u8; 4];
    let mut written = 0usize;
    let rc = unsafe {
        tenso_encode_dense_into(
            data.as_ptr(),
            data.len(),
            1,
            shape.as_ptr(),
            shape.len(),
            false,
            false,
            0,
            tiny.as_mut_ptr(),
            tiny.len(),
            &mut written,
        )
    };
    assert_eq!(rc, TENSO_ERR_BUFFER_TOO_SMALL);
}
