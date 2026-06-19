//! tenso-ffi: stable C ABI over tenso-core.
//!
//! All exported functions are `extern "C"`, `#[no_mangle]`, snake_case with a
//! `tenso_` prefix. Only opaque handles cross the boundary. Buffer ownership:
//!   - Mode A (caller-allocates): `tenso_dense_required_size` + `tenso_encode_dense_into`.
//!   - Mode B (core-allocates):   `tenso_decode` returns an opaque `TensoView`
//!     handle, freed by `tenso_view_free`. (There is no raw-byte-buffer Mode B
//!     output, hence no `tenso_bytes_free`.)
//!
//! `cbindgen` regenerates `include/tenso.h` from these signatures (see
//! `build.rs` + `cbindgen.toml`).
//!
//! Safety contract: every incoming pointer is treated as untrusted. Null
//! pointers and zero-with-nonnull / nonzero-with-null `(ptr, len)` mismatches
//! are rejected with a `TENSO_ERR_NULL` status (or NULL return for
//! pointer-returning entry points) and a thread-local error message. No panic
//! is ever allowed to unwind across the FFI boundary: the decode/encode calls
//! into `tenso-core` are wrapped in `catch_unwind`.

#![allow(clippy::missing_safety_doc)]

use core::ffi::{c_char, c_int};
use std::cell::RefCell;
use std::ffi::CString;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use tenso_core::{decode, dense_required_size, encode_dense_into, parse_header};
use tenso_core::{ArraySpec, Decoded, Dtype, EncodeOpts, TensoError};

// =============================================================================
// Status codes (returned by integer-returning entry points)
// =============================================================================

pub const TENSO_OK: c_int = 0;
pub const TENSO_ERR_TOO_SHORT: c_int = -1;
pub const TENSO_ERR_BAD_MAGIC: c_int = -2;
pub const TENSO_ERR_UNSUPPORTED_VERSION: c_int = -3;
pub const TENSO_ERR_BAD_DTYPE: c_int = -4;
pub const TENSO_ERR_TOO_MANY_DIMS: c_int = -5;
pub const TENSO_ERR_TOO_MANY_ELEMENTS: c_int = -6;
pub const TENSO_ERR_INTEGRITY: c_int = -7;
pub const TENSO_ERR_BAD_BUNDLE: c_int = -8;
pub const TENSO_ERR_LZ4: c_int = -9;
pub const TENSO_ERR_BUFFER_TOO_SMALL: c_int = -10;
pub const TENSO_ERR_NULL: c_int = -11;
pub const TENSO_ERR_MALFORMED: c_int = -12;
/// A Rust panic was caught at the boundary (should not normally happen).
pub const TENSO_ERR_PANIC: c_int = -13;
/// The decoded packet is structured (bundle/sparse) and has no flat dense body;
/// use a structured decode path instead of `tenso_view_*`.
pub const TENSO_ERR_UNSUPPORTED_KIND: c_int = -14;

// =============================================================================
// Thread-local last-error
// =============================================================================

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store a message for `tenso_last_error()` on this thread. NUL bytes in the
/// message are stripped so `CString::new` cannot fail.
fn set_last_error(msg: &str) {
    let sanitized: String = msg.chars().filter(|&c| c != '\0').collect();
    let c = CString::new(sanitized).unwrap_or_else(|_| CString::new("tenso: error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Map a `TensoError` to a status code and record a human-readable message.
fn report_error(err: &TensoError) -> c_int {
    let (code, msg): (c_int, String) = match err {
        TensoError::TooShort => (TENSO_ERR_TOO_SHORT, "tenso: input too short".into()),
        TensoError::BadMagic => (TENSO_ERR_BAD_MAGIC, "tenso: bad magic".into()),
        TensoError::UnsupportedVersion(v) => (
            TENSO_ERR_UNSUPPORTED_VERSION,
            format!("tenso: unsupported version {v}"),
        ),
        TensoError::BadDtype(c) => (TENSO_ERR_BAD_DTYPE, format!("tenso: bad dtype code {c}")),
        TensoError::TooManyDims => (TENSO_ERR_TOO_MANY_DIMS, "tenso: too many dims".into()),
        TensoError::TooManyElements => (
            TENSO_ERR_TOO_MANY_ELEMENTS,
            "tenso: too many elements".into(),
        ),
        TensoError::IntegrityMismatch => (TENSO_ERR_INTEGRITY, "tenso: integrity mismatch".into()),
        TensoError::BadBundle => (TENSO_ERR_BAD_BUNDLE, "tenso: bad bundle".into()),
        TensoError::Lz4(reason) => (TENSO_ERR_LZ4, format!("tenso: lz4: {reason}")),
        TensoError::BufferTooSmall => {
            (TENSO_ERR_BUFFER_TOO_SMALL, "tenso: buffer too small".into())
        }
        TensoError::Malformed => (TENSO_ERR_MALFORMED, "tenso: malformed packet".into()),
    };
    set_last_error(&msg);
    code
}

// =============================================================================
// Opaque handles
// =============================================================================

/// Header view, returned by value through an out-pointer. `#[repr(C)]` so the
/// layout is stable across the boundary; this is NOT opaque.
#[repr(C)]
pub struct TensoHeader {
    pub version: u8,
    pub flags: u16,
    pub dtype_code: u8,
    pub ndim: u32,
    pub base_size: u32,
}

/// Opaque decoded-view handle. Internals are private to the ffi crate.
///
/// We own the packet bytes for the view's lifetime and pre-extract the owned
/// `shape`/`dtype`/`body` so accessor pointers stay stable and valid until
/// `tenso_view_free`. `body` is a fresh allocation (owned), which keeps the
/// pointer valid even when the underlying decode produced an owned (e.g.
/// decompressed) buffer.
pub struct TensoView {
    dtype_code: u8,
    shape: Vec<u32>,
    body: Vec<u8>,
}

// =============================================================================
// Header
// =============================================================================

/// Parse a packet header into `out`. Returns a TENSO_* status code.
///
/// # Safety
/// `data` must point to `len` readable bytes; `out` must be a valid, writable
/// `TensoHeader`.
#[no_mangle]
pub unsafe extern "C" fn tenso_parse_header(
    data: *const u8,
    len: usize,
    out: *mut TensoHeader,
) -> c_int {
    clear_last_error();
    if out.is_null() {
        set_last_error("tenso: null out pointer");
        return TENSO_ERR_NULL;
    }
    let bytes = match slice_from_raw(data, len) {
        Ok(b) => b,
        Err(code) => return code,
    };

    let result = catch_unwind(AssertUnwindSafe(|| parse_header(bytes)));
    match result {
        Ok(Ok(h)) => {
            // ndim/base_size are bounded (MAX_NDIM, header sizes); the casts to
            // u32 are lossless in practice but saturate defensively.
            let hdr = TensoHeader {
                version: h.version,
                flags: h.flags,
                dtype_code: h.dtype_code,
                ndim: h.ndim.min(u32::MAX as usize) as u32,
                base_size: h.base_size.min(u32::MAX as usize) as u32,
            };
            ptr::write(out, hdr);
            TENSO_OK
        }
        Ok(Err(e)) => report_error(&e),
        Err(_) => {
            set_last_error("tenso: panic in parse_header");
            TENSO_ERR_PANIC
        }
    }
}

// =============================================================================
// Dense encode (Mode A: caller-allocates)
// =============================================================================

/// Build an `ArraySpec`/`EncodeOpts` from C args, validating pointers.
///
/// Returns `Err(status)` on any pointer/dtype problem (with the thread-local
/// error already set), or `Ok((spec, opts))` on success.
///
/// # Safety
/// Callers guarantee `data`/`shape` point to `data_len`/`ndim` valid elements.
unsafe fn build_spec_opts<'a>(
    data: *const u8,
    data_len: usize,
    dtype_code: u8,
    shape: *const u32,
    ndim: usize,
    check_integrity: bool,
    compress: bool,
    alignment: usize,
) -> Result<(ArraySpec<'a>, EncodeOpts), c_int> {
    let dtype = match Dtype::from_code(dtype_code) {
        Ok(d) => d,
        Err(e) => return Err(report_error(&e)),
    };
    let data_slice = slice_from_raw(data, data_len)?;
    let shape_slice = slice_from_raw_t::<u32>(shape, ndim)?;

    // Default alignment (0) falls back to the protocol default; otherwise it
    // must be a power of two. tenso-core re-validates, but a clear message here
    // helps callers.
    let alignment = if alignment == 0 {
        tenso_core::ALIGNMENT
    } else {
        alignment
    };

    let spec = ArraySpec {
        data: data_slice,
        dtype,
        shape: shape_slice,
    };
    let opts = EncodeOpts {
        check_integrity,
        compress,
        alignment,
    };
    Ok((spec, opts))
}

/// Compute the required output size for a dense encode into `*out_size`.
///
/// # Safety
/// `data` must point to `data_len` readable bytes; `shape` to `ndim` u32s;
/// `out_size` must be a valid, writable `usize`.
#[no_mangle]
pub unsafe extern "C" fn tenso_dense_required_size(
    data: *const u8,
    data_len: usize,
    dtype_code: u8,
    shape: *const u32,
    ndim: usize,
    check_integrity: bool,
    compress: bool,
    alignment: usize,
    out_size: *mut usize,
) -> c_int {
    clear_last_error();
    if out_size.is_null() {
        set_last_error("tenso: null out_size pointer");
        return TENSO_ERR_NULL;
    }

    let (spec, opts) = match build_spec_opts(
        data,
        data_len,
        dtype_code,
        shape,
        ndim,
        check_integrity,
        compress,
        alignment,
    ) {
        Ok(v) => v,
        Err(code) => return code,
    };

    let result = catch_unwind(AssertUnwindSafe(|| dense_required_size(&spec, &opts)));
    match result {
        Ok(Ok(n)) => {
            ptr::write(out_size, n);
            TENSO_OK
        }
        Ok(Err(e)) => report_error(&e),
        Err(_) => {
            set_last_error("tenso: panic in dense_required_size");
            TENSO_ERR_PANIC
        }
    }
}

/// Encode a dense tensor into a caller-allocated `out` buffer.
/// Writes the byte count to `*written`.
///
/// # Safety
/// All pointers must be valid; `out` must have `out_cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn tenso_encode_dense_into(
    data: *const u8,
    data_len: usize,
    dtype_code: u8,
    shape: *const u32,
    ndim: usize,
    check_integrity: bool,
    compress: bool,
    alignment: usize,
    out: *mut u8,
    out_cap: usize,
    written: *mut usize,
) -> c_int {
    clear_last_error();
    if written.is_null() {
        set_last_error("tenso: null written pointer");
        return TENSO_ERR_NULL;
    }

    let (spec, opts) = match build_spec_opts(
        data,
        data_len,
        dtype_code,
        shape,
        ndim,
        check_integrity,
        compress,
        alignment,
    ) {
        Ok(v) => v,
        Err(code) => return code,
    };

    let out_slice = match mut_slice_from_raw(out, out_cap) {
        Ok(s) => s,
        Err(code) => return code,
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        encode_dense_into(&spec, out_slice, &opts)
    }));
    match result {
        Ok(Ok(n)) => {
            ptr::write(written, n);
            TENSO_OK
        }
        Ok(Err(e)) => report_error(&e),
        Err(_) => {
            set_last_error("tenso: panic in encode_dense_into");
            TENSO_ERR_PANIC
        }
    }
}

// =============================================================================
// Decode (Mode B: core-allocates an opaque view)
// =============================================================================

/// Decode a packet, returning an opaque `TensoView*` (NULL on error; check
/// `tenso_last_error`). Free with `tenso_view_free`.
///
/// Only flat dense and quantized tensors expose a single body through the
/// `tenso_view_*` accessors. Bundles, sparse, string, ragged, and IPC-ref
/// packets decode successfully in core but have no single flat body; this entry
/// point reports `TENSO_ERR_UNSUPPORTED_KIND` for them (via `tenso_last_error`)
/// and returns NULL.
///
/// # Safety
/// `data` must point to `len` readable bytes. The returned view owns its own
/// copy of the relevant body, so it stays valid after `data` is freed.
#[no_mangle]
pub unsafe extern "C" fn tenso_decode(data: *const u8, len: usize) -> *mut TensoView {
    clear_last_error();
    let bytes = match slice_from_raw(data, len) {
        Ok(b) => b,
        Err(_) => return ptr::null_mut(),
    };

    // Copy the packet so the borrowed `Decoded` cannot outlive `data`, and so
    // the eventual view is fully self-owned.
    let owned: Vec<u8> = bytes.to_vec();

    let result = catch_unwind(AssertUnwindSafe(|| build_view(&owned)));
    match result {
        Ok(Ok(view)) => Box::into_raw(Box::new(view)),
        Ok(Err(code)) => {
            // build_view already set the thread-local message for the success-
            // but-unsupported case; map_decode_err set it for core errors.
            let _ = code;
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("tenso: panic in decode");
            ptr::null_mut()
        }
    }
}

/// Decode `owned` and extract an owned `TensoView` for the flat-body kinds.
fn build_view(owned: &[u8]) -> Result<TensoView, c_int> {
    let decoded = decode(owned).map_err(|e| report_error(&e))?;
    match decoded {
        Decoded::Dense(t) => Ok(TensoView {
            dtype_code: t.dtype.code(),
            shape: t.shape,
            body: t.body.to_vec(),
        }),
        Decoded::Quantized(q) => Ok(TensoView {
            dtype_code: q.dtype.code(),
            shape: q.shape,
            // Quantized "body" surfaced through the flat accessors is the packed
            // payload. Richer scale/zero-point access is a future structured FFI.
            body: q.packed.to_vec(),
        }),
        Decoded::Bundle(_)
        | Decoded::Sparse { .. }
        | Decoded::String { .. }
        | Decoded::Ragged { .. }
        | Decoded::IpcRef(_) => {
            set_last_error(
                "tenso: decoded packet is structured (bundle/sparse/string/ragged/ipc-ref); \
                 no flat body — use a structured decode path",
            );
            Err(TENSO_ERR_UNSUPPORTED_KIND)
        }
    }
}

/// Borrow a `&TensoView` from a raw pointer, returning `None` on null.
///
/// # Safety
/// `view` must be a live pointer from `tenso_decode` or null.
unsafe fn view_ref<'a>(view: *const TensoView) -> Option<&'a TensoView> {
    if view.is_null() {
        None
    } else {
        Some(&*view)
    }
}

/// Dtype code of a decoded view (0 if `view` is null).
///
/// # Safety
/// `view` must be a live pointer from `tenso_decode` or null.
#[no_mangle]
pub unsafe extern "C" fn tenso_view_dtype(view: *const TensoView) -> u8 {
    match view_ref(view) {
        Some(v) => v.dtype_code,
        None => 0,
    }
}

/// Number of dimensions of a decoded view (0 if `view` is null).
///
/// # Safety
/// `view` must be a live pointer from `tenso_decode` or null.
#[no_mangle]
pub unsafe extern "C" fn tenso_view_ndim(view: *const TensoView) -> usize {
    match view_ref(view) {
        Some(v) => v.shape.len(),
        None => 0,
    }
}

/// Pointer to the view's shape array (`tenso_view_ndim` u32 entries), or NULL.
///
/// For a 0-d (scalar) tensor `ndim == 0` and this may return a non-null but
/// zero-length pointer; callers must gate on `tenso_view_ndim`.
///
/// # Safety
/// `view` must be a live pointer from `tenso_decode` or null. The returned
/// pointer is valid until `tenso_view_free(view)`.
#[no_mangle]
pub unsafe extern "C" fn tenso_view_shape(view: *const TensoView) -> *const u32 {
    match view_ref(view) {
        Some(v) => v.shape.as_ptr(),
        None => ptr::null(),
    }
}

/// Pointer to the view's body bytes, or NULL.
///
/// # Safety
/// `view` must be a live pointer from `tenso_decode` or null. The returned
/// pointer is valid until `tenso_view_free(view)`.
#[no_mangle]
pub unsafe extern "C" fn tenso_view_body_ptr(view: *const TensoView) -> *const u8 {
    match view_ref(view) {
        Some(v) => v.body.as_ptr(),
        None => ptr::null(),
    }
}

/// Length in bytes of the view's body (0 if `view` is null).
///
/// # Safety
/// `view` must be a live pointer from `tenso_decode` or null.
#[no_mangle]
pub unsafe extern "C" fn tenso_view_body_len(view: *const TensoView) -> usize {
    match view_ref(view) {
        Some(v) => v.body.len(),
        None => 0,
    }
}

/// Free a view returned by `tenso_decode`. No-op on null.
///
/// # Safety
/// `view` must be a pointer from `tenso_decode`, freed at most once. After this
/// call the pointer and any accessor results derived from it are dangling.
#[no_mangle]
pub unsafe extern "C" fn tenso_view_free(view: *mut TensoView) {
    if !view.is_null() {
        drop(Box::from_raw(view));
    }
}

// NOTE: there is intentionally no `tenso_bytes_free`. The only Mode B
// (core-allocates) path is `tenso_decode`, which returns an opaque `TensoView`
// handle freed by `tenso_view_free`; no exported function hands a raw
// `(ptr, len)` byte buffer to the caller. A free fn with no matching allocator
// is dead and a latent-UB footgun (`Vec::from_raw_parts` requires a global-
// allocator `Vec<u8>` with `capacity == len`), so it was removed.

// =============================================================================
// Error reporting
// =============================================================================

/// Return a pointer to a thread-local, NUL-terminated description of the last
/// error on this thread (valid until the next ffi call on the same thread).
///
/// Returns an empty string `""` (not NULL) when there is no recorded error.
#[no_mangle]
pub extern "C" fn tenso_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(c) => c.as_ptr(),
        None => EMPTY.as_ptr() as *const c_char,
    })
}

/// Static empty C string used when no error is set.
static EMPTY: [u8; 1] = [0];

// =============================================================================
// Pointer/slice guards (untrusted inputs)
// =============================================================================

/// Build a `&[u8]` from a raw pointer + length, rejecting null/len mismatches.
///
/// - `(null, 0)` -> empty slice (legal: a zero-length input).
/// - `(null, n>0)` -> `TENSO_ERR_NULL`.
/// - `(nonnull, 0)` -> empty slice (we never deref).
///
/// # Safety
/// If non-null, `data` must be readable for `len` bytes.
unsafe fn slice_from_raw<'a>(data: *const u8, len: usize) -> Result<&'a [u8], c_int> {
    if data.is_null() {
        if len == 0 {
            Ok(&[])
        } else {
            set_last_error("tenso: null data pointer with non-zero length");
            Err(TENSO_ERR_NULL)
        }
    } else if len == 0 {
        Ok(&[])
    } else {
        Ok(std::slice::from_raw_parts(data, len))
    }
}

/// Typed variant of `slice_from_raw` for `*const T` arrays (e.g. shape u32s).
///
/// # Safety
/// If non-null, `data` must be readable for `count` `T`s and properly aligned.
unsafe fn slice_from_raw_t<'a, T>(data: *const T, count: usize) -> Result<&'a [T], c_int> {
    if data.is_null() {
        if count == 0 {
            Ok(&[])
        } else {
            set_last_error("tenso: null array pointer with non-zero count");
            Err(TENSO_ERR_NULL)
        }
    } else if count == 0 {
        Ok(&[])
    } else {
        Ok(std::slice::from_raw_parts(data, count))
    }
}

/// Mutable `&mut [u8]` from a raw pointer + capacity, rejecting null/len issues.
///
/// # Safety
/// If non-null, `out` must be writable for `cap` bytes.
unsafe fn mut_slice_from_raw<'a>(out: *mut u8, cap: usize) -> Result<&'a mut [u8], c_int> {
    if out.is_null() {
        if cap == 0 {
            Ok(&mut [])
        } else {
            set_last_error("tenso: null out pointer with non-zero capacity");
            Err(TENSO_ERR_NULL)
        }
    } else if cap == 0 {
        Ok(&mut [])
    } else {
        Ok(std::slice::from_raw_parts_mut(out, cap))
    }
}
