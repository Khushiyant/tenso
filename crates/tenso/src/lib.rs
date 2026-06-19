//! tenso: pure-Rust engine for the Tenso tensor wire format.
//!
//! `no_std + alloc` friendly; `std` feature gates streaming writers + lz4_flex. No pyo3/numpy/libc.
//! Byte-level authority: root `src/lib.rs`, `src/tenso/{config,core,quantize,ragged}.py`.
//! Protocol constants, `Dtype`, `Header`, `parse_header`, `write_v4_header` and IpcRef framing
//! are load-bearing across crates and must stay byte-identical to those files.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code, unused)]

extern crate alloc;

use alloc::string::String;
#[allow(unused_imports)]
use alloc::vec; // `vec!` macro under no_std
use alloc::vec::Vec;

// =============================================================================
// Protocol constants (REAL — must match config.py / root src/lib.rs exactly)
// =============================================================================

/// Magic number prefixing every Tenso packet.
pub const MAGIC: [u8; 4] = *b"TNSO";
/// Current protocol version.
pub const VERSION: u8 = 4;
/// v3 header size: magic(4) + ver(1) + flags_u8(1) + dtype(1) + ndim(1).
pub const HEADER_BASE_V3: usize = 8;
/// v4 header size: magic(4) + ver(1) + flags_u16(2) + dtype(1) + ndim(1) + reserved(1).
pub const HEADER_BASE_V4: usize = 10;
/// Default body alignment (AVX-512 friendly).
pub const ALIGNMENT: usize = 64;

/// Maximum number of dimensions (DoS protection).
pub const MAX_NDIM: usize = 32;
/// Maximum elements per tensor (DoS protection).
pub const MAX_ELEMENTS: u64 = 1_000_000_000;

// --- Protocol flags (u16 in v4) ---
pub const FLAG_ALIGNED: u16 = 1;
pub const FLAG_INTEGRITY: u16 = 2;
pub const FLAG_COMPRESSION: u16 = 4;
pub const FLAG_SPARSE_COO: u16 = 8;
pub const FLAG_BUNDLE: u16 = 16;
pub const FLAG_SPARSE_CSR: u16 = 32;
pub const FLAG_SPARSE_CSC: u16 = 64;
pub const FLAG_CUST_ALIGN: u16 = 128;
pub const FLAG_STRING: u16 = 256;
pub const FLAG_RAGGED: u16 = 512;
/// NEW (bit 10): packet is a GPU IPC reference rather than an inline body.
pub const FLAG_GPU_IPC_REF: u16 = 1024;

// --- dtype codes (must match config.py) ---
pub const DCODE_F32: u8 = 1;
pub const DCODE_I32: u8 = 2;
pub const DCODE_F64: u8 = 3;
pub const DCODE_I64: u8 = 4;
pub const DCODE_U8: u8 = 5;
pub const DCODE_U16: u8 = 6;
pub const DCODE_BOOL: u8 = 7;
pub const DCODE_F16: u8 = 8;
pub const DCODE_I8: u8 = 9;
pub const DCODE_I16: u8 = 10;
pub const DCODE_U32: u8 = 11;
pub const DCODE_U64: u8 = 12;
pub const DCODE_C64: u8 = 13;
pub const DCODE_C128: u8 = 14;
pub const DCODE_BF16: u8 = 15;
pub const DCODE_QINT8: u8 = 16;
pub const DCODE_QUINT8: u8 = 17;
pub const DCODE_QINT4: u8 = 18;
pub const DCODE_QUINT4: u8 = 19;
pub const DCODE_STR: u8 = 20;
pub const DCODE_BYTES: u8 = 21;

// --- Quantization schemes (must match config.py) ---
pub const QUANT_PER_TENSOR: u8 = 0;
pub const QUANT_PER_CHANNEL: u8 = 1;
pub const QUANT_PER_GROUP: u8 = 2;

// LZ4 frame magic (little-endian on the wire): 04 22 4D 18.
const LZ4_FRAME_MAGIC: [u8; 4] = [0x04, 0x22, 0x4D, 0x18];

// =============================================================================
// GPU IpcRef framing constants (REAL — load-bearing across crates)
// =============================================================================
//
// v4 header with FLAG_GPU_IPC_REF + reserved byte (offset 9) == discriminator 1, then a
// fixed 96-byte LE body: handle[64], byte_offset:u64, nbytes:u64, device_uuid[16].
// Never combine with FLAG_INTEGRITY or an inline body; reject device_uuid mismatch on import.

/// Value placed in the v4 reserved byte (offset 9) to mark an IpcRef packet.
pub const IPC_REF_DISCRIMINATOR: u8 = 1;
/// Size of the opaque IPC handle blob.
pub const IPC_REF_HANDLE_LEN: usize = 64;
/// Size of the device UUID blob.
pub const IPC_REF_DEVICE_UUID_LEN: usize = 16;
/// Fixed IpcRef body size: handle(64) + byte_offset(8) + nbytes(8) + uuid(16).
pub const IPC_REF_BODY_LEN: usize = IPC_REF_HANDLE_LEN + 8 + 8 + IPC_REF_DEVICE_UUID_LEN; // 96
/// Total IpcRef packet size: v4 header + fixed body.
pub const IPC_REF_PACKET_LEN: usize = HEADER_BASE_V4 + IPC_REF_BODY_LEN; // 106

/// Decoded GPU IPC reference (no GPU access here; just the wire payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcRef {
    pub handle: [u8; IPC_REF_HANDLE_LEN],
    pub byte_offset: u64,
    pub nbytes: u64,
    pub device_uuid: [u8; IPC_REF_DEVICE_UUID_LEN],
}

// =============================================================================
// Dtype (REAL — load-bearing)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    I32,
    F64,
    I64,
    U8,
    U16,
    Bool,
    F16,
    I8,
    I16,
    U32,
    U64,
    C64,
    C128,
    BF16,
    QInt8,
    QUInt8,
    QInt4,
    QUInt4,
    Str,
    Bytes,
}

impl Dtype {
    /// Map a wire dtype code to a `Dtype`.
    pub fn from_code(code: u8) -> Result<Dtype, TensoError> {
        let d = match code {
            DCODE_F32 => Dtype::F32,
            DCODE_I32 => Dtype::I32,
            DCODE_F64 => Dtype::F64,
            DCODE_I64 => Dtype::I64,
            DCODE_U8 => Dtype::U8,
            DCODE_U16 => Dtype::U16,
            DCODE_BOOL => Dtype::Bool,
            DCODE_F16 => Dtype::F16,
            DCODE_I8 => Dtype::I8,
            DCODE_I16 => Dtype::I16,
            DCODE_U32 => Dtype::U32,
            DCODE_U64 => Dtype::U64,
            DCODE_C64 => Dtype::C64,
            DCODE_C128 => Dtype::C128,
            DCODE_BF16 => Dtype::BF16,
            DCODE_QINT8 => Dtype::QInt8,
            DCODE_QUINT8 => Dtype::QUInt8,
            DCODE_QINT4 => Dtype::QInt4,
            DCODE_QUINT4 => Dtype::QUInt4,
            DCODE_STR => Dtype::Str,
            DCODE_BYTES => Dtype::Bytes,
            other => return Err(TensoError::BadDtype(other)),
        };
        Ok(d)
    }

    /// The wire dtype code for this `Dtype`.
    pub fn code(&self) -> u8 {
        match self {
            Dtype::F32 => DCODE_F32,
            Dtype::I32 => DCODE_I32,
            Dtype::F64 => DCODE_F64,
            Dtype::I64 => DCODE_I64,
            Dtype::U8 => DCODE_U8,
            Dtype::U16 => DCODE_U16,
            Dtype::Bool => DCODE_BOOL,
            Dtype::F16 => DCODE_F16,
            Dtype::I8 => DCODE_I8,
            Dtype::I16 => DCODE_I16,
            Dtype::U32 => DCODE_U32,
            Dtype::U64 => DCODE_U64,
            Dtype::C64 => DCODE_C64,
            Dtype::C128 => DCODE_C128,
            Dtype::BF16 => DCODE_BF16,
            Dtype::QInt8 => DCODE_QINT8,
            Dtype::QUInt8 => DCODE_QUINT8,
            Dtype::QInt4 => DCODE_QINT4,
            Dtype::QUInt4 => DCODE_QUINT4,
            Dtype::Str => DCODE_STR,
            Dtype::Bytes => DCODE_BYTES,
        }
    }

    /// Fixed per-element byte size, or `None` for sub-byte (4-bit) / variable-length dtypes.
    pub fn item_size(&self) -> Option<usize> {
        let s = match self {
            Dtype::F32 => 4,
            Dtype::I32 => 4,
            Dtype::F64 => 8,
            Dtype::I64 => 8,
            Dtype::U8 => 1,
            Dtype::U16 => 2,
            Dtype::Bool => 1,
            Dtype::F16 => 2,
            Dtype::I8 => 1,
            Dtype::I16 => 2,
            Dtype::U32 => 4,
            Dtype::U64 => 8,
            Dtype::C64 => 8,
            Dtype::C128 => 16,
            Dtype::BF16 => 2,
            Dtype::QInt8 => 1,
            Dtype::QUInt8 => 1,
            // sub-byte packed (2 per byte): no whole-byte item size
            Dtype::QInt4 => return None,
            Dtype::QUInt4 => return None,
            // variable-length
            Dtype::Str => return None,
            Dtype::Bytes => return None,
        };
        Some(s)
    }

    /// Canonical numpy-style name.
    pub fn name(&self) -> &'static str {
        match self {
            Dtype::F32 => "float32",
            Dtype::I32 => "int32",
            Dtype::F64 => "float64",
            Dtype::I64 => "int64",
            Dtype::U8 => "uint8",
            Dtype::U16 => "uint16",
            Dtype::Bool => "bool",
            Dtype::F16 => "float16",
            Dtype::I8 => "int8",
            Dtype::I16 => "int16",
            Dtype::U32 => "uint32",
            Dtype::U64 => "uint64",
            Dtype::C64 => "complex64",
            Dtype::C128 => "complex128",
            Dtype::BF16 => "bfloat16",
            Dtype::QInt8 => "qint8",
            Dtype::QUInt8 => "quint8",
            Dtype::QInt4 => "qint4",
            Dtype::QUInt4 => "quint4",
            Dtype::Str => "string",
            Dtype::Bytes => "bytes",
        }
    }

    /// True for the four quantized dtype codes (16..=19).
    fn is_quantized(&self) -> bool {
        matches!(
            self,
            Dtype::QInt8 | Dtype::QUInt8 | Dtype::QInt4 | Dtype::QUInt4
        )
    }

    /// True for the two 4-bit packed quant dtypes (18, 19).
    fn is_4bit(&self) -> bool {
        matches!(self, Dtype::QInt4 | Dtype::QUInt4)
    }
}

// =============================================================================
// Header (REAL — load-bearing)
// =============================================================================

/// Parsed packet header (version-aware: v3 = 8-byte base, v4 = 10-byte base).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub version: u8,
    pub flags: u16,
    pub dtype_code: u8,
    pub ndim: usize,
    pub base_size: usize,
}

/// Parse a v3/v4 packet header (mirrors root `parse_packet_header_raw`).
pub fn parse_header(bytes: &[u8]) -> Result<Header, TensoError> {
    if bytes.len() < HEADER_BASE_V3 {
        return Err(TensoError::TooShort);
    }
    if bytes[0..4] != MAGIC {
        return Err(TensoError::BadMagic);
    }
    let version = bytes[4];
    match version {
        3 => Ok(Header {
            version,
            flags: bytes[5] as u16,
            dtype_code: bytes[6],
            ndim: bytes[7] as usize,
            base_size: HEADER_BASE_V3,
        }),
        4 => {
            if bytes.len() < HEADER_BASE_V4 {
                return Err(TensoError::TooShort);
            }
            Ok(Header {
                version,
                flags: u16::from_le_bytes([bytes[5], bytes[6]]),
                dtype_code: bytes[7],
                ndim: bytes[8] as usize,
                base_size: HEADER_BASE_V4,
            })
        }
        v => Err(TensoError::UnsupportedVersion(v)),
    }
}

/// Write a v4 header into `out[0..10]` (mirrors root `write_v4_header`).
pub fn write_v4_header(out: &mut [u8], flags: u16, dtype_code: u8, ndim: u8) {
    out[0..4].copy_from_slice(&MAGIC);
    out[4] = VERSION;
    out[5..7].copy_from_slice(&flags.to_le_bytes());
    out[7] = dtype_code;
    out[8] = ndim;
    out[9] = 0; // reserved
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Padding to advance `pos` to the next `alignment` boundary (0/1 => no padding).
#[inline]
fn padding_for(pos: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return 0;
    }
    let remainder = pos % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

/// Product of shape dims as u64; overflow => `TooManyElements`.
#[inline]
fn shape_num_elements(shape: &[u32]) -> Result<u64, TensoError> {
    let mut acc: u64 = 1;
    for &d in shape {
        acc = acc
            .checked_mul(d as u64)
            .ok_or(TensoError::TooManyElements)?;
    }
    Ok(acc)
}

/// Single-pass XXH3-64 over `data` (matches Python `xxhash.xxh3_64_intdigest`).
#[cfg(feature = "integrity")]
#[inline]
fn integrity_hash(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}

/// Read a little-endian u32 from `bytes[off..off+4]`, bounds-checked.
#[inline]
fn read_u32(bytes: &[u8], off: usize) -> Result<u32, TensoError> {
    let end = off.checked_add(4).ok_or(TensoError::Malformed)?;
    if end > bytes.len() {
        return Err(TensoError::TooShort);
    }
    Ok(u32::from_le_bytes([
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
    ]))
}

/// Read a little-endian u64 from `bytes[off..off+8]`, bounds-checked.
#[inline]
fn read_u64(bytes: &[u8], off: usize) -> Result<u64, TensoError> {
    let end = off.checked_add(8).ok_or(TensoError::Malformed)?;
    if end > bytes.len() {
        return Err(TensoError::TooShort);
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[off..off + 8]);
    Ok(u64::from_le_bytes(arr))
}

/// Decompress an LZ4 *frame* (matching Python's `lz4.frame`).
#[cfg(feature = "compression")]
fn lz4_decompress_frame(body: &[u8]) -> Result<Vec<u8>, TensoError> {
    use std::io::Read;
    let mut decoder = lz4_flex::frame::FrameDecoder::new(body);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|_| TensoError::Lz4("LZ4 frame decompression failed"))?;
    Ok(out)
}

/// Compress `data` into an LZ4 *frame* (matching Python's `lz4.frame`).
#[cfg(feature = "compression")]
fn lz4_compress_frame(data: &[u8]) -> Result<Vec<u8>, TensoError> {
    use std::io::Write;
    let mut out = Vec::with_capacity(data.len() / 2 + 32);
    {
        let mut encoder = lz4_flex::frame::FrameEncoder::new(&mut out);
        encoder
            .write_all(data)
            .map_err(|_| TensoError::Lz4("LZ4 frame encode failed"))?;
        encoder
            .finish()
            .map_err(|_| TensoError::Lz4("LZ4 frame finish failed"))?;
    }
    Ok(out)
}

// =============================================================================
// Dense encode API
// =============================================================================

/// The core's only dense input: raw bytes + dtype + shape (no numpy).
pub struct ArraySpec<'a> {
    pub data: &'a [u8],
    pub dtype: Dtype,
    pub shape: &'a [u32],
}

/// Encode options.
pub struct EncodeOpts {
    pub check_integrity: bool,
    pub compress: bool,
    pub alignment: usize,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        EncodeOpts {
            check_integrity: false,
            compress: false,
            alignment: ALIGNMENT,
        }
    }
}

/// Shared pre-flight layout for `dense_required_size` / `encode_dense_into`.
struct DenseLayout {
    use_custom_align: bool,
    header_len: usize,
    padding_len: usize,
}

fn dense_layout(spec: &ArraySpec, opts: &EncodeOpts) -> Result<DenseLayout, TensoError> {
    if !opts.alignment.is_power_of_two() {
        // alignment must be a power of two
        return Err(TensoError::Malformed);
    }
    let ndim = spec.shape.len();
    if ndim > MAX_NDIM {
        return Err(TensoError::TooManyDims);
    }
    let num_elements = shape_num_elements(spec.shape)?;
    if num_elements > MAX_ELEMENTS {
        return Err(TensoError::TooManyElements);
    }
    // Dense path is fixed-size dtypes only (variable/quant dtypes have their own packets).
    let item = spec
        .dtype
        .item_size()
        .ok_or(TensoError::BadDtype(spec.dtype.code()))?;
    let expected = (num_elements as usize)
        .checked_mul(item)
        .ok_or(TensoError::TooManyElements)?;
    if spec.data.len() != expected {
        // buffer must be exactly shape*item_size
        return Err(TensoError::Malformed);
    }

    let use_custom_align = opts.alignment != ALIGNMENT;
    let mut header_len = HEADER_BASE_V4 + ndim * 4;
    if use_custom_align {
        header_len += 1;
    }
    let padding_len = padding_for(header_len, opts.alignment);
    Ok(DenseLayout {
        use_custom_align,
        header_len,
        padding_len,
    })
}

/// Compute the exact (uncompressed) / upper-bound (compressed) buffer size
/// needed to encode `spec` with `opts`.
pub fn dense_required_size(spec: &ArraySpec, opts: &EncodeOpts) -> Result<usize, TensoError> {
    let layout = dense_layout(spec, opts)?;
    let footer_len = if opts.check_integrity { 8 } else { 0 };

    let body_len = if opts.compress {
        #[cfg(feature = "compression")]
        {
            // LZ4 frame upper bound: block worst-case + frame overhead + per-block size prefix.
            let nbytes = spec.data.len();
            let block_worst = lz4_flex::block::get_maximum_output_size(nbytes);
            block_worst + 64 + (nbytes / 65536 + 1) * 4
        }
        #[cfg(not(feature = "compression"))]
        {
            return Err(TensoError::Lz4(
                "compression requires the `compression` feature",
            ));
        }
    } else {
        spec.data.len()
    };

    Ok(layout.header_len + layout.padding_len + body_len + footer_len)
}

/// Encode `spec` into `out`, returning the number of bytes written.
pub fn encode_dense_into(
    spec: &ArraySpec,
    out: &mut [u8],
    opts: &EncodeOpts,
) -> Result<usize, TensoError> {
    let layout = dense_layout(spec, opts)?;
    let ndim = spec.shape.len();

    // Resolve the (optionally compressed) body up front to know total length.
    #[cfg(feature = "compression")]
    let compressed: Option<Vec<u8>> = if opts.compress {
        Some(lz4_compress_frame(spec.data)?)
    } else {
        None
    };
    #[cfg(not(feature = "compression"))]
    let compressed: Option<Vec<u8>> = if opts.compress {
        return Err(TensoError::Lz4(
            "compression requires the `compression` feature",
        ));
    } else {
        None
    };

    let body: &[u8] = match &compressed {
        Some(c) => c.as_slice(),
        None => spec.data,
    };
    let body_len = body.len();
    let footer_len = if opts.check_integrity { 8 } else { 0 };
    let total_len = layout.header_len + layout.padding_len + body_len + footer_len;

    if out.len() < total_len {
        return Err(TensoError::BufferTooSmall);
    }

    // Flags: ALIGNED unless custom alignment; INTEGRITY; COMPRESSION.
    let mut flags: u16 = 0;
    if layout.use_custom_align {
        flags |= FLAG_CUST_ALIGN;
    } else {
        flags |= FLAG_ALIGNED;
    }
    if opts.check_integrity {
        flags |= FLAG_INTEGRITY;
    }
    if opts.compress {
        flags |= FLAG_COMPRESSION;
    }

    write_v4_header(out, flags, spec.dtype.code(), ndim as u8);

    let mut cursor = HEADER_BASE_V4;
    for &dim in spec.shape {
        out[cursor..cursor + 4].copy_from_slice(&dim.to_le_bytes());
        cursor += 4;
    }
    if layout.use_custom_align {
        out[cursor] = opts.alignment.trailing_zeros() as u8;
        cursor += 1;
    }
    // cursor == header_len; zero the padding
    debug_assert_eq!(cursor, layout.header_len);
    let body_start = layout.header_len + layout.padding_len;
    for b in &mut out[layout.header_len..body_start] {
        *b = 0;
    }

    out[body_start..body_start + body_len].copy_from_slice(body);

    if opts.check_integrity {
        #[cfg(feature = "integrity")]
        {
            // Integrity covers the on-wire body (compressed or raw); matches core.py.
            let hash = integrity_hash(body);
            let footer_start = body_start + body_len;
            out[footer_start..footer_start + 8].copy_from_slice(&hash.to_le_bytes());
        }
        #[cfg(not(feature = "integrity"))]
        {
            return Err(TensoError::IntegrityMismatch);
        }
    }

    Ok(total_len)
}

// =============================================================================
// Bundle encode API
// =============================================================================

/// Maximum bundle entries: the v4 header stores the count in `ndim` (one byte).
pub const MAX_BUNDLE_ENTRIES: usize = 255;

/// Exact buffer size for a bundle packet.
///
/// Layout: v4 header (FLAG_BUNDLE, dtype 0, ndim = entry count), then per entry
/// `klen:u32 LE | key utf8 | vlen:u32 LE | value packet`. >`MAX_BUNDLE_ENTRIES` errors.
pub fn bundle_required_size(entries: &[(&str, &[u8])]) -> Result<usize, TensoError> {
    if entries.len() > MAX_BUNDLE_ENTRIES {
        return Err(TensoError::BadBundle);
    }
    let mut total = HEADER_BASE_V4;
    for (key, value) in entries {
        // klen(4) + key + vlen(4) + value
        total = total
            .checked_add(4)
            .and_then(|t| t.checked_add(key.len()))
            .and_then(|t| t.checked_add(4))
            .and_then(|t| t.checked_add(value.len()))
            .ok_or(TensoError::Malformed)?;
    }
    Ok(total)
}

/// Encode `entries` as a bundle packet into `out` (layout: [`bundle_required_size`]).
pub fn encode_bundle_into(entries: &[(&str, &[u8])], out: &mut [u8]) -> Result<usize, TensoError> {
    let total = bundle_required_size(entries)?;
    if out.len() < total {
        return Err(TensoError::BufferTooSmall);
    }

    // ndim holds the entry count (<= 255, guaranteed by bundle_required_size).
    write_v4_header(out, FLAG_BUNDLE, 0, entries.len() as u8);

    let mut cursor = HEADER_BASE_V4;
    for (key, value) in entries {
        let key_bytes = key.as_bytes();
        out[cursor..cursor + 4].copy_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        cursor += 4;
        out[cursor..cursor + key_bytes.len()].copy_from_slice(key_bytes);
        cursor += key_bytes.len();
        out[cursor..cursor + 4].copy_from_slice(&(value.len() as u32).to_le_bytes());
        cursor += 4;
        out[cursor..cursor + value.len()].copy_from_slice(value);
        cursor += value.len();
    }

    debug_assert_eq!(cursor, total);
    Ok(total)
}

// =============================================================================
// Sparse encode API
// =============================================================================

/// Sparse component opts: dense, no integrity/compression, 64-byte align (matches Python).
#[inline]
fn sparse_component_opts() -> EncodeOpts {
    EncodeOpts {
        check_integrity: false,
        compress: false,
        alignment: ALIGNMENT,
    }
}

/// Exact buffer size for a sparse packet.
///
/// Layout: v4 header (format flag, dtype 0, ndim = shape.len()), shape[ndim] (u32 LE),
/// then 3x `size:u32 LE | dense sub-packet`. Exactly three components required.
pub fn sparse_required_size(shape: &[u32], components: &[ArraySpec]) -> Result<usize, TensoError> {
    if components.len() != 3 {
        return Err(TensoError::Malformed);
    }
    if shape.len() > MAX_NDIM {
        return Err(TensoError::TooManyDims);
    }
    let opts = sparse_component_opts();
    let mut total = HEADER_BASE_V4
        .checked_add(shape.len().checked_mul(4).ok_or(TensoError::Malformed)?)
        .ok_or(TensoError::Malformed)?;
    for comp in components {
        let sub = dense_required_size(comp, &opts)?;
        // size prefix(4) + sub-packet
        total = total
            .checked_add(4)
            .and_then(|t| t.checked_add(sub))
            .ok_or(TensoError::Malformed)?;
    }
    Ok(total)
}

/// Encode a sparse packet into `out` (layout: [`sparse_required_size`]).
/// Components are dense sub-packets, no integrity/compression, 64-byte aligned.
pub fn encode_sparse_into(
    format: SparseFormat,
    shape: &[u32],
    components: &[ArraySpec],
    out: &mut [u8],
) -> Result<usize, TensoError> {
    if components.len() != 3 {
        return Err(TensoError::Malformed);
    }
    if shape.len() > MAX_NDIM {
        return Err(TensoError::TooManyDims);
    }
    let opts = sparse_component_opts();
    let total = sparse_required_size(shape, components)?;
    if out.len() < total {
        return Err(TensoError::BufferTooSmall);
    }

    let flags = format.flag();
    write_v4_header(out, flags, 0, shape.len() as u8);

    let mut cursor = HEADER_BASE_V4;
    for &dim in shape {
        out[cursor..cursor + 4].copy_from_slice(&dim.to_le_bytes());
        cursor += 4;
    }

    for comp in components {
        let sub_len = dense_required_size(comp, &opts)?;
        out[cursor..cursor + 4].copy_from_slice(&(sub_len as u32).to_le_bytes());
        cursor += 4;
        let written = encode_dense_into(comp, &mut out[cursor..cursor + sub_len], &opts)?;
        debug_assert_eq!(written, sub_len);
        cursor += written;
    }

    debug_assert_eq!(cursor, total);
    Ok(total)
}

// =============================================================================
// Quantized encode API
// =============================================================================

/// A quantized tensor to encode (mirrors `core.py::_serialize_quantized`).
/// `data` is the already-packed body (4-bit dtypes pack 2 elements/byte).
pub struct QuantSpec<'a> {
    pub dtype: Dtype,
    pub shape: &'a [u32],
    pub scheme: u8,
    pub axis: u8,
    pub group_size: u32,
    /// Raw LE f32 bytes, one per tensor/channel/group (on-wire `numpy.float32.tobytes()`).
    pub scales: &'a [u8],
    pub zero_points: &'a [u8],
    pub data: &'a [u8],
}

struct QuantLayout {
    use_custom_align: bool,
    quant_meta_len: usize,
    header_len: usize,
    padding_len: usize,
    body_len: usize,
}

fn quant_layout(spec: &QuantSpec, opts: &EncodeOpts) -> Result<QuantLayout, TensoError> {
    if !opts.alignment.is_power_of_two() {
        return Err(TensoError::Malformed);
    }
    if !matches!(
        spec.dtype,
        Dtype::QInt8 | Dtype::QUInt8 | Dtype::QInt4 | Dtype::QUInt4
    ) {
        return Err(TensoError::BadDtype(spec.dtype.code()));
    }
    let ndim = spec.shape.len();
    if ndim > MAX_NDIM {
        return Err(TensoError::TooManyDims);
    }
    // scales/zero_points: raw f32 bytes, equal length, multiple of 4
    if spec.scales.len() != spec.zero_points.len() || !spec.scales.len().is_multiple_of(4) {
        return Err(TensoError::Malformed);
    }
    let num_elements = shape_num_elements(spec.shape)?;
    if num_elements > MAX_ELEMENTS {
        return Err(TensoError::TooManyElements);
    }
    // meta: scheme(1) + axis(1) + group_size(4) + num_scales(4) + scales + zp
    let quant_meta_len = 1 + 1 + 4 + 4 + spec.scales.len() + spec.zero_points.len();
    let use_custom_align = opts.alignment != ALIGNMENT;
    let mut header_len = HEADER_BASE_V4 + ndim * 4 + quant_meta_len;
    if use_custom_align {
        header_len += 1;
    }
    let padding_len = padding_for(header_len, opts.alignment);
    let body_len = if spec.dtype.is_4bit() {
        (num_elements as usize).div_ceil(2)
    } else {
        num_elements as usize
    };
    if spec.data.len() < body_len {
        return Err(TensoError::Malformed);
    }
    Ok(QuantLayout {
        use_custom_align,
        quant_meta_len,
        header_len,
        padding_len,
        body_len,
    })
}

/// Exact buffer size needed to encode `spec` as a quantized packet.
pub fn quantized_required_size(spec: &QuantSpec, opts: &EncodeOpts) -> Result<usize, TensoError> {
    let l = quant_layout(spec, opts)?;
    let footer_len = if opts.check_integrity { 8 } else { 0 };
    Ok(l.header_len + l.padding_len + l.body_len + footer_len)
}

/// Encode a quantized tensor into `out` (byte-identical to `core.py::_serialize_quantized`).
pub fn encode_quantized_into(
    spec: &QuantSpec,
    out: &mut [u8],
    opts: &EncodeOpts,
) -> Result<usize, TensoError> {
    let l = quant_layout(spec, opts)?;
    let ndim = spec.shape.len();
    let num_scales = spec.scales.len() / 4;
    let footer_len = if opts.check_integrity { 8 } else { 0 };
    let total_len = l.header_len + l.padding_len + l.body_len + footer_len;
    if out.len() < total_len {
        return Err(TensoError::BufferTooSmall);
    }

    let mut flags: u16 = 0;
    if l.use_custom_align {
        flags |= FLAG_CUST_ALIGN;
    } else {
        flags |= FLAG_ALIGNED;
    }
    if opts.check_integrity {
        flags |= FLAG_INTEGRITY;
    }
    write_v4_header(out, flags, spec.dtype.code(), ndim as u8);

    let mut cursor = HEADER_BASE_V4;
    for &dim in spec.shape {
        out[cursor..cursor + 4].copy_from_slice(&dim.to_le_bytes());
        cursor += 4;
    }
    let meta_start = cursor;
    out[cursor] = spec.scheme;
    cursor += 1;
    out[cursor] = spec.axis;
    cursor += 1;
    out[cursor..cursor + 4].copy_from_slice(&spec.group_size.to_le_bytes());
    cursor += 4;
    out[cursor..cursor + 4].copy_from_slice(&(num_scales as u32).to_le_bytes());
    cursor += 4;
    out[cursor..cursor + spec.scales.len()].copy_from_slice(spec.scales);
    cursor += spec.scales.len();
    out[cursor..cursor + spec.zero_points.len()].copy_from_slice(spec.zero_points);
    cursor += spec.zero_points.len();
    debug_assert_eq!(cursor - meta_start, l.quant_meta_len);
    if l.use_custom_align {
        out[cursor] = opts.alignment.trailing_zeros() as u8;
        cursor += 1;
    }
    debug_assert_eq!(cursor, l.header_len);
    let body_start = l.header_len + l.padding_len;
    for b in &mut out[l.header_len..body_start] {
        *b = 0;
    }
    out[body_start..body_start + l.body_len].copy_from_slice(&spec.data[..l.body_len]);

    if opts.check_integrity {
        #[cfg(feature = "integrity")]
        {
            // Integrity covers quant metadata + packed body (matches core.py).
            let mut hasher = xxhash_rust::xxh3::Xxh3::new();
            hasher.update(&out[meta_start..meta_start + l.quant_meta_len]);
            hasher.update(&out[body_start..body_start + l.body_len]);
            let footer_start = body_start + l.body_len;
            out[footer_start..footer_start + 8].copy_from_slice(&hasher.digest().to_le_bytes());
        }
        #[cfg(not(feature = "integrity"))]
        {
            return Err(TensoError::IntegrityMismatch);
        }
    }
    Ok(total_len)
}

// =============================================================================
// String (StringTensor) encode API
// =============================================================================

/// Exact buffer size for a StringTensor packet (`count` strings, `payload_len`
/// total UTF-8 bytes).
pub fn string_required_size(
    count: u32,
    payload_len: usize,
    check_integrity: bool,
) -> Result<usize, TensoError> {
    let header_len = HEADER_BASE_V4 + 4;
    let padding_len = padding_for(header_len, ALIGNMENT);
    let offsets_len = (count as usize + 1)
        .checked_mul(8)
        .ok_or(TensoError::Malformed)?;
    let body_len = offsets_len
        .checked_add(payload_len)
        .ok_or(TensoError::Malformed)?;
    let footer_len = if check_integrity { 8 } else { 0 };
    Ok(header_len + padding_len + body_len + footer_len)
}

/// Encode a StringTensor into `out` (byte-identical to `ragged.py::StringTensor.dumps`;
/// 64-aligned, dtype Str, ndim 1). `offsets` is `count+1` u64s from 0, monotonic, ending at `payload.len()`.
pub fn encode_string_into(
    count: u32,
    offsets: &[u64],
    payload: &[u8],
    out: &mut [u8],
    check_integrity: bool,
) -> Result<usize, TensoError> {
    if offsets.len() != count as usize + 1 {
        return Err(TensoError::Malformed);
    }
    if offsets[0] != 0 {
        return Err(TensoError::Malformed);
    }
    let mut prev = 0u64;
    for &o in offsets {
        if o < prev {
            return Err(TensoError::Malformed);
        }
        prev = o;
    }
    if prev as usize != payload.len() {
        return Err(TensoError::Malformed);
    }

    let header_len = HEADER_BASE_V4 + 4;
    let padding_len = padding_for(header_len, ALIGNMENT);
    let offsets_len = (count as usize + 1) * 8;
    let body_len = offsets_len + payload.len();
    let footer_len = if check_integrity { 8 } else { 0 };
    let total_len = header_len + padding_len + body_len + footer_len;
    if out.len() < total_len {
        return Err(TensoError::BufferTooSmall);
    }

    let mut flags = FLAG_STRING | FLAG_ALIGNED;
    if check_integrity {
        flags |= FLAG_INTEGRITY;
    }
    write_v4_header(out, flags, DCODE_STR, 1);
    out[HEADER_BASE_V4..HEADER_BASE_V4 + 4].copy_from_slice(&count.to_le_bytes());

    let body_start = header_len + padding_len;
    for b in &mut out[header_len..body_start] {
        *b = 0;
    }
    let mut cursor = body_start;
    for &o in offsets {
        out[cursor..cursor + 8].copy_from_slice(&o.to_le_bytes());
        cursor += 8;
    }
    out[cursor..cursor + payload.len()].copy_from_slice(payload);

    if check_integrity {
        #[cfg(feature = "integrity")]
        {
            // Integrity covers body (offsets + payload); matches ragged.py.
            let hash = integrity_hash(&out[body_start..body_start + body_len]);
            let footer_start = body_start + body_len;
            out[footer_start..footer_start + 8].copy_from_slice(&hash.to_le_bytes());
        }
        #[cfg(not(feature = "integrity"))]
        {
            return Err(TensoError::IntegrityMismatch);
        }
    }
    Ok(total_len)
}

// =============================================================================
// Decode API
// =============================================================================

/// A zero-copy (or owned, when decompressed) view over a dense tensor body.
pub struct TensorView<'a> {
    pub dtype: Dtype,
    pub shape: Vec<u32>,
    pub body: &'a [u8],
    /// True if `body` borrows directly from the input packet (no copy).
    pub zero_copy: bool,
}

/// A view over a quantized tensor (scale/zero-point metadata + packed payload).
pub struct QuantView<'a> {
    pub dtype: Dtype,
    pub shape: Vec<u32>,
    pub scheme: u8,
    pub axis: i32,
    pub group_size: u32,
    pub scales: &'a [u8],
    pub zero_points: &'a [u8],
    pub packed: &'a [u8],
    pub zero_copy: bool,
}

/// The decoded form of a packet.
pub enum Decoded<'a> {
    Dense(TensorView<'a>),
    Bundle(Vec<(String, Decoded<'a>)>),
    Sparse {
        format: SparseFormat,
        shape: Vec<u32>,
        components: Vec<Decoded<'a>>,
    },
    Quantized(QuantView<'a>),
    String {
        shape: Vec<u32>,
        offsets: &'a [u8],
        payload: &'a [u8],
    },
    Ragged {
        shape: Vec<u32>,
        offsets: &'a [u8],
        payload: &'a [u8],
    },
    /// A GPU IPC reference (no inline body).
    IpcRef(IpcRef),
}

/// Sparse storage format discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    Coo,
    Csr,
    Csc,
}

impl SparseFormat {
    /// The single header flag bit identifying this sparse format on the wire.
    /// COO uses `FLAG_SPARSE_COO` (== Python's `FLAG_SPARSE`, value 8).
    pub fn flag(&self) -> u16 {
        match self {
            SparseFormat::Coo => FLAG_SPARSE_COO,
            SparseFormat::Csr => FLAG_SPARSE_CSR,
            SparseFormat::Csc => FLAG_SPARSE_CSC,
        }
    }
}

/// Decode a Tenso packet. Bodies are zero-copy borrows; compressed dense bodies
/// can't be borrowed so are rejected here (use the owning FFI/device decode path).
pub fn decode(bytes: &[u8]) -> Result<Decoded<'_>, TensoError> {
    decode_depth(bytes, 0)
}

/// Max bundle/sparse nesting depth (DoS guard against stack overflow; deeper => `Malformed`).
pub const MAX_DECODE_DEPTH: usize = 64;

fn decode_depth(bytes: &[u8], depth: usize) -> Result<Decoded<'_>, TensoError> {
    if depth > MAX_DECODE_DEPTH {
        return Err(TensoError::Malformed);
    }
    let hdr = parse_header(bytes)?;
    let flags = hdr.flags;
    let ndim = hdr.ndim;
    let base = hdr.base_size;
    let dtype_code = hdr.dtype_code;

    // --- GPU IpcRef packet: dedicated 96-byte body, no shape ---
    if flags & FLAG_GPU_IPC_REF != 0 {
        let ipc = parse_ipc_ref(bytes)?;
        return Ok(Decoded::IpcRef(ipc));
    }

    // --- Bundle: ndim is the entry count ---
    if flags & FLAG_BUNDLE != 0 {
        let count = ndim;
        let mut entries: Vec<(String, Decoded)> = Vec::with_capacity(count);
        let mut cursor = base;
        for _ in 0..count {
            let k_len = read_u32(bytes, cursor)? as usize;
            cursor += 4;
            let k_end = cursor.checked_add(k_len).ok_or(TensoError::Malformed)?;
            if k_end > bytes.len() {
                return Err(TensoError::TooShort);
            }
            let key =
                core::str::from_utf8(&bytes[cursor..k_end]).map_err(|_| TensoError::BadBundle)?;
            cursor = k_end;

            let v_len = read_u32(bytes, cursor)? as usize;
            cursor += 4;
            let v_end = cursor.checked_add(v_len).ok_or(TensoError::Malformed)?;
            if v_end > bytes.len() {
                return Err(TensoError::TooShort);
            }
            let val = decode_depth(&bytes[cursor..v_end], depth + 1)?;
            cursor = v_end;
            entries.push((String::from(key), val));
        }
        return Ok(Decoded::Bundle(entries));
    }

    // --- Sparse (COO / CSR / CSC): shape then 3 length-prefixed sub-packets ---
    if let Some(format) = sparse_format(flags) {
        let shape = read_shape(bytes, base, ndim)?;
        let shape_end = base + ndim * 4;
        let mut cursor = shape_end;
        let mut components: Vec<Decoded> = Vec::with_capacity(3);
        for _ in 0..3 {
            let sub_len = read_u32(bytes, cursor)? as usize;
            cursor += 4;
            let sub_end = cursor.checked_add(sub_len).ok_or(TensoError::Malformed)?;
            if sub_end > bytes.len() {
                return Err(TensoError::TooShort);
            }
            components.push(decode_depth(&bytes[cursor..sub_end], depth + 1)?);
            cursor = sub_end;
        }
        return Ok(Decoded::Sparse {
            format,
            shape,
            components,
        });
    }

    // --- String / Ragged packets ---
    if flags & FLAG_STRING != 0 {
        return decode_string(bytes, &hdr);
    }
    if flags & FLAG_RAGGED != 0 {
        // Python serializes RaggedArray as a bundle; support the direct
        // (shape+offsets+payload) framing here for completeness.
        return decode_ragged(bytes, &hdr);
    }

    // --- Quantized dtypes (16..=19) ---
    if (DCODE_QINT8..=DCODE_QUINT4).contains(&dtype_code) {
        return decode_quantized(bytes, &hdr);
    }

    // --- Dense ---
    decode_dense(bytes, &hdr)
}

#[inline]
fn sparse_format(flags: u16) -> Option<SparseFormat> {
    if flags & FLAG_SPARSE_COO != 0 {
        Some(SparseFormat::Coo)
    } else if flags & FLAG_SPARSE_CSR != 0 {
        Some(SparseFormat::Csr)
    } else if flags & FLAG_SPARSE_CSC != 0 {
        Some(SparseFormat::Csc)
    } else {
        None
    }
}

fn read_shape(bytes: &[u8], base: usize, ndim: usize) -> Result<Vec<u32>, TensoError> {
    if ndim > MAX_NDIM {
        return Err(TensoError::TooManyDims);
    }
    let shape_end = base
        .checked_add(ndim.checked_mul(4).ok_or(TensoError::Malformed)?)
        .ok_or(TensoError::Malformed)?;
    if bytes.len() < shape_end {
        return Err(TensoError::TooShort);
    }
    let mut shape = Vec::with_capacity(ndim);
    let mut cursor = base;
    for _ in 0..ndim {
        shape.push(read_u32(bytes, cursor)?);
        cursor += 4;
    }
    Ok(shape)
}

/// Resolve body alignment from flags + optional custom-align exponent byte after the shape.
/// Returns (alignment, header_len), where header_len includes the align byte if present.
fn resolve_alignment(
    bytes: &[u8],
    flags: u16,
    shape_end: usize,
) -> Result<(usize, usize), TensoError> {
    if flags & FLAG_CUST_ALIGN != 0 {
        if bytes.len() < shape_end + 1 {
            return Err(TensoError::TooShort);
        }
        let exponent = bytes[shape_end] as u32;
        if exponent >= usize::BITS {
            return Err(TensoError::Malformed);
        }
        let alignment = 1usize << exponent;
        Ok((alignment, shape_end + 1))
    } else if flags & FLAG_ALIGNED != 0 {
        Ok((ALIGNMENT, shape_end))
    } else {
        Ok((1, shape_end))
    }
}

fn decode_dense<'a>(bytes: &'a [u8], hdr: &Header) -> Result<Decoded<'a>, TensoError> {
    let flags = hdr.flags;
    let ndim = hdr.ndim;
    let base = hdr.base_size;
    let dtype = Dtype::from_code(hdr.dtype_code)?;
    let item = dtype
        .item_size()
        .ok_or(TensoError::BadDtype(hdr.dtype_code))?;

    let shape = read_shape(bytes, base, ndim)?;
    let shape_end = base + ndim * 4;
    let num_elements = shape_num_elements(&shape)?;
    if num_elements > MAX_ELEMENTS {
        return Err(TensoError::TooManyElements);
    }

    let (alignment, header_len) = resolve_alignment(bytes, flags, shape_end)?;
    let padding_len = padding_for(header_len, alignment);
    let body_start = header_len
        .checked_add(padding_len)
        .ok_or(TensoError::Malformed)?;
    if bytes.len() < body_start {
        return Err(TensoError::TooShort);
    }

    let footer_len = if flags & FLAG_INTEGRITY != 0 { 8 } else { 0 };
    let uncompressed_len = (num_elements as usize)
        .checked_mul(item)
        .ok_or(TensoError::TooManyElements)?;

    let body_len = if flags & FLAG_COMPRESSION != 0 {
        // compressed body runs to the footer
        let avail = bytes.len();
        if avail < body_start + footer_len {
            return Err(TensoError::TooShort);
        }
        avail - body_start - footer_len
    } else {
        uncompressed_len
    };

    let body_end = body_start
        .checked_add(body_len)
        .ok_or(TensoError::Malformed)?;
    let total_end = body_end
        .checked_add(footer_len)
        .ok_or(TensoError::Malformed)?;
    if bytes.len() < total_end {
        return Err(TensoError::TooShort);
    }

    let body = &bytes[body_start..body_end];

    // Verify integrity over the on-wire body (compressed or raw).
    if flags & FLAG_INTEGRITY != 0 {
        #[cfg(feature = "integrity")]
        {
            let expected = read_u64(bytes, body_end)?;
            if integrity_hash(body) != expected {
                return Err(TensoError::IntegrityMismatch);
            }
        }
        #[cfg(not(feature = "integrity"))]
        {
            return Err(TensoError::IntegrityMismatch);
        }
    }

    if flags & FLAG_COMPRESSION != 0 {
        // Can't borrow a compressed body zero-copy; reject so callers don't
        // mistake compressed bytes for tensor data (owning paths decompress).
        return Err(TensoError::Lz4(
            "compressed dense bodies require an owning decode path",
        ));
    }

    Ok(Decoded::Dense(TensorView {
        dtype,
        shape,
        body,
        zero_copy: true,
    }))
}

/// Owning dense decode (decompresses compressed bodies); the borrowing [`decode`] rejects those.
/// Returns `(dtype, shape, owned_body)`; integrity is verified before decompression.
#[cfg(feature = "compression")]
pub fn decode_dense_to_owned(bytes: &[u8]) -> Result<(Dtype, Vec<u32>, Vec<u8>), TensoError> {
    // Uncompressed dense reuses the borrowing decoder (also runs integrity); non-dense errors.
    match decode(bytes) {
        Ok(Decoded::Dense(v)) => return Ok((v.dtype, v.shape, v.body.to_vec())),
        Ok(_) => return Err(TensoError::Malformed),
        // Compressed dense: `decode` verified integrity, then rejected the borrow.
        Err(TensoError::Lz4(_)) => {}
        Err(e) => return Err(e),
    }

    let hdr = parse_header(bytes)?;
    let dtype = Dtype::from_code(hdr.dtype_code)?;
    let shape = read_shape(bytes, hdr.base_size, hdr.ndim)?;
    let shape_end = hdr.base_size + hdr.ndim * 4;
    let (alignment, header_len) = resolve_alignment(bytes, hdr.flags, shape_end)?;
    let padding_len = padding_for(header_len, alignment);
    let body_start = header_len
        .checked_add(padding_len)
        .ok_or(TensoError::Malformed)?;
    let footer_len = if hdr.flags & FLAG_INTEGRITY != 0 {
        8
    } else {
        0
    };
    if bytes.len() < body_start + footer_len {
        return Err(TensoError::TooShort);
    }
    // Compressed body: body_start to the footer (no length prefix; bounds checked above).
    let body = &bytes[body_start..bytes.len() - footer_len];
    let owned = lz4_decompress_frame(body)?;
    Ok((dtype, shape, owned))
}

fn decode_quantized<'a>(bytes: &'a [u8], hdr: &Header) -> Result<Decoded<'a>, TensoError> {
    let flags = hdr.flags;
    let ndim = hdr.ndim;
    let base = hdr.base_size;
    let dtype = Dtype::from_code(hdr.dtype_code)?;

    let shape = read_shape(bytes, base, ndim)?;
    let shape_end = base + ndim * 4;
    let num_elements = shape_num_elements(&shape)?;
    if num_elements > MAX_ELEMENTS {
        return Err(TensoError::TooManyElements);
    }

    // meta: scheme(1) + axis(1) + group_size(4) + num_scales(4) + scales(n*4) + zero_points(n*4)
    let meta_start = shape_end;
    let mut cursor = meta_start;
    if bytes.len() < cursor + 10 {
        return Err(TensoError::TooShort);
    }
    let scheme = bytes[cursor];
    cursor += 1;
    let axis_byte = bytes[cursor];
    cursor += 1;
    let group_size = read_u32(bytes, cursor)?;
    cursor += 4;
    let num_scales = read_u32(bytes, cursor)? as usize;
    cursor += 4;

    let sz_bytes = num_scales.checked_mul(4).ok_or(TensoError::Malformed)?;
    let scales_end = cursor.checked_add(sz_bytes).ok_or(TensoError::Malformed)?;
    if scales_end > bytes.len() {
        return Err(TensoError::TooShort);
    }
    let scales = &bytes[cursor..scales_end];
    cursor = scales_end;
    let zp_end = cursor.checked_add(sz_bytes).ok_or(TensoError::Malformed)?;
    if zp_end > bytes.len() {
        return Err(TensoError::TooShort);
    }
    let zero_points = &bytes[cursor..zp_end];
    cursor = zp_end;
    let meta_len = cursor - meta_start;

    let (alignment, header_len) = resolve_alignment(bytes, flags, cursor)?;
    let padding_len = padding_for(header_len, alignment);
    let body_start = header_len
        .checked_add(padding_len)
        .ok_or(TensoError::Malformed)?;

    // Packed body: 4-bit dtypes store 2 elements/byte.
    let body_len = if dtype.is_4bit() {
        (num_elements as usize).div_ceil(2)
    } else {
        num_elements as usize
    };
    let body_end = body_start
        .checked_add(body_len)
        .ok_or(TensoError::Malformed)?;
    let footer_len = if flags & FLAG_INTEGRITY != 0 { 8 } else { 0 };
    let total_end = body_end
        .checked_add(footer_len)
        .ok_or(TensoError::Malformed)?;
    if bytes.len() < total_end {
        return Err(TensoError::TooShort);
    }
    let packed = &bytes[body_start..body_end];

    // Integrity covers quant metadata + packed body (matches core.py).
    if flags & FLAG_INTEGRITY != 0 {
        #[cfg(feature = "integrity")]
        {
            let expected = read_u64(bytes, body_end)?;
            let mut hasher = xxhash_rust::xxh3::Xxh3::new();
            hasher.update(&bytes[meta_start..meta_start + meta_len]);
            hasher.update(packed);
            if hasher.digest() != expected {
                return Err(TensoError::IntegrityMismatch);
            }
        }
        #[cfg(not(feature = "integrity"))]
        {
            return Err(TensoError::IntegrityMismatch);
        }
    }

    Ok(Decoded::Quantized(QuantView {
        dtype,
        shape,
        scheme,
        // axis is a single unsigned wire byte (Python writes `axis & 0xFF`); zero-extend to i32.
        axis: axis_byte as i32,
        group_size,
        scales,
        zero_points,
        packed,
        zero_copy: true,
    }))
}

fn decode_string<'a>(bytes: &'a [u8], hdr: &Header) -> Result<Decoded<'a>, TensoError> {
    let (shape, offsets, payload) = decode_offset_payload(bytes, hdr)?;
    Ok(Decoded::String {
        shape,
        offsets,
        payload,
    })
}

fn decode_ragged<'a>(bytes: &'a [u8], hdr: &Header) -> Result<Decoded<'a>, TensoError> {
    let (shape, offsets, payload) = decode_offset_payload(bytes, hdr)?;
    Ok(Decoded::Ragged {
        shape,
        offsets,
        payload,
    })
}

/// Decoded offset+payload: `(offsets-as-u32-shape, offsets_bytes, payload_bytes)`.
type OffsetPayload<'a> = (Vec<u32>, &'a [u8], &'a [u8]);

/// Shared String/Ragged decode. Layout: header + shape[ndim] + (cust-align?) + padding
/// + offsets((count+1)*u64) + payload + (integrity footer?). `shape[0]` == count.
fn decode_offset_payload<'a>(
    bytes: &'a [u8],
    hdr: &Header,
) -> Result<OffsetPayload<'a>, TensoError> {
    let flags = hdr.flags;
    let ndim = hdr.ndim;
    let base = hdr.base_size;

    let shape = read_shape(bytes, base, ndim)?;
    let shape_end = base + ndim * 4;
    let count = if let Some(&first) = shape.first() {
        first as usize
    } else {
        0
    };

    let (alignment, header_len) = resolve_alignment(bytes, flags, shape_end)?;
    let padding_len = padding_for(header_len, alignment);
    let body_start = header_len
        .checked_add(padding_len)
        .ok_or(TensoError::Malformed)?;

    let offsets_len = (count + 1).checked_mul(8).ok_or(TensoError::Malformed)?;
    let offsets_end = body_start
        .checked_add(offsets_len)
        .ok_or(TensoError::Malformed)?;
    if bytes.len() < offsets_end {
        return Err(TensoError::TooShort);
    }
    let offsets = &bytes[body_start..offsets_end];

    // Offsets must start at 0 and be monotonic (mirrors ragged.py; guards untrusted input).
    // Last offset is the payload length.
    let mut prev = 0u64;
    for i in 0..=count {
        let o = read_u64(bytes, body_start + i * 8)?;
        if i == 0 && o != 0 {
            return Err(TensoError::Malformed);
        }
        if o < prev {
            return Err(TensoError::Malformed);
        }
        prev = o;
    }
    let total_payload = prev as usize;

    let payload_end = offsets_end
        .checked_add(total_payload)
        .ok_or(TensoError::Malformed)?;
    if bytes.len() < payload_end {
        return Err(TensoError::TooShort);
    }
    let payload = &bytes[offsets_end..payload_end];

    if flags & FLAG_INTEGRITY != 0 {
        #[cfg(feature = "integrity")]
        {
            if bytes.len() < payload_end + 8 {
                return Err(TensoError::TooShort);
            }
            let expected = read_u64(bytes, payload_end)?;
            // Integrity covers body = offsets + payload (matches ragged.py).
            if integrity_hash(&bytes[body_start..payload_end]) != expected {
                return Err(TensoError::IntegrityMismatch);
            }
        }
        #[cfg(not(feature = "integrity"))]
        {
            return Err(TensoError::IntegrityMismatch);
        }
    }

    Ok((shape, offsets, payload))
}

// =============================================================================
// IpcRef framing helpers
// =============================================================================

/// Encode an `IpcRef` into a fresh `IPC_REF_PACKET_LEN`-byte packet.
/// Layout: v4 header (FLAG_GPU_IPC_REF, offset-9 byte = IPC_REF_DISCRIMINATOR), then 96-byte
/// body: handle[64], byte_offset:u64 LE, nbytes:u64 LE, device_uuid[16].
pub fn write_ipc_ref(out: &mut [u8], ipc: &IpcRef) -> Result<usize, TensoError> {
    if out.len() < IPC_REF_PACKET_LEN {
        return Err(TensoError::BufferTooSmall);
    }
    // ndim 0 (no shape); GPU_IPC_REF only, never INTEGRITY
    write_v4_header(out, FLAG_GPU_IPC_REF, 0, 0);
    out[9] = IPC_REF_DISCRIMINATOR;

    let mut cursor = HEADER_BASE_V4;
    out[cursor..cursor + IPC_REF_HANDLE_LEN].copy_from_slice(&ipc.handle);
    cursor += IPC_REF_HANDLE_LEN;
    out[cursor..cursor + 8].copy_from_slice(&ipc.byte_offset.to_le_bytes());
    cursor += 8;
    out[cursor..cursor + 8].copy_from_slice(&ipc.nbytes.to_le_bytes());
    cursor += 8;
    out[cursor..cursor + IPC_REF_DEVICE_UUID_LEN].copy_from_slice(&ipc.device_uuid);
    cursor += IPC_REF_DEVICE_UUID_LEN;

    debug_assert_eq!(cursor, IPC_REF_PACKET_LEN);
    Ok(IPC_REF_PACKET_LEN)
}

/// Parse an IpcRef packet; validate discriminator, reject illegal flags (INTEGRITY) / inline body.
pub fn parse_ipc_ref(bytes: &[u8]) -> Result<IpcRef, TensoError> {
    let hdr = parse_header(bytes)?;
    if hdr.flags & FLAG_GPU_IPC_REF == 0 {
        return Err(TensoError::Malformed);
    }
    // IpcRef never carries an integrity footer or inline body
    if hdr.flags & FLAG_INTEGRITY != 0 {
        return Err(TensoError::Malformed);
    }
    if hdr.version != VERSION {
        return Err(TensoError::UnsupportedVersion(hdr.version));
    }
    // offset-9 byte is the discriminator; ndim must be 0
    if bytes.len() < HEADER_BASE_V4 || bytes[9] != IPC_REF_DISCRIMINATOR {
        return Err(TensoError::Malformed);
    }
    if hdr.ndim != 0 {
        return Err(TensoError::Malformed);
    }
    if bytes.len() < IPC_REF_PACKET_LEN {
        return Err(TensoError::TooShort);
    }

    let mut cursor = HEADER_BASE_V4;
    let mut handle = [0u8; IPC_REF_HANDLE_LEN];
    handle.copy_from_slice(&bytes[cursor..cursor + IPC_REF_HANDLE_LEN]);
    cursor += IPC_REF_HANDLE_LEN;
    let byte_offset = read_u64(bytes, cursor)?;
    cursor += 8;
    let nbytes = read_u64(bytes, cursor)?;
    cursor += 8;
    let mut device_uuid = [0u8; IPC_REF_DEVICE_UUID_LEN];
    device_uuid.copy_from_slice(&bytes[cursor..cursor + IPC_REF_DEVICE_UUID_LEN]);

    Ok(IpcRef {
        handle,
        byte_offset,
        nbytes,
        device_uuid,
    })
}

// =============================================================================
// Errors (REAL enum shape; variants per contract)
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensoError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u8),
    BadDtype(u8),
    TooManyDims,
    TooManyElements,
    IntegrityMismatch,
    BadBundle,
    /// LZ4 (de)compression error; static reason string keeps the enum no_std-clonable.
    Lz4(&'static str),
    /// Buffer supplied to an `*_into` function was too small.
    BufferTooSmall,
    /// Generic malformed-packet catch-all for the decode paths.
    Malformed,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- header round-trips ----

    #[test]
    fn header_v4_roundtrip() {
        let mut buf = [0u8; HEADER_BASE_V4];
        write_v4_header(&mut buf, 0xBEEF, DCODE_I8, 3);
        let h = parse_header(&buf).unwrap();
        assert_eq!(h.version, 4);
        assert_eq!(h.flags, 0xBEEF);
        assert_eq!(h.dtype_code, DCODE_I8);
        assert_eq!(h.ndim, 3);
        assert_eq!(h.base_size, HEADER_BASE_V4);
    }

    #[test]
    fn header_v3_parse() {
        let bytes = [b'T', b'N', b'S', b'O', 3, 0b1010_0011, 7, 4];
        let h = parse_header(&bytes).unwrap();
        assert_eq!(h.version, 3);
        assert_eq!(h.flags, 0b1010_0011);
        assert_eq!(h.dtype_code, 7);
        assert_eq!(h.ndim, 4);
        assert_eq!(h.base_size, HEADER_BASE_V3);
    }

    #[test]
    fn header_errors() {
        assert_eq!(parse_header(&[]), Err(TensoError::TooShort));
        assert_eq!(
            parse_header(&[b'X', b'X', b'X', b'X', 4, 0, 0, 0]),
            Err(TensoError::BadMagic)
        );
        assert_eq!(
            parse_header(&[b'T', b'N', b'S', b'O', 9, 0, 0, 0]),
            Err(TensoError::UnsupportedVersion(9))
        );
        // v4 truncated (only 8 bytes present)
        assert_eq!(
            parse_header(&[b'T', b'N', b'S', b'O', 4, 0, 0, 0]),
            Err(TensoError::TooShort)
        );
    }

    // ---- dtype ----

    #[test]
    fn dtype_code_roundtrip() {
        for code in 1u8..=21 {
            let d = Dtype::from_code(code).unwrap();
            assert_eq!(d.code(), code);
        }
        assert_eq!(Dtype::from_code(0), Err(TensoError::BadDtype(0)));
        assert_eq!(Dtype::from_code(99), Err(TensoError::BadDtype(99)));
        assert_eq!(Dtype::QInt4.item_size(), None);
        assert_eq!(Dtype::Str.item_size(), None);
        assert_eq!(Dtype::F32.item_size(), Some(4));
    }

    // ---- dense encode/decode round-trips ----

    #[test]
    fn dense_required_size_default() {
        // f32 vec of 5 -> header(10)+shape(4)=14, pad to 64 -> body at 64, +20 = 84.
        let data = [0u8; 20];
        let shape = [5u32];
        let spec = ArraySpec {
            data: &data,
            dtype: Dtype::F32,
            shape: &shape,
        };
        let sz = dense_required_size(&spec, &EncodeOpts::default()).unwrap();
        assert_eq!(sz, 84);
    }

    #[test]
    fn dense_roundtrip_f32() {
        let vals: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut data = Vec::new();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let shape = [5u32];
        let spec = ArraySpec {
            data: &data,
            dtype: Dtype::F32,
            shape: &shape,
        };
        let opts = EncodeOpts::default();
        let sz = dense_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; sz];
        let n = encode_dense_into(&spec, &mut out, &opts).unwrap();
        assert_eq!(n, sz);

        match decode(&out).unwrap() {
            Decoded::Dense(v) => {
                assert_eq!(v.dtype, Dtype::F32);
                assert_eq!(v.shape, vec![5u32]);
                assert!(v.zero_copy);
                assert_eq!(v.body, &data[..]);
            }
            _ => panic!("expected dense"),
        }
    }

    #[test]
    #[cfg(feature = "integrity")]
    fn dense_roundtrip_integrity() {
        let mut data = Vec::new();
        for i in 0i32..8 {
            data.extend_from_slice(&i.to_le_bytes());
        }
        let shape = [8u32];
        let spec = ArraySpec {
            data: &data,
            dtype: Dtype::I32,
            shape: &shape,
        };
        let opts = EncodeOpts {
            check_integrity: true,
            ..Default::default()
        };
        let sz = dense_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; sz];
        encode_dense_into(&spec, &mut out, &opts).unwrap();

        match decode(&out).unwrap() {
            Decoded::Dense(v) => assert_eq!(v.body, &data[..]),
            _ => panic!("expected dense"),
        }
        // Corrupt the body -> integrity mismatch.
        let body_start = 64;
        out[body_start] ^= 0xFF;
        match decode(&out) {
            Err(TensoError::IntegrityMismatch) => {}
            other => panic!("expected IntegrityMismatch, got {:?}", other.is_ok()),
        }
    }

    #[test]
    fn dense_buffer_too_small() {
        let data = [0u8; 20];
        let shape = [5u32];
        let spec = ArraySpec {
            data: &data,
            dtype: Dtype::F32,
            shape: &shape,
        };
        let mut out = [0u8; 10];
        assert_eq!(
            encode_dense_into(&spec, &mut out, &EncodeOpts::default()),
            Err(TensoError::BufferTooSmall)
        );
    }

    #[test]
    fn dense_too_many_dims() {
        let shape = [1u32; MAX_NDIM + 1];
        let data = [0u8; 4];
        let spec = ArraySpec {
            data: &data,
            dtype: Dtype::U8,
            shape: &shape,
        };
        assert_eq!(
            dense_required_size(&spec, &EncodeOpts::default()),
            Err(TensoError::TooManyDims)
        );
    }

    // ---- ipc ref ----

    #[test]
    fn ipc_ref_roundtrip() {
        let ipc = IpcRef {
            handle: [7u8; 64],
            byte_offset: 0x1122_3344_5566_7788,
            nbytes: 4096,
            device_uuid: [0xAB; 16],
        };
        let mut out = [0u8; IPC_REF_PACKET_LEN];
        let n = write_ipc_ref(&mut out, &ipc).unwrap();
        assert_eq!(n, IPC_REF_PACKET_LEN);
        // header sanity
        assert_eq!(&out[0..4], b"TNSO");
        assert_eq!(out[4], VERSION);
        assert_eq!(u16::from_le_bytes([out[5], out[6]]), FLAG_GPU_IPC_REF);
        assert_eq!(out[9], IPC_REF_DISCRIMINATOR);

        let parsed = parse_ipc_ref(&out).unwrap();
        assert_eq!(parsed, ipc);

        match decode(&out).unwrap() {
            Decoded::IpcRef(p) => assert_eq!(p, ipc),
            _ => panic!("expected IpcRef"),
        }
    }

    #[test]
    fn ipc_ref_rejects_integrity_flag() {
        let ipc = IpcRef {
            handle: [0; 64],
            byte_offset: 0,
            nbytes: 0,
            device_uuid: [0; 16],
        };
        let mut out = [0u8; IPC_REF_PACKET_LEN];
        write_ipc_ref(&mut out, &ipc).unwrap();
        // Inject the (illegal) INTEGRITY flag.
        let flags = FLAG_GPU_IPC_REF | FLAG_INTEGRITY;
        out[5..7].copy_from_slice(&flags.to_le_bytes());
        assert_eq!(parse_ipc_ref(&out), Err(TensoError::Malformed));
    }

    #[test]
    fn ipc_ref_rejects_bad_discriminator() {
        let ipc = IpcRef {
            handle: [0; 64],
            byte_offset: 0,
            nbytes: 0,
            device_uuid: [0; 16],
        };
        let mut out = [0u8; IPC_REF_PACKET_LEN];
        write_ipc_ref(&mut out, &ipc).unwrap();
        out[9] = 0; // wrong discriminator
        assert_eq!(parse_ipc_ref(&out), Err(TensoError::Malformed));
    }

    // ---- bundle ----

    #[test]
    fn bundle_roundtrip() {
        // Build "a": f32[2]=[1,2], "b": i64[3]=[10,20,30] using the encoder.
        let a_data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let a_shape = [2u32];
        let a_spec = ArraySpec {
            data: &a_data,
            dtype: Dtype::F32,
            shape: &a_shape,
        };
        let a_sz = dense_required_size(&a_spec, &EncodeOpts::default()).unwrap();
        let mut a_pkt = vec![0u8; a_sz];
        encode_dense_into(&a_spec, &mut a_pkt, &EncodeOpts::default()).unwrap();

        let b_data: Vec<u8> = [10i64, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b_shape = [3u32];
        let b_spec = ArraySpec {
            data: &b_data,
            dtype: Dtype::I64,
            shape: &b_shape,
        };
        let b_sz = dense_required_size(&b_spec, &EncodeOpts::default()).unwrap();
        let mut b_pkt = vec![0u8; b_sz];
        encode_dense_into(&b_spec, &mut b_pkt, &EncodeOpts::default()).unwrap();

        // Assemble a bundle packet manually to exercise the decoder.
        let mut pkt = vec![0u8; HEADER_BASE_V4];
        write_v4_header(&mut pkt, FLAG_BUNDLE, 0, 2);
        for (key, body) in [("a", &a_pkt), ("b", &b_pkt)] {
            pkt.extend_from_slice(&(key.len() as u32).to_le_bytes());
            pkt.extend_from_slice(key.as_bytes());
            pkt.extend_from_slice(&(body.len() as u32).to_le_bytes());
            pkt.extend_from_slice(body);
        }

        match decode(&pkt).unwrap() {
            Decoded::Bundle(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].0, "a");
                assert_eq!(entries[1].0, "b");
                match &entries[1].1 {
                    Decoded::Dense(v) => {
                        assert_eq!(v.dtype, Dtype::I64);
                        assert_eq!(v.shape, vec![3u32]);
                    }
                    _ => panic!("expected dense in bundle"),
                }
            }
            _ => panic!("expected bundle"),
        }
    }

    // ---- bundle encode round-trips ----

    #[test]
    fn encode_bundle_roundtrip() {
        // Build "a": f32[2]=[1,2], "b": i64[3]=[10,20,30] as dense sub-packets.
        let a_data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let a_shape = [2u32];
        let a_spec = ArraySpec {
            data: &a_data,
            dtype: Dtype::F32,
            shape: &a_shape,
        };
        let a_sz = dense_required_size(&a_spec, &EncodeOpts::default()).unwrap();
        let mut a_pkt = vec![0u8; a_sz];
        encode_dense_into(&a_spec, &mut a_pkt, &EncodeOpts::default()).unwrap();

        let b_data: Vec<u8> = [10i64, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b_shape = [3u32];
        let b_spec = ArraySpec {
            data: &b_data,
            dtype: Dtype::I64,
            shape: &b_shape,
        };
        let b_sz = dense_required_size(&b_spec, &EncodeOpts::default()).unwrap();
        let mut b_pkt = vec![0u8; b_sz];
        encode_dense_into(&b_spec, &mut b_pkt, &EncodeOpts::default()).unwrap();

        let entries: [(&str, &[u8]); 2] = [("a", &a_pkt), ("b", &b_pkt)];
        let sz = bundle_required_size(&entries).unwrap();
        let mut out = vec![0u8; sz];
        let n = encode_bundle_into(&entries, &mut out).unwrap();
        assert_eq!(n, sz);
        // ndim byte must equal the entry count.
        let hdr = parse_header(&out).unwrap();
        assert_eq!(hdr.flags, FLAG_BUNDLE);
        assert_eq!(hdr.ndim, 2);

        match decode(&out).unwrap() {
            Decoded::Bundle(decoded) => {
                assert_eq!(decoded.len(), 2);
                assert_eq!(decoded[0].0, "a");
                assert_eq!(decoded[1].0, "b");
                match &decoded[0].1 {
                    Decoded::Dense(v) => {
                        assert_eq!(v.dtype, Dtype::F32);
                        assert_eq!(v.shape, vec![2u32]);
                        assert_eq!(v.body, &a_data[..]);
                    }
                    _ => panic!("expected dense in bundle"),
                }
                match &decoded[1].1 {
                    Decoded::Dense(v) => {
                        assert_eq!(v.dtype, Dtype::I64);
                        assert_eq!(v.shape, vec![3u32]);
                        assert_eq!(v.body, &b_data[..]);
                    }
                    _ => panic!("expected dense in bundle"),
                }
            }
            _ => panic!("expected bundle"),
        }
    }

    #[test]
    fn encode_bundle_empty() {
        let entries: [(&str, &[u8]); 0] = [];
        let sz = bundle_required_size(&entries).unwrap();
        assert_eq!(sz, HEADER_BASE_V4);
        let mut out = vec![0u8; sz];
        encode_bundle_into(&entries, &mut out).unwrap();
        match decode(&out).unwrap() {
            Decoded::Bundle(d) => assert!(d.is_empty()),
            _ => panic!("expected bundle"),
        }
    }

    #[test]
    fn encode_bundle_too_many_entries() {
        // 256 entries: count does not fit in one ndim byte -> error.
        let body: Vec<u8> = Vec::new();
        let key = String::from("k");
        let entries: Vec<(&str, &[u8])> = (0..256).map(|_| (key.as_str(), &body[..])).collect();
        assert_eq!(bundle_required_size(&entries), Err(TensoError::BadBundle));
        let mut out = vec![0u8; 4096];
        assert_eq!(
            encode_bundle_into(&entries, &mut out),
            Err(TensoError::BadBundle)
        );

        // Exactly 255 is allowed.
        let entries255: Vec<(&str, &[u8])> = (0..255).map(|_| (key.as_str(), &body[..])).collect();
        assert!(bundle_required_size(&entries255).is_ok());
    }

    #[test]
    fn encode_bundle_buffer_too_small() {
        let entries: [(&str, &[u8]); 1] = [("a", &[1u8, 2, 3])];
        let mut out = [0u8; 4];
        assert_eq!(
            encode_bundle_into(&entries, &mut out),
            Err(TensoError::BufferTooSmall)
        );
    }

    #[test]
    fn decode_rejects_too_deeply_nested_bundle() {
        // Wrap a dense packet in 1-entry bundles past MAX_DECODE_DEPTH; decode() must
        // reject with Malformed instead of overflowing the stack (DoS guard).
        let data = 1.0f32.to_le_bytes().to_vec();
        let shape = [1u32];
        let spec = ArraySpec {
            data: &data,
            dtype: Dtype::F32,
            shape: &shape,
        };
        let opts = EncodeOpts::default();
        let mut pkt = vec![0u8; dense_required_size(&spec, &opts).unwrap()];
        encode_dense_into(&spec, &mut pkt, &opts).unwrap();

        let wrap = |inner: &[u8]| -> Vec<u8> {
            let entries: [(&str, &[u8]); 1] = [("a", inner)];
            let mut out = vec![0u8; bundle_required_size(&entries).unwrap()];
            encode_bundle_into(&entries, &mut out).unwrap();
            out
        };

        let mut deep = pkt.clone();
        for _ in 0..(MAX_DECODE_DEPTH + 2) {
            deep = wrap(&deep);
        }
        assert!(matches!(decode(&deep), Err(TensoError::Malformed)));

        // Moderate nesting (well within the cap) still decodes successfully.
        let mut shallow = pkt;
        for _ in 0..8 {
            shallow = wrap(&shallow);
        }
        assert!(decode(&shallow).is_ok());
    }

    #[test]
    fn decode_rejects_oversized_element_count() {
        // Dense f32 shape=[u32::MAX] exceeds MAX_ELEMENTS; decode() must reject (no huge alloc).
        let mut pkt = vec![0u8; HEADER_BASE_V4 + 4];
        write_v4_header(&mut pkt, 0, Dtype::F32.code(), 1);
        pkt[HEADER_BASE_V4..HEADER_BASE_V4 + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&pkt).is_err());
    }

    // ---- sparse encode round-trips ----

    fn make_dense(data: Vec<u8>, dtype: Dtype, shape: Vec<u32>) -> (Vec<u8>, Vec<u32>, Dtype) {
        (data, shape, dtype)
    }

    #[test]
    fn encode_sparse_coo_roundtrip() {
        // COO over a 4x4 matrix: data f32[3], row i32[3], col i32[3].
        let shape = [4u32, 4u32];
        let (d_data, d_shape, d_dt) = make_dense(
            [1.5f32, 2.5, 3.5]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            Dtype::F32,
            vec![3u32],
        );
        let (r_data, r_shape, r_dt) = make_dense(
            [0i32, 1, 3].iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dtype::I32,
            vec![3u32],
        );
        let (c_data, c_shape, c_dt) = make_dense(
            [2i32, 0, 3].iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dtype::I32,
            vec![3u32],
        );
        let components = [
            ArraySpec {
                data: &d_data,
                dtype: d_dt,
                shape: &d_shape,
            },
            ArraySpec {
                data: &r_data,
                dtype: r_dt,
                shape: &r_shape,
            },
            ArraySpec {
                data: &c_data,
                dtype: c_dt,
                shape: &c_shape,
            },
        ];

        let sz = sparse_required_size(&shape, &components).unwrap();
        let mut out = vec![0u8; sz];
        let n = encode_sparse_into(SparseFormat::Coo, &shape, &components, &mut out).unwrap();
        assert_eq!(n, sz);

        let hdr = parse_header(&out).unwrap();
        assert_eq!(hdr.flags, FLAG_SPARSE_COO);
        assert_eq!(hdr.dtype_code, 0);
        assert_eq!(hdr.ndim, 2);

        match decode(&out).unwrap() {
            Decoded::Sparse {
                format,
                shape: got_shape,
                components: comps,
            } => {
                assert_eq!(format, SparseFormat::Coo);
                assert_eq!(got_shape, vec![4u32, 4u32]);
                assert_eq!(comps.len(), 3);
                match &comps[0] {
                    Decoded::Dense(v) => {
                        assert_eq!(v.dtype, Dtype::F32);
                        assert_eq!(v.shape, vec![3u32]);
                        assert_eq!(v.body, &d_data[..]);
                    }
                    _ => panic!("expected dense component"),
                }
                match &comps[1] {
                    Decoded::Dense(v) => {
                        assert_eq!(v.dtype, Dtype::I32);
                        assert_eq!(v.body, &r_data[..]);
                    }
                    _ => panic!("expected dense component"),
                }
                match &comps[2] {
                    Decoded::Dense(v) => {
                        assert_eq!(v.dtype, Dtype::I32);
                        assert_eq!(v.body, &c_data[..]);
                    }
                    _ => panic!("expected dense component"),
                }
            }
            _ => panic!("expected sparse"),
        }
    }

    #[test]
    fn encode_sparse_csr_csc_formats() {
        // Minimal CSR/CSC: data f32[1], indices i32[1], indptr i32[3].
        let shape = [2u32, 2u32];
        let d_data: Vec<u8> = [9.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let i_data: Vec<u8> = [0i32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let p_data: Vec<u8> = [0i32, 1, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_shape = [1u32];
        let i_shape = [1u32];
        let p_shape = [3u32];
        let components = [
            ArraySpec {
                data: &d_data,
                dtype: Dtype::F32,
                shape: &d_shape,
            },
            ArraySpec {
                data: &i_data,
                dtype: Dtype::I32,
                shape: &i_shape,
            },
            ArraySpec {
                data: &p_data,
                dtype: Dtype::I32,
                shape: &p_shape,
            },
        ];

        for (fmt, flag) in [
            (SparseFormat::Csr, FLAG_SPARSE_CSR),
            (SparseFormat::Csc, FLAG_SPARSE_CSC),
        ] {
            let sz = sparse_required_size(&shape, &components).unwrap();
            let mut out = vec![0u8; sz];
            encode_sparse_into(fmt, &shape, &components, &mut out).unwrap();
            assert_eq!(parse_header(&out).unwrap().flags, flag);
            match decode(&out).unwrap() {
                Decoded::Sparse {
                    format,
                    shape: got_shape,
                    components: comps,
                } => {
                    assert_eq!(format, fmt);
                    assert_eq!(got_shape, vec![2u32, 2u32]);
                    assert_eq!(comps.len(), 3);
                    match &comps[2] {
                        Decoded::Dense(v) => {
                            assert_eq!(v.shape, vec![3u32]);
                            assert_eq!(v.body, &p_data[..]);
                        }
                        _ => panic!("expected dense indptr"),
                    }
                }
                _ => panic!("expected sparse"),
            }
        }
    }

    #[test]
    fn encode_sparse_requires_three_components() {
        let shape = [4u32];
        let data: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let cshape = [1u32];
        let two = [
            ArraySpec {
                data: &data,
                dtype: Dtype::F32,
                shape: &cshape,
            },
            ArraySpec {
                data: &data,
                dtype: Dtype::F32,
                shape: &cshape,
            },
        ];
        assert_eq!(
            sparse_required_size(&shape, &two),
            Err(TensoError::Malformed)
        );
        let mut out = vec![0u8; 4096];
        assert_eq!(
            encode_sparse_into(SparseFormat::Coo, &shape, &two, &mut out),
            Err(TensoError::Malformed)
        );
    }

    // ---- string ----

    #[test]
    fn string_decode_layout() {
        // ["hi","","world","ñ"] -> count=4, offsets [0,2,2,7,9], payload "hiworldñ".
        let strings: [&[u8]; 4] = [b"hi", b"", b"world", "ñ".as_bytes()];
        let mut payload = Vec::new();
        let mut offsets = vec![0u64];
        for s in strings {
            payload.extend_from_slice(s);
            offsets.push(payload.len() as u64);
        }
        let count = 4u32;
        let header_len = HEADER_BASE_V4 + 4;
        let pad = padding_for(header_len, ALIGNMENT);
        let body_start = header_len + pad;
        let mut pkt = vec![0u8; body_start];
        write_v4_header(&mut pkt, FLAG_STRING | FLAG_ALIGNED, DCODE_STR, 1);
        pkt[HEADER_BASE_V4..HEADER_BASE_V4 + 4].copy_from_slice(&count.to_le_bytes());
        for o in &offsets {
            pkt.extend_from_slice(&o.to_le_bytes());
        }
        pkt.extend_from_slice(&payload);

        match decode(&pkt).unwrap() {
            Decoded::String {
                shape,
                offsets: off,
                payload: pl,
            } => {
                assert_eq!(shape, vec![4u32]);
                assert_eq!(off.len(), (4 + 1) * 8);
                assert_eq!(pl, &b"hiworld\xc3\xb1"[..]);
            }
            _ => panic!("expected string"),
        }
    }

    // ---- quantized ----

    #[test]
    fn quantized_int8_per_tensor_roundtrip() {
        // shape [4], per_tensor, 1 scale + 1 zp, body = 4 int8 bytes.
        let shape = [4u32];
        let ndim = 1usize;
        let scale: f32 = 0.5;
        let zp: f32 = 1.0;
        let body: [u8; 4] = [10, 20, 30, 40];

        // meta = scheme1 axis1 gs4 num4 + 1*4 scales + 1*4 zp = 18
        let header_len_base = HEADER_BASE_V4 + ndim * 4; // shape
        let mut pkt = vec![0u8; header_len_base];
        write_v4_header(&mut pkt, FLAG_ALIGNED, DCODE_QINT8, ndim as u8);
        pkt[HEADER_BASE_V4..HEADER_BASE_V4 + 4].copy_from_slice(&shape[0].to_le_bytes());
        // meta
        pkt.push(QUANT_PER_TENSOR);
        pkt.push(0); // axis
        pkt.extend_from_slice(&0u32.to_le_bytes()); // group_size
        pkt.extend_from_slice(&1u32.to_le_bytes()); // num_scales
        pkt.extend_from_slice(&scale.to_le_bytes());
        pkt.extend_from_slice(&zp.to_le_bytes());
        // align body to 64
        let cur = pkt.len();
        let pad = padding_for(cur, ALIGNMENT);
        pkt.resize(cur + pad, 0);
        pkt.extend_from_slice(&body);

        match decode(&pkt).unwrap() {
            Decoded::Quantized(q) => {
                assert_eq!(q.dtype, Dtype::QInt8);
                assert_eq!(q.shape, vec![4u32]);
                assert_eq!(q.scheme, QUANT_PER_TENSOR);
                assert_eq!(q.axis, 0);
                assert_eq!(q.scales.len(), 4);
                assert_eq!(q.zero_points.len(), 4);
                assert_eq!(q.packed, &body[..]);
            }
            _ => panic!("expected quantized"),
        }
    }

    #[test]
    fn quantized_int4_body_len() {
        // shape [5] 4-bit -> body = ceil(5/2) = 3 bytes.
        let shape = [5u32];
        let ndim = 1usize;
        let mut pkt = vec![0u8; HEADER_BASE_V4 + ndim * 4];
        write_v4_header(&mut pkt, FLAG_ALIGNED, DCODE_QINT4, ndim as u8);
        pkt[HEADER_BASE_V4..HEADER_BASE_V4 + 4].copy_from_slice(&shape[0].to_le_bytes());
        pkt.push(QUANT_PER_TENSOR);
        pkt.push(0);
        pkt.extend_from_slice(&0u32.to_le_bytes());
        pkt.extend_from_slice(&1u32.to_le_bytes());
        pkt.extend_from_slice(&1.0f32.to_le_bytes());
        pkt.extend_from_slice(&0.0f32.to_le_bytes());
        let cur = pkt.len();
        pkt.resize(cur + padding_for(cur, ALIGNMENT), 0);
        pkt.extend_from_slice(&[0xAB, 0xCD, 0x0E]); // 3 packed bytes

        match decode(&pkt).unwrap() {
            Decoded::Quantized(q) => {
                assert_eq!(q.dtype, Dtype::QInt4);
                assert_eq!(q.packed.len(), 3);
            }
            _ => panic!("expected quantized"),
        }
    }

    #[test]
    fn encode_quantized_roundtrip_per_tensor() {
        let scales = 0.5f32.to_le_bytes();
        let zps = 3.0f32.to_le_bytes();
        let data = [10u8, 20, 30, 40];
        let shape = [2u32, 2];
        let spec = QuantSpec {
            dtype: Dtype::QInt8,
            shape: &shape,
            scheme: QUANT_PER_TENSOR,
            axis: 0,
            group_size: 0,
            scales: &scales,
            zero_points: &zps,
            data: &data,
        };
        // check_integrity false so this runs without the `integrity` feature
        let opts = EncodeOpts {
            check_integrity: false,
            compress: false,
            alignment: ALIGNMENT,
        };
        let need = quantized_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; need];
        assert_eq!(encode_quantized_into(&spec, &mut out, &opts).unwrap(), need);
        match decode(&out).unwrap() {
            Decoded::Quantized(q) => {
                assert_eq!(q.dtype, Dtype::QInt8);
                assert_eq!(q.shape, vec![2u32, 2]);
                assert_eq!(q.scheme, QUANT_PER_TENSOR);
                assert_eq!(q.packed, &data[..]);
                assert_eq!(q.scales, &0.5f32.to_le_bytes()[..]);
                assert_eq!(q.zero_points, &3.0f32.to_le_bytes()[..]);
            }
            _ => panic!("expected quantized"),
        }
    }

    #[test]
    fn encode_quantized_int4_body_len() {
        // 5 elements 4-bit -> ceil(5/2) = 3 body bytes.
        let scales = 1.0f32.to_le_bytes();
        let zps = 0.0f32.to_le_bytes();
        let data = [0xABu8, 0xCD, 0x0E, 0x99];
        let shape = [5u32];
        let spec = QuantSpec {
            dtype: Dtype::QInt4,
            shape: &shape,
            scheme: QUANT_PER_TENSOR,
            axis: 0,
            group_size: 0,
            scales: &scales,
            zero_points: &zps,
            data: &data,
        };
        let opts = EncodeOpts::default();
        let need = quantized_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; need];
        encode_quantized_into(&spec, &mut out, &opts).unwrap();
        match decode(&out).unwrap() {
            Decoded::Quantized(q) => assert_eq!(q.packed.len(), 3),
            _ => panic!("expected quantized"),
        }
    }

    #[test]
    fn encode_string_roundtrip_and_matches_fixture() {
        // Same strings as the string_mixed_utf8.tenso fixture; encoder must be byte-identical.
        let parts: [&[u8]; 4] = [b"hi", b"", b"world", "\u{00f1}".as_bytes()];
        let mut payload: Vec<u8> = Vec::new();
        let mut offsets: Vec<u64> = vec![0];
        for p in parts {
            payload.extend_from_slice(p);
            offsets.push(payload.len() as u64);
        }
        let need = string_required_size(4, payload.len(), false).unwrap();
        let mut out = vec![0u8; need];
        encode_string_into(4, &offsets, &payload, &mut out, false).unwrap();
        match decode(&out).unwrap() {
            Decoded::String { shape, .. } => assert_eq!(shape, vec![4u32]),
            _ => panic!("expected string"),
        }
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/string_mixed_utf8.tenso");
        if let Ok(fixture) = std::fs::read(&fixture_path) {
            assert_eq!(
                out, fixture,
                "Rust string encoder must match the Python fixture byte-for-byte"
            );
        }
    }
}

// =============================================================================
// Conformance tests: frozen .tenso fixtures round-trip + byte-identical encode.
// Fixtures (tests/fixtures, root) may be absent in a packaged build; tests skip if missing.
// =============================================================================

#[cfg(all(test, feature = "std"))]
mod conformance {
    use super::*;
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        // CARGO_MANIFEST_DIR = <root>/crates/tenso
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); // crates
        p.pop(); // root
        p.push("tests");
        p.push("fixtures");
        p
    }

    fn read_fixture(name: &str) -> Option<Vec<u8>> {
        let path = fixtures_dir().join(name);
        std::fs::read(path).ok()
    }

    #[test]
    fn fixture_dense_f32_vec_roundtrip_and_byte_identical() {
        let Some(data) = read_fixture("dense_f32_vec.tenso") else {
            eprintln!("skip: fixture missing");
            return;
        };
        assert_eq!(&data[0..4], b"TNSO");

        // Decode and check values.
        let expected: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        match decode(&data).unwrap() {
            Decoded::Dense(v) => {
                assert_eq!(v.dtype, Dtype::F32);
                assert_eq!(v.shape, vec![5u32]);
                let mut got = Vec::new();
                for chunk in v.body.chunks_exact(4) {
                    got.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                assert_eq!(got, expected);
            }
            _ => panic!("expected dense"),
        }

        // Re-encode from raw bytes and assert byte-for-byte equality.
        let mut raw = Vec::new();
        for v in expected {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let shape = [5u32];
        let spec = ArraySpec {
            data: &raw,
            dtype: Dtype::F32,
            shape: &shape,
        };
        let opts = EncodeOpts::default();
        let sz = dense_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; sz];
        encode_dense_into(&spec, &mut out, &opts).unwrap();
        assert_eq!(
            out, data,
            "core encoder must reproduce dense_f32_vec byte-for-byte"
        );
    }

    #[test]
    fn fixture_dense_f64_mat_byte_identical() {
        let Some(data) = read_fixture("dense_f64_mat.tenso") else {
            eprintln!("skip: fixture missing");
            return;
        };
        // arange(12) f64 reshape (3,4)
        let mut raw = Vec::new();
        for i in 0u64..12 {
            raw.extend_from_slice(&(i as f64).to_le_bytes());
        }
        let shape = [3u32, 4u32];
        let spec = ArraySpec {
            data: &raw,
            dtype: Dtype::F64,
            shape: &shape,
        };
        let opts = EncodeOpts::default();
        let sz = dense_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; sz];
        encode_dense_into(&spec, &mut out, &opts).unwrap();
        assert_eq!(out, data);

        match decode(&data).unwrap() {
            Decoded::Dense(v) => {
                assert_eq!(v.dtype, Dtype::F64);
                assert_eq!(v.shape, vec![3u32, 4u32]);
            }
            _ => panic!("expected dense"),
        }
    }

    #[test]
    #[cfg(feature = "integrity")]
    fn fixture_dense_i32_integrity_byte_identical() {
        let Some(data) = read_fixture("dense_i32_integrity.tenso") else {
            eprintln!("skip: fixture missing");
            return;
        };
        let mut raw = Vec::new();
        for i in 0i32..8 {
            raw.extend_from_slice(&i.to_le_bytes());
        }
        let shape = [8u32];
        let spec = ArraySpec {
            data: &raw,
            dtype: Dtype::I32,
            shape: &shape,
        };
        let opts = EncodeOpts {
            check_integrity: true,
            ..Default::default()
        };
        let sz = dense_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; sz];
        encode_dense_into(&spec, &mut out, &opts).unwrap();
        assert_eq!(
            out, data,
            "core encoder must reproduce dense_i32_integrity byte-for-byte"
        );

        // Decode (verifies the embedded XXH3 footer).
        match decode(&data).unwrap() {
            Decoded::Dense(v) => assert_eq!(v.shape, vec![8u32]),
            _ => panic!("expected dense"),
        }
    }

    #[test]
    fn fixture_dense_u8_3d_zeros_byte_identical() {
        let Some(data) = read_fixture("dense_u8_3d_zeros.tenso") else {
            eprintln!("skip: fixture missing");
            return;
        };
        let raw = vec![0u8; 4 * 4 * 4];
        let shape = [4u32, 4u32, 4u32];
        let spec = ArraySpec {
            data: &raw,
            dtype: Dtype::U8,
            shape: &shape,
        };
        let opts = EncodeOpts::default();
        let sz = dense_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; sz];
        encode_dense_into(&spec, &mut out, &opts).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn fixture_dense_bf16_vec_byte_identical() {
        let Some(data) = read_fixture("dense_bf16_vec.tenso") else {
            eprintln!("skip: fixture missing");
            return;
        };
        // Read the body bytes from the fixture (no bf16 crate dep) and round-trip through the encoder.
        let hdr = parse_header(&data).unwrap();
        assert_eq!(hdr.dtype_code, DCODE_BF16);
        // header(10)+shape(4)=14, pad to 64 -> body at 64, 4*2=8 bytes.
        let body = &data[64..64 + 8];
        let shape = [4u32];
        let spec = ArraySpec {
            data: body,
            dtype: Dtype::BF16,
            shape: &shape,
        };
        let opts = EncodeOpts::default();
        let sz = dense_required_size(&spec, &opts).unwrap();
        let mut out = vec![0u8; sz];
        encode_dense_into(&spec, &mut out, &opts).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn fixture_bundle_mixed_roundtrip() {
        let Some(data) = read_fixture("bundle_mixed.tenso") else {
            eprintln!("skip: fixture missing");
            return;
        };
        match decode(&data).unwrap() {
            Decoded::Bundle(entries) => {
                assert_eq!(entries.len(), 2);
                let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
                assert!(keys.contains(&"a"));
                assert!(keys.contains(&"b"));
                for (k, v) in &entries {
                    match (k.as_str(), v) {
                        ("a", Decoded::Dense(t)) => {
                            assert_eq!(t.dtype, Dtype::F32);
                            assert_eq!(t.shape, vec![2u32]);
                        }
                        ("b", Decoded::Dense(t)) => {
                            assert_eq!(t.dtype, Dtype::I64);
                            assert_eq!(t.shape, vec![3u32]);
                        }
                        _ => panic!("unexpected bundle entry"),
                    }
                }
            }
            _ => panic!("expected bundle"),
        }
    }

    #[test]
    fn fixture_string_mixed_utf8_roundtrip() {
        let Some(data) = read_fixture("string_mixed_utf8.tenso") else {
            eprintln!("skip: fixture missing");
            return;
        };
        match decode(&data).unwrap() {
            Decoded::String {
                shape,
                offsets,
                payload,
            } => {
                assert_eq!(shape, vec![4u32]);
                assert_eq!(offsets.len(), (4 + 1) * 8);
                // Reconstruct the strings from offsets+payload.
                let mut strs: Vec<&str> = Vec::new();
                for i in 0..4 {
                    let start = read_u64(offsets, i * 8).unwrap() as usize;
                    let end = read_u64(offsets, (i + 1) * 8).unwrap() as usize;
                    strs.push(core::str::from_utf8(&payload[start..end]).unwrap());
                }
                assert_eq!(strs, vec!["hi", "", "world", "ñ"]);
            }
            _ => panic!("expected string"),
        }
    }
}
