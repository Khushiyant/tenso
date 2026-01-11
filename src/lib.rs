use numpy::PyArrayDyn;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::convert::TryInto;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
#[cfg(unix)]
use std::os::unix::io::{RawFd, FromRawFd};
#[cfg(windows)]
use std::os::windows::io::{RawHandle, FromRawHandle};
use std::mem::ManuallyDrop;
use std::io::{self, Write, Read};
use std::fs::File;
use lz4_flex::{compress as lz4_compress, decompress as lz4_decompress};
use lz4_flex::frame::{FrameEncoder, FrameDecoder};

// -----------------------------------------------------------------------------
// DType Definition
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    Float32,
    Int32,
    Float64,
    Int64,
    Uint8,
    Uint16,
    Bool,
    Float16,
    // New types
    Int8,
    Int16,
    Uint32,
    Uint64,
    Complex64,
    Complex128,
}

impl DType {
    fn from_code(code: u8) -> PyResult<Self> {
        match code {
            1 => Ok(DType::Float32),
            2 => Ok(DType::Int32),
            3 => Ok(DType::Float64),
            4 => Ok(DType::Int64),
            5 => Ok(DType::Uint8),
            6 => Ok(DType::Uint16),
            7 => Ok(DType::Bool),
            8 => Ok(DType::Float16),
            9 => Ok(DType::Int8),
            10 => Ok(DType::Int16),
            11 => Ok(DType::Uint32),
            12 => Ok(DType::Uint64),
            13 => Ok(DType::Complex64),
            14 => Ok(DType::Complex128),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported or unknown dtype code: {}",
                code
            ))),
        }
    }

    fn code(&self) -> u8 {
        match self {
            DType::Float32 => 1,
            DType::Int32 => 2,
            DType::Float64 => 3,
            DType::Int64 => 4,
            DType::Uint8 => 5,
            DType::Uint16 => 6,
            DType::Bool => 7,
            DType::Float16 => 8,
            DType::Int8 => 9,
            DType::Int16 => 10,
            DType::Uint32 => 11,
            DType::Uint64 => 12,
            DType::Complex64 => 13,
            DType::Complex128 => 14,
        }
    }

    fn item_size(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Int32 => 4,
            DType::Float64 => 8,
            DType::Int64 => 8,
            DType::Uint8 => 1,
            DType::Uint16 => 2,
            DType::Bool => 1,
            DType::Float16 => 2,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Uint32 => 4,
            DType::Uint64 => 8,
            DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Int32 => "int32",
            DType::Float64 => "float64",
            DType::Int64 => "int64",
            DType::Uint8 => "uint8",
            DType::Uint16 => "uint16",
            DType::Bool => "bool",
            DType::Float16 => "float16",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Uint32 => "uint32",
            DType::Uint64 => "uint64",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        }
    }
}

// -----------------------------------------------------------------------------
// Helper Functions & Structs
// -----------------------------------------------------------------------------

fn compute_integrity_hash(data: &[u8]) -> u64 {
    const PARALLEL_THRESHOLD: usize = 1024 * 1024; // 1MB
    if data.len() < PARALLEL_THRESHOLD {
        return xxh3_64(data);
    }
    
    // Parallel Merkle-ish hash
    let hashes: Vec<u8> = data.par_chunks(PARALLEL_THRESHOLD)
        .map(|chunk| xxh3_64(chunk))
        .collect::<Vec<u64>>()
        .into_iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
        
    xxh3_64(&hashes)
}

struct WrapperWriter<W> {
    inner: W,
    hasher: Option<Xxh3>,
    written: usize,
}

impl<W: Write> WrapperWriter<W> {
    fn new(inner: W, track_hash: bool) -> Self {
        Self {
            inner,
            hasher: if track_hash { Some(Xxh3::new()) } else { None },
            written: 0,
        }
    }

    fn finish(self) -> (W, Option<u64>, usize) {
        let hash = self.hasher.map(|h| h.digest());
        (self.inner, hash, self.written)
    }
}

impl<W: Write> Write for WrapperWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(h) = &mut self.hasher {
            h.update(buf);
        }
        let n = self.inner.write(buf)?;
        self.written += n;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// -----------------------------------------------------------------------------
// Serialization Implementation (In-Memory)
// -----------------------------------------------------------------------------

macro_rules! serialize_impl {
    ($py:expr, $array:expr, $dtype_code:ty, $dtype_enum:expr, $check_integrity:expr, $compress:expr, $alignment:expr) => {{
        let array = $array.downcast::<PyArrayDyn<$dtype_code>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Type mismatch in serialization dispatch")
        })?;

        let ndim = array.ndim();
        let shape = array.shape();

        let use_custom_align = $alignment != 64;
        let mut header_len = 8 + (ndim * 4);
        if use_custom_align {
            header_len += 1;
        }
        
        let alignment = $alignment;
        let remainder = header_len % alignment;
        let padding_len = if remainder == 0 {
            0
        } else {
            alignment - remainder
        };

        let total_elements: usize = shape.iter().product();
        let dtype = $dtype_enum;
        let item_size = dtype.item_size();
        let uncompressed_len = total_elements * item_size;

        let slice = unsafe { array.as_slice() }.map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("Array must be C-Contiguous")
        })?; 
        let u8_slice = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * item_size)
        };

        let (compressed_data, body_len, flags_compression) = if $compress {
             let data = lz4_compress(u8_slice);
             let len = data.len();
             (Some(data), len, 4u8)
        } else {
             (None, uncompressed_len, 0u8)
        };
        
        let footer_len = if $check_integrity { 8 } else { 0 };
        let total_len = header_len + padding_len + body_len + footer_len;
        
        const PARALLEL_THRESHOLD: usize = 1024 * 1024;

        PyBytes::new_with($py, total_len, |bytes: &mut [u8]| {
            bytes[0..4].copy_from_slice(b"TNSO");
            bytes[4] = 3; 
            
            let mut flags = 0;
            if use_custom_align {
                flags |= 128;
            } else {
                flags |= 1;
            }
            if $check_integrity {
                flags |= 2;
            }
            flags |= flags_compression;
            bytes[5] = flags;

            bytes[6] = dtype.code();
            bytes[7] = ndim as u8;

            let mut cursor = 8;
            for &dim in shape {
                bytes[cursor..cursor + 4].copy_from_slice(&(dim as u32).to_le_bytes());
                cursor += 4;
            }

            if use_custom_align {
                bytes[cursor] = alignment.trailing_zeros() as u8;
            }

            let body_start = header_len + padding_len;
            let target_slice = &mut bytes[body_start..body_start + body_len];

            if let Some(ref c_data) = compressed_data {
                target_slice.copy_from_slice(c_data);
                if $check_integrity {
                     let hash = compute_integrity_hash(c_data);
                     let footer_start = body_start + body_len;
                     bytes[footer_start..footer_start + 8].copy_from_slice(&hash.to_le_bytes());
                }
            } else {
                if body_len >= PARALLEL_THRESHOLD {
                    target_slice.par_chunks_mut(128 * 1024)
                        .zip(u8_slice.par_chunks(128 * 1024))
                        .for_each(|(t, s)| t.copy_from_slice(s));
                } else {
                    target_slice.copy_from_slice(u8_slice);
                }

                if $check_integrity {
                    let hash = compute_integrity_hash(u8_slice);
                    let footer_start = body_start + body_len;
                    bytes[footer_start..footer_start + 8].copy_from_slice(&hash.to_le_bytes());
                }
            }
            Ok(())
        })
    }};
}

// -----------------------------------------------------------------------------
// DUMP TO FD Implementation (Optimized)
// -----------------------------------------------------------------------------

#[cfg(unix)]
type RawFileDescriptor = RawFd;
#[cfg(windows)]
type RawFileDescriptor = RawHandle;

fn generic_dump<T: numpy::Element>(
    py: Python,
    array: &PyAny,
    fd: RawFileDescriptor,
    check_integrity: bool,
    compress: bool,
    alignment: usize,
    dtype_enum: DType,
) -> PyResult<usize> {
    let array = array.downcast::<PyArrayDyn<T>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("Type mismatch in serialization dispatch")
    })?;

    let ndim = array.ndim();
    let shape = array.shape().to_vec();

    let slice = unsafe { array.as_slice() }.map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("Array must be C-Contiguous")
    })?;
    
    let item_size = dtype_enum.item_size();
    let total_bytes = slice.len() * item_size;
    
    // Safety: Extract raw pointer to pass to other thread.
    // The PyArray object is held by the 'array' reference which is tied to the GIL-holding thread stack,
    // which is blocked by allow_threads. So the data will not move or be GC'd.
    let data_ptr = slice.as_ptr() as *const u8 as usize;
    
    py.allow_threads(move || {
        let u8_slice = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, total_bytes) };
        #[cfg(unix)]
        let mut file = unsafe { ManuallyDrop::new(File::from_raw_fd(fd)) };
        #[cfg(windows)]
        let mut file = unsafe { ManuallyDrop::new(File::from_raw_handle(fd)) };

        let use_custom_align = alignment != 64;
        let mut header_len = 8 + (ndim * 4);
        if use_custom_align { header_len += 1; }
        
        let remainder = header_len % alignment;
        let padding_len = if remainder == 0 { 0 } else { alignment - remainder };
        
        let mut header_buf = Vec::with_capacity(header_len + padding_len);
        header_buf.extend_from_slice(b"TNSO");
        header_buf.push(3); // Version 3
        
        let mut flags = 0;
        if use_custom_align { flags |= 128; } else { flags |= 1; }
        if check_integrity { flags |= 2; }
        if compress { flags |= 4; }
        
        header_buf.push(flags);
        header_buf.push(dtype_enum.code());
        header_buf.push(ndim as u8);
        
        for &dim in &shape {
            header_buf.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        
        if use_custom_align {
             header_buf.push(alignment.trailing_zeros() as u8);
        }
        
        header_buf.resize(header_len + padding_len, 0);
        file.write_all(&header_buf).map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // WrapperWriter computes hash of the data being written to file.
        // If compress=True, we write compressed data to file, so we hash compressed data.
        let mut wrapper = WrapperWriter::new(&mut *file, check_integrity);
        
        if compress {
            // Streaming Compression
            let mut encoder = FrameEncoder::new(&mut wrapper);
            encoder.write_all(u8_slice).map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            encoder.finish().map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        } else {
            wrapper.write_all(u8_slice).map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        
        let (_, hash_opt, bytes_written) = wrapper.finish();

        if let Some(hash) = hash_opt {
             let hash: u64 = hash;
             file.write_all(&hash.to_le_bytes()).map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        
        let footer_len = if check_integrity { 8 } else { 0 };
        Ok(header_len + padding_len + bytes_written + footer_len)
    })
}

#[pyfunction]
#[pyo3(signature = (array, fd, check_integrity=false, compress=false, alignment=64))]
fn dump_to_fd_rs<'py>(
    py: Python<'py>,
    array: &'py PyAny,
    fd: RawFileDescriptor,
    check_integrity: bool,
    compress: bool,
    alignment: usize
) -> PyResult<usize> {
    if !alignment.is_power_of_two() {
        return Err(pyo3::exceptions::PyValueError::new_err("Alignment must be a power of two"));
    }
    
    let dtype = array.getattr("dtype")?;
    let name: String = dtype.getattr("name")?.extract()?;

    match name.as_str() {
        "float32" => generic_dump::<f32>(py, array, fd, check_integrity, compress, alignment, DType::Float32),
        "int32" => generic_dump::<i32>(py, array, fd, check_integrity, compress, alignment, DType::Int32),
        "float64" => generic_dump::<f64>(py, array, fd, check_integrity, compress, alignment, DType::Float64),
        "int64" => generic_dump::<i64>(py, array, fd, check_integrity, compress, alignment, DType::Int64),
        "uint8" => generic_dump::<u8>(py, array, fd, check_integrity, compress, alignment, DType::Uint8),
        "uint16" => generic_dump::<u16>(py, array, fd, check_integrity, compress, alignment, DType::Uint16),
        "bool" => generic_dump::<bool>(py, array, fd, check_integrity, compress, alignment, DType::Bool),
        "int8" => generic_dump::<i8>(py, array, fd, check_integrity, compress, alignment, DType::Int8),
        "int16" => generic_dump::<i16>(py, array, fd, check_integrity, compress, alignment, DType::Int16),
        "uint32" => generic_dump::<u32>(py, array, fd, check_integrity, compress, alignment, DType::Uint32),
        "uint64" => generic_dump::<u64>(py, array, fd, check_integrity, compress, alignment, DType::Uint64),
        "complex64" => generic_dump::<Complex32>(py, array, fd, check_integrity, compress, alignment, DType::Complex64),
        "complex128" => generic_dump::<Complex64>(py, array, fd, check_integrity, compress, alignment, DType::Complex128),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            name
        ))),
    }
}

// -----------------------------------------------------------------------------
// Main Dumps Function
// -----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (array, check_integrity=false, compress=false, alignment=64))]
fn dumps_rs<'py>(py: Python<'py>, array: &'py PyAny, check_integrity: bool, compress: bool, alignment: usize) -> PyResult<&'py PyBytes> {
    if !alignment.is_power_of_two() {
        return Err(pyo3::exceptions::PyValueError::new_err("Alignment must be a power of two"));
    }

    let dtype = array.getattr("dtype")?;
    let name: String = dtype.getattr("name")?.extract()?;

    match name.as_str() {
        "float32" => serialize_impl!(py, array, f32, DType::Float32, check_integrity, compress, alignment),
        "int32" => serialize_impl!(py, array, i32, DType::Int32, check_integrity, compress, alignment),
        "float64" => serialize_impl!(py, array, f64, DType::Float64, check_integrity, compress, alignment),
        "int64" => serialize_impl!(py, array, i64, DType::Int64, check_integrity, compress, alignment),
        "uint8" => serialize_impl!(py, array, u8, DType::Uint8, check_integrity, compress, alignment),
        "uint16" => serialize_impl!(py, array, u16, DType::Uint16, check_integrity, compress, alignment),
        "bool" => serialize_impl!(py, array, bool, DType::Bool, check_integrity, compress, alignment),
        "int8" => serialize_impl!(py, array, i8, DType::Int8, check_integrity, compress, alignment),
        "int16" => serialize_impl!(py, array, i16, DType::Int16, check_integrity, compress, alignment),
        "uint32" => serialize_impl!(py, array, u32, DType::Uint32, check_integrity, compress, alignment),
        "uint64" => serialize_impl!(py, array, u64, DType::Uint64, check_integrity, compress, alignment),
        "complex64" => serialize_impl!(py, array, Complex32, DType::Complex64, check_integrity, compress, alignment),
        "complex128" => serialize_impl!(py, array, Complex64, DType::Complex128, check_integrity, compress, alignment),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            name
        ))),
    }
}

// -----------------------------------------------------------------------------
// Metadata Extraction Function
// -----------------------------------------------------------------------------

#[pyfunction]
fn get_packet_info_rs<'py>(py: Python<'py>, data: &'py PyAny) -> PyResult<&'py PyDict> {
    let buffer: PyBuffer<u8> = PyBuffer::get(data)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, buffer.len_bytes())
    };

    if bytes.len() < 8 {
        return Err(pyo3::exceptions::PyValueError::new_err("Packet too short"));
    }

    if &bytes[0..4] != b"TNSO" {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid tenso packet"));
    }

    let ver = bytes[4];
    let flags = bytes[5];
    let dtype_code = bytes[6];
    let ndim = bytes[7] as usize;

    let shape_end = 8 + (ndim * 4);
    if bytes.len() < shape_end {
        return Err(pyo3::exceptions::PyValueError::new_err("Packet too short to contain shape"));
    }

    let mut shape = Vec::with_capacity(ndim);
    let mut cursor = 8;
    for _ in 0..ndim {
        let dim_bytes: [u8; 4] = bytes[cursor..cursor+4].try_into().map_err(|_| {
             pyo3::exceptions::PyValueError::new_err("Failed to read shape")
        })?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        shape.push(dim);
        cursor += 4;
    }

    let dict = PyDict::new(py);
    dict.set_item("version", ver)?;
    dict.set_item("flags", flags)?;
    dict.set_item("dtype_code", dtype_code)?;
    dict.set_item("ndim", ndim)?;
    
    let total_elements: usize = struct_prod(&shape);
    dict.set_item("total_elements", total_elements)?;
    dict.set_item("shape", PyTuple::new(py, shape))?;
    
    dict.set_item("aligned", (flags & 1) != 0)?;
    dict.set_item("integrity_protected", (flags & 2) != 0)?;

    Ok(dict)
}

fn struct_prod(shape: &[usize]) -> usize {
    shape.iter().product()
}

// -----------------------------------------------------------------------------
// Loads Function
// -----------------------------------------------------------------------------

fn deserialize_impl<'py>(
    py: Python<'py>, 
    root_data: &'py PyAny, 
    bytes: &[u8], 
    absolute_offset: usize
) -> PyResult<Option<PyObject>> {
    const MAX_NDIM: usize = 32;
    const MAX_ELEMENTS: usize = 1_000_000_000;
    const FLAG_BUNDLE: u8 = 16;

    if bytes.len() < 8 {
        return Err(pyo3::exceptions::PyValueError::new_err("Packet too short"));
    }

    if &bytes[0..4] != b"TNSO" {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid tenso packet"));
    }

    let ver = bytes[4];
    let flags = bytes[5];
    let dtype_code = bytes[6];
    let ndim = bytes[7] as usize;

    // Check for unsupported flags
    let supported_mask = 1 | 2 | 4 | 16 | 128;
    if (flags & !supported_mask) != 0 {
        return Ok(None);
    }

    // --- BUNDLE PATH ---
    if (flags & FLAG_BUNDLE) != 0 {
        let res = PyDict::new(py);
        let count = ndim; // For bundles, ndim holds the item count
        let mut cursor = 8;

        for _ in 0..count {
            // Read Key Length
            if cursor + 4 > bytes.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("Packet truncated (key len)"));
            }
            let k_len = u32::from_le_bytes(bytes[cursor..cursor+4].try_into().unwrap()) as usize;
            cursor += 4;

            // Read Key
            if cursor + k_len > bytes.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("Packet truncated (key)"));
            }
            let key_str = std::str::from_utf8(&bytes[cursor..cursor+k_len])
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid UTF-8 key"))?;
            cursor += k_len;

            // Read Value Packet Length
            if cursor + 4 > bytes.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("Packet truncated (val len)"));
            }
            let v_len = u32::from_le_bytes(bytes[cursor..cursor+4].try_into().unwrap()) as usize;
            cursor += 4;

            // Check boundaries
            if cursor + v_len > bytes.len() {
                 return Err(pyo3::exceptions::PyValueError::new_err("Packet truncated (value body)"));
            }

            // Recursive Call
            // Note: We pass the slice for the value packet
            let val_bytes = &bytes[cursor..cursor+v_len];
            let val_offset = absolute_offset + cursor;
            
            let val_obj = deserialize_impl(py, root_data, val_bytes, val_offset)?;
            
            match val_obj {
                Some(obj) => res.set_item(key_str, obj)?,
                None => return Ok(None), // Recursion failed (e.g. nested unsupported packet)
            }

            cursor += v_len;
        }
        return Ok(Some(res.into()));
    }

    // --- DENSE TENSOR PATH ---

    // DoS Protection
    if ndim > MAX_NDIM {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("Packet exceeds maximum dimensions ({} > {})", ndim, MAX_NDIM)));
    }

    let mut header_len = 8 + (ndim * 4);
    let mut alignment = 1;
    let use_custom_align = (flags & 128) != 0;

    if use_custom_align {
        if bytes.len() < header_len + 1 {
            return Err(pyo3::exceptions::PyValueError::new_err("Packet header incomplete (missing alignment byte)"));
        }
        let exponent = bytes[header_len];
        alignment = 1 << exponent;
        header_len += 1;
    } else if (flags & 1) != 0 {
        alignment = 64;
    }

    if bytes.len() < header_len {
        return Err(pyo3::exceptions::PyValueError::new_err("Packet header incomplete"));
    }

    // Parse shape
    let mut shape = Vec::with_capacity(ndim);
    let mut cursor = 8;
    for _ in 0..ndim {
        let dim_bytes: [u8; 4] = bytes[cursor..cursor+4].try_into().map_err(|_| {
             pyo3::exceptions::PyValueError::new_err("Failed to read shape")
        })?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        shape.push(dim);
        cursor += 4;
    }

    let remainder = header_len % alignment;
    let padding_len = if remainder == 0 { 0 } else { alignment - remainder };
    
    let body_start_rel = header_len + padding_len;

    let dtype = DType::from_code(dtype_code)?;
    let dtype_name = dtype.name();
    let item_size = dtype.item_size();
    
    let num_elements: usize = shape.iter().product();

    // DoS Protection
    if num_elements > MAX_ELEMENTS {
         return Err(pyo3::exceptions::PyValueError::new_err(format!("Packet exceeds maximum elements ({} > {})", num_elements, MAX_ELEMENTS)));
    }

    let uncompressed_len = num_elements * item_size;
    let footer_len = if (flags & 2) != 0 { 8 } else { 0 };
    
    let available = bytes.len();
    if available < body_start_rel {
         return Err(pyo3::exceptions::PyValueError::new_err("Packet too short for header/padding"));
    }
    
    // Determine body length (compressed or raw)
    let body_len = if (flags & 4) != 0 {
        // Compressed: Body is everything until footer
        if available < body_start_rel + footer_len {
             return Err(pyo3::exceptions::PyValueError::new_err("Packet too short for compressed body"));
        }
        available - body_start_rel - footer_len
    } else {
        // Raw: Body is exactly uncompressed_len
        uncompressed_len
    };

    if available < body_start_rel + body_len + footer_len {
         return Err(pyo3::exceptions::PyValueError::new_err("Packet too short for body"));
    }
    
    // Integrity check
    if (flags & 2) != 0 { // FLAG_INTEGRITY
         let footer_start = body_start_rel + body_len;
         let expected_hash_bytes: [u8; 8] = bytes[footer_start..footer_start+8].try_into().map_err(|_| {
             pyo3::exceptions::PyValueError::new_err("Failed to read hash")
         })?;
         let expected_hash = u64::from_le_bytes(expected_hash_bytes);
         
         let body_slice = &bytes[body_start_rel..body_start_rel+body_len];
         
         let actual_hash = if ver >= 3 {
             compute_integrity_hash(body_slice)
         } else {
             xxh3_64(body_slice)
         };
         
         if actual_hash != expected_hash {
              return Err(pyo3::exceptions::PyValueError::new_err("Integrity check failed: XXH3 mismatch"));
         }
    }

    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype_name)?;
    
    let array = if (flags & 4) != 0 {
        // Compressed Path: Decompress -> New Array
        let body_slice = &bytes[body_start_rel..body_start_rel+body_len];
        
        // Check for LZ4 Frame Magic: 0x04 0x22 0x4D 0x18
        let is_frame = body_slice.len() >= 4 && body_slice[0..4] == [0x04, 0x22, 0x4D, 0x18];

        let decompressed = if is_frame {
            let mut decoder = FrameDecoder::new(body_slice);
            let mut out = Vec::with_capacity(uncompressed_len);
            decoder.read_to_end(&mut out).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("LZ4 Frame Decompression failed: {}", e)))?;
            out
        } else {
             lz4_decompress(body_slice, uncompressed_len)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("LZ4 Block Decompression failed: {}", e)))?
        };
            
        let py_bytes = PyBytes::new(py, &decompressed);
        kwargs.set_item("count", num_elements)?;
        numpy.call_method("frombuffer", (py_bytes,), Some(kwargs))?
    } else {
        // Zero-Copy Path
        kwargs.set_item("offset", absolute_offset + body_start_rel)?;
        kwargs.set_item("count", num_elements)?;
        numpy.call_method("frombuffer", (root_data,), Some(kwargs))?
    };

    let reshaped = array.call_method1("reshape", (shape,))?;
    
    // Set writeable=False
    let flags_attr = reshaped.getattr("flags")?;
    flags_attr.setattr("writeable", false)?;

    Ok(Some(reshaped.into()))
}

#[pyfunction]
fn loads_rs<'py>(py: Python<'py>, data: &'py PyAny) -> PyResult<Option<PyObject>> {
    let buffer: PyBuffer<u8> = PyBuffer::get(data)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, buffer.len_bytes())
    };

    deserialize_impl(py, data, bytes, 0)
}


#[pymodule]
fn tenso_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_packet_info_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dumps_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dump_to_fd_rs, m)?)?;
    m.add_function(wrap_pyfunction!(loads_rs, m)?)?;
    Ok(())
}
