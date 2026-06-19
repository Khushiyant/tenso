// PyO3 0.22 gil-refs API; Bound migration pending. Deprecation +
// useless_conversion (fires in macro-generated code) silenced until then.
#![allow(deprecated, clippy::useless_conversion)]

//! Thin PyO3 binding over `tenso`. Owns only Python-facing concerns
//! (numpy extraction, object construction, dense/bundle/sparse dispatch,
//! zero-copy `loads`, shm_mutex); all wire-format codec lives in `tenso`.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::fs::File;
use std::io::Write;
use std::mem::ManuallyDrop;
#[cfg(unix)]
use std::os::unix::io::FromRawFd;
#[cfg(windows)]
use std::os::windows::io::FromRawHandle;

use tenso::{
    bundle_required_size, decode, dense_required_size, encode_bundle_into, encode_dense_into,
    encode_quantized_into, encode_sparse_into, encode_string_into, parse_header,
    quantized_required_size, sparse_required_size, string_required_size, ArraySpec, Decoded, Dtype,
    EncodeOpts, QuantSpec, SparseFormat, TensoError, FLAG_ALIGNED, FLAG_BUNDLE, FLAG_INTEGRITY,
};

// -----------------------------------------------------------------------------
// Error mapping + small extraction helpers
// -----------------------------------------------------------------------------

/// Map a `TensoError` to a Python exception (messages match the old codec).
fn map_err(e: TensoError) -> PyErr {
    let msg = match e {
        TensoError::TooShort => "Packet too short".to_string(),
        TensoError::BadMagic => "Invalid tenso packet".to_string(),
        TensoError::UnsupportedVersion(v) => format!("Unsupported protocol version: {}", v),
        TensoError::BadDtype(c) => format!("Unsupported or unknown dtype code: {}", c),
        TensoError::TooManyDims => "Packet exceeds maximum dimensions".to_string(),
        TensoError::TooManyElements => "Packet exceeds maximum elements".to_string(),
        TensoError::IntegrityMismatch => "Integrity check failed: XXH3 mismatch".to_string(),
        TensoError::BadBundle => {
            "Bundle has too many entries; the wire format encodes the entry count in a \
             single byte, so at most 255 entries are supported"
                .to_string()
        }
        TensoError::Lz4(reason) => reason.to_string(),
        TensoError::BufferTooSmall => "Buffer too small".to_string(),
        TensoError::Malformed => "Malformed tenso packet".to_string(),
    };
    pyo3::exceptions::PyValueError::new_err(msg)
}

/// Convert shape dims to wire-format `u32`, rejecting oversized dims (a silent
/// `as u32` truncation would corrupt the array on reserialization, issue #4).
fn shape_to_u32(dims: &[usize]) -> PyResult<Vec<u32>> {
    dims.iter()
        .map(|&d| {
            u32::try_from(d).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Dimension {} exceeds the wire-format limit of {} \
                     (each dimension is stored as a 32-bit integer)",
                    d,
                    u32::MAX
                ))
            })
        })
        .collect()
}

/// Resolve a numpy dtype `name` string to a `tenso::Dtype`.
fn dtype_from_name(name: &str) -> PyResult<Dtype> {
    let d = match name {
        "float32" => Dtype::F32,
        "int32" => Dtype::I32,
        "float64" => Dtype::F64,
        "int64" => Dtype::I64,
        "uint8" => Dtype::U8,
        "uint16" => Dtype::U16,
        "bool" => Dtype::Bool,
        "float16" => Dtype::F16,
        "int8" => Dtype::I8,
        "int16" => Dtype::I16,
        "uint32" => Dtype::U32,
        "uint64" => Dtype::U64,
        "complex64" => Dtype::C64,
        "complex128" => Dtype::C128,
        "bfloat16" => Dtype::BF16,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dtype: {}",
                name
            )))
        }
    };
    Ok(d)
}

/// View of a numpy array: dtype + LE body bytes + u32 shape. `data` borrows the
/// array's buffer for `array`'s lifetime; caller must hold the GIL while used.
struct NumpyView<'a> {
    dtype: Dtype,
    data: &'a [u8],
    shape: Vec<u32>,
}

fn extract_numpy<'a>(array: &'a PyAny) -> PyResult<NumpyView<'a>> {
    let dtype_obj = array.getattr("dtype")?;
    let name: String = dtype_obj.getattr("name")?.extract()?;
    let dtype = dtype_from_name(&name)?;

    // Require C-contiguous: we read `nbytes` from `ctypes.data` row-major, which
    // is wrong (and can OOB) for strided/transposed/Fortran views. dumps_rs is
    // exported, so enforce here too (Python wrapper also coerces).
    let c_contiguous: bool = array.getattr("flags")?.getattr("c_contiguous")?.extract()?;
    if !c_contiguous {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Array must be C-contiguous",
        ));
    }

    let nbytes: usize = array.getattr("nbytes")?.extract()?;
    let data_ptr: usize = array.getattr("ctypes")?.getattr("data")?.extract()?;
    let shape_usize: Vec<usize> = array.getattr("shape")?.extract()?;
    let shape = shape_to_u32(&shape_usize)?;

    // SAFETY: C-contiguous (checked above), so `data_ptr` covers `nbytes` of LE
    // array memory while the GIL is held and `array` is alive.
    let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, nbytes) };
    Ok(NumpyView { dtype, data, shape })
}

/// Build `EncodeOpts` from the Python-facing flags.
fn opts(check_integrity: bool, compress: bool, alignment: usize) -> EncodeOpts {
    EncodeOpts {
        check_integrity,
        compress,
        alignment,
    }
}

// -----------------------------------------------------------------------------
// Container helpers (bundle / sparse) — collect bytes, then delegate to core
// -----------------------------------------------------------------------------

/// Recursively serialize each dict value via `dumps_rs` into owned
/// `(key, value_packet)` pairs that outlive the encode call.
fn collect_bundle_entries(
    py: Python,
    dict: &PyDict,
    check_integrity: bool,
    compress: bool,
    alignment: usize,
) -> PyResult<Vec<(String, Vec<u8>)>> {
    if dict.len() > tenso::MAX_BUNDLE_ENTRIES {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Bundle has {} entries; the wire format encodes the entry count in a \
             single byte, so at most 255 entries are supported",
            dict.len()
        )));
    }
    let mut entries = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let val_packet_obj = dumps_rs(py, value, check_integrity, compress, alignment)?;
        let val_bytes = val_packet_obj.as_bytes().to_vec();
        entries.push((key_str, val_bytes));
    }
    Ok(entries)
}

/// Sparse tensor's format flag + its three component arrays (coerced
/// C-contiguous) with owned LE body bytes.
struct SparseComponents {
    format: SparseFormat,
    shape: Vec<u32>,
    // (dtype, owned body bytes, shape) per component.
    comps: Vec<(Dtype, Vec<u8>, Vec<u32>)>,
}

fn collect_sparse_components<'py>(
    py: Python<'py>,
    tensor: &'py PyAny,
    format: &str,
) -> PyResult<SparseComponents> {
    let (sparse_format, names): (SparseFormat, [&str; 3]) = match format {
        "coo" => (SparseFormat::Coo, ["data", "row", "col"]),
        "csr" => (SparseFormat::Csr, ["data", "indices", "indptr"]),
        "csc" => (SparseFormat::Csc, ["data", "indices", "indptr"]),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unknown sparse format",
            ))
        }
    };

    let shape_usize: Vec<usize> = tensor.getattr("shape")?.extract()?;
    let shape = shape_to_u32(&shape_usize)?;

    let mut comps = Vec::with_capacity(3);
    for name in names {
        let arr = tensor.getattr(name)?;
        let contig = if arr.getattr("flags")?.getattr("c_contiguous")?.extract()? {
            arr
        } else {
            let numpy = py.import("numpy")?;
            numpy.call_method1("ascontiguousarray", (arr,))?
        };
        let view = extract_numpy(contig)?;
        comps.push((view.dtype, view.data.to_vec(), view.shape));
    }

    Ok(SparseComponents {
        format: sparse_format,
        shape,
        comps,
    })
}

// -----------------------------------------------------------------------------
// dumps_rs — numpy / dict / sparse -> packet bytes (PyBytes)
// -----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (array, check_integrity=false, compress=false, alignment=64))]
fn dumps_rs<'py>(
    py: Python<'py>,
    array: &'py PyAny,
    check_integrity: bool,
    compress: bool,
    alignment: usize,
) -> PyResult<&'py PyBytes> {
    if !alignment.is_power_of_two() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Alignment must be a power of two",
        ));
    }

    // CHECK FOR BUNDLE
    if let Ok(dict) = array.downcast::<PyDict>() {
        let entries = collect_bundle_entries(py, dict, check_integrity, compress, alignment)?;
        let refs: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let size = bundle_required_size(&refs).map_err(map_err)?;
        return PyBytes::new_with(py, size, |out: &mut [u8]| {
            encode_bundle_into(&refs, out).map_err(map_err)?;
            Ok(())
        });
    }

    // CHECK FOR SPARSE
    if let Ok(format_attr) = array.getattr("format") {
        if let Ok(format_str) = format_attr.extract::<String>() {
            let sc = collect_sparse_components(py, array, &format_str)?;
            let specs: Vec<ArraySpec> = sc
                .comps
                .iter()
                .map(|(dtype, data, shape)| ArraySpec {
                    data: data.as_slice(),
                    dtype: *dtype,
                    shape: shape.as_slice(),
                })
                .collect();
            let size = sparse_required_size(&sc.shape, &specs).map_err(map_err)?;
            return PyBytes::new_with(py, size, |out: &mut [u8]| {
                encode_sparse_into(sc.format, &sc.shape, &specs, out).map_err(map_err)?;
                Ok(())
            });
        }
    }

    // DENSE PATH
    let view = extract_numpy(array)?;
    let spec = ArraySpec {
        data: view.data,
        dtype: view.dtype,
        shape: &view.shape,
    };
    let o = opts(check_integrity, compress, alignment);
    let size = dense_required_size(&spec, &o).map_err(map_err)?;

    if compress {
        // Compressed size is an upper bound; encode into scratch and copy out
        // the exact bytes (trailing slack would break the integrity footer).
        let mut scratch = vec![0u8; size];
        let actual = encode_dense_into(&spec, &mut scratch, &o).map_err(map_err)?;
        Ok(PyBytes::new(py, &scratch[..actual]))
    } else {
        // Uncompressed: `size` is exact, fill the PyBytes in place (no copy).
        PyBytes::new_with(py, size, |out: &mut [u8]| {
            encode_dense_into(&spec, out, &o).map_err(map_err)?;
            Ok(())
        })
    }
}

// -----------------------------------------------------------------------------
// dumps_quantized_rs / dumps_string_rs — encode non-dense Python types via core.
// -----------------------------------------------------------------------------

/// Encode a `QuantizedTensor` (extracted field-by-field) via `tenso`.
#[pyfunction]
#[pyo3(signature = (qt, check_integrity=false, alignment=64))]
fn dumps_quantized_rs<'py>(
    py: Python<'py>,
    qt: &'py PyAny,
    check_integrity: bool,
    alignment: usize,
) -> PyResult<&'py PyBytes> {
    if !alignment.is_power_of_two() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Alignment must be a power of two",
        ));
    }
    let dtype_code: u8 = qt.getattr("dtype_code")?.extract()?;
    let dtype = Dtype::from_code(dtype_code).map_err(map_err)?;
    let scheme: u8 = qt.getattr("quant_scheme")?.extract()?;
    let group_size: u32 = qt.getattr("group_size")?.extract()?;
    // Python stores `axis & 0xFF` as a single byte.
    let axis_i: i64 = qt.getattr("axis")?.extract()?;
    let axis = (axis_i & 0xFF) as u8;
    let shape_usize: Vec<usize> = qt.getattr("shape")?.extract()?;
    let shape = shape_to_u32(&shape_usize)?;
    // scales / zero_points / data: numpy arrays -> raw LE bytes.
    let scales: Vec<u8> = qt.getattr("scales")?.call_method0("tobytes")?.extract()?;
    let zero_points: Vec<u8> = qt
        .getattr("zero_points")?
        .call_method0("tobytes")?
        .extract()?;
    let data: Vec<u8> = qt.getattr("data")?.call_method0("tobytes")?.extract()?;

    let spec = QuantSpec {
        dtype,
        shape: &shape,
        scheme,
        axis,
        group_size,
        scales: &scales,
        zero_points: &zero_points,
        data: &data,
    };
    let o = opts(check_integrity, false, alignment);
    let size = quantized_required_size(&spec, &o).map_err(map_err)?;
    PyBytes::new_with(py, size, |out: &mut [u8]| {
        encode_quantized_into(&spec, out, &o).map_err(map_err)?;
        Ok(())
    })
}

/// Encode a `StringTensor` from raw `offsets` ((count+1) u64 LE) + `payload`.
#[pyfunction]
#[pyo3(signature = (offsets, payload, count, check_integrity=false))]
fn dumps_string_rs<'py>(
    py: Python<'py>,
    offsets: &[u8],
    payload: &[u8],
    count: u32,
    check_integrity: bool,
) -> PyResult<&'py PyBytes> {
    if !offsets.len().is_multiple_of(8) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "offsets length must be a multiple of 8",
        ));
    }
    let offs: Vec<u64> = offsets
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let size = string_required_size(count, payload.len(), check_integrity).map_err(map_err)?;
    PyBytes::new_with(py, size, |out: &mut [u8]| {
        encode_string_into(count, &offs, payload, out, check_integrity).map_err(map_err)?;
        Ok(())
    })
}

/// Frame already-encoded `(key, value_packet)` entries into a bundle packet,
/// letting Python orchestrate recursion while the wire frame stays in Rust.
#[pyfunction]
fn encode_bundle_rs<'py>(
    py: Python<'py>,
    entries: Vec<(String, Vec<u8>)>,
) -> PyResult<&'py PyBytes> {
    let refs: Vec<(&str, &[u8])> = entries
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let size = bundle_required_size(&refs).map_err(map_err)?;
    PyBytes::new_with(py, size, |out: &mut [u8]| {
        encode_bundle_into(&refs, out).map_err(map_err)?;
        Ok(())
    })
}

// -----------------------------------------------------------------------------
// dump_to_buffer_rs — same dispatch, encode into a caller-provided buffer
// -----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (array, buffer, check_integrity=false, compress=false, alignment=64))]
fn dump_to_buffer_rs<'py>(
    py: Python<'py>,
    array: &'py PyAny,
    buffer: &'py PyAny,
    check_integrity: bool,
    compress: bool,
    alignment: usize,
) -> PyResult<usize> {
    let py_buf: PyBuffer<u8> = PyBuffer::get(buffer)?;
    if py_buf.readonly() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Buffer is read-only",
        ));
    }

    let buf_len = py_buf.len_bytes();
    let buf_ptr = py_buf.buf_ptr() as *mut u8;
    // SAFETY: PyBuffer guarantees `buf_ptr` is writable for `buf_len` bytes
    // while the GIL is held; `array` and `buffer` are distinct allocations.
    let target = unsafe { std::slice::from_raw_parts_mut(buf_ptr, buf_len) };

    if !alignment.is_power_of_two() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Alignment must be a power of two",
        ));
    }

    // CHECK FOR BUNDLE
    if let Ok(dict) = array.downcast::<PyDict>() {
        let entries = collect_bundle_entries(py, dict, check_integrity, compress, alignment)?;
        let refs: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        return encode_bundle_into(&refs, target).map_err(map_err);
    }

    // CHECK FOR SPARSE
    if let Ok(format_attr) = array.getattr("format") {
        if let Ok(format_str) = format_attr.extract::<String>() {
            let sc = collect_sparse_components(py, array, &format_str)?;
            let specs: Vec<ArraySpec> = sc
                .comps
                .iter()
                .map(|(dtype, data, shape)| ArraySpec {
                    data: data.as_slice(),
                    dtype: *dtype,
                    shape: shape.as_slice(),
                })
                .collect();
            return encode_sparse_into(sc.format, &sc.shape, &specs, target).map_err(map_err);
        }
    }

    // DENSE PATH
    let view = extract_numpy(array)?;
    let spec = ArraySpec {
        data: view.data,
        dtype: view.dtype,
        shape: &view.shape,
    };
    let o = opts(check_integrity, compress, alignment);
    encode_dense_into(&spec, target, &o).map_err(map_err)
}

// -----------------------------------------------------------------------------
// dump_to_fd_rs — dense only -> encode into a Vec -> write_all to the fd
// -----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (array, fd, check_integrity=false, compress=false, alignment=64))]
fn dump_to_fd_rs<'py>(
    py: Python<'py>,
    array: &'py PyAny,
    fd: i32,
    check_integrity: bool,
    compress: bool,
    alignment: usize,
) -> PyResult<usize> {
    if !alignment.is_power_of_two() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Alignment must be a power of two",
        ));
    }

    let is_contiguous: bool = array.getattr("flags")?.getattr("c_contiguous")?.extract()?;
    if !is_contiguous {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Array must be C-Contiguous",
        ));
    }

    let view = extract_numpy(array)?;
    let spec = ArraySpec {
        data: view.data,
        dtype: view.dtype,
        shape: &view.shape,
    };
    let o = opts(check_integrity, compress, alignment);

    // Encode into an owned buffer, then stream to the fd without the GIL.
    // Trim to actual length (compressed-size estimate is an upper bound).
    let size = dense_required_size(&spec, &o).map_err(map_err)?;
    let mut packet = vec![0u8; size];
    let actual = encode_dense_into(&spec, &mut packet, &o).map_err(map_err)?;
    packet.truncate(actual);

    py.allow_threads(move || {
        #[cfg(unix)]
        let mut file = unsafe { ManuallyDrop::new(File::from_raw_fd(fd)) };
        #[cfg(windows)]
        let mut file = unsafe {
            // C runtime fd -> Windows HANDLE.
            extern "C" {
                fn _get_osfhandle(fd: i32) -> isize;
            }
            let handle = _get_osfhandle(fd) as *mut std::ffi::c_void;
            ManuallyDrop::new(File::from_raw_handle(handle))
        };
        file.write_all(&packet)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(actual)
    })
}

// -----------------------------------------------------------------------------
// loads_rs — decode via tenso, zero-copy into the original Python buffer
// -----------------------------------------------------------------------------

/// Build a zero-copy numpy view over the dense body, mapped into `root_data`
/// at offset `body_ptr - base_ptr` (decode borrows the body from the packet).
fn dense_to_numpy<'py>(
    py: Python<'py>,
    root_data: &'py PyAny,
    base_ptr: usize,
    dtype: Dtype,
    shape: &[u32],
    body: &[u8],
) -> PyResult<PyObject> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype.name())?;

    let num_elements: usize = shape.iter().map(|&d| d as usize).product();
    kwargs.set_item("count", num_elements)?;

    let offset = body.as_ptr() as usize - base_ptr;
    kwargs.set_item("offset", offset)?;

    let array = numpy.call_method("frombuffer", (root_data,), Some(kwargs))?;
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let reshaped = array.call_method1("reshape", (shape_usize,))?;
    let flags_attr = reshaped.getattr("flags")?;
    flags_attr.setattr("writeable", false)?;
    Ok(reshaped.into())
}

/// Convert a `Decoded` to a Python object, recursing for bundles. `Ok(None)`
/// signals a kind the Python fallback owns (ragged, IpcRef).
fn decoded_to_py<'py>(
    py: Python<'py>,
    root_data: &'py PyAny,
    base_ptr: usize,
    decoded: Decoded,
) -> PyResult<Option<PyObject>> {
    match decoded {
        Decoded::Dense(view) => Ok(Some(dense_to_numpy(
            py,
            root_data,
            base_ptr,
            view.dtype,
            &view.shape,
            view.body,
        )?)),
        Decoded::Bundle(entries) => {
            let res = PyDict::new(py);
            for (key, val) in entries {
                match decoded_to_py(py, root_data, base_ptr, val)? {
                    Some(obj) => res.set_item(key, obj)?,
                    // A nested unsupported kind makes the whole packet fall back.
                    None => return Ok(None),
                }
            }
            Ok(Some(res.into()))
        }
        Decoded::Quantized(q) => {
            // Assemble a QuantizedTensor from the Rust-parsed fields.
            let np = py.import("numpy")?;
            let qcls = py.import("tenso.quantize")?.getattr("QuantizedTensor")?;
            let data = np
                .call_method1("frombuffer", (PyBytes::new(py, q.packed), "uint8"))?
                .call_method0("copy")?;
            let scales = np
                .call_method1("frombuffer", (PyBytes::new(py, q.scales), "float32"))?
                .call_method0("copy")?;
            let zero_points = np
                .call_method1("frombuffer", (PyBytes::new(py, q.zero_points), "float32"))?
                .call_method0("copy")?;
            let shape = PyTuple::new(py, q.shape.iter().map(|&d| d as usize));
            let kwargs = PyDict::new(py);
            kwargs.set_item("data", data)?;
            kwargs.set_item("scales", scales)?;
            kwargs.set_item("zero_points", zero_points)?;
            kwargs.set_item("shape", shape)?;
            kwargs.set_item("dtype_code", q.dtype.code())?;
            kwargs.set_item("quant_scheme", q.scheme)?;
            kwargs.set_item("group_size", q.group_size)?;
            kwargs.set_item("axis", q.axis)?;
            Ok(Some(qcls.call((), Some(kwargs))?.into()))
        }
        Decoded::String {
            shape,
            offsets,
            payload,
        } => {
            // StringTensor via its raw-field classmethod.
            let np = py.import("numpy")?;
            let scls = py.import("tenso.ragged")?.getattr("StringTensor")?;
            let offs = np
                .call_method1("frombuffer", (PyBytes::new(py, offsets), "uint64"))?
                .call_method0("copy")?;
            let count = *shape.first().unwrap_or(&0) as usize;
            Ok(Some(
                scls.call_method1("_from_raw", (offs, PyBytes::new(py, payload), count))?
                    .into(),
            ))
        }
        Decoded::Sparse {
            format,
            shape,
            components,
        } => {
            // Components are dense sub-packets; build them, then assemble scipy.
            let sparse = py.import("scipy.sparse").map_err(|_| {
                pyo3::exceptions::PyImportError::new_err(
                    "scipy is required for sparse deserialization",
                )
            })?;
            let mut comps: Vec<PyObject> = Vec::with_capacity(components.len());
            for c in components {
                match decoded_to_py(py, root_data, base_ptr, c)? {
                    Some(o) => comps.push(o),
                    None => return Ok(None),
                }
            }
            if comps.len() != 3 {
                return Ok(None);
            }
            let shape_t = PyTuple::new(py, shape.iter().map(|&d| d as usize));
            let (c0, c1, c2) = (&comps[0], &comps[1], &comps[2]);
            let obj = match format {
                SparseFormat::Coo => {
                    let coords = PyTuple::new(py, [c1, c2]);
                    sparse.call_method1("coo_matrix", ((c0, coords), shape_t))?
                }
                SparseFormat::Csr => sparse.call_method1("csr_matrix", ((c0, c1, c2), shape_t))?,
                SparseFormat::Csc => sparse.call_method1("csc_matrix", ((c0, c1, c2), shape_t))?,
            };
            Ok(Some(obj.into()))
        }
        // Ragged is a bundle in practice; IpcRef stays a Python concern.
        _ => Ok(None),
    }
}

#[pyfunction]
fn loads_rs<'py>(py: Python<'py>, data: &'py PyAny) -> PyResult<Option<PyObject>> {
    let buffer: PyBuffer<u8> = PyBuffer::get(data)?;
    let base_ptr = buffer.buf_ptr() as usize;
    // SAFETY: PyBuffer keeps `data` readable for `len_bytes()` under the GIL.
    let bytes =
        unsafe { std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, buffer.len_bytes()) };

    match decode(bytes) {
        Ok(decoded) => decoded_to_py(py, data, base_ptr, decoded),
        // Compressed dense: decompress via core, build an owned numpy array.
        Err(TensoError::Lz4(_)) => {
            let (dtype, shape, owned) = tenso::decode_dense_to_owned(bytes).map_err(map_err)?;
            let np = py.import("numpy")?;
            let arr = np
                .call_method1("frombuffer", (PyBytes::new(py, &owned), dtype.name()))?
                .call_method0("copy")?;
            let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            Ok(Some(arr.call_method1("reshape", (shape_usize,))?.into()))
        }
        Err(e) => Err(map_err(e)),
    }
}

// -----------------------------------------------------------------------------
// get_packet_info_rs — header + shape metadata dict
// -----------------------------------------------------------------------------

#[pyfunction]
fn get_packet_info_rs<'py>(py: Python<'py>, data: &'py PyAny) -> PyResult<&'py PyDict> {
    let buffer: PyBuffer<u8> = PyBuffer::get(data)?;
    let bytes =
        unsafe { std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, buffer.len_bytes()) };

    let hdr = parse_header(bytes).map_err(map_err)?;

    let dict = PyDict::new(py);
    dict.set_item("version", bytes[4])?;
    dict.set_item("flags", hdr.flags)?;
    dict.set_item("dtype_code", hdr.dtype_code)?;
    dict.set_item("ndim", hdr.ndim)?;
    dict.set_item("aligned", (hdr.flags & FLAG_ALIGNED) != 0)?;
    dict.set_item("integrity_protected", (hdr.flags & FLAG_INTEGRITY) != 0)?;

    if hdr.flags & FLAG_BUNDLE != 0 {
        // For a bundle, `ndim` is the entry count and post-header bytes are
        // key-length prefixes, not dims. Report entry count + empty shape.
        dict.set_item("entry_count", hdr.ndim)?;
        dict.set_item("shape", PyTuple::empty(py))?;
        dict.set_item("total_elements", 0usize)?;
    } else {
        // dense / sparse / quantized / string: post-header bytes are the shape.
        let shape_end = hdr.base_size + (hdr.ndim * 4);
        if bytes.len() < shape_end {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Packet too short to contain shape",
            ));
        }
        let mut shape = Vec::with_capacity(hdr.ndim);
        let mut cursor = hdr.base_size;
        for _ in 0..hdr.ndim {
            let dim = u32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]) as usize;
            shape.push(dim);
            cursor += 4;
        }
        // Saturating product: dims are attacker-controlled, so a plain product
        // could wrap. (decode() separately enforces MAX_NDIM / MAX_ELEMENTS.)
        let total_elements: usize = shape.iter().fold(1usize, |acc, &d| acc.saturating_mul(d));
        dict.set_item("total_elements", total_elements)?;
        dict.set_item("shape", PyTuple::new(py, shape))?;
    }

    Ok(dict)
}

// -----------------------------------------------------------------------------
// Cross-Process Robust Mutex (POSIX pthread_mutex in shared memory)
// -----------------------------------------------------------------------------

#[cfg(unix)]
mod shm_mutex {
    use std::time::Duration;

    // pthread_mutex_t in shared memory (macOS 64B, Linux 40B); reserve 64.
    pub const MUTEX_SIZE: usize = 64;

    extern "C" {
        fn pthread_mutexattr_init(attr: *mut libc::pthread_mutexattr_t) -> libc::c_int;
        fn pthread_mutexattr_setpshared(
            attr: *mut libc::pthread_mutexattr_t,
            pshared: libc::c_int,
        ) -> libc::c_int;
        fn pthread_mutexattr_destroy(attr: *mut libc::pthread_mutexattr_t) -> libc::c_int;
        fn pthread_mutex_init(
            mutex: *mut libc::pthread_mutex_t,
            attr: *const libc::pthread_mutexattr_t,
        ) -> libc::c_int;
        #[allow(dead_code)] // declared for API completeness; lock path uses trylock
        fn pthread_mutex_lock(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_unlock(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_trylock(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_destroy(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;

        // Robust mutex (Linux)
        #[cfg(target_os = "linux")]
        fn pthread_mutexattr_setrobust(
            attr: *mut libc::pthread_mutexattr_t,
            robust: libc::c_int,
        ) -> libc::c_int;
        #[cfg(target_os = "linux")]
        fn pthread_mutex_consistent(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;
    }

    /// Init a process-shared mutex at `ptr` (>= MUTEX_SIZE zeroed shm bytes).
    pub unsafe fn init_mutex(ptr: *mut u8) -> Result<(), String> {
        let mutex = ptr as *mut libc::pthread_mutex_t;
        let mut attr: libc::pthread_mutexattr_t = std::mem::zeroed();

        if pthread_mutexattr_init(&mut attr) != 0 {
            return Err("pthread_mutexattr_init failed".into());
        }
        if pthread_mutexattr_setpshared(&mut attr, libc::PTHREAD_PROCESS_SHARED) != 0 {
            pthread_mutexattr_destroy(&mut attr);
            return Err("pthread_mutexattr_setpshared failed".into());
        }

        // Linux: robust mutex to recover from crashed holders.
        #[cfg(target_os = "linux")]
        {
            if pthread_mutexattr_setrobust(&mut attr, libc::PTHREAD_MUTEX_ROBUST) != 0 {
                pthread_mutexattr_destroy(&mut attr);
                return Err("pthread_mutexattr_setrobust failed".into());
            }
        }

        let rc = pthread_mutex_init(mutex, &attr);
        pthread_mutexattr_destroy(&mut attr);
        if rc != 0 {
            return Err(format!("pthread_mutex_init failed: {}", rc));
        }
        Ok(())
    }

    /// Lock with timeout. `Ok(true)` if recovered from a dead owner (Linux).
    pub unsafe fn lock_mutex(ptr: *mut u8, timeout: Duration) -> Result<bool, String> {
        let mutex = ptr as *mut libc::pthread_mutex_t;
        let deadline = std::time::Instant::now() + timeout;

        loop {
            let rc = pthread_mutex_trylock(mutex);
            if rc == 0 {
                return Ok(false); // normal
            }

            // EOWNERDEAD: prior holder crashed (Linux robust mutex)
            #[cfg(target_os = "linux")]
            if rc == libc::EOWNERDEAD {
                pthread_mutex_consistent(mutex);
                return Ok(true); // recovered
            }

            if rc == libc::EBUSY {
                if std::time::Instant::now() >= deadline {
                    return Err(format!("Mutex lock timed out after {:?}", timeout));
                }
                std::thread::sleep(Duration::from_micros(50));
                continue;
            }

            return Err(format!("pthread_mutex_trylock failed: {}", rc));
        }
    }

    /// Unlock the mutex.
    pub unsafe fn unlock_mutex(ptr: *mut u8) -> Result<(), String> {
        let mutex = ptr as *mut libc::pthread_mutex_t;
        let rc = pthread_mutex_unlock(mutex);
        if rc != 0 {
            return Err(format!("pthread_mutex_unlock failed: {}", rc));
        }
        Ok(())
    }

    pub unsafe fn destroy_mutex(ptr: *mut u8) -> Result<(), String> {
        let mutex = ptr as *mut libc::pthread_mutex_t;
        let rc = pthread_mutex_destroy(mutex);
        if rc != 0 {
            return Err(format!("pthread_mutex_destroy failed: {}", rc));
        }
        Ok(())
    }
}

/// Init a POSIX process-shared mutex at `offset` in a shm-backed buffer
/// (needs >= 64 bytes there).
#[cfg(unix)]
#[pyfunction]
fn shm_mutex_init(buffer: &PyAny, offset: usize) -> PyResult<()> {
    let py_buf: PyBuffer<u8> = PyBuffer::get(buffer)?;
    if py_buf.readonly() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Buffer is read-only",
        ));
    }
    let buf_len = py_buf.len_bytes();
    let ptr = py_buf.buf_ptr() as *mut u8;

    if offset + shm_mutex::MUTEX_SIZE > buf_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Insufficient space: need {} bytes at offset {}, buffer is {} bytes",
            shm_mutex::MUTEX_SIZE,
            offset,
            buf_len
        )));
    }

    unsafe {
        shm_mutex::init_mutex(ptr.add(offset)).map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}

/// Lock a POSIX process-shared mutex. True if recovered from a dead owner.
#[cfg(unix)]
#[pyfunction]
#[pyo3(signature = (buffer, offset, timeout_secs=5.0))]
fn shm_mutex_lock(buffer: &PyAny, offset: usize, timeout_secs: f64) -> PyResult<bool> {
    let py_buf: PyBuffer<u8> = PyBuffer::get(buffer)?;
    if py_buf.readonly() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Buffer is read-only",
        ));
    }
    let ptr = py_buf.buf_ptr() as *mut u8;
    let timeout = std::time::Duration::from_secs_f64(timeout_secs);

    unsafe {
        shm_mutex::lock_mutex(ptr.add(offset), timeout)
            .map_err(pyo3::exceptions::PyTimeoutError::new_err)
    }
}

/// Unlock a POSIX process-shared mutex.
#[cfg(unix)]
#[pyfunction]
fn shm_mutex_unlock(buffer: &PyAny, offset: usize) -> PyResult<()> {
    let py_buf: PyBuffer<u8> = PyBuffer::get(buffer)?;
    if py_buf.readonly() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Buffer is read-only",
        ));
    }
    let ptr = py_buf.buf_ptr() as *mut u8;

    unsafe {
        shm_mutex::unlock_mutex(ptr.add(offset)).map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}

/// Destroy a POSIX process-shared mutex.
#[cfg(unix)]
#[pyfunction]
fn shm_mutex_destroy(buffer: &PyAny, offset: usize) -> PyResult<()> {
    let py_buf: PyBuffer<u8> = PyBuffer::get(buffer)?;
    if py_buf.readonly() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Buffer is read-only",
        ));
    }
    let ptr = py_buf.buf_ptr() as *mut u8;

    unsafe {
        shm_mutex::destroy_mutex(ptr.add(offset)).map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}

/// Return the size in bytes needed for one POSIX process-shared mutex.
#[cfg(unix)]
#[pyfunction]
fn shm_mutex_size() -> usize {
    shm_mutex::MUTEX_SIZE
}

#[pymodule]
fn tenso_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_packet_info_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dumps_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dumps_quantized_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dumps_string_rs, m)?)?;
    m.add_function(wrap_pyfunction!(encode_bundle_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dump_to_buffer_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dump_to_fd_rs, m)?)?;
    m.add_function(wrap_pyfunction!(loads_rs, m)?)?;
    #[cfg(unix)]
    {
        m.add_function(wrap_pyfunction!(shm_mutex_init, m)?)?;
        m.add_function(wrap_pyfunction!(shm_mutex_lock, m)?)?;
        m.add_function(wrap_pyfunction!(shm_mutex_unlock, m)?)?;
        m.add_function(wrap_pyfunction!(shm_mutex_destroy, m)?)?;
        m.add_function(wrap_pyfunction!(shm_mutex_size, m)?)?;
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Fuzzing shims (only under `--cfg fuzzing`): byte-in/Result-out API over
// tenso so fuzz targets can parse without PyO3. Not for prod/tests.
// -----------------------------------------------------------------------------

#[cfg(fuzzing)]
pub mod fuzz_api {
    use tenso::{parse_header, TensoError, HEADER_BASE_V4};

    /// Result of a successful header + shape parse.
    #[derive(Debug)]
    pub struct ParsedShape {
        pub version: u8,
        pub flags: u16,
        pub dtype_code: u8,
        pub ndim: usize,
        pub base_size: usize,
        pub shape: Vec<usize>,
    }

    /// Pure-Rust wrapper around `tenso::parse_header` for fuzzing.
    pub fn fuzz_parse_header(bytes: &[u8]) -> Result<(u16, u8, usize, usize), TensoError> {
        let h = parse_header(bytes)?;
        Ok((h.flags, h.dtype_code, h.ndim, h.base_size))
    }

    /// Parse header + per-dim shape (the dense pre-body decode), delegating
    /// to `tenso`.
    /// TODO(fuzz): rest of decode (LZ4/integrity/numpy) is in core or PyO3.
    pub fn fuzz_parse_header_and_shape(bytes: &[u8]) -> Result<ParsedShape, TensoError> {
        let h = parse_header(bytes)?;
        let shape_end = h
            .base_size
            .checked_add(h.ndim.checked_mul(4).ok_or(TensoError::Malformed)?)
            .ok_or(TensoError::Malformed)?;
        if bytes.len() < shape_end {
            return Err(TensoError::TooShort);
        }
        let mut shape = Vec::with_capacity(h.ndim);
        let mut cursor = h.base_size;
        for _ in 0..h.ndim {
            let dim_bytes: [u8; 4] = bytes[cursor..cursor + 4]
                .try_into()
                .map_err(|_| TensoError::TooShort)?;
            shape.push(u32::from_le_bytes(dim_bytes) as usize);
            cursor += 4;
        }
        Ok(ParsedShape {
            version: bytes[4],
            flags: h.flags,
            dtype_code: h.dtype_code,
            ndim: h.ndim,
            base_size: h.base_size,
            shape,
        })
    }

    /// Re-export the header base size so fuzz targets can sanity-check offsets.
    pub const HEADER_BASE_V4_PUB: usize = HEADER_BASE_V4;
}
