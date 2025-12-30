use numpy::PyArrayDyn;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::convert::TryInto;
use std::mem;

// -----------------------------------------------------------------------------
// Helper Macro for Dtype Dispatch
// -----------------------------------------------------------------------------

macro_rules! serialize_impl {
    ($py:expr, $array:expr, $dtype_code:ty) => {{
        let array = $array.downcast::<PyArrayDyn<$dtype_code>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Type mismatch in serialization dispatch")
        })?;

        let ndim = array.ndim();
        let shape = array.shape();

        let header_len = 8 + (ndim * 4);
        let alignment = 64;
        let remainder = header_len % alignment;
        let padding_len = if remainder == 0 {
            0
        } else {
            alignment - remainder
        };

        let total_elements: usize = shape.iter().product();
        let item_size = mem::size_of::<$dtype_code>();
        let body_len = total_elements * item_size;
        let total_len = header_len + padding_len + body_len;

        PyBytes::new_with($py, total_len, |bytes: &mut [u8]| {
            bytes[0..4].copy_from_slice(b"TNSO");
            bytes[4] = 2; // Version
            bytes[5] = 1; // Flags (Aligned)

            let type_id = std::any::TypeId::of::<$dtype_code>();

            bytes[6] = if type_id == std::any::TypeId::of::<f32>() {
                1
            } else if type_id == std::any::TypeId::of::<i32>() {
                2
            } else if type_id == std::any::TypeId::of::<f64>() {
                3
            } else if type_id == std::any::TypeId::of::<i64>() {
                4
            } else if type_id == std::any::TypeId::of::<u8>() {
                5
            } else if type_id == std::any::TypeId::of::<u16>() {
                6
            } else if type_id == std::any::TypeId::of::<bool>() {
                7
            } else {
                0
            };

            bytes[7] = ndim as u8;

            let mut cursor = 8;
            for &dim in shape {
                bytes[cursor..cursor + 4].copy_from_slice(&(dim as u32).to_le_bytes());
                cursor += 4;
            }

            let body_start = header_len + padding_len;

            let slice = unsafe { array.as_slice() }.map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("Array must be C-Contiguous")
            })?;

            let u8_slice = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * item_size)
            };
            bytes[body_start..body_start + body_len].copy_from_slice(u8_slice);

            Ok(())
        })
    }};
}

// -----------------------------------------------------------------------------
// Main Dumps Function
// -----------------------------------------------------------------------------

#[pyfunction]
fn dumps_rs<'py>(py: Python<'py>, array: &'py PyAny) -> PyResult<&'py PyBytes> {
    let dtype = array.getattr("dtype")?;
    let name: String = dtype.getattr("name")?.extract()?;

    match name.as_str() {
        "float32" => serialize_impl!(py, array, f32),
        "int32" => serialize_impl!(py, array, i32),
        "float64" => serialize_impl!(py, array, f64),
        "int64" => serialize_impl!(py, array, i64),
        "uint8" => serialize_impl!(py, array, u8),
        "uint16" => serialize_impl!(py, array, u16),
        "bool" => serialize_impl!(py, array, bool),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dtype: {}",
            name
        ))),
    }
}

// -----------------------------------------------------------------------------
// Packet Info (Fixed for PyBuffer/ReadOnlyCell)
// -----------------------------------------------------------------------------

#[pyfunction]
fn get_packet_info_rs(py: Python, data: PyBuffer<u8>) -> PyResult<PyObject> {
    // 1. Get the ReadOnlyCell slice
    let raw_slice = data.as_slice(py).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Buffer is not contiguous or readable")
    })?;

    // 2. Cast to &[u8] for easier handling
    // SAFETY: ReadOnlyCell<u8> is repr(transparent) wrapping UnsafeCell<u8>.
    // Since we are treating it as read-only bytes, casting to &[u8] is safe here.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(raw_slice.as_ptr() as *const u8, raw_slice.len()) };

    // 3. Basic Bounds Check
    if bytes.len() < 8 {
        return Err(pyo3::exceptions::PyValueError::new_err("Packet too short"));
    }

    // 4. Magic Check
    if &bytes[0..4] != b"TNSO" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Invalid tenso packet",
        ));
    }

    // 5. Header Parsing
    let ver = bytes[4];
    let flags = bytes[5];
    let dtype_code = bytes[6];
    let ndim = bytes[7] as usize;

    // 6. Shape Parsing
    let shape_start = 8;
    let shape_end = shape_start + (ndim * 4);

    if bytes.len() < shape_end {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Packet too short to contain shape",
        ));
    }

    let mut shape_vec = Vec::with_capacity(ndim);
    let mut total_elements: usize = 1;

    for i in 0..ndim {
        let start = shape_start + (i * 4);
        let end = start + 4;
        let dim_bytes: [u8; 4] = bytes[start..end].try_into().unwrap();
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        shape_vec.push(dim);
        total_elements *= dim;
    }

    // 7. Build Result Dictionary
    let dict = PyDict::new(py);
    let shape_tuple = PyTuple::new(py, shape_vec);

    dict.set_item("version", ver)?;
    dict.set_item("flags", flags)?;
    dict.set_item("dtype_code", dtype_code)?;
    dict.set_item("ndim", ndim)?;
    dict.set_item("shape", shape_tuple)?;
    dict.set_item("total_elements", total_elements)?;

    let aligned = (flags & 1) != 0;
    let integrity = (flags & 2) != 0;
    dict.set_item("aligned", aligned)?;
    dict.set_item("integrity_protected", integrity)?;

    Ok(dict.to_object(py))
}

#[pymodule]
fn tenso_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_packet_info_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dumps_rs, m)?)?;
    Ok(())
}