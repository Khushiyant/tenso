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
                pyo3::exceptions::PyValueError::new_err("Array must be C-Contiguous")  // array.flags.c_contiguous
            })?; 

            let u8_slice = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * item_size) // array.tobytes()
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
        "float32" => serialize_impl!(py, array, f32), // serialize_f32(array)
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


#[pymodule]
fn tenso_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(get_packet_info_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dumps_rs, m)?)?;
    Ok(())
}