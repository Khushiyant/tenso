//! tenso vs Apache Arrow IPC — native Rust, no Python (the Python binding hid
//! the difference). Reports payload size + encode/decode (ns), single core.
use arrow::array::{ArrayRef, UInt8Array};
use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use std::time::Instant;
use tenso::{decode, dense_required_size, encode_dense_into, ArraySpec, Dtype, EncodeOpts};

fn time_ns<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64 * 1e9
}

fn arrow_encode(data: &[u8]) -> Vec<u8> {
    let arr = UInt8Array::from(data.to_vec());
    let batch = RecordBatch::try_from_iter([("d", Arc::new(arr) as ArrayRef)]).unwrap();
    let mut buf = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut buf, &batch.schema()).unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
    }
    buf
}

fn run(name: &str, data: &[u8], shape: &[u32]) {
    let iters = 20_000;
    let opts = EncodeOpts::default();
    let spec = ArraySpec { data, dtype: Dtype::U8, shape };
    let mut tb = vec![0u8; dense_required_size(&spec, &opts).unwrap()];
    let tn = encode_dense_into(&spec, &mut tb, &opts).unwrap();
    let tb = &tb[..tn];
    let ab = arrow_encode(data);

    let e_t = time_ns(iters, || {
        let mut b = vec![0u8; dense_required_size(&spec, &opts).unwrap()];
        encode_dense_into(&spec, &mut b, &opts).unwrap();
    });
    let e_a = time_ns(iters, || {
        let _ = arrow_encode(data);
    });
    let d_t = time_ns(iters, || {
        let _ = decode(tb).unwrap();
    });
    let d_a = time_ns(iters, || {
        let mut r = StreamReader::try_new(ab.as_slice(), None).unwrap();
        let _ = r.next().unwrap().unwrap();
    });

    println!("\n== {name} (raw {} B) ==", data.len());
    println!("  {:8} {:>9} {:>11} {:>11}  read", "fmt", "size", "enc ns", "dec ns");
    println!("  {:8} {:>9} {:>11.0} {:>11.0}  zero-copy borrow", "tenso", tb.len(), e_t, d_t);
    println!("  {:8} {:>9} {:>11.0} {:>11.0}  IPC parse + rebuild", "arrow", ab.len(), e_a, d_a);
}

fn main() {
    run("96x96x3 u8 (TinyML)", &vec![123u8; 96 * 96 * 3], &[96, 96, 3]);
    run("720p RGB frame", &vec![200u8; 1280 * 720 * 3], &[720, 1280, 3]);
}
