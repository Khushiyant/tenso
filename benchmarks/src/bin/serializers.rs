//! tenso vs the serializers used on microcontrollers / the edge: CBOR,
//! MessagePack, and postcard (the de-facto embedded-Rust format). All native,
//! no Python. Reports payload size + encode/decode (ns), single core.
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tenso::{decode, dense_required_size, encode_dense_into, ArraySpec, Dtype, EncodeOpts};

#[derive(Serialize, Deserialize)]
struct TensorMsg {
    shape: Vec<u32>,
    dtype: u8,
    data: Vec<u8>,
}

fn time_ns<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64 * 1e9
}

fn run(name: &str, data: &[u8], dtype: Dtype, shape: &[u32]) {
    let iters = 50_000;
    let opts = EncodeOpts::default();
    let spec = ArraySpec { data, dtype, shape };
    let mk = || TensorMsg { shape: shape.to_vec(), dtype: dtype.code(), data: data.to_vec() };

    let mut tbuf = vec![0u8; dense_required_size(&spec, &opts).unwrap()];
    let tn = encode_dense_into(&spec, &mut tbuf, &opts).unwrap();
    let tbuf = &tbuf[..tn];
    let cb = {
        let mut v = Vec::new();
        ciborium::into_writer(&mk(), &mut v).unwrap();
        v
    };
    let mp = rmp_serde::to_vec(&mk()).unwrap();
    let pc = postcard::to_allocvec(&mk()).unwrap();

    let e_tenso = time_ns(iters, || {
        let mut b = vec![0u8; dense_required_size(&spec, &opts).unwrap()];
        encode_dense_into(&spec, &mut b, &opts).unwrap();
    });
    let e_cbor = time_ns(iters, || {
        let mut v = Vec::new();
        ciborium::into_writer(&mk(), &mut v).unwrap();
    });
    let e_mp = time_ns(iters, || {
        let _ = rmp_serde::to_vec(&mk()).unwrap();
    });
    let e_pc = time_ns(iters, || {
        let _ = postcard::to_allocvec(&mk()).unwrap();
    });

    let d_tenso = time_ns(iters, || {
        let _ = decode(tbuf).unwrap();
    });
    let d_cbor = time_ns(iters, || {
        let _: TensorMsg = ciborium::from_reader(cb.as_slice()).unwrap();
    });
    let d_mp = time_ns(iters, || {
        let _: TensorMsg = rmp_serde::from_slice(&mp).unwrap();
    });
    let d_pc = time_ns(iters, || {
        let _: TensorMsg = postcard::from_bytes(&pc).unwrap();
    });

    println!("\n== {name} (raw {} B) ==", data.len());
    println!("  {:10} {:>8} {:>10} {:>10}  read", "fmt", "size", "enc ns", "dec ns");
    println!("  {:10} {:>8} {:>10.0} {:>10.0}  zero-copy, 64B-aligned", "tenso", tbuf.len(), e_tenso, d_tenso);
    println!("  {:10} {:>8} {:>10.0} {:>10.0}  parse + alloc", "postcard", pc.len(), e_pc, d_pc);
    println!("  {:10} {:>8} {:>10.0} {:>10.0}  parse + alloc", "msgpack", mp.len(), e_mp, d_mp);
    println!("  {:10} {:>8} {:>10.0} {:>10.0}  parse + alloc", "cbor", cb.len(), e_cbor, d_cbor);
}

fn main() {
    run("96x96x3 u8 (TinyML input)", &vec![123u8; 96 * 96 * 3], Dtype::U8, &[96, 96, 3]);
    let sensor: Vec<u8> = (0..256u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    run("256 f32 (sensor window)", &sensor, Dtype::F32, &[256]);
}
