//! tenso vs CDR — the serialization ROS 2's DDS uses on the wire for every
//! message (e.g. sensor_msgs/Image). Native Rust, no Python, no ROS install.
//! NOTE: this is the serde-based `cdr` crate; production C++ ROS CDR is faster
//! (closer to a memcpy), so treat the *multiplier* as an upper bound — the
//! durable win is structural (zero-copy decode + 64B alignment vs copy+alloc).
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tenso::{decode, dense_required_size, encode_dense_into, ArraySpec, Dtype, EncodeOpts};

#[derive(Serialize, Deserialize)]
struct ImageMsg {
    width: u32,
    height: u32,
    channels: u8,
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

fn run(name: &str, data: &[u8], shape: &[u32]) {
    let iters = 20_000;
    let opts = EncodeOpts::default();
    let spec = ArraySpec { data, dtype: Dtype::U8, shape };
    let mk = || ImageMsg { width: shape[1], height: shape[0], channels: 3, data: data.to_vec() };

    let mut tb = vec![0u8; dense_required_size(&spec, &opts).unwrap()];
    let tn = encode_dense_into(&spec, &mut tb, &opts).unwrap();
    let tb = &tb[..tn];
    let cb = cdr::serialize::<_, _, cdr::CdrLe>(&mk(), cdr::Infinite).unwrap();

    let e_t = time_ns(iters, || {
        let mut b = vec![0u8; dense_required_size(&spec, &opts).unwrap()];
        encode_dense_into(&spec, &mut b, &opts).unwrap();
    });
    let e_c = time_ns(iters, || {
        let _ = cdr::serialize::<_, _, cdr::CdrLe>(&mk(), cdr::Infinite).unwrap();
    });
    let d_t = time_ns(iters, || {
        let _ = decode(tb).unwrap();
    });
    let d_c = time_ns(iters, || {
        let _: ImageMsg = cdr::deserialize::<ImageMsg>(&cb).unwrap();
    });

    println!("\n== {name} (raw {} B) ==", data.len());
    println!("  {:14} {:>9} {:>11} {:>11}  read", "fmt", "size", "enc ns", "dec ns");
    println!("  {:14} {:>9} {:>11.0} {:>11.0}  zero-copy borrow", "tenso", tb.len(), e_t, d_t);
    println!("  {:14} {:>9} {:>11.0} {:>11.0}  copy + alloc", "CDR (ROS/DDS)", cb.len(), e_c, d_c);
}

fn main() {
    run("96x96x3 (TinyML)", &vec![123u8; 96 * 96 * 3], &[96, 96, 3]);
    run("720p RGB frame", &vec![200u8; 1280 * 720 * 3], &[720, 1280, 3]);
}
