# tenso benchmarks

Native benchmarks comparing tenso against the formats used to move tensors — on
the edge, in Apache Arrow pipelines, and in ROS 2. Every number is reproducible
with the scripts in this directory.

tenso's decode is a zero-copy borrow: it returns an aligned view over the bytes
with no parse and no allocation (~15 ns, flat, regardless of tensor size). The
serializers below parse the stream and allocate + copy the whole array on every
read. That is the gap.

## Edge serializers — CBOR, MessagePack, postcard

What runs on microcontrollers (TinyCBOR/QCBOR, MessagePack, embedded-Rust postcard).

**96x96x3 uint8 (TinyML input, 27,648 B raw)**

| format   | size     | encode     | decode       | read                   |
| -------- | -------- | ---------- | ------------ | ---------------------- |
| tenso    | 27,712 B | 0.9 us     | 0.015 us     | zero-copy, 64B-aligned |
| postcard | 27,656 B | 12.4 us    | 17.3 us      | parse + alloc          |
| msgpack  | 27,657 B | 40.7 us    | 39.6 us      | parse + alloc          |
| cbor     | 55,324 B | 88.6 us    | 492 us       | parse + alloc          |

**256 float32 (sensor window, 1,024 B raw):** tenso 0.04 us / 0.014 us · postcard
0.53 / 0.63 us · msgpack 1.7 / 1.5 us · cbor 3.1 / 18.3 us.

## Apache Arrow (IPC)

| case           | format | size     | encode  | decode    |
| -------------- | ------ | -------- | ------- | --------- |
| 96x96x3        | tenso  | 27,712 B | 0.9 us  | 0.02 us   |
|                | arrow  | 31,432 B | 2.8 us  | 1.4 us    |
| 720p RGB frame | tenso  | 2.76 MB  | 148 us  | 0.015 us  |
|                | arrow  | 3.11 MB  | 3.19 ms | 180 us    |

## ROS 2 — CDR (the DDS wire format)

| case           | format | size     | encode  | decode    |
| -------------- | ------ | -------- | ------- | --------- |
| 96x96x3        | tenso  | 27,712 B | 1.0 us  | 0.015 us  |
|                | CDR    | 27,668 B | 12.3 us | 22.8 us   |
| 720p RGB frame | tenso  | 2.76 MB  | 150 us  | 0.015 us  |
|                | CDR    | 2.76 MB  | 2.70 ms | 2.37 ms   |

In ROS you don't replace DDS — you ride on it: publish a `tenso_msgs/TensoBlob`
(a byte payload) instead of `sensor_msgs/Image`, and DDS moves an opaque blob
while the tensor stays zero-copy and aligned on both ends.

## Footprint

| | size | extra runtime |
| --- | --- | --- |
| tenso (`libtenso_ffi.so`, stripped) | 376 KB | none |
| Fast DDS (ROS 2 default) | several MB | none |
| Cyclone DDS | 1-2 MB | none |
| iceoryx (DDS zero-copy path) | MBs | a background daemon (RouDi) |

## Runs on the edge

The core is `no_std + alloc` and builds for microcontroller targets:

| target | device class |
| --- | --- |
| `thumbv7em-none-eabi` | ARM Cortex-M4 |
| `riscv32imc-unknown-none-elf` | ESP32-C3 / C6 (RISC-V) |

`compression` (LZ4) needs `std` and is excluded on bare metal; `integrity`
(XXH3) is `no_std` and builds.

## Reproduce

```bash
cd benchmarks
cargo run --release --bin serializers   # vs CBOR / MessagePack / postcard
cargo run --release --bin arrow_bench    # vs Apache Arrow IPC
cargo run --release --bin cdr_bench      # vs ROS 2 / CDR

# embedded build proofs (core only)
rustup target add thumbv7em-none-eabi riscv32imc-unknown-none-elf
cargo build -p tenso --no-default-features --features integrity --target thumbv7em-none-eabi
cargo build -p tenso --no-default-features --features integrity --target riscv32imc-unknown-none-elf
```

Machine: Intel Core Ultra 7 255HX, Linux, `--release`, single core, 20k-50k
iterations averaged. Native Rust, no Python binding.
