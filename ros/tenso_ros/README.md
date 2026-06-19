# tenso_ros — Zero-copy tensors in ROS2

ROS2 integration for the [Tenso](../../README.md) tensor wire format. It moves
tensors between nodes as opaque, self-describing Tenso packets and decodes them
**zero-copy** via an `rclcpp::TypeAdapter`.

> **Linux-only.** ROS2 is not built on macOS/Windows in this project. Both
> `tenso_msgs` and `tenso_ros` `CMakeLists.txt` hard-fail on non-Linux systems.

## Packages

| Package      | What it provides |
|--------------|------------------|
| `tenso_msgs` | `TensoBlob.msg` — `std_msgs/Header header` + `uint8[] payload` (a complete Tenso packet). |
| `tenso_ros`  | `TypeAdapter<TensorView, TensoBlob>` + publisher/subscriber skeletons + this demo. |

## How the zero-copy path works

1. The producer encodes a tensor **once** into a Tenso packet using the native C
   core (`cpp/tenso.hpp` → `include/tenso.h` → `tenso-ffi`). No Python, no
   hand-rolled byte layout.
2. It publishes a `tenso_ros::TensorView` (an owner of those encoded bytes) on a
   publisher declared with the *adapted* type:
   `rclcpp::Publisher<tenso_ros::TensoTensorAdapted>`.
3. With `use_intra_process_comms(true)` and a subscription using the **same**
   adapted type, rclcpp moves the buffer to the subscriber without
   re-serializing.
4. The subscriber calls `tensor.view()` (`tenso::read`) to get a
   `tenso::Packet` whose body **borrows** from the received buffer — the tensor
   data is never copied on decode.

The adapter lives in
[`include/tenso_ros/type_adapter.hpp`](include/tenso_ros/type_adapter.hpp). The
`convert_to_ros_message` / `convert_to_custom` hooks only marshal the byte array;
they never re-encode the wire format.

### Lifetime seam to watch

`tenso::Packet` from `TensorView::view()` borrows the body from the
`TensorView`'s internal `std::vector<uint8_t>`. Do not let the `Packet` (or any
`data_as<T>()` pointer) outlive the `TensorView` it came from. The subscriber
skeleton consumes the packet inside the callback, which is the safe pattern.

## Build

```bash
# 1. Build the Tenso C ABI library the shim links against.
cd <tenso-repo-root>
cargo build -p tenso-ffi --release        # produces target/release/libtenso_ffi.*
                                          # and regenerates include/tenso.h

# 2. Build the ROS packages in a colcon workspace (symlink or copy ros/* in).
mkdir -p ~/tenso_ws/src
ln -s <tenso-repo-root>/ros/tenso_msgs ~/tenso_ws/src/
ln -s <tenso-repo-root>/ros/tenso_ros  ~/tenso_ws/src/
cd ~/tenso_ws
colcon build --packages-select tenso_msgs tenso_ros \
  --cmake-args -DTENSO_REPO_ROOT=<tenso-repo-root>
source install/setup.bash
```

`tenso_ros/CMakeLists.txt` discovers `libtenso_ffi` under
`<repo>/target/{release,debug}` (or `$TENSO_FFI_LIB_DIR` /
`-DTENSO_FFI_LIB=...`) and adds `include/` + `cpp/` to the include path.

## "hello tensor" A/B demo — Tenso vs `sensor_msgs/Image`

Goal: show that a self-describing Tenso `TensoBlob` carries arbitrary dtype /
rank tensors with a zero-copy intra-process hop, where `sensor_msgs/Image`
forces you to flatten everything into `uint8[]` + a stringly-typed `encoding`
and (typically) a serialize/deserialize copy across nodes.

### Side A — Tenso

```bash
# Same process/container -> intra-process zero-copy.
ros2 run tenso_ros tensor_publisher &
ros2 run tenso_ros tensor_subscriber
```

The subscriber logs `dtype=float32 ndim=2 nelem=8 first=...`. The dtype, rank,
and shape all came from inside the packet; the transport layer only saw bytes.

### Side B — `sensor_msgs/Image` (for contrast)

A `sensor_msgs/Image` can only describe 2D/3D image-shaped data via `height`,
`width`, `step`, and a string `encoding` (e.g. `"32FC1"`). To send the same
`[1, 8]` float32 tensor you must:

- pick an `encoding` string and pack it as `height=1, width=8, step=32`,
- copy your floats into `data` as raw `uint8[]`,
- and have the consumer re-derive dtype/shape from `encoding` + dims.

Arbitrary rank (e.g. a `[2, 3, 4, 5]` tensor) or non-image dtypes (int64, bf16,
quantized) don't map cleanly at all. Tenso's `TensoBlob` keeps the metadata
*with the data*, decodes zero-copy, and round-trips byte-identically with the
Rust core and the Python library.

### What to compare

| Aspect                    | `TensoBlob` (this pkg)            | `sensor_msgs/Image`            |
|---------------------------|----------------------------------|--------------------------------|
| Arbitrary rank / shape    | yes (in-packet shape)            | image-shaped only              |
| Dtype expressiveness      | full Tenso dtype set incl. bf16/quant | string `encoding` subset  |
| Self-describing payload   | yes                              | partial (dims + encoding)      |
| Intra-process zero-copy   | yes (TypeAdapter + loaned msg)   | possible but copy-prone        |
| Cross-language identity    | byte-identical w/ Rust + Python | n/a                            |

## Status

This is **scaffolding**: the message, the `TypeAdapter`, and runnable
publisher/subscriber skeletons. The encode/transport/decode seams are wired to
the Tenso C core; the integrator should verify the `libtenso_ffi` discovery in
CMake and the intra-process loaned-message behavior on the target ROS2 distro.
