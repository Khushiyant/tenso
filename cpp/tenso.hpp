/*
 * tenso.hpp — Thin RAII C++17 client for the Tenso binary protocol.
 *
 * This header is a *shim* over the stable C ABI declared in <tenso.h>
 * (cbindgen-generated from crates/tenso-ffi). It deliberately does NOT
 * reimplement the wire-format parser/encoder: every read/write call below
 * delegates to the native `tenso_*` functions, so C++ clients always agree
 * byte-for-byte with the Rust core and with Python.
 *
 * Build/link:
 *   - #include "tenso.hpp"  (this header)
 *   - ensure <tenso.h> is on the include path (../include/tenso.h)
 *   - link against the tenso-ffi cdylib/staticlib (e.g. libtenso_ffi)
 *
 * Usage:
 *   #include "tenso.hpp"
 *
 *   // Reading: decode a packet into an owning Packet (frees in dtor).
 *   tenso::Packet pkt = tenso::read(buffer, buffer_size);
 *   const float* data = pkt.data_as<float>();
 *   auto shape        = pkt.shape();   // std::vector<uint32_t>
 *   tenso::DType dt   = pkt.dtype();
 *
 *   // Writing: Mode A (caller-allocates) dense encode via the C ABI.
 *   std::vector<uint8_t> out =
 *       tenso::write(data_ptr, data_bytes, shape, tenso::DType::Float32);
 *
 * Requirements: C++17 or later. Depends only on <tenso.h> + the FFI library.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TENSO_HPP
#define TENSO_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// The C ABI. Adjust the include path in your build if <tenso.h> is not on the
// default search path (the workspace ships it at ../include/tenso.h).
#include "tenso.h"

namespace tenso {

// ---------------------------------------------------------------------------
// Protocol constants (mirrors src/tenso/config.py / crates/tenso-core).
// Kept here for client convenience; parsing itself lives in the C core.
// ---------------------------------------------------------------------------

static constexpr uint8_t MAGIC[4] = {'T', 'N', 'S', 'O'};
static constexpr uint8_t VERSION = 4;
static constexpr size_t  DEFAULT_ALIGNMENT = 64;

// Flag bits (u16 in v4).
static constexpr uint16_t FLAG_ALIGNED     = 1;
static constexpr uint16_t FLAG_INTEGRITY   = 2;
static constexpr uint16_t FLAG_COMPRESSION = 4;
static constexpr uint16_t FLAG_SPARSE      = 8;     // SPARSE_COO
static constexpr uint16_t FLAG_BUNDLE      = 16;
static constexpr uint16_t FLAG_SPARSE_CSR  = 32;
static constexpr uint16_t FLAG_SPARSE_CSC  = 64;
static constexpr uint16_t FLAG_CUST_ALIGN  = 128;
static constexpr uint16_t FLAG_STRING      = 256;
static constexpr uint16_t FLAG_RAGGED      = 512;
static constexpr uint16_t FLAG_GPU_IPC_REF = 1024;  // bit 10 (v4 GPU IpcRef)

// ---------------------------------------------------------------------------
// DType enumeration (codes match the wire format).
// ---------------------------------------------------------------------------

enum class DType : uint8_t {
    Float32    = 1,
    Int32      = 2,
    Float64    = 3,
    Int64      = 4,
    Uint8      = 5,
    Uint16     = 6,
    Bool       = 7,
    Float16    = 8,
    Int8       = 9,
    Int16      = 10,
    Uint32     = 11,
    Uint64     = 12,
    Complex64  = 13,
    Complex128 = 14,
    BFloat16   = 15,
    // Quantized / variable-length codes are decode-only through the C core;
    // they have no fixed item size and are not valid inputs to write().
    QInt8      = 16,
    QUInt8     = 17,
    QInt4      = 18,
    QUInt4     = 19,
    Str        = 20,
    Bytes      = 21,
};

// Item size in bytes for fixed-width dtypes. Returns 0 for variable-width /
// quantized codes (qint4/quint4/str/bytes have no fixed item size).
inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32:    return 4;
        case DType::Int32:      return 4;
        case DType::Float64:    return 8;
        case DType::Int64:      return 8;
        case DType::Uint8:      return 1;
        case DType::Uint16:     return 2;
        case DType::Bool:       return 1;
        case DType::Float16:    return 2;
        case DType::Int8:       return 1;
        case DType::Int16:      return 2;
        case DType::Uint32:     return 4;
        case DType::Uint64:     return 8;
        case DType::Complex64:  return 8;
        case DType::Complex128: return 16;
        case DType::BFloat16:   return 2;
        case DType::QInt8:      return 1;
        case DType::QUInt8:     return 1;
        default:                return 0;  // qint4/quint4/str/bytes: no fixed size
    }
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32:    return "float32";
        case DType::Int32:      return "int32";
        case DType::Float64:    return "float64";
        case DType::Int64:      return "int64";
        case DType::Uint8:      return "uint8";
        case DType::Uint16:     return "uint16";
        case DType::Bool:       return "bool";
        case DType::Float16:    return "float16";
        case DType::Int8:       return "int8";
        case DType::Int16:      return "int16";
        case DType::Uint32:     return "uint32";
        case DType::Uint64:     return "uint64";
        case DType::Complex64:  return "complex64";
        case DType::Complex128: return "complex128";
        case DType::BFloat16:   return "bfloat16";
        case DType::QInt8:      return "qint8";
        case DType::QUInt8:     return "quint8";
        case DType::QInt4:      return "qint4";
        case DType::QUInt4:     return "quint4";
        case DType::Str:        return "string";
        case DType::Bytes:      return "bytes";
        default:                return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Error type — carries the thread-local message from tenso_last_error().
// ---------------------------------------------------------------------------

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {

// Pull the most recent error string from the C core (thread-local), falling
// back to a generic message when the core reported nothing.
inline std::string last_error(const char* fallback) {
    const char* msg = tenso_last_error();
    if (msg != nullptr && msg[0] != '\0') return std::string(msg);
    return std::string(fallback);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Packet — an owning RAII wrapper around an opaque TensoView*.
//
// Decoding is performed by the C core (`tenso_decode`); this class only owns
// the resulting handle and exposes typed, zero-copy accessors over it. The
// destructor calls `tenso_view_free` exactly once. Move-only (no copy) so the
// underlying handle has a single owner.
//
// IMPORTANT: the body bytes are borrowed from the source buffer passed to
// read()/decode(). That buffer must outlive the Packet.
// ---------------------------------------------------------------------------

class Packet {
public:
    Packet() = default;

    explicit Packet(::TensoView* view) : view_(view) {}

    // Move-only.
    Packet(const Packet&)            = delete;
    Packet& operator=(const Packet&) = delete;

    Packet(Packet&& other) noexcept : view_(other.view_) {
        other.view_ = nullptr;
    }
    Packet& operator=(Packet&& other) noexcept {
        if (this != &other) {
            reset();
            view_       = other.view_;
            other.view_ = nullptr;
        }
        return *this;
    }

    ~Packet() { reset(); }

    bool valid() const { return view_ != nullptr; }
    explicit operator bool() const { return valid(); }

    // Raw opaque handle (for advanced interop). Ownership stays with Packet.
    const ::TensoView* handle() const { return view_; }

    DType dtype() const {
        ensure();
        return static_cast<DType>(tenso_view_dtype(view_));
    }

    size_t ndim() const {
        ensure();
        return static_cast<size_t>(tenso_view_ndim(view_));
    }

    std::vector<uint32_t> shape() const {
        ensure();
        const uint32_t* s = tenso_view_shape(view_);
        size_t n          = static_cast<size_t>(tenso_view_ndim(view_));
        if (s == nullptr) return {};
        return std::vector<uint32_t>(s, s + n);
    }

    size_t num_elements() const {
        ensure();
        const uint32_t* s = tenso_view_shape(view_);
        size_t n          = static_cast<size_t>(tenso_view_ndim(view_));
        if (s == nullptr || n == 0) return 0;
        size_t prod = 1;
        for (size_t i = 0; i < n; ++i) prod *= static_cast<size_t>(s[i]);
        return prod;
    }

    // Pointer to the body bytes (borrowed from the source buffer).
    const uint8_t* body_ptr() const {
        ensure();
        return tenso_view_body_ptr(view_);
    }

    size_t body_bytes() const {
        ensure();
        return static_cast<size_t>(tenso_view_body_len(view_));
    }

    // Typed, zero-copy view of the body. No alignment/dtype validation beyond
    // what the core already performed; caller must request the right T.
    template <typename T>
    const T* data_as() const {
        return reinterpret_cast<const T*>(body_ptr());
    }

    // Copy the body into a caller-provided buffer.
    void copy_to(void* dest, size_t dest_size) const {
        size_t n = body_bytes();
        if (dest_size < n) throw Error("tenso: destination buffer too small");
        const uint8_t* src = body_ptr();
        for (size_t i = 0; i < n; ++i)
            static_cast<uint8_t*>(dest)[i] = src[i];
    }

    // Explicitly free the underlying view early.
    void reset() {
        if (view_ != nullptr) {
            tenso_view_free(view_);
            view_ = nullptr;
        }
    }

private:
    void ensure() const {
        if (view_ == nullptr) throw Error("tenso: operation on an empty Packet");
    }

    ::TensoView* view_ = nullptr;
};

// ---------------------------------------------------------------------------
// read()/decode() — decode a packet via the C core into an owning Packet.
//
// The body is borrowed from `buf`; keep `buf` alive while the Packet is used.
// Throws tenso::Error on any decode failure (NULL view), surfacing the
// core's thread-local error message.
// ---------------------------------------------------------------------------

inline Packet decode(const uint8_t* buf, size_t len) {
    ::TensoView* v = tenso_decode(buf, static_cast<uintptr_t>(len));
    if (v == nullptr) throw Error(detail::last_error("tenso: decode failed"));
    return Packet(v);
}

inline Packet read(const uint8_t* buf, size_t len) { return decode(buf, len); }

inline Packet read(const std::vector<uint8_t>& buf) {
    return decode(buf.data(), buf.size());
}

// ---------------------------------------------------------------------------
// parse_header() — thin wrapper over tenso_parse_header (no body copy).
// ---------------------------------------------------------------------------

inline ::TensoHeader parse_header(const uint8_t* buf, size_t len) {
    ::TensoHeader h{};
    int rc = tenso_parse_header(buf, static_cast<uintptr_t>(len), &h);
    if (rc != TENSO_OK)
        throw Error(detail::last_error("tenso: parse_header failed"));
    return h;
}

// ---------------------------------------------------------------------------
// write() — serialize a dense array via the C core (Mode A: caller-allocates).
//
// Two-call protocol: query required size, then encode into a sized buffer.
// All wire-format work happens in the native core; this only marshals args.
// ---------------------------------------------------------------------------

inline std::vector<uint8_t> write(const void*                  data,
                                  size_t                       data_bytes,
                                  const std::vector<uint32_t>& shape,
                                  DType                        dtype,
                                  size_t                       alignment = DEFAULT_ALIGNMENT,
                                  bool                         check_integrity = false,
                                  bool                         compress = false) {
    const uint8_t* data_ptr = static_cast<const uint8_t*>(data);
    const uint32_t* shape_ptr = shape.empty() ? nullptr : shape.data();
    uintptr_t       ndim      = static_cast<uintptr_t>(shape.size());

    uintptr_t needed = 0;
    int rc = tenso_dense_required_size(data_ptr,
                                       static_cast<uintptr_t>(data_bytes),
                                       static_cast<uint8_t>(dtype),
                                       shape_ptr,
                                       ndim,
                                       check_integrity,
                                       compress,
                                       static_cast<uintptr_t>(alignment),
                                       &needed);
    if (rc != TENSO_OK)
        throw Error(detail::last_error("tenso: dense_required_size failed"));

    std::vector<uint8_t> out(static_cast<size_t>(needed));
    uintptr_t written = 0;
    rc = tenso_encode_dense_into(data_ptr,
                                 static_cast<uintptr_t>(data_bytes),
                                 static_cast<uint8_t>(dtype),
                                 shape_ptr,
                                 ndim,
                                 check_integrity,
                                 compress,
                                 static_cast<uintptr_t>(alignment),
                                 out.data(),
                                 static_cast<uintptr_t>(out.size()),
                                 &written);
    if (rc != TENSO_OK)
        throw Error(detail::last_error("tenso: encode_dense_into failed"));

    out.resize(static_cast<size_t>(written));
    return out;
}

// Convenience overload: append into a caller-owned vector (clears it first).
inline void write(std::vector<uint8_t>&        out,
                  const void*                  data,
                  size_t                       data_bytes,
                  const std::vector<uint32_t>& shape,
                  DType                        dtype,
                  size_t                       alignment = DEFAULT_ALIGNMENT,
                  bool                         check_integrity = false,
                  bool                         compress = false) {
    out = write(data, data_bytes, shape, dtype, alignment, check_integrity, compress);
}

}  // namespace tenso

#endif  // TENSO_HPP
