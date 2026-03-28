/*
 * tenso.hpp — Header-only C++ client for the Tenso binary protocol.
 *
 * Enables C++ inference engines (TensorRT, ONNXRuntime, LibTorch) to
 * read and write Tenso packets without any Python dependency.
 *
 * Usage:
 *   #include "tenso.hpp"
 *
 *   // Reading a packet from a buffer
 *   tenso::Packet pkt = tenso::read(buffer, buffer_size);
 *   float* data = pkt.data_as<float>();
 *   // shape: pkt.shape(), dtype: pkt.dtype()
 *
 *   // Writing a packet to a buffer
 *   std::vector<uint8_t> out;
 *   tenso::write(out, data_ptr, shape, tenso::DType::Float32);
 *
 * Requirements: C++17 or later. No external dependencies.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TENSO_HPP
#define TENSO_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenso {

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

static constexpr uint8_t MAGIC[4] = {'T', 'N', 'S', 'O'};
static constexpr uint8_t VERSION = 3;
static constexpr size_t DEFAULT_ALIGNMENT = 64;

// Flag bits
static constexpr uint8_t FLAG_ALIGNED     = 1;
static constexpr uint8_t FLAG_INTEGRITY   = 2;
static constexpr uint8_t FLAG_COMPRESSION = 4;
static constexpr uint8_t FLAG_SPARSE      = 8;
static constexpr uint8_t FLAG_BUNDLE      = 16;
static constexpr uint8_t FLAG_SPARSE_CSR  = 32;
static constexpr uint8_t FLAG_SPARSE_CSC  = 64;
static constexpr uint8_t FLAG_CUST_ALIGN  = 128;

// ---------------------------------------------------------------------------
// DType enumeration
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
};

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
        default:
            throw std::runtime_error("Unknown dtype code: " +
                                     std::to_string(static_cast<int>(dt)));
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
        default:                return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Packet — a parsed view into a Tenso buffer (zero-copy for dense)
// ---------------------------------------------------------------------------

struct Packet {
    DType                    dtype_val;
    uint8_t                  flags;
    uint8_t                  version;
    std::vector<uint32_t>    shape_vec;
    const uint8_t*           body_ptr;    // points into the source buffer
    size_t                   body_bytes;
    bool                     integrity_ok;

    DType dtype() const { return dtype_val; }

    const std::vector<uint32_t>& shape() const { return shape_vec; }

    size_t ndim() const { return shape_vec.size(); }

    size_t num_elements() const {
        if (shape_vec.empty()) return 0;
        size_t n = 1;
        for (auto d : shape_vec) n *= d;
        return n;
    }

    /// Return a typed pointer to the body data.
    template <typename T>
    const T* data_as() const {
        return reinterpret_cast<const T*>(body_ptr);
    }

    /// Copy the body into a caller-provided buffer.
    void copy_to(void* dest, size_t dest_size) const {
        if (dest_size < body_bytes)
            throw std::runtime_error("Destination buffer too small");
        std::memcpy(dest, body_ptr, body_bytes);
    }
};

// ---------------------------------------------------------------------------
// read() — parse a Tenso dense packet from a byte buffer
// ---------------------------------------------------------------------------

inline Packet read(const uint8_t* buf, size_t len) {
    if (len < 8)
        throw std::runtime_error("Tenso packet too short");

    if (std::memcmp(buf, MAGIC, 4) != 0)
        throw std::runtime_error("Invalid Tenso magic");

    Packet pkt{};
    pkt.version   = buf[4];
    pkt.flags     = buf[5];
    pkt.dtype_val = static_cast<DType>(buf[6]);
    uint8_t ndim  = buf[7];

    // Bundles, sparse, and compressed packets need special handling
    if (pkt.flags & (FLAG_BUNDLE | FLAG_SPARSE | FLAG_SPARSE_CSR |
                     FLAG_SPARSE_CSC | FLAG_COMPRESSION))
        throw std::runtime_error(
            "tenso.hpp read() only supports dense uncompressed packets. "
            "Use the Python library for bundles, sparse, or compressed data.");

    size_t cursor = 8;
    if (cursor + ndim * 4 > len)
        throw std::runtime_error("Packet too short for shape");

    pkt.shape_vec.resize(ndim);
    for (uint8_t i = 0; i < ndim; ++i) {
        uint32_t dim;
        std::memcpy(&dim, buf + cursor, 4);
        pkt.shape_vec[i] = dim;
        cursor += 4;
    }

    // Alignment
    size_t alignment = 1;
    if (pkt.flags & FLAG_CUST_ALIGN) {
        if (cursor >= len)
            throw std::runtime_error("Missing alignment byte");
        alignment = size_t(1) << buf[cursor];
        cursor += 1;
    } else if (pkt.flags & FLAG_ALIGNED) {
        alignment = DEFAULT_ALIGNMENT;
    }

    size_t header_len = cursor;
    size_t remainder = header_len % alignment;
    size_t padding = (remainder == 0) ? 0 : (alignment - remainder);
    size_t body_start = header_len + padding;

    size_t item_sz = dtype_size(pkt.dtype_val);
    size_t body_len = pkt.num_elements() * item_sz;
    size_t footer_len = (pkt.flags & FLAG_INTEGRITY) ? 8 : 0;

    if (body_start + body_len + footer_len > len)
        throw std::runtime_error("Packet too short for body");

    pkt.body_ptr = buf + body_start;
    pkt.body_bytes = body_len;
    pkt.integrity_ok = true; // Integrity verification left to caller if needed

    return pkt;
}

// ---------------------------------------------------------------------------
// write() — serialize a dense array into a Tenso packet
// ---------------------------------------------------------------------------

inline void write(
    std::vector<uint8_t>& out,
    const void* data,
    const std::vector<uint32_t>& shape,
    DType dtype,
    size_t alignment = DEFAULT_ALIGNMENT,
    bool check_integrity = false
) {
    uint8_t ndim = static_cast<uint8_t>(shape.size());
    size_t num_elements = 1;
    for (auto d : shape) num_elements *= d;
    size_t body_len = num_elements * dtype_size(dtype);

    bool custom_align = (alignment != 64);
    size_t header_len = 8 + ndim * 4 + (custom_align ? 1 : 0);
    size_t remainder = header_len % alignment;
    size_t padding = (remainder == 0) ? 0 : (alignment - remainder);
    size_t footer_len = check_integrity ? 8 : 0;
    size_t total = header_len + padding + body_len + footer_len;

    out.resize(total, 0);
    uint8_t* buf = out.data();

    // Header
    std::memcpy(buf, MAGIC, 4);
    buf[4] = VERSION;

    uint8_t flags = 0;
    if (custom_align) flags |= FLAG_CUST_ALIGN;
    else              flags |= FLAG_ALIGNED;
    if (check_integrity) flags |= FLAG_INTEGRITY;
    buf[5] = flags;

    buf[6] = static_cast<uint8_t>(dtype);
    buf[7] = ndim;

    size_t cursor = 8;
    for (uint8_t i = 0; i < ndim; ++i) {
        std::memcpy(buf + cursor, &shape[i], 4);
        cursor += 4;
    }

    if (custom_align) {
        // trailing_zeros equivalent
        uint8_t exponent = 0;
        size_t a = alignment;
        while (a > 1) { a >>= 1; exponent++; }
        buf[cursor] = exponent;
    }

    // Body
    size_t body_start = header_len + padding;
    std::memcpy(buf + body_start, data, body_len);

    // Integrity footer (XXH3 not included — caller can append externally)
    // For a fully self-contained implementation, integrate xxHash C header.
}

// ---------------------------------------------------------------------------
// Convenience: read from a std::vector
// ---------------------------------------------------------------------------

inline Packet read(const std::vector<uint8_t>& buf) {
    return read(buf.data(), buf.size());
}

} // namespace tenso

#endif // TENSO_HPP
