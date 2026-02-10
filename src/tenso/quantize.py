"""
Quantized Tensor support for Tenso.

Provides QuantizedTensor for 4-bit and 8-bit quantized representations
with per-tensor, per-channel, and per-group quantization schemes.
"""

import math

import numpy as np

from .config import (
    QDTYPE_QINT4,
    QDTYPE_QINT8,
    QDTYPE_QUINT4,
    QUANT_PER_CHANNEL,
    QUANT_PER_GROUP,
    QUANT_PER_TENSOR,
    _QDTYPE_FROM_NAME,
    _QDTYPE_NAMES,
)


def _compute_scale_zp(data: np.ndarray, qmin: int, qmax: int):
    """Compute min-max affine quantization scale and zero point."""
    dmin = float(data.min())
    dmax = float(data.max())
    if dmin == dmax:
        return np.float32(1.0), np.float32(0.0)
    scale = np.float32((dmax - dmin) / (qmax - qmin))
    zero_point = np.float32(qmin - dmin / scale)
    zero_point = np.float32(np.clip(zero_point, qmin, qmax))
    return scale, zero_point


def _pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack two 4-bit values per byte. Lower nibble = even index, upper nibble = odd index."""
    n = len(values)
    num_bytes = math.ceil(n / 2)
    packed = np.zeros(num_bytes, dtype=np.uint8)
    # Mask to 4 bits
    masked = values.astype(np.uint8) & 0x0F
    # Even indices go to lower nibble
    packed[: n // 2] = masked[0::2][: n // 2]
    # Odd indices go to upper nibble
    packed[: n // 2] |= masked[1::2][: n // 2].astype(np.uint8) << 4
    # Handle last element if odd count
    if n % 2 == 1:
        packed[-1] = masked[-1]
    return packed


def _unpack_int4(packed: np.ndarray, num_elements: int, signed: bool) -> np.ndarray:
    """Unpack 4-bit values from packed bytes."""
    result = np.empty(num_elements, dtype=np.int8 if signed else np.uint8)
    n_pairs = num_elements // 2
    if n_pairs > 0:
        # Lower nibble = even indices
        result[0 : 2 * n_pairs : 2] = (packed[:n_pairs] & 0x0F).astype(result.dtype)
        # Upper nibble = odd indices
        result[1 : 2 * n_pairs : 2] = ((packed[:n_pairs] >> 4) & 0x0F).astype(
            result.dtype
        )
    if num_elements % 2 == 1:
        result[-1] = packed[-1] & 0x0F
    if signed:
        # Sign-extend: if bit 3 is set, value is negative (two's complement in 4 bits)
        mask = result > 7
        result[mask] = result[mask] - 16
    return result


class QuantizedTensor:
    """A quantized tensor with scale/zero_point metadata."""

    def __init__(
        self,
        data: np.ndarray,
        scales: np.ndarray,
        zero_points: np.ndarray,
        shape: tuple,
        dtype_code: int,
        quant_scheme: int = QUANT_PER_TENSOR,
        group_size: int = 0,
    ):
        self.data = data
        self.scales = np.asarray(scales, dtype=np.float32).ravel()
        self.zero_points = np.asarray(zero_points, dtype=np.float32).ravel()
        self.shape = tuple(shape)
        self.dtype_code = dtype_code
        self.quant_scheme = quant_scheme
        self.group_size = group_size

    @property
    def dtype_name(self) -> str:
        return _QDTYPE_NAMES[self.dtype_code]

    @property
    def is_4bit(self) -> bool:
        return self.dtype_code in (QDTYPE_QINT4, QDTYPE_QUINT4)

    @property
    def is_signed(self) -> bool:
        return self.dtype_code in (QDTYPE_QINT8, QDTYPE_QINT4)

    @property
    def nbytes(self) -> int:
        n = int(np.prod(self.shape))
        if self.is_4bit:
            return math.ceil(n / 2)
        return n

    @classmethod
    def quantize(
        cls,
        tensor: np.ndarray,
        dtype: str,
        scheme: str = "per_tensor",
        group_size: int = 0,
        axis: int = 0,
    ) -> "QuantizedTensor":
        """Quantize a float tensor.

        Parameters
        ----------
        tensor : np.ndarray
            Input tensor (will be converted to float32).
        dtype : str
            Target quantized dtype name: "qint8", "quint8", "qint4", "quint4".
        scheme : str
            Quantization scheme: "per_tensor", "per_channel", "per_group".
        group_size : int
            Group size for per_group scheme.
        axis : int
            Channel axis for per_channel scheme.
        """
        dtype_code = _QDTYPE_FROM_NAME[dtype]
        is_4bit = dtype_code in (QDTYPE_QINT4, QDTYPE_QUINT4)
        is_signed = dtype_code in (QDTYPE_QINT8, QDTYPE_QINT4)

        if is_4bit:
            qmin, qmax = (-8, 7) if is_signed else (0, 15)
        else:
            qmin, qmax = (-128, 127) if is_signed else (0, 255)

        scheme_map = {
            "per_tensor": QUANT_PER_TENSOR,
            "per_channel": QUANT_PER_CHANNEL,
            "per_group": QUANT_PER_GROUP,
        }
        quant_scheme = scheme_map[scheme]

        float_data = tensor.astype(np.float32)
        shape = tensor.shape
        flat = float_data.ravel()

        if quant_scheme == QUANT_PER_TENSOR:
            scale, zp = _compute_scale_zp(flat, qmin, qmax)
            scales = np.array([scale], dtype=np.float32)
            zero_points = np.array([zp], dtype=np.float32)
            quantized = np.clip(np.round(flat / scale + zp), qmin, qmax).astype(
                np.int8 if is_signed else np.uint8
            )

        elif quant_scheme == QUANT_PER_CHANNEL:
            num_channels = shape[axis]
            scales = np.empty(num_channels, dtype=np.float32)
            zero_points = np.empty(num_channels, dtype=np.float32)
            quantized = np.empty(flat.shape, dtype=np.int8 if is_signed else np.uint8)

            # Move axis to front for easier slicing
            moved = np.moveaxis(float_data, axis, 0)
            moved_flat = moved.reshape(num_channels, -1)
            q_flat = np.empty_like(moved_flat, dtype=np.int8 if is_signed else np.uint8)

            for ch in range(num_channels):
                s, z = _compute_scale_zp(moved_flat[ch], qmin, qmax)
                scales[ch] = s
                zero_points[ch] = z
                q_flat[ch] = np.clip(
                    np.round(moved_flat[ch] / s + z), qmin, qmax
                ).astype(q_flat.dtype)

            # Move axis back and flatten
            q_shaped = q_flat.reshape(moved.shape)
            q_original = np.moveaxis(q_shaped, 0, axis)
            quantized = q_original.ravel()

        elif quant_scheme == QUANT_PER_GROUP:
            if group_size <= 0:
                raise ValueError("group_size must be > 0 for per_group scheme")
            n = len(flat)
            num_groups = math.ceil(n / group_size)
            scales = np.empty(num_groups, dtype=np.float32)
            zero_points = np.empty(num_groups, dtype=np.float32)
            quantized = np.empty(n, dtype=np.int8 if is_signed else np.uint8)

            for g in range(num_groups):
                start = g * group_size
                end = min(start + group_size, n)
                group = flat[start:end]
                s, z = _compute_scale_zp(group, qmin, qmax)
                scales[g] = s
                zero_points[g] = z
                quantized[start:end] = np.clip(
                    np.round(group / s + z), qmin, qmax
                ).astype(quantized.dtype)

        if is_4bit:
            packed = _pack_int4(quantized)
        else:
            packed = quantized.view(np.uint8)

        return cls(
            data=packed,
            scales=scales,
            zero_points=zero_points,
            shape=shape,
            dtype_code=dtype_code,
            quant_scheme=quant_scheme,
            group_size=group_size,
        )

    def dequantize(self) -> np.ndarray:
        """Reconstruct a float32 approximation of the original tensor."""
        num_elements = int(np.prod(self.shape))
        is_signed = self.is_signed

        if self.is_4bit:
            values = _unpack_int4(self.data, num_elements, is_signed).astype(
                np.float32
            )
        else:
            raw = self.data.view(np.int8 if is_signed else np.uint8)
            values = raw[:num_elements].astype(np.float32)

        if self.quant_scheme == QUANT_PER_TENSOR:
            result = (values - self.zero_points[0]) * self.scales[0]

        elif self.quant_scheme == QUANT_PER_CHANNEL:
            # Need to figure out which axis was used — we use axis=0 convention
            # since shape is preserved
            num_channels = len(self.scales)
            result = np.empty_like(values)
            channel_size = num_elements // num_channels
            reshaped = values.reshape(num_channels, channel_size)
            for ch in range(num_channels):
                reshaped[ch] = (reshaped[ch] - self.zero_points[ch]) * self.scales[ch]
            result = reshaped.ravel()

        elif self.quant_scheme == QUANT_PER_GROUP:
            result = np.empty_like(values)
            for g in range(len(self.scales)):
                start = g * self.group_size
                end = min(start + self.group_size, num_elements)
                result[start:end] = (
                    values[start:end] - self.zero_points[g]
                ) * self.scales[g]

        return result.reshape(self.shape)

    def __repr__(self) -> str:
        return (
            f"QuantizedTensor(dtype={self.dtype_name}, shape={self.shape}, "
            f"scheme={self.quant_scheme}, nbytes={self.nbytes})"
        )
