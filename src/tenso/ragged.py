"""
String Tensor and Ragged Array support for Tenso.

Provides efficient serialization of variable-length string batches
and ragged/jagged arrays without padding, suitable for NLP pipelines
and dynamic batching in LLM inference.

String Tensor Format:
    Header (8 bytes) + offsets array (n+1 uint64) + packed UTF-8 data

Ragged Array Format:
    Header (8 bytes) + shape_0 (4 bytes) + offsets (shape_0+1 uint64)
    + flat values as dense Tenso packet
"""

import struct
from typing import List, Optional, Sequence, Union

import numpy as np
import xxhash

from .config import (
    _ALIGNMENT,
    _MAGIC,
    _VERSION,
    DTYPE_BYTES,
    DTYPE_STRING,
    FLAG_ALIGNED,
    FLAG_INTEGRITY,
    FLAG_RAGGED,
    FLAG_STRING,
)


class StringTensor:
    """
    A batch of variable-length strings stored as packed UTF-8 with offsets.

    This is a drop-in replacement for object-dtype numpy arrays of strings,
    but serializes compactly without padding.

    Example::

        st = StringTensor(["hello", "world", "foo"])
        packet = st.dumps()
        restored = StringTensor.loads(packet)
        assert restored[0] == "hello"
        assert len(restored) == 3
    """

    __slots__ = ("_offsets", "_data", "_count")

    def __init__(self, strings: Sequence[Union[str, bytes]]):
        parts = []
        offsets = [0]
        for s in strings:
            encoded = s.encode("utf-8") if isinstance(s, str) else s
            parts.append(encoded)
            offsets.append(offsets[-1] + len(encoded))
        self._offsets = np.array(offsets, dtype=np.uint64)
        self._data = b"".join(parts)
        self._count = len(strings)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx: int) -> str:
        if idx < 0:
            idx += self._count
        if idx < 0 or idx >= self._count:
            raise IndexError(f"index {idx} out of range for StringTensor of size {self._count}")
        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])
        return self._data[start:end].decode("utf-8")

    def to_list(self) -> List[str]:
        result = []
        for i in range(self._count):
            start = int(self._offsets[i])
            end = int(self._offsets[i + 1])
            result.append(self._data[start:end].decode("utf-8"))
        return result

    @property
    def nbytes(self) -> int:
        return len(self._data)

    @property
    def shape(self) -> tuple:
        return (self._count,)

    def dumps(self, check_integrity: bool = False) -> memoryview:
        """Serialize to a Tenso packet."""
        offsets_bytes = self._offsets.tobytes()
        data_bytes = self._data

        flags = FLAG_STRING | FLAG_ALIGNED
        if check_integrity:
            flags |= FLAG_INTEGRITY

        # Header(8) + ndim=1 shape(4) + offsets + data
        header_len = 8 + 4  # 8-byte header + 1 dim (4 bytes)
        padding_len = (_ALIGNMENT - (header_len % _ALIGNMENT)) % _ALIGNMENT
        body = offsets_bytes + data_bytes
        footer_len = 8 if check_integrity else 0
        total = header_len + padding_len + len(body) + footer_len

        buf = bytearray(total)
        # Encode flags as two bytes if >255, using high byte in dtype_code position
        flags_lo = flags & 0xFF
        flags_hi = (flags >> 8) & 0xFF
        struct.pack_into("<4sBBBB", buf, 0, _MAGIC, _VERSION, flags_lo, DTYPE_STRING | (flags_hi << 4), 1)
        struct.pack_into("<I", buf, 8, self._count)

        body_start = header_len + padding_len
        buf[body_start:body_start + len(body)] = body

        if check_integrity:
            digest = xxhash.xxh3_64_intdigest(body)
            struct.pack_into("<Q", buf, body_start + len(body), digest)

        return memoryview(buf)

    @classmethod
    def loads(cls, data: Union[bytes, bytearray, memoryview]) -> "StringTensor":
        """Deserialize from a Tenso packet."""
        mv = memoryview(data)
        if len(mv) < 12:
            raise ValueError("StringTensor packet too short")

        magic = bytes(mv[0:4])
        if magic != _MAGIC:
            raise ValueError("Invalid tenso packet")

        flags_byte = mv[5]
        count = struct.unpack("<I", mv[8:12])[0]

        header_len = 12
        padding_len = (_ALIGNMENT - (header_len % _ALIGNMENT)) % _ALIGNMENT
        body_start = header_len + padding_len

        offsets_size = (count + 1) * 8
        offsets = np.frombuffer(mv[body_start:body_start + offsets_size], dtype=np.uint64).copy()
        data_start = body_start + offsets_size
        total_str_bytes = int(offsets[-1])
        raw_data = bytes(mv[data_start:data_start + total_str_bytes])

        obj = object.__new__(cls)
        obj._offsets = offsets
        obj._data = raw_data
        obj._count = count
        return obj

    def __repr__(self) -> str:
        preview = self.to_list()[:3]
        suffix = ", ..." if self._count > 3 else ""
        return f"StringTensor({preview}{suffix}, count={self._count})"


class RaggedArray:
    """
    A ragged (jagged) array: a sequence of variable-length 1-D arrays
    stored without padding.

    Useful for dynamic batching in LLM inference where sequence lengths vary.

    Example::

        ra = RaggedArray([
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0]),
            np.array([6.0]),
        ])
        packet = ra.dumps()
        restored = RaggedArray.loads(packet)
        assert len(restored) == 3
        assert list(restored[1]) == [4.0, 5.0]
    """

    __slots__ = ("_offsets", "_flat", "_count")

    def __init__(self, arrays: Sequence[np.ndarray]):
        if not arrays:
            self._offsets = np.array([0], dtype=np.uint64)
            self._flat = np.array([], dtype=np.float32)
            self._count = 0
            return

        dtype = arrays[0].dtype
        offsets = [0]
        parts = []
        for arr in arrays:
            arr = np.asarray(arr).ravel()
            if arr.dtype != dtype:
                arr = arr.astype(dtype)
            parts.append(arr)
            offsets.append(offsets[-1] + len(arr))

        self._offsets = np.array(offsets, dtype=np.uint64)
        self._flat = np.concatenate(parts)
        self._count = len(arrays)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self._count
        if idx < 0 or idx >= self._count:
            raise IndexError(f"index {idx} out of range for RaggedArray of size {self._count}")
        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])
        return self._flat[start:end].copy()

    @property
    def dtype(self) -> np.dtype:
        return self._flat.dtype

    @property
    def flat_values(self) -> np.ndarray:
        return self._flat

    @property
    def row_splits(self) -> np.ndarray:
        return self._offsets

    def to_list(self) -> List[np.ndarray]:
        result = []
        for i in range(self._count):
            start = int(self._offsets[i])
            end = int(self._offsets[i + 1])
            result.append(self._flat[start:end].copy())
        return result

    def dumps(self, check_integrity: bool = False) -> memoryview:
        """Serialize to a Tenso packet using bundle format internally."""
        from .core import dumps as core_dumps

        bundle = {
            "__ragged_offsets__": self._offsets,
            "__ragged_flat__": self._flat,
            "__ragged_count__": np.array([self._count], dtype=np.uint64),
        }
        return core_dumps(bundle, check_integrity=check_integrity)

    @classmethod
    def loads(cls, data: Union[bytes, bytearray, memoryview, dict]) -> "RaggedArray":
        """Deserialize from a Tenso packet or already-deserialized dict."""
        if not isinstance(data, dict):
            from .core import loads as core_loads
            data = core_loads(data)

        if not isinstance(data, dict) or "__ragged_offsets__" not in data:
            raise ValueError("Not a valid RaggedArray packet")

        offsets = np.asarray(data["__ragged_offsets__"], dtype=np.uint64)
        flat = np.asarray(data["__ragged_flat__"])
        count = int(data["__ragged_count__"][0])

        obj = object.__new__(cls)
        obj._offsets = offsets
        obj._flat = flat
        obj._count = count
        return obj

    def __repr__(self) -> str:
        return (
            f"RaggedArray(count={self._count}, dtype={self._flat.dtype}, "
            f"total_elements={len(self._flat)})"
        )
