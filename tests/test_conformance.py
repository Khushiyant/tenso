"""
Conformance tests for the Tenso binary wire format.

Each ``.tenso`` fixture under ``tests/fixtures/`` is a frozen snapshot of
the encoder output for a specific input.  We assert two things:

1. The fixture decodes back to the exact original value (dtype, shape, data).
2. The fixture's SHA-256 still matches the expected value.

The SHA-256 check is the canary for accidental wire-format drift: if a
refactor changes the encoder's byte layout, every downstream consumer of
old packets breaks, so we want CI to catch it immediately.

To regenerate fixtures (after an *intentional* wire-format change), run::

    .venv/bin/python tests/fixtures/_generate.py

and copy the new ``EXPECTED_SHA256`` dict it prints into this file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import tenso
from tenso import StringTensor


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# SHA-256 of each fixture file, captured from `_generate.py` output.
# DO NOT edit by hand — re-run the generator if you change the encoder.
EXPECTED_SHA256 = {
    "dense_f32_vec.tenso":
        "ba5f47405c10351b48a59d4a4f46d73b9d5754bdea85488f0a93c114be0199c6",
    "dense_f64_mat.tenso":
        "85276d4eae66333c21f617f428bab6ac841de39421f6bd1ca39df402d0658785",
    "dense_i32_integrity.tenso":
        "a31de660a53dded66288457cfe0e74d81d9ec3481d357c3ce3de949701d82870",
    "dense_u8_3d_zeros.tenso":
        "5c3bae036d3f545d3e10064ba48d9bf3699d2ef17b46b0cf790baf1690bea0f5",
    "dense_bf16_vec.tenso":
        "fc9615243289d8a3819f6f0a0494898f788287163800208169c8b982e45ac87f",
    "bundle_mixed.tenso":
        "848d656444bb1d8ebdd921667a17d75d42d27f3b01d13a2df646cc7b6c6e9ed5",
    "string_mixed_utf8.tenso":
        "a7d2668fe1672083c47e5714a2ff45318fff252c79952ad06391f2835b008c30",
    "dense_f64_compressed.tenso":
        "9608679ed6964e6933d2dc1abc688fe0323f792cb9d5365db9e3e36890f59fa6",
    "quantized_qint8_per_tensor.tenso":
        "db8d4e0ed8e583f6a84c2fb1de0c6fdb622b52a447fe243f9087409150605c1c",
    "sparse_coo_f32.tenso":
        "b909f9fdf4c3054f3983aaf376fab0a6eb343716c5c7555f784dabd360cf612d",
}


def _read_fixture(name: str) -> bytes:
    path = FIXTURES_DIR / name
    assert path.exists(), f"missing fixture: {path}"
    data = path.read_bytes()
    # Every Tenso packet starts with the magic 'TNSO'.
    assert data[:4] == b"TNSO", f"{name!r} doesn't start with TNSO magic"
    return data


def _assert_sha256(name: str, data: bytes) -> None:
    expected = EXPECTED_SHA256.get(name)
    assert expected is not None, f"no expected SHA-256 recorded for {name!r}"
    actual = hashlib.sha256(data).hexdigest()
    assert actual == expected, (
        f"wire-format drift for {name!r}: expected {expected}, got {actual}. "
        "If this change is intentional, re-run tests/fixtures/_generate.py "
        "and update EXPECTED_SHA256."
    )


def test_dense_f32_vec():
    name = "dense_f32_vec.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    arr = tenso.loads(data)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    assert arr.dtype == np.float32
    assert arr.shape == (5,)
    np.testing.assert_array_equal(arr, expected)


def test_dense_f64_mat():
    name = "dense_f64_mat.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    arr = tenso.loads(data)
    expected = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert arr.dtype == np.float64
    assert arr.shape == (3, 4)
    np.testing.assert_array_equal(arr, expected)


def test_dense_i32_integrity():
    name = "dense_i32_integrity.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    arr = tenso.loads(data)
    expected = np.arange(8, dtype=np.int32)
    assert arr.dtype == np.int32
    assert arr.shape == (8,)
    np.testing.assert_array_equal(arr, expected)


def test_dense_u8_3d_zeros():
    name = "dense_u8_3d_zeros.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    arr = tenso.loads(data)
    expected = np.zeros((4, 4, 4), dtype=np.uint8)
    assert arr.dtype == np.uint8
    assert arr.shape == (4, 4, 4)
    np.testing.assert_array_equal(arr, expected)


def test_dense_bf16_vec():
    name = "dense_bf16_vec.tenso"
    if not (FIXTURES_DIR / name).exists():
        pytest.skip("bfloat16 fixture absent (ml_dtypes was missing when generated)")

    try:
        from ml_dtypes import bfloat16  # type: ignore
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    data = _read_fixture(name)
    _assert_sha256(name, data)

    arr = tenso.loads(data)
    expected = np.array([1.0, -1.5, 0.25, 3.75], dtype=bfloat16)
    assert arr.dtype == np.dtype(bfloat16)
    assert arr.shape == (4,)
    np.testing.assert_array_equal(arr, expected)


def test_bundle_mixed():
    name = "bundle_mixed.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    obj = tenso.loads(data)
    assert isinstance(obj, dict)
    assert set(obj.keys()) == {"a", "b"}

    assert obj["a"].dtype == np.float32
    assert obj["a"].shape == (2,)
    np.testing.assert_array_equal(obj["a"], np.array([1.0, 2.0], dtype=np.float32))

    assert obj["b"].dtype == np.int64
    assert obj["b"].shape == (3,)
    np.testing.assert_array_equal(obj["b"], np.array([10, 20, 30], dtype=np.int64))


def test_string_mixed_utf8():
    name = "string_mixed_utf8.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    # StringTensor packets use a dedicated reader.
    st = StringTensor.loads(data)
    assert isinstance(st, StringTensor)
    assert len(st) == 4
    assert st.to_list() == ["hi", "", "world", "ñ"]


def test_dense_f64_compressed():
    name = "dense_f64_compressed.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    arr = tenso.loads(data)
    expected = np.tile(np.arange(64, dtype=np.float64), 16)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, expected)


def test_quantized_qint8_per_tensor():
    from tenso import QuantizedTensor

    name = "quantized_qint8_per_tensor.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    qt = tenso.loads(data)
    assert isinstance(qt, QuantizedTensor)
    assert qt.shape == (4, 4)
    expected = np.linspace(-2.0, 2.0, 16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_allclose(qt.dequantize(), expected, atol=0.05)


def test_sparse_coo_f32():
    pytest.importorskip("scipy")

    name = "sparse_coo_f32.tenso"
    data = _read_fixture(name)
    _assert_sha256(name, data)

    m = tenso.loads(data)
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 0], dense[0, 2], dense[1, 1], dense[2, 0] = 1.0, 2.0, 3.0, 4.0
    np.testing.assert_array_equal(m.toarray(), dense)


def test_v3_dense_backward_read():
    """The decoder must still read a v3 packet (8-byte header) — there is no v3
    writer, so this hand-frames a minimal v3 dense packet and asserts it decodes.
    This is the backward-compatibility canary the v4-only fixtures cannot give."""
    import struct as _struct

    from tenso.config import _ALIGNMENT, _MAGIC, FLAG_ALIGNED

    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype_code, ndim = 1, 1  # float32, 1-D
    # v3 header: magic(4) + ver=3 + flags_u8 + dtype + ndim  (8 bytes)
    header = _struct.pack("<4sBBBB", _MAGIC, 3, FLAG_ALIGNED, dtype_code, ndim)
    shape = _struct.pack("<I", values.size)
    cursor = 8 + ndim * 4
    pad = (_ALIGNMENT - cursor % _ALIGNMENT) % _ALIGNMENT
    packet = header + shape + b"\x00" * pad + values.tobytes()

    out = tenso.loads(packet)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, values)
