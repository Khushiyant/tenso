"""
Generate the binary conformance fixtures.

Run once to regenerate ``.tenso`` files under this directory::

    .venv/bin/python tests/fixtures/_generate.py

The fixtures lock the on-disk wire format so we can detect accidental
encoder drift between releases.  All fixtures are produced by the
real encoder (``tenso.dumps`` / ``StringTensor.dumps``) — never hand-rolled
bytes — so they always reflect what the current implementation emits.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

import tenso
from tenso import StringTensor


FIXTURES_DIR = Path(__file__).resolve().parent


def _write(name: str, packet) -> tuple[str, int, str]:
    """Write a packet to ``FIXTURES_DIR/<name>`` and return (name, size, sha256)."""
    path = FIXTURES_DIR / name
    data = bytes(packet)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    return name, len(data), digest


def main() -> None:
    results: list[tuple[str, int, str]] = []

    # 1. Dense float32 vector, shape (5,) — [1, 2, 3, 4, 5]
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    results.append(_write("dense_f32_vec.tenso", tenso.dumps(arr)))

    # 2. Dense float64 matrix, shape (3, 4)
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    results.append(_write("dense_f64_mat.tenso", tenso.dumps(arr)))

    # 3. Dense int32 vector, shape (8,), values 0..7, integrity flag set
    arr = np.arange(8, dtype=np.int32)
    results.append(
        _write("dense_i32_integrity.tenso", tenso.dumps(arr, check_integrity=True))
    )

    # 4. Dense uint8, shape (4, 4, 4), all zeros
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    results.append(_write("dense_u8_3d_zeros.tenso", tenso.dumps(arr)))

    # 5. bfloat16 vector (optional; only if ml_dtypes is present)
    try:
        from ml_dtypes import bfloat16  # type: ignore

        arr = np.array([1.0, -1.5, 0.25, 3.75], dtype=bfloat16)
        results.append(_write("dense_bf16_vec.tenso", tenso.dumps(arr)))
    except ImportError:
        pass

    # 6. Bundle: {"a": float32[2], "b": int64[3]}
    bundle = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([10, 20, 30], dtype=np.int64),
    }
    results.append(_write("bundle_mixed.tenso", tenso.dumps(bundle)))

    # 7. StringTensor with mixed-length UTF-8 strings
    st = StringTensor(["hi", "", "world", "ñ"])
    results.append(_write("string_mixed_utf8.tenso", st.dumps()))

    # Print a summary that can be pasted into test_conformance.py
    width = max(len(name) for name, _, _ in results)
    print(f"\nGenerated {len(results)} fixtures in {FIXTURES_DIR}:")
    print()
    print("EXPECTED_SHA256 = {")
    for name, size, digest in results:
        print(f'    "{name}": "{digest}",  # {size} bytes')
    print("}")


if __name__ == "__main__":
    main()
