#!/usr/bin/env python3
"""Generate a small corpus of valid Tenso packets for atheris seeding.

Run from anywhere:

    python pyfuzz/_make_seeds.py

The seeds directory is intentionally checked into git so CI doesn't need
to regenerate them. Re-run this script whenever the protocol changes in
a way that meaningfully expands the surface area we want fuzzed.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

import tenso

SEED_DIR = pathlib.Path(__file__).parent / "seeds"


def _seed(name: str, payload: bytes) -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    out = SEED_DIR / name
    out.write_bytes(payload)
    print(f"  wrote {out.relative_to(SEED_DIR.parent)} ({len(payload)} bytes)")


def main() -> int:
    print(f"Generating seeds into {SEED_DIR}")

    # 1) Tiny 1-D float32 array.
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    _seed("dense_f32_1d.bin", bytes(tenso.dumps(a)))

    # 2) 2-D float32 matrix.
    b = np.arange(12, dtype=np.float32).reshape(3, 4)
    _seed("dense_f32_2d.bin", bytes(tenso.dumps(b)))

    # 3) 3-D float32 tensor.
    c = np.zeros((2, 2, 2), dtype=np.float32)
    _seed("dense_f32_3d.bin", bytes(tenso.dumps(c)))

    # 4) Integrity-protected variant exercises the footer/hash path.
    _seed(
        "dense_f32_integrity.bin",
        bytes(tenso.dumps(a, check_integrity=True)),
    )

    # 5) Compressed variant exercises the LZ4 path.
    big = np.zeros(1024, dtype=np.float32)
    _seed("dense_f32_compressed.bin", bytes(tenso.dumps(big, compress=True)))

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
