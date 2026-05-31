#!/usr/bin/env python3
"""Atheris fuzz target for ``tenso.get_packet_info``.

This exercises both the Rust fast path (``get_packet_info_rs``) and the
Python fallback that ``utils.get_packet_info`` drops to when the Rust
extension is unavailable or rejects the buffer.

Contract under test: same as ``fuzz_loads`` — ``ValueError`` /
``EOFError`` / ``TypeError`` are fine, anything else is a parser bug.

Run:

    pip install atheris
    python pyfuzz/fuzz_get_packet_info.py pyfuzz/seeds/
"""
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import tenso  # noqa: E402


_EXPECTED = (ValueError, EOFError, TypeError)


def _one(data: bytes) -> None:
    try:
        tenso.get_packet_info(data)
    except _EXPECTED:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"tenso.get_packet_info raised unexpected {type(exc).__name__}: {exc!r}"
        ) from exc


def main() -> None:
    atheris.Setup(sys.argv, _one)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
