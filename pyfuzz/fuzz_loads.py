#!/usr/bin/env python3
"""Atheris fuzz target for ``tenso.loads``.

Contract under test: ``tenso.loads(data)`` may raise ``ValueError``,
``EOFError``, or ``TypeError`` on malformed input, but MUST NOT segfault,
hang, or raise any other exception type. A surfaced ``MemoryError`` or
``OverflowError`` is also a parser bug (the protocol caps ndim and
total_elements precisely so we never end up there from untrusted input).

Run:

    pip install atheris
    python pyfuzz/fuzz_loads.py pyfuzz/seeds/
"""
from __future__ import annotations

import sys

import atheris

# Instrument tenso (and its deps) so libFuzzer gets coverage feedback on
# the Python branches. The Rust ``.so`` is opaque to atheris — atheris
# only sees coverage for Python frames — but that still meaningfully
# exercises the Python fallback paths.
with atheris.instrument_imports():
    import tenso  # noqa: E402


# Exceptions we consider "expected" for malformed input. Anything else
# escaping ``tenso.loads`` is a bug worth a crash file.
_EXPECTED = (ValueError, EOFError, TypeError)


def _one(data: bytes) -> None:
    try:
        tenso.loads(data)
    except _EXPECTED:
        return
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        # Re-raise as a typed assertion so atheris records the input.
        raise AssertionError(
            f"tenso.loads raised unexpected {type(exc).__name__}: {exc!r}"
        ) from exc


def main() -> None:
    atheris.Setup(sys.argv, _one)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
