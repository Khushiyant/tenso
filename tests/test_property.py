"""
Property-based round-trip tests for the Tenso protocol.

These tests don't pin specific byte layouts (that's ``test_conformance``'s
job).  Instead, they assert that for a wide range of randomly-generated
inputs the encode/decode pair is the identity::

    loads(dumps(x)) == x

Hypothesis explores shape/dtype/value space, so any regression where a
particular combination fails is much more likely to surface here than in
hand-written examples.

We constrain array size aggressively (total elements <= 256) so the full
suite finishes well under five seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

import tenso
from tenso import StringTensor


# Dtypes that Tenso dense-path supports and that hypothesis can sample
# directly via ``hypothesis.extra.numpy.arrays``.
DENSE_DTYPES = [
    np.dtype("float32"),
    np.dtype("float64"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("uint8"),
    np.dtype("int8"),
]


# Shapes up to 4 dimensions, with each dim small enough that total
# element count comfortably stays <= 256 even at max rank.
_shapes = array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=4)


def _dense_array_strategy(dtype: np.dtype):
    """Generate a contiguous array of a given dtype with safe values.

    Floats are restricted to finite values; otherwise NaN-vs-NaN equality
    makes assertions awkward.  Integers/uints use their natural width.
    """
    if np.issubdtype(dtype, np.floating):
        elements = st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            width=dtype.itemsize * 8,
        )
    else:
        info = np.iinfo(dtype)
        elements = st.integers(min_value=int(info.min), max_value=int(info.max))
    return arrays(dtype=dtype, shape=_shapes, elements=elements)


# Cap example count so the suite stays fast; suppress the "function-scoped
# fixture" health check since we don't use any fixtures here.
_settings = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_settings
@given(data=st.data())
@pytest.mark.parametrize("dtype", DENSE_DTYPES, ids=lambda d: d.name)
def test_dense_roundtrip(data, dtype):
    """``loads(dumps(arr)) == arr`` for all supported dense dtypes."""
    arr = data.draw(_dense_array_strategy(dtype))
    packet = tenso.dumps(arr)
    restored = tenso.loads(packet)

    assert restored.dtype == arr.dtype
    assert restored.shape == arr.shape
    np.testing.assert_array_equal(restored, arr)


@_settings
@given(data=st.data())
@pytest.mark.parametrize("dtype", DENSE_DTYPES, ids=lambda d: d.name)
def test_dense_roundtrip_with_integrity(data, dtype):
    """Same as above but with the integrity (XXH3) footer enabled."""
    arr = data.draw(_dense_array_strategy(dtype))
    packet = tenso.dumps(arr, check_integrity=True)
    restored = tenso.loads(packet)

    assert restored.dtype == arr.dtype
    assert restored.shape == arr.shape
    np.testing.assert_array_equal(restored, arr)


# Keys for bundles: short ASCII identifiers are enough to exercise the
# UTF-8 length-prefixed path without hitting weird edge cases that aren't
# the point of *this* test (StringTensor covers unicode below).
_bundle_keys = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
    min_size=1,
    max_size=8,
)


def _small_array_strategy():
    """Pick a random supported dtype, then build an array of that dtype."""
    return st.sampled_from(DENSE_DTYPES).flatmap(_dense_array_strategy)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    bundle=st.dictionaries(
        keys=_bundle_keys,
        values=_small_array_strategy(),
        min_size=1,
        max_size=4,
    )
)
def test_bundle_roundtrip(bundle):
    """``loads(dumps(d)) == d`` for dict-of-arrays bundles."""
    packet = tenso.dumps(bundle)
    restored = tenso.loads(packet)

    assert isinstance(restored, dict)
    assert set(restored.keys()) == set(bundle.keys())
    for key, expected in bundle.items():
        got = restored[key]
        assert got.dtype == expected.dtype
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)


# Unicode strings: any printable code point, including surrogate-free
# ranges, with bounded length so total payload stays small.
_unicode_strings = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),  # Skip surrogate halves; not valid UTF-8.
    ),
    min_size=0,
    max_size=16,
)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(strings=st.lists(_unicode_strings, min_size=0, max_size=16))
def test_string_tensor_roundtrip(strings):
    """``StringTensor.loads(st.dumps()) == st`` for arbitrary unicode lists."""
    st_obj = StringTensor(strings)
    packet = st_obj.dumps()
    restored = StringTensor.loads(packet)

    assert len(restored) == len(strings)
    assert restored.to_list() == strings
