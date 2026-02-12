"""Comprehensive test suite for TensoCache."""

import multiprocessing
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

from tenso.cache import TensoCache, _MAX_KEY_LEN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache():
    """Provide a fresh 4MB TensoCache that auto-cleans on teardown."""
    c = TensoCache("4MB")
    yield c
    c.close()
    c.unlink()


@pytest.fixture
def small_cache():
    """Provide a very small cache for eviction tests."""
    c = TensoCache("64KB")
    yield c
    c.close()
    c.unlink()


# ---------------------------------------------------------------------------
# Basic put/get roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64, np.int32, np.int64,
        np.uint8, np.int8, np.float16, np.int16,
        np.uint16, np.uint32, np.uint64, np.bool_,
    ])
    def test_various_dtypes(self, cache, dtype):
        arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        cache.put("t", arr)
        result = cache.get("t")
        assert result is not None
        np.testing.assert_array_equal(arr, result)

    def test_multidimensional(self, cache):
        arr = np.random.randn(10, 20, 3).astype(np.float32)
        cache.put("3d", arr)
        result = cache.get("3d")
        np.testing.assert_array_equal(arr, result)

    def test_scalar(self, cache):
        arr = np.array(42.0, dtype=np.float64)
        cache.put("scalar", arr)
        result = cache.get("scalar")
        assert result is not None
        np.testing.assert_array_equal(arr, result)

    def test_empty_array(self, cache):
        arr = np.array([], dtype=np.float32)
        cache.put("empty", arr)
        result = cache.get("empty")
        assert result is not None
        assert result.shape == (0,)

    def test_large_array(self, cache):
        arr = np.random.randn(100, 100).astype(np.float32)
        cache.put("large", arr)
        result = cache.get("large")
        np.testing.assert_array_equal(arr, result)

    def test_1d_array(self, cache):
        arr = np.arange(1000, dtype=np.int64)
        cache.put("1d", arr)
        result = cache.get("1d")
        np.testing.assert_array_equal(arr, result)


# ---------------------------------------------------------------------------
# Zero-copy verification
# ---------------------------------------------------------------------------

class TestZeroCopy:
    def test_get_returns_readonly_by_default(self, cache):
        arr = np.ones((10, 10), dtype=np.float32)
        cache.put("ro", arr)
        result = cache.get("ro")
        assert result is not None
        np.testing.assert_array_equal(arr, result)

    def test_copy_flag_returns_writeable(self, cache):
        arr = np.ones((10, 10), dtype=np.float32)
        cache.put("cp", arr)
        result = cache.get("cp", copy=True)
        assert result is not None
        np.testing.assert_array_equal(arr, result)
        # copy should be independent
        result[0, 0] = 999.0
        original = cache.get("cp")
        assert original[0, 0] != 999.0


# ---------------------------------------------------------------------------
# In-place update
# ---------------------------------------------------------------------------

class TestInPlaceUpdate:
    def test_same_size_reuses_allocation(self, cache):
        arr1 = np.ones(100, dtype=np.float32)
        arr2 = np.ones(100, dtype=np.float32) * 2
        cache.put("x", arr1)
        stats_before = cache.stats
        cache.put("x", arr2)
        result = cache.get("x")
        np.testing.assert_array_equal(arr2, result)
        # Active entry count should remain 1
        assert cache.stats['entries'] == 1

    def test_larger_update_reallocates(self, cache):
        arr_small = np.ones(10, dtype=np.float32)
        arr_large = np.ones(10000, dtype=np.float32) * 2
        cache.put("y", arr_small)
        cache.put("y", arr_large)
        result = cache.get("y")
        np.testing.assert_array_equal(arr_large, result)
        assert cache.stats['entries'] == 1


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_lru_eviction_order(self):
        """Insert entries, promote one via get(), then force a single eviction.
        The promoted entry should survive; the LRU-tail entry should be evicted."""
        cache = TensoCache("16KB")
        try:
            arr = np.ones(64, dtype=np.float32)

            # Fill all entry slots: with 16KB pool we get ~9 max_entries
            max_ent = cache._max_entries
            for i in range(max_ent):
                cache.put(f"k{i}", arr * i)

            # LRU order after sequential inserts (head → tail):
            #   k(max_ent-1) → ... → k1 → k0
            # k0 is the LRU tail.

            # Promote k0 to MRU via get()
            cache.get("k0")
            # LRU order: k0 → k(max_ent-1) → ... → k2 → k1  (k1 is now the tail)

            # Insert one more entry — this forces eviction of the tail (k1)
            cache.put("new", arr * 99)

            # k0 was promoted, should survive
            result_k0 = cache.get("k0")
            assert result_k0 is not None, "'k0' should survive (was promoted to MRU)"
            np.testing.assert_array_equal(result_k0, arr * 0)

            # k1 was the LRU tail, should be evicted
            assert cache.get("k1") is None, "'k1' should be evicted (was LRU tail)"

            # "new" should be present
            result_new = cache.get("new")
            assert result_new is not None
            np.testing.assert_array_equal(result_new, arr * 99)
        finally:
            cache.close()
            cache.unlink()


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

class TestTTL:
    def test_ttl_expiry(self, cache):
        arr = np.array([1, 2, 3], dtype=np.float32)
        cache.put("ttl_key", arr, ttl=0.1)
        # Should be available immediately
        assert cache.get("ttl_key") is not None
        # Wait for expiry
        time.sleep(0.2)
        assert cache.get("ttl_key") is None

    def test_ttl_not_expired(self, cache):
        arr = np.array([1, 2, 3], dtype=np.float32)
        cache.put("long_ttl", arr, ttl=10.0)
        result = cache.get("long_ttl")
        assert result is not None
        np.testing.assert_array_equal(arr, result)

    def test_no_ttl_persists(self, cache):
        arr = np.array([1, 2, 3], dtype=np.float32)
        cache.put("no_ttl", arr)
        time.sleep(0.1)
        assert cache.get("no_ttl") is not None

    def test_ttl_contains_and_keys(self, cache):
        arr = np.array([1], dtype=np.float32)
        cache.put("exp", arr, ttl=0.1)
        assert "exp" in cache
        time.sleep(0.2)
        assert "exp" not in cache
        assert "exp" not in cache.keys()


# ---------------------------------------------------------------------------
# delete() and free-list reuse
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_existing(self, cache):
        arr = np.ones(100, dtype=np.float32)
        cache.put("del_me", arr)
        assert "del_me" in cache
        assert cache.delete("del_me") is True
        assert "del_me" not in cache
        assert cache.get("del_me") is None

    def test_delete_nonexistent(self, cache):
        assert cache.delete("nope") is False

    def test_free_space_reused(self, cache):
        """After deleting, freed space should be reusable."""
        arr = np.ones(500, dtype=np.float32)
        cache.put("a", arr)
        free_before = cache.stats['free_bytes']
        cache.delete("a")
        free_after = cache.stats['free_bytes']
        assert free_after > free_before
        # Should be able to insert again
        cache.put("b", arr)
        assert cache.get("b") is not None


# ---------------------------------------------------------------------------
# keys(), clear(), len(), 'in' operator
# ---------------------------------------------------------------------------

class TestCollectionOps:
    def test_keys(self, cache):
        cache.put("a", np.array([1], dtype=np.float32))
        cache.put("b", np.array([2], dtype=np.float32))
        k = cache.keys()
        assert set(k) == {"a", "b"}

    def test_len(self, cache):
        assert len(cache) == 0
        cache.put("x", np.array([1], dtype=np.float32))
        assert len(cache) == 1
        cache.put("y", np.array([2], dtype=np.float32))
        assert len(cache) == 2
        cache.delete("x")
        assert len(cache) == 1

    def test_contains(self, cache):
        cache.put("here", np.array([1], dtype=np.float32))
        assert "here" in cache
        assert "not_here" not in cache

    def test_clear(self, cache):
        for i in range(5):
            cache.put(f"k{i}", np.arange(10, dtype=np.float32))
        assert len(cache) == 5
        cache.clear()
        assert len(cache) == 0
        assert cache.keys() == []
        # Free bytes should be restored
        stats = cache.stats
        assert stats['free_bytes'] == stats['data_region_size']


# ---------------------------------------------------------------------------
# info() metadata
# ---------------------------------------------------------------------------

class TestInfo:
    def test_info_shape_dtype(self, cache):
        arr = np.random.randn(5, 10).astype(np.float32)
        cache.put("info_test", arr)
        info = cache.info("info_test")
        assert info is not None
        assert info['shape'] == (5, 10)
        assert info['ndim'] == 2
        assert info['dtype'] == np.dtype('float32')
        assert info['size_bytes'] > 0
        assert info['ttl'] is None

    def test_info_with_ttl(self, cache):
        arr = np.array([1, 2], dtype=np.int32)
        cache.put("ttl_info", arr, ttl=5.0)
        info = cache.info("ttl_info")
        assert info is not None
        assert info['ttl'] == 5.0
        assert info['age'] >= 0.0

    def test_info_nonexistent(self, cache):
        assert cache.info("nope") is None

    def test_info_expired(self, cache):
        arr = np.array([1], dtype=np.float32)
        cache.put("exp_info", arr, ttl=0.05)
        time.sleep(0.1)
        assert cache.info("exp_info") is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_reads(self, cache):
        """Multiple threads reading the same key concurrently."""
        arr = np.random.randn(100).astype(np.float32)
        cache.put("shared", arr)
        errors = []

        def reader():
            try:
                for _ in range(50):
                    result = cache.get("shared")
                    if result is not None:
                        np.testing.assert_array_equal(arr, result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_put_get(self, cache):
        """Concurrent puts and gets should not crash."""
        errors = []

        def writer(key_prefix):
            try:
                for i in range(20):
                    arr = np.array([i], dtype=np.float32)
                    cache.put(f"{key_prefix}_{i}", arr)
            except Exception as e:
                errors.append(e)

        def reader(key_prefix):
            try:
                for i in range(20):
                    cache.get(f"{key_prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for p in range(3):
            threads.append(threading.Thread(target=writer, args=(f"w{p}",)))
            threads.append(threading.Thread(target=reader, args=(f"w{p}",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# Cross-process test
# ---------------------------------------------------------------------------

class TestCrossProcess:
    def test_subprocess_reads(self, cache):
        """A subprocess attaches to the same SHM by name and reads data."""
        arr = np.arange(10, dtype=np.float32)
        cache.put("cross", arr)

        code = textwrap.dedent(f"""\
        import sys
        import numpy as np
        from tenso.cache import TensoCache
        c = TensoCache(name="{cache.name}", create=False)
        result = c.get("cross")
        if result is None:
            sys.exit(1)
        expected = np.arange(10, dtype=np.float32)
        if not np.array_equal(result, expected):
            sys.exit(2)
        c.close()
        sys.exit(0)
        """)

        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10
        )
        assert proc.returncode == 0, (
            f"Subprocess failed (rc={proc.returncode}): "
            f"stdout={proc.stdout}, stderr={proc.stderr}"
        )


# ---------------------------------------------------------------------------
# MemoryError when pool exhausted
# ---------------------------------------------------------------------------

class TestMemoryExhaustion:
    def test_raises_memory_error(self):
        """Single allocation larger than entire data region should raise MemoryError."""
        c = TensoCache("16KB")
        try:
            with pytest.raises(MemoryError):
                # ~40KB array into a 16KB pool — cannot fit even after eviction
                c.put("huge", np.ones(10000, dtype=np.float32))
        finally:
            c.close()
            c.unlink()


# ---------------------------------------------------------------------------
# Key length validation
# ---------------------------------------------------------------------------

class TestKeyValidation:
    def test_key_too_long(self, cache):
        long_key = "x" * (_MAX_KEY_LEN + 1)
        with pytest.raises(ValueError, match="Key too long"):
            cache.put(long_key, np.array([1], dtype=np.float32))

    def test_max_length_key_works(self, cache):
        key = "k" * _MAX_KEY_LEN
        arr = np.array([42], dtype=np.float32)
        cache.put(key, arr)
        result = cache.get(key)
        assert result is not None
        np.testing.assert_array_equal(arr, result)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_cleanup(self):
        with TensoCache("1MB") as c:
            c.put("ctx", np.array([1, 2], dtype=np.float32))
            assert c.get("ctx") is not None
            name = c.name
        # After exiting, cache should be closed
        with pytest.raises(RuntimeError, match="closed"):
            c.get("ctx")

    def test_context_manager_unlinks(self):
        name = None
        with TensoCache("1MB") as c:
            name = c.name
            c.put("test", np.array([1], dtype=np.float32))
        # SHM segment should be unlinked; re-creating should succeed
        with TensoCache("1MB", name=name) as c2:
            assert len(c2) == 0


# ---------------------------------------------------------------------------
# stats property
# ---------------------------------------------------------------------------

class TestStats:
    def test_initial_stats(self, cache):
        s = cache.stats
        assert s['entries'] == 0
        assert s['hits'] == 0
        assert s['misses'] == 0
        assert s['free_bytes'] > 0
        assert s['pool_size'] > 0

    def test_stats_after_operations(self, cache):
        arr = np.ones(100, dtype=np.float32)
        cache.put("s1", arr)
        cache.get("s1")        # hit
        cache.get("nonexist")  # miss

        s = cache.stats
        assert s['entries'] == 1
        assert s['hits'] == 1
        assert s['misses'] == 1
        assert s['used_bytes'] > 0
        assert 0.0 < s['hit_rate'] < 1.0

    def test_hit_rate_calculation(self, cache):
        arr = np.array([1], dtype=np.float32)
        cache.put("hr", arr)
        for _ in range(10):
            cache.get("hr")
        s = cache.stats
        assert s['hits'] == 10
        assert s['hit_rate'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_overwrite_same_key_many_times(self, cache):
        for i in range(50):
            cache.put("loop", np.array([i], dtype=np.float32))
        result = cache.get("loop")
        np.testing.assert_array_equal(result, np.array([49], dtype=np.float32))
        assert cache.stats['entries'] == 1

    def test_get_after_close_raises(self, cache):
        cache.put("x", np.array([1], dtype=np.float32))
        cache.close()
        with pytest.raises(RuntimeError):
            cache.get("x")

    def test_repr(self, cache):
        r = repr(cache)
        assert "TensoCache" in r
        assert cache.name in r
        assert "open" in r

    def test_multiple_keys(self, cache):
        for i in range(20):
            cache.put(f"multi_{i}", np.array([i], dtype=np.float64))
        for i in range(20):
            result = cache.get(f"multi_{i}")
            assert result is not None
            np.testing.assert_array_equal(result, np.array([i], dtype=np.float64))

    def test_create_false_validates_magic(self):
        """Attaching to non-TensoCache SHM should raise ValueError."""
        import multiprocessing.shared_memory as shm_mod
        seg = shm_mod.SharedMemory(create=True, size=8192)
        try:
            seg.buf[:4] = b"XXXX"
            with pytest.raises(ValueError, match="Invalid TensoCache magic"):
                TensoCache(name=seg.name, create=False)
        finally:
            seg.close()
            seg.unlink()

    def test_create_false_requires_name(self):
        with pytest.raises(ValueError, match="name is required"):
            TensoCache(create=False)
