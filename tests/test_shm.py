
import pytest
import numpy as np
import sys
import multiprocessing.shared_memory

try:
    from tenso.shm import TensoShm
    HAS_SHM_IMPL = True
except ImportError:
    HAS_SHM_IMPL = False

@pytest.mark.skipif(not HAS_SHM_IMPL, reason="TensoShm not available")
def test_shm_roundtrip():
    name = "tenso_test_shm_01"
    shape = (100, 100)
    data = np.random.rand(*shape).astype(np.float32)

    shm_writer = None
    try:
        shm_writer = TensoShm.create_from(name, data, check_integrity=True)
        assert shm_writer.name == name

        # Zero-copy read
        read_data = shm_writer.get()
        assert read_data is not None
        np.testing.assert_array_equal(data, read_data)
        del read_data

        with TensoShm(name) as shm_reader:
            read_data_2 = shm_reader.get()
            np.testing.assert_array_equal(data, read_data_2)
            del read_data_2

    finally:
        if shm_writer:
            shm_writer.close()
            shm_writer.unlink()

@pytest.mark.skipif(not HAS_SHM_IMPL, reason="TensoShm not available")
def test_shm_manual_put():
    name = "tenso_test_shm_02"
    data = np.array([1, 2, 3, 4], dtype=np.int32)

    shm = multiprocessing.shared_memory.SharedMemory(name=name, create=True, size=1024)
    ts = TensoShm(name)

    try:
        bytes_written = ts.put(data)
        assert bytes_written > 0

        res = ts.get()
        np.testing.assert_array_equal(data, res)
        del res
    finally:
        ts.close()
        shm.close()
        shm.unlink()
