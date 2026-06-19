import numpy as np
import tenso
import pytest


# --- Introspection Tests ---


def test_get_packet_info_bundle_reports_entry_count():
    """For a bundle, `ndim` is the entry count and the post-header bytes are
    key-length prefixes, not dims — so get_packet_info must report entry_count
    and an empty shape, not a meaningless one."""
    bundle = {
        "a": np.zeros((3, 4), dtype=np.float32),
        "b": np.arange(5, dtype=np.int64),
    }
    info = tenso.get_packet_info(tenso.dumps(bundle))
    assert info["entry_count"] == 2
    assert info["shape"] == ()
    assert info["total_elements"] == 0


def test_get_packet_info():
    """Verify introspection returns correct metadata without deserializing."""
    data = np.random.rand(32, 128).astype(np.float32)
    packet = tenso.dumps(data)

    info = tenso.get_packet_info(packet)

    assert info["version"] == 4
    assert info["dtype"] == np.dtype("float32")
    assert info["shape"] == (32, 128)
    assert info["ndim"] == 2
    assert info["aligned"] is True
    assert info["total_elements"] == 32 * 128
    assert info["data_size_bytes"] == 32 * 128 * 4  # float32 is 4 bytes


def test_packet_info_invalid():
    """Verify introspection raises errors on bad data."""
    with pytest.raises(ValueError, match="Packet too short"):
        tenso.get_packet_info(b"short")

    with pytest.raises(ValueError, match="Invalid tenso packet"):
        tenso.get_packet_info(b"JUNK____")
