"""
Ray Integration for Tenso.

Registers Tenso as a custom serializer for Ray, replacing pickle-based
serialization with zero-copy tensor transfer for numpy arrays and
optionally PyTorch tensors.

Usage::

    import ray
    from tenso.ray import register

    ray.init()
    register()  # Register Tenso as the serializer for numpy arrays

    # All ray.put/get operations now use Tenso for numpy arrays
    ref = ray.put(np.zeros((1000, 1000)))
    arr = ray.get(ref)  # Deserialized via Tenso (46x less CPU than pickle)

    # Works transparently with remote functions and actors
    @ray.remote
    def process(tensor):
        return tensor.mean()

    ray.get(process.remote(np.random.randn(1000, 1000)))
"""

from typing import List, Optional

import numpy as np

from . import dumps, loads
from .quantize import QuantizedTensor

# Tracks the classes for which register() installed Tenso serializers, so
# that unregister() can remove exactly those (including torch/jax) and
# correctly revert to pickle.
_registered_types: List[type] = []


def _serialize_ndarray(arr: np.ndarray) -> bytes:
    """Serialize a numpy array using Tenso protocol."""
    return bytes(dumps(arr))


def _deserialize_ndarray(data: bytes) -> np.ndarray:
    """Deserialize a numpy array from Tenso protocol."""
    return loads(data, copy=True)


def _serialize_dict(d: dict) -> bytes:
    """Serialize a dict bundle using Tenso protocol."""
    return bytes(dumps(d))


def _deserialize_dict(data: bytes) -> dict:
    """Deserialize a dict bundle from Tenso protocol."""
    return loads(data, copy=True)


def _serialize_quantized(qt: QuantizedTensor) -> bytes:
    """Serialize a QuantizedTensor using Tenso protocol."""
    return bytes(dumps(qt))


def _deserialize_quantized(data: bytes) -> QuantizedTensor:
    """Deserialize a QuantizedTensor from Tenso protocol."""
    return loads(data, copy=True)


def register(
    include_torch: bool = False,
    include_jax: bool = False,
) -> None:
    """
    Register Tenso as the custom serializer for tensor types in Ray.

    After calling this, all ``ray.put()``, ``ray.get()``, remote function
    arguments, and actor method arguments involving registered types will
    be serialized using Tenso instead of pickle.

    Parameters
    ----------
    include_torch : bool, default False
        Also register serializers for ``torch.Tensor``. Requires PyTorch.
    include_jax : bool, default False
        Also register serializers for JAX arrays. Requires JAX.

    Raises
    ------
    ImportError
        If ray is not installed or if optional frameworks are not available.

    Examples
    --------
    >>> import ray
    >>> from tenso.ray import register
    >>> ray.init()
    >>> register()
    >>> ref = ray.put(np.zeros((100, 100)))
    >>> arr = ray.get(ref)
    """
    try:
        from ray.util.serialization import register_serializer
    except ImportError:
        raise ImportError(
            "Ray is required for this integration. "
            "Install it with: pip install 'tenso[api]'"
        )

    # Reset tracking so unregister() only ever reverts this call's serializers.
    _registered_types.clear()

    # numpy arrays
    register_serializer(
        np.ndarray,
        serializer=_serialize_ndarray,
        deserializer=_deserialize_ndarray,
    )
    _registered_types.append(np.ndarray)

    # dict bundles
    register_serializer(
        dict,
        serializer=_serialize_dict,
        deserializer=_deserialize_dict,
    )
    _registered_types.append(dict)

    # QuantizedTensor
    register_serializer(
        QuantizedTensor,
        serializer=_serialize_quantized,
        deserializer=_deserialize_quantized,
    )
    _registered_types.append(QuantizedTensor)

    # PyTorch tensors
    if include_torch:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for torch tensor serialization. "
                "Install it with: pip install torch"
            )

        def _serialize_torch(t: torch.Tensor) -> bytes:
            arr = t.detach().cpu().numpy()
            return bytes(dumps(arr))

        def _deserialize_torch(data: bytes) -> torch.Tensor:
            arr = loads(data, copy=True)
            return torch.from_numpy(arr)

        register_serializer(
            torch.Tensor,
            serializer=_serialize_torch,
            deserializer=_deserialize_torch,
        )
        _registered_types.append(torch.Tensor)

    # JAX arrays
    if include_jax:
        try:
            import jax.numpy as jnp
            from jax import Array as JaxArray
        except ImportError:
            raise ImportError(
                "JAX is required for jax array serialization. "
                "Install it with: pip install jax"
            )

        def _serialize_jax(t: JaxArray) -> bytes:
            arr = np.asarray(t)
            return bytes(dumps(arr))

        def _deserialize_jax(data: bytes) -> JaxArray:
            arr = loads(data, copy=True)
            return jnp.array(arr)

        register_serializer(
            JaxArray,
            serializer=_serialize_jax,
            deserializer=_deserialize_jax,
        )
        _registered_types.append(JaxArray)


def unregister() -> None:
    """
    Remove Tenso serializers from Ray, reverting to default pickle behavior.

    This deregisters exactly the types that were registered by the most recent
    :func:`register` call (including dict bundles and any torch/jax types),
    so that all of them revert to pickle.
    """
    try:
        from ray.util.serialization import deregister_serializer
    except ImportError:
        return

    while _registered_types:
        cls = _registered_types.pop()
        try:
            deregister_serializer(cls)
        except Exception:
            pass
