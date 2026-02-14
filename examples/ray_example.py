"""
Example: Using Tenso as Ray's tensor serializer.

Tenso replaces Ray's default pickle serialization with zero-copy
binary protocol, reducing CPU usage by ~46x for tensor operations.

Usage:
    pip install 'tenso[api]'
    python ray_example.py
"""

import time

import numpy as np
import ray

from tenso.ray import register

# Initialize Ray and register Tenso serializers
ray.init()
register()
print("Tenso registered as Ray serializer\n")


# --- Example 1: Basic put/get ---
print("--- Example 1: Object Store ---")
tensor = np.random.randn(1000, 1000).astype(np.float32)

start = time.perf_counter()
ref = ray.put(tensor)
result = ray.get(ref)
elapsed = (time.perf_counter() - start) * 1000

print(f"  Shape: {tensor.shape}, Dtype: {tensor.dtype}")
print(f"  Size: {tensor.nbytes / 1e6:.1f} MB")
print(f"  Put + Get: {elapsed:.1f} ms")
print(f"  Match: {np.array_equal(tensor, result)}\n")


# --- Example 2: Remote functions ---
print("--- Example 2: Remote Functions ---")


@ray.remote
def normalize(tensor):
    """Normalize a tensor to zero mean, unit variance."""
    return (tensor - tensor.mean()) / tensor.std()


@ray.remote
def dot_product(a, b):
    """Compute matrix multiplication."""
    return a @ b


a = np.random.randn(500, 500).astype(np.float32)
b = np.random.randn(500, 500).astype(np.float32)

start = time.perf_counter()
norm_ref = normalize.remote(a)
dot_ref = dot_product.remote(a, b)
norm_result, dot_result = ray.get([norm_ref, dot_ref])
elapsed = (time.perf_counter() - start) * 1000

print(f"  Normalize mean: {norm_result.mean():.6f} (should be ~0)")
print(f"  Dot product shape: {dot_result.shape}")
print(f"  Parallel execution: {elapsed:.1f} ms\n")


# --- Example 3: Actor with state ---
print("--- Example 3: Stateful Actor ---")


@ray.remote
class ModelWeightStore:
    """Actor that accumulates gradient updates to model weights."""

    def __init__(self, shape):
        self.weights = np.zeros(shape, dtype=np.float32)
        self.step = 0

    def update(self, gradient, lr=0.01):
        self.weights -= lr * gradient
        self.step += 1
        return self.step

    def get_weights(self):
        return self.weights


store = ModelWeightStore.remote((100, 100))

# Simulate 5 gradient updates
for i in range(5):
    grad = np.random.randn(100, 100).astype(np.float32)
    step = ray.get(store.update.remote(grad))

weights = ray.get(store.get_weights.remote())
print(f"  After {step} updates: weights norm = {np.linalg.norm(weights):.4f}\n")


# --- Example 4: Pipeline with multiple tensors ---
print("--- Example 4: Pipeline ---")


@ray.remote
def embed(batch):
    """Simulate embedding lookup."""
    vocab_size, embed_dim = 10000, 256
    embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    return embeddings[batch]


@ray.remote
def transform(embeddings):
    """Simulate a transformer layer."""
    return embeddings @ np.random.randn(256, 256).astype(np.float32)


@ray.remote
def classify(features):
    """Simulate classification head."""
    logits = features.mean(axis=0) @ np.random.randn(256, 10).astype(np.float32)
    return logits.argmax()


batch = np.random.randint(0, 10000, size=(32,))

start = time.perf_counter()
emb_ref = embed.remote(batch)
feat_ref = transform.remote(emb_ref)  # Chained - no intermediate ray.get()
pred = ray.get(classify.remote(feat_ref))
elapsed = (time.perf_counter() - start) * 1000

print(f"  Predicted class: {pred}")
print(f"  Pipeline latency: {elapsed:.1f} ms\n")

ray.shutdown()
print("Done.")
