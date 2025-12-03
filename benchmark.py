import time
import json
import pickle
import io
import msgpack
import numpy as np
import tenso

# --- CONFIGURATION ---
ITERATIONS = 50  # Run each test 50 times and take the average

# --- COMPETITORS ---
def bench_json(data):
    # JSON requires .tolist() which is very slow
    enc = lambda x: json.dumps(x.tolist()).encode('utf-8')
    dec = lambda x: np.array(json.loads(x), dtype=data.dtype)
    return enc, dec

def bench_pickle(data):
    enc = lambda x: pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
    dec = lambda x: pickle.loads(x)
    return enc, dec

def bench_msgpack(data):
    # Msgpack requires manual bytes handling for numpy usually
    enc = lambda x: msgpack.packb(x.tobytes())
    # Note: simple msgpack loses shape/dtype info, we are being generous to it here
    dec = lambda x: np.frombuffer(msgpack.unpackb(x), dtype=data.dtype).reshape(data.shape)
    return enc, dec

def bench_numpy_native(data):
    def enc(x):
        f = io.BytesIO()
        np.save(f, x)
        return f.getvalue()
    def dec(x):
        f = io.BytesIO(x)
        return np.load(f)
    return enc, dec

def bench_tenso(data):
    return tenso.dumps, tenso.loads

# --- SCENARIOS ---
SCENARIOS = [
    {
        "name": "API Vector",
        "desc": "Small 1D Embedding",
        "shape": (1536,), 
        "dtype": np.float32
    },
    {
        "name": "CV Batch",
        "desc": "32x 256x256 Images",
        "shape": (32, 256, 256, 3), 
        "dtype": np.uint8
    },
    {
        "name": "LLM Layer",
        "desc": "4096^2 Matrix",
        "shape": (4096, 4096), 
        "dtype": np.float32
    },
    {
        "name": "Masking",
        "desc": "Boolean Mask",
        "shape": (1024, 1024), 
        "dtype": np.bool
    }
]

# --- RUNNER ---
print(f"{'SCENARIO':<15} | {'FORMAT':<10} | {'SIZE':<10} | {'SERIALIZE':<10} | {'DESERIALIZE':<10} | {'vs JSON'}")
print("-" * 85)

for scen in SCENARIOS:
    # Generate Data
    if scen['dtype'] == np.bool:
        data = np.random.randint(0, 2, scen['shape']).astype(bool)
    elif scen['dtype'] == np.uint8:
        data = np.random.randint(0, 255, scen['shape']).astype(np.uint8)
    else:
        data = np.random.rand(*scen['shape']).astype(scen['dtype'])
    
    # Define Competitors
    competitors = {
        "JSON": bench_json(data),
        "Pickle": bench_pickle(data),
        "MsgPack": bench_msgpack(data),
        "Numpy": bench_numpy_native(data),
        "Tenso": bench_tenso(data)
    }

    results = {}

    # Run Tests
    for name, (enc_func, dec_func) in competitors.items():
        # JSON fails on large binary/bools gracefully sometimes, skip if needed
        if name == "JSON" and (scen['dtype'] == np.uint8 or scen['name'] == "LLM Layer"):
            results[name] = {"ser": float('inf'), "des": float('inf'), "size": 0}
            continue

        # Warmup
        try:
            encoded = enc_func(data)
            _ = dec_func(encoded)
        except Exception as e:
            print(f"Failed {name}: {e}")
            continue

        # Timing Serialization
        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            encoded = enc_func(data)
        t_ser = ((time.perf_counter() - t0) / ITERATIONS) * 1000 # ms

        # Timing Deserialization
        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            _ = dec_func(encoded)
        t_des = ((time.perf_counter() - t0) / ITERATIONS) * 1000 # ms

        results[name] = {
            "ser": t_ser,
            "des": t_des,
            "size": len(encoded)
        }

    # Print Rows
    baseline = results.get("JSON", {}).get("ser", float('inf'))
    
    for name, metrics in results.items():
        size_str = f"{metrics['size']/1024/1024:.2f} MB"
        if metrics['size'] < 1024:
            size_str = f"{metrics['size']} B"
        elif metrics['size'] < 1024*1024:
            size_str = f"{metrics['size']/1024:.2f} KB"

        speedup = "N/A"
        if baseline != float('inf') and metrics['ser'] > 0:
            factor = baseline / metrics['ser']
            speedup = f"{factor:.1f}x"
        elif name == "JSON":
            speedup = "1.0x"

        print(f"{scen['name']:<15} | {name:<10} | {size_str:<10} | {metrics['ser']:>7.3f} ms | {metrics['des']:>7.3f} ms | {speedup}")
    print("-" * 85)