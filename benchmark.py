import argparse
import time
import json
import pickle
import io
import os
import socket
import threading
import tempfile
import struct
import numpy as np
import tenso

# Optional dependencies for comparison
try:
    import msgpack
    import pyarrow as pa
    from safetensors.numpy import save as st_save, load as st_load
except ImportError:
    print("Warning: Missing benchmark dependencies (msgpack, pyarrow, safetensors). Some tests skipped.")

# --- 1. SERIALIZATION BENCHMARK HELPERS ---

def bench_json(data):
    enc = lambda x: json.dumps(x.tolist()).encode('utf-8')
    dec = lambda x: np.array(json.loads(x), dtype=data.dtype)
    return enc, dec

def bench_pickle(data):
    enc = lambda x: pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
    dec = lambda x: pickle.loads(x)
    return enc, dec

def bench_msgpack(data):
    enc = lambda x: msgpack.packb(x.tobytes())
    dec = lambda x: np.frombuffer(msgpack.unpackb(x), dtype=data.dtype).reshape(data.shape)
    return enc, dec

def bench_safetensors(data):
    def enc(x): return st_save({"t": x})
    def dec(x): return st_load(x)["t"]
    return enc, dec

def bench_arrow(data):
    def enc(x):
        arr = pa.array(x.flatten())
        batch = pa.RecordBatch.from_arrays([arr], names=['t'])
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
        return sink.getvalue()
    
    def dec(x):
        with pa.ipc.open_stream(x) as reader:
            batch = reader.read_next_batch()
            return batch.column(0).to_numpy(zero_copy_only=False).reshape(data.shape)
    return enc, dec

def bench_tenso(data):
    return tenso.dumps, tenso.loads

# --- BENCHMARK RUNNERS ---

def run_serialization():
    print("\n" + "="*80)
    print("BENCHMARK 1: IN-MEMORY SERIALIZATION (CPU Overhead)")
    print("="*80)
    
    SCENARIOS = [
        {"name": "API Vector", "shape": (1536,), "dtype": np.float32},
        {"name": "CV Batch", "shape": (32, 256, 256, 3), "dtype": np.uint8},
        {"name": "LLM Layer", "shape": (4096, 4096), "dtype": np.float32}
    ]
    
    print(f"{'SCENARIO':<15} | {'FORMAT':<12} | {'SIZE':<10} | {'SERIALIZE':<10} | {'DESERIALIZE':<10}")
    print("-" * 75)

    for scen in SCENARIOS:
        if scen['dtype'] == np.uint8:
            data = np.random.randint(0, 255, scen['shape']).astype(np.uint8)
        else:
            data = np.random.rand(*scen['shape']).astype(scen['dtype'])
        
        competitors = {
            "Pickle": bench_pickle(data),
            "Tenso": bench_tenso(data)
        }
        if 'msgpack' in globals(): competitors["MsgPack"] = bench_msgpack(data)
        if 'st_save' in globals(): competitors["Safetensors"] = bench_safetensors(data)
        if 'pa' in globals(): competitors["Arrow"] = bench_arrow(data)

        for name, (enc_func, dec_func) in competitors.items():
            try:
                # Warmup
                encoded = enc_func(data)
                _ = dec_func(encoded)
                
                ITERATIONS = 10
                t0 = time.perf_counter()
                for _ in range(ITERATIONS): encoded = enc_func(data)
                t_ser = ((time.perf_counter() - t0) / ITERATIONS) * 1000

                t0 = time.perf_counter()
                for _ in range(ITERATIONS): _ = dec_func(encoded)
                t_des = ((time.perf_counter() - t0) / ITERATIONS) * 1000
                
                size_str = f"{len(encoded)/1024/1024:.2f} MB" if len(encoded) > 1024**2 else f"{len(encoded)/1024:.2f} KB"
                print(f"{scen['name']:<15} | {name:<12} | {size_str:<10} | {t_ser:>7.3f} ms | {t_des:>7.3f} ms")
            except Exception as e:
                print(f"{scen['name']:<15} | {name:<12} | FAILED ({e})")
        print("-" * 75)

def run_io():
    print("\n" + "="*80)
    print("BENCHMARK 2: DISK I/O (Read/Write & Memory Mapping)")
    print("="*80)
    
    shape = (8192, 8192) 
    data = np.random.rand(*shape).astype(np.float32)
    size_mb = data.nbytes / (1024 * 1024)
    print(f"Dataset: {size_mb:.0f} MB Matrix {shape}")
    
    print(f"{'FORMAT':<15} | {'WRITE (ms)':<10} | {'READ (ms)':<10}")
    print("-" * 60)

    # 1. Tenso
    with tempfile.NamedTemporaryFile(delete=False) as f: path = f.name
    try:
        t0 = time.perf_counter()
        with open(path, "wb") as f: tenso.dump(data, f)
        t_write = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with open(path, "rb") as f: tenso.load(f, mmap_mode=True)
        t_read = (time.perf_counter() - t0) * 1000
        print(f"{'Tenso':<15} | {t_write:>10.2f} | {t_read:>10.2f}")
    finally: os.remove(path)

    # 2. Numpy
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as f: path = f.name
    try:
        t0 = time.perf_counter()
        np.save(path, data)
        t_write = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        np.load(path, mmap_mode='r')
        t_read = (time.perf_counter() - t0) * 1000
        print(f"{'Numpy .npy':<15} | {t_write:>10.2f} | {t_read:>10.2f}")
    finally: os.remove(path)

    # 3. Pickle
    with tempfile.NamedTemporaryFile(delete=False) as f: path = f.name
    try:
        t0 = time.perf_counter()
        with open(path, "wb") as f: pickle.dump(data, f)
        t_write = (time.perf_counter() - t0) * 1000

        with open(path, "rb") as f: 
            t0 = time.perf_counter()
            pickle.load(f)
            t_read = (time.perf_counter() - t0) * 1000
        print(f"{'Pickle':<15} | {t_write:>10.2f} | {t_read:>10.2f}")
    finally: os.remove(path)

def run_stream_read():
    print("\n" + "="*80)
    print("BENCHMARK 3: STREAM READ (Throughput & Memory Churn)")
    print("="*80)
    
    class FastStream(io.BytesIO): pass # IOBase supports readinto
    
    data = np.random.rand(5000, 5000).astype(np.float32)
    packet = tenso.dumps(data)
    size_mb = len(packet) / (1024 * 1024)
    
    print(f"Dataset: {size_mb:.0f} MB Packet")
    print(f"{'METHOD':<20} | {'TIME (ms)':<10} | {'THROUGHPUT':<15}")
    print("-" * 60)

    # Optimized
    stream = FastStream(packet)
    t0 = time.perf_counter()
    tenso.read_stream(stream)
    t_opt = (time.perf_counter() - t0) * 1000
    print(f"{'Tenso read_stream':<20} | {t_opt:>7.2f} ms | {size_mb/(t_opt/1000):>7.2f} MB/s")

    # Legacy Loop Simulation
    stream.seek(0)
    t0 = time.perf_counter()
    buffer = b''
    while True:
        chunk = stream.read(65536)
        if not chunk: break
        buffer += chunk
    tenso.loads(buffer)
    t_old = (time.perf_counter() - t0) * 1000
    print(f"{'Naive Loop':<20} | {t_old:>7.2f} ms | {size_mb/(t_old/1000):>7.2f} MB/s")
    
    print("-" * 60)
    print(f"Speedup: {t_old/t_opt:.1f}x")

def run_stream_write():
    print("\n" + "="*80)
    print("BENCHMARK 4: NETWORK WRITE (Latency & Atomic Packets)")
    print("="*80)
    
    PORT = 9998
    
    def sink_server():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('localhost', PORT))
        s.listen(1)
        conn, _ = s.accept()
        while True:
            try:
                if not conn.recv(1024*1024): break
            except: break
        conn.close()
        s.close()

    t = threading.Thread(target=sink_server, daemon=True)
    t.start()
    time.sleep(0.5)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', PORT))

    data = np.random.rand(16, 16).astype(np.float32)
    COUNT = 10000
    
    print(f"Sending {COUNT} tensors (1KB each) over localhost TCP...")

    # Optimized write_stream
    t0 = time.perf_counter()
    for _ in range(COUNT):
        tenso.write_stream(data, client)
    t_total = time.perf_counter() - t0
    
    client.close()
    
    print(f"Total Time: {t_total:.4f}s")
    print(f"Throughput: {COUNT/t_total:.0f} packets/sec")
    print(f"Latency:    {(t_total/COUNT)*1_000_000:.1f} µs/packet")

# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tenso Benchmarks")
    parser.add_argument("mode", nargs="?", choices=["all", "ser", "io", "read", "write"], default="all",
                        help="Benchmark mode: ser (Serialization), io (Disk), read (Stream Read), write (Stream Write)")
    
    args = parser.parse_args()
    
    if args.mode in ["all", "ser"]: run_serialization()
    if args.mode in ["all", "io"]: run_io()
    if args.mode in ["all", "read"]: run_stream_read()
    if args.mode in ["all", "write"]: run_stream_write()