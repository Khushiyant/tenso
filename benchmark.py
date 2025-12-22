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
import sys
import psutil
import numpy as np
import tenso
import asyncio

# Optional dependencies for comparison
try:
    import msgpack
    import pyarrow as pa
    from safetensors.numpy import save as st_save, load as st_load
except ImportError:
    print("Warning: Missing benchmark dependencies (msgpack, pyarrow, safetensors). Some tests skipped.")


try:
    from tenso.async_core import awrite_stream, aread_stream
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False

# Global state for integrity check
USE_INTEGRITY = False

# --- HELPER: Resource Monitoring ---

class ResourceMonitor:
    """Monitor CPU and memory during benchmarks."""
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
    def snapshot(self):
        return {
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'cpu_percent': self.process.cpu_percent()
        }
    
    def memory_delta(self):
        current = self.process.memory_info().rss / 1024 / 1024
        return current - self.baseline_memory

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
    # Wrap dumps to respect the global USE_INTEGRITY flag
    def enc(x): return tenso.dumps(x, check_integrity=USE_INTEGRITY)
    return enc, tenso.loads

# --- BENCHMARK RUNNERS ---

def run_serialization():
    print("\n" + "="*80)
    print(f"BENCHMARK 1: IN-MEMORY SERIALIZATION (Integrity: {USE_INTEGRITY})")
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
    print(f"BENCHMARK 2: DISK I/O (Integrity: {USE_INTEGRITY})")
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
        with open(path, "wb") as f: tenso.dump(data, f, check_integrity=USE_INTEGRITY)
        t_write = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        # load automatically detects integrity footer from header
        with open(path, "rb") as f: tenso.load(f, mmap_mode=True)
        t_read = (time.perf_counter() - t0) * 1000
        print(f"{'Tenso':<15} | {t_write:>10.2f} | {t_read:>10.2f}")
    finally: 
        if os.path.exists(path): os.remove(path)

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
    finally: 
        if os.path.exists(path): os.remove(path)

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
    finally: 
        if os.path.exists(path): os.remove(path)

def run_stream_read():
    print("\n" + "="*80)
    print(f"BENCHMARK 3: STREAM READ (Integrity: {USE_INTEGRITY})")
    print("="*80)
    
    class FastStream(io.BytesIO): pass # IOBase supports readinto
    
    data = np.random.rand(5000, 5000).astype(np.float32)
    packet = tenso.dumps(data, check_integrity=USE_INTEGRITY)
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
    print(f"BENCHMARK 4: NETWORK WRITE (Integrity: {USE_INTEGRITY})")
    print("="*80)
    
    PORT = 9998
    
    def sink_server():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('localhost', PORT))
        s.listen(1)
        try:
            conn, _ = s.accept()
            while True:
                try:
                    if not conn.recv(1024*1024): break
                except: break
            conn.close()
        except: pass
        s.close()

    t = threading.Thread(target=sink_server, daemon=True)
    t.start()
    time.sleep(0.5)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(('localhost', PORT))

        data = np.random.rand(16, 16).astype(np.float32)
        COUNT = 10000
        
        print(f"Sending {COUNT} tensors (1KB each) over localhost TCP...")

        # Optimized write_stream
        t0 = time.perf_counter()
        for _ in range(COUNT):
            tenso.write_stream(data, client, check_integrity=USE_INTEGRITY)
        t_total = time.perf_counter() - t0
        
        client.close()
        
        print(f"Total Time: {t_total:.4f}s")
        print(f"Throughput: {COUNT/t_total:.0f} packets/sec")
        print(f"Latency:    {(t_total/COUNT)*1_000_000:.1f} µs/packet")
    except:
        print("Network write benchmark failed. Connection issues.")

def run_memory_overhead():
    print("\n" + "="*80)
    print(f"BENCHMARK 5: MEMORY OVERHEAD (Integrity: {USE_INTEGRITY})")
    print("="*80)
    
    shapes = [
        (1000, 1000),
        (2000, 2000), 
        (4000, 4000)
    ]
    
    print(f"{'SIZE':<15} | {'FORMAT':<12} | {'RAW (MB)':<10} | {'SERIALIZED (MB)':<15} | {'OVERHEAD':<10}")
    print("-" * 80)
    
    for shape in shapes:
        data = np.random.rand(*shape).astype(np.float32)
        raw_size = data.nbytes / 1024 / 1024
        
        results = {}
        
        # Tenso
        packet = tenso.dumps(data, check_integrity=USE_INTEGRITY)
        results['Tenso'] = len(packet) / 1024 / 1024
        
        # Pickle
        packet = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        results['Pickle'] = len(packet) / 1024 / 1024
        
        # Arrow
        if 'pa' in globals():
            arr = pa.array(data.flatten())
            batch = pa.RecordBatch.from_arrays([arr], names=['t'])
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, batch.schema) as writer:
                writer.write_batch(batch)
            results['Arrow'] = len(sink.getvalue()) / 1024 / 1024
        
        # SafeTensors
        if 'st_save' in globals():
            packet = st_save({"t": data})
            results['Safetensors'] = len(packet) / 1024 / 1024
        
        for name, size in results.items():
            overhead = ((size - raw_size) / raw_size) * 100
            print(f"{str(shape):<15} | {name:<12} | {raw_size:>8.2f} | {size:>13.2f} | {overhead:>8.2f}%")
        
        print("-" * 80)

def run_cpu_usage():
    print("\n" + "="*80)
    print(f"BENCHMARK 6: CPU USAGE (Integrity: {USE_INTEGRITY})")
    print("="*80)
    
    data = np.random.rand(4096, 4096).astype(np.float32)
    monitor = ResourceMonitor()
    
    print(f"{'FORMAT':<15} | {'SERIALIZE CPU%':<15} | {'DESERIALIZE CPU%':<18}")
    print("-" * 60)
    
    formats = {
        'Tenso': (lambda x: tenso.dumps(x, check_integrity=USE_INTEGRITY), tenso.loads),
        'Pickle': (lambda x: pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL), pickle.loads)
    }
    
    if 'pa' in globals():
        def arrow_enc(x):
            arr = pa.array(x.flatten())
            batch = pa.RecordBatch.from_arrays([arr], names=['t'])
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, batch.schema) as writer:
                writer.write_batch(batch)
            return sink.getvalue()
        
        def arrow_dec(x):
            with pa.ipc.open_stream(x) as reader:
                batch = reader.read_next_batch()
                return batch.column(0).to_numpy(zero_copy_only=False).reshape(data.shape)
        
        formats['Arrow'] = (arrow_enc, arrow_dec)
    
    if 'st_save' in globals():
        formats['Safetensors'] = (lambda x: st_save({"t": x}), lambda x: st_load(x)["t"])
    
    for name, (enc, dec) in formats.items():
        # Warmup
        packet = enc(data)
        _ = dec(packet)
        
        # Measure serialization CPU
        monitor.process.cpu_percent()  # Reset
        time.sleep(0.1)
        for _ in range(20):
            _ = enc(data)
        ser_cpu = monitor.process.cpu_percent()
        
        # Measure deserialization CPU
        monitor.process.cpu_percent()  # Reset
        time.sleep(0.1)
        for _ in range(20):
            _ = dec(packet)
        des_cpu = monitor.process.cpu_percent()
        
        print(f"{name:<15} | {ser_cpu:>13.1f}% | {des_cpu:>16.1f}%")

async def _bench_async_throughput(data, count=1000):
    """Measure async write/read throughput."""
    
    # 1. Async Write Benchmark
    # We write to a null sink to measure purely the serialization + overhead
    class NullWriter:
        def write(self, d): pass
        async def drain(self): pass
    
    writer = NullWriter()
    t0 = time.perf_counter()
    for _ in range(count):
        await awrite_stream(data, writer, check_integrity=USE_INTEGRITY)
    t_write = (time.perf_counter() - t0)
    
    return t_write

def run_async_benchmark():
    print("\n" + "="*80)
    print(f"BENCHMARK: ASYNC I/O (Integrity: {USE_INTEGRITY})")
    print("="*80)
    
    if not HAS_ASYNC:
        print("Async core not available.")
        return

    shape = (128, 128) # Small-ish tensor
    data = np.random.rand(*shape).astype(np.float32)
    count = 5000
    total_mb = (data.nbytes * count) / (1024**2)
    
    print(f"Streaming {count} tensors of shape {shape} ({data.nbytes/1024:.1f} KB each)...")
    
    # Run Async Write
    t_write = asyncio.run(_bench_async_throughput(data, count))
    
    fps = count / t_write
    mbps = total_mb / t_write
    
    print(f"{'Async Write':<20} | {t_write:>8.4f}s | {fps:>10.0f} tensors/s | {mbps:>8.2f} MB/s")
    print("-" * 80)

def run_dtype_coverage():
    print("\n" + "="*80)
    print(f"BENCHMARK 7: DTYPE COVERAGE (Integrity: {USE_INTEGRITY})")
    print("="*80)
    
    dtypes = [
        np.float16, np.float32, np.float64,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.bool_, np.complex64, np.complex128
    ]
    
    # Try adding bfloat16
    try:
        from ml_dtypes import bfloat16
        dtypes.append(bfloat16)
    except ImportError:
        pass
    
    shape = (100, 100)
    print(f"{'DTYPE':<25} | {'STATUS':<10} | {'SIZE (KB)':<12} | {'ROUNDTRIP (ms)':<15}")
    print("-" * 75)
    
    for dtype in dtypes:
        try:
            if dtype == np.bool_:
                data = np.random.randint(0, 2, shape).astype(dtype)
            elif hasattr(dtype, 'name') and dtype.name == 'bfloat16':
                 data = np.random.randn(*shape).astype(np.float32).astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                data = np.random.randint(0, 100, shape).astype(dtype)
            elif np.issubdtype(dtype, np.complexfloating):
                data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
            else:
                data = np.random.randn(*shape).astype(dtype)
            
            t0 = time.perf_counter()
            packet = tenso.dumps(data, check_integrity=USE_INTEGRITY)
            restored = tenso.loads(packet)
            roundtrip = (time.perf_counter() - t0) * 1000
            
            status = "✓ PASS" if np.array_equal(data, restored) else "✗ FAIL"
            
            print(f"{str(dtype):<25} | {status:<10} | {len(packet)/1024:>10.2f} | {roundtrip:>13.3f}")
        except Exception as e:
            print(f"{str(dtype):<25} | {'✗ ERROR':<10} | {'-':<12} | {str(e)[:20]}")

def run_arrow_comparison():
    print("\n" + "="*80)
    print(f"BENCHMARK 8: ARROW vs TENSO (Integrity: {USE_INTEGRITY})")
    print("="*80)
    
    if 'pa' not in globals():
        print("Arrow not available. Skipping.")
        return
    
    sizes = [
        ("Small", (512, 512)),
        ("Medium", (2048, 2048)),
        ("Large", (4096, 4096)),
        ("XLarge", (8192, 8192))
    ]
    
    print(f"{'SIZE':<10} | {'TENSO SER':<12} | {'ARROW SER':<12} | {'TENSO DES':<12} | {'ARROW DES':<12} | {'SPEEDUP':<10}")
    print("-" * 95)
    
    for name, shape in sizes:
        data = np.random.rand(*shape).astype(np.float32)
        
        # Tenso
        t0 = time.perf_counter()
        tenso_packet = tenso.dumps(data, check_integrity=USE_INTEGRITY)
        tenso_ser = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        _ = tenso.loads(tenso_packet)
        tenso_des = (time.perf_counter() - t0) * 1000
        
        # Arrow
        arr = pa.array(data.flatten())
        batch = pa.RecordBatch.from_arrays([arr], names=['t'])
        
        t0 = time.perf_counter()
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
        arrow_packet = sink.getvalue()
        arrow_ser = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        with pa.ipc.open_stream(arrow_packet) as reader:
            batch = reader.read_next_batch()
            _ = batch.column(0).to_numpy(zero_copy_only=False).reshape(shape)
        arrow_des = (time.perf_counter() - t0) * 1000
        
        speedup = arrow_des / tenso_des
        
        print(f"{name:<10} | {tenso_ser:>10.3f}ms | {arrow_ser:>10.3f}ms | {tenso_des:>10.3f}ms | {arrow_des:>10.3f}ms | {speedup:>8.1f}x")
    
    print("-" * 95)

def run_correctness():
    print("\n" + "="*80)
    print(f"BENCHMARK 9: CORRECTNESS (Integrity: {USE_INTEGRITY})")
    print("="*80)
    
    tests = [
        ("Zeros", lambda: np.zeros((100, 100), dtype=np.float32)),
        ("Ones", lambda: np.ones((100, 100), dtype=np.float32)),
        ("Random", lambda: np.random.rand(100, 100).astype(np.float32)),
        ("Negative", lambda: -np.random.rand(100, 100).astype(np.float32)),
        ("Mixed", lambda: np.random.randn(100, 100).astype(np.float32)),
        ("Large Values", lambda: np.random.rand(100, 100).astype(np.float32) * 1e6),
        ("Small Values", lambda: np.random.rand(100, 100).astype(np.float32) * 1e-6),
    ]
    
    print(f"{'TEST':<20} | {'STATUS':<10} | {'MAX ERROR':<15}")
    print("-" * 55)
    
    for name, gen_func in tests:
        data = gen_func()
        packet = tenso.dumps(data, check_integrity=USE_INTEGRITY)
        restored = tenso.loads(packet)
        
        if np.allclose(data, restored, rtol=1e-7, atol=1e-7):
            max_error = np.abs(data - restored).max()
            print(f"{name:<20} | {'✓ PASS':<10} | {max_error:<15.2e}")
        else:
            print(f"{name:<20} | {'✗ FAIL':<10} | {'MISMATCH':<15}")
    
    print("-" * 55)

def print_summary():
    """Print system info and summary."""
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"NumPy Version:  {np.__version__}")
    print(f"Tenso Version:  {tenso.__version__ if hasattr(tenso, '__version__') else 'unknown'}")
    print(f"CPU Cores:      {psutil.cpu_count()}")
    print(f"Total Memory:   {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Platform:       {sys.platform}")
    
    if 'pa' in globals():
        print(f"Arrow Version:  {pa.__version__}")
    print("="*80)

# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tenso Comprehensive Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark Modes:
  all         - Run all benchmarks (default)
  ser         - Serialization speed
  io          - Disk I/O performance
  read        - Stream reading
  write       - Network writing
  memory      - Memory overhead
  cpu         - CPU usage
  dtypes      - Dtype coverage
  arrow       - Arrow comparison
  correctness - Data integrity
  quick       - Quick overview (ser + arrow)
        """
    )
    
    parser.add_argument(
        "mode", 
        nargs="?", 
        choices=["all", "ser", "io", "read", "write", "memory", "cpu", "dtypes", "arrow", "correctness", "quick"], 
        default="all",
        help="Benchmark mode to run"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip system information summary"
    )

    parser.add_argument(
        "--integrity",
        action="store_true",
        help="Enable XXH3 integrity checks for Tenso benchmarks"
    )
    
    args = parser.parse_args()
    
    # Set the global flag
    USE_INTEGRITY = args.integrity
    
    if not args.no_summary:
        print_summary()
    
    if args.mode == "all":
        run_serialization()
        run_io()
        run_stream_read()
        run_stream_write()
        run_memory_overhead()
        run_cpu_usage()
        run_dtype_coverage()
        run_arrow_comparison()
        run_correctness()
        run_async_benchmark()
        run_dtype_coverage()
    elif args.mode == "async":
        run_async_benchmark()
    elif args.mode == "dtypes":
        run_dtype_coverage()
    elif args.mode == "quick":
        run_serialization()
        run_arrow_comparison()
    elif args.mode == "ser":
        run_serialization()
    elif args.mode == "io":
        run_io()
    elif args.mode == "read":
        run_stream_read()
    elif args.mode == "write":
        run_stream_write()
    elif args.mode == "memory":
        run_memory_overhead()
    elif args.mode == "cpu":
        run_cpu_usage()
    elif args.mode == "arrow":
        run_arrow_comparison()
    elif args.mode == "correctness":
        run_correctness()
    
    print("\n" + "="*80)
    print("BENCHMARKS COMPLETE")
    print("="*80)