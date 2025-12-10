import time
import numpy as np
import tenso
import pickle
import tempfile
import os

def benchmark_final():
    print(f"{'FORMAT':<15} | {'SIZE':<10} | {'WRITE (ms)':<10} | {'READ (ms)':<10}")
    print("-" * 60)

    # 256 MB Matrix
    shape = (8192, 8192) 
    data = np.random.rand(*shape).astype(np.float32)
    size_mb = data.nbytes / (1024 * 1024)
    size_str = f"{size_mb:.0f} MB"

    # --- Tenso (Optimized Dump + mmap) ---
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    
    try:
        t0 = time.perf_counter()
        with open(path, "wb") as f:
            tenso.dump(data, f)
        t_write = (time.perf_counter() - t0) * 1000

        # Measure Load Time ONLY (excluding file close/OS cleanup)
        with open(path, "rb") as f:
            t0 = time.perf_counter()
            res = tenso.load(f, mmap_mode=True)
            t_read = (time.perf_counter() - t0) * 1000
            
        print(f"{'Tenso':<15} | {size_str:<10} | {t_write:>10.2f} | {t_read:>10.2f}")
    finally:
        if os.path.exists(path):
            os.remove(path)

    # --- Numpy Native (.npy) ---
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as f:
        path = f.name
    
    try:
        t0 = time.perf_counter()
        np.save(path, data)
        t_write = (time.perf_counter() - t0) * 1000

        # Numpy Native mmap
        t0 = time.perf_counter()
        res = np.load(path, mmap_mode='r')
        t_read = (time.perf_counter() - t0) * 1000

        print(f"{'Numpy .npy':<15} | {size_str:<10} | {t_write:>10.2f} | {t_read:>10.2f}")
    finally:
        if os.path.exists(path):
            os.remove(path)

    # --- Pickle ---
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    
    try:
        t0 = time.perf_counter()
        with open(path, "wb") as f:
            pickle.dump(data, f)
        t_write = (time.perf_counter() - t0) * 1000

        with open(path, "rb") as f:
            t0 = time.perf_counter()
            res = pickle.load(f)
            t_read = (time.perf_counter() - t0) * 1000

        print(f"{'Pickle':<15} | {size_str:<10} | {t_write:>10.2f} | {t_read:>10.2f}")
    finally:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    benchmark_final()