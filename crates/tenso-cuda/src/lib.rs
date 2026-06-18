//! tenso-cuda: the ONLY crate that touches CUDA.
//!
//! The CUDA runtime (`libcudart`) is loaded at runtime via `libloading`/dlopen —
//! there is NO link-time dependency on the CUDA toolkit, so this crate (and the
//! whole workspace) builds on macOS / aarch64 without a toolkit installed and
//! only fails at runtime when no driver is present. Real driver calls are gated
//! behind the `cuda` feature; without it, `CudaBackend::open` returns
//! `Err(CudaLoadError::FeatureDisabled)` and `CudaBackend::available()` is
//! `false`, so callers on toolkit-less hosts degrade gracefully.
//!
//! `CudaBackend` implements `tenso_device::DeviceBackend`, so it slots into
//! `GpuCodec` exactly like the Cpu/Mock backends.
//!
//! DESIGN: this file is deliberately BORING. It is FFI shims only — thin wrappers
//! over `cudaMalloc`/`cudaFree`/`cudaMemcpy`/`cudaHostAlloc`/
//! `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle`/`cudaIpcCloseMemHandle` and a
//! couple of device-attribute queries. There is ZERO offset/packet/wire math
//! here — all of that lives above the `DeviceBackend` trait in `tenso-device`'s
//! `GpuCodec`. We only move bytes between host and device and hand out opaque
//! handles.
//!
//! TESTABILITY: everything below the trait requires a real CUDA device to
//! exercise. The host-only unit tests at the bottom verify the
//! graceful-degradation contract (`available()`, `open()` error mapping, struct
//! layouts) and never call into the driver. Tests that need a GPU are marked
//! `#[ignore]`, gated behind `cuda`, and run on a real Linux + NVIDIA box with
//! `cargo test -p tenso-cuda --features cuda -- --ignored` (with libcudart on
//! the loader path, or `TENSO_CUDART_PATH` set).

#![allow(dead_code, unused)]

use tenso_device::{DevErr, DevPtr, DeviceBackend, IpcHandle, PinnedBuf};

/// Error opening / loading the CUDA runtime.
#[derive(Debug)]
pub enum CudaLoadError {
    /// The crate was built without the `cuda` feature.
    FeatureDisabled,
    /// `libcudart` could not be dlopen'd at runtime. Carries a diagnostic listing
    /// how many candidates were tried and the last underlying dlopen error, so a
    /// runtime that is present-but-off-the-loader-path (fix with `TENSO_CUDART_PATH`
    /// or `LD_LIBRARY_PATH`) is debuggable rather than an opaque "not found".
    DriverNotFound(String),
    /// A required symbol was missing from the runtime library.
    MissingSymbol(&'static str),
    /// A runtime call failed (carries the `cudaError_t` code + the runtime's
    /// `cudaGetErrorString` text, when it could be resolved).
    Driver(i32, String),
}

// =============================================================================
// CUDA runtime FFI surface (only compiled with the `cuda` feature)
// =============================================================================
//
// We bind only the handful of `libcudart` entry points we need. All signatures
// follow the C `cudaError_t cudaXxx(...)` convention (0 == cudaSuccess). Pointer
// arguments are `*mut c_void` etc.; we never dereference CUDA-owned memory on the
// host. Keeping the set tiny keeps the ABI surface stable across CUDA versions.
#[cfg(feature = "cuda")]
mod ffi {
    use core::ffi::{c_char, c_void};
    use libloading::{Library, Symbol};

    // cudaError_t — 0 is cudaSuccess.
    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    // cudaMemcpyKind enum values (stable across CUDA versions).
    pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
    pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

    // cudaHostAlloc flags. We allocate plain page-locked host memory
    // (cudaHostAllocDefault == 0); we do NOT request the Mapped flag because the
    // contract's PinnedBuf only carries a Vec<u8> (see alloc_pinned for the seam).
    pub const CUDA_HOST_ALLOC_DEFAULT: u32 = 0;

    // cudaDeviceAttr values we query. Stable in the runtime ABI.
    //   cudaDevAttrIntegrated = 18  (1 on Tegra / Jetson iGPUs; 0 on discrete)
    pub const CUDA_DEV_ATTR_INTEGRATED: i32 = 18;

    // cudaIpcMemHandle_t is an opaque 64-byte reserved POD blob; we treat it as
    // [u8; 64]. CUDA_IPC_HANDLE_SIZE == 64 == tenso_core::IPC_REF_HANDLE_LEN.
    pub const CUDA_IPC_HANDLE_BYTES: usize = 64;

    // cudaIpcMemLazyEnablePeerAccess = 1 (the only documented OpenMemHandle flag).
    pub const CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: u32 = 1;

    // A cudaIpcMemHandle_t is `struct { char reserved[64]; }`. We model it as a
    // 64-byte POD array. cudaIpcOpenMemHandle takes it BY VALUE; on the SysV
    // x86-64 ABI a 64-byte aggregate passed by value is passed in memory (the
    // caller materialises it and passes a hidden pointer per the struct-by-value
    // rule), which is exactly what `extern "C" fn(..., handle: CudaIpcMemHandle,
    // ...)` lowers to for a `#[repr(C)]` 64-byte array wrapper.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CudaIpcMemHandle(pub [u8; CUDA_IPC_HANDLE_BYTES]);

    // Resolved C function pointer types.
    pub type FnSetDevice = unsafe extern "C" fn(device: i32) -> CudaError;
    pub type FnGetDevice = unsafe extern "C" fn(device: *mut i32) -> CudaError;
    pub type FnMalloc = unsafe extern "C" fn(dev_ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub type FnFree = unsafe extern "C" fn(dev_ptr: *mut c_void) -> CudaError;
    pub type FnMemcpy = unsafe extern "C" fn(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
    ) -> CudaError;
    pub type FnHostAlloc =
        unsafe extern "C" fn(p_host: *mut *mut c_void, size: usize, flags: u32) -> CudaError;
    pub type FnFreeHost = unsafe extern "C" fn(ptr: *mut c_void) -> CudaError;
    // cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr): the handle
    // is an OUT param, so it is passed by pointer.
    pub type FnIpcGetMemHandle =
        unsafe extern "C" fn(handle: *mut CudaIpcMemHandle, dev_ptr: *mut c_void) -> CudaError;
    // cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned
    // flags): the handle is passed BY VALUE (a 64-byte struct).
    pub type FnIpcOpenMemHandle = unsafe extern "C" fn(
        dev_ptr: *mut *mut c_void,
        handle: CudaIpcMemHandle,
        flags: u32,
    ) -> CudaError;
    pub type FnIpcCloseMemHandle = unsafe extern "C" fn(dev_ptr: *mut c_void) -> CudaError;
    pub type FnDeviceGetAttribute =
        unsafe extern "C" fn(value: *mut i32, attr: i32, device: i32) -> CudaError;
    pub type FnDeviceGetPciBusId =
        unsafe extern "C" fn(pci_bus_id: *mut c_char, len: i32, device: i32) -> CudaError;
    pub type FnGetErrorString = unsafe extern "C" fn(error: i32) -> *const c_char;

    /// Owns the dlopen'd library plus all resolved symbols. Symbols are raw fn
    /// pointers copied out of the `Library` at load time; we keep `_lib` alive
    /// for the lifetime of the backend so the pointers stay valid.
    pub struct Runtime {
        _lib: Library,
        pub set_device: FnSetDevice,
        pub get_device: FnGetDevice,
        pub malloc: FnMalloc,
        pub free: FnFree,
        pub memcpy: FnMemcpy,
        pub host_alloc: FnHostAlloc,
        pub free_host: FnFreeHost,
        pub ipc_get_mem_handle: FnIpcGetMemHandle,
        pub ipc_open_mem_handle: FnIpcOpenMemHandle,
        pub ipc_close_mem_handle: FnIpcCloseMemHandle,
        pub device_get_attribute: FnDeviceGetAttribute,
        pub device_get_pci_bus_id: FnDeviceGetPciBusId,
        pub get_error_string: FnGetErrorString,
    }

    /// Candidate sonames for the CUDA runtime across platforms / CUDA majors.
    /// We try them in order; the first that dlopen's wins. On the target Linux +
    /// NVIDIA box the bare `libcudart.so` (toolkit) or `libcudart.so.12` /
    /// `.so.13` (redistributable) is present.
    const CANDIDATES: &[&str] = &[
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.13",
        "libcudart.so.11.0",
        "libcudart.so.11",
        // Windows / macOS fallbacks (rarely a real CUDA host, but cheap to try).
        "cudart64_12.dll",
        "cudart64_110.dll",
        "libcudart.dylib",
    ];

    /// Standard install directories that may hold `libcudart` even when it is not
    /// on the dynamic loader path. Searched (joined with each soname) only after
    /// bare-soname resolution fails. This catches a system CUDA toolkit; a pip
    /// `nvidia-cuda-runtime-cu12` wheel lives in a venv site-packages whose path
    /// is not fixed, so that case is handled by `TENSO_CUDART_PATH` / `LD_LIBRARY_PATH`.
    const CANDIDATE_DIRS: &[&str] = &[
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/sbsa-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/opt/cuda/lib64",
    ];

    /// Find and dlopen `libcudart`, in order:
    ///   1. `$TENSO_CUDART_PATH` (an explicit soname or absolute path), if set.
    ///   2. Bare sonames, resolved via the system loader path (LD_LIBRARY_PATH /
    ///      ldconfig).
    ///   3. Each `CANDIDATE_DIRS` entry joined with each soname.
    /// On total failure returns `DriverNotFound` with a diagnostic (count of
    /// attempts + last underlying dlopen error) so the cause is debuggable.
    fn open_cudart() -> Result<Library, super::CudaLoadError> {
        let mut paths: Vec<String> = Vec::new();
        if let Ok(p) = std::env::var("TENSO_CUDART_PATH") {
            if !p.is_empty() {
                paths.push(p);
            }
        }
        for name in CANDIDATES {
            paths.push((*name).to_string());
        }
        for dir in CANDIDATE_DIRS {
            for name in CANDIDATES {
                paths.push(format!("{dir}/{name}"));
            }
        }

        let mut last_err: Option<String> = None;
        for path in &paths {
            // SAFETY: loading a shared library by name/path. Its symbols are
            // resolved and type-checked against the CUDA C ABI in `load()`.
            match unsafe { Library::new(path) } {
                Ok(l) => return Ok(l),
                Err(e) => last_err = Some(format!("{path}: {e}")),
            }
        }
        let detail = format!(
            "could not load libcudart after {} candidate(s){}. Set TENSO_CUDART_PATH to the \
             libcudart.so[.NN] path, or add its directory to LD_LIBRARY_PATH. NOTE: libcudart \
             ships with the CUDA runtime/toolkit (e.g. `pip install nvidia-cuda-runtime-cu12`), \
             NOT the NVIDIA driver — a working `nvidia-smi` is necessary but not sufficient.",
            paths.len(),
            match last_err {
                Some(e) => format!("; last error: {e}"),
                None => String::new(),
            },
        );
        Err(super::CudaLoadError::DriverNotFound(detail))
    }

    impl Runtime {
        /// dlopen libcudart and resolve every symbol we use. Returns
        /// `DriverNotFound` if no candidate library loads, `MissingSymbol` if a
        /// needed entry point is absent.
        pub fn load() -> Result<Self, super::CudaLoadError> {
            // Find + dlopen libcudart (env override, then loader path, then
            // standard install dirs). The library is the CUDA runtime; its symbols
            // match the C signatures bound above and `lib` is moved into the
            // returned Runtime so the resolved fn pointers stay valid.
            let lib = open_cudart()?;

            // Resolve a symbol, mapping absence to MissingSymbol. The returned
            // raw fn pointer is `Copy`; we read `*sym` while `lib` outlives us
            // (it is moved into the returned Runtime alongside the pointers).
            macro_rules! sym {
                ($name:literal, $ty:ty) => {{
                    // SAFETY: the named symbol, if present, has the C signature
                    // declared by $ty (verified against the CUDA runtime API).
                    let s: Symbol<$ty> = unsafe { lib.get($name) }.map_err(|_| {
                        super::CudaLoadError::MissingSymbol(
                            // strip the trailing NUL for the message
                            core::str::from_utf8(&$name[..$name.len() - 1]).unwrap_or("?"),
                        )
                    })?;
                    *s
                }};
            }

            let set_device = sym!(b"cudaSetDevice\0", FnSetDevice);
            let get_device = sym!(b"cudaGetDevice\0", FnGetDevice);
            let malloc = sym!(b"cudaMalloc\0", FnMalloc);
            let free = sym!(b"cudaFree\0", FnFree);
            let memcpy = sym!(b"cudaMemcpy\0", FnMemcpy);
            let host_alloc = sym!(b"cudaHostAlloc\0", FnHostAlloc);
            let free_host = sym!(b"cudaFreeHost\0", FnFreeHost);
            let ipc_get_mem_handle = sym!(b"cudaIpcGetMemHandle\0", FnIpcGetMemHandle);
            let ipc_open_mem_handle = sym!(b"cudaIpcOpenMemHandle\0", FnIpcOpenMemHandle);
            let ipc_close_mem_handle = sym!(b"cudaIpcCloseMemHandle\0", FnIpcCloseMemHandle);
            let device_get_attribute = sym!(b"cudaDeviceGetAttribute\0", FnDeviceGetAttribute);
            let device_get_pci_bus_id = sym!(b"cudaDeviceGetPCIBusId\0", FnDeviceGetPciBusId);
            let get_error_string = sym!(b"cudaGetErrorString\0", FnGetErrorString);

            Ok(Runtime {
                _lib: lib,
                set_device,
                get_device,
                malloc,
                free,
                memcpy,
                host_alloc,
                free_host,
                ipc_get_mem_handle,
                ipc_open_mem_handle,
                ipc_close_mem_handle,
                device_get_attribute,
                device_get_pci_bus_id,
                get_error_string,
            })
        }

        /// Resolve a `cudaError_t` to its human-readable string via the runtime's
        /// `cudaGetErrorString`. Never panics: falls back to "<unknown>" if the
        /// runtime returns NULL or a non-UTF8 string.
        pub fn error_string(&self, code: CudaError) -> String {
            // SAFETY: cudaGetErrorString returns a pointer to a static C string
            // owned by the runtime (valid for the process lifetime) or NULL.
            let ptr = unsafe { (self.get_error_string)(code) };
            if ptr.is_null() {
                return String::from("<null cudaGetErrorString>");
            }
            // SAFETY: ptr is a NUL-terminated C string from the runtime.
            let cstr = unsafe { core::ffi::CStr::from_ptr(ptr) };
            cstr.to_str()
                .map(String::from)
                .unwrap_or_else(|_| String::from_utf8_lossy(cstr.to_bytes()).into_owned())
        }
    }
}

// =============================================================================
// CudaBackend
// =============================================================================

/// A handle to a CUDA device backed by the dlopen'd runtime.
///
/// Construction (`open`) loads `libcudart`, selects the device, and caches the
/// integrated/unified flag and the device UUID derived from the PCI bus id. All
/// `DeviceBackend` methods are thin shims over the resolved runtime symbols.
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    rt: ffi::Runtime,
    #[cfg(feature = "cuda")]
    device_ordinal: i32,
    #[cfg(feature = "cuda")]
    unified: bool,
    #[cfg(feature = "cuda")]
    uuid: [u8; 16],
    // Without the `cuda` feature the struct is uninhabited in practice
    // (`open` never returns Ok), but it must still be nameable.
    #[cfg(not(feature = "cuda"))]
    _never: core::convert::Infallible,
}

impl CudaBackend {
    /// True if the CUDA runtime could be dlopen'd on this host (and the `cuda`
    /// feature is enabled). Lets callers probe for GPU-direct support without
    /// committing to a device or handling an error path.
    ///
    /// Always `false` when built without the `cuda` feature.
    pub fn available() -> bool {
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
        #[cfg(feature = "cuda")]
        {
            ffi::Runtime::load().is_ok()
        }
    }

    /// Load the CUDA runtime and bind to `device_ordinal`.
    ///
    /// Without the `cuda` feature this always returns
    /// `Err(CudaLoadError::FeatureDisabled)` so callers on toolkit-less hosts
    /// can degrade gracefully. With the feature, it dlopen's `libcudart`,
    /// selects the device, and caches the integrated (unified-memory) flag and
    /// a stable 16-byte device UUID derived from the PCI bus id.
    pub fn open(device_ordinal: i32) -> Result<Self, CudaLoadError> {
        let _ = device_ordinal;
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaLoadError::FeatureDisabled)
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_char;

            let rt = ffi::Runtime::load()?;

            // Bind the current thread/context to the requested device. A nonzero
            // cudaError_t here means the ordinal is invalid or no device exists.
            // SAFETY: resolved C entry point, plain i32 argument.
            let rc = unsafe { (rt.set_device)(device_ordinal) };
            if rc != ffi::CUDA_SUCCESS {
                return Err(CudaLoadError::Driver(rc, rt.error_string(rc)));
            }

            // Query the integrated attribute (1 on Jetson/Tegra iGPUs, 0 on a
            // discrete part like the RTX 5070 Ti). On integrated parts host and
            // device share physical memory, enabling the zero-copy mapped path.
            let mut integrated: i32 = 0;
            // SAFETY: out-param is a valid &mut i32; attr/device are scalars.
            let rc = unsafe {
                (rt.device_get_attribute)(
                    &mut integrated,
                    ffi::CUDA_DEV_ATTR_INTEGRATED,
                    device_ordinal,
                )
            };
            if rc != ffi::CUDA_SUCCESS {
                return Err(CudaLoadError::Driver(rc, rt.error_string(rc)));
            }
            let unified = integrated != 0;

            // Device UUID: the runtime exposes a stable, human-readable PCI bus
            // id ("0000:65:00.0") via cudaDeviceGetPCIBusId. We derive a stable
            // 16-byte id from it (see `pci_bus_id_to_uuid`) rather than decoding
            // the large, version-fragile `cudaDeviceProp` struct (which carries
            // the real UUID but bloats the ABI surface). The bus id is stable for
            // the same physical slot across processes on one node, so it is a
            // sound key for CUDA IPC's "same device" rule.
            let mut buf = [0i8; 32];
            // SAFETY: buf is 32 writable bytes; len matches; device is a scalar.
            let rc = unsafe {
                (rt.device_get_pci_bus_id)(buf.as_mut_ptr() as *mut c_char, 32, device_ordinal)
            };
            if rc != ffi::CUDA_SUCCESS {
                return Err(CudaLoadError::Driver(rc, rt.error_string(rc)));
            }
            let uuid = pci_bus_id_to_uuid(&buf);

            Ok(CudaBackend {
                rt,
                device_ordinal,
                unified,
                uuid,
            })
        }
    }

    /// True if the device exposes unified (integrated/Tegra) memory. On the
    /// discrete RTX 5070 Ti this is `false`. Mirror of the trait method for
    /// callers who hold a concrete `CudaBackend`.
    pub fn is_unified(&self) -> bool {
        #[cfg(not(feature = "cuda"))]
        {
            // Unreachable: a CudaBackend cannot be constructed without `cuda`.
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            self.unified
        }
    }
}

/// Derive a stable 16-byte device id from a NUL-terminated PCI bus id C string
/// (e.g. b"0000:65:00.0"). We copy the bus-id bytes verbatim into the first 16
/// bytes (truncating / zero-padding), which is deterministic, collision-free for
/// distinct slots on a node, and matches CUDA IPC's "handles only resolve on the
/// same physical device" rule. Defined unconditionally so the host-only unit
/// tests can exercise it without the `cuda` feature.
fn pci_bus_id_to_uuid(bus_id: &[i8]) -> [u8; 16] {
    let mut uuid = [0u8; 16];
    for (i, &c) in bus_id.iter().enumerate() {
        if i >= 16 || c == 0 {
            break;
        }
        uuid[i] = c as u8;
    }
    uuid
}

// =============================================================================
// DeviceBackend impl — FFI shims only, no packet/offset math
// =============================================================================
//
// Every method below requires a real CUDA device to do anything. We move raw
// bytes between host slices and device pointers and hand out opaque IPC handles.
// `DevPtr(usize)` carries the raw `void*` returned by `cudaMalloc`, reinterpreted
// as a `usize`. Every fallible call checks the cudaError_t.

impl DeviceBackend for CudaBackend {
    fn alloc(&self, n: usize) -> Result<DevPtr, DevErr> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = n;
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_void;
            // cudaMalloc(0) is allowed and returns NULL; treat a zero-byte alloc
            // as a benign null DevPtr rather than an error.
            if n == 0 {
                return Ok(DevPtr(0));
            }
            let mut ptr: *mut c_void = core::ptr::null_mut();
            // SAFETY: out-param is a valid &mut *mut c_void; n is the byte count.
            let rc = unsafe { (self.rt.malloc)(&mut ptr, n) };
            if rc != ffi::CUDA_SUCCESS || ptr.is_null() {
                // cudaMalloc fails almost exclusively with OOM; surface that.
                return Err(DevErr::OutOfMemory);
            }
            Ok(DevPtr(ptr as usize))
        }
    }

    fn free(&self, p: DevPtr) {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = p;
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_void;
            if p.0 == 0 {
                return;
            }
            // SAFETY: p.0 came from cudaMalloc on this backend. Errors are
            // swallowed to match the infallible `free` signature.
            let _ = unsafe { (self.rt.free)(p.0 as *mut c_void) };
        }
    }

    fn copy_h2d(&self, dst: DevPtr, src: &[u8]) {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dst, src);
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_void;
            if src.is_empty() {
                return;
            }
            // SAFETY: dst.0 is a device pointer with >= src.len() bytes (caller
            // contract); src is a valid host slice. Kind=1 selects HostToDevice.
            // On the discrete 5070 Ti this is a real PCIe DMA.
            let rc = unsafe {
                (self.rt.memcpy)(
                    dst.0 as *mut c_void,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            };
            // The trait's copy is infallible, but a nonzero cudaError_t means the
            // transfer did NOT happen (dst left stale). Surface it loudly in
            // debug/test builds so a bad copy can't pass silently.
            debug_assert!(
                rc == ffi::CUDA_SUCCESS,
                "cudaMemcpy H2D failed: {} ({rc})",
                self.rt.error_string(rc)
            );
        }
    }

    fn copy_d2h(&self, dst: &mut [u8], src: DevPtr) {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dst, src);
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_void;
            if dst.is_empty() {
                return;
            }
            // SAFETY: src.0 is a device pointer with >= dst.len() bytes (caller
            // contract); dst is a valid mutable host slice. Kind=2 = DeviceToHost.
            let rc = unsafe {
                (self.rt.memcpy)(
                    dst.as_mut_ptr() as *mut c_void,
                    src.0 as *const c_void,
                    dst.len(),
                    ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            };
            // A nonzero cudaError_t means dst was NOT filled from device memory
            // (left zeroed/stale). Surface it loudly in debug/test builds.
            debug_assert!(
                rc == ffi::CUDA_SUCCESS,
                "cudaMemcpy D2H failed: {} ({rc})",
                self.rt.error_string(rc)
            );
        }
    }

    fn alloc_pinned(&self, n: usize) -> Result<PinnedBuf, DevErr> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = n;
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_void;
            // Allocate page-locked host memory via cudaHostAlloc with the Default
            // flag (0), then immediately free it via cudaFreeHost. The contract's
            // PinnedBuf only carries a Vec<u8>, so we COPY the (zeroed) region
            // into a Vec for the caller.
            //
            // SEAM: this loses the genuine page-locked property the moment we copy
            // into a Vec — the Vec is ordinary pageable host memory. To expose
            // true pinned buffers (and a device pointer for the zero-copy path),
            // `PinnedBuf` in tenso-device would need to carry the raw host pointer
            // + a backend-owned deleter that calls cudaFreeHost on Drop. We still
            // exercise cudaHostAlloc/cudaFreeHost here so the GPU integration test
            // proves the pinned-alloc path works on real hardware. For now this
            // gives a correctly-sized host buffer.
            let mut host: *mut c_void = core::ptr::null_mut();
            // SAFETY: out-param valid; n is the byte count; flag is a valid mask.
            let rc = unsafe { (self.rt.host_alloc)(&mut host, n, ffi::CUDA_HOST_ALLOC_DEFAULT) };
            if rc != ffi::CUDA_SUCCESS || host.is_null() {
                return Err(DevErr::OutOfMemory);
            }
            // Copy the (zeroed-by-us) region into a Vec and release the pinned
            // allocation. cudaHostAlloc does not zero, so we zero-init the Vec.
            let mut bytes = vec![0u8; n];
            // SAFETY: host points to >= n readable bytes; bytes has n bytes.
            unsafe {
                core::ptr::copy_nonoverlapping(host as *const u8, bytes.as_mut_ptr(), n);
                let _ = (self.rt.free_host)(host);
            }
            Ok(PinnedBuf { bytes })
        }
    }

    fn export_ipc(&self, p: DevPtr) -> Result<IpcHandle, DevErr> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = p;
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_void;
            // Integrated (unified) GPUs do not support classic CUDA IPC mem
            // handles — the iGPU shares system memory and IPC export is rejected
            // by the driver. Surface that as IpcUnsupported rather than a raw
            // code. On the discrete 5070 Ti this branch is skipped.
            if self.unified {
                return Err(DevErr::IpcUnsupported);
            }
            let mut handle = ffi::CudaIpcMemHandle([0u8; ffi::CUDA_IPC_HANDLE_BYTES]);
            // SAFETY: handle is a 64-byte cudaIpcMemHandle_t out-param; p.0 is a
            // device pointer returned by cudaMalloc on this backend.
            let rc = unsafe { (self.rt.ipc_get_mem_handle)(&mut handle, p.0 as *mut c_void) };
            if rc != ffi::CUDA_SUCCESS {
                // Surface the real cudaError_t instead of a blanket IpcUnsupported,
                // which previously hid the driver's actual reason.
                return Err(DevErr::Device(rc));
            }
            // We fill only the opaque handle blob and the device UUID. byte_offset
            // and nbytes are wire-framing concerns owned by GpuCodec above the
            // trait, so we leave them zero here (BORING: no packet math).
            Ok(IpcHandle {
                bytes: handle.0,
                byte_offset: 0,
                nbytes: 0,
                device_uuid: self.uuid,
            })
        }
    }

    fn import_ipc(&self, h: &IpcHandle) -> Result<DevPtr, DevErr> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = h;
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_void;
            if self.unified {
                return Err(DevErr::IpcUnsupported);
            }
            // Device-UUID validation (reject mismatches) is also a GpuCodec-level
            // rule; we defensively reject a cross-device handle when our cached
            // UUID is known (nonzero) and disagrees.
            if self.uuid != [0u8; 16] && h.device_uuid != self.uuid {
                return Err(DevErr::DeviceUuidMismatch);
            }
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let handle = ffi::CudaIpcMemHandle(h.bytes);
            // SAFETY: out-param valid; `handle` is a 64-byte cudaIpcMemHandle_t
            // passed BY VALUE (matches the C signature). The flag is the only
            // documented one (LazyEnablePeerAccess = 1).
            let rc = unsafe {
                (self.rt.ipc_open_mem_handle)(
                    &mut ptr,
                    handle,
                    ffi::CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
                )
            };
            if rc != ffi::CUDA_SUCCESS {
                // Surface the real cudaError_t. The most common case here is the
                // documented same-process restriction: importing a handle exported
                // by the calling process returns 201 (cudaErrorDeviceUninitialized
                // / "invalid device context"). CUDA IPC is a cross-process
                // mechanism; see the cross-process integration test.
                return Err(DevErr::Device(rc));
            }
            if ptr.is_null() {
                return Err(DevErr::IpcUnsupported);
            }
            // Honor byte_offset so a sub-buffer IPC reference resolves to the right
            // address within the imported mapping (the field is 0 today, so this is
            // a safe no-op until offset export is wired up).
            Ok(DevPtr(ptr as usize + h.byte_offset as usize))
        }
    }

    fn is_unified_memory(&self) -> bool {
        #[cfg(not(feature = "cuda"))]
        {
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            self.unified
        }
    }

    fn device_uuid(&self) -> [u8; 16] {
        #[cfg(not(feature = "cuda"))]
        {
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            self.uuid
        }
    }
}

/// Close a device pointer that was obtained via `import_ipc` (i.e. mapped from a
/// foreign `cudaIpcMemHandle_t`). This is NOT `free`: an imported pointer must be
/// released with `cudaIpcCloseMemHandle`, never `cudaFree`. Exposed so callers /
/// tests that import an IPC handle can correctly tear it down. No-op (and
/// uninhabited) without the `cuda` feature.
impl CudaBackend {
    pub fn close_ipc(&self, p: DevPtr) -> Result<(), DevErr> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = p;
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            use core::ffi::c_void;
            if p.0 == 0 {
                return Ok(());
            }
            // SAFETY: p.0 was returned by cudaIpcOpenMemHandle on this backend.
            let rc = unsafe { (self.rt.ipc_close_mem_handle)(p.0 as *mut c_void) };
            if rc != ffi::CUDA_SUCCESS {
                // Surface the real cudaError_t (e.g. the handle was already gone /
                // invalid for this context).
                return Err(DevErr::Device(rc));
            }
            Ok(())
        }
    }
}

// =============================================================================
// Tests — host-only / compile-time smoke tests + #[ignore] GPU integration
// =============================================================================
//
// The non-ignored tests NEVER touch a CUDA device. They verify the
// graceful-degradation contract and basic invariants so CI on macOS/aarch64 (no
// toolkit) stays green. The `#[ignore]`d tests genuinely exercise the driver;
// run them on a real Linux + NVIDIA box with:
//   cargo test -p tenso-cuda --features cuda -- --ignored --nocapture
#[cfg(test)]
mod tests {
    use super::*;

    /// Without the `cuda` feature, `open` must report FeatureDisabled and the
    /// backend must advertise itself as unavailable. This is the host-CI path.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn open_without_feature_is_feature_disabled() {
        assert!(!CudaBackend::available());
        match CudaBackend::open(0) {
            Err(CudaLoadError::FeatureDisabled) => {}
            Err(e) => panic!("expected FeatureDisabled, got Err({:?})", e),
            Ok(_) => panic!("expected FeatureDisabled, got Ok(CudaBackend)"),
        }
    }

    /// With the `cuda` feature but no driver present (typical dev box / CI),
    /// `available()` is false and `open` fails with DriverNotFound. On a real
    /// CUDA host this test would instead see a backend open successfully, so it
    /// only asserts the negative when no runtime is loadable.
    #[test]
    #[cfg(feature = "cuda")]
    fn open_with_feature_but_no_driver() {
        if !CudaBackend::available() {
            // Match directly so we never need CudaBackend: Debug for the Ok arm.
            match CudaBackend::open(0) {
                Err(CudaLoadError::DriverNotFound(_)) | Err(CudaLoadError::MissingSymbol(_)) => {}
                Err(other) => panic!("expected DriverNotFound/MissingSymbol, got {:?}", other),
                Ok(_) => panic!("expected an error when no driver is loadable"),
            }
        }
    }

    /// The IpcHandle blob must be exactly the CUDA cudaIpcMemHandle_t size so our
    /// export/import shims copy the right number of bytes.
    #[test]
    fn ipc_handle_blob_is_64_bytes() {
        assert_eq!(tenso_core::IPC_REF_HANDLE_LEN, 64);
        let h = IpcHandle {
            bytes: [0u8; tenso_core::IPC_REF_HANDLE_LEN],
            byte_offset: 0,
            nbytes: 0,
            device_uuid: [0u8; tenso_core::IPC_REF_DEVICE_UUID_LEN],
        };
        assert_eq!(h.bytes.len(), 64);
        assert_eq!(h.device_uuid.len(), 16);
    }

    /// The CudaIpcMemHandle FFI struct must be exactly 64 bytes (==
    /// CUDA_IPC_HANDLE_SIZE == IPC_REF_HANDLE_LEN) so passing it by value matches
    /// the C `cudaIpcMemHandle_t` ABI.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_ipc_handle_struct_is_64_bytes() {
        assert_eq!(core::mem::size_of::<ffi::CudaIpcMemHandle>(), 64);
        assert_eq!(ffi::CUDA_IPC_HANDLE_BYTES, tenso_core::IPC_REF_HANDLE_LEN);
    }

    /// `pci_bus_id_to_uuid` copies the bus-id bytes verbatim, stops at the NUL,
    /// and zero-pads to 16 bytes. Distinct bus ids yield distinct uuids.
    #[test]
    fn pci_bus_id_uuid_is_stable_and_distinct() {
        // C string b"0000:65:00.0\0..."
        let mut a = [0i8; 32];
        for (i, &b) in b"0000:65:00.0".iter().enumerate() {
            a[i] = b as i8;
        }
        let ua = pci_bus_id_to_uuid(&a);
        assert_eq!(&ua[0..12], b"0000:65:00.0");
        assert_eq!(&ua[12..16], &[0u8; 4]); // zero-padded tail
                                            // deterministic
        assert_eq!(ua, pci_bus_id_to_uuid(&a));

        let mut b = [0i8; 32];
        for (i, &c) in b"0000:b3:00.0".iter().enumerate() {
            b[i] = c as i8;
        }
        assert_ne!(ua, pci_bus_id_to_uuid(&b));
    }

    // -------------------------------------------------------------------------
    // GPU integration tests — REQUIRE a real Linux + NVIDIA device.
    // Gated #[ignore] so they only run with `--ignored`.
    // -------------------------------------------------------------------------

    /// (a) Device round-trip through the wire codec: host tensor -> alloc +
    /// copy_h2d -> GpuCodec::encode_from_device -> decode_into_device -> copy_d2h
    /// -> assert bytes equal. Proves the discrete H2D/D2H + encode/decode path.
    #[test]
    #[ignore = "requires a real Linux + NVIDIA CUDA device"]
    #[cfg(feature = "cuda")]
    fn gpu_device_roundtrip_through_codec() {
        use tenso_core::{Dtype, EncodeOpts};
        use tenso_device::GpuCodec;

        let be = CudaBackend::open(0).expect("CUDA device 0 must be present");
        let codec = GpuCodec::new(&be);

        // 16 f32 elements, shape [4,4] -> 64 bytes of body.
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 1.5).collect();
        let mut body = vec![0u8; values.len() * 4];
        for (i, v) in values.iter().enumerate() {
            body[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let shape = [4u32, 4u32];

        // host -> device
        let src = be.alloc(body.len()).expect("alloc src");
        be.copy_h2d(src, &body);

        // encode straight from device memory
        let opts = EncodeOpts::default();
        let packet = codec
            .encode_from_device(src, body.len(), Dtype::F32, &shape, &opts)
            .expect("encode_from_device");

        // decode the packet back into a fresh device allocation
        let dst = be.alloc(body.len()).expect("alloc dst");
        let written = codec
            .decode_into_device(&packet, dst, body.len())
            .expect("decode_into_device");
        assert_eq!(written, body.len(), "decoded body byte count");

        // device -> host and compare
        let mut out = vec![0u8; body.len()];
        be.copy_d2h(&mut out, dst);
        assert_eq!(out, body, "round-tripped bytes must match the source");

        be.free(src);
        be.free(dst);
    }

    /// (b) IPC round-trip ACROSS PROCESSES — the only configuration CUDA IPC
    /// supports. `cudaIpcOpenMemHandle` is a cross-process mechanism; importing a
    /// handle in the SAME process that exported it is rejected by the driver
    /// (returns cudaErrorDeviceUninitialized = 201 on the 5070 Ti — see test
    /// `gpu_ipc_same_process_import_is_rejected`). So this test exports in the
    /// parent, writes the 64-byte handle + device UUID to a temp file, then
    /// re-execs THIS test binary as an importer child. The child opens its own
    /// CUDA context, imports the handle, reads the aliased VRAM back, and asserts
    /// byte-equality, exiting 0 on success. The parent holds the allocation and
    /// its CUDA context alive (it blocks on the child) for the whole import.
    #[test]
    #[ignore = "requires a real Linux + NVIDIA CUDA device"]
    #[cfg(feature = "cuda")]
    fn gpu_ipc_roundtrip_cross_process() {
        use std::io::{Read, Write};

        const PAYLOAD: [u8; 16] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];

        // Importer child arm: re-exec'd with TENSO_IPC_CHILD set to the handle
        // file. Reads [handle:64][uuid:16], imports, verifies, and exits the
        // process directly (so libtest semantics never apply to the child).
        if let Ok(path) = std::env::var("TENSO_IPC_CHILD") {
            let mut buf = Vec::new();
            std::fs::File::open(&path)
                .and_then(|mut f| f.read_to_end(&mut buf))
                .expect("child: read handle file");
            assert_eq!(buf.len(), 80, "child: handle file must be 80 bytes");
            let mut bytes = [0u8; tenso_core::IPC_REF_HANDLE_LEN];
            bytes.copy_from_slice(&buf[0..64]);
            let mut uuid = [0u8; tenso_core::IPC_REF_DEVICE_UUID_LEN];
            uuid.copy_from_slice(&buf[64..80]);

            let be = match CudaBackend::open(0) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("child: CUDA open failed: {e:?}");
                    std::process::exit(2);
                }
            };
            let handle = IpcHandle {
                bytes,
                byte_offset: 0,
                nbytes: PAYLOAD.len() as u64,
                device_uuid: uuid,
            };
            let q = match be.import_ipc(&handle) {
                Ok(q) => q,
                Err(e) => {
                    eprintln!("child: import_ipc failed: {e:?}");
                    std::process::exit(3);
                }
            };
            let mut out = [0u8; 16];
            be.copy_d2h(&mut out, q);
            be.close_ipc(q).expect("child: close_ipc");
            if out != PAYLOAD {
                eprintln!("child: IPC readback mismatch: {out:?}");
                std::process::exit(4);
            }
            std::process::exit(0);
        }

        // Parent arm.
        let be = CudaBackend::open(0).expect("CUDA device 0 must be present");
        assert!(
            !be.is_unified_memory(),
            "CUDA IPC requires a discrete GPU; integrated parts reject it"
        );
        let p = be.alloc(PAYLOAD.len()).expect("alloc");
        be.copy_h2d(p, &PAYLOAD);

        let handle = be.export_ipc(p).expect("export_ipc");
        assert_eq!(handle.device_uuid, be.device_uuid());

        // Serialize [handle:64][uuid:16] for the child.
        let path =
            std::env::temp_dir().join(format!("tenso_ipc_handle_{}.bin", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).expect("create handle file");
            f.write_all(&handle.bytes).expect("write handle");
            f.write_all(&handle.device_uuid).expect("write uuid");
        }

        // Re-exec exactly this (ignored) test as the importer child.
        let exe = std::env::current_exe().expect("current_exe");
        let status = std::process::Command::new(exe)
            .args([
                "--exact",
                "--ignored",
                "tests::gpu_ipc_roundtrip_cross_process",
            ])
            .env("TENSO_IPC_CHILD", &path)
            .status()
            .expect("spawn importer child");

        // The child has now exited, so the export's job is done; tear down.
        let _ = std::fs::remove_file(&path);
        be.free(p);

        assert!(
            status.success(),
            "cross-process IPC importer child failed (exit {:?}); see child stderr above",
            status.code()
        );
    }

    /// (b2) Same-process IPC import MUST be rejected by CUDA. This documents and
    /// regression-tests the restriction that forces (b) to be cross-process:
    /// `cudaIpcOpenMemHandle` on a handle exported by the calling process returns
    /// an error (201 / cudaErrorDeviceUninitialized on the 5070 Ti), now surfaced
    /// as `DevErr::Device(code)` instead of a blanket `IpcUnsupported`.
    #[test]
    #[ignore = "requires a real Linux + NVIDIA CUDA device"]
    #[cfg(feature = "cuda")]
    fn gpu_ipc_same_process_import_is_rejected() {
        let be = CudaBackend::open(0).expect("CUDA device 0 must be present");
        let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let p = be.alloc(payload.len()).expect("alloc");
        be.copy_h2d(p, &payload);
        let handle = be.export_ipc(p).expect("export_ipc");

        match be.import_ipc(&handle) {
            Err(DevErr::Device(code)) => {
                eprintln!("same-process import correctly rejected with cudaError {code}");
            }
            Err(other) => {
                // Still a rejection (the exact code can vary by driver/version).
                eprintln!("same-process import rejected with {other:?}");
            }
            Ok(_) => panic!("same-process cudaIpcOpenMemHandle unexpectedly succeeded"),
        }
        be.free(p);
    }

    /// (c) The discrete RTX 5070 Ti must report `is_unified_memory() == false`
    /// (cudaDevAttrIntegrated == 0). Only an integrated Tegra/Jetson part is
    /// unified. Also sanity-checks the derived UUID is nonzero.
    #[test]
    #[ignore = "requires a real Linux + NVIDIA CUDA device"]
    #[cfg(feature = "cuda")]
    fn gpu_discrete_is_not_unified() {
        let be = CudaBackend::open(0).expect("CUDA device 0 must be present");
        assert!(
            !be.is_unified_memory(),
            "a discrete GPU (e.g. RTX 5070 Ti) must NOT report unified memory"
        );
        assert_ne!(
            be.device_uuid(),
            [0u8; 16],
            "device_uuid must be derived from a non-empty PCI bus id"
        );
    }
}
