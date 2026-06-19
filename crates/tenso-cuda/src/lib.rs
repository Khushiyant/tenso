//! tenso-cuda: the only crate that touches CUDA.
//!
//! `libcudart` is dlopen'd at runtime (no link-time toolkit dependency), so the
//! workspace builds on toolkit-less hosts. Driver calls are gated behind `cuda`;
//! without it `open` returns `FeatureDisabled` and `available()` is `false`.
//!
//! `CudaBackend` implements `tenso_device::DeviceBackend` (slots into `GpuCodec`
//! like the Cpu/Mock backends). FFI shims only — no offset/packet/wire math
//! (that lives in `GpuCodec` above the trait); we just move bytes and hand out
//! opaque IPC handles. Host-only unit tests verify graceful degradation; GPU
//! tests are `#[ignore]`d (run with `--features cuda -- --ignored`).

#![allow(dead_code, unused)]

use tenso_device::{DevErr, DevPtr, DeviceBackend, IpcHandle, PinnedBuf};

/// Error opening / loading the CUDA runtime.
#[derive(Debug)]
pub enum CudaLoadError {
    /// The crate was built without the `cuda` feature.
    FeatureDisabled,
    /// `libcudart` could not be dlopen'd. Carries a diagnostic (attempts + last
    /// dlopen error) so a present-but-off-loader-path runtime is debuggable.
    DriverNotFound(String),
    /// A required symbol was missing from the runtime library.
    MissingSymbol(&'static str),
    /// A runtime call failed (carries the `cudaError_t` + `cudaGetErrorString` text).
    Driver(i32, String),
}

// =============================================================================
// CUDA runtime FFI surface (only compiled with the `cuda` feature)
// =============================================================================
//
// Binds only the entry points we need; all follow `cudaError_t cudaXxx(...)`
// (0 == cudaSuccess). We never deref CUDA-owned memory on the host.
#[cfg(feature = "cuda")]
mod ffi {
    use core::ffi::{c_char, c_void};
    use libloading::{Library, Symbol};

    // cudaError_t — 0 is cudaSuccess.
    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    // cudaMemcpyKind enum values.
    pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
    pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

    // cudaHostAllocDefault: plain page-locked memory, no Mapped flag (PinnedBuf
    // only carries a Vec<u8>; see alloc_pinned for the seam).
    pub const CUDA_HOST_ALLOC_DEFAULT: u32 = 0;

    // cudaDevAttrIntegrated = 18 (1 on Tegra/Jetson iGPUs; 0 on discrete).
    pub const CUDA_DEV_ATTR_INTEGRATED: i32 = 18;

    // cudaIpcMemHandle_t size: opaque 64-byte blob == tenso::IPC_REF_HANDLE_LEN.
    pub const CUDA_IPC_HANDLE_BYTES: usize = 64;

    // cudaIpcMemLazyEnablePeerAccess = 1 (the only documented OpenMemHandle flag).
    pub const CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: u32 = 1;

    // cudaIpcMemHandle_t is `struct { char reserved[64]; }`. cudaIpcOpenMemHandle
    // takes it BY VALUE; a `#[repr(C)]` 64-byte array wrapper matches that ABI.
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
    // cudaIpcGetMemHandle: handle is an OUT param (by pointer).
    pub type FnIpcGetMemHandle =
        unsafe extern "C" fn(handle: *mut CudaIpcMemHandle, dev_ptr: *mut c_void) -> CudaError;
    // cudaIpcOpenMemHandle: handle passed BY VALUE (64-byte struct).
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

    /// Owns the dlopen'd library + resolved fn pointers. `_lib` is kept alive so
    /// the pointers stay valid.
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

    /// Candidate `libcudart` sonames across platforms/majors, tried in order.
    const CANDIDATES: &[&str] = &[
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.13",
        "libcudart.so.11.0",
        "libcudart.so.11",
        // Windows / macOS fallbacks.
        "cudart64_12.dll",
        "cudart64_110.dll",
        "libcudart.dylib",
    ];

    /// Standard install dirs (joined with each soname), searched only after
    /// bare-soname resolution fails. pip-wheel installs go via `TENSO_CUDART_PATH`.
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

    /// Find + dlopen `libcudart`: `$TENSO_CUDART_PATH`, then bare sonames (loader
    /// path), then `CANDIDATE_DIRS`×sonames. On failure returns `DriverNotFound`
    /// with a diagnostic (attempts + last dlopen error).
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
            // SAFETY: dlopen by name/path; symbols type-checked vs CUDA ABI in load().
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
        /// dlopen libcudart and resolve every symbol. `DriverNotFound` if nothing
        /// loads, `MissingSymbol` if an entry point is absent.
        pub fn load() -> Result<Self, super::CudaLoadError> {
            // `lib` is moved into the returned Runtime so the fn pointers stay valid.
            let lib = open_cudart()?;

            // Resolve a symbol, mapping absence to MissingSymbol.
            macro_rules! sym {
                ($name:literal, $ty:ty) => {{
                    // SAFETY: the named symbol, if present, has the C signature $ty.
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

        /// Resolve a `cudaError_t` to text via `cudaGetErrorString`. Never panics
        /// (falls back on NULL / non-UTF8).
        pub fn error_string(&self, code: CudaError) -> String {
            // SAFETY: returns a static runtime-owned C string (process lifetime) or NULL.
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

/// A handle to a CUDA device backed by the dlopen'd runtime. `open` selects the
/// device and caches the unified flag + UUID (from the PCI bus id).
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    rt: ffi::Runtime,
    #[cfg(feature = "cuda")]
    device_ordinal: i32,
    #[cfg(feature = "cuda")]
    unified: bool,
    #[cfg(feature = "cuda")]
    uuid: [u8; 16],
    // Without `cuda` the struct is uninhabited (`open` never returns Ok) but
    // must still be nameable.
    #[cfg(not(feature = "cuda"))]
    _never: core::convert::Infallible,
}

impl CudaBackend {
    /// True if `libcudart` could be dlopen'd (and `cuda` is enabled); lets callers
    /// probe for GPU support. Always `false` without the `cuda` feature.
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

    /// Load the runtime and bind to `device_ordinal`. Without `cuda` returns
    /// `FeatureDisabled`; otherwise selects the device and caches the unified
    /// flag + a stable 16-byte UUID from the PCI bus id.
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

            // Bind to the requested device (nonzero rc = bad ordinal / no device).
            // SAFETY: resolved C entry point, plain i32 argument.
            let rc = unsafe { (rt.set_device)(device_ordinal) };
            if rc != ffi::CUDA_SUCCESS {
                return Err(CudaLoadError::Driver(rc, rt.error_string(rc)));
            }

            // Integrated attr: 1 on Jetson/Tegra iGPUs (shared memory, zero-copy
            // path), 0 on discrete parts.
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

            // Derive a 16-byte UUID from the PCI bus id ("0000:65:00.0") rather
            // than the version-fragile `cudaDeviceProp`. The bus id is stable per
            // slot across processes, a sound key for IPC's "same device" rule.
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

    /// True if the device exposes unified (integrated/Tegra) memory. Mirror of
    /// the trait method for callers holding a concrete `CudaBackend`.
    pub fn is_unified(&self) -> bool {
        #[cfg(not(feature = "cuda"))]
        {
            // Unreachable: CudaBackend cannot be constructed without `cuda`.
            match self._never {}
        }
        #[cfg(feature = "cuda")]
        {
            self.unified
        }
    }
}

/// Derive a stable 16-byte device id from a NUL-terminated PCI bus id (e.g.
/// b"0000:65:00.0"): bytes copied verbatim, truncated/zero-padded to 16.
/// Deterministic, collision-free per slot. Unconditional so host-only tests run.
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
// `DevPtr(usize)` carries the raw `void*` from `cudaMalloc`. Every fallible call
// checks the cudaError_t.

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
            // cudaMalloc(0) returns NULL; treat as a benign null DevPtr.
            if n == 0 {
                return Ok(DevPtr(0));
            }
            let mut ptr: *mut c_void = core::ptr::null_mut();
            // SAFETY: out-param is a valid &mut *mut c_void; n is the byte count.
            let rc = unsafe { (self.rt.malloc)(&mut ptr, n) };
            if rc != ffi::CUDA_SUCCESS || ptr.is_null() {
                // cudaMalloc fails almost exclusively with OOM.
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
            // SAFETY: p.0 came from cudaMalloc on this backend; errors swallowed
            // to match the infallible `free` signature.
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
            // SAFETY: dst.0 has >= src.len() bytes (caller contract); src is a
            // valid host slice; Kind=1 = HostToDevice.
            let rc = unsafe {
                (self.rt.memcpy)(
                    dst.0 as *mut c_void,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            };
            // Nonzero rc means the copy did not happen (dst stale); fail loudly in debug.
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
            // SAFETY: src.0 has >= dst.len() bytes (caller contract); dst is a
            // valid mutable host slice; Kind=2 = DeviceToHost.
            let rc = unsafe {
                (self.rt.memcpy)(
                    dst.as_mut_ptr() as *mut c_void,
                    src.0 as *const c_void,
                    dst.len(),
                    ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            };
            // Nonzero rc means dst was not filled (stale); fail loudly in debug.
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
            // SEAM: cudaHostAlloc + immediate cudaFreeHost, copying into a Vec<u8>
            // (all PinnedBuf carries). The Vec is pageable, so the pinned property
            // is lost; true pinned buffers would need PinnedBuf to hold the raw
            // host ptr + a cudaFreeHost-on-Drop deleter. We still exercise the
            // alloc/free pair so the GPU test proves it works on real hardware.
            let mut host: *mut c_void = core::ptr::null_mut();
            // SAFETY: out-param valid; n is the byte count; flag is a valid mask.
            let rc = unsafe { (self.rt.host_alloc)(&mut host, n, ffi::CUDA_HOST_ALLOC_DEFAULT) };
            if rc != ffi::CUDA_SUCCESS || host.is_null() {
                return Err(DevErr::OutOfMemory);
            }
            // cudaHostAlloc does not zero, so zero-init the Vec.
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
            // Integrated GPUs reject CUDA IPC mem handles (shared system memory);
            // surface as IpcUnsupported rather than a raw code.
            if self.unified {
                return Err(DevErr::IpcUnsupported);
            }
            let mut handle = ffi::CudaIpcMemHandle([0u8; ffi::CUDA_IPC_HANDLE_BYTES]);
            // SAFETY: handle is a 64-byte cudaIpcMemHandle_t out-param; p.0 came
            // from cudaMalloc on this backend.
            let rc = unsafe { (self.rt.ipc_get_mem_handle)(&mut handle, p.0 as *mut c_void) };
            if rc != ffi::CUDA_SUCCESS {
                // Surface the real cudaError_t, not a blanket IpcUnsupported.
                return Err(DevErr::Device(rc));
            }
            // Fill only the handle blob + UUID; byte_offset/nbytes are GpuCodec
            // wire-framing concerns, left zero here.
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
            // Defensively reject a cross-device handle when our cached UUID is
            // known (nonzero) and disagrees (UUID validation is a GpuCodec rule).
            if self.uuid != [0u8; 16] && h.device_uuid != self.uuid {
                return Err(DevErr::DeviceUuidMismatch);
            }
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let handle = ffi::CudaIpcMemHandle(h.bytes);
            // SAFETY: out-param valid; `handle` is a 64-byte cudaIpcMemHandle_t
            // passed BY VALUE (matches C); flag is LazyEnablePeerAccess = 1.
            let rc = unsafe {
                (self.rt.ipc_open_mem_handle)(
                    &mut ptr,
                    handle,
                    ffi::CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
                )
            };
            if rc != ffi::CUDA_SUCCESS {
                // Surface the real cudaError_t. CUDA IPC is cross-process only:
                // same-process import returns 201 (cudaErrorDeviceUninitialized).
                return Err(DevErr::Device(rc));
            }
            if ptr.is_null() {
                return Err(DevErr::IpcUnsupported);
            }
            // Honor byte_offset for sub-buffer refs (0 today, so a no-op).
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

/// Release a pointer from `import_ipc` via `cudaIpcCloseMemHandle` (NOT `free` /
/// `cudaFree`). No-op (uninhabited) without the `cuda` feature.
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
                // Surface the real cudaError_t (e.g. handle already gone/invalid).
                return Err(DevErr::Device(rc));
            }
            Ok(())
        }
    }
}

// =============================================================================
// Tests — host-only smoke tests + #[ignore] GPU integration
// =============================================================================
//
// Non-ignored tests never touch a device (verify graceful degradation, keeping
// toolkit-less CI green). `#[ignore]`d tests need a real Linux + NVIDIA box:
//   cargo test -p tenso-cuda --features cuda -- --ignored --nocapture
#[cfg(test)]
mod tests {
    use super::*;

    /// Without `cuda`: `open` reports FeatureDisabled and `available()` is false.
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

    /// With `cuda` but no driver: `available()` false and `open` fails with
    /// DriverNotFound. Only asserts the negative when no runtime is loadable.
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

    /// IpcHandle blob must be exactly cudaIpcMemHandle_t size (64 bytes).
    #[test]
    fn ipc_handle_blob_is_64_bytes() {
        assert_eq!(tenso::IPC_REF_HANDLE_LEN, 64);
        let h = IpcHandle {
            bytes: [0u8; tenso::IPC_REF_HANDLE_LEN],
            byte_offset: 0,
            nbytes: 0,
            device_uuid: [0u8; tenso::IPC_REF_DEVICE_UUID_LEN],
        };
        assert_eq!(h.bytes.len(), 64);
        assert_eq!(h.device_uuid.len(), 16);
    }

    /// CudaIpcMemHandle must be 64 bytes so by-value passing matches the C ABI.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_ipc_handle_struct_is_64_bytes() {
        assert_eq!(core::mem::size_of::<ffi::CudaIpcMemHandle>(), 64);
        assert_eq!(ffi::CUDA_IPC_HANDLE_BYTES, tenso::IPC_REF_HANDLE_LEN);
    }

    /// `pci_bus_id_to_uuid` copies bytes verbatim, stops at NUL, zero-pads to 16;
    /// distinct bus ids yield distinct uuids.
    #[test]
    fn pci_bus_id_uuid_is_stable_and_distinct() {
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
    // GPU integration tests — require a real Linux + NVIDIA device (#[ignore]d).
    // -------------------------------------------------------------------------

    /// (a) Device round-trip through the codec: h2d -> encode_from_device ->
    /// decode_into_device -> d2h -> assert equal.
    #[test]
    #[ignore = "requires a real Linux + NVIDIA CUDA device"]
    #[cfg(feature = "cuda")]
    fn gpu_device_roundtrip_through_codec() {
        use tenso::{Dtype, EncodeOpts};
        use tenso_device::GpuCodec;

        let be = CudaBackend::open(0).expect("CUDA device 0 must be present");
        let codec = GpuCodec::new(&be);

        // 16 f32, shape [4,4] -> 64 bytes.
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

    /// (b) IPC round-trip ACROSS PROCESSES (the only config CUDA IPC supports;
    /// same-process import returns 201). Parent exports + writes handle+UUID to a
    /// temp file, then re-execs this binary as an importer child that maps the
    /// handle, reads the aliased VRAM, and asserts byte-equality. Parent blocks on
    /// the child, keeping the allocation + context alive.
    #[test]
    #[ignore = "requires a real Linux + NVIDIA CUDA device"]
    #[cfg(feature = "cuda")]
    fn gpu_ipc_roundtrip_cross_process() {
        use std::io::{Read, Write};

        const PAYLOAD: [u8; 16] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];

        // Importer child arm (re-exec'd with TENSO_IPC_CHILD = handle file):
        // reads [handle:64][uuid:16], imports, verifies, exits directly.
        if let Ok(path) = std::env::var("TENSO_IPC_CHILD") {
            let mut buf = Vec::new();
            std::fs::File::open(&path)
                .and_then(|mut f| f.read_to_end(&mut buf))
                .expect("child: read handle file");
            assert_eq!(buf.len(), 80, "child: handle file must be 80 bytes");
            let mut bytes = [0u8; tenso::IPC_REF_HANDLE_LEN];
            bytes.copy_from_slice(&buf[0..64]);
            let mut uuid = [0u8; tenso::IPC_REF_DEVICE_UUID_LEN];
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

    /// (b2) Same-process IPC import must be rejected (201 / DeviceUninitialized),
    /// surfaced as `DevErr::Device(code)`. This is why (b) must be cross-process.
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

    /// (c) A discrete GPU reports `is_unified_memory() == false` and a nonzero UUID.
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
