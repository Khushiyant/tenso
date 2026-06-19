//! tenso-device: mockable device-backend layer (no real GPU needed).
//!
//! `DeviceBackend` trait, `CpuBackend` (host mem as device mem, IPC via a
//! process-local registry), `MockBackend` (Vec<u8> VRAM + op log + fault
//! injection), and `GpuCodec` (tenso encode/decode + IPC framing).

#![allow(dead_code, unused)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use tenso::{
    Decoded, Dtype, EncodeOpts, IpcRef, TensoError, FLAG_GPU_IPC_REF, FLAG_INTEGRITY,
    HEADER_BASE_V4, IPC_REF_DEVICE_UUID_LEN, IPC_REF_DISCRIMINATOR, IPC_REF_HANDLE_LEN,
    IPC_REF_PACKET_LEN, MAGIC, VERSION,
};

// ---- Device handle / error types ----

/// Opaque device pointer (offset/handle into a backend's address space).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevPtr(pub usize);

/// A pinned host buffer suitable for fast H2D/D2H transfers.
pub struct PinnedBuf {
    /// Host-visible bytes (a plain Vec for the mock/cpu backends).
    pub bytes: Vec<u8>,
}

/// Opaque IPC handle blob (CUDA cudaIpcMemHandle_t analogue).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcHandle {
    pub bytes: [u8; IPC_REF_HANDLE_LEN],
    pub byte_offset: u64,
    pub nbytes: u64,
    pub device_uuid: [u8; IPC_REF_DEVICE_UUID_LEN],
}

/// Device-layer error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevErr {
    OutOfMemory,
    InvalidPointer,
    IpcUnsupported,
    DeviceUuidMismatch,
    FaultInjected,
    Core(TensoError),
    /// Driver call failed with native status `code` (e.g. CUDA `cudaError_t`);
    /// distinct from `IpcUnsupported` (platform cannot do IPC at all).
    Device(i32),
    Other(&'static str),
}

// ---- DeviceBackend trait ----

pub trait DeviceBackend {
    fn alloc(&self, n: usize) -> Result<DevPtr, DevErr>;
    fn free(&self, p: DevPtr);
    fn copy_h2d(&self, dst: DevPtr, src: &[u8]);
    fn copy_d2h(&self, dst: &mut [u8], src: DevPtr);
    fn alloc_pinned(&self, n: usize) -> Result<PinnedBuf, DevErr>;
    fn export_ipc(&self, p: DevPtr) -> Result<IpcHandle, DevErr>;
    fn import_ipc(&self, h: &IpcHandle) -> Result<DevPtr, DevErr>;
    fn is_unified_memory(&self) -> bool;
    fn device_uuid(&self) -> [u8; 16];
}

// ---- CpuBackend (host memory; IPC via a process-local registry) ----
//
// "Device memory" is a boxed Vec<u8>; DevPtr is its first byte's host address.
// IPC is a process-local handle registry: a handle only resolves on a backend
// whose UUID matches the exporter (preserves cross-process semantics on CI).

struct CpuAlloc {
    // Boxed so the backing address stays stable for the DevPtr.
    storage: Box<[u8]>,
    len: usize,
}

struct CpuRegistry {
    // addr -> live allocation
    live: HashMap<usize, CpuAlloc>,
    // handle bytes -> (addr, len, exporter uuid)
    ipc: HashMap<[u8; IPC_REF_HANDLE_LEN], (usize, usize, [u8; IPC_REF_DEVICE_UUID_LEN])>,
    // seed for synthesising IPC handle bytes
    handle_seed: u64,
}

impl CpuRegistry {
    fn new() -> Self {
        CpuRegistry {
            live: HashMap::new(),
            ipc: HashMap::new(),
            handle_seed: 0,
        }
    }
}

fn cpu_registry() -> &'static Mutex<CpuRegistry> {
    static REG: OnceLock<Mutex<CpuRegistry>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(CpuRegistry::new()))
}

/// Host-memory backend: memcpy transfers, process-local IPC, unified memory.
pub struct CpuBackend {
    uuid: [u8; IPC_REF_DEVICE_UUID_LEN],
}

impl CpuBackend {
    pub fn new() -> Self {
        // Distinct per-process UUID so import_ipc rejects foreign handles.
        static CTR: AtomicU64 = AtomicU64::new(1);
        let id = CTR.fetch_add(1, Ordering::Relaxed);
        let mut uuid = [0u8; IPC_REF_DEVICE_UUID_LEN];
        uuid[0..4].copy_from_slice(b"CPU\0");
        uuid[8..16].copy_from_slice(&id.to_le_bytes());
        CpuBackend { uuid }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        CpuBackend::new()
    }
}

impl DeviceBackend for CpuBackend {
    fn alloc(&self, n: usize) -> Result<DevPtr, DevErr> {
        let storage = vec![0u8; n].into_boxed_slice();
        let addr = storage.as_ptr() as usize;
        let mut reg = cpu_registry().lock().unwrap();
        reg.live.insert(addr, CpuAlloc { storage, len: n });
        Ok(DevPtr(addr))
    }

    fn free(&self, p: DevPtr) {
        let mut reg = cpu_registry().lock().unwrap();
        reg.live.remove(&p.0);
    }

    fn copy_h2d(&self, dst: DevPtr, src: &[u8]) {
        let mut reg = cpu_registry().lock().unwrap();
        if let Some(a) = reg.live.get_mut(&dst.0) {
            let n = src.len().min(a.len);
            a.storage[..n].copy_from_slice(&src[..n]);
        }
    }

    fn copy_d2h(&self, dst: &mut [u8], src: DevPtr) {
        let reg = cpu_registry().lock().unwrap();
        if let Some(a) = reg.live.get(&src.0) {
            let n = dst.len().min(a.len);
            dst[..n].copy_from_slice(&a.storage[..n]);
        }
    }

    fn alloc_pinned(&self, n: usize) -> Result<PinnedBuf, DevErr> {
        Ok(PinnedBuf {
            bytes: vec![0u8; n],
        })
    }

    fn export_ipc(&self, p: DevPtr) -> Result<IpcHandle, DevErr> {
        let mut reg = cpu_registry().lock().unwrap();
        let len = reg.live.get(&p.0).ok_or(DevErr::InvalidPointer)?.len;
        reg.handle_seed += 1;
        let seed = reg.handle_seed;
        let mut bytes = [0u8; IPC_REF_HANDLE_LEN];
        // Address + uniquifying seed => collision-free in-process handle key.
        bytes[0..8].copy_from_slice(&(p.0 as u64).to_le_bytes());
        bytes[8..16].copy_from_slice(&seed.to_le_bytes());
        reg.ipc.insert(bytes, (p.0, len, self.uuid));
        Ok(IpcHandle {
            bytes,
            byte_offset: 0,
            nbytes: len as u64,
            device_uuid: self.uuid,
        })
    }

    fn import_ipc(&self, h: &IpcHandle) -> Result<DevPtr, DevErr> {
        if h.device_uuid != self.uuid {
            return Err(DevErr::DeviceUuidMismatch);
        }
        let reg = cpu_registry().lock().unwrap();
        let (addr, _len, exp_uuid) = *reg.ipc.get(&h.bytes).ok_or(DevErr::InvalidPointer)?;
        if exp_uuid != self.uuid {
            return Err(DevErr::DeviceUuidMismatch);
        }
        Ok(DevPtr(addr))
    }

    fn is_unified_memory(&self) -> bool {
        true
    }

    fn device_uuid(&self) -> [u8; 16] {
        self.uuid
    }
}

// ---- MockBackend (Vec<u8> VRAM, op recording, fault injection) ----

/// Records each backend operation for assertions in tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockOp {
    Alloc(usize),
    Free(DevPtr),
    CopyH2D { dst: DevPtr, len: usize },
    CopyD2H { src: DevPtr, len: usize },
    AllocPinned(usize),
    ExportIpc(DevPtr),
    ImportIpc,
}

struct MockSlab {
    // One Vec per allocation = fake VRAM.
    data: Vec<u8>,
}

struct MockState {
    // DevPtr.0 is a synthetic 1-based id into the slab map.
    slabs: HashMap<usize, MockSlab>,
    next_id: usize,
    log: Vec<MockOp>,
    // when op_counter == fault_at, the next faultable op fails once.
    op_counter: usize,
    fault_at: Option<usize>,
    // exported handles: handle bytes -> (id, len, uuid)
    ipc: HashMap<[u8; IPC_REF_HANDLE_LEN], (usize, usize, [u8; IPC_REF_DEVICE_UUID_LEN])>,
    handle_seed: u64,
    // capacity (None = unbounded) and current bytes in use
    capacity: Option<usize>,
    in_use: usize,
}

/// Mock GPU backend: Vec<u8> VRAM, op log, fault injection, configurable
/// unified-memory flag (for Jetson zero-copy paths without a GPU).
pub struct MockBackend {
    state: Mutex<MockState>,
    unified: bool,
    alignment: usize,
    uuid: [u8; IPC_REF_DEVICE_UUID_LEN],
}

impl MockBackend {
    /// Build a mock backend; `unified` toggles `is_unified_memory()`.
    pub fn new(unified: bool) -> Self {
        static CTR: AtomicU64 = AtomicU64::new(1);
        let id = CTR.fetch_add(1, Ordering::Relaxed);
        let mut uuid = [0u8; IPC_REF_DEVICE_UUID_LEN];
        uuid[0..4].copy_from_slice(b"MOCK");
        uuid[8..16].copy_from_slice(&id.to_le_bytes());
        MockBackend {
            state: Mutex::new(MockState {
                slabs: HashMap::new(),
                next_id: 1,
                log: Vec::new(),
                op_counter: 0,
                fault_at: None,
                ipc: HashMap::new(),
                handle_seed: 0,
                capacity: None,
                in_use: 0,
            }),
            unified,
            alignment: tenso::ALIGNMENT,
            uuid,
        }
    }

    /// Override the device UUID (useful for cross-device IPC rejection tests).
    pub fn with_uuid(mut self, uuid: [u8; IPC_REF_DEVICE_UUID_LEN]) -> Self {
        self.uuid = uuid;
        self
    }

    /// Configure the reported alignment.
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// Cap total VRAM so `alloc` can return `OutOfMemory` past `capacity` bytes.
    pub fn with_capacity(self, capacity: usize) -> Self {
        {
            let mut st = self.state.lock().unwrap();
            st.capacity = Some(capacity);
        }
        self
    }

    /// Reported (configurable) alignment.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Inject a one-shot fault on the Nth (zero-based) faultable op (alloc /
    /// copy_h2d / copy_d2h / alloc_pinned / export_ipc / import_ipc).
    pub fn inject_fault_at(&self, op_index: usize) {
        let mut st = self.state.lock().unwrap();
        st.fault_at = Some(op_index);
    }

    /// Return the recorded op log.
    pub fn ops(&self) -> Vec<MockOp> {
        self.state.lock().unwrap().log.clone()
    }

    /// Read back a slab's current bytes (test helper; no real GPU).
    pub fn slab_bytes(&self, p: DevPtr) -> Option<Vec<u8>> {
        let st = self.state.lock().unwrap();
        st.slabs.get(&p.0).map(|s| s.data.clone())
    }

    // True (and consumes the fault) if this faultable op should fail.
    fn check_fault(st: &mut MockState) -> bool {
        let trip = st.fault_at == Some(st.op_counter);
        st.op_counter += 1;
        if trip {
            st.fault_at = None;
        }
        trip
    }
}

impl DeviceBackend for MockBackend {
    fn alloc(&self, n: usize) -> Result<DevPtr, DevErr> {
        let mut st = self.state.lock().unwrap();
        st.log.push(MockOp::Alloc(n));
        if Self::check_fault(&mut st) {
            return Err(DevErr::FaultInjected);
        }
        if let Some(cap) = st.capacity {
            if st.in_use + n > cap {
                return Err(DevErr::OutOfMemory);
            }
        }
        let id = st.next_id;
        st.next_id += 1;
        st.in_use += n;
        st.slabs.insert(id, MockSlab { data: vec![0u8; n] });
        Ok(DevPtr(id))
    }

    fn free(&self, p: DevPtr) {
        let mut st = self.state.lock().unwrap();
        st.log.push(MockOp::Free(p));
        if let Some(slab) = st.slabs.remove(&p.0) {
            st.in_use = st.in_use.saturating_sub(slab.data.len());
        }
    }

    fn copy_h2d(&self, dst: DevPtr, src: &[u8]) {
        let mut st = self.state.lock().unwrap();
        st.log.push(MockOp::CopyH2D {
            dst,
            len: src.len(),
        });
        // Faults on void-returning copies can't surface an error; the slab is
        // left untouched.
        if Self::check_fault(&mut st) {
            return;
        }
        if let Some(slab) = st.slabs.get_mut(&dst.0) {
            let n = src.len().min(slab.data.len());
            slab.data[..n].copy_from_slice(&src[..n]);
        }
    }

    fn copy_d2h(&self, dst: &mut [u8], src: DevPtr) {
        let mut st = self.state.lock().unwrap();
        st.log.push(MockOp::CopyD2H {
            src,
            len: dst.len(),
        });
        if Self::check_fault(&mut st) {
            return;
        }
        if let Some(slab) = st.slabs.get(&src.0) {
            let n = dst.len().min(slab.data.len());
            dst[..n].copy_from_slice(&slab.data[..n]);
        }
    }

    fn alloc_pinned(&self, n: usize) -> Result<PinnedBuf, DevErr> {
        let mut st = self.state.lock().unwrap();
        st.log.push(MockOp::AllocPinned(n));
        if Self::check_fault(&mut st) {
            return Err(DevErr::FaultInjected);
        }
        Ok(PinnedBuf {
            bytes: vec![0u8; n],
        })
    }

    fn export_ipc(&self, p: DevPtr) -> Result<IpcHandle, DevErr> {
        let mut st = self.state.lock().unwrap();
        st.log.push(MockOp::ExportIpc(p));
        if Self::check_fault(&mut st) {
            return Err(DevErr::FaultInjected);
        }
        let len = match st.slabs.get(&p.0) {
            Some(s) => s.data.len(),
            None => return Err(DevErr::InvalidPointer),
        };
        st.handle_seed += 1;
        let seed = st.handle_seed;
        let mut bytes = [0u8; IPC_REF_HANDLE_LEN];
        bytes[0..8].copy_from_slice(&(p.0 as u64).to_le_bytes());
        bytes[8..16].copy_from_slice(&seed.to_le_bytes());
        let uuid = self.uuid;
        st.ipc.insert(bytes, (p.0, len, uuid));
        Ok(IpcHandle {
            bytes,
            byte_offset: 0,
            nbytes: len as u64,
            device_uuid: uuid,
        })
    }

    fn import_ipc(&self, h: &IpcHandle) -> Result<DevPtr, DevErr> {
        let mut st = self.state.lock().unwrap();
        st.log.push(MockOp::ImportIpc);
        if Self::check_fault(&mut st) {
            return Err(DevErr::FaultInjected);
        }
        if h.device_uuid != self.uuid {
            return Err(DevErr::DeviceUuidMismatch);
        }
        match st.ipc.get(&h.bytes) {
            Some(&(id, len, exp_uuid)) => {
                if exp_uuid != self.uuid {
                    Err(DevErr::DeviceUuidMismatch)
                } else if h.byte_offset as usize > len {
                    // Offset past the allocation is invalid.
                    Err(DevErr::InvalidPointer)
                } else {
                    // Honor byte_offset: sub-buffer resolves to base + offset.
                    Ok(DevPtr(id + h.byte_offset as usize))
                }
            }
            None => Err(DevErr::InvalidPointer),
        }
    }

    fn is_unified_memory(&self) -> bool {
        self.unified
    }

    fn device_uuid(&self) -> [u8; 16] {
        self.uuid
    }
}

// ---- IpcRef packet framing (mirror of tenso's wire layout) ----
//
// Builds/parses the 106-byte IpcRef packet from tenso's framing constants,
// byte-identical to tenso so packets round-trip through its parse/decode.
// Layout (LE): header[0..10] = magic, ver=4, flags=FLAG_GPU_IPC_REF, dtype=0,
// ndim=0, IPC_REF_DISCRIMINATOR; body[10..106] = handle[64], byte_offset:u64,
// nbytes:u64, device_uuid[16].

fn build_ipc_packet(ipc: &IpcRef) -> Vec<u8> {
    let mut out = vec![0u8; IPC_REF_PACKET_LEN];
    // v4 header
    out[0..4].copy_from_slice(&MAGIC);
    out[4] = VERSION;
    out[5..7].copy_from_slice(&FLAG_GPU_IPC_REF.to_le_bytes());
    // dtype_code (offset 7) = 0: matches tenso's write_ipc_ref; an IpcRef
    // has no element dtype.
    out[7] = 0;
    out[8] = 0; // ndim: no inline shape
    out[9] = IPC_REF_DISCRIMINATOR;
    // body
    let b = HEADER_BASE_V4;
    out[b..b + IPC_REF_HANDLE_LEN].copy_from_slice(&ipc.handle);
    let o = b + IPC_REF_HANDLE_LEN;
    out[o..o + 8].copy_from_slice(&ipc.byte_offset.to_le_bytes());
    out[o + 8..o + 16].copy_from_slice(&ipc.nbytes.to_le_bytes());
    out[o + 16..o + 16 + IPC_REF_DEVICE_UUID_LEN].copy_from_slice(&ipc.device_uuid);
    out
}

fn parse_ipc_packet(bytes: &[u8]) -> Result<IpcRef, TensoError> {
    if bytes.len() < IPC_REF_PACKET_LEN {
        return Err(TensoError::TooShort);
    }
    if bytes[0..4] != MAGIC {
        return Err(TensoError::BadMagic);
    }
    if bytes[4] != VERSION {
        return Err(TensoError::UnsupportedVersion(bytes[4]));
    }
    let flags = u16::from_le_bytes([bytes[5], bytes[6]]);
    if flags & FLAG_GPU_IPC_REF == 0 {
        return Err(TensoError::Malformed);
    }
    // An IpcRef must never carry an integrity footer.
    if flags & FLAG_INTEGRITY != 0 {
        return Err(TensoError::Malformed);
    }
    if bytes[9] != IPC_REF_DISCRIMINATOR {
        return Err(TensoError::Malformed);
    }
    let b = HEADER_BASE_V4;
    let mut handle = [0u8; IPC_REF_HANDLE_LEN];
    handle.copy_from_slice(&bytes[b..b + IPC_REF_HANDLE_LEN]);
    let o = b + IPC_REF_HANDLE_LEN;
    let byte_offset = u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let nbytes = u64::from_le_bytes(bytes[o + 8..o + 16].try_into().unwrap());
    let mut device_uuid = [0u8; IPC_REF_DEVICE_UUID_LEN];
    device_uuid.copy_from_slice(&bytes[o + 16..o + 16 + IPC_REF_DEVICE_UUID_LEN]);
    Ok(IpcRef {
        handle,
        byte_offset,
        nbytes,
        device_uuid,
    })
}

// ---- GpuCodec (orchestration over tenso) ----

/// Result of decoding into device memory.
pub struct DecodeResult {
    pub ptr: DevPtr,
    pub nbytes: usize,
    /// True when unified memory avoided an H2D copy.
    pub zero_copy: bool,
}

/// Drives tenso encode/decode against a `DeviceBackend`, incl. IPC framing.
pub struct GpuCodec<'b, B: DeviceBackend> {
    pub backend: &'b B,
}

impl<'b, B: DeviceBackend> GpuCodec<'b, B> {
    pub fn new(backend: &'b B) -> Self {
        GpuCodec { backend }
    }

    /// Encode a tensor whose body lives in device memory (stages D2H, then
    /// runs tenso's dense encoder; D2H may be a no-cost view if unified).
    pub fn encode_from_device(
        &self,
        ptr: DevPtr,
        nbytes: usize,
        dtype: Dtype,
        shape: &[u32],
        opts: &EncodeOpts,
    ) -> Result<Vec<u8>, DevErr> {
        // Stage device -> host.
        let mut host = vec![0u8; nbytes];
        self.backend.copy_d2h(&mut host, ptr);

        let spec = tenso::ArraySpec {
            data: &host,
            dtype,
            shape,
        };
        let need = tenso::dense_required_size(&spec, opts).map_err(DevErr::Core)?;
        let mut out = vec![0u8; need];
        let written = tenso::encode_dense_into(&spec, &mut out, opts).map_err(DevErr::Core)?;
        out.truncate(written);
        Ok(out)
    }

    /// Decode a packet's dense body into a caller-provided device pointer of
    /// `dst_capacity` bytes; errors `OutOfMemory` rather than truncate.
    pub fn decode_into_device(
        &self,
        bytes: &[u8],
        dst: DevPtr,
        dst_capacity: usize,
    ) -> Result<usize, DevErr> {
        let body = self.decode_dense_body(bytes)?;
        if body.len() > dst_capacity {
            return Err(DevErr::OutOfMemory);
        }
        self.backend.copy_h2d(dst, body);
        Ok(body.len())
    }

    /// Decode a packet into device memory, reporting whether the path was
    /// zero-copy (true under unified memory, where no H2D staging is needed).
    pub fn decode_to_device(&self, bytes: &[u8]) -> Result<DecodeResult, DevErr> {
        let body = self.decode_dense_body(bytes)?;
        self.place_body_on_device(body)
    }

    /// Allocate a device slab for `body` and stage it; `zero_copy` follows the
    /// backend's unified-memory flag. Factored out for testability.
    fn place_body_on_device(&self, body: &[u8]) -> Result<DecodeResult, DevErr> {
        let nbytes = body.len();
        let ptr = self.backend.alloc(nbytes)?;
        let zero_copy = self.backend.is_unified_memory();
        // Unified hardware needs no staging DMA; discrete needs an H2D. We
        // always mirror into the (mock) slab so the DevPtr is dereferenceable;
        // zero_copy records which physical path real hardware would take.
        self.backend.copy_h2d(ptr, body);
        Ok(DecodeResult {
            ptr,
            nbytes,
            zero_copy,
        })
    }

    /// Export a device allocation as a GPU IPC reference packet (106 bytes).
    pub fn export_ipc(&self, ptr: DevPtr, nbytes: usize) -> Result<Vec<u8>, DevErr> {
        let h = self.backend.export_ipc(ptr)?;
        let ipc = IpcRef {
            handle: h.bytes,
            byte_offset: h.byte_offset,
            // Prefer caller-stated size; fall back to the handle's nbytes.
            nbytes: if nbytes as u64 != 0 {
                nbytes as u64
            } else {
                h.nbytes
            },
            device_uuid: h.device_uuid,
        };
        Ok(build_ipc_packet(&ipc))
    }

    /// Import a GPU IPC reference packet, validating the device UUID against
    /// this backend before resolving the handle to a device pointer.
    pub fn import_ipc(&self, bytes: &[u8]) -> Result<DecodeResult, DevErr> {
        let ipc = parse_ipc_packet(bytes).map_err(DevErr::Core)?;
        // Reject UUID mismatch here too (defence in depth; backend re-checks).
        if ipc.device_uuid != self.backend.device_uuid() {
            return Err(DevErr::DeviceUuidMismatch);
        }
        let handle = IpcHandle {
            bytes: ipc.handle,
            byte_offset: ipc.byte_offset,
            nbytes: ipc.nbytes,
            device_uuid: ipc.device_uuid,
        };
        let ptr = self.backend.import_ipc(&handle)?;
        Ok(DecodeResult {
            ptr,
            nbytes: ipc.nbytes as usize,
            // Imported IPC is inherently zero-copy: shared, never staged.
            zero_copy: true,
        })
    }

    /// Decode `bytes` and borrow the dense body; non-dense/IPC packets error.
    fn decode_dense_body<'p>(&self, bytes: &'p [u8]) -> Result<&'p [u8], DevErr> {
        match tenso::decode(bytes).map_err(DevErr::Core)? {
            Decoded::Dense(view) => Ok(view.body),
            // Only dense bodies here; IpcRef goes through import_ipc, structured
            // packets aren't a single contiguous device body.
            _ => Err(DevErr::Other("expected dense packet for device decode")),
        }
    }
}

// ---- Tests (no GPU: CpuBackend, MockBackend, IPC frame) ----

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MockBackend: basic alloc / copy round-trip + op log ----

    #[test]
    fn mock_alloc_copy_roundtrip_and_oplog() {
        let be = MockBackend::new(false);
        let p = be.alloc(8).unwrap();
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        be.copy_h2d(p, &src);
        let mut dst = [0u8; 8];
        be.copy_d2h(&mut dst, p);
        assert_eq!(dst, src);
        be.free(p);

        let log = be.ops();
        assert_eq!(log[0], MockOp::Alloc(8));
        assert_eq!(log[1], MockOp::CopyH2D { dst: p, len: 8 });
        assert_eq!(log[2], MockOp::CopyD2H { src: p, len: 8 });
        assert_eq!(log[3], MockOp::Free(p));
    }

    #[test]
    fn mock_copy_truncates_to_slab_len() {
        let be = MockBackend::new(false);
        let p = be.alloc(4).unwrap();
        be.copy_h2d(p, &[9, 9, 9, 9, 9, 9]); // longer than slab
        assert_eq!(be.slab_bytes(p).unwrap(), vec![9, 9, 9, 9]);
    }

    // ---- MockBackend: configurable unified flag, alignment, OOM ----

    #[test]
    fn mock_unified_flag_is_configurable() {
        assert!(!MockBackend::new(false).is_unified_memory());
        assert!(MockBackend::new(true).is_unified_memory());
    }

    #[test]
    fn mock_alignment_is_configurable() {
        let be = MockBackend::new(false).with_alignment(128);
        assert_eq!(be.alignment(), 128);
        assert_eq!(MockBackend::new(false).alignment(), tenso::ALIGNMENT);
    }

    #[test]
    fn mock_capacity_triggers_oom() {
        let be = MockBackend::new(false).with_capacity(16);
        let _a = be.alloc(10).unwrap();
        // 10 + 10 > 16 => OOM
        assert_eq!(be.alloc(10), Err(DevErr::OutOfMemory));
        // freeing reclaims capacity
        be.free(_a);
        let _b = be.alloc(10).unwrap();
    }

    // ---- MockBackend: fault injection ----

    #[test]
    fn mock_fault_injection_on_nth_op() {
        let be = MockBackend::new(false);
        // op 0 = first alloc OK, fault the SECOND faultable op (the next alloc)
        be.alloc(4).unwrap(); // op_counter 0
        be.inject_fault_at(1); // trip when op_counter == 1
        assert_eq!(be.alloc(4), Err(DevErr::FaultInjected));
        // fault is one-shot
        be.alloc(4).unwrap();
    }

    #[test]
    fn mock_fault_injection_on_export() {
        let be = MockBackend::new(false);
        let p = be.alloc(4).unwrap(); // op 0
        be.inject_fault_at(1); // next faultable op = export
        assert_eq!(be.export_ipc(p), Err(DevErr::FaultInjected));
    }

    // ---- MockBackend: IPC export/import + uuid rejection ----

    #[test]
    fn mock_ipc_export_import_roundtrip() {
        let be = MockBackend::new(false);
        let p = be.alloc(32).unwrap();
        be.copy_h2d(p, &[7u8; 32]);
        let h = be.export_ipc(p).unwrap();
        let p2 = be.import_ipc(&h).unwrap();
        assert_eq!(p2, p);
        let mut out = [0u8; 32];
        be.copy_d2h(&mut out, p2);
        assert_eq!(out, [7u8; 32]);
    }

    #[test]
    fn mock_ipc_import_rejects_uuid_mismatch() {
        let exporter = MockBackend::new(false);
        let importer = MockBackend::new(false); // different uuid
        let p = exporter.alloc(8).unwrap();
        let h = exporter.export_ipc(p).unwrap();
        assert_eq!(importer.import_ipc(&h), Err(DevErr::DeviceUuidMismatch));
    }

    // ---- CpuBackend: real host memory round-trip + IPC ----

    #[test]
    fn cpu_alloc_copy_roundtrip() {
        let be = CpuBackend::new();
        let p = be.alloc(16).unwrap();
        let src: Vec<u8> = (0..16).collect();
        be.copy_h2d(p, &src);
        let mut dst = vec![0u8; 16];
        be.copy_d2h(&mut dst, p);
        assert_eq!(dst, src);
        be.free(p);
    }

    #[test]
    fn cpu_is_unified() {
        assert!(CpuBackend::new().is_unified_memory());
    }

    #[test]
    fn cpu_ipc_roundtrip_and_uuid_rejection() {
        let exporter = CpuBackend::new();
        let p = exporter.alloc(8).unwrap();
        exporter.copy_h2d(p, &[3, 1, 4, 1, 5, 9, 2, 6]);
        let h = exporter.export_ipc(p).unwrap();
        // Same backend imports fine.
        let p2 = exporter.import_ipc(&h).unwrap();
        let mut out = [0u8; 8];
        exporter.copy_d2h(&mut out, p2);
        assert_eq!(out, [3, 1, 4, 1, 5, 9, 2, 6]);
        // A different CpuBackend (different uuid) must reject.
        let other = CpuBackend::new();
        assert_eq!(other.import_ipc(&h), Err(DevErr::DeviceUuidMismatch));
    }

    // ---- IPC packet framing: build / parse round-trip + rule enforcement ----

    fn sample_ipc() -> IpcRef {
        let mut handle = [0u8; IPC_REF_HANDLE_LEN];
        for (i, b) in handle.iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut uuid = [0u8; IPC_REF_DEVICE_UUID_LEN];
        uuid.copy_from_slice(b"0123456789abcdef");
        IpcRef {
            handle,
            byte_offset: 0xdead_beef,
            nbytes: 4096,
            device_uuid: uuid,
        }
    }

    #[test]
    fn ipc_packet_roundtrip() {
        let ipc = sample_ipc();
        let pkt = build_ipc_packet(&ipc);
        assert_eq!(pkt.len(), IPC_REF_PACKET_LEN);
        // header sanity
        assert_eq!(&pkt[0..4], &MAGIC);
        assert_eq!(pkt[4], VERSION);
        assert_eq!(
            u16::from_le_bytes([pkt[5], pkt[6]]) & FLAG_GPU_IPC_REF,
            FLAG_GPU_IPC_REF
        );
        assert_eq!(pkt[9], IPC_REF_DISCRIMINATOR);
        // never combine with integrity / inline body
        assert_eq!(u16::from_le_bytes([pkt[5], pkt[6]]) & FLAG_INTEGRITY, 0);
        let parsed = parse_ipc_packet(&pkt).unwrap();
        assert_eq!(parsed, ipc);
    }

    #[test]
    fn ipc_packet_rejects_integrity_flag() {
        let ipc = sample_ipc();
        let mut pkt = build_ipc_packet(&ipc);
        // illegally set the INTEGRITY bit
        let flags = u16::from_le_bytes([pkt[5], pkt[6]]) | FLAG_INTEGRITY;
        pkt[5..7].copy_from_slice(&flags.to_le_bytes());
        assert_eq!(parse_ipc_packet(&pkt), Err(TensoError::Malformed));
    }

    #[test]
    fn ipc_packet_rejects_bad_discriminator() {
        let ipc = sample_ipc();
        let mut pkt = build_ipc_packet(&ipc);
        pkt[9] = 0; // not the IpcRef discriminator
        assert_eq!(parse_ipc_packet(&pkt), Err(TensoError::Malformed));
    }

    #[test]
    fn ipc_packet_rejects_short() {
        assert_eq!(parse_ipc_packet(&[0u8; 10]), Err(TensoError::TooShort));
    }

    // ---- GpuCodec: IPC export/import through the codec (uuid enforcement) ----

    #[test]
    fn codec_ipc_export_import_roundtrip() {
        let be = MockBackend::new(false);
        let codec = GpuCodec::new(&be);
        let p = be.alloc(64).unwrap();
        let pkt = codec.export_ipc(p, 64).unwrap();
        assert_eq!(pkt.len(), IPC_REF_PACKET_LEN);
        let res = codec.import_ipc(&pkt).unwrap();
        assert_eq!(res.ptr, p);
        assert_eq!(res.nbytes, 64);
        assert!(res.zero_copy);
    }

    #[test]
    fn codec_import_rejects_foreign_uuid() {
        let exporter = MockBackend::new(false);
        let importer = MockBackend::new(false);
        let exp_codec = GpuCodec::new(&exporter);
        let imp_codec = GpuCodec::new(&importer);
        let p = exporter.alloc(16).unwrap();
        let pkt = exp_codec.export_ipc(p, 16).unwrap();
        // The codec rejects on UUID before ever touching the backend registry.
        match imp_codec.import_ipc(&pkt) {
            Err(DevErr::DeviceUuidMismatch) => {}
            other => panic!(
                "expected DeviceUuidMismatch, got {:?}",
                other.map(|r| r.ptr)
            ),
        }
    }

    #[test]
    fn codec_export_uses_handle_nbytes_when_zero() {
        let be = MockBackend::new(false);
        let codec = GpuCodec::new(&be);
        let p = be.alloc(48).unwrap();
        // nbytes=0 => fall back to the backend handle's reported size (48).
        let pkt = codec.export_ipc(p, 0).unwrap();
        let parsed = parse_ipc_packet(&pkt).unwrap();
        assert_eq!(parsed.nbytes, 48);
    }

    // ---- GpuCodec: decode_to_device unified-vs-staged branch ----
    // Exercises place_body_on_device directly. Contract:
    // DecodeResult.zero_copy == backend.is_unified_memory().

    #[test]
    fn decode_branch_unified_reports_zero_copy() {
        let unified = MockBackend::new(true);
        let codec = GpuCodec::new(&unified);
        let body = [42u8; 24];
        let res = codec.place_body_on_device(&body).unwrap();
        assert!(res.zero_copy, "unified backend must report zero_copy");
        assert_eq!(res.nbytes, 24);
        assert_eq!(unified.slab_bytes(res.ptr).unwrap(), body.to_vec());
    }

    #[test]
    fn decode_branch_discrete_reports_staged() {
        let discrete = MockBackend::new(false);
        let codec = GpuCodec::new(&discrete);
        let body = [7u8; 24];
        let res = codec.place_body_on_device(&body).unwrap();
        assert!(!res.zero_copy, "discrete backend must NOT report zero_copy");
        assert_eq!(res.nbytes, 24);
        // The discrete path still performs an explicit H2D staging copy.
        let log = discrete.ops();
        assert!(log
            .iter()
            .any(|op| matches!(op, MockOp::CopyH2D { len: 24, .. })));
    }

    #[test]
    fn decode_branch_propagates_oom() {
        let be = MockBackend::new(false).with_capacity(8);
        let codec = GpuCodec::new(&be);
        let body = [0u8; 16];
        match codec.place_body_on_device(&body) {
            Err(DevErr::OutOfMemory) => {}
            other => panic!("expected OutOfMemory, got {:?}", other.map(|r| r.nbytes)),
        }
    }
}
