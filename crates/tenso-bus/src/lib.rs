//! tenso-bus: a shared-memory tensor bus built over tenso-core.
//!
//! Two transport shapes, both designed for the robotics hot path:
//!
//!   - [`LatestValueBus`] — a triple-buffered **seqlock** single-slot buffer for
//!     telemetry-style publish/subscribe. A single writer publishes; any number
//!     of readers always observe the newest *complete* frame with a lock-free,
//!     wait-free (bounded-retry) read. No torn reads.
//!
//!   - [`RingBus`] — an **SPMC** ring of fixed-stride slots with a configurable
//!     [`OverflowPolicy`] (`DropOldest`, `Block`, `Error`). One producer writes,
//!     many consumers each track their own read cursor and can detect lag.
//!
//! Packets on the wire are exactly tenso-core packets, so any reader (Rust,
//! C/C++, ROS2) decodes them with `tenso_core::decode`. Every packet is
//! validated with `tenso_core::parse_header` before it is accepted.
//!
//! ## Synchronization model
//!
//! The fast paths are lock-free:
//!   - latest-value uses a seqlock (odd sequence ⇒ write in progress) plus a
//!     triple-buffer index so the writer never blocks readers and readers never
//!     block the writer.
//!   - the ring uses a monotonically increasing write index with per-slot
//!     publication sequence numbers (a slot-local seqlock) so readers detect a
//!     half-written or overwritten slot.
//!
//! A robust POSIX `pthread_mutex` (see the [`shm_mutex`] module) lives in the
//! header for the **slow path only**: serializing multiple *publishers* /
//! *producers* across processes and recovering (on Linux) from a producer that
//! crashed mid-publish. On macOS the robust attribute is unavailable, so the
//! mutex degrades to a plain process-shared mutex (no `EOWNERDEAD` recovery) and
//! the seqlock alone still prevents torn reads.
//!
//! ## Instrumentation
//!
//! Both buses keep [`BusStats`] counters (`copies`, `drops`, `lag_events`,
//! `recoveries`) in the shared header so producers and consumers in any process
//! can observe throughput/loss.

#![allow(clippy::missing_safety_doc)]

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use tenso_core::TensoError;

/// 64-byte SIMD/cache-line alignment used for all slot bodies (matches the
/// tenso-core wire alignment so a decoded body is itself well-aligned).
pub const ALIGNMENT: usize = 64;

/// Errors from bus operations.
#[derive(Debug)]
pub enum BusError {
    /// Failed to create/open/map the shared-memory segment.
    Shm(&'static str),
    /// Ring is full and the policy is to reject rather than overwrite.
    Full,
    /// No message currently available (non-blocking read).
    Empty,
    /// A reader fell too far behind and its slot was overwritten.
    Lagged,
    /// Caller's output buffer is smaller than the stored packet.
    OutputTooSmall,
    /// Packet does not fit in a slot / exceeds the configured slot stride.
    PacketTooLarge,
    /// Underlying packet was malformed.
    Core(TensoError),
}

impl From<TensoError> for BusError {
    fn from(e: TensoError) -> Self {
        BusError::Core(e)
    }
}

/// Overflow behaviour for [`RingBus::push`] when the ring is full.
///
/// BACKPRESSURE LIMITATION: `Block` and `Error` decide "is a slot still unread?"
/// from the PRODUCER handle's own `read_cursor`, which the producer only advances
/// when it also consumes (loopback). In a true SPMC topology (one producer handle,
/// separate consumer handles/processes) the producer cannot see the independent
/// consumer cursors — there is no shared minimum-consumer-cursor in the ring
/// header — so `Block` can never observe a slot being freed and will busy-wait to
/// the lock timeout and then return `Full`, and `Error` reports `Full` purely from
/// the producer's own non-advancing cursor. Use `Block`/`Error` only in a
/// single-handle (loopback) configuration; for multi-process consumers use the
/// default `DropOldest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Overwrite the oldest unread slot (lossy, never blocks the producer).
    /// This is the default for the telemetry hot path and the only policy that
    /// behaves correctly for multi-process SPMC consumers.
    DropOldest,
    /// Block (busy-wait with a short backoff) until a consumer frees a slot.
    /// Only effective in a single-handle/loopback config (see the type-level
    /// BACKPRESSURE LIMITATION note); in true SPMC it times out and returns `Full`.
    Block,
    /// Return [`BusError::Full`] immediately when the producer's own cursor shows
    /// the ring is full. See the type-level BACKPRESSURE LIMITATION note: this is
    /// meaningful only in a single-handle/loopback config.
    Error,
}

impl Default for OverflowPolicy {
    fn default() -> Self {
        OverflowPolicy::DropOldest
    }
}

/// Bus configuration shared by both transports.
pub struct BusConfig<'a> {
    /// POSIX shm name (e.g. "/tenso_bus_camera"). Must start with '/'.
    pub name: &'a str,
    /// Capacity hint.
    ///
    /// - For [`RingBus`] this is the number of slots.
    /// - For [`LatestValueBus`] it is ignored (always triple-buffered).
    pub capacity: usize,
    /// Create (true) vs attach to an existing segment (false).
    pub create: bool,
}

impl<'a> BusConfig<'a> {
    /// Convenience constructor.
    pub fn new(name: &'a str, capacity: usize, create: bool) -> Self {
        BusConfig {
            name,
            capacity,
            create,
        }
    }
}

/// Snapshot of bus instrumentation counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BusStats {
    /// Number of payload `memcpy`s performed (publish + read).
    pub copies: u64,
    /// Frames dropped (overwritten before being read, or `DropOldest`).
    pub drops: u64,
    /// Times a reader detected it had lagged and was forced to skip.
    pub lag_events: u64,
    /// Times a crashed writer's mutex was recovered (Linux robust mutex).
    pub recoveries: u64,
}

// =============================================================================
// Cross-process robust mutex (POSIX pthread_mutex in shared memory)
//
// Ported from the root crate's `shm_mutex` module (src/lib.rs). Used here only
// for the slow path: serializing multiple writers/publishers across processes
// and recovering from a writer that died mid-critical-section.
// =============================================================================

#[cfg(unix)]
pub mod shm_mutex {
    use core::time::Duration;

    /// Bytes reserved for one `pthread_mutex_t` in shared memory.
    ///
    /// On macOS a `pthread_mutex_t` is 64 bytes; on Linux it is 40. We reserve
    /// 64 unconditionally so the header layout is identical across platforms.
    pub const MUTEX_SIZE: usize = 64;

    extern "C" {
        fn pthread_mutexattr_init(attr: *mut libc::pthread_mutexattr_t) -> libc::c_int;
        fn pthread_mutexattr_setpshared(
            attr: *mut libc::pthread_mutexattr_t,
            pshared: libc::c_int,
        ) -> libc::c_int;
        fn pthread_mutexattr_destroy(attr: *mut libc::pthread_mutexattr_t) -> libc::c_int;
        fn pthread_mutex_init(
            mutex: *mut libc::pthread_mutex_t,
            attr: *const libc::pthread_mutexattr_t,
        ) -> libc::c_int;
        fn pthread_mutex_unlock(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_trylock(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_destroy(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;

        // Robust mutex support (Linux only).
        #[cfg(target_os = "linux")]
        fn pthread_mutexattr_setrobust(
            attr: *mut libc::pthread_mutexattr_t,
            robust: libc::c_int,
        ) -> libc::c_int;
        #[cfg(target_os = "linux")]
        fn pthread_mutex_consistent(mutex: *mut libc::pthread_mutex_t) -> libc::c_int;
    }

    /// Initialize a process-shared mutex at the given memory location.
    ///
    /// # Safety
    /// `ptr` must point to at least [`MUTEX_SIZE`] bytes of zeroed shared memory
    /// that outlives every user of the mutex.
    pub unsafe fn init_mutex(ptr: *mut u8) -> Result<(), &'static str> {
        let mutex = ptr as *mut libc::pthread_mutex_t;
        let mut attr: libc::pthread_mutexattr_t = core::mem::zeroed();

        if pthread_mutexattr_init(&mut attr) != 0 {
            return Err("pthread_mutexattr_init failed");
        }
        if pthread_mutexattr_setpshared(&mut attr, libc::PTHREAD_PROCESS_SHARED) != 0 {
            pthread_mutexattr_destroy(&mut attr);
            return Err("pthread_mutexattr_setpshared failed");
        }

        // On Linux, enable a robust mutex so we can recover from crashed holders.
        // macOS lacks PTHREAD_MUTEX_ROBUST; it gracefully degrades to a plain
        // process-shared mutex.
        #[cfg(target_os = "linux")]
        {
            if pthread_mutexattr_setrobust(&mut attr, libc::PTHREAD_MUTEX_ROBUST) != 0 {
                pthread_mutexattr_destroy(&mut attr);
                return Err("pthread_mutexattr_setrobust failed");
            }
        }

        let rc = pthread_mutex_init(mutex, &attr);
        pthread_mutexattr_destroy(&mut attr);
        if rc != 0 {
            return Err("pthread_mutex_init failed");
        }
        Ok(())
    }

    /// Lock the mutex with a timeout.
    ///
    /// Returns `Ok(true)` if the lock was acquired after recovering from a dead
    /// owner (Linux robust mutex), `Ok(false)` on a normal acquisition.
    ///
    /// # Safety
    /// `ptr` must reference a mutex previously initialized with [`init_mutex`].
    pub unsafe fn lock_mutex(ptr: *mut u8, timeout: Duration) -> Result<bool, &'static str> {
        let mutex = ptr as *mut libc::pthread_mutex_t;
        let deadline = std::time::Instant::now() + timeout;

        loop {
            let rc = pthread_mutex_trylock(mutex);
            if rc == 0 {
                return Ok(false); // normal acquisition
            }

            // EOWNERDEAD: previous holder crashed (Linux robust mutex).
            #[cfg(target_os = "linux")]
            if rc == libc::EOWNERDEAD {
                pthread_mutex_consistent(mutex);
                return Ok(true); // recovered
            }

            if rc == libc::EBUSY {
                if std::time::Instant::now() >= deadline {
                    return Err("mutex lock timed out");
                }
                std::thread::sleep(Duration::from_micros(50));
                continue;
            }

            return Err("pthread_mutex_trylock failed");
        }
    }

    /// Unlock the mutex.
    ///
    /// # Safety
    /// `ptr` must reference a mutex this thread currently holds.
    pub unsafe fn unlock_mutex(ptr: *mut u8) -> Result<(), &'static str> {
        let mutex = ptr as *mut libc::pthread_mutex_t;
        if pthread_mutex_unlock(mutex) != 0 {
            return Err("pthread_mutex_unlock failed");
        }
        Ok(())
    }

    /// Destroy the mutex.
    ///
    /// # Safety
    /// `ptr` must reference a mutex no other thread/process is using.
    pub unsafe fn destroy_mutex(ptr: *mut u8) -> Result<(), &'static str> {
        let mutex = ptr as *mut libc::pthread_mutex_t;
        if pthread_mutex_destroy(mutex) != 0 {
            return Err("pthread_mutex_destroy failed");
        }
        Ok(())
    }
}

// =============================================================================
// Shared-memory segment (POSIX shm_open + mmap), #[cfg(unix)]
// =============================================================================

#[cfg(unix)]
mod shm {
    use super::BusError;

    /// An mmap'd POSIX shared-memory segment.
    ///
    /// Owns the mapping; unmaps on drop. The backing shm object is *not*
    /// unlinked on drop (the bus may outlive a single process), but the creator
    /// can call [`Segment::unlink`] explicitly.
    pub struct Segment {
        ptr: *mut u8,
        len: usize,
        name: std::ffi::CString,
        creator: bool,
    }

    // The mapping is process-shared; raw pointer access is synchronized by the
    // bus's seqlock/mutex discipline, not by the type system.
    unsafe impl Send for Segment {}
    unsafe impl Sync for Segment {}

    impl Segment {
        /// Create or attach a shm segment of exactly `len` bytes.
        ///
        /// When `create` is true the object is created (or truncated to `len`)
        /// and zero-filled by the kernel. When false it is attached read/write
        /// and must already be at least `len` bytes.
        pub fn open(name: &str, len: usize, create: bool) -> Result<Self, BusError> {
            if len == 0 {
                return Err(BusError::Shm("zero-length segment"));
            }
            let cname =
                std::ffi::CString::new(name).map_err(|_| BusError::Shm("shm name contains NUL"))?;

            let mut oflag = libc::O_RDWR;
            if create {
                oflag |= libc::O_CREAT;
            }
            let mode: libc::mode_t = 0o600;

            let fd = unsafe { libc::shm_open(cname.as_ptr(), oflag, mode as libc::c_uint) };
            if fd < 0 {
                return Err(BusError::Shm("shm_open failed"));
            }

            if create {
                // Size the object. ftruncate-grown pages are zero-filled.
                let rc = unsafe { libc::ftruncate(fd, len as libc::off_t) };
                if rc != 0 {
                    unsafe { libc::close(fd) };
                    return Err(BusError::Shm("ftruncate failed"));
                }
            } else {
                // Verify the existing object is large enough.
                let mut st: libc::stat = unsafe { core::mem::zeroed() };
                if unsafe { libc::fstat(fd, &mut st) } != 0 {
                    unsafe { libc::close(fd) };
                    return Err(BusError::Shm("fstat failed"));
                }
                if (st.st_size as usize) < len {
                    unsafe { libc::close(fd) };
                    return Err(BusError::Shm("existing segment too small"));
                }
            }

            let ptr = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                )
            };
            // The fd is no longer needed once mapped.
            unsafe { libc::close(fd) };

            if ptr == libc::MAP_FAILED {
                return Err(BusError::Shm("mmap failed"));
            }

            Ok(Segment {
                ptr: ptr as *mut u8,
                len,
                name: cname,
                creator: create,
            })
        }

        #[inline]
        pub fn as_ptr(&self) -> *mut u8 {
            self.ptr
        }

        #[inline]
        pub fn len(&self) -> usize {
            self.len
        }

        #[inline]
        pub fn is_creator(&self) -> bool {
            self.creator
        }

        /// Remove the shm object from the namespace (existing mappings persist).
        pub fn unlink(&self) {
            unsafe {
                libc::shm_unlink(self.name.as_ptr());
            }
        }
    }

    impl Drop for Segment {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

// =============================================================================
// Shared header layouts
// =============================================================================

/// Magic stamped at the front of a bus segment so attachers can sanity-check.
const BUS_MAGIC: u32 = u32::from_le_bytes(*b"TBUS");

/// Layout of the latest-value (triple-buffer seqlock) segment header.
///
/// `#[repr(C)]` so the layout is stable across processes/architectures of the
/// same ABI. All multi-process mutable fields are atomics so the compiler does
/// not assume exclusive access.
#[repr(C)]
struct LvbHeader {
    magic: AtomicU32,
    _pad0: u32,
    /// Seqlock: even = stable, odd = write in progress. Bumped twice per publish.
    seq: AtomicU64,
    /// Index (0..3) of the buffer holding the most recently published frame.
    front: AtomicU32,
    _pad1: u32,
    /// Per-buffer published payload length (only meaningful for `front`).
    lens: [AtomicU32; 3],
    _pad2: u32,
    /// Max payload bytes per buffer (set at create).
    slot_stride: AtomicU32,
    _pad3: u32,
    // Instrumentation
    copies: AtomicU64,
    drops: AtomicU64,
    lag_events: AtomicU64,
    recoveries: AtomicU64,
    // Slow-path publisher mutex (raw pthread_mutex storage).
    #[cfg(unix)]
    mutex: [u8; shm_mutex::MUTEX_SIZE],
}

/// Layout of the SPMC ring segment header.
#[repr(C)]
struct RingHeader {
    magic: AtomicU32,
    _pad0: u32,
    /// Number of slots (power of two not required; we use modulo).
    capacity: AtomicU32,
    /// Bytes per slot body (payload region, 64-byte aligned).
    slot_stride: AtomicU32,
    /// Monotonic count of packets ever written. Slot index = write_index % cap.
    write_index: AtomicU64,
    // Instrumentation
    copies: AtomicU64,
    drops: AtomicU64,
    lag_events: AtomicU64,
    recoveries: AtomicU64,
    // Slow-path producer mutex.
    #[cfg(unix)]
    mutex: [u8; shm_mutex::MUTEX_SIZE],
}

/// Per-slot metadata header in the ring (immediately precedes each slot body).
#[repr(C)]
struct SlotMeta {
    /// Slot-local seqlock: even = stable, odd = write in progress.
    seq: AtomicU64,
    /// Absolute write index of the packet currently in this slot (for lag
    /// detection: a reader knows the slot was reused if this advanced).
    write_index: AtomicU64,
    /// Payload byte length.
    len: AtomicU32,
    _pad: u32,
}

#[inline]
const fn align_up(n: usize, a: usize) -> usize {
    (n + a - 1) & !(a - 1)
}

/// Round a struct size up to 64 so the following region is cache-aligned.
#[inline]
const fn aligned_hdr(n: usize) -> usize {
    align_up(n, ALIGNMENT)
}

// =============================================================================
// LatestValueBus — triple-buffered seqlock (lock-free reader, single writer)
// =============================================================================

/// A shared-memory single-slot buffer where the newest packet wins.
///
/// Triple-buffered: the writer always writes into the buffer that is neither the
/// current `front` nor the one a reader could currently be reading, then flips
/// `front`. Readers use the seqlock to retry if a publish raced with their read,
/// guaranteeing tear-free reads without ever blocking the writer.
pub struct LatestValueBus {
    #[cfg(unix)]
    seg: shm::Segment,
    /// Cached pointers/strides resolved from the header at open time.
    slot_stride: usize,
    /// Default timeout for the slow-path publisher mutex.
    lock_timeout: core::time::Duration,
}

impl LatestValueBus {
    #[inline]
    fn header_bytes() -> usize {
        aligned_hdr(core::mem::size_of::<LvbHeader>())
    }

    /// Total segment size for a given per-slot stride.
    fn segment_size(slot_stride: usize) -> usize {
        Self::header_bytes() + 3 * align_up(slot_stride, ALIGNMENT)
    }

    /// Create or attach a latest-value bus.
    ///
    /// `cfg.capacity` is interpreted as the maximum payload bytes per frame
    /// (the slot stride). It is rounded up to 64 bytes. A floor of 64 bytes
    /// applies so tiny configs still hold a header-only packet.
    #[cfg(unix)]
    pub fn open(cfg: &BusConfig) -> Result<Self, BusError> {
        let slot_stride = align_up(cfg.capacity.max(ALIGNMENT), ALIGNMENT);
        let size = Self::segment_size(slot_stride);
        let seg = shm::Segment::open(cfg.name, size, cfg.create)?;

        // Safety: segment is at least `size` bytes; LvbHeader fits at offset 0.
        let hdr = unsafe { &*(seg.as_ptr() as *const LvbHeader) };

        if cfg.create {
            // Zero-filled by the kernel. Initialize the mutex first, then the
            // header fields; `magic_init` publishes `magic` last so an attacher
            // that observes the magic also observes a ready mutex + fields.
            unsafe {
                let mptr = seg.as_ptr().add(Self::mutex_offset());
                shm_mutex::init_mutex(mptr).map_err(BusError::Shm)?;
            }
            hdr.magic_init(slot_stride as u32);
        } else if hdr.magic.load(Ordering::Acquire) != BUS_MAGIC {
            return Err(BusError::Shm("not a tenso latest-value bus"));
        }

        let resolved_stride = hdr.slot_stride.load(Ordering::Acquire) as usize;
        Ok(LatestValueBus {
            seg,
            slot_stride: resolved_stride,
            lock_timeout: core::time::Duration::from_secs(5),
        })
    }

    #[cfg(unix)]
    #[inline]
    fn mutex_offset() -> usize {
        // The mutex is the trailing field of LvbHeader.
        core::mem::size_of::<LvbHeader>() - shm_mutex::MUTEX_SIZE
    }

    #[cfg(unix)]
    #[inline]
    fn header(&self) -> &LvbHeader {
        unsafe { &*(self.seg.as_ptr() as *const LvbHeader) }
    }

    #[cfg(unix)]
    #[inline]
    fn buffer_ptr(&self, idx: usize) -> *mut u8 {
        let base = Self::header_bytes();
        unsafe {
            self.seg
                .as_ptr()
                .add(base + idx * align_up(self.slot_stride, ALIGNMENT))
        }
    }

    /// Atomically replace the stored packet with `packet` (single writer).
    ///
    /// The packet must be a valid tenso-core packet and fit in the slot stride.
    #[cfg(unix)]
    pub fn publish(&self, packet: &[u8]) -> Result<(), BusError> {
        // Validate the wire packet before accepting it.
        tenso_core::parse_header(packet)?;
        if packet.len() > self.slot_stride {
            return Err(BusError::PacketTooLarge);
        }

        let hdr = self.header();

        // Slow path: serialize concurrent publishers across processes. With a
        // single writer this is uncontended; with several it provides ordering
        // and (on Linux) crash recovery.
        unsafe {
            let mptr = self.seg.as_ptr().add(Self::mutex_offset());
            match shm_mutex::lock_mutex(mptr, self.lock_timeout) {
                Ok(recovered) => {
                    if recovered {
                        hdr.recoveries.fetch_add(1, Ordering::Relaxed);
                        // A writer died mid-publish (EOWNERDEAD) holding the lock,
                        // possibly leaving the seqlock ODD ("write in progress").
                        // Each publish adds 2 (parity-preserving), so an odd seq
                        // would wedge every reader's tear-check FOREVER. Restore
                        // even parity by completing the abandoned increment before
                        // republishing. `front` already points at a fully-written
                        // buffer (the dead writer copied + stored len before going
                        // odd, and either had not flipped front or flipped it to
                        // that complete buffer), so readers observe an intact frame.
                        if hdr.seq.load(Ordering::Acquire) & 1 == 1 {
                            hdr.seq.fetch_add(1, Ordering::AcqRel); // odd -> even
                        }
                    }
                }
                Err(_) => return Err(BusError::Shm("publisher mutex lock failed")),
            }
        }

        // Pick a back buffer: anything that is not the current front. With three
        // buffers a reader can hold at most one (the front it latched), so the
        // writer always has a free buffer that no reader is touching.
        let front = hdr.front.load(Ordering::Acquire) as usize;
        let back = match front {
            0 => 1,
            1 => 2,
            _ => 0,
        };

        // Copy payload into the back buffer (no seqlock needed yet: not visible).
        unsafe {
            core::ptr::copy_nonoverlapping(packet.as_ptr(), self.buffer_ptr(back), packet.len());
        }
        hdr.lens[back].store(packet.len() as u32, Ordering::Release);
        hdr.copies.fetch_add(1, Ordering::Relaxed);

        // Publish via seqlock: odd (writing) -> flip front -> even (stable).
        hdr.seq.fetch_add(1, Ordering::AcqRel); // -> odd
        hdr.front.store(back as u32, Ordering::Release);
        hdr.seq.fetch_add(1, Ordering::AcqRel); // -> even

        unsafe {
            let mptr = self.seg.as_ptr().add(Self::mutex_offset());
            let _ = shm_mutex::unlock_mutex(mptr);
        }

        Ok(())
    }

    /// Read the latest packet into `out` (tear-free), returning its length.
    ///
    /// Lock-free, bounded-retry. Returns [`BusError::Empty`] if nothing has been
    /// published yet, [`BusError::OutputTooSmall`] if `out` cannot hold the
    /// frame.
    ///
    /// **Seqlock guarantee:** the writer rotates through 3 buffers, so it cannot
    /// reuse the buffer this read latched until two further publishes occur.
    /// The post-copy `seq` re-check discards any copy that overlapped a publish,
    /// so a *torn* frame is never returned. A reader stalled across two-plus
    /// publishes simply retries and observes the newest stable frame.
    #[cfg(unix)]
    pub fn read_latest(&self, out: &mut [u8]) -> Result<usize, BusError> {
        let hdr = self.header();

        // Bounded retry: a publish is two atomic increments, so a reader can be
        // starved only by an adversarial publish storm. 1024 attempts is far
        // beyond any real scheduling window.
        for _ in 0..1024 {
            let s0 = hdr.seq.load(Ordering::Acquire);
            if s0 & 1 == 1 {
                core::hint::spin_loop();
                continue; // write in progress
            }
            if s0 == 0 {
                return Err(BusError::Empty); // nothing published yet
            }

            let front = hdr.front.load(Ordering::Acquire) as usize;
            let len = hdr.lens[front].load(Ordering::Acquire) as usize;

            if len > out.len() {
                // Re-check the seqlock so we don't surface a length from a torn
                // read; if stable, the length is real and the buffer is small.
                if hdr.seq.load(Ordering::Acquire) == s0 {
                    return Err(BusError::OutputTooSmall);
                }
                hdr.lag_events.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            unsafe {
                core::ptr::copy_nonoverlapping(self.buffer_ptr(front), out.as_mut_ptr(), len);
            }

            // Acquire fence between the (plain, non-atomic) payload copy above and
            // the validating seq load below. An Acquire *load* alone would not stop
            // the data reads from being sunk past it, so on a weakly-ordered target
            // (ARM/aarch64) the tear-check could pass while bytes were sampled after
            // a concurrent publish. The standalone fence pins the data reads before
            // the re-read of `seq`, which is the canonical seqlock reader barrier.
            core::sync::atomic::fence(Ordering::Acquire);

            // Validate the seqlock did not move (no concurrent publish flipped
            // the buffer under us). Triple-buffering means the front we read was
            // not the writer's target, but a fresh publish could have advanced
            // `front`; retry to always return the newest stable frame.
            let s1 = hdr.seq.load(Ordering::Acquire);
            if s1 == s0 {
                hdr.copies.fetch_add(1, Ordering::Relaxed);
                return Ok(len);
            }
            hdr.lag_events.fetch_add(1, Ordering::Relaxed);
            core::hint::spin_loop();
        }
        Err(BusError::Lagged)
    }

    /// Snapshot the instrumentation counters.
    #[cfg(unix)]
    pub fn stats(&self) -> BusStats {
        let hdr = self.header();
        BusStats {
            copies: hdr.copies.load(Ordering::Relaxed),
            drops: hdr.drops.load(Ordering::Relaxed),
            lag_events: hdr.lag_events.load(Ordering::Relaxed),
            recoveries: hdr.recoveries.load(Ordering::Relaxed),
        }
    }

    /// Unlink the backing shm object (call once, from the creator, at shutdown).
    #[cfg(unix)]
    pub fn unlink(&self) {
        self.seg.unlink();
    }

    // --- non-unix stubs so the crate still type-checks on, e.g., Windows ---

    #[cfg(not(unix))]
    pub fn open(_cfg: &BusConfig) -> Result<Self, BusError> {
        Err(BusError::Shm(
            "tenso-bus requires a unix shared-memory backend",
        ))
    }
    #[cfg(not(unix))]
    pub fn publish(&self, _packet: &[u8]) -> Result<(), BusError> {
        Err(BusError::Shm("unsupported platform"))
    }
    #[cfg(not(unix))]
    pub fn read_latest(&self, _out: &mut [u8]) -> Result<usize, BusError> {
        Err(BusError::Shm("unsupported platform"))
    }
}

#[cfg(unix)]
impl LvbHeader {
    /// Initialize a freshly zeroed (kernel zero-filled) header.
    fn magic_init(&self, slot_stride: u32) {
        self.slot_stride.store(slot_stride, Ordering::Release);
        self.front.store(0, Ordering::Release);
        // seq starts at 0 (== "nothing published"); first publish leaves it == 2.
        self.seq.store(0, Ordering::Release);
        // Store `magic` last (Release) so an attacher that observes the magic
        // with an Acquire load also observes the initialized fields above.
        self.magic.store(BUS_MAGIC, Ordering::Release);
    }
}

// =============================================================================
// RingBus — SPMC ring with OverflowPolicy
// =============================================================================

/// A shared-memory ring of Tenso packets (single producer, multiple consumers).
///
/// Each consumer should hold its own [`RingBus`] handle and call [`pop`] which
/// advances a *per-handle* read cursor. The producer calls [`push`]. Slots are
/// 64-byte aligned; each carries a slot-local seqlock so a consumer can detect a
/// half-written or recycled slot and report [`BusError::Lagged`].
///
/// [`pop`]: RingBus::pop
pub struct RingBus {
    #[cfg(unix)]
    seg: shm::Segment,
    capacity: usize,
    slot_stride: usize,
    policy: OverflowPolicy,
    /// Per-handle (per-consumer) read cursor. Interior-mutable (atomic) so `pop`
    /// keeps the contract's `&self` signature while the handle stays `Send +
    /// Sync` (each consumer still owns its own handle; the atomic just avoids
    /// `Cell`'s `!Sync`).
    read_cursor: AtomicU64,
    lock_timeout: core::time::Duration,
}

impl RingBus {
    #[inline]
    fn header_bytes() -> usize {
        aligned_hdr(core::mem::size_of::<RingHeader>())
    }

    /// Bytes occupied by one slot: aligned SlotMeta + aligned payload.
    #[inline]
    fn slot_bytes(slot_stride: usize) -> usize {
        aligned_hdr(core::mem::size_of::<SlotMeta>()) + align_up(slot_stride, ALIGNMENT)
    }

    fn segment_size(capacity: usize, slot_stride: usize) -> usize {
        Self::header_bytes() + capacity * Self::slot_bytes(slot_stride)
    }

    /// Default per-slot payload stride when attaching needs a fallback.
    const DEFAULT_SLOT_STRIDE: usize = 1 << 20; // 1 MiB

    /// Create or attach a ring bus with the default [`OverflowPolicy`].
    ///
    /// `cfg.capacity` is the number of slots. The per-slot payload stride is
    /// derived: when creating we use [`DEFAULT_SLOT_STRIDE`]; see
    /// [`RingBus::open_with`] for explicit control.
    ///
    /// [`DEFAULT_SLOT_STRIDE`]: RingBus::DEFAULT_SLOT_STRIDE
    #[cfg(unix)]
    pub fn open(cfg: &BusConfig) -> Result<Self, BusError> {
        Self::open_with(cfg, Self::DEFAULT_SLOT_STRIDE, OverflowPolicy::default())
    }

    /// Create or attach with an explicit slot stride and overflow policy.
    #[cfg(unix)]
    pub fn open_with(
        cfg: &BusConfig,
        slot_stride: usize,
        policy: OverflowPolicy,
    ) -> Result<Self, BusError> {
        let capacity = cfg.capacity.max(1);
        let stride = align_up(slot_stride.max(ALIGNMENT), ALIGNMENT);
        let size = Self::segment_size(capacity, stride);
        let seg = shm::Segment::open(cfg.name, size, cfg.create)?;

        let hdr = unsafe { &*(seg.as_ptr() as *const RingHeader) };

        if cfg.create {
            hdr.capacity.store(capacity as u32, Ordering::Release);
            hdr.slot_stride.store(stride as u32, Ordering::Release);
            hdr.write_index.store(0, Ordering::Release);
            unsafe {
                let mptr = seg.as_ptr().add(Self::mutex_offset());
                shm_mutex::init_mutex(mptr).map_err(BusError::Shm)?;
            }
            // Store magic last (Release) so attachers observing it also observe
            // the initialized capacity/stride/write_index above.
            hdr.magic.store(BUS_MAGIC, Ordering::Release);
        } else if hdr.magic.load(Ordering::Acquire) != BUS_MAGIC {
            return Err(BusError::Shm("not a tenso ring bus"));
        }

        let resolved_cap = hdr.capacity.load(Ordering::Acquire) as usize;
        let resolved_stride = hdr.slot_stride.load(Ordering::Acquire) as usize;
        // Attaching consumers start reading from the current head so they only
        // see frames published after they joined (telemetry semantics).
        let start = hdr.write_index.load(Ordering::Acquire);

        Ok(RingBus {
            seg,
            capacity: resolved_cap,
            slot_stride: resolved_stride,
            policy,
            read_cursor: AtomicU64::new(start),
            lock_timeout: core::time::Duration::from_secs(5),
        })
    }

    /// Override this handle's overflow policy.
    pub fn with_policy(mut self, policy: OverflowPolicy) -> Self {
        self.policy = policy;
        self
    }

    #[cfg(unix)]
    #[inline]
    fn mutex_offset() -> usize {
        core::mem::size_of::<RingHeader>() - shm_mutex::MUTEX_SIZE
    }

    #[cfg(unix)]
    #[inline]
    fn header(&self) -> &RingHeader {
        unsafe { &*(self.seg.as_ptr() as *const RingHeader) }
    }

    #[cfg(unix)]
    #[inline]
    fn slot_base(&self, slot: usize) -> *mut u8 {
        let base = Self::header_bytes();
        unsafe {
            self.seg
                .as_ptr()
                .add(base + slot * Self::slot_bytes(self.slot_stride))
        }
    }

    #[cfg(unix)]
    #[inline]
    fn slot_meta(&self, slot: usize) -> &SlotMeta {
        unsafe { &*(self.slot_base(slot) as *const SlotMeta) }
    }

    #[cfg(unix)]
    #[inline]
    fn slot_payload(&self, slot: usize) -> *mut u8 {
        unsafe {
            self.slot_base(slot)
                .add(aligned_hdr(core::mem::size_of::<SlotMeta>()))
        }
    }

    /// Publish a fully-formed Tenso packet onto the ring (single producer).
    #[cfg(unix)]
    pub fn push(&self, packet: &[u8]) -> Result<(), BusError> {
        tenso_core::parse_header(packet)?;
        if packet.len() > self.slot_stride {
            return Err(BusError::PacketTooLarge);
        }

        let hdr = self.header();

        // Serialize concurrent producers (slow path / crash recovery).
        unsafe {
            let mptr = self.seg.as_ptr().add(Self::mutex_offset());
            match shm_mutex::lock_mutex(mptr, self.lock_timeout) {
                Ok(recovered) => {
                    if recovered {
                        hdr.recoveries.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => return Err(BusError::Shm("producer mutex lock failed")),
            }
        }

        let result = self.push_locked(hdr, packet);

        unsafe {
            let mptr = self.seg.as_ptr().add(Self::mutex_offset());
            let _ = shm_mutex::unlock_mutex(mptr);
        }
        result
    }

    #[cfg(unix)]
    fn push_locked(&self, hdr: &RingHeader, packet: &[u8]) -> Result<(), BusError> {
        let widx = hdr.write_index.load(Ordering::Acquire);

        // Capacity check: a slot is "occupied/unread" when the oldest in-flight
        // index (widx - capacity) has not yet been consumed. We can't know every
        // consumer's cursor across processes cheaply, so the policy governs how
        // we treat a full ring. For DropOldest we simply overwrite (each
        // consumer detects the overwrite via the slot's write_index jump).
        match self.policy {
            OverflowPolicy::DropOldest => {
                if widx >= self.capacity as u64 {
                    hdr.drops.fetch_add(1, Ordering::Relaxed);
                }
            }
            OverflowPolicy::Error => {
                // "Full" only ever matters relative to *this* handle's own
                // consumption; with a global producer and independent consumers
                // we approximate full as: the slot we are about to stomp still
                // holds an unread index for this handle.
                let slot = (widx as usize) % self.capacity;
                let meta = self.slot_meta(slot);
                let occupant = meta.write_index.load(Ordering::Acquire);
                if widx >= self.capacity as u64
                    && occupant + 1 > self.read_cursor.load(Ordering::Relaxed)
                {
                    return Err(BusError::Full);
                }
            }
            OverflowPolicy::Block => {
                let slot = (widx as usize) % self.capacity;
                let meta = self.slot_meta(slot);
                let deadline = std::time::Instant::now() + self.lock_timeout;
                while widx >= self.capacity as u64 {
                    let occupant = meta.write_index.load(Ordering::Acquire);
                    // Free once the prospective occupant has been read by this
                    // handle (best-effort single-host backpressure).
                    if occupant + 1 <= self.read_cursor.load(Ordering::Relaxed) {
                        break;
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err(BusError::Full);
                    }
                    std::thread::sleep(core::time::Duration::from_micros(50));
                }
            }
        }

        let slot = (widx as usize) % self.capacity;
        let meta = self.slot_meta(slot);

        // Slot-local seqlock publish: odd -> write body -> even.
        meta.seq.fetch_add(1, Ordering::AcqRel); // -> odd
        unsafe {
            core::ptr::copy_nonoverlapping(packet.as_ptr(), self.slot_payload(slot), packet.len());
        }
        meta.len.store(packet.len() as u32, Ordering::Release);
        meta.write_index.store(widx, Ordering::Release);
        meta.seq.fetch_add(1, Ordering::AcqRel); // -> even

        hdr.copies.fetch_add(1, Ordering::Relaxed);
        hdr.write_index.store(widx + 1, Ordering::Release);
        Ok(())
    }

    /// Pop the next packet (relative to this handle's read cursor) into `out`,
    /// returning its byte length, or [`BusError::Empty`].
    ///
    /// If the producer has lapped this consumer, advances the cursor to the
    /// oldest still-available frame and returns [`BusError::Lagged`] so the
    /// caller knows it missed frames; the next `pop` then succeeds.
    #[cfg(unix)]
    pub fn pop(&self, out: &mut [u8]) -> Result<usize, BusError> {
        let hdr = self.header();
        let cursor = self.read_cursor.load(Ordering::Relaxed);
        let widx = hdr.write_index.load(Ordering::Acquire);

        if cursor >= widx {
            return Err(BusError::Empty);
        }

        // Lag check: if the producer has advanced more than `capacity` past our
        // cursor, the slot we want has been overwritten. Skip to the oldest
        // available frame.
        let oldest = widx.saturating_sub(self.capacity as u64);
        if cursor < oldest {
            self.read_cursor.store(oldest, Ordering::Relaxed);
            hdr.lag_events.fetch_add(1, Ordering::Relaxed);
            return Err(BusError::Lagged);
        }

        let slot = (cursor as usize) % self.capacity;
        let meta = self.slot_meta(slot);

        // Slot-local seqlock read with bounded retry.
        for _ in 0..1024 {
            let s0 = meta.seq.load(Ordering::Acquire);
            if s0 & 1 == 1 {
                core::hint::spin_loop();
                continue; // write in progress
            }
            let occupant = meta.write_index.load(Ordering::Acquire);
            if occupant != cursor {
                // Slot was recycled out from under us between the widx check and
                // here: we lagged. Resync and report.
                let widx2 = hdr.write_index.load(Ordering::Acquire);
                let oldest2 = widx2.saturating_sub(self.capacity as u64);
                self.read_cursor
                    .store(oldest2.max(cursor + 1).min(widx2), Ordering::Relaxed);
                hdr.lag_events.fetch_add(1, Ordering::Relaxed);
                return Err(BusError::Lagged);
            }
            let len = meta.len.load(Ordering::Acquire) as usize;
            if len > out.len() {
                if meta.seq.load(Ordering::Acquire) == s0 {
                    return Err(BusError::OutputTooSmall);
                }
                continue;
            }
            unsafe {
                core::ptr::copy_nonoverlapping(self.slot_payload(slot), out.as_mut_ptr(), len);
            }
            // Acquire fence pinning the payload copy above before the validating
            // seq re-read below (see read_latest for the full rationale): without
            // it, a weakly-ordered target could sink the data reads past the check
            // and return a torn frame.
            core::sync::atomic::fence(Ordering::Acquire);
            if meta.seq.load(Ordering::Acquire) == s0 {
                self.read_cursor.store(cursor + 1, Ordering::Relaxed);
                hdr.copies.fetch_add(1, Ordering::Relaxed);
                return Ok(len);
            }
            core::hint::spin_loop();
        }
        // Producer kept the slot churning long enough that we never got a stable
        // read; treat as lag.
        hdr.lag_events.fetch_add(1, Ordering::Relaxed);
        Err(BusError::Lagged)
    }

    /// Snapshot the instrumentation counters.
    #[cfg(unix)]
    pub fn stats(&self) -> BusStats {
        let hdr = self.header();
        BusStats {
            copies: hdr.copies.load(Ordering::Relaxed),
            drops: hdr.drops.load(Ordering::Relaxed),
            lag_events: hdr.lag_events.load(Ordering::Relaxed),
            recoveries: hdr.recoveries.load(Ordering::Relaxed),
        }
    }

    /// Number of slots.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Unlink the backing shm object (call once, from the creator, at shutdown).
    #[cfg(unix)]
    pub fn unlink(&self) {
        self.seg.unlink();
    }

    // --- non-unix stubs ---

    #[cfg(not(unix))]
    pub fn open(_cfg: &BusConfig) -> Result<Self, BusError> {
        Err(BusError::Shm(
            "tenso-bus requires a unix shared-memory backend",
        ))
    }
    #[cfg(not(unix))]
    pub fn push(&self, _packet: &[u8]) -> Result<(), BusError> {
        Err(BusError::Shm("unsupported platform"))
    }
    #[cfg(not(unix))]
    pub fn pop(&self, _out: &mut [u8]) -> Result<usize, BusError> {
        Err(BusError::Shm("unsupported platform"))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64 as StdAtomicU64, Ordering as O};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Build a minimal valid v4 dense f32 packet (header + shape + body), so
    /// `tenso_core::parse_header` accepts what we push. We don't need a fully
    /// round-trippable body for transport tests — only a valid header/shape.
    fn make_packet(payload_marker: u8, n_elems: u32) -> Vec<u8> {
        // v4 header is 10 bytes; shape is ndim * u32; body is n_elems * 4 (f32).
        let ndim: u8 = 1;
        let header_base = tenso_core::HEADER_BASE_V4;
        let shape_bytes = ndim as usize * 4;
        let body_bytes = n_elems as usize * 4;
        let mut buf = vec![0u8; header_base + shape_bytes + body_bytes];
        // flags=0, dtype f32=1, ndim=1
        tenso_core::write_v4_header(&mut buf, 0, 1, ndim);
        // shape[0] = n_elems (LE u32)
        buf[header_base..header_base + 4].copy_from_slice(&n_elems.to_le_bytes());
        // mark the body so we can verify content survived transport
        for b in &mut buf[header_base + shape_bytes..] {
            *b = payload_marker;
        }
        // sanity: must parse
        assert!(
            tenso_core::parse_header(&buf).is_ok(),
            "test packet must parse"
        );
        buf
    }

    fn unique_name(tag: &str) -> String {
        // POSIX shm names are capped at 31 chars on macOS (PSHMNAMLEN), incl.
        // the leading '/'. We fold the tag + pid + a monotonic counter + a
        // process-start nanos seed into a short hex token so collisions across
        // concurrent tests / re-runs are astronomically unlikely while staying
        // comfortably under the limit. `tag` is ignored for length but mixed in.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let pid = std::process::id() as u64;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let ctr = COUNTER.fetch_add(1, Ordering::Relaxed);
        // 64-bit FNV-1a over the entropy sources + tag bytes.
        let mut h: u64 = 0xcbf29ce484222325;
        for b in tag
            .as_bytes()
            .iter()
            .chain(&pid.to_le_bytes())
            .chain(&nanos.to_le_bytes())
            .chain(&ctr.to_le_bytes())
        {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        // e.g. "/tenso_t_0a1b2c3d4e5f6071" => 1 + 8 + 16 = 25 chars (< 31).
        format!("/tenso_t_{:016x}", h)
    }

    // ----- LatestValueBus -----

    #[test]
    fn lvb_publish_read_roundtrip() {
        let name = unique_name("lvb_rt");
        let cfg = BusConfig::new(&name, 4096, true);
        let bus = LatestValueBus::open(&cfg).unwrap();

        let mut out = vec![0u8; 8192];
        // empty before first publish
        assert!(matches!(bus.read_latest(&mut out), Err(BusError::Empty)));

        let p1 = make_packet(0xAB, 16);
        bus.publish(&p1).unwrap();
        let n = bus.read_latest(&mut out).unwrap();
        assert_eq!(n, p1.len());
        assert_eq!(&out[..n], &p1[..]);

        // newest wins
        let p2 = make_packet(0xCD, 32);
        bus.publish(&p2).unwrap();
        let n = bus.read_latest(&mut out).unwrap();
        assert_eq!(n, p2.len());
        assert_eq!(&out[..n], &p2[..]);

        bus.unlink();
    }

    #[test]
    fn lvb_output_too_small() {
        let name = unique_name("lvb_small");
        let cfg = BusConfig::new(&name, 4096, true);
        let bus = LatestValueBus::open(&cfg).unwrap();
        let p = make_packet(0x11, 64); // header+shape+256 body
        bus.publish(&p).unwrap();
        let mut tiny = [0u8; 8];
        assert!(matches!(
            bus.read_latest(&mut tiny),
            Err(BusError::OutputTooSmall)
        ));
        bus.unlink();
    }

    #[test]
    fn lvb_rejects_malformed_packet() {
        let name = unique_name("lvb_bad");
        let cfg = BusConfig::new(&name, 256, true);
        let bus = LatestValueBus::open(&cfg).unwrap();
        let bad = [0u8; 12]; // bad magic
        assert!(matches!(bus.publish(&bad), Err(BusError::Core(_))));
        bus.unlink();
    }

    #[test]
    fn lvb_packet_too_large() {
        let name = unique_name("lvb_big");
        let cfg = BusConfig::new(&name, 64, true); // 64-byte stride
        let bus = LatestValueBus::open(&cfg).unwrap();
        let p = make_packet(0x22, 256); // ~1KB > 64
        assert!(matches!(bus.publish(&p), Err(BusError::PacketTooLarge)));
        bus.unlink();
    }

    #[test]
    fn lvb_concurrent_seqlock_no_tearing() {
        // Single writer, many readers; readers must never see a torn frame.
        // We encode the marker into every body byte, so a torn read (mixing two
        // publishes) would show two distinct marker values.
        let name = unique_name("lvb_seq");
        let cfg = BusConfig::new(&name, 8192, true);
        let writer = Arc::new(LatestValueBus::open(&cfg).unwrap());

        let stop = Arc::new(AtomicBool::new(false));
        let n_elems = 200u32; // 800-byte body
        let mut readers = Vec::new();
        for _ in 0..4 {
            let rbus = LatestValueBus::open(&BusConfig::new(&name, 8192, false)).unwrap();
            let stop_c = stop.clone();
            readers.push(std::thread::spawn(move || {
                let mut out = vec![0u8; 8192];
                let mut reads = 0u64;
                while !stop_c.load(O::Relaxed) {
                    match rbus.read_latest(&mut out) {
                        Ok(n) => {
                            // body starts after header(10) + shape(4)
                            let body = &out[14..n];
                            if !body.is_empty() {
                                let first = body[0];
                                assert!(
                                    body.iter().all(|&b| b == first),
                                    "torn read: body not uniform"
                                );
                            }
                            reads += 1;
                        }
                        Err(BusError::Empty) => {}
                        // A reader can legitimately fall behind a publish storm
                        // and get Lagged (the bounded seqlock retry gave up). That
                        // is NOT a torn read, so it must not fail this no-tearing
                        // test -- otherwise the test is flaky on slow/contended CI
                        // (e.g. the macOS runner). The Ok branch still asserts no
                        // tearing on every frame that is read successfully.
                        Err(BusError::Lagged) => {}
                        Err(e) => panic!("reader error: {:?}", e),
                    }
                }
                reads
            }));
        }

        let writer_c = writer.clone();
        let stop_w = stop.clone();
        let wt = std::thread::spawn(move || {
            let mut marker = 1u8;
            let mut count = 0u64;
            while !stop_w.load(O::Relaxed) {
                let p = make_packet(marker, n_elems);
                writer_c.publish(&p).unwrap();
                marker = marker.wrapping_add(1).max(1);
                count += 1;
            }
            count
        });

        std::thread::sleep(Duration::from_millis(200));
        stop.store(true, O::Relaxed);

        let published = wt.join().unwrap();
        let mut total_reads = 0;
        for r in readers {
            total_reads += r.join().unwrap();
        }
        assert!(published > 0, "writer made no progress");
        assert!(total_reads > 0, "readers made no progress");

        writer.unlink();
    }

    #[test]
    fn lvb_single_host_latency() {
        // Smoke latency test (runs on macOS): publish->read round trip should be
        // sub-millisecond in the common case. We only assert it completes and
        // report the median via stats; no hard threshold to avoid CI flakiness.
        let name = unique_name("lvb_lat");
        let cfg = BusConfig::new(&name, 4096, true);
        let bus = LatestValueBus::open(&cfg).unwrap();
        let p = make_packet(0x77, 64);
        let mut out = vec![0u8; 4096];

        let iters = 10_000;
        let start = Instant::now();
        for _ in 0..iters {
            bus.publish(&p).unwrap();
            let n = bus.read_latest(&mut out).unwrap();
            assert_eq!(n, p.len());
        }
        let elapsed = start.elapsed();
        let per = elapsed / iters;
        // Generous ceiling: 1ms per round trip even on a slow CI box.
        assert!(
            per < Duration::from_millis(1),
            "latency too high: {:?}/op",
            per
        );
        bus.unlink();
    }

    // ----- RingBus -----

    #[test]
    fn ring_push_pop_fifo() {
        let name = unique_name("ring_fifo");
        let cfg = BusConfig::new(&name, 8, true);
        let prod = RingBus::open_with(&cfg, 4096, OverflowPolicy::Error).unwrap();
        let cons = RingBus::open_with(
            &BusConfig::new(&name, 8, false),
            4096,
            OverflowPolicy::Error,
        )
        .unwrap();

        let mut out = vec![0u8; 4096];
        assert!(matches!(cons.pop(&mut out), Err(BusError::Empty)));

        for i in 0..5u8 {
            let p = make_packet(i + 1, 8);
            prod.push(&p).unwrap();
        }
        for i in 0..5u8 {
            let n = cons.pop(&mut out).unwrap();
            let expected = make_packet(i + 1, 8);
            assert_eq!(&out[..n], &expected[..], "FIFO order violated at {}", i);
        }
        assert!(matches!(cons.pop(&mut out), Err(BusError::Empty)));
        prod.unlink();
    }

    #[test]
    fn ring_drop_oldest_overwrites() {
        let name = unique_name("ring_drop");
        let cfg = BusConfig::new(&name, 4, true);
        let prod = RingBus::open_with(&cfg, 1024, OverflowPolicy::DropOldest).unwrap();
        let cons = RingBus::open_with(
            &BusConfig::new(&name, 4, false),
            1024,
            OverflowPolicy::DropOldest,
        )
        .unwrap();

        // Push 8 into a 4-slot ring; consumer joined at head so it sees frames
        // 0..8 but lags: oldest available is 4..8.
        for i in 0..8u8 {
            prod.push(&make_packet(i + 1, 4)).unwrap();
        }

        let mut out = vec![0u8; 1024];
        // First pop should report Lagged (cursor 0 < oldest 4), then resync.
        let r = cons.pop(&mut out);
        assert!(
            matches!(r, Err(BusError::Lagged)),
            "expected lag, got {:?}",
            r
        );

        // Now we should read frames 5,6,7,8 (markers 5..8).
        let mut got = Vec::new();
        loop {
            match cons.pop(&mut out) {
                Ok(n) => got.push(out[14]), // body marker byte
                Err(BusError::Empty) => break,
                Err(BusError::Lagged) => continue,
                Err(e) => panic!("unexpected {:?}", e),
            }
        }
        assert_eq!(got, vec![5, 6, 7, 8], "drop-oldest tail mismatch");
        assert!(prod.stats().drops >= 4, "expected drops recorded");
        prod.unlink();
    }

    #[test]
    fn ring_error_policy_when_full() {
        let name = unique_name("ring_err");
        let cfg = BusConfig::new(&name, 2, true);
        let prod = RingBus::open_with(&cfg, 256, OverflowPolicy::Error).unwrap();
        // Producer's own cursor never advances (it doesn't consume), so after
        // filling capacity the next push that would stomp an unread slot errors.
        prod.push(&make_packet(1, 2)).unwrap();
        prod.push(&make_packet(2, 2)).unwrap();
        let r = prod.push(&make_packet(3, 2));
        assert!(
            matches!(r, Err(BusError::Full)),
            "expected Full, got {:?}",
            r
        );
        prod.unlink();
    }

    #[test]
    fn ring_spmc_concurrent() {
        // One producer, several consumers; each consumer reads an independent
        // stream. With DropOldest no data race / torn read may occur.
        let name = unique_name("ring_spmc");
        let cfg = BusConfig::new(&name, 256, true);
        let prod = Arc::new(RingBus::open_with(&cfg, 256, OverflowPolicy::DropOldest).unwrap());

        let stop = Arc::new(AtomicBool::new(false));
        let total = Arc::new(StdAtomicU64::new(0));
        let mut handles = Vec::new();
        for _ in 0..3 {
            let cons = RingBus::open_with(
                &BusConfig::new(&name, 256, false),
                256,
                OverflowPolicy::DropOldest,
            )
            .unwrap();
            let stop_c = stop.clone();
            let total_c = total.clone();
            handles.push(std::thread::spawn(move || {
                let mut out = vec![0u8; 256];
                while !stop_c.load(O::Relaxed) {
                    match cons.pop(&mut out) {
                        Ok(n) => {
                            let body = &out[14..n];
                            if !body.is_empty() {
                                let f = body[0];
                                assert!(body.iter().all(|&b| b == f), "torn ring read");
                            }
                            total_c.fetch_add(1, O::Relaxed);
                        }
                        Err(BusError::Empty) | Err(BusError::Lagged) => {
                            std::thread::yield_now();
                        }
                        Err(e) => panic!("consumer error {:?}", e),
                    }
                }
            }));
        }

        let prod_c = prod.clone();
        let stop_p = stop.clone();
        let pt = std::thread::spawn(move || {
            let mut marker = 1u8;
            let mut pushed = 0u64;
            while !stop_p.load(O::Relaxed) {
                prod_c.push(&make_packet(marker, 8)).unwrap();
                marker = marker.wrapping_add(1).max(1);
                pushed += 1;
            }
            pushed
        });

        std::thread::sleep(Duration::from_millis(200));
        stop.store(true, O::Relaxed);
        let pushed = pt.join().unwrap();
        for h in handles {
            h.join().unwrap();
        }
        assert!(pushed > 0);
        assert!(total.load(O::Relaxed) > 0, "consumers read nothing");
        prod.unlink();
    }

    #[test]
    fn ring_single_host_latency() {
        let name = unique_name("ring_lat");
        let cfg = BusConfig::new(&name, 64, true);
        let prod = RingBus::open_with(&cfg, 512, OverflowPolicy::DropOldest).unwrap();
        let cons = RingBus::open_with(
            &BusConfig::new(&name, 64, false),
            512,
            OverflowPolicy::DropOldest,
        )
        .unwrap();
        let p = make_packet(0x99, 16);
        let mut out = vec![0u8; 512];
        let iters = 10_000;
        let start = Instant::now();
        for _ in 0..iters {
            prod.push(&p).unwrap();
            let _ = cons.pop(&mut out);
        }
        let per = start.elapsed() / iters;
        assert!(
            per < Duration::from_millis(1),
            "ring latency too high: {:?}",
            per
        );
        prod.unlink();
    }

    // ----- shm_mutex (basic, all unix) -----

    #[test]
    fn shm_mutex_init_lock_unlock() {
        let mut storage = vec![0u8; shm_mutex::MUTEX_SIZE];
        unsafe {
            shm_mutex::init_mutex(storage.as_mut_ptr()).unwrap();
            let recovered =
                shm_mutex::lock_mutex(storage.as_mut_ptr(), Duration::from_secs(1)).unwrap();
            assert!(!recovered, "fresh mutex should not report recovery");
            shm_mutex::unlock_mutex(storage.as_mut_ptr()).unwrap();
            shm_mutex::destroy_mutex(storage.as_mut_ptr()).unwrap();
        }
    }
}

// =============================================================================
// Linux-only: robust mutex crash recovery (EOWNERDEAD) via a forked child.
// =============================================================================

#[cfg(all(test, target_os = "linux"))]
mod linux_robust_tests {
    use super::*;
    use std::time::Duration;

    /// A child process locks a shared, robust mutex and then `_exit`s while
    /// holding it (simulating a crashed writer). The parent must then acquire
    /// the mutex and observe the `recovered == true` signal (EOWNERDEAD ->
    /// pthread_mutex_consistent).
    #[test]
    fn robust_mutex_recovers_from_dead_owner() {
        // Place the mutex in an anonymous MAP_SHARED region so the fork child
        // shares it.
        let len = shm_mutex::MUTEX_SIZE;
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap failed");
        let base = ptr as *mut u8;

        unsafe { shm_mutex::init_mutex(base).unwrap() };

        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");

        if pid == 0 {
            // Child: lock and die while holding it.
            unsafe {
                let _ = shm_mutex::lock_mutex(base, Duration::from_secs(2));
                // Exit WITHOUT unlocking -> robust mutex marks EOWNERDEAD.
                libc::_exit(0);
            }
        }

        // Parent: reap the child, then try to acquire.
        let mut status = 0i32;
        unsafe { libc::waitpid(pid, &mut status, 0) };

        let recovered = unsafe { shm_mutex::lock_mutex(base, Duration::from_secs(5)) }
            .expect("parent should acquire the orphaned robust mutex");
        assert!(
            recovered,
            "expected EOWNERDEAD recovery (recovered == true) from dead owner"
        );
        unsafe {
            // After consistency, normal unlock.
            shm_mutex::unlock_mutex(base).unwrap();
            shm_mutex::destroy_mutex(base).unwrap();
            libc::munmap(ptr, len);
        }
    }
}
