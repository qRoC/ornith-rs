//! Thin Metal layer: device, the compiled kernel library, a pipeline cache, a
//! buffer pool and command encoding.
//!
//! This is the one place that talks to the OS. It mirrors
//! `mlx/backend/metal/device.{h,cpp}` in shape but not in size: mlx also
//! carries a JIT compiler, residency sets, fences and multiple streams, none of
//! which a single-stream inference loop needs.
//!
//! Ordering model, in two parts.
//!
//! *Inside* an encoder, dispatches are concurrent and ordered by the barriers
//! [`Hazards`] works out -- mlx's rule, and the reason a dispatch that depends
//! on nothing before it does not have to wait for it. `MTLDispatchTypeSerial`,
//! which barriers between every pair, is still one `ORNITH_SERIAL=1` away.
//!
//! *Across* encoders the queue orders the command buffers, and mlx relies on
//! exactly that -- its `commit()` starts the next one with no synchronisation
//! of its own. What it does add is a fence per encoder
//! (`CommandEncoder::end_encoding`), waited on by whichever later encoder
//! reads that encoder's outputs, and it needs them because its encoders are
//! concurrent. So are these, now, which is what makes the fence here load
//! bearing rather than belt and braces: every encoder signals one before it
//! ends and the next waits on it, ordering the boundary whatever the
//! dependency across it. There are ~14 boundaries in a decode step, so it
//! costs nothing measurable.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::rc::Rc;

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLBarrierScope, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice,
    MTLDispatchType, MTLFence, MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::safetensors::Mapping;

/// The metallib built by `build.rs` from `kernels/`.
static METALLIB: &[u8] = include_bytes!(env!("ORNITH_METALLIB"));

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ComputeEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type MTLBuf = Retained<ProtocolObject<dyn MTLBuffer>>;
type Fence = Retained<ProtocolObject<dyn MTLFence>>;

/// A Metal function constant value.
#[derive(Clone, Copy, Debug)]
pub enum FConst {
    Bool(bool),
    Int(i32),
}

// --- buffers ----------------------------------------------------------------

/// A device buffer. Dropping it hands the allocation back to the pool it came
/// from, so a steady-state decode step allocates nothing.
pub struct Buffer {
    raw: MTLBuf,
    pub len: usize,
    /// Which allocation this is. A fresh number every time a buffer is handed
    /// out, including when the pool hands the same Metal object back, so two
    /// unrelated arrays that land on one address can be told apart. The
    /// hazard tracking does not use it -- the barrier is needed either way,
    /// since the bytes really are shared -- but the attribution does, to say
    /// whether a barrier was forced by data or by the allocator.
    serial: u64,
    pool: Option<Rc<RefCell<Pool>>>,
}

impl Buffer {
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// Identity for hazard tracking: the Metal object's address.
    fn id(&self) -> usize {
        buf_id(&self.raw)
    }

    /// Host pointer. Valid because every allocation is `StorageModeShared`:
    /// on Apple silicon CPU and GPU address the same physical pages.
    pub fn contents(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }

    /// # Safety
    /// The caller must not read while a dispatch that writes this buffer is
    /// still in flight; call [`Context::sync`] first.
    pub unsafe fn as_slice<T: Copy>(&self, offset_bytes: usize, count: usize) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.contents().add(offset_bytes) as *const T, count)
        }
    }

    /// # Safety
    /// Same as [`Buffer::as_slice`], plus no concurrent GPU reads.
    pub unsafe fn as_mut_slice<T: Copy>(&self, offset_bytes: usize, count: usize) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.contents().add(offset_bytes) as *mut T, count)
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            pool.borrow_mut().put(self.len, self.raw.clone());
        }
    }
}

/// Size-bucketed free list. mlx's allocator does the same thing for the same
/// reason: `newBufferWithLength` is a kernel call, and a decode step asks for a
/// few hundred short-lived intermediates.
#[derive(Default)]
struct Pool {
    free: HashMap<usize, Vec<MTLBuf>>,
    live_bytes: usize,
    peak_bytes: usize,
}

impl Pool {
    fn take(&mut self, len: usize) -> Option<MTLBuf> {
        let v = self.free.get_mut(&len)?;
        v.pop()
    }

    /// `take`, but skipping any buffer the current barrier epoch has already
    /// touched. Handing one of those out puts an unrelated array on an address
    /// a live dispatch is using, which the tracking can only read as a
    /// dependency. This is the default; `ORNITH_RECYCLE=1` takes the other
    /// branch, which is what mlx does.
    fn take_untouched(&mut self, len: usize, touched: &dyn Fn(BufId) -> bool) -> Option<MTLBuf> {
        let v = self.free.get_mut(&len)?;
        let i = v.iter().rposition(|b| !touched(buf_id(b)))?;
        Some(v.remove(i))
    }

    fn put(&mut self, len: usize, buf: MTLBuf) {
        self.free.entry(len).or_default().push(buf);
    }
}

/// A Metal buffer's identity: the address of the objc object.
fn buf_id(raw: &MTLBuf) -> usize {
    &**raw as *const ProtocolObject<dyn MTLBuffer> as *const () as usize
}

/// Round an allocation up so similar sizes share a bucket.
fn bucket(len: usize) -> usize {
    const MIN: usize = 256;
    if len <= MIN {
        return MIN;
    }
    // Round to the next multiple of 1/8th of the enclosing power of two: a
    // ~12% worst-case overhead, few distinct buckets.
    let pow = usize::BITS - (len - 1).leading_zeros();
    let step = (1usize << pow) / 8;
    len.div_ceil(step) * step
}

// --- profiling ---------------------------------------------------------------

/// Per-run counters, off unless `ORNITH_PROF` is set.
///
/// The question they answer is where a decode step's wall time goes: if the
/// summed GPU busy time is well under it, the step is bound by encoding or by
/// the round trip to the host, not by the kernels.
#[derive(Default)]
pub struct Prof {
    pub on: bool,
    /// Barriers actually inserted, i.e. how often the tracking found a
    /// dependency. Against the dispatch count it says what the marking bought.
    pub barriers: usize,
    /// Of those, the ones where no cause was a dependency the data has: every
    /// buffer that forced it was either a range of an allocation the earlier
    /// dispatch did not touch, or a different allocation that happens to sit
    /// at the same address because the pool recycled it.
    pub barriers_spurious: usize,
    /// `(kernel, culprit, kind, why, spurious) -> count`, keyed on the first
    /// hazard the dispatch hit: which pair of dispatches forced a barrier and
    /// what kind of sharing was between them.
    barrier_causes: HashMap<(Rc<str>, Rc<str>, Kind, Why, bool), usize>,
    /// How many dispatches each closed epoch held. A barrier orders everything
    /// before it against everything after, so an epoch is the unit that can
    /// actually overlap -- and its size, not the barrier count, is what says
    /// how much of the step runs at once.
    epoch_sizes: HashMap<usize, usize>,
    /// Dispatches in the epoch now open.
    epoch_dispatches: usize,
    /// One decode step's dispatches in order, with the barrier before each --
    /// enough to read the dependency chain rather than infer it from
    /// histograms. `ORNITH_TRACE=<path>` turns it on; it stops after
    /// [`TRACE_BUDGET`] dispatches, which is about one token.
    trace: Option<Vec<String>>,
    pub commits: usize,
    pub dispatches: usize,
    /// Round trips to the host: every one drains the queue and leaves the GPU
    /// with nothing to do until the CPU queues the next thing.
    pub syncs: usize,
    /// Summed `GPUEndTime - GPUStartTime` over committed command buffers.
    pub gpu_s: f64,
    /// Time blocked in `waitUntilCompleted`.
    pub wait_s: f64,
    pub by_kernel: HashMap<String, usize>,
    /// GPU seconds per kernel. Only filled when a command buffer holds a
    /// single dispatch (`ORNITH_COMMIT_EVERY=1`), since a buffer's timestamps
    /// cover everything in it.
    pub gpu_by_kernel: HashMap<String, f64>,
    /// GPU seconds in buffers that held more than one dispatch.
    pub gpu_mixed: f64,
    /// Kernel names dispatched into the buffer now being built.
    cb_names: Vec<String>,
    /// Committed but not yet harvested; their timestamps are only readable
    /// once the GPU is done with them. Carries the one kernel name when the
    /// buffer holds exactly one dispatch.
    pending: Vec<(CommandBuffer, Option<String>)>,
}

/// Lines `ORNITH_TRACE` records before it stops. One decode step is ~1669
/// dispatches plus its barriers, so this covers a step and a little over; the
/// interest is in the shape of one step, not all of them.
const TRACE_BUDGET: usize = 4096;

// --- hazards ----------------------------------------------------------------

/// How the encoder orders one dispatch against the next.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BarrierMode {
    /// mlx's rule: a concurrent encoder, and a barrier only where a dispatch
    /// touches a buffer an earlier one in the same epoch used the other way.
    Selective,
    /// `MTLDispatchTypeSerial`, which barriers between every pair. What this
    /// port did before the tracking existed; kept as the control to measure
    /// and to fall back on (`ORNITH_SERIAL=1`).
    Serial,
    /// Concurrent with no barriers at all: wrong, and only useful for putting
    /// a bound on what the barriers cost (`ORNITH_NO_BARRIER=1`).
    None,
}

/// Which of the three hazards forced a barrier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    /// Read after write.
    Raw,
    /// Write after read.
    War,
    /// Write after write.
    Waw,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Raw => "RAW",
            Kind::War => "WAR",
            Kind::Waw => "WAW",
        }
    }
}

/// Whether the hazard behind a barrier is a dependency the data has, or an
/// artefact of tracking whole buffers by address -- which is what mlx does,
/// and so what this port does. Only the attribution looks at this; the barrier
/// goes in either way, because in both cases the bytes really are shared.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Why {
    /// Overlapping bytes of one allocation: a dependency the data really has.
    Real,
    /// One allocation, two ranges that provably do not overlap -- two halves
    /// of a fused projection, two experts of a stacked weight, two rows of a
    /// cache. Ordering them is not something the data asks for.
    Disjoint,
    /// Two different allocations that share an address, because the pool
    /// handed the freed one straight back out. Nothing flows between them.
    Recycled,
}

impl Why {
    pub fn as_str(self) -> &'static str {
        match self {
            Why::Real => "real",
            Why::Disjoint => "disjoint",
            Why::Recycled => "recycled",
        }
    }
}

/// A buffer's identity for hazard tracking: the address of the Metal object.
/// Two views of one allocation share it, and so does a buffer the pool has
/// handed out again -- both are hazards, and both want the barrier.
type BufId = usize;

/// Port of the bookkeeping in `mlx`'s `CommandEncoder`
/// (`set_input_array` / `register_output_array` / `maybeInsertBarrier`).
///
/// A barrier splits the encoder into epochs. Within one, dispatches may run in
/// any order and overlap, so a dispatch needs a barrier before it if it reads
/// what an earlier dispatch in this epoch wrote (RAW), writes what one read
/// (WAR), or writes what one wrote (WAW). `prev_*` is what the epoch has seen,
/// `next_*` what the dispatch now being encoded binds.
#[derive(Default)]
struct Hazards {
    prev_in: HashSet<BufId>,
    prev_out: HashSet<BufId>,
    next_in: HashSet<BufId>,
    next_out: HashSet<BufId>,
    needs_barrier: bool,
    /// The rest is attribution, maintained only when profiling: who last read
    /// and last wrote each buffer in this epoch, and what the dispatch now
    /// being encoded collided with. `*_by` mirror the sets above key for key.
    prev_in_by: HashMap<BufId, Touch>,
    prev_out_by: HashMap<BufId, Touch>,
    next_in_by: HashMap<BufId, Touch>,
    next_out_by: HashMap<BufId, Touch>,
    causes: Vec<(Kind, Why, Rc<str>)>,
}

/// One epoch-local touch of a buffer, for attribution.
#[derive(Clone)]
struct Touch {
    serial: u64,
    start: usize,
    end: usize,
    kernel: Rc<str>,
}

impl Hazards {
    fn clear(&mut self) {
        self.prev_in.clear();
        self.prev_out.clear();
        self.next_in.clear();
        self.next_out.clear();
        self.needs_barrier = false;
        self.prev_in_by.clear();
        self.prev_out_by.clear();
        self.next_in_by.clear();
        self.next_out_by.clear();
        self.causes.clear();
    }

    /// Has this epoch bound `id` at all? What `ORNITH_NO_RECYCLE=1` asks the
    /// pool before handing a buffer back out.
    fn touched(&self, id: BufId) -> bool {
        self.prev_in.contains(&id)
            || self.prev_out.contains(&id)
            || self.next_in.contains(&id)
            || self.next_out.contains(&id)
    }
}

/// Why a barrier between `t` and `b` was inserted: the two either share bytes,
/// or only an address.
fn why(t: &Touch, b: &Bound) -> Why {
    if t.serial != b.serial {
        Why::Recycled
    } else if t.end <= b.start || b.end <= t.start {
        Why::Disjoint
    } else {
        Why::Real
    }
}

// --- context ----------------------------------------------------------------

pub struct Context {
    device: Device,
    queue: Queue,
    /// e.g. "applegpu_g16s". mlx keys several dispatch decisions off this.
    pub arch: String,
    /// The two digits in the architecture name: 16 for an M4, 17 for an M5.
    pub arch_gen: i32,
    library: Library,
    pipelines: RefCell<HashMap<String, Pipeline>>,
    pool: Rc<RefCell<Pool>>,
    cb: RefCell<Option<CommandBuffer>>,
    enc: RefCell<Option<ComputeEncoder>>,
    /// Buffers kept alive until the current command buffer completes.
    retained: RefCell<Vec<Rc<Buffer>>>,
    dispatches: RefCell<usize>,
    /// Signalled at the end of each encoder and waited on by the next, so a
    /// command-buffer boundary carries the dependencies across it. Rotated so
    /// a fence is never re-signalled while an earlier wait on it is still
    /// outstanding; mlx gives every encoder its own for the same reason.
    fences: Vec<Fence>,
    next_fence: RefCell<usize>,
    /// The fence the previous encoder signalled, if there was one.
    last_fence: RefCell<Option<Fence>>,
    /// The most recently committed command buffer, kept so [`Context::sync`]
    /// has something to wait on when `flush` already submitted everything.
    last_committed: RefCell<Option<CommandBuffer>>,
    /// Commit after this many dispatches, trading two costs against each
    /// other: a command buffer costs ~60 us of GPU time, and a large one
    /// leaves the GPU idle while the CPU fills it. Measured on this machine at
    /// 256 greedy tokens -- 32: 68 tok/s, 128: 82, 512: 79, where 512 is the
    /// end where the GPU starts waiting. `ORNITH_COMMIT_EVERY` overrides it.
    pub max_dispatches_per_commit: usize,
    prof: RefCell<Prof>,
    barriers: BarrierMode,
    hazards: RefCell<Hazards>,
    /// The kernel now being encoded, so a failed hazard check can name it.
    last_kernel: RefCell<String>,
    /// The same name, shareable, for the barrier attribution to keep in its
    /// tables. Only maintained when profiling.
    cur_kernel: RefCell<Rc<str>>,
    /// Hands out [`Buffer::serial`].
    next_serial: Cell<u64>,
    /// Whether the pool may hand back a buffer this barrier epoch has already
    /// touched. mlx allows it; this port does not, because handing one out
    /// puts an unrelated array on an address a live dispatch is using and the
    /// tracking can only read that as a dependency -- 30 barriers a token
    /// that no data needs. Declining costs nothing: the buffer is reused in
    /// the next epoch instead, so the pool's peak is the same 76.8 MB either
    /// way, and it measured -0.8% of GPU busy. `ORNITH_RECYCLE=1` puts mlx's
    /// behaviour back as the control.
    recycle_in_epoch: bool,
    /// Buffers some dispatch has written, or the host filled at allocation.
    ///
    /// Reading one that is not in here means an op bound a fresh pooled buffer
    /// as an input -- which is how a missed `out` shows up: the op that wrote
    /// the buffer called it an input, so the tracking recorded a read where
    /// there was a write and will not order the next reader against it. That
    /// is a race, and a race is not something a golden test can be trusted to
    /// catch, so it is an assert instead.
    written: RefCell<HashSet<BufId>>,
}

impl Context {
    pub fn new() -> Result<Context, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("newCommandQueue failed")?;
        let data = DispatchData::from_bytes(METALLIB);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|e| format!("newLibraryWithData: {e}"))?;
        let arch = device.architecture().name().to_string();
        // Same parse as mlx's Device::Device: the two digits before the
        // trailing size letter.
        let b = arch.as_bytes();
        let arch_gen = if b.len() >= 3 {
            let tens = (b[b.len() - 3] as char).to_digit(10).unwrap_or(0) as i32;
            let ones = (b[b.len() - 2] as char).to_digit(10).unwrap_or(0) as i32;
            tens * 10 + ones
        } else {
            0
        };
        let fences: Vec<Fence> =
            (0..8).map(|_| device.newFence().expect("newFence failed")).collect();
        Ok(Context {
            device,
            queue,
            arch,
            arch_gen,
            library,
            pipelines: RefCell::new(HashMap::new()),
            pool: Rc::new(RefCell::new(Pool::default())),
            cb: RefCell::new(None),
            enc: RefCell::new(None),
            retained: RefCell::new(Vec::new()),
            dispatches: RefCell::new(0),
            fences,
            next_fence: RefCell::new(0),
            last_fence: RefCell::new(None),
            last_committed: RefCell::new(None),
            max_dispatches_per_commit: std::env::var("ORNITH_COMMIT_EVERY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(128),
            prof: RefCell::new(Prof {
                on: std::env::var_os("ORNITH_PROF").is_some(),
                ..Prof::default()
            }),
            barriers: if std::env::var_os("ORNITH_NO_BARRIER").is_some() {
                BarrierMode::None
            } else if std::env::var_os("ORNITH_SERIAL").is_some() {
                BarrierMode::Serial
            } else {
                BarrierMode::Selective
            },
            hazards: RefCell::new(Hazards::default()),
            written: RefCell::new(HashSet::new()),
            last_kernel: RefCell::new(String::new()),
            cur_kernel: RefCell::new(Rc::from("")),
            next_serial: Cell::new(1),
            recycle_in_epoch: std::env::var_os("ORNITH_RECYCLE").is_some(),
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// mlx's `metal::is_nax_available()`: the Metal 4 tensor-op kernels need
    /// macOS 26.2+ and GPU gen 17+ (M5). On an M4 (`applegpu_g16s`) this is
    /// false, so none of mlx's `*_nax` kernels can run and this port does not
    /// carry them.
    pub fn nax_available(&self) -> bool {
        let last = self.arch.as_bytes().last().copied().unwrap_or(b'g');
        self.arch_gen >= if last == b'p' { 18 } else { 17 }
    }

    pub fn recommended_working_set(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    // --- allocation ---------------------------------------------------------

    /// The next [`Buffer::serial`].
    fn serial(&self) -> u64 {
        let n = self.next_serial.get();
        self.next_serial.set(n + 1);
        n
    }

    pub fn alloc(&self, nbytes: usize) -> Buffer {
        let len = bucket(nbytes.max(1));
        let recycled = if self.recycle_in_epoch {
            self.pool.borrow_mut().take(len)
        } else {
            let h = self.hazards.borrow();
            self.pool.borrow_mut().take_untouched(len, &|id| h.touched(id))
        };
        if let Some(raw) = recycled {
            self.written.borrow_mut().remove(&buf_id(&raw));
            return Buffer { raw, len, serial: self.serial(), pool: Some(self.pool.clone()) };
        }
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .unwrap_or_else(|| panic!("newBufferWithLength({len}) failed"));
        {
            let mut p = self.pool.borrow_mut();
            p.live_bytes += len;
            p.peak_bytes = p.peak_bytes.max(p.live_bytes);
        }
        Buffer { raw, len, serial: self.serial(), pool: Some(self.pool.clone()) }
    }

    /// Allocate outside the pool, at exactly the requested size. Weights are
    /// long-lived and large, so bucket rounding would only waste memory.
    pub fn alloc_exact(&self, nbytes: usize) -> Buffer {
        let len = nbytes.max(1);
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .unwrap_or_else(|| panic!("newBufferWithLength({len}) failed"));
        self.written.borrow_mut().insert(buf_id(&raw));
        Buffer { raw, len, serial: self.serial(), pool: None }
    }

    /// Wrap a whole mmap'd safetensors shard with no copy. The mapping stays
    /// alive for the process, which is what the returned buffer borrows.
    pub fn wrap_mapping(&self, map: &Mapping) -> Buffer {
        let ptr = std::ptr::NonNull::new(map.as_mut_ptr()).expect("null mapping");
        let raw = unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                map.map_len,
                // Read-only weights; Shared so the GPU reads the page cache
                // pages directly.
                MTLResourceOptions::StorageModeShared,
                // No deallocator: the Mapping's Drop owns the munmap.
                None,
            )
        }
        .expect("newBufferWithBytesNoCopy failed");
        self.written.borrow_mut().insert(buf_id(&raw));
        Buffer { raw, len: map.map_len, serial: self.serial(), pool: None }
    }

    pub fn pool_stats(&self) -> (usize, usize) {
        let p = self.pool.borrow();
        (p.live_bytes, p.peak_bytes)
    }

    // --- pipelines ----------------------------------------------------------

    pub fn pipeline(&self, name: &str) -> Pipeline {
        self.pipeline_with(name, &[])
    }

    pub fn pipeline_with(&self, name: &str, consts: &[(usize, bool)]) -> Pipeline {
        let typed: Vec<(usize, FConst)> = consts.iter().map(|(i, v)| (*i, FConst::Bool(*v))).collect();
        self.pipeline_typed(name, &typed)
    }

    /// Build (or fetch) a pipeline, specialising the kernel's function
    /// constants. mlx does the same in `Device::get_kernel(..., func_consts)`:
    /// e.g. rms_norm's `has_w`, rope's `forward`/`traditional`, the steel
    /// gemm's `align_M`/`align_N`/`align_K` and sdpa's `BLOCKS`.
    pub fn pipeline_typed(&self, name: &str, consts: &[(usize, FConst)]) -> Pipeline {
        let mut key = String::from(name);
        for (i, v) in consts {
            match v {
                FConst::Bool(b) => key.push_str(&format!("#{i}=b{}", *b as u8)),
                FConst::Int(n) => key.push_str(&format!("#{i}=i{n}")),
            }
        }
        if let Some(p) = self.pipelines.borrow().get(&key) {
            return p.clone();
        }
        let ns = NSString::from_str(name);
        let func = if consts.is_empty() {
            self.library
                .newFunctionWithName(&ns)
                .unwrap_or_else(|| panic!("kernel {name:?} is not in the metallib"))
        } else {
            let values = MTLFunctionConstantValues::new();
            for (i, v) in consts {
                match v {
                    FConst::Bool(b) => unsafe {
                        values.setConstantValue_type_atIndex(
                            std::ptr::NonNull::from(b).cast::<c_void>(),
                            MTLDataType::Bool,
                            *i,
                        )
                    },
                    FConst::Int(n) => unsafe {
                        values.setConstantValue_type_atIndex(
                            std::ptr::NonNull::from(n).cast::<c_void>(),
                            MTLDataType::Int,
                            *i,
                        )
                    },
                }
            }
            self.library
                .newFunctionWithName_constantValues_error(&ns, &values)
                .unwrap_or_else(|e| panic!("specialising {name:?}: {e}"))
        };
        let pipe = self
            .device
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap_or_else(|e| panic!("pipeline for {name:?}: {e}"));
        self.pipelines.borrow_mut().insert(key, pipe.clone());
        pipe
    }

    // --- encoding -----------------------------------------------------------

    fn ensure_encoder(&self) -> ComputeEncoder {
        if let Some(e) = self.enc.borrow().as_ref() {
            return e.clone();
        }
        let cb = self.queue.commandBuffer().expect("commandBuffer failed");
        let enc = match self.barriers {
            BarrierMode::Serial => {
                cb.computeCommandEncoder().expect("computeCommandEncoder failed")
            }
            BarrierMode::Selective | BarrierMode::None => cb
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .expect("computeCommandEncoder failed"),
        };
        // A new encoder is a new epoch: the fence below orders it against
        // everything the previous one did, so nothing carries over.
        self.hazards.borrow_mut().clear();
        if let Some(f) = self.last_fence.borrow().as_ref() {
            enc.waitForFence(f);
        }
        *self.cb.borrow_mut() = Some(cb);
        *self.enc.borrow_mut() = Some(enc.clone());
        enc
    }

    /// Signal this encoder's fence and end it. Everything the next encoder
    /// does waits on that fence, so the boundary is ordered.
    fn end_encoder(&self, enc: &ComputeEncoder) {
        {
            let mut p = self.prof.borrow_mut();
            if p.on && p.epoch_dispatches > 0 {
                let n = std::mem::take(&mut p.epoch_dispatches);
                *p.epoch_sizes.entry(n).or_default() += 1;
                if let Some(t) = p.trace.as_mut()
                    && t.len() < TRACE_BUDGET
                {
                    t.push("-- end of encoder".to_string());
                }
            }
        }
        let mut idx = self.next_fence.borrow_mut();
        let fence = self.fences[*idx % self.fences.len()].clone();
        *idx += 1;
        drop(idx);
        enc.updateFence(&fence);
        enc.endEncoding();
        *self.last_fence.borrow_mut() = Some(fence);
    }

    /// Start a dispatch of `kernel`.
    pub fn kernel<'a>(&'a self, name: &str) -> Dispatch<'a> {
        self.kernel_with(name, &[])
    }

    pub fn kernel_with<'a>(&'a self, name: &str, consts: &[(usize, bool)]) -> Dispatch<'a> {
        let pipe = self.pipeline_with(name, consts);
        let enc = self.ensure_encoder();
        enc.setComputePipelineState(&pipe);
        self.prof_name(name);
        self.note_kernel(name);
        return Dispatch::new(self, enc, pipe);
    }

    pub fn kernel_typed<'a>(&'a self, name: &str, consts: &[(usize, FConst)]) -> Dispatch<'a> {
        let pipe = self.pipeline_typed(name, consts);
        let enc = self.ensure_encoder();
        enc.setComputePipelineState(&pipe);
        self.prof_name(name);
        self.note_kernel(name);
        Dispatch::new(self, enc, pipe)
    }

    /// End the current encoder and submit. Does not wait.
    pub fn flush(&self) {
        if let Some(enc) = self.enc.borrow_mut().take() {
            self.end_encoder(&enc);
        }
        if let Some(cb) = self.cb.borrow_mut().take() {
            cb.commit();
            self.prof_commit(&cb);
            *self.dispatches.borrow_mut() = 0;
            // The command buffer retains its resources until completion, so
            // dropping our own references here is safe.
            self.retained.borrow_mut().clear();
            *self.last_committed.borrow_mut() = Some(cb);
        }
    }

    /// Submit and wait for the GPU. Required before reading any buffer.
    ///
    /// "Nothing open" is not the same as "nothing running": a dispatch that
    /// lands exactly on a commit boundary leaves the encoder and the command
    /// buffer empty while the GPU is still working. Waiting has to fall back
    /// to the last buffer `flush` committed, or the host reads a buffer whose
    /// kernel has not run -- which reads as zeros, and is the whole forward
    /// pass when the dispatch that lands there is the embedding gather.
    pub fn sync(&self) {
        if let Some(enc) = self.enc.borrow_mut().take() {
            self.end_encoder(&enc);
        }
        let open = self.cb.borrow_mut().take();
        let waiting_on = match open {
            Some(cb) => {
                cb.commit();
                self.prof_commit(&cb);
                *self.dispatches.borrow_mut() = 0;
                self.retained.borrow_mut().clear();
                *self.last_committed.borrow_mut() = None;
                Some(cb)
            }
            None => self.last_committed.borrow_mut().take(),
        };
        {
            let mut p = self.prof.borrow_mut();
            if p.on {
                p.syncs += 1;
            }
        }
        if let Some(cb) = waiting_on {
            let t0 = self.prof.borrow().on.then(std::time::Instant::now);
            cb.waitUntilCompleted();
            if let Some(t0) = t0 {
                self.prof.borrow_mut().wait_s += t0.elapsed().as_secs_f64();
            }
        }
        // The queue is serial, so everything committed before that buffer has
        // completed too and the timestamps are readable now.
        self.prof_harvest();
    }

    /// Keep a buffer alive until the current batch completes.
    pub fn retain(&self, b: Rc<Buffer>) {
        self.retained.borrow_mut().push(b);
    }

    // --- hazards ------------------------------------------------------------

    /// What the dispatch now being encoded does to `b`, for the attribution.
    fn touch(&self, b: &Bound) -> Touch {
        Touch {
            serial: b.serial,
            start: b.start,
            end: b.end,
            kernel: self.cur_kernel.borrow().clone(),
        }
    }

    /// `set_input_array`: a read of `b` needs a barrier if this epoch has
    /// already written it. `wrote` is what the dispatch does to `b`, which
    /// only the attribution cares about -- it is the difference between
    /// read-after-write and write-after-write.
    fn note_input(&self, b: &Bound, wrote: bool) {
        assert!(
            self.written.borrow().contains(&b.id),
            "{} reads a pooled buffer nothing has written: whichever op writes \
             it bound it with `arr` instead of `out`",
            self.last_kernel.borrow()
        );
        let prof_on = self.prof.borrow().on;
        let mut h = self.hazards.borrow_mut();
        let raw_after_write = h.prev_out.contains(&b.id);
        if prof_on {
            if raw_after_write && let Some(t) = h.prev_out_by.get(&b.id).cloned() {
                let kind = if wrote { Kind::Waw } else { Kind::Raw };
                h.causes.push((kind, why(&t, b), t.kernel));
            }
            let t = self.touch(b);
            h.next_in_by.insert(b.id, t);
        }
        h.next_in.insert(b.id);
        h.needs_barrier |= raw_after_write;
    }

    /// `register_output_array`: a write also counts as a read (so it picks up
    /// write-after-write), and needs a barrier if this epoch has read `b`.
    fn note_output(&self, b: &Bound) {
        self.written.borrow_mut().insert(b.id);
        self.note_input(b, true);
        let prof_on = self.prof.borrow().on;
        let mut h = self.hazards.borrow_mut();
        let war_after_read = h.prev_in.contains(&b.id);
        if prof_on {
            if war_after_read && let Some(t) = h.prev_in_by.get(&b.id).cloned() {
                h.causes.push((Kind::War, why(&t, b), t.kernel));
            }
            let t = self.touch(b);
            h.next_out_by.insert(b.id, t);
        }
        h.next_out.insert(b.id);
        h.needs_barrier |= war_after_read;
    }

    /// `maybeInsertBarrier`, called once the dispatch's bindings are known and
    /// before it is encoded. A barrier starts a new epoch, so the dispatch
    /// that forced it becomes all the epoch has seen.
    fn maybe_barrier(&self, enc: &ComputeEncoder) {
        if self.barriers != BarrierMode::Selective {
            return;
        }
        let mut h = self.hazards.borrow_mut();
        if h.needs_barrier {
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            {
                let mut prof = self.prof.borrow_mut();
                if prof.on {
                    prof.barriers += 1;
                    let kernel = self.cur_kernel.borrow().clone();
                    record_causes(&mut prof, &kernel, &h.causes);
                    let n = std::mem::take(&mut prof.epoch_dispatches);
                    *prof.epoch_sizes.entry(n).or_default() += 1;
                    if let Some(t) = prof.trace.as_mut()
                        && t.len() < TRACE_BUDGET
                    {
                        let c = h.causes.first();
                        t.push(match c {
                            Some((kind, why, culprit)) => format!(
                                "-- barrier {} {} after {culprit}",
                                kind.as_str(),
                                why.as_str()
                            ),
                            None => "-- barrier".to_string(),
                        });
                    }
                }
            }
            h.needs_barrier = false;
            let next_in = std::mem::take(&mut h.next_in);
            let next_out = std::mem::take(&mut h.next_out);
            h.prev_in = next_in;
            h.prev_out = next_out;
            let in_by = std::mem::take(&mut h.next_in_by);
            let out_by = std::mem::take(&mut h.next_out_by);
            h.prev_in_by = in_by;
            h.prev_out_by = out_by;
        } else {
            let next_in = std::mem::take(&mut h.next_in);
            let next_out = std::mem::take(&mut h.next_out);
            h.prev_in.extend(next_in);
            h.prev_out.extend(next_out);
            let in_by = std::mem::take(&mut h.next_in_by);
            let out_by = std::mem::take(&mut h.next_out_by);
            h.prev_in_by.extend(in_by);
            h.prev_out_by.extend(out_by);
        }
        h.causes.clear();
    }

    // --- profiling ----------------------------------------------------------

    /// Remember the kernel being encoded (for the hazard assert's message).
    fn note_kernel(&self, name: &str) {
        if self.barriers != BarrierMode::Selective {
            return;
        }
        let mut k = self.last_kernel.borrow_mut();
        k.clear();
        k.push_str(name);
        if self.prof.borrow().on {
            *self.cur_kernel.borrow_mut() = Rc::from(name);
        }
    }

    fn prof_name(&self, name: &str) {
        let mut p = self.prof.borrow_mut();
        if !p.on {
            return;
        }
        *p.by_kernel.entry(name.to_string()).or_default() += 1;
        p.cb_names.push(name.to_string());
    }

    fn prof_commit(&self, cb: &CommandBuffer) {
        let mut p = self.prof.borrow_mut();
        if !p.on {
            return;
        }
        p.commits += 1;
        let names = std::mem::take(&mut p.cb_names);
        let one = (names.len() == 1).then(|| names[0].clone());
        p.pending.push((cb.clone(), one));
    }

    /// Sum the GPU busy time of every command buffer that has completed.
    fn prof_harvest(&self) {
        let mut p = self.prof.borrow_mut();
        if !p.on {
            return;
        }
        for (cb, name) in std::mem::take(&mut p.pending) {
            let dt = cb.GPUEndTime() - cb.GPUStartTime();
            p.gpu_s += dt;
            match name {
                Some(n) => *p.gpu_by_kernel.entry(n).or_default() += dt,
                None => p.gpu_mixed += dt,
            }
        }
    }

    /// `(commits, dispatches, gpu_s, wait_s)` since the last reset.
    pub fn prof_totals(&self) -> (usize, usize, f64, f64) {
        let p = self.prof.borrow();
        (p.commits, p.dispatches, p.gpu_s, p.wait_s)
    }

    /// Host round trips since the last reset.
    pub fn prof_syncs(&self) -> usize {
        self.prof.borrow().syncs
    }

    /// Barriers inserted since the last reset.
    pub fn prof_barriers(&self) -> usize {
        self.prof.borrow().barriers
    }

    /// One dispatch's line in the trace: the kernel, then every binding as
    /// `r|w:buffer:allocation:start:end`. That is everything the scheduling
    /// question needs -- which dispatches could have run together is a
    /// property of the bindings, not of the order they were encoded in.
    fn trace_dispatch(&self, bound: &[Bound], marked: bool) {
        let mut p = self.prof.borrow_mut();
        let Some(t) = p.trace.as_mut() else { return };
        if t.len() >= TRACE_BUDGET {
            return;
        }
        let mut line = self.cur_kernel.borrow().to_string();
        for b in bound {
            let dir = if b.is_out || !marked { 'w' } else { 'r' };
            line.push_str(&format!(
                " {dir}:{:x}:{}:{}:{}",
                b.id, b.serial, b.start, b.end
            ));
        }
        t.push(line);
    }

    /// The recorded step, if `ORNITH_TRACE` asked for one.
    pub fn prof_trace(&self) -> Vec<String> {
        self.prof.borrow().trace.clone().unwrap_or_default()
    }

    /// `(dispatches in the epoch, how many epochs held that many)`, smallest
    /// first. A barrier closes an epoch and so does an encoder.
    pub fn prof_epoch_sizes(&self) -> Vec<(usize, usize)> {
        let p = self.prof.borrow();
        let mut v: Vec<(usize, usize)> = p.epoch_sizes.iter().map(|(k, n)| (*k, *n)).collect();
        v.sort_by_key(|e| e.0);
        v
    }

    /// Of those, the ones no data flowed through: see [`Why`].
    pub fn prof_barriers_spurious(&self) -> usize {
        self.prof.borrow().barriers_spurious
    }

    /// `(kernel, culprit, kind, why, spurious, count)`, most frequent first:
    /// which dispatch forced a barrier, which earlier one it collided with,
    /// over what kind of sharing, and whether the barrier as a whole had any
    /// real dependency behind it.
    pub fn prof_barrier_causes(&self) -> Vec<(String, String, Kind, Why, bool, usize)> {
        let p = self.prof.borrow();
        let mut v: Vec<(String, String, Kind, Why, bool, usize)> = p
            .barrier_causes
            .iter()
            .map(|((k, c, kind, why, sp), n)| {
                (k.to_string(), c.to_string(), *kind, *why, *sp, *n)
            })
            .collect();
        v.sort_by(|a, b| b.5.cmp(&a.5).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        v
    }

    pub fn prof_on(&self) -> bool {
        self.prof.borrow().on
    }

    /// `(kernel, dispatches, gpu_s)`, most GPU time first. The GPU column is
    /// only meaningful under `ORNITH_COMMIT_EVERY=1`; otherwise the time
    /// lands in `gpu_mixed` and comes back zero here.
    pub fn prof_kernel_times(&self) -> (Vec<(String, usize, f64)>, f64) {
        let p = self.prof.borrow();
        let mut v: Vec<(String, usize, f64)> = p
            .by_kernel
            .iter()
            .map(|(k, n)| (k.clone(), *n, p.gpu_by_kernel.get(k).copied().unwrap_or(0.0)))
            .collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2).then(b.1.cmp(&a.1)).then(a.0.cmp(&b.0)));
        (v, p.gpu_mixed)
    }

    /// Dispatch counts by kernel, most dispatched first.
    pub fn prof_by_kernel(&self) -> Vec<(String, usize)> {
        let p = self.prof.borrow();
        let mut v: Vec<(String, usize)> =
            p.by_kernel.iter().map(|(k, n)| (k.clone(), *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    /// Zero the counters, keeping them on. Called once the prompt is in, so
    /// the numbers describe decode steps only.
    pub fn prof_reset(&self) {
        let mut p = self.prof.borrow_mut();
        if !p.on {
            return;
        }
        p.commits = 0;
        p.syncs = 0;
        p.barriers = 0;
        p.barriers_spurious = 0;
        p.barrier_causes.clear();
        p.epoch_sizes.clear();
        p.epoch_dispatches = 0;
        if std::env::var_os("ORNITH_TRACE").is_some() {
            p.trace = Some(Vec::new());
        }
        p.dispatches = 0;
        p.gpu_s = 0.0;
        p.wait_s = 0.0;
        p.by_kernel.clear();
        p.gpu_by_kernel.clear();
        p.gpu_mixed = 0.0;
    }

    fn count_dispatch(&self) {
        {
            let mut p = self.prof.borrow_mut();
            if p.on {
                p.dispatches += 1;
                p.epoch_dispatches += 1;
            }
        }
        let mut n = self.dispatches.borrow_mut();
        *n += 1;
        let over = *n >= self.max_dispatches_per_commit;
        drop(n);
        if over {
            self.flush();
        }
    }
}

/// Fold one barrier's causes into the table. A barrier is only ever inserted
/// once per dispatch, so it is keyed on the first hazard the dispatch hit --
/// and counted as spurious only if *none* of its causes was a real one, since
/// a single real dependency is enough to need it.
fn record_causes(prof: &mut Prof, kernel: &Rc<str>, causes: &[(Kind, Why, Rc<str>)]) {
    let spurious = !causes.is_empty() && causes.iter().all(|(_, why, _)| *why != Why::Real);
    if spurious {
        prof.barriers_spurious += 1;
    }
    if let Some((kind, why, culprit)) = causes.first() {
        *prof
            .barrier_causes
            .entry((kernel.clone(), culprit.clone(), *kind, *why, spurious))
            .or_default() += 1;
    }
}

// --- dispatch builder -------------------------------------------------------

/// Bindings a single dispatch can track. No kernel here comes close; the
/// assert in `bind` says so out loud rather than silently dropping one.
const MAX_BOUND: usize = 16;

/// One binding, as the hazard tracking sees it: which buffer, which
/// allocation of it, which bytes, and whether the kernel writes them. Only the
/// id and the direction decide a barrier; the serial and the range are what
/// the attribution needs to say whether that barrier was earned.
#[derive(Clone, Copy)]
struct Bound {
    id: BufId,
    serial: u64,
    /// Byte range bound. `arr` and `out` know the array's extent; a
    /// whole-buffer binding runs to the end, which is the conservative
    /// direction -- it can only make a range look less disjoint than it is.
    start: usize,
    end: usize,
    is_out: bool,
}

impl Bound {
    const EMPTY: Bound = Bound { id: 0, serial: 0, start: 0, end: 0, is_out: false };
}

pub struct Dispatch<'a> {
    ctx: &'a Context,
    enc: ComputeEncoder,
    pipe: Pipeline,
    slot: usize,
    /// The buffers bound so far and whether the kernel writes them, for the
    /// hazard tracking in [`Context::maybe_barrier`].
    bound: [Bound; MAX_BOUND],
    n_bound: usize,
    /// Set by the first [`Dispatch::out`]. Until an op says which of its
    /// bindings it writes, every one of them counts as a write, which forces
    /// the barrier an unmigrated op used to get from the serial encoder.
    marked: bool,
}

impl Drop for Context {
    /// An encoder that is still open when its command buffer is released
    /// trips `-[_MTLCommandEncoder dealloc]: Command encoder released without
    /// endEncoding`, which aborts the process. A test binary that finishes
    /// mid-batch would otherwise die on the way out with the results already
    /// printed.
    fn drop(&mut self) {
        self.sync();
    }
}

impl<'a> Dispatch<'a> {
    fn new(ctx: &'a Context, enc: ComputeEncoder, pipe: Pipeline) -> Dispatch<'a> {
        Dispatch {
            ctx,
            enc,
            pipe,
            slot: 0,
            bound: [Bound::EMPTY; MAX_BOUND],
            n_bound: 0,
            marked: false,
        }
    }

    /// Bind a buffer at the next argument index.
    pub fn buf(self, b: &Buffer) -> Self {
        self.buf_off(b, 0)
    }

    pub fn buf_off(self, b: &Buffer, offset: usize) -> Self {
        unsafe { self.enc.setBuffer_offset_atIndex(Some(b.raw()), offset, self.slot) };
        self.bind(b, offset, b.len - offset.min(b.len), false)
    }

    /// Bind the buffer this kernel *writes*. Same as [`Dispatch::buf`] to
    /// Metal; the difference is that the hazard tracking stops having to
    /// assume every binding is written.
    pub fn out_buf(self, b: &Buffer) -> Self {
        unsafe { self.enc.setBuffer_offset_atIndex(Some(b.raw()), 0, self.slot) };
        self.bind(b, 0, b.len, true)
    }

    /// [`Dispatch::arr`] for the array the kernel writes.
    pub fn out(self, a: &crate::array::Array) -> Self {
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(a.buf.raw()), a.offset, self.slot)
        };
        self.bind(&a.buf, a.offset, a.nbytes(), true)
    }

    fn bind(mut self, b: &Buffer, offset: usize, nbytes: usize, is_out: bool) -> Self {
        if self.ctx.barriers == BarrierMode::Selective {
            assert!(self.n_bound < MAX_BOUND, "more than {MAX_BOUND} buffers in one dispatch");
            self.bound[self.n_bound] = Bound {
                id: b.id(),
                serial: b.serial,
                start: offset,
                end: offset + nbytes,
                is_out,
            };
            self.n_bound += 1;
            self.marked |= is_out;
        }
        Self { slot: self.slot + 1, ..self }
    }

    /// Hand the bindings to the tracker and let it decide on a barrier. Must
    /// run before the dispatch is encoded.
    fn order(&self) {
        if self.ctx.barriers != BarrierMode::Selective {
            return;
        }
        for b in &self.bound[..self.n_bound] {
            if b.is_out || !self.marked {
                self.ctx.note_output(b);
            } else {
                self.ctx.note_input(b, false);
            }
        }
        self.ctx.maybe_barrier(&self.enc);
        self.ctx.trace_dispatch(&self.bound[..self.n_bound], self.marked);
    }

    /// Bind an array: its buffer at the array's byte offset. This is how
    /// every op binds inputs and outputs, so views cost nothing.
    pub fn arr(self, a: &crate::array::Array) -> Self {
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(a.buf.raw()), a.offset, self.slot)
        };
        self.bind(&a.buf, a.offset, a.nbytes(), false)
    }

    /// Bind a small value inline (`setBytes:`), as mlx does for shapes,
    /// strides and scalar parameters.
    pub fn bytes<T: Copy>(self, v: &T) -> Self {
        let ptr = std::ptr::NonNull::from(v).cast::<c_void>();
        unsafe { self.enc.setBytes_length_atIndex(ptr, size_of::<T>(), self.slot) };
        Self { slot: self.slot + 1, ..self }
    }

    pub fn bytes_slice<T: Copy>(self, v: &[T]) -> Self {
        assert!(!v.is_empty(), "setBytes with an empty slice");
        let ptr = std::ptr::NonNull::new(v.as_ptr() as *mut c_void).unwrap();
        unsafe { self.enc.setBytes_length_atIndex(ptr, size_of_val(v), self.slot) };
        Self { slot: self.slot + 1, ..self }
    }

    /// Skip an argument slot (mlx leaves unused optional inputs unbound).
    pub fn skip(self) -> Self {
        Self { slot: self.slot + 1, ..self }
    }

    pub fn threadgroup_memory(self, bytes: usize, index: usize) -> Self {
        unsafe { self.enc.setThreadgroupMemoryLength_atIndex(bytes, index) };
        self
    }

    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.pipe.maxTotalThreadsPerThreadgroup()
    }

    pub fn simd_width(&self) -> usize {
        self.pipe.threadExecutionWidth()
    }

    /// `dispatchThreadgroups:` -- grid is counted in threadgroups, like
    /// mlx's `dispatch_threadgroups`.
    pub fn groups(self, grid: (usize, usize, usize), tg: (usize, usize, usize)) {
        self.order();
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 },
            MTLSize { width: tg.0, height: tg.1, depth: tg.2 },
        );
        self.ctx.count_dispatch();
    }

    /// `dispatchThreads:` -- grid is counted in threads, like mlx's
    /// `dispatch_threads`, which the JIT `mx.fast.metal_kernel` path uses.
    pub fn threads(self, grid: (usize, usize, usize), tg: (usize, usize, usize)) {
        self.order();
        self.enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 },
            MTLSize { width: tg.0, height: tg.1, depth: tg.2 },
        );
        self.ctx.count_dispatch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_rounding() {
        assert_eq!(bucket(1), 256);
        assert_eq!(bucket(256), 256);
        assert_eq!(bucket(257), 320); // next 1/8 step of 512
        assert_eq!(bucket(1024), 1024);
        assert_eq!(bucket(1025), 1280); // next 1/8 step of 2048
        assert!(bucket(100_000) >= 100_000);
    }

    #[test]
    fn library_loads_and_pipelines_resolve() {
        let ctx = Context::new().unwrap();
        assert!(!ctx.device_name().is_empty());
        assert!(ctx.arch.starts_with("applegpu_"), "arch: {}", ctx.arch);
        assert!(ctx.arch_gen >= 13, "arch_gen: {}", ctx.arch_gen);
        // One instantiation from each family we compiled.
        for k in [
            "affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0",
            "affine_gather_qmv_fast_bfloat16_t_gs_32_b_4",
            "affine_dequantize_bfloat16_t_gs_64_b_8",
        ] {
            ctx.pipeline(k);
        }
        // Kernels with function constants must be specialised first.
        ctx.pipeline("rmsbfloat16");
        ctx.pipeline_with("rope_bfloat16", &[(1, true), (2, false), (3, false)]);
        ctx.pipeline_with(
            "affine_qmm_t_bfloat16_t_gs_32_b_4_alN_true_batch_0",
            &[(200, true), (201, true), (202, true)],
        );
    }

    #[test]
    fn pool_recycles() {
        let ctx = Context::new().unwrap();
        let p0 = {
            let b = ctx.alloc(4096);
            b.raw().gpuAddress()
        };
        let b2 = ctx.alloc(4096);
        assert_eq!(p0, b2.raw().gpuAddress(), "second alloc should reuse the freed buffer");
    }
}
