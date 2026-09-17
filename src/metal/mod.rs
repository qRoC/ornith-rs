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
//! *Inside* an encoder, Metal's default `MTLDispatchTypeSerial` inserts a
//! barrier between consecutive dispatches, so writes from dispatch N are
//! visible to dispatch N+1 with no explicit synchronisation, and a recycled
//! buffer can never be written by a later dispatch while an earlier one still
//! reads it.
//!
//! *Across* encoders the queue is what orders the work, and mlx relies on
//! exactly that -- its `commit()` starts the next command buffer with no
//! synchronisation of its own. What mlx does add is a fence per encoder
//! (`CommandEncoder::end_encoding`), waited on by whichever later encoder
//! reads that encoder's outputs; it needs them because its encoders are
//! concurrent. This port signals one fence at the end of every encoder and
//! waits on it in the next, which is the coarse version of the same thing. It
//! costs nothing measurable and it makes the boundary explicit rather than
//! inherited from the queue.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLDispatchType,
    MTLFence, MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSize,
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
    pool: Option<Rc<RefCell<Pool>>>,
}

impl Buffer {
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
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

    fn put(&mut self, len: usize, buf: MTLBuf) {
        self.free.entry(len).or_default().push(buf);
    }
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
    pub commits: usize,
    pub dispatches: usize,
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
    no_barrier: bool,
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
            no_barrier: std::env::var_os("ORNITH_NO_BARRIER").is_some(),
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

    pub fn alloc(&self, nbytes: usize) -> Buffer {
        let len = bucket(nbytes.max(1));
        if let Some(raw) = self.pool.borrow_mut().take(len) {
            return Buffer { raw, len, pool: Some(self.pool.clone()) };
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
        Buffer { raw, len, pool: Some(self.pool.clone()) }
    }

    /// Allocate outside the pool, at exactly the requested size. Weights are
    /// long-lived and large, so bucket rounding would only waste memory.
    pub fn alloc_exact(&self, nbytes: usize) -> Buffer {
        let len = nbytes.max(1);
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .unwrap_or_else(|| panic!("newBufferWithLength({len}) failed"));
        Buffer { raw, len, pool: None }
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
        Buffer { raw, len: map.map_len, pool: None }
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
        // `ORNITH_NO_BARRIER=1` is a measurement mode, not an option: a
        // concurrent encoder with no barriers at all lets every dispatch in a
        // buffer overlap, which is wrong but puts a ceiling on what removing
        // the serial encoder's per-dispatch barrier could ever buy.
        let enc = if self.no_barrier {
            cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .expect("computeCommandEncoder failed")
        } else {
            cb.computeCommandEncoder().expect("computeCommandEncoder failed")
        };
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
        return Dispatch { ctx: self, enc, pipe, slot: 0 };
    }

    pub fn kernel_typed<'a>(&'a self, name: &str, consts: &[(usize, FConst)]) -> Dispatch<'a> {
        let pipe = self.pipeline_typed(name, consts);
        let enc = self.ensure_encoder();
        enc.setComputePipelineState(&pipe);
        self.prof_name(name);
        Dispatch { ctx: self, enc, pipe, slot: 0 }
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

    // --- profiling ----------------------------------------------------------

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

// --- dispatch builder -------------------------------------------------------

pub struct Dispatch<'a> {
    ctx: &'a Context,
    enc: ComputeEncoder,
    pipe: Pipeline,
    slot: usize,
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
    /// Bind a buffer at the next argument index.
    pub fn buf(self, b: &Buffer) -> Self {
        self.buf_off(b, 0)
    }

    pub fn buf_off(self, b: &Buffer, offset: usize) -> Self {
        unsafe { self.enc.setBuffer_offset_atIndex(Some(b.raw()), offset, self.slot) };
        Self { slot: self.slot + 1, ..self }
    }

    /// Bind an array: its buffer at the array's byte offset. This is how
    /// every op binds inputs and outputs, so views cost nothing.
    pub fn arr(self, a: &crate::array::Array) -> Self {
        self.buf_off(&a.buf, a.offset)
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 },
            MTLSize { width: tg.0, height: tg.1, depth: tg.2 },
        );
        self.ctx.count_dispatch();
    }

    /// `dispatchThreads:` -- grid is counted in threads, like mlx's
    /// `dispatch_threads`, which the JIT `mx.fast.metal_kernel` path uses.
    pub fn threads(self, grid: (usize, usize, usize), tg: (usize, usize, usize)) {
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
