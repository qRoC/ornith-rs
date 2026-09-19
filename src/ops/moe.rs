//! The sparse mixture-of-experts block.
//!
//! Port of `Qwen3NextSparseMoeBlock` and `SwitchGLU`
//! (vendor/mlx-lm/mlx_lm/models/qwen3_next.py and switch_layers.py) with the
//! kernel dispatch from `GatherQMM::eval_gpu`.
//!
//! The Python is short, and every line of it is a decision that shows up in the
//! numbers:
//!
//! ```text
//!     gates = self.gate(x)
//!     gates = mx.softmax(gates, axis=-1, precise=True)
//!     inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
//!     scores = mx.take_along_axis(gates, inds, axis=-1)
//!     scores = scores / scores.sum(axis=-1, keepdims=True)      # norm_topk_prob
//!     y = self.switch_mlp(x, inds)
//!     y = (y * scores[..., None]).sum(axis=-2)
//!     shared_y = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
//!     y = y + shared_y
//! ```

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::linear::Linear;

use super::quantized::{QuantMatrix, quantized_matmul};
use super::reduce::{sum_last_axis, sum_second_last_axis};

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        Dtype::U32 => "uint32",
        d => panic!("no moe kernel for {}", d.name()),
    }
}

/// Launch geometry for a flat one-thread-per-element kernel.
///
/// The grid is the exact element count, never rounded up to a multiple of the
/// threadgroup: `dispatchThreads:` handles a non-uniform last threadgroup, and
/// rounding up instead would run threads past the end of the buffer. None of
/// these kernels bounds-check, because with an exact grid they do not have to
/// -- and a rounded grid corrupted memory in a way that only showed up at
/// sequence lengths whose element count was not a multiple of 256.
fn flat(n: usize) -> ((usize, usize, usize), (usize, usize, usize)) {
    let tg = 256.min(n.next_power_of_two()).max(1);
    ((n, 1, 1), (tg, 1, 1))
}

/// `mx.argsort(x, axis=-1)` for a row-contiguous 2-D `x`.
///
/// Port of `single_block_sort`. mlx routes `argpartition` here too, so the
/// router's top-k comes out of a full ascending sort.
pub fn argsort_last_axis(ctx: &Context, x: &Array) -> Array {
    let n = x.dim(-1);
    let n_rows = x.numel() / n;
    let out = Array::empty(ctx, &x.shape, Dtype::U32);

    // The instrument, not a kernel: see `router_fake_order` in
    // kernels/moe.metal. It answers what the sort costs where it runs, which
    // dropping the sort cannot, and it makes the model produce nonsense.
    // Since Phase 31 the sort is not on the default path at all, so this
    // needs `ORNITH_ROUTER_SORT=1` beside it to reach anything.
    if fake_sort() {
        let (grid, tg) = flat(n_rows * n);
        ctx.kernel(&format!("router_fake_order_{}", tname(x.dtype)))
            .arr(x)
            .out(&out)
            .bytes(&(n as u32))
            .threads(grid, tg);
        return out;
    }

    // `gpu_merge_sort`: tn is always 4, and bn is the first bucket that fits.
    let tn = 4usize;
    let potential_bn = n.div_ceil(tn);
    let mut bn = if potential_bn > 256 {
        512
    } else if potential_bn > 128 {
        256
    } else if potential_bn > 64 {
        128
    } else if potential_bn > 32 {
        64
    } else {
        32
    };
    if bn == 512 && x.dtype.size() > 4 {
        bn = 256;
    }
    let n_blocks = n.div_ceil(bn * tn);
    assert_eq!(
        n_blocks, 1,
        "a {n}-wide sort needs mlx's multi_block_sort, which this port does not carry"
    );

    let name = format!("carg_block_sort_{}_uint32_bn{bn}_tn{tn}", tname(x.dtype));
    ctx.kernel(&name)
        .arr(x)
        .out(&out)
        .bytes(&(n as i32)) // size_sorted_axis
        .bytes(&1i32) // in_stride_sorted_axis
        .bytes(&1i32) // out_stride_sorted_axis
        .bytes(&(n as i32)) // in_stride_segment_axis
        .bytes(&(n as i32)) // out_stride_segment_axis
        .groups((1, n_rows, 1), (bn, 1, 1));
    out
}

/// `argpartition(x, kth=-k)[..., -k:]` plus the gathered scores, i.e. the
/// router's expert selection.
pub fn top_k(ctx: &Context, gates: &Array, k: usize) -> (Array, Array) {
    let n = gates.dim(-1);
    let rows = gates.numel() / n;

    // One selection kernel instead of a full sort, a slice and a gather.
    // Exact, and `kernels/moe.metal` explains why: the last k of a stable
    // ascending sort are the k largest by value then by index, and that is
    // what k rounds of selection under the same order produce. Worth 0.385 ms
    // a token -- `REBUILD.md` Phase 31. `ORNITH_ROUTER_SORT=1` puts mlx's
    // sort back, and so does a k this kernel does not carry.
    if router_select() && k <= 8 && n <= 256 {
        let inds = Array::empty(ctx, &[rows, k], Dtype::U32);
        let scores = Array::empty(ctx, &[rows, k], gates.dtype);
        ctx.kernel(&format!("router_top_k_{}", tname(gates.dtype)))
            .arr(gates)
            .out(&inds)
            .out(&scores)
            .bytes(&(n as u32))
            .bytes(&(k as u32))
            .groups((rows, 1, 1), (32, 1, 1));
        return (inds, scores);
    }

    let order = argsort_last_axis(ctx, gates);

    let inds = Array::empty(ctx, &[rows, k], Dtype::U32);
    let (grid, tg) = flat(rows * k);
    ctx.kernel("slice_tail_uint32")
        .arr(&order)
        .out(&inds)
        .bytes(&(n as u32))
        .bytes(&(k as u32))
        .threads(grid, tg);

    let scores = Array::empty(ctx, &[rows, k], gates.dtype);
    ctx.kernel(&format!("take_along_last_{}", tname(gates.dtype)))
        .arr(gates)
        .arr(&inds)
        .out(&scores)
        .bytes(&(n as u32))
        .bytes(&(k as u32))
        .threads(grid, tg);

    (inds, scores)
}

/// `scores / scores.sum(axis=-1, keepdims=True)`.
pub fn normalize_scores(ctx: &Context, scores: &Array) -> Array {
    let k = scores.dim(-1);
    let rows = scores.numel() / k;
    let denom = Array::empty(ctx, &[rows, 1], scores.dtype);
    sum_last_axis(ctx, scores, &denom);

    let out = Array::empty(ctx, &scores.shape, scores.dtype);
    let (grid, tg) = flat(scores.numel());
    ctx.kernel(&format!("div_by_row_{}", tname(scores.dtype)))
        .arr(scores)
        .arr(&denom)
        .out(&out)
        .bytes(&(k as u32))
        .threads(grid, tg);
    out
}

/// `SwitchGLU.__call__`: route each token to its `k` experts.
///
/// `x` is `[tokens, dim]`, `inds` is `[tokens, k]`. Returns `[tokens, k, dim]`.
pub fn switch_glu(
    ctx: &Context,
    x: &Array,
    inds: &Array,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
) -> Array {
    switch_glu_sorted(ctx, x, inds, gate_proj, up_proj, down_proj, None)
}

/// `force_sort` overrides mlx-lm's threshold; sorting is a permutation, so it
/// must not change the result, which makes this a useful cross-check.
///
/// Returns `[tokens, k, dim]` in token order, which is what mlx-lm's
/// `SwitchGLU` returns. The block itself takes [`switch_glu_rows`] instead
/// and skips the pass that puts it back.
pub fn switch_glu_sorted(
    ctx: &Context,
    x: &Array,
    inds: &Array,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
    force_sort: Option<bool>,
) -> Array {
    let (tokens, k) = (x.shape[0], inds.shape[1]);
    let (y, rows) = switch_glu_rows(ctx, x, inds, gate_proj, up_proj, down_proj, force_sort);
    let dim = y.shape[1];
    match rows {
        Some(inv) => unsort_rows(ctx, &y, &inv).reshape(&[tokens, k, dim]),
        None => y.reshape(&[tokens, k, dim]),
    }
}

/// The experts' outputs as `[tokens * k, dim]`, and -- when they are left in
/// expert-sorted order -- the row of them each `(token, j)` pair landed in.
///
/// mlx-lm's `_gather_sort` / `_scatter_unsort` are a pair of copies that
/// exist only to put rows where the next kernel expects them. At a decode
/// step the sorted path is not taken at all; at a prefill chunk of 2048 the
/// inventory read them at 71.7 ms and 78.9 ms over 40 layers, **~10% of the
/// chunk**. Neither consumer needs its copy:
///
///   * the gathered matmul already reads its activations through
///     `lhs_indices` -- that is `mx.gather_qmm`'s contract, and the
///     *unsorted* branch below is built on it -- so where that holds it can
///     take `x` and the permutation in place of a gathered `xs`;
///   * the only consumer of the unsorted output is the block's tail, which
///     indexes `y` a row at a time anyway, so it can take the inverse
///     permutation in place of a permuted `y`.
///
/// Both are addressing and not arithmetic: the same rows meet the same
/// weights, and the tail folds the same `k` rows in the same order. So both
/// are bit-exact and what moves is the cache. `ORNITH_FUSE_MOE_GATHER=0` and
/// `ORNITH_FUSE_MOE_UNSORT=0` are the controls.
///
/// **And that 10% is mostly not what this removes.** Against the copies as
/// they were -- one element a thread -- removing them read -8.3% of a
/// chunk; against the same copies at sixteen bytes a thread it reads
/// **-0.6%, 7 of 8 pairs down** (`REBUILD.md` Phase 36). The kernels were
/// moving 67 and 134 MB at 37 and 68 GB/s on a machine that does 465, and
/// most of what the inventory was pointing at was their access width and
/// not their existence. See [`gather_rows`]. This is kept because it is
/// strictly less work and consistently signed, not because it is the large
/// half -- and because a control has to be the best version of what it
/// replaces, the number above is the one against the wide copy.
///
/// **The front one has a precondition the back one does not**, and it is
/// worth naming because `gather_matmul` hides it: mlx's sorted kernel,
/// `gather_qmm_rhs`, takes no `lhs_indices` at all. It reads row `r` of `x`
/// for output row `r`, which is exactly why mlx-lm gathers first. So on an
/// artifact whose experts are one of mlx's modes the copy stays, and on one
/// whose experts are k-quants it goes, because `kquant_mm_tile` indexes
/// every row it loads. [`Linear::gather_reads_lhs`] is that asked of the
/// weight rather than assumed of the call.
pub fn switch_glu_rows(
    ctx: &Context,
    x: &Array,
    inds: &Array,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
    force_sort: Option<bool>,
) -> (Array, Option<Array>) {
    let tokens = x.shape[0];
    let dim = x.shape[1];
    let k = inds.shape[1];
    let rows = tokens * k;

    // mlx-lm sorts once the batch is big enough for the weight reads to
    // dominate: `do_sort = indices.size >= 64`.
    let do_sort = force_sort.unwrap_or(rows >= 64);

    if !do_sort {
        // Unsorted: each token's row is reused for all k experts, which the
        // kernel expresses through `lhs_indices`.
        let lhs: Vec<u32> = (0..rows as u32).map(|i| i / k as u32).collect();
        let lhs_indices = Array::from_slice(ctx, &lhs, &[rows], Dtype::U32);
        let idx = inds.reshape(&[rows]);
        return (glu(ctx, x, &lhs_indices, &idx, gate_proj, up_proj, down_proj, false), None);
    }

    let (order, inv_order) = counting_sort(ctx, inds, gate_proj.num_experts());

    // `idx` is the expert each sorted row goes to, `src` the row of `x` it
    // reads. One dispatch for the pair, and it is the same one either way:
    // see `gather_expert_rows` in kernels/moe.metal.
    let idx = Array::empty(ctx, &[rows], Dtype::U32);
    let src = Array::empty(ctx, &[rows], Dtype::U32);
    let (g2, t2) = flat(rows);
    ctx.kernel("gather_expert_rows")
        .arr(inds)
        .arr(&order)
        .out(&idx)
        .out(&src)
        .bytes(&(k as u32))
        .threads(g2, t2);

    let indexed = fuse_moe_gather()
        && up_proj.gather_reads_lhs(rows, true)
        && gate_proj.gather_reads_lhs(rows, true);

    let y = if indexed {
        glu(ctx, x, &src, &idx, gate_proj, up_proj, down_proj, true)
    } else {
        let xs = Array::empty(ctx, &[rows, dim], x.dtype);
        gather_rows(ctx, x, &order, &xs, dim, k);
        // Sorted rows are one-to-one with the gathered copy.
        let identity: Vec<u32> = (0..rows as u32).collect();
        let lhs_indices = Array::from_slice(ctx, &identity, &[rows], Dtype::U32);
        glu(ctx, &xs, &lhs_indices, &idx, gate_proj, up_proj, down_proj, true)
    };

    if fuse_moe_unsort() { (y, Some(inv_order)) } else { (unsort_rows(ctx, &y, &inv_order), None) }
}

/// The three projections and the activation between them, over whatever rows
/// `lhs` and `idx` name.
fn glu(
    ctx: &Context,
    x: &Array,
    lhs: &Array,
    idx: &Array,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
    sorted: bool,
) -> Array {
    let rows = idx.numel();
    let x_up = up_proj.gather_matmul(ctx, x, lhs, idx, sorted);
    let x_gate = gate_proj.gather_matmul(ctx, x, lhs, idx, sorted);
    let act = super::elementwise::swiglu(ctx, &x_gate, &x_up);
    // down_proj consumes the activation row-for-row, so its lhs indices are
    // the identity regardless of sorting.
    let identity: Vec<u32> = (0..rows as u32).collect();
    let lhs_down = Array::from_slice(ctx, &identity, &[rows], Dtype::U32);
    down_proj.gather_matmul(ctx, &act, &lhs_down, idx, sorted)
}

/// `xs[r] = x[order[r] / k]`: mlx-lm's `_gather_sort`, a whole pass over `x`.
///
/// Two kernels for one copy, and the difference between them is only how
/// many bytes a thread moves. One element a thread is 33.5M threads and 67
/// MB at a prefill chunk, and it measured **37 GB/s** against this
/// machine's 465; sixteen bytes a thread is the same copy at an access
/// width the machine is built for, and it was worth **-4.6% of a 6621-token
/// prompt, 3 of 3 down** -- four times what removing the copy outright is
/// worth. Nothing about this kernel is scattered enough to explain 37 GB/s:
/// consecutive threadgroups walk one row. The width was the whole of it.
/// A row has to be a whole number of vectors for the wide form, which every
/// hidden size in this model is, and `ORNITH_MOE_GATHER_WIDE=0` is the
/// control -- which also picks the width of [`unsort_rows`], since the two
/// are one question. Bit-exact either way: it is a copy, and neither form
/// touches a value.
fn gather_rows(ctx: &Context, x: &Array, order: &Array, xs: &Array, dim: usize, k: usize) {
    let lanes = 16 / x.dtype.size();
    let wide = gather_wide() && lanes > 0 && dim % lanes == 0;
    let n = if wide { xs.numel() / lanes } else { xs.numel() };
    let (grid, tg) = flat(n);
    let name = if wide { "gather_rows_div_wide" } else { "gather_rows_div" };
    ctx.kernel(&format!("{name}_{}", tname(x.dtype)))
        .arr(x)
        .arr(order)
        .out(xs)
        .bytes(&(dim as u32))
        .bytes(&(k as u32))
        .threads(grid, tg);
}

/// Whether the gather moves sixteen bytes a thread. See [`gather_rows`].
static GATHER_WIDE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static GATHER_WIDE_ENV: std::sync::Once = std::sync::Once::new();

fn gather_wide() -> bool {
    GATHER_WIDE_ENV.call_once(|| {
        if std::env::var("ORNITH_MOE_GATHER_WIDE").as_deref() == Ok("0") {
            GATHER_WIDE.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    GATHER_WIDE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the wide gather on or off at runtime; returns what it was.
pub fn set_gather_wide(on: bool) -> bool {
    let was = gather_wide();
    GATHER_WIDE.store(on, std::sync::atomic::Ordering::Relaxed);
    was
}

/// `y[inv_order]`: mlx-lm's `_scatter_unsort`, a whole pass over `y`.
fn unsort_rows(ctx: &Context, y: &Array, inv: &Array) -> Array {
    let (rows, dim) = (y.shape[0], y.shape[1]);
    let out = Array::empty(ctx, &[rows, dim], y.dtype);
    let lanes = 16 / y.dtype.size();
    let wide = gather_wide() && lanes > 0 && dim % lanes == 0;
    let n = if wide { out.numel() / lanes } else { out.numel() };
    let (grid, tg) = flat(n);
    let name = if wide { "unsort_rows_wide" } else { "unsort_rows" };
    ctx.kernel(&format!("{name}_{}", tname(y.dtype)))
        .arr(y)
        .arr(inv)
        .out(&out)
        .bytes(&(dim as u32))
        .threads(grid, tg);
    out
}

/// Exposed for the unsort debug test.
pub fn unsort_for_test(ctx: &Context, y: &Array, inv: &Array) -> Array {
    unsort_rows(ctx, y, inv)
}

/// Exposed for the gather debug test.
pub fn gather_rows_for_test(ctx: &Context, x: &Array, order: &Array, k: usize) -> Array {
    let rows = order.numel();
    let dim = x.shape[1];
    let xs = Array::empty(ctx, &[rows, dim], x.dtype);
    gather_rows(ctx, x, order, &xs, dim, k);
    xs
}

/// Exposed so tests can compare the permutation with mlx's `mx.argsort`.
pub fn counting_sort_for_test(ctx: &Context, inds: &Array, n_experts: usize) -> (Array, Array) {
    counting_sort(ctx, inds, n_experts)
}

/// `(order, inv_order)` for mlx-lm's `_gather_sort`. See the kernel comment in
/// kernels/moe.metal for why a counting sort reproduces mlx's argsort here.
///
/// Three dispatches above a block, one below it. The one-thread kernel was
/// written for a decode step, where `n` is `k` -- eight entries -- and the
/// sorted path is not even taken; a prefill chunk of 2048 sends it 16384 and
/// it cost **11% of the chunk** (`REBUILD.md` Phase 35). The parallel form
/// is the same sort with its serial depth cut from `n` to one block, and it
/// is bit-identical by construction. `ORNITH_PARALLEL_SORT=0` is the
/// control.
fn counting_sort(ctx: &Context, inds: &Array, n_experts: usize) -> (Array, Array) {
    /// `ORNITH_SORT_BLOCK` in kernels/moe.metal.
    const BLOCK: usize = 256;
    let n = inds.numel();
    let order = Array::empty(ctx, &[n], Dtype::U32);
    let inv_order = Array::empty(ctx, &[n], Dtype::U32);
    let n_blocks = n.div_ceil(BLOCK);

    // One block is one thread's worth of work either way, and the histogram
    // lives in a 1024-entry threadgroup array.
    if !parallel_sort() || n_blocks <= 1 || n_experts > 1024 {
        let counts = Array::empty(ctx, &[n_experts], Dtype::U32);
        ctx.kernel("counting_sort_experts")
            .arr(inds)
            .out(&order)
            .out(&inv_order)
            .out(&counts)
            .bytes(&(n as u32))
            .bytes(&(n_experts as u32))
            .threads((1, 1, 1), (1, 1, 1));
        return (order, inv_order);
    }

    let blk = Array::empty(ctx, &[n_experts * n_blocks], Dtype::U32);
    ctx.kernel("sort_block_hist")
        .arr(inds)
        .out(&blk)
        .bytes(&(n as u32))
        .bytes(&(n_experts as u32))
        .bytes(&(n_blocks as u32))
        .groups((n_blocks, 1, 1), (BLOCK, 1, 1));
    ctx.kernel("sort_block_offsets")
        .out(&blk)
        .bytes(&(n_experts as u32))
        .bytes(&(n_blocks as u32))
        .groups((1, 1, 1), (BLOCK, 1, 1));
    ctx.kernel("sort_block_scatter")
        .arr(inds)
        .out(&blk)
        .out(&order)
        .out(&inv_order)
        .bytes(&(n as u32))
        .bytes(&(n_blocks as u32))
        .threads((n_blocks, 1, 1), (BLOCK.min(n_blocks), 1, 1));
    (order, inv_order)
}

/// Whether the gather sort is parallel over its input. See [`counting_sort`].
static PARALLEL_SORT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static PARALLEL_SORT_ENV: std::sync::Once = std::sync::Once::new();

fn parallel_sort() -> bool {
    PARALLEL_SORT_ENV.call_once(|| {
        if std::env::var("ORNITH_PARALLEL_SORT").as_deref() == Ok("0") {
            PARALLEL_SORT.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    PARALLEL_SORT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the parallel sort on or off at runtime; returns what it was.
pub fn set_parallel_sort(on: bool) -> bool {
    let was = parallel_sort();
    PARALLEL_SORT.store(on, std::sync::atomic::Ordering::Relaxed);
    was
}

/// The whole tail of the block in one dispatch: the top-k scores normalised,
/// the expert outputs weighted and mixed, the shared expert gated, and the
/// two added.
///
/// `scores` is the router's top-k as [`top_k`] returns it -- not normalised,
/// since the normalising is the first thing this does. The six kernels it
/// replaces did no work at all: 240 dispatches a token, 0.40 ms of a 6.70 ms
/// trunk, all of it launch (`REBUILD.md` Phase 32). `kernels/moe.metal` says
/// what had to be reproduced to keep it bit-exact -- both sums accumulate in
/// bfloat16, in ascending order, because that is what mlx's `Sum` does.
///
/// `rows` is [`switch_glu_rows`]'s permutation when the expert outputs were
/// left in expert order, and `None` when `y` is already `[tokens, k, dim]`.
/// Taking it here is what removes `unsort_rows` -- see the kernel comment.
///
/// `ORNITH_FUSE_MOE_TAIL=0` issues the six separately, as the control. Those
/// six are mlx's own ops and read `y` in token order, so under that control
/// the copy comes back: a control has to compute the same thing.
pub fn combine(
    ctx: &Context,
    scores: &Array,
    y: &Array,
    rows: Option<&Array>,
    gate: &Array,
    shared: &Array,
    normalize: bool,
) -> Array {
    let k = scores.dim(-1);
    let tokens = scores.numel() / k;
    assert_eq!(gate.numel(), tokens, "the shared expert's gate is one number a token");
    let dim = shared.numel() / tokens;
    assert_eq!(shared.numel(), tokens * dim, "the shared expert is one row a token");
    assert_eq!(y.numel(), tokens * k * dim, "expert outputs are k rows a token");
    if let Some(r) = rows {
        assert_eq!(r.numel(), tokens * k, "one row of `y` per (token, expert) pair");
    }

    if !fuse_moe_tail() {
        let y = match rows {
            Some(r) => unsort_rows(ctx, &y.reshape(&[tokens * k, dim]), r),
            None => y.clone(),
        };
        let y = y.reshape(&[tokens, k, dim]);
        let scores = if normalize { normalize_scores(ctx, scores) } else { scores.clone() };
        let mixed = mix_experts(ctx, &y, &scores);
        let gated = super::elementwise::sigmoid_gate(ctx, gate, shared);
        return super::elementwise::add(ctx, &mixed, &gated);
    }

    let out = Array::empty(ctx, &[tokens, dim], y.dtype);
    let (grid, tg) = flat(tokens * dim);
    let name = match rows {
        Some(_) => format!("moe_combine_rows_{}", tname(y.dtype)),
        None => format!("moe_combine_{}", tname(y.dtype)),
    };
    let d = ctx.kernel(&name).arr(scores).arr(y).arr(gate).arr(shared);
    let d = match rows {
        Some(r) => d.arr(r),
        None => d,
    };
    d.out(&out)
        .bytes(&(k as u32))
        .bytes(&(dim as u32))
        .bytes(&(normalize as u32))
        .threads(grid, tg);
    out
}

/// Whether the expert gemm reads `x` through the permutation or a gathered
/// copy of it. See [`switch_glu_rows`].
static FUSE_MOE_GATHER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static FUSE_MOE_GATHER_ENV: std::sync::Once = std::sync::Once::new();

fn fuse_moe_gather() -> bool {
    FUSE_MOE_GATHER_ENV.call_once(|| {
        if std::env::var("ORNITH_FUSE_MOE_GATHER").as_deref() == Ok("0") {
            FUSE_MOE_GATHER.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    FUSE_MOE_GATHER.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the indexed gather on or off at runtime; returns what it was.
pub fn set_fuse_moe_gather(on: bool) -> bool {
    let was = fuse_moe_gather();
    FUSE_MOE_GATHER.store(on, std::sync::atomic::Ordering::Relaxed);
    was
}

/// Whether the block's tail reads `y` through the permutation or a
/// reordered copy of it. See [`switch_glu_rows`].
static FUSE_MOE_UNSORT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static FUSE_MOE_UNSORT_ENV: std::sync::Once = std::sync::Once::new();

fn fuse_moe_unsort() -> bool {
    FUSE_MOE_UNSORT_ENV.call_once(|| {
        if std::env::var("ORNITH_FUSE_MOE_UNSORT").as_deref() == Ok("0") {
            FUSE_MOE_UNSORT.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    FUSE_MOE_UNSORT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the indexed tail on or off at runtime; returns what it was.
pub fn set_fuse_moe_unsort(on: bool) -> bool {
    let was = fuse_moe_unsort();
    FUSE_MOE_UNSORT.store(on, std::sync::atomic::Ordering::Relaxed);
    was
}

/// Whether the block's tail is one dispatch or six. See [`combine`].
static FUSE_MOE_TAIL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static FUSE_MOE_TAIL_ENV: std::sync::Once = std::sync::Once::new();

fn fuse_moe_tail() -> bool {
    FUSE_MOE_TAIL_ENV.call_once(|| {
        if std::env::var("ORNITH_FUSE_MOE_TAIL").as_deref() == Ok("0") {
            FUSE_MOE_TAIL.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    FUSE_MOE_TAIL.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the fusion on or off at runtime; returns what it was.
pub fn set_fuse_moe_tail(on: bool) -> bool {
    let was = fuse_moe_tail();
    FUSE_MOE_TAIL.store(on, std::sync::atomic::Ordering::Relaxed);
    was
}

/// The whole `Qwen3NextSparseMoeBlock`, given the layer's weights.
#[allow(clippy::too_many_arguments)]
pub fn sparse_moe_block(
    ctx: &Context,
    x: &Array,
    gate: &Linear,
    switch_gate: &Linear,
    switch_up: &Linear,
    switch_down: &Linear,
    shared_gate: &Linear,
    shared_up: &Linear,
    shared_down: &Linear,
    shared_expert_gate: &Linear,
    top_k_n: usize,
    norm_topk_prob: bool,
) -> Array {
    let dim = x.dim(-1);
    let tokens = x.numel() / dim;
    let x2 = x.reshape(&[tokens, dim]);

    // Issued interleaved, not in the order the Python reads. The shared expert
    // runs on every token and depends on `x2` alone, so nothing in it waits on
    // the router; the router's tail, meanwhile, is a chain of small dispatches
    // that each hold a barrier epoch on their own. Issuing the two together
    // puts the shared expert's reads into those epochs instead of giving them
    // five more. Every kernel still sees the same inputs and writes the same
    // bytes, so the result is bit-exact -- only the issue order moves. See
    // HANDOFF, "Where the 1216 barriers come from".
    let gates = gate.matmul(ctx, &x2);
    let sg = shared_gate.matmul(ctx, &x2);
    let su = shared_up.matmul(ctx, &x2);
    let gate_pos = shared_gate_pos();
    let mut gate_scalar = (gate_pos == 0).then(|| shared_expert_gate.matmul(ctx, &x2));

    let gates = super::softmax::softmax(ctx, &gates, true);
    let sact = super::elementwise::swiglu(ctx, &sg, &su);

    let (inds, scores) = top_k(ctx, &gates, top_k_n);
    let shared_y = shared_down.matmul(ctx, &sact);

    // `shared_expert_gate` is one row of 2048 -- 6 KB, one output number --
    // and it is the most expensive small dispatch in the step: 12.4 us,
    // because a one-row `qmv` is 32 threads walking K in a chain of dependent
    // loads and there is nothing else in it to hide them behind. Its own
    // epoch is where that shows. It depends on `x2` alone and nothing reads
    // it until the block's tail, so it may be issued into any epoch in
    // between, and which one is worth ~3% of a decode step. See
    // [`shared_gate_pos`] for the three, and `ORNITH_MOE_GATE_POS` to pick
    // one. Same dispatch, same inputs, same output; only where it sits in
    // the queue moves. Moving the shared expert's *down* projection along
    // with it measured the same to within a thousandth of a millisecond, so
    // that stays where it is.
    if gate_pos == 1 {
        gate_scalar = Some(shared_expert_gate.matmul(ctx, &x2));
    }

    let (y, rows) = switch_glu_rows(ctx, &x2, &inds, switch_gate, switch_up, switch_down, None);

    // Position 2, measured and rejected. Fusing the tail (Phase 32) took
    // `sigmoid_gate` out of the epoch that used to cover position 1, and
    // `tools/epoch_slack.py` then found the gate alone at the head of its
    // own epoch again -- 38 of 40 of them, 6.2 us each, 0.235 ms a token if
    // each moved one epoch on, into the last gather's 14-16 us of memory.
    // Issued here it **costs 0.104 ms, 0 of 8 pairs down**: at position 1
    // there are two barriers of gather between the gate and the tail that
    // reads it, and that is what hides its dependent-load chain, not the
    // epoch it shares. The model ranks; it does not predict.
    if gate_pos == 2 {
        gate_scalar = Some(shared_expert_gate.matmul(ctx, &x2));
    }

    let out = combine(
        ctx,
        &scores,
        &y,
        rows.as_ref(),
        gate_scalar.as_ref().unwrap(),
        &shared_y,
        norm_topk_prob,
    );
    out.reshape(&x.shape)
}

/// Whether the router's top-k selects rather than sorting. See [`top_k`].
static ROUTER_SELECT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static ROUTER_SELECT_ENV: std::sync::Once = std::sync::Once::new();

fn router_select() -> bool {
    ROUTER_SELECT_ENV.call_once(|| {
        if std::env::var("ORNITH_ROUTER_SORT").as_deref() == Ok("1") {
            ROUTER_SELECT.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    ROUTER_SELECT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Swap between the selection and mlx's sort at runtime; returns what it was.
pub fn set_router_select(on: bool) -> bool {
    let was = router_select();
    ROUTER_SELECT.store(on, std::sync::atomic::Ordering::Relaxed);
    was
}

/// Whether the router's argsort is replaced by the stand-in that prices it.
static FAKE_SORT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FAKE_SORT_ENV: std::sync::Once = std::sync::Once::new();

fn fake_sort() -> bool {
    FAKE_SORT_ENV.call_once(|| {
        if std::env::var("ORNITH_FAKE_SORT").as_deref() == Ok("1") {
            FAKE_SORT.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    FAKE_SORT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Swap the router's sort for the stand-in at runtime. **The model's output
/// is wrong while this is on.** Returns what it was.
pub fn set_fake_sort(on: bool) -> bool {
    let was = fake_sort();
    FAKE_SORT.store(on, std::sync::atomic::Ordering::Relaxed);
    was
}

/// Which epoch the shared expert's gate is issued into, in
/// [`sparse_moe_block`]:
///
///   0. beside the router, where the Python has it;
///   1. beside the routed experts' gathers -- Phase 31, worth 3.3% then,
///      and the default;
///   2. after them, in the epoch the down gather holds -- what
///      `tools/epoch_slack.py` asked for after Phase 32 moved the epochs,
///      and worth -0.104 ms when it was measured. Kept as the control that
///      says so.
///
/// Since the tail was fused, 0 and 1 measure the same (-0.007 ms, 5 of 8
/// pairs down): the fusion took what the reorder was buying.
///
/// `ORNITH_MOE_GATE_POS` picks one; `ORNITH_MOE_EARLY_GATE=1` is the older
/// spelling of 0 and still works, since it is the control the record quotes.
static GATE_POS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);
static GATE_POS_ENV: std::sync::Once = std::sync::Once::new();

fn shared_gate_pos() -> u8 {
    GATE_POS_ENV.call_once(|| {
        let p = match std::env::var("ORNITH_MOE_GATE_POS").as_deref() {
            Ok(v) => v.parse().expect("ORNITH_MOE_GATE_POS is 0, 1 or 2"),
            Err(_) if std::env::var("ORNITH_MOE_EARLY_GATE").as_deref() == Ok("1") => 0,
            Err(_) => return,
        };
        assert!(p <= 2, "ORNITH_MOE_GATE_POS is 0, 1 or 2");
        GATE_POS.store(p, std::sync::atomic::Ordering::Relaxed);
    });
    GATE_POS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Move it at runtime, so one process can measure all three. Returns what it
/// was.
pub fn set_shared_gate_pos(p: u8) -> u8 {
    let was = shared_gate_pos();
    GATE_POS.store(p, std::sync::atomic::Ordering::Relaxed);
    was
}

/// `(y * scores[..., None]).sum(axis=-2)`: weight each expert's output by its
/// router score and mix them.
pub fn mix_experts(ctx: &Context, y: &Array, scores: &Array) -> Array {
    let (tokens, dim) = (y.shape[0], y.shape[2]);
    let weighted = scale_rows(ctx, y, scores);
    let mixed = Array::empty(ctx, &[tokens, dim], y.dtype);
    sum_second_last_axis(ctx, &weighted, &mixed);
    mixed
}

/// `y * scores[..., None]` for `y` `[tokens, k, dim]` and `scores`
/// `[tokens, k]`.
fn scale_rows(ctx: &Context, y: &Array, scores: &Array) -> Array {
    let dim = y.shape[2];
    let out = Array::empty(ctx, &y.shape, y.dtype);
    let (grid, tg) = flat(y.numel());
    // `sigmoid_gate` is the wrong op here; this is a plain broadcast multiply,
    // so reuse the row-broadcast shape with a dedicated kernel.
    ctx.kernel(&format!("mul_by_row_{}", tname(y.dtype)))
        .arr(scores)
        .arr(y)
        .out(&out)
        .bytes(&(dim as u32))
        .threads(grid, tg);
    out
}

// --- the unquantized block ---------------------------------------------------
//
// The MTP head carries the same 256-expert block as a trunk layer, but stored
// bfloat16 rather than quantized, so every `quantized_matmul` becomes a plain
// `matmul` and `gather_qmm_rows` becomes `gather_mv`. The routing, the sort and
// the mixing are dtype-independent and are shared with the quantized path
// above.

/// `SwitchGLU.__call__` over stacked bfloat16 experts.
///
/// Each weight is `[experts, out, in]`, the layout `stack_numbered_experts`
/// leaves behind.
pub fn switch_glu_dense(
    ctx: &Context,
    x: &Array,
    inds: &Array,
    gate_proj: &Array,
    up_proj: &Array,
    down_proj: &Array,
) -> Array {
    let tokens = x.shape[0];
    let dim = x.shape[1];
    let k = inds.shape[1];
    let rows = tokens * k;
    let experts = gate_proj.shape[0];

    let do_sort = rows >= 64;
    assert!(
        !do_sort,
        "an unquantized SwitchGLU with {rows} rows sorts, which routes to \
         mlx's steel gather gemm; only the M == 1 gather_mv path is ported"
    );
    let _ = experts;

    // Unsorted: each token's row feeds all k experts, which the kernel reads
    // through `lhs_indices`.
    let idx = inds.reshape(&[rows]);
    let lhs: Vec<u32> = (0..rows as u32).map(|i| i / k as u32).collect();
    let lhs_indices = Array::from_slice(ctx, &lhs, &[rows], Dtype::U32);

    let x_up = crate::ops::matmul::gather_mv(ctx, x, up_proj, &lhs_indices, &idx);
    let x_gate = crate::ops::matmul::gather_mv(ctx, x, gate_proj, &lhs_indices, &idx);
    let act = super::elementwise::swiglu(ctx, &x_gate, &x_up);
    let identity: Vec<u32> = (0..rows as u32).collect();
    let lhs_down = Array::from_slice(ctx, &identity, &[rows], Dtype::U32);
    let y = crate::ops::matmul::gather_mv(ctx, &act, down_proj, &lhs_down, &idx);
    y.reshape(&[tokens, k, dim])
}

/// `qwen3_next.SparseMoeBlock` with bfloat16 weights throughout.
#[allow(clippy::too_many_arguments)]
pub fn sparse_moe_block_dense(
    ctx: &Context,
    x: &Array,
    gate: &Array,
    switch_gate: &Array,
    switch_up: &Array,
    switch_down: &Array,
    shared_gate: &Array,
    shared_up: &Array,
    shared_down: &Array,
    shared_expert_gate: &Array,
    top_k_n: usize,
    norm_topk_prob: bool,
) -> Array {
    use crate::ops::matmul::matmul;

    let dim = x.dim(-1);
    let tokens = x.numel() / dim;
    let x2 = x.reshape(&[tokens, dim]);

    let gates = matmul(ctx, &x2, gate, true);
    let gates = super::softmax::softmax(ctx, &gates, true);

    let (inds, scores) = top_k(ctx, &gates, top_k_n);

    let y = switch_glu_dense(ctx, &x2, &inds, switch_gate, switch_up, switch_down);

    let sg = matmul(ctx, &x2, shared_gate, true);
    let su = matmul(ctx, &x2, shared_up, true);
    let sact = super::elementwise::swiglu(ctx, &sg, &su);
    let shared_y = matmul(ctx, &sact, shared_down, true);
    let gate_scalar = matmul(ctx, &x2, shared_expert_gate, true);

    let out = combine(ctx, &scores, &y, None, &gate_scalar, &shared_y, norm_topk_prob);
    out.reshape(&x.shape)
}
