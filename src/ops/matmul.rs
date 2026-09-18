//! Unquantized matmul.
//!
//! Port of `steel_matmul_regular_axpby` and the branches of `Matmul::eval_gpu`
//! this model reaches (vendor/mlx/mlx/backend/metal/matmul.cpp), over mlx's own
//! steel GEMM kernel.
//!
//! Used by attention prefill (mlx's unfused SDPA at head_dim 256), the MTP head
//! and the vision tower -- everything whose weights are native bfloat16.

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::invariant;

/// `mlx::steel::GEMMParams` (kernels/steel/gemm/params.h), field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: i64,
    batch_stride_b: i64,
    batch_stride_d: i64,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

/// Tile shape for this device class. `GEMM_TPARAM_MACRO` keys off the last
/// letter of the architecture name; `applegpu_g16s` is neither 'g'/'p' (small)
/// nor 'd' (large), so it takes the medium-device branch, which is a single
/// unconditional tile.
fn tile_params(ctx: &Context) -> (usize, usize, usize, usize, usize) {
    let devc = ctx.arch.as_bytes().last().copied().unwrap_or(b'g');
    assert!(
        devc != b'g' && devc != b'p' && devc != b'd',
        "only the medium-device tile (64x64x16, wm=wn=2) is instantiated; \
         this GPU ({}) would select a different one",
        ctx.arch
    );
    (64, 64, 16, 2, 2)
}

/// The `c` of `mx.addmm(c, a, b)`: the array plus the strides and scalars
/// mlx reads off it. `ldc` and `fdc` are `c.strides()[-2]` and
/// `c.strides()[-1]`, which for a broadcast bias row are 0 and 1.
pub struct AddMmSource<'a> {
    pub c: &'a Array,
    pub ldc: i32,
    pub fdc: i32,
    pub batch_stride: i64,
    pub alpha: f32,
    pub beta: f32,
}

/// `mlx::steel::GEMMAddMMParams`, the addmm source's description.
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmAddMmParams {
    ldc: i32,
    fdc: i32,
    batch_stride_c: i64,
    alpha: f32,
    beta: f32,
}

/// Batched `a @ b` where the two leading axes may broadcast, plus mlx's
/// `addmm` source `c`.
///
/// `a` is `[..., M, K]` (or `[..., K, M]` when `transpose_a`), `b` is
/// `[..., K, N]` (or `[..., N, K]` when `transpose_b`). `batch_shape` and the
/// per-operand batch strides are passed through to the kernel exactly as
/// mlx does, which is what makes a broadcast batch dimension free.
///
/// `c` is `None` for a plain `matmul` (mlx's `CHECK_AB = false`
/// instantiation) and `Some((c, ldc, fdc, batch_stride_c, alpha, beta))` for
/// an `AddMM`. Note which function constants that sets: `use_out_source` is
/// `alpha != 0 || beta != 1` and `do_axpby` is `use_out_source && (alpha != 1
/// || beta != 1)`, so `nn.Linear`'s bias -- `mx.addmm(b, x, w.T)` with the
/// default alpha = beta = 1 -- turns `use_out_source` on and leaves `do_axpby`
/// off, and the kernel adds `c` without scaling it.
#[allow(clippy::too_many_arguments)]
pub fn steel_matmul(
    ctx: &Context,
    a: &Array,
    b: &Array,
    c: Option<&AddMmSource<'_>>,
    out: &Array,
    m: usize,
    n: usize,
    k: usize,
    transpose_a: bool,
    transpose_b: bool,
    batch_shape: &[i32],
    a_batch_strides: &[i64],
    b_batch_strides: &[i64],
) {
    assert!(!ctx.nax_available(), "this GPU would take mlx's nax gemm path");
    assert_eq!(a.dtype, b.dtype);
    assert_eq!(a.dtype, out.dtype);

    let batch_size_out: usize = batch_shape.iter().map(|d| *d as usize).product();
    let (bm, bn, bk, wm, wn) = tile_params(ctx);

    let lda = if transpose_a { m } else { k };
    let ldb = if transpose_b { k } else { n };
    let ldd = n;

    let has_batch = batch_shape.len() > 1;
    let align_m = m % bm == 0;
    let align_n = n % bn == 0;
    let align_k = k % bk == 0;
    let (use_out_source, do_axpby) = match c {
        Some(c) => (
            c.alpha != 0.0 || c.beta != 1.0,
            (c.alpha != 0.0 || c.beta != 1.0) && (c.alpha != 1.0 || c.beta != 1.0),
        ),
        None => (false, false),
    };

    let tname = match a.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no steel gemm for {}", d.name()),
    };
    let name = format!(
        "steel_gemm_fused_{}{}_{tname}_{tname}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}",
        if transpose_a { 't' } else { 'n' },
        if transpose_b { 't' } else { 'n' },
    );
    // Function constants: has_batch(10), use_out_source(100), do_axpby(110),
    // align_M/N/K(200/201/202).
    let consts = [
        (10usize, has_batch),
        (100, use_out_source),
        (110, do_axpby),
        (200, align_m),
        (201, align_n),
        (202, align_k),
    ];

    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let swizzle_log = 0;

    let params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: lda as i32,
        ldb: ldb as i32,
        ldd: ldd as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        batch_stride_a: a_batch_strides.last().copied().unwrap_or(0),
        batch_stride_b: b_batch_strides.last().copied().unwrap_or(0),
        batch_stride_d: (m * n) as i64,
        swizzle_log,
        gemm_k_iterations_aligned: (k / bk) as i32,
        batch_ndim: batch_shape.len() as i32,
    };

    // The kernel reads a and b batch strides from one flat array.
    let mut strides = a_batch_strides.to_vec();
    strides.extend_from_slice(b_batch_strides);

    let addmm_params = c.map(|c| GemmAddMmParams {
        ldc: c.ldc,
        fdc: c.fdc,
        batch_stride_c: c.batch_stride,
        alpha: c.alpha,
        beta: c.beta,
    });
    let d = ctx.kernel_with(&name, &consts).arr(a).arr(b);
    let d = match (&c, &addmm_params) {
        (Some(c), Some(p)) => d.arr(c.c).out(out).bytes(&params).bytes(p),
        _ => d
            .skip() // buffer 2 is the addmm source `c`
            .out(out)
            .bytes(&params)
            .skip(), // buffer 5 is GEMMAddMMParams
    };
    let d = if has_batch {
        d.bytes_slice(batch_shape).bytes_slice(&strides)
    } else {
        d
    };
    d.groups((tn, tm, batch_size_out), (32, wn, wm));
}

/// `mlx::steel::GEMMSpiltKParams`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmSplitKParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    tiles_n: i32,
    tiles_m: i32,
    split_k_partitions: i32,
    split_k_partition_stride: i32,
    split_k_partition_size: i32,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
}

fn next_power_of_2(n: i32) -> i32 {
    if n <= 1 { 1 } else { (n as u32 - 1).next_power_of_two() as i32 }
}

/// `steel_gemm_splitk_axpby`: partition K, accumulate the partial products in
/// float32, then sum them into the output dtype.
#[allow(clippy::too_many_arguments)]
pub fn steel_gemm_splitk(
    ctx: &Context,
    a: &Array,
    b: &Array,
    c: Option<&AddMmSource<'_>>,
    out: &Array,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    transpose_a: bool,
    transpose_b: bool,
) {
    let _tm = m.div_ceil(32) as i32;
    let _tn = n.div_ceil(32) as i32;
    let _tk = (k / 16) as i32;

    let bm = if m < 40 { 16usize } else { 32 };
    let bn = if n < 40 { 16usize } else { 32 };
    let bk = 16usize;
    let (wm, wn) = (2usize, 2usize);

    let split_k_partitions = next_power_of_2(_tk / (_tm * _tn)).max(2).min(32) as usize;
    let split_k_partition_stride = m * n;
    let gemm_k_iterations = (k / bk) / split_k_partitions;
    let split_k_partition_size = gemm_k_iterations * bk;

    // The partials are always float32, whatever the inputs are.
    let c_split = Array::empty(ctx, &[split_k_partitions, m, n], Dtype::F32);

    let mn_aligned = m % bm == 0 && n % bn == 0;
    let k_aligned = k % bk == 0;
    let tname = match a.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no split-k gemm for {}", d.name()),
    };
    let name = format!(
        "steel_gemm_splitk_{}{}_{tname}_float32_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}_MN_{}aligned_K_{}aligned",
        if transpose_a { 't' } else { 'n' },
        if transpose_b { 't' } else { 'n' },
        if mn_aligned { 't' } else { 'n' },
        if k_aligned { 't' } else { 'n' },
    );

    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let params = GemmSplitKParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: lda as i32,
        ldb: ldb as i32,
        ldc: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        split_k_partitions: split_k_partitions as i32,
        split_k_partition_stride: split_k_partition_stride as i32,
        split_k_partition_size: split_k_partition_size as i32,
        swizzle_log: 0,
        gemm_k_iterations_aligned: gemm_k_iterations as i32,
    };
    ctx.kernel(&name)
        .arr(a)
        .arr(b)
        .out(&c_split)
        .bytes(&params)
        .groups((tn, tm, split_k_partitions), (32, wn, wm));

    let out_tname = match out.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no split-k accum for {}", d.name()),
    };
    // The accumulate kernel carries the addmm epilogue here, not the gemm:
    // `do_axpby` is `alpha != 1 || beta != 0`, so unlike the fused gemm an
    // `nn.Linear` bias (alpha = beta = 1) *does* take the `_axbpy` variant.
    let do_axpby = c.map(|c| c.alpha != 1.0 || c.beta != 0.0).unwrap_or(false);
    let group = super::rope::block_dims(n, m, 1);
    let name = format!(
        "steel_gemm_splitk_accum_{out_tname}_float32{}",
        if do_axpby { "_axbpy" } else { "" }
    );
    let d = ctx
        .kernel(&name)
        // `gemm_splitk_accum` takes the partials `const device`: the gemm
        // above wrote them, this pass sums them into `out`.
        .arr(&c_split)
        .out(out)
        .bytes(&(split_k_partitions as i32))
        .bytes(&(split_k_partition_stride as i32))
        .bytes(&(n as i32));
    let d = match c {
        Some(c) if do_axpby => d
            .arr(c.c)
            .bytes(&c.ldc)
            .bytes(&c.fdc)
            .bytes(&c.alpha)
            .bytes(&c.beta),
        _ => d,
    };
    d.threads((n, m, 1), group);
}

/// `gemv` / `gemv_t`: `Matmul::eval_gpu` routes here whenever `min(M, N) == 1`,
/// which covers every unquantized linear during decode.
///
/// Port of `gemv_axbpy<false>` from matmul.cpp.
// `sn` is assigned once at the top and again in both branches, which is how
// mlx's `gemv_axbpy` writes it; keeping the shape keeps the diff readable.
#[allow(clippy::too_many_arguments, unused_assignments)]
pub fn gemv(
    ctx: &Context,
    a: &Array,
    b: &Array,
    out: &Array,
    m: usize,
    n: usize,
    k: usize,
    batch_size_out: usize,
    lda: usize,
    ldb: usize,
    transpose_a: bool,
    transpose_b: bool,
    batch_shape: &[i32],
    a_batch_strides: &[i64],
    b_batch_strides: &[i64],
) {
    let is_b_matrix = n != 1;
    let (mat, vec) = if is_b_matrix { (b, a) } else { (a, b) };
    let transpose_mat = if is_b_matrix { !transpose_b } else { transpose_a };
    let in_vector_len = k;
    let out_vector_len = if is_b_matrix { n } else { m };
    let mat_ld = if is_b_matrix { ldb } else { lda };
    let (strides_mat, strides_vec) = if is_b_matrix {
        (b_batch_strides, a_batch_strides)
    } else {
        (a_batch_strides, b_batch_strides)
    };
    let contiguous_kernel = batch_shape.len() == 1;

    let (mut tm, mut tn) = (4usize, 4usize);
    let (mut sm, mut sn) = (1usize, 32usize);
    let (mut bm, mut bn) = (1usize, 1usize);
    let n_out_per_tgp;
    let base;

    if transpose_mat {
        if in_vector_len >= 8192 && out_vector_len >= 2048 {
            sm = 4;
            sn = 8;
        } else {
            sm = 8;
            sn = 4;
        }
        bn = if out_vector_len >= 2048 {
            16
        } else if out_vector_len >= 512 {
            4
        } else {
            2
        };
        if out_vector_len < tn {
            tn = 1;
        }
        n_out_per_tgp = bn * sn * tn;
        base = "gemv_t";
    } else {
        bm = if out_vector_len >= 4096 { 8 } else { 4 };
        sn = 32;
        if k <= 64 {
            bm = 1;
            sm = 8;
            sn = 4;
        } else if k >= 16 * out_vector_len {
            bm = 1;
            bn = 8;
        }
        if out_vector_len < tm {
            tm = 1;
        }
        n_out_per_tgp = bm * sm * tm;
        base = "gemv";
    }

    let tname = match out.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no gemv for {}", d.name()),
    };
    let name = format!(
        "{base}_{tname}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc{}_axpby0",
        u8::from(!contiguous_kernel)
    );

    let n_tgp = out_vector_len.div_ceil(n_out_per_tgp);
    let vec_strides: Vec<i64> =
        if strides_vec.is_empty() { vec![0] } else { strides_vec.to_vec() };
    let mat_strides: Vec<i64> =
        if strides_mat.is_empty() { vec![0] } else { strides_mat.to_vec() };
    ctx.kernel(&name)
        .arr(mat)
        .arr(vec)
        .skip() // buffer 2 is the axpby source
        .out(out)
        .bytes(&(in_vector_len as i32))
        .bytes(&(out_vector_len as i32))
        .bytes(&(mat_ld as i32))
        .skip() // alpha
        .skip() // beta
        .bytes(&(batch_shape.len() as i32))
        .bytes_slice(batch_shape)
        .bytes_slice(&vec_strides)
        .bytes_slice(&mat_strides)
        .groups((n_tgp, 1, batch_size_out), (32, bn, bm));
}

/// `mx.matmul` for the row-contiguous, same-dtype case this port needs.
///
/// Shapes follow numpy broadcasting on the batch axes; `a` is `[..., M, K]`,
/// `b` is `[..., K, N]` or, with `transpose_b`, `[..., N, K]`.
pub fn matmul(ctx: &Context, a: &Array, b: &Array, transpose_b: bool) -> Array {
    let m = a.shape[a.ndim() - 2];
    let k = a.shape[a.ndim() - 1];
    let (bk, n) = if transpose_b {
        (b.shape[b.ndim() - 1], b.shape[b.ndim() - 2])
    } else {
        (b.shape[b.ndim() - 2], b.shape[b.ndim() - 1])
    };
    assert_eq!(k, bk, "inner dimensions do not match: {k} vs {bk}");

    let a_batch = &a.shape[..a.ndim() - 2];
    let b_batch = &b.shape[..b.ndim() - 2];
    let rank = a_batch.len().max(b_batch.len());
    let mut batch_shape = vec![0i32; rank];
    let mut a_strides = vec![0i64; rank];
    let mut b_strides = vec![0i64; rank];

    // Broadcast the batch axes, right-aligned, and build the per-operand
    // strides: a broadcast axis gets stride 0, which is how mlx expresses
    // "the same matrix for every head" without materialising it.
    let mut a_stride = (m * k) as i64;
    let mut b_stride = (k * n) as i64;
    for i in (0..rank).enumerate().map(|(_, i)| i).rev() {
        let ai = (i + a_batch.len()).checked_sub(rank).and_then(|j| a_batch.get(j)).copied();
        let bi = (i + b_batch.len()).checked_sub(rank).and_then(|j| b_batch.get(j)).copied();
        let da = ai.unwrap_or(1);
        let db = bi.unwrap_or(1);
        let dim = da.max(db);
        assert!(da == dim || da == 1, "batch axis {i} does not broadcast: {da} vs {dim}");
        assert!(db == dim || db == 1, "batch axis {i} does not broadcast: {db} vs {dim}");
        batch_shape[i] = dim as i32;
        a_strides[i] = if da == 1 { 0 } else { a_stride };
        b_strides[i] = if db == 1 { 0 } else { b_stride };
        a_stride *= da as i64;
        b_stride *= db as i64;
    }

    let mut out_shape: Vec<usize> = batch_shape.iter().map(|d| *d as usize).collect();
    out_shape.push(m);
    out_shape.push(n);
    let out = Array::empty(ctx, &out_shape, a.dtype);

    // Batch invariance: the rows of an M-row product go to the steel gemm and
    // the rows of a 1-row product go to gemv, and the two associate the sum
    // differently. Under `ops::invariant` each row takes the kernel it would
    // have taken alone -- which is the same move `quantized_matmul` makes, and
    // it matters for the same reason: `in_proj_a` and `in_proj_b` are bf16, so
    // the quantized flag alone leaves a gated-delta layer batch-dependent.
    // `src/bin/invariance.rs` found it there, three windows in sixty.
    //
    // Only the unbatched shape is split. The batched products here are the
    // unfused attention's two, which mlx reaches at `l * gqa_factor > 32` --
    // never at a decode step or a verify window of four rows, which is where
    // this flag makes its promise, and always at prefill, where a chunk is not
    // a batch of rows that would otherwise have run alone.
    let trivial_batch = batch_shape.iter().all(|&d| d == 1);
    if invariant::for_rows(m) && n > 1 && trivial_batch {
        let a2 = a.reshape(&[m, k]);
        let o2 = out.reshape(&[m, n]);
        let lda = k;
        let ldb = if transpose_b { k } else { n };
        for r in 0..m {
            gemv(
                ctx,
                &a2.slice_leading(r, 1),
                b,
                &o2.slice_leading(r, 1),
                1,
                n,
                k,
                1,
                lda,
                ldb,
                false,
                transpose_b,
                &[1],
                &[0],
                &[0],
            );
        }
        return out;
    }

    // `Matmul::eval_gpu` sends a vector-shaped product to gemv before it ever
    // reaches the steel gemm.
    if m.min(n) == 1 {
        let batch_size_out: usize = batch_shape.iter().map(|d| *d as usize).product();
        let lda = k;
        let ldb = if transpose_b { k } else { n };
        gemv(
            ctx,
            a,
            b,
            &out,
            m,
            n,
            k,
            batch_size_out,
            lda,
            ldb,
            false,
            transpose_b,
            if batch_shape.is_empty() { &[1] } else { &batch_shape },
            &a_strides,
            &b_strides,
        );
        return out;
    }

    // `steel_matmul`'s split-K specialisation: a small output tile with a long
    // K, and a single batch.
    let batch_size_out: usize = batch_shape.iter().map(|d| *d as usize).product();
    let _tm = m.div_ceil(16) as i64;
    let _tn = n.div_ceil(16) as i64;
    let _tk = (k / 16) as i64;
    let devc = ctx.arch.as_bytes().last().copied().unwrap_or(b'g');
    let min_tmn_threshold: i64 = if devc == b's' || devc == b'd' { 2048 } else { 1024 };
    if batch_size_out == 1
        && _tm * _tn <= min_tmn_threshold
        && _tk >= 8
        && k >= m.max(n)
    {
        let lda = k;
        let ldb = if transpose_b { k } else { n };
        steel_gemm_splitk(ctx, a, b, None, &out, m, n, k, lda, ldb, false, transpose_b);
        return out;
    }

    steel_matmul(
        ctx,
        a,
        b,
        None,
        &out,
        m,
        n,
        k,
        false,
        transpose_b,
        &batch_shape,
        &a_strides,
        &b_strides,
    );
    out
}

/// `nn.Linear` with a bias, i.e. `mx.addmm(bias, x, weight.T)` -- which is
/// `AddMM::eval_gpu`, not `Matmul::eval_gpu`.
///
/// Every linear in the vision tower has a bias, and nothing else in this model
/// does, so this is the only caller. `x` is `[M, K]`, `w` is `[N, K]` as the
/// checkpoint stores it (mlx-lm hands `addmm` a `w.T` view, which
/// `check_transpose` reports as transposed with `b_cols = K`), and `bias` is
/// `[N]`, broadcast to `[M, N]` with strides `(0, 1)`.
///
/// `AddMM::eval_gpu` tries three routes in order; the two this never reaches
/// are asserted rather than silently mis-dispatched.
pub fn addmm(ctx: &Context, x: &Array, w: &Array, bias: &Array) -> Array {
    let m = x.shape[x.ndim() - 2];
    let k = x.shape[x.ndim() - 1];
    let n = w.shape[0];
    assert_eq!(w.shape[1], k, "weight is [out, in]");
    assert_eq!(x.ndim(), 2, "addmm here is always over a flat [M, K]");
    assert_eq!(bias.numel(), n, "bias must match the output width");

    // `gemv_wide` is tried first. `gemv_wide_config` gives up when
    // `passes = (M + 4) / 5` exceeds 3, i.e. for M > 15; an image is never
    // that short.
    assert!(m > 15, "this shape would take mlx's gemv_wide path (M = {m})");
    // ... and then `min(M, N) == 1` would take `gemv_axbpy`.
    assert!(m.min(n) > 1, "this shape would take mlx's gemv_axbpy path");

    let out = Array::empty(ctx, &[m, n], x.dtype);
    let c = AddMmSource { c: bias, ldc: 0, fdc: 1, batch_stride: 0, alpha: 1.0, beta: 1.0 };

    // `steel_matmul_axpby`'s split-K specialisation, the same test the plain
    // matmul makes: one batch, a small output tile, a long K.
    let _tm = m.div_ceil(16) as i64;
    let _tn = n.div_ceil(16) as i64;
    let _tk = (k / 16) as i64;
    let devc = ctx.arch.as_bytes().last().copied().unwrap_or(b'g');
    let min_tmn_threshold: i64 = if devc == b's' || devc == b'd' { 2048 } else { 1024 };
    if _tm * _tn <= min_tmn_threshold && _tk >= 8 && k >= m.max(n) {
        steel_gemm_splitk(ctx, x, w, Some(&c), &out, m, n, k, k, k, false, true);
        return out;
    }

    steel_matmul(ctx, x, w, Some(&c), &out, m, n, k, false, true, &[1], &[0], &[0]);
    out
}

/// `gather_mv` from `mlx/backend/metal/matmul.cpp`: the M == 1 route of
/// `GatherMM::eval_gpu`, i.e. an *unquantized* `SwitchLinear`.
///
/// `w` is the stacked expert weight as the checkpoint stores it,
/// `[experts, out, in]`. mlx-lm hands `gather_mm` a `swapaxes(-1, -2)` view of
/// it, so `check_transpose` sees `stx == 1` and reports it transposed with
/// `mat_cols = in`; then `transpose_mat = (!is_mv) ^ transpose_mat` flips it
/// back, because a SwitchLinear is a vector-matrix product. The kernel that
/// runs is therefore the plain `gemv_gather` with `marix_ld = in` -- which is
/// exactly the row stride of the `[out, in]` block the checkpoint stores, so
/// no copy or transpose is ever materialised.
///
/// `x` is `[rows, in]`, the index arrays are `[rows]`, the output is
/// `[rows, out]`.
pub fn gather_mv(
    ctx: &Context,
    x: &Array,
    w: &Array,
    lhs_indices: &Array,
    rhs_indices: &Array,
) -> Array {
    let k = x.dim(-1);
    let (experts, n) = (w.shape[0], w.shape[1]);
    assert_eq!(w.shape[2], k, "expert weight is [E, out, in]");
    let rows = rhs_indices.numel();
    assert_eq!(lhs_indices.numel(), rows);
    let out = Array::empty(ctx, &[rows, n], x.dtype);

    let (tm, tn) = (if n < 4 { 1 } else { 4 }, 4);
    let (sm, sn) = (1usize, 32usize);
    let (bm, bn) = (if n >= 4096 { 8 } else { 4 }, 1usize);
    let n_out_per_tgp = bm * sm * tm;
    let n_tgp = n.div_ceil(n_out_per_tgp);

    let tname = match out.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no gather gemv for {}", d.name()),
    };
    let name =
        format!("gemv_gather_{tname}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}");

    // Batch bookkeeping: one "batch" per output row. The index arrays are
    // materialised contiguous, so their strides are 1; `index_batch_strides`
    // is the vector's strides followed by the matrix's, which is what the
    // kernel slices with `+ batch_ndim`.
    let batch_shape = [rows as i32];
    let index_batch_strides = [1i64, 1i64];
    let vec_shape = [x.shape[0] as i32];
    let vec_stride = [k as i64];
    let mat_shape = [experts as i32];
    let mat_stride = [(n * k) as i64];

    ctx.kernel(&name)
        .arr(w)
        .arr(x)
        .skip() // buffer 2 is the axpby bias
        .out(&out)
        .bytes(&(k as i32))
        .bytes(&(n as i32))
        .bytes(&(k as i32)) // marix_ld
        .skip() // alpha
        .skip() // beta
        .bytes(&1i32) // batch_ndim
        .bytes_slice(&batch_shape)
        .bytes_slice(&index_batch_strides)
        .bytes(&1i32) // vector_batch_ndim
        .bytes_slice(&vec_shape)
        .bytes_slice(&vec_stride)
        .bytes(&1i32) // matrix_batch_ndim
        .bytes_slice(&mat_shape)
        .bytes_slice(&mat_stride)
        .arr(lhs_indices)
        .arr(rhs_indices)
        .groups((n_tgp, 1, rows), (32, bn, bm));
    out
}
