//! `mx.fast.scaled_dot_product_attention`.
//!
//! Port of `ScaledDotProductAttention::use_fallback` / `eval_gpu` and the
//! `sdpa_vector*` dispatchers in
//! `vendor/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp`, plus the
//! unfused fallback from `vendor/mlx/mlx/fast.cpp`.
//!
//! This model has `head_dim = 256` and a GQA factor of 8 (16 query heads over
//! 2 kv heads), which lands it on both sides of mlx's routing table:
//!
//!   fused vector kernels, when every one of these holds
//!     * query length <= 8          (`eval_gpu`'s vector-mode test)
//!     * query length <= key length (`has_fused_kernel`)
//!     * query length * gqa_factor <= 32, i.e. at most 4 query rows here
//!     * `use_fallback`'s tail, which at head_dim 256 rejects nothing below 9
//!
//!   otherwise the unfused path: `use_fallback` returns true and mlx runs
//!   `softmax(scale*Q @ K^T + mask) @ V` as ordinary ops.
//!
//! So decode (1 row) and MTP verification (up to 4 rows) are fused, and
//! prefill is not. Taking the fused path for prefill would be faster and would
//! not match mlx.

use crate::array::{Array, Dtype};
use crate::metal::{Context, FConst};

use super::elementwise::{causal_mask, scalar_mul};
use super::matmul::steel_matmul;
use super::softmax::softmax;

/// `q`: `[B, Hq, L, D]`, `k`/`v`: `[B, Hkv, S, D]`, all row-contiguous.
/// Returns `[B, Hq, L, D]`.
pub fn scaled_dot_product_attention(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    causal: bool,
) -> Array {
    assert_eq!(q.ndim(), 4);
    let l = q.shape[2];
    let kl = k.shape[2];
    let gqa_factor = q.shape[1] / k.shape[1];
    // `has_fused_kernel` (the vector branch) followed by `use_fallback`.
    let fused = l <= 8 && l <= kl && l * gqa_factor <= 32;
    if fused {
        sdpa_fused_vector(ctx, q, k, v, scale, causal)
    } else {
        sdpa_unfused(ctx, q, k, v, scale, causal)
    }
}

/// Attention against a KV cache whose sequence axis is over-allocated.
///
/// mlx's `KVCache.keys_and_values()` hands the kernels a `[..., :offset, :]`
/// view of a `[B, H, capacity, D]` buffer. The fused kernels take the key and
/// value strides as arguments, so that view costs nothing; the unfused path
/// goes through `matmul`, whose `check_transpose` would materialise a
/// contiguous copy, so this does the same.
pub fn attention_with_cache(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    capacity: usize,
    scale: f32,
    causal: bool,
) -> Array {
    let l = q.shape[2];
    let kl = k.shape[2];
    let gqa_factor = q.shape[1] / k.shape[1];
    let fused = l <= 8 && l <= kl && l * gqa_factor <= 32;
    if fused {
        sdpa_fused_vector_strided(ctx, q, k, v, capacity, scale, causal)
    } else if capacity == kl {
        sdpa_unfused(ctx, q, k, v, scale, causal)
    } else {
        let kc = compact_cache(ctx, k, capacity);
        let vc = compact_cache(ctx, v, capacity);
        sdpa_unfused(ctx, q, &kc, &vc, scale, causal)
    }
}

/// Copy `[B, H, used, D]` out of a `[B, H, capacity, D]` buffer.
fn compact_cache(ctx: &Context, x: &Array, capacity: usize) -> Array {
    let (b, h, used, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let out = Array::empty(ctx, &[b, h, used, d], x.dtype);
    let n = out.numel();
    let tg = 256.min(n.next_power_of_two()).max(1);
    let tname = match x.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no cache copy for {}", d.name()),
    };
    ctx.kernel(&format!("cache_copy_{tname}"))
        .arr(x)
        .arr(&out)
        .bytes(&(capacity as u32))
        .bytes(&(used as u32))
        .bytes(&(used as u32))
        .bytes(&(d as u32))
        .bytes(&0u32)
        .threads((n, 1, 1), (tg, 1, 1));
    out
}

/// mlx's fused vector kernels, the decode path.
fn sdpa_fused_vector(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    causal: bool,
) -> Array {
    let cap = k.shape[2];
    sdpa_fused_vector_strided(ctx, q, k, v, cap, scale, causal)
}

#[allow(clippy::too_many_arguments)]
fn sdpa_fused_vector_strided(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    capacity: usize,
    scale: f32,
    causal: bool,
) -> Array {
    let (b, hq, l, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let hkv = k.shape[1];
    let n = k.shape[2];
    let vd = v.shape[3];
    let gqa_factor = hq / hkv;
    let out = Array::empty(ctx, &[b, hq, l, vd], q.dtype);

    // `do_causal` is only meaningful with more than one query row; mlx sets
    // `do_causal_ && q.shape(2) > 1`.
    let do_causal = causal && l > 1;

    // Strides in elements, as mlx reads them off the arrays. They come from
    // the cache's capacity, not from the logical length: a partially filled
    // cache is a strided view.
    let kd = k.shape[3];
    let vdim = v.shape[3];
    let k_head_stride = if hkv == 1 { (hkv * capacity * kd) as u64 } else { (capacity * kd) as u64 };
    let k_seq_stride = kd as u64;
    let v_head_stride =
        if v.shape[1] == 1 { (v.shape[1] * capacity * vdim) as u64 } else { (capacity * vdim) as u64 };
    let v_seq_stride = vdim as u64;

    let tname = type_name(q.dtype);
    // has_mask(20), query_transposed(21), do_causal(22), bool_mask(23),
    // float_mask(24), has_sinks(25). This port always hands the kernels
    // row-contiguous queries and never uses an array mask or sinks.
    let consts = [
        (20usize, FConst::Bool(false)),
        (21, FConst::Bool(false)),
        (22, FConst::Bool(do_causal)),
        (23, FConst::Bool(false)),
        (24, FConst::Bool(false)),
        (25, FConst::Bool(false)),
    ];

    // Route to the 2-pass kernel exactly where mlx does: a 'd' or 's' device
    // with a key sequence of at least 1024, or GQA past 4096.
    let devc = ctx.arch.as_bytes().last().copied().unwrap_or(b'g');
    let two_pass =
        ((devc == b'd' || devc == b's') && n >= 1024) || (hkv < hq && n >= 4096);

    if !two_pass {
        let name = format!("sdpa_vector_{tname}_{d}_{vd}");
        ctx.kernel_typed(&name, &consts)
            .arr(q)
            .arr(k)
            .arr(v)
            .arr(&out)
            .bytes(&(gqa_factor as i32))
            .bytes(&(n as i32))
            .bytes(&k_head_stride)
            .bytes(&k_seq_stride)
            .bytes(&v_head_stride)
            .bytes(&v_seq_stride)
            .bytes(&scale)
            .groups((b * hq, l, 1), (1024, 1, 1));
        return out;
    }

    // --- two pass -----------------------------------------------------------
    let n_simds = gqa_factor * l;
    let blocks = sdpa_blocks(devc, n, n_simds);

    let partials = Array::empty(ctx, &[b, hq, l, blocks, vd], q.dtype);
    let sums = Array::empty(ctx, &[b, hq, l, blocks], Dtype::F32);
    let maxs = Array::empty(ctx, &[b, hq, l, blocks], Dtype::F32);

    let mut consts1 = consts.to_vec();
    consts1.push((26, FConst::Int(blocks as i32)));

    let name1 = format!("sdpa_vector_2pass_1_{tname}_{d}_{vd}");
    ctx.kernel_typed(&name1, &consts1)
        .arr(q)
        .arr(k)
        .arr(v)
        .arr(&partials)
        .arr(&sums)
        .arr(&maxs)
        .skip() // buffer 6 is unused
        .bytes(&(n as i32))
        .bytes(&k_head_stride)
        .bytes(&k_seq_stride)
        .bytes(&v_head_stride)
        .bytes(&v_seq_stride)
        .bytes(&scale)
        .groups((hkv, b, blocks), (32, gqa_factor, l));

    let name2 = format!("sdpa_vector_2pass_2_{tname}_{vd}");
    ctx.kernel(&name2)
        .arr(&partials)
        .arr(&sums)
        .arr(&maxs)
        .arr(&out)
        .bytes(&(blocks as i32))
        .groups((b * hq, l, 1), (1024, 1, 1));
    out
}

/// The block count for the 2-pass kernel, from `sdpa_vector_2pass`.
fn sdpa_blocks(devc: u8, n: usize, n_simds: usize) -> usize {
    if devc == b's' {
        let mut blocks = 64;
        if n > 1024 && n_simds > 4 {
            blocks = if n <= 8192 {
                128
            } else if n <= 32768 {
                256
            } else if n <= 65536 {
                512
            } else {
                1024
            };
        }
        blocks
    } else if devc == b'd' {
        let mut blocks = 128;
        if n_simds <= 2 && n > 8192 {
            blocks = 256;
        } else if n_simds >= 6 {
            if (16384..65536).contains(&n) {
                blocks = 512;
            } else if n >= 65536 {
                blocks = 1024;
            }
        }
        blocks
    } else if n_simds >= 4 {
        64
    } else {
        32
    }
}

/// `mlx::steel::AttnParams` (kernels/steel/attn/params.h), field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParams {
    b: i32,
    h: i32,
    d: i32,
    q_l: i32,
    k_l: i32,
    gqa_factor: i32,
    scale: f32,
    nq: i32,
    nk: i32,
    nq_aligned: i32,
    nk_aligned: i32,
    q_l_rem: i32,
    k_l_rem: i32,
    q_l_off: i32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}

/// `sdpa_full_self_attention_metal`: mlx's *full* (non-vector) fused
/// attention, over `steel_attention`.
///
/// The trunk never reaches it, because at head_dim 256 `use_fallback` sends
/// anything longer than 8 query rows to the unfused path. The vision tower
/// does: MTPLX pads its head dim from 72 to 80 before calling
/// `mx.fast.scaled_dot_product_attention`, and at head dim 80 with more than 8
/// query rows `has_fused_kernel` accepts and `use_fallback`'s tail
/// (`query_head_dim == 192 || query_head_dim == 256`) rejects.
///
/// No mask, no causal, no sinks -- the vision tower attends within a frame and
/// splits the sequence rather than masking it.
pub fn steel_attention(ctx: &Context, q: &Array, k: &Array, v: &Array, scale: f32) -> Array {
    assert!(
        !ctx.nax_available(),
        "this GPU would take mlx's nax attention path, and its 72 -> 96 pad"
    );
    let (b, h, q_l, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let k_l = k.shape[2];
    assert_eq!(v.shape[3], d, "the full kernel needs matching q/v head dims");
    assert_eq!(d, 80, "only head dim 80 is instantiated (the padded vision 72)");
    let gqa_factor = h / k.shape[1];

    let (wm, wn) = (4usize, 1usize);
    let bd = d;
    let bq = 32usize;
    let bk = if bd < 128 { 32usize } else { 16 };

    let align_q = q_l % bq == 0;
    let align_k = k_l % bk == 0;
    // align_Q(200), align_K(201), has_mask(300), do_causal(301), has_sinks(302)
    let consts = [
        (200usize, align_q),
        (201, align_k),
        (300, false),
        (301, false),
        (302, false),
    ];

    let tname = match q.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        dt => panic!("no steel attention for {}", dt.name()),
    };
    // The mask type name is the query's own when there is no mask.
    let name = format!(
        "steel_attention_{tname}_bq{bq}_bk{bk}_bd{bd}_wm{wm}_wn{wn}_mask{tname}"
    );

    let out = Array::empty(ctx, &[b, h, q_l, d], q.dtype);

    let nq = q_l.div_ceil(bq);
    let nk = k_l.div_ceil(bk);
    let nq_aligned = q_l / bq;
    let nk_aligned = k_l / bk;

    let qs = q.strides();
    let ks = k.strides();
    let vs = v.strides();
    let os = out.strides();
    let params = AttnParams {
        b: b as i32,
        h: h as i32,
        d: d as i32,
        q_l: q_l as i32,
        k_l: k_l as i32,
        gqa_factor: gqa_factor as i32,
        scale,
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: nq_aligned as i32,
        nk_aligned: nk_aligned as i32,
        q_l_rem: (q_l - nq_aligned * bq) as i32,
        k_l_rem: (k_l - nk_aligned * bk) as i32,
        q_l_off: (k_l - q_l) as i32,
        q_strides: [qs[0], qs[1], qs[2]],
        k_strides: [ks[0], ks[1], ks[2]],
        v_strides: [vs[0], vs[1], vs[2]],
        o_strides: [os[0], os[1], os[2]],
    };

    ctx.kernel_with(&name, &consts)
        .arr(q)
        .arr(k)
        .arr(v)
        .arr(&out)
        .bytes(&params)
        .groups((nq, h, b), (32, wm, wn));
    out
}

/// mlx's unfused fallback, used for prefill at head_dim 256.
///
/// `q = scale * q` (rounded to q's dtype), GQA handled by an extra axis with a
/// broadcast batch stride, causal mask applied as a select, then a precise
/// softmax and a second matmul -- the exact sequence in `fast.cpp`.
fn sdpa_unfused(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    causal: bool,
) -> Array {
    let (b, hq, lq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let hkv = k.shape[1];
    let lk = k.shape[2];
    let vd = v.shape[3];
    let n_repeats = hq / hkv;

    let qs = scalar_mul(ctx, q, scale);

    // [B, Hq, Lq, D] -> [B, Hkv, nrep, Lq, D]; k and v gain a length-1 axis in
    // the same place, which the gemm broadcasts with a zero batch stride.
    let batch_shape: Vec<i32> = vec![b as i32, hkv as i32, n_repeats as i32];
    let q_batch_strides: Vec<i64> = vec![
        (hkv * n_repeats * lq * d) as i64,
        (n_repeats * lq * d) as i64,
        (lq * d) as i64,
    ];
    let kv_batch_strides: Vec<i64> =
        vec![(hkv * lk * d) as i64, (lk * d) as i64, 0];

    // scores = q @ k^T
    let scores = Array::empty(ctx, &[b, hkv, n_repeats, lq, lk], q.dtype);
    steel_matmul(
        ctx,
        &qs,
        k,
        None,
        &scores,
        lq,
        lk,
        d,
        false,
        true, // k is [.., Lk, D]: transpose to get [.., D, Lk]
        &batch_shape,
        &q_batch_strides,
        &kv_batch_strides,
    );

    if causal {
        causal_mask(ctx, &scores, lq, lk);
    }
    let probs = softmax(ctx, &scores, true);

    // out = probs @ v
    let out = Array::empty(ctx, &[b, hkv, n_repeats, lq, vd], q.dtype);
    let probs_batch_strides: Vec<i64> = vec![
        (hkv * n_repeats * lq * lk) as i64,
        (n_repeats * lq * lk) as i64,
        (lq * lk) as i64,
    ];
    steel_matmul(
        ctx,
        &probs,
        v,
        None,
        &out,
        lq,
        vd,
        lk,
        false,
        false,
        &batch_shape,
        &probs_batch_strides,
        &kv_batch_strides,
    );

    // [B, Hkv, nrep, Lq, D] -> [B, Hq, Lq, D]
    out.reshape(&[b, hq, lq, vd])
}

fn type_name(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16_t",
        Dtype::F32 => "float",
        d => panic!("no sdpa kernel for {}", d.name()),
    }
}
