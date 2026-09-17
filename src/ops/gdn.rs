//! The gated delta rule: `mlx_lm/models/gated_delta.py` plus the pieces of
//! `qwen3_5.GatedDeltaNet` that are op dispatch rather than layer structure.

use crate::array::{Array, Dtype};
use crate::metal::Context;

use super::rope::block_dims;

/// `mx.concatenate([head, tail], axis=1)` for `[B, T, D]` arrays.
pub fn concat_time(ctx: &Context, head: &Array, tail: &Array) -> Array {
    assert_eq!(head.ndim(), 3);
    assert_eq!(tail.ndim(), 3);
    let (b, dim) = (head.shape[0], head.shape[2]);
    assert_eq!(tail.shape[0], b);
    assert_eq!(tail.shape[2], dim);
    let (ht, tt) = (head.shape[1], tail.shape[1]);
    let out = Array::empty(ctx, &[b, ht + tt, dim], head.dtype);
    let n = out.numel();
    let tg = 256.min(n.next_power_of_two()).max(1);
    ctx.kernel(&format!("concat_time_{}", tname(head.dtype)))
        .arr(head)
        .arr(tail)
        .out(&out)
        .bytes(&(ht as u32))
        .bytes(&(tt as u32))
        .bytes(&(dim as u32))
        .threads((n, 1, 1), (tg, 1, 1));
    out
}

/// `nn.Conv1d(D, D, kernel_size=K, groups=D, padding=0)` over `[B, T_in, D]`,
/// producing `[B, T_in - K + 1, D]`. Weight is `[D, K, 1]`.
///
/// Port of `depthwise_conv_1D_gpu` (vendor/mlx/mlx/backend/metal/conv.cpp).
pub fn depthwise_conv1d(ctx: &Context, x: &Array, w: &Array) -> Array {
    assert_eq!(x.ndim(), 3);
    let (b, t_in, d) = (x.shape[0], x.shape[1], x.shape[2]);
    let k = w.shape[1];
    assert_eq!(w.shape[0], d, "depthwise conv needs one filter per channel");
    assert_eq!(w.shape[2], 1, "conv1d weights must already be sanitized to [D, K, 1]");
    let t_out = t_in - k + 1;
    let out = Array::empty(ctx, &[b, t_out, d], x.dtype);

    let large = x.numel() > i32::MAX as usize;
    assert!(!large, "the large-index conv kernel is not needed at these sizes");
    let strides = x.strides();
    let s32: [i32; 3] = [strides[0] as i32, strides[1] as i32, strides[2] as i32];
    let group = block_dims(d, t_out, b);
    ctx.kernel(&format!("depthwise_conv_1d_{}", tname(x.dtype)))
        .arr(x)
        .arr(w)
        .out(&out)
        .bytes_slice(&s32)
        .bytes(&(k as i32))
        .threads((d, t_out, b), group);
    out
}

/// `compute_g(A_log, a, dt_bias)` -> float32 `[B, T, Hv]`.
pub fn compute_g(ctx: &Context, a_log: &Array, a: &Array, dt_bias: &Array) -> Array {
    let n_heads = a.dim(-1);
    assert_eq!(a_log.numel(), n_heads);
    assert_eq!(dt_bias.numel(), n_heads);
    let out = Array::empty(ctx, &a.shape, Dtype::F32);
    let n = a.numel();
    let tg = 256.min(n.next_power_of_two()).max(1);
    ctx.kernel(&format!("compute_g_{}", tname(a.dtype)))
        .arr(a_log)
        .arr(a)
        .arr(dt_bias)
        .out(&out)
        .bytes(&(n_heads as u32))
        .threads((n, 1, 1), (tg, 1, 1));
    out
}

/// `beta = mx.sigmoid(b)`.
pub fn beta(ctx: &Context, b: &Array) -> Array {
    let out = Array::empty(ctx, &b.shape, b.dtype);
    let n = b.numel();
    let tg = 256.min(n.next_power_of_two()).max(1);
    ctx.kernel(&format!("gdn_beta_{}", tname(b.dtype)))
        .arr(b)
        .out(&out)
        .threads((n, 1, 1), (tg, 1, 1));
    out
}

/// `gated_delta_update` from mlx_lm/models/gated_delta.py.
///
/// `q`, `k` are `[B, T, Hk, Dk]`, `v` is `[B, T, Hv, Dv]`, `a` and `b` are
/// `[B, T, Hv]`, `state` is `[B, Hv, Dv, Dk]` float32. Returns `(y, state)`
/// with `y` shaped `[B, T, Hv, Dv]`.
///
/// The caller folds the `Dk**-0.5` readout scale into q and k -- that is the
/// contract stated in mlx-lm's docstring, and `GatedDeltaNet.__call__` is what
/// applies it.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_update(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> (Array, Array) {
    let beta = beta(ctx, b);
    let g = compute_g(ctx, a_log, a, dt_bias);
    gated_delta_kernel(ctx, q, k, v, &g, &beta, state, mask)
}

/// mlx-lm's `gated_delta_kernel_unpacked`: the original one-value-row-per-
/// SIMD-group kernel, kept as the comparator for the packed one.
pub fn gated_delta_kernel_unpacked(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> (Array, Array) {
    gated_delta_impl(ctx, q, k, v, g, beta, state, None, false)
}

/// `gated_delta_kernel`: `_gated_delta_kernel_impl` with `allow_packed=true`.
pub fn gated_delta_kernel(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> (Array, Array) {
    gated_delta_impl(ctx, q, k, v, g, beta, state, mask, true)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_impl(
    ctx: &Context,
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
    allow_packed: bool,
) -> (Array, Array) {
    let (b, t, hk, dk) = (k.shape[0], k.shape[1], k.shape[2], k.shape[3]);
    let (hv, dv) = (v.shape[2], v.shape[3]);
    assert_eq!(state.shape, vec![b, hv, dv, dk]);
    assert_eq!(state.dtype, Dtype::F32, "mamba_ssm_dtype is float32");
    assert_eq!(g.dtype, Dtype::F32);

    let y = Array::empty(ctx, &[b, t, hv, dv], q.dtype);
    let state_out = Array::empty(ctx, &state.shape, state.dtype);

    // `packed_eligible` from mlx-lm: no mask, scalar gate, Dk == 128,
    // Dv % 8 == 0, float32 gate and state.
    let packed_eligible =
        allow_packed && mask.is_none() && g.ndim() == 3 && dk == 128 && dv % 8 == 0;
    let shape_suffix = format!(
        "{}_{}_{dk}_{dv}_{hk}_{hv}",
        tname(q.dtype),
        tname(state.dtype)
    );

    if packed_eligible {
        let name = format!("gated_delta_packed_{shape_suffix}");
        ctx.kernel(&name)
            .arr(q)
            .arr(k)
            .arr(v)
            .arr(g)
            .arr(beta)
            .arr(state)
            .bytes(&(t as i32))
            .out(&y)
            .out(&state_out)
            .threads((32, dv / 8, b * hv), (32, 2, 1));
    } else if let Some(mask) = mask {
        let name = format!("gated_delta_mask_{shape_suffix}");
        ctx.kernel(&name)
            .arr(q)
            .arr(k)
            .arr(v)
            .arr(g)
            .arr(beta)
            .arr(state)
            .bytes(&(t as i32))
            .arr(mask)
            .out(&y)
            .out(&state_out)
            .threads((32, dv, b * hv), (32, 4, 1));
    } else {
        let name = format!("gated_delta_{shape_suffix}");
        ctx.kernel(&name)
            .arr(q)
            .arr(k)
            .arr(v)
            .arr(g)
            .arr(beta)
            .arr(state)
            .bytes(&(t as i32))
            .out(&y)
            .out(&state_out)
            .threads((32, dv, b * hv), (32, 4, 1));
    }
    (y, state_out)
}

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no gated-delta kernel for {}", d.name()),
    }
}
