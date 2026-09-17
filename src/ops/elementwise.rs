//! Elementwise ops, including the fused forms mlx reaches via `mx.compile`.

use crate::array::{Array, Dtype};
use crate::metal::Context;

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        Dtype::U32 => "uint32",
        Dtype::I32 => "int32",
        d => panic!("no elementwise kernel for {}", d.name()),
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

/// `out = a` with a dtype conversion.
pub fn copy_cast(ctx: &Context, a: &Array, dtype: Dtype) -> Array {
    let out = Array::empty(ctx, &a.shape, dtype);
    let name = format!("copy_cast_{}_{}", tname(a.dtype), tname(dtype));
    let (grid, tg) = flat(a.numel());
    ctx.kernel(&name).arr(a).out(&out).threads(grid, tg);
    out
}

/// `nn.silu(gate) * x`, mlx_lm's `swiglu`.
pub fn swiglu(ctx: &Context, gate: &Array, x: &Array) -> Array {
    assert_eq!(gate.shape, x.shape);
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let (grid, tg) = flat(x.numel());
    ctx.kernel(&format!("swiglu_{}", tname(x.dtype))).arr(gate).arr(x).out(&out).threads(grid, tg);
    out
}

/// mlx_lm's `precise_swiglu`: silu and the product in float32, result cast
/// back. `h` only supplies the output dtype.
pub fn precise_swiglu(ctx: &Context, gate: &Array, x: &Array) -> Array {
    assert_eq!(gate.shape, x.shape);
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let (grid, tg) = flat(x.numel());
    ctx.kernel(&format!("precise_swiglu_{}", tname(x.dtype)))
        .arr(gate)
        .arr(x)
        .out(&out)
        .threads(grid, tg);
    out
}

/// `sigmoid(gate) * x`, with each gate value covering `inner` values of `x`.
pub fn sigmoid_gate(ctx: &Context, gate: &Array, x: &Array) -> Array {
    let inner = x.numel() / gate.numel();
    assert_eq!(gate.numel() * inner, x.numel(), "gate does not divide x");
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let (grid, tg) = flat(x.numel());
    ctx.kernel(&format!("sigmoid_gate_{}", tname(x.dtype)))
        .arr(gate)
        .arr(x)
        .out(&out)
        .bytes(&(inner as u32))
        .threads(grid, tg);
    out
}

/// `nn.silu(x)` on its own.
pub fn silu(ctx: &Context, x: &Array) -> Array {
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let (grid, tg) = flat(x.numel());
    ctx.kernel(&format!("silu_{}", tname(x.dtype))).arr(x).out(&out).threads(grid, tg);
    out
}

/// `scale * x` in x's dtype.
pub fn scalar_mul(ctx: &Context, x: &Array, scale: f32) -> Array {
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let (grid, tg) = flat(x.numel());
    ctx.kernel(&format!("scalar_mul_{}", tname(x.dtype)))
        .arr(x)
        .out(&out)
        .bytes(&scale)
        .threads(grid, tg);
    out
}

/// In-place `where(q_idx >= k_idx, scores, finfo.min)` over the last two axes.
pub fn causal_mask(ctx: &Context, scores: &Array, q_len: usize, k_len: usize) {
    let rows = scores.numel() / k_len;
    let tgx = 32.min(k_len.next_power_of_two()).max(1);
    let tgy = (1024 / tgx).min(rows.next_power_of_two()).max(1);
    ctx.kernel(&format!("causal_mask_{}", tname(scores.dtype)))
        .out(scores)
        .bytes(&(q_len as u32))
        .bytes(&(k_len as u32))
        .bytes(&(rows as u32))
        .threads((k_len, rows, 1), (tgx, tgy, 1));
}

/// `nn.GELU()` -- the exact, erf-based gelu the vision tower's `PatchMerger`
/// uses.
pub fn gelu_exact(ctx: &Context, x: &Array) -> Array {
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let (grid, tg) = flat(x.numel());
    ctx.kernel(&format!("gelu_exact_{}", tname(x.dtype))).arr(x).out(&out).threads(grid, tg);
    out
}

/// `nn.GELU(approx="tanh")`, the `gelu_pytorch_tanh` the vision blocks' MLP
/// uses.
pub fn gelu_tanh(ctx: &Context, x: &Array) -> Array {
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let (grid, tg) = flat(x.numel());
    ctx.kernel(&format!("gelu_tanh_{}", tname(x.dtype))).arr(x).out(&out).threads(grid, tg);
    out
}

/// `mx.pad(x, [(0, 0), ..., (0, out_row - in_row)])`: widen the last axis with
/// zeros, keeping every leading axis.
pub fn pad_last(ctx: &Context, x: &Array, out_row: usize) -> Array {
    let in_row = x.dim(-1);
    assert!(out_row >= in_row);
    if out_row == in_row {
        return x.clone();
    }
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = out_row;
    let out = Array::empty(ctx, &shape, x.dtype);
    let (grid, tg) = flat(out.numel());
    ctx.kernel(&format!("pad_last_{}", tname(x.dtype)))
        .arr(x)
        .out(&out)
        .bytes(&(in_row as u32))
        .bytes(&(out_row as u32))
        .threads(grid, tg);
    out
}

/// `x + bias`, with `bias` broadcast over the last axis. `nn.Conv3d` adds its
/// bias this way, as a separate op after the convolution.
pub fn add_bias_last(ctx: &Context, x: &Array, bias: &Array) -> Array {
    let row = x.dim(-1);
    assert_eq!(bias.numel(), row, "bias must match the last axis");
    let out = Array::empty(ctx, &x.shape, x.dtype);
    let (grid, tg) = flat(x.numel());
    ctx.kernel(&format!("add_bias_last_{}", tname(x.dtype)))
        .arr(x)
        .arr(bias)
        .out(&out)
        .bytes(&(row as u32))
        .threads(grid, tg);
    out
}

/// `a + b`.
pub fn add(ctx: &Context, a: &Array, b: &Array) -> Array {
    assert_eq!(a.shape, b.shape);
    let out = Array::empty(ctx, &a.shape, a.dtype);
    let (grid, tg) = flat(a.numel());
    ctx.kernel(&format!("add_elem_{}", tname(a.dtype))).arr(a).arr(b).out(&out).threads(grid, tg);
    out
}
