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
    let lanes = lanes(x.dtype);
    let wide = wide_elem() && vec16(gate, x.numel()) && vec16(x, x.numel()) && vec16(&out, x.numel());
    let (grid, tg) = flat(if wide { x.numel() / lanes } else { x.numel() });
    let name = if wide { "swiglu_wide" } else { "swiglu" };
    ctx.kernel(&format!("{name}_{}", tname(x.dtype)))
        .arr(gate)
        .arr(x)
        .out(&out)
        .threads(grid, tg);
    out
}

/// Elements in a 16-byte access, which is the width the machine is built
/// for. See [`vec16`].
fn lanes(d: Dtype) -> usize {
    16 / d.size()
}

/// Whether a 16-byte vector access is legal on this array over `n` elements.
///
/// Two conditions, and the second is the one a first attempt forgets: the
/// count must divide, **and the base must be 16-byte aligned**. An [`Array`]
/// may be a view -- `slice_leading`, `slice_inner` and every expert view set
/// a byte `offset`, and `setBuffer:offset:` hands it to the kernel -- so a
/// bf16 view starting at an odd element is 2-byte aligned, and reading a
/// 16-byte vector from it is undefined rather than slow. See the same guard
/// in `ops::moe`, which is where the mistake was made first.
fn vec16(a: &Array, n: usize) -> bool {
    let l = lanes(a.dtype);
    l > 0 && n % l == 0 && a.offset % 16 == 0
}

/// Whether the elementwise kernels move sixteen bytes a thread.
///
/// Phase 37's re-taken inventory reads `swiglu` at 106 GB/s and
/// `sigmoid_gate` at 105 on a machine that does 465, where `add_rms` reads
/// 425 and `block_softmax` 404. The difference is the access width and
/// nothing else, which `REBUILD.md` Phase 36 established on the MoE copies.
/// `ORNITH_WIDE_ELEM=0` is the control.
static WIDE_ELEM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static WIDE_ELEM_ENV: std::sync::Once = std::sync::Once::new();

pub(crate) fn wide_elem() -> bool {
    WIDE_ELEM_ENV.call_once(|| {
        if std::env::var("ORNITH_WIDE_ELEM").as_deref() == Ok("0") {
            WIDE_ELEM.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    WIDE_ELEM.load(std::sync::atomic::Ordering::Relaxed)
}

/// Turn the wide elementwise kernels on or off at runtime; returns what it
/// was.
pub fn set_wide_elem(on: bool) -> bool {
    let was = wide_elem();
    WIDE_ELEM.store(on, std::sync::atomic::Ordering::Relaxed);
    was
}

/// Whether [`swiglu`] would take its wide path, so a test can say which one
/// it covered rather than leaving a green pair to mean nothing.
pub fn would_widen(a: &Array, n: usize) -> bool {
    wide_elem() && vec16(a, n)
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
    let lanes = lanes(x.dtype);
    // `inner` too: a thread's sixteen bytes must not straddle two gates.
    let wide = wide_elem()
        && inner % lanes == 0
        && vec16(x, x.numel())
        && vec16(&out, x.numel());
    let (grid, tg) = flat(if wide { x.numel() / lanes } else { x.numel() });
    let name = if wide { "sigmoid_gate_wide" } else { "sigmoid_gate" };
    ctx.kernel(&format!("{name}_{}", tname(x.dtype)))
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
