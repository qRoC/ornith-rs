//! Slicing and transposes. See kernels/layout.metal for why these are copies.

use crate::array::{Array, Dtype};
use crate::metal::Context;

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no layout kernel for {}", d.name()),
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

/// `x[..., start..start + len]`, keeping the leading axes.
pub fn slice_inner(ctx: &Context, x: &Array, start: usize, len: usize) -> Array {
    let row = x.dim(-1);
    assert!(start + len <= row);
    if start == 0 && len == row {
        return x.clone();
    }
    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = len;
    let out = Array::empty(ctx, &shape, x.dtype);
    let (grid, tg) = flat(out.numel());
    ctx.kernel(&format!("slice_inner_{}", tname(x.dtype)))
        .arr(x)
        .arr(&out)
        .bytes(&(row as u32))
        .bytes(&(len as u32))
        .bytes(&(start as u32))
        .threads(grid, tg);
    out
}

/// `[B, L, H, D] -> [B, H, L, D]` (`transpose(0, 2, 1, 3)`).
pub fn transpose_lh(ctx: &Context, x: &Array) -> Array {
    let (b, l, h, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    if l == 1 || h == 1 {
        // The move is a no-op in memory; only the shape changes.
        return x.reshape(&[b, h, l, d]);
    }
    let out = Array::empty(ctx, &[b, h, l, d], x.dtype);
    let (grid, tg) = flat(out.numel());
    ctx.kernel(&format!("transpose_lh_{}", tname(x.dtype)))
        .arr(x)
        .arr(&out)
        .bytes(&(l as u32))
        .bytes(&(h as u32))
        .bytes(&(d as u32))
        .threads(grid, tg);
    out
}

/// `[B, H, L, D] -> [B, L, H, D]`.
pub fn transpose_hl(ctx: &Context, x: &Array) -> Array {
    let (b, h, l, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    if l == 1 || h == 1 {
        return x.reshape(&[b, l, h, d]);
    }
    let out = Array::empty(ctx, &[b, l, h, d], x.dtype);
    let (grid, tg) = flat(out.numel());
    ctx.kernel(&format!("transpose_hl_{}", tname(x.dtype)))
        .arr(x)
        .arr(&out)
        .bytes(&(l as u32))
        .bytes(&(h as u32))
        .bytes(&(d as u32))
        .threads(grid, tg);
    out
}

/// `mx.concatenate([a, b], axis=-1)`.
pub fn concat_inner(ctx: &Context, a: &Array, b: &Array) -> Array {
    assert_eq!(a.dtype, b.dtype);
    let (ar, br) = (a.dim(-1), b.dim(-1));
    assert_eq!(a.numel() / ar, b.numel() / br, "rows must match");
    let mut shape = a.shape.clone();
    *shape.last_mut().unwrap() = ar + br;
    let out = Array::empty(ctx, &shape, a.dtype);
    let (grid, tg) = flat(out.numel());
    ctx.kernel(&format!("concat_inner_{}", tname(a.dtype)))
        .arr(a)
        .arr(b)
        .arr(&out)
        .bytes(&(ar as u32))
        .bytes(&(br as u32))
        .threads(grid, tg);
    out
}

/// `mx.concatenate(parts, axis=0)` for row-contiguous arrays that share every
/// axis but the first. MTPLX uses it to glue one tower call per image into the
/// single row block a `VisionSplice` carries.
pub fn concat_leading(ctx: &Context, parts: &[&Array]) -> Array {
    assert!(!parts.is_empty());
    if parts.len() == 1 {
        return parts[0].clone();
    }
    let dtype = parts[0].dtype;
    let rows: usize = parts.iter().map(|p| p.shape[0]).sum();
    let mut shape = parts[0].shape.clone();
    shape[0] = rows;
    let out = Array::empty(ctx, &shape, dtype);
    let mut off = 0usize;
    for p in parts {
        assert_eq!(p.dtype, dtype);
        assert_eq!(p.shape[1..], shape[1..]);
        let dst = out.slice_leading(off, p.shape[0]);
        let n = dst.numel();
        let (grid, tg) = flat(n);
        ctx.kernel(&format!("copy_cast_{0}_{0}", tname(dtype))).arr(p).arr(&dst).threads(grid, tg);
        off += p.shape[0];
    }
    out
}
