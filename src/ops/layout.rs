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
    // One row's slice is already contiguous, so it is a view rather than a
    // copy: the kernel would move a few kilobytes and cost a dispatch. Decode
    // runs a single token, which is where most of these come from -- the
    // gated-delta q/k/v split alone is three per layer.
    if x.numel() == row {
        return x.inner_view(start, len);
    }
    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = len;
    let out = Array::empty(ctx, &shape, x.dtype);
    // Sixteen bytes a thread when every stride in the copy is a whole
    // number of vectors and both bases are aligned; see the same guard in
    // `ops::elementwise`, and `REBUILD.md` Phase 36 for why the width is
    // worth checking at all.
    let lanes = 16 / x.dtype.size();
    let wide = crate::ops::elementwise::wide_elem()
        && lanes > 0
        && row % lanes == 0
        && len % lanes == 0
        && start % lanes == 0
        && x.offset % 16 == 0
        && out.offset % 16 == 0;
    let (grid, tg) = flat(if wide { out.numel() / lanes } else { out.numel() });
    let name = if wide { "slice_inner_wide" } else { "slice_inner" };
    ctx.kernel(&format!("{name}_{}", tname(x.dtype)))
        .arr(x)
        .out(&out)
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
    let (grid, tg, name) = transpose_launch(x, &out);
    ctx.kernel(&format!("transpose_lh{name}_{}", tname(x.dtype)))
        .arr(x)
        .out(&out)
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
    let (grid, tg, name) = transpose_launch(x, &out);
    ctx.kernel(&format!("transpose_hl{name}_{}", tname(x.dtype)))
        .arr(x)
        .out(&out)
        .bytes(&(l as u32))
        .bytes(&(h as u32))
        .bytes(&(d as u32))
        .threads(grid, tg);
    out
}

/// Whether a transpose of `x` with innermost axis `d` moves sixteen bytes a
/// thread, and the launch geometry that goes with it.
///
/// **A transpose here is a permutation of contiguous runs, not a strided
/// copy.** `L` and `H` both sit outside `D`, so run `d` of the source lands
/// whole somewhere in the destination and both ends of the move are already
/// coalesced; what the one-element-a-thread form was losing is the width
/// `swiglu`, `sigmoid_gate` and `slice_inner` were losing in Phase 36/37,
/// and it comes back the same way. The guard is theirs too -- `d` a whole
/// number of vectors and **both bases 16-byte aligned**, because an
/// [`Array`] may be a view with a byte offset (see `ops::elementwise::vec16`
/// for the trap) -- and so is the control, `ORNITH_WIDE_ELEM=0`.
fn transpose_launch(
    x: &Array,
    out: &Array,
) -> ((usize, usize, usize), (usize, usize, usize), &'static str) {
    let lanes = 16 / x.dtype.size();
    // The destination is a fresh allocation, so its base is the allocator's
    // rather than a view's -- but it is checked rather than assumed, and
    // `transpose_would_widen` is the same predicate so the two cannot drift.
    if transpose_would_widen(x) && out.offset % 16 == 0 {
        let (grid, tg) = flat(out.numel() / lanes);
        (grid, tg, "_wide")
    } else {
        let (grid, tg) = flat(out.numel());
        (grid, tg, "")
    }
}

/// Whether [`transpose_lh`] / [`transpose_hl`] move sixteen bytes a thread on
/// this array -- the guard itself, so a test can assert which path it
/// covered rather than leaving a green wide/narrow pair to mean nothing. The
/// first such assertion in this port failed on its first run; see
/// `ops::elementwise::would_widen`.
///
/// **A transpose here is a permutation of contiguous runs, not a strided
/// copy.** `L` and `H` both sit outside `D`, so run `d` of the source lands
/// whole somewhere in the destination and both ends of the move are already
/// coalesced; what the one-element-a-thread form was losing is the width
/// `swiglu`, `sigmoid_gate` and `slice_inner` were losing in Phase 36/37,
/// and it comes back the same way. The conditions are theirs too -- `d` a
/// whole number of vectors and **the base 16-byte aligned**, because an
/// [`Array`] may be a view with a byte offset (see `ops::elementwise::vec16`
/// for the trap) -- and so is the control, `ORNITH_WIDE_ELEM=0`.
pub fn transpose_would_widen(x: &Array) -> bool {
    let lanes = 16 / x.dtype.size();
    crate::ops::elementwise::wide_elem()
        && lanes > 0
        && x.shape[3] % lanes == 0
        && x.offset % 16 == 0
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
        .out(&out)
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
