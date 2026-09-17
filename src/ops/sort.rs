//! `multi_block_sort` from `vendor/mlx/mlx/backend/metal/sort.cpp`.
//!
//! `gpu_merge_sort` sends a row wider than `bn * tn` here: a blockwise sort
//! into 2048-element runs, then log2(n_blocks) rounds of partition + merge,
//! ping-ponging between two pairs of buffers. The sampler's 248320-wide argsort
//! is 122 blocks and 7 merge rounds.
//!
//! `mx.argpartition` has no kernel of its own -- `ArgPartition::eval_gpu` says
//! "We direct arg partition to sort for now" -- so `apply_top_k`'s
//! `argpartition(-logprobs, kth=19)` is a full ascending argsort as well, and
//! the order of everything past the top 20 is fully determined.

use crate::array::{Array, Dtype};
use crate::metal::Context;

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        Dtype::U32 => "uint32",
        d => panic!("no sort kernel for {}", d.name()),
    }
}

/// `gpu_merge_sort`'s tile choice.
pub fn sort_tiles(n: usize, dtype: Dtype) -> (usize, usize, usize) {
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
    if bn == 512 && dtype.size() > 4 {
        bn = 256;
    }
    let n_blocks = n.div_ceil(bn * tn);
    (bn, tn, n_blocks)
}

/// `mx.argsort(x, axis=-1)` for a row-contiguous 2-D `x` too wide for one
/// block.
pub fn multi_block_argsort(ctx: &Context, x: &Array) -> Array {
    let n = x.dim(-1);
    let n_rows = x.numel() / n;
    let (bn, tn, n_blocks) = sort_tiles(n, x.dtype);
    assert!(n_blocks > 1, "use the single-block sort for {n} elements");
    let t = tname(x.dtype);

    // `multi_block_sort`'s five temporaries.
    let vals0 = Array::empty(ctx, &[n_rows, n], x.dtype);
    let vals1 = Array::empty(ctx, &[n_rows, n], x.dtype);
    let idxs0 = Array::empty(ctx, &[n_rows, n], Dtype::U32);
    let idxs1 = Array::empty(ctx, &[n_rows, n], Dtype::U32);
    let partitions = Array::empty(ctx, &[n_rows, n_blocks + 1], Dtype::U32);

    // The sorted axis is the last one, so the "non-contiguous" shape/stride
    // pair describes the rows: shape [n_rows], stride [n].
    let nc_shape = [n_rows as i32];
    let nc_str = [n as i64];

    ctx.kernel(&format!("sort_mbsort_{t}_uint32_bn{bn}_tn{tn}"))
        .arr(x)
        .arr(&vals0)
        .arr(&idxs0)
        .bytes(&(n as i32))
        .bytes(&1i32) // stride_sorted_axis
        .bytes(&1i32) // nc_dim
        .bytes_slice(&nc_shape)
        .bytes_slice(&nc_str)
        .groups((n_blocks, n_rows, 1), (bn, 1, 1));

    let mut ping = false;
    let mut out_idxs = &idxs0;
    let n_thr_per_group = (n_blocks + 1).min(1024);

    let mut merge_tiles = 2usize;
    while merge_tiles / 2 < n_blocks {
        let (vin, iin, vout, iout) = if ping {
            (&vals1, &idxs1, &vals0, &idxs0)
        } else {
            (&vals0, &idxs0, &vals1, &idxs1)
        };
        ping = !ping;

        ctx.kernel(&format!("partition_mbsort_{t}_uint32_bn{bn}_tn{tn}"))
            .arr(&partitions)
            .arr(vin)
            .arr(iin)
            .bytes(&(n as i32))
            .bytes(&(merge_tiles as i32))
            .bytes(&(n_blocks as i32))
            .groups((1, n_rows, 1), (n_thr_per_group, 1, 1));

        ctx.kernel(&format!("merge_mbsort_{t}_uint32_bn{bn}_tn{tn}"))
            .arr(&partitions)
            .arr(vin)
            .arr(iin)
            .arr(vout)
            .arr(iout)
            .bytes(&(n as i32))
            .bytes(&(merge_tiles as i32))
            .bytes(&(n_blocks as i32))
            .groups((n_blocks, n_rows, 1), (bn, 1, 1));

        out_idxs = iout;
        merge_tiles *= 2;
    }

    // `copy_gpu_inplace(dev_idxs_out, out, ..., CopyType::Vector)`: the sorted
    // axis is last, so this is a straight contiguous copy.
    let out = Array::empty(ctx, &x.shape, Dtype::U32);
    let total = out.numel();
    ctx.kernel("sampler_copy_uint32")
        .arr(out_idxs)
        .arr(&out)
        .threads((total, 1, 1), (1024.min(total), 1, 1));
    out
}

/// `mx.argsort(x, axis=-1)`, picking the block or multi-block path the way
/// `gpu_merge_sort` does.
pub fn argsort_last_axis(ctx: &Context, x: &Array) -> Array {
    let (_, _, n_blocks) = sort_tiles(x.dim(-1), x.dtype);
    if n_blocks > 1 {
        multi_block_argsort(ctx, x)
    } else {
        crate::ops::moe::argsort_last_axis(ctx, x)
    }
}
