//! `mx.fast.rope`.
//!
//! Port of `RoPE::eval_gpu` (vendor/mlx/mlx/backend/metal/rope.cpp) over mlx's
//! `rope.metal`.
//!
//! This model uses `nn.RoPE(dims=64, traditional=False, base=1e7, scale=1.0)`:
//! `initialize_rope` is handed `rope_parameters` whose `type` is `"default"`,
//! so despite the `mrope_section`/`mrope_interleaved` entries sitting next to
//! it, the text path takes the plain rotary branch (mlx_lm/models/rope_utils.py).
//! `dims` is `head_dim * partial_rotary_factor = 256 * 0.25`, so only the first
//! 64 of each head's 256 channels rotate and the rest pass through -- which is
//! why mlx copies the input to the output first and then rotates in place.

use crate::array::Array;
use crate::metal::Context;

use super::elementwise::copy_cast;

const N_PER_THREAD: usize = 4;

/// `rope(x, dims, base, scale, offset)` for a row-contiguous `x` shaped
/// `[B, N, T, D]` (batch, heads, time, head_dim).
pub fn rope(ctx: &Context, x: &Array, dims: usize, base: f32, scale: f32, offset: usize) -> Array {
    assert_eq!(x.ndim(), 4, "expected [B, N, T, D]");
    let (b, n, t, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    assert!(dims <= d);
    let mat_size = t * d;

    // dims < D: mlx copies the whole input into the output and then rotates
    // the leading `dims` channels of that copy in place ("donated").
    let out = if dims < d { copy_cast(ctx, x, x.dtype) } else { Array::empty(ctx, &x.shape, x.dtype) };
    let src = if dims < d { &out } else { x };

    let strides: [i64; 3] = [mat_size as i64, d as i64, 1];
    let out_strides: [i64; 3] = strides;

    // `single` is mlx's inference special case: one time step, contiguous, one
    // offset. That is every decode step.
    let single = t == 1;
    let log2_base = base.log2();

    let type_name = match x.dtype {
        crate::array::Dtype::BF16 => "bfloat16",
        crate::array::Dtype::F32 => "float32",
        dt => panic!("no rope kernel for {}", dt.name()),
    };
    let name = format!("rope_{}{}", if single { "single_" } else { "" }, type_name);
    // forward = true, traditional = false, hs_transpose = false.
    let consts = [(1usize, true), (2, false), (3, false)];

    if single {
        let dim0 = dims / 2;
        let dim1 = b * n;
        let group = block_dims(dim0, dim1, 1);
        ctx.kernel_with(&name, &consts)
            .arr(src)
            .arr(&out)
            .bytes(&(offset as i32))
            .bytes(&scale)
            .bytes(&out_strides[0])
            .skip() // buffers 5..9 are unused by rope_single
            .skip()
            .skip()
            .skip()
            .skip()
            .bytes(&log2_base)
            .threads((dim0, dim1, 1), group);
    } else {
        // The general kernel reads a per-batch offset array.
        let offsets = Array::from_slice(ctx, &vec![offset as i32; b], &[b], crate::array::Dtype::I32);
        let dim0 = dims / 2;
        let dim1 = t;
        let dim2 = b * n.div_ceil(N_PER_THREAD);
        let group = block_dims(dim0, dim1, dim2);
        ctx.kernel_with(&name, &consts)
            .arr(src)
            .arr(&out)
            .arr(&offsets)
            .bytes(&scale)
            .bytes_slice(&strides)
            .bytes_slice(&out_strides)
            .bytes(&1i64) // offset_stride
            .bytes(&(n as i32)) // n_head
            .skip()
            .skip()
            .bytes(&log2_base)
            .threads((dim0, dim1, dim2), group);
    }
    out
}

/// `get_block_dims_common` from vendor/mlx/mlx/backend/common/utils.cpp: grow
/// the three block dimensions one power of two at a time until 2^10 threads.
pub fn block_dims(dim0: usize, dim1: usize, dim2: usize) -> (usize, usize, usize) {
    let dims = [dim0, dim1, dim2];
    let mut pows = [0u32; 3];
    let mut sum = 0;
    loop {
        let presum = sum;
        for i in 0..3 {
            if dims[i] >= (1 << (pows[i] + 1)) {
                pows[i] += 1;
                sum += 1;
            }
            if sum == 10 {
                return (1 << pows[0], 1 << pows[1], 1 << pows[2]);
            }
        }
        if sum == presum {
            return (1 << pows[0], 1 << pows[1], 1 << pows[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::block_dims;

    #[test]
    fn block_dims_match_mlx() {
        // dims/2 = 32 channels x B*N = 16 rows: 32 * 16 = 512 <= 1024.
        assert_eq!(block_dims(32, 16, 1), (32, 16, 1));
        // Capped at 2^10 total threads.
        let (a, b, c) = block_dims(4096, 4096, 4096);
        assert_eq!(a * b * c, 1024);
    }
}
