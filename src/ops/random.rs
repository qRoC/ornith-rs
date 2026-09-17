//! `mx.random`: the threefry2x32 bit generator and the key sequence built on
//! it.
//!
//! Ports of `RandomBits::eval_gpu` (`mlx/backend/metal/primitives.cpp`) and of
//! `KeySequence` / `split` / `uniform` from `mlx/random.cpp`. The kernel
//! itself is mlx's, vendored verbatim.
//!
//! The sampler depends on this being exact, not merely random: with the same
//! seed mlx and this port must draw the same token.

use crate::array::{Array, Dtype};
use crate::metal::Context;

/// `random::key(seed)`: the 64-bit seed split into two words, high first.
pub fn key(seed: u64) -> [u32; 2] {
    [(seed >> 32) as u32, seed as u32]
}

/// `random::bits(shape, width=4, key)`.
///
/// `keys` is the `[2]` key array; `shape` is the output shape in uint32
/// elements.
pub fn bits(ctx: &Context, keys: &Array, shape: &[usize]) -> Array {
    assert_eq!(keys.shape, vec![2], "bits expects a single [2] key");
    let out = Array::empty(ctx, shape, Dtype::U32);
    let num_keys = keys.numel() / 2;
    let elems_per_key = out.numel() / num_keys;
    let bytes_per_key = out.dtype.size() * elems_per_key;
    let out_per_key = bytes_per_key.div_ceil(4);
    let half_size = out_per_key / 2;
    let odd = out_per_key % 2 == 1;

    let grid = (num_keys, half_size + usize::from(odd), 1);
    ctx.kernel("rbitsc")
        .arr(keys)
        .arr(&out)
        .bytes(&odd)
        .bytes(&(bytes_per_key as u64))
        .threads(grid, super::indexing::block_dims(grid.0, grid.1, 1));
    out
}

/// `KeySequence`, the state `mx.random.seed` sets and every keyless random
/// call advances.
///
/// `next()` is `split(key_)`: `bits({2, 2})` off the current key, the first
/// row kept as the new state and the second handed out.
pub struct KeySequence {
    pub key: Array,
}

impl KeySequence {
    pub fn seed(ctx: &Context, seed: u64) -> KeySequence {
        KeySequence { key: Array::from_slice(ctx, &key(seed), &[2], Dtype::U32) }
    }

    pub fn next(&mut self, ctx: &Context) -> Array {
        let pair = bits(ctx, &self.key, &[2, 2]);
        // `mlx::core::split(pair, 2)` then `reshape(..., {2})`.
        let first = pair.slice_leading(0, 1).reshape(&[2]);
        let second = pair.slice_leading(1, 1).reshape(&[2]);
        // The new state must not alias the buffer the next `bits` writes, and
        // `slice_leading` is a view, so keep the view -- `bits` allocates a
        // fresh output every call, so the aliasing never arises.
        self.key = first;
        second
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_splits_the_seed() {
        assert_eq!(key(0), [0, 0]);
        assert_eq!(key(1), [0, 1]);
        assert_eq!(key(0x0123_4567_89ab_cdef), [0x0123_4567, 0x89ab_cdef]);
    }
}
