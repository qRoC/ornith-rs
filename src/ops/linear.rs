//! One quantized linear layer, in whichever family its mode belongs to.
//!
//! Until Q4_K there was only one family: every quantized module in the
//! artifact was one of mlx's modes, so a [`QuantMatrix`] and a call to
//! [`quantized_matmul`] were the whole story. ggml's Q4_K is not an mlx mode
//! -- different tensors, different kernels, a different oracle -- so the model
//! now holds a thing that is one or the other and asks it to multiply.
//!
//! Only the modules a Q4_K artifact actually converts carry this: `lm_head`
//! and the three gated-delta projections, which are the 91 that
//! `int4-head-linattn` moved to 4 bits and the ones `Q4K.md` scopes the first
//! cut to. Everything else -- the experts, the routers, the full-attention
//! projections -- stays a `QuantMatrix` and keeps its exact MLX contract, so
//! the golden tests that cover them are untouched.

use crate::array::Array;
use crate::metal::Context;
use crate::ops::q4k::{self, Q4kMatrix};
use crate::ops::quantized::{QuantMatrix, QuantMode, quantized_matmul};

/// A quantized `[O, I]` matrix that `x @ w.T` can be asked of.
#[derive(Clone)]
pub enum Linear {
    /// One of mlx's modes, held to `max_abs == 0.0` against MLX.
    Mlx(QuantMatrix),
    /// ggml's Q4_K, held to `max_abs == 0.0` against ggml.
    Q4k(Q4kMatrix),
}

impl Linear {
    /// `x @ w.T`, with `x` `[.., M, I]` and the result `[.., M, O]`.
    pub fn matmul(&self, ctx: &Context, x: &Array) -> Array {
        match self {
            Linear::Mlx(q) => quantized_matmul(ctx, x, q),
            Linear::Q4k(q) => q4k::qmv(ctx, x, q),
        }
    }

    pub fn out_dims(&self) -> usize {
        match self {
            Linear::Mlx(q) => q.out_dims(),
            Linear::Q4k(q) => q.out_dims(),
        }
    }

    pub fn in_dims(&self) -> usize {
        match self {
            Linear::Mlx(q) => q.in_dims(),
            Linear::Q4k(q) => q.in_dims(),
        }
    }

    pub fn mode(&self) -> QuantMode {
        match self {
            Linear::Mlx(q) => q.mode,
            Linear::Q4k(_) => QuantMode::Q4k,
        }
    }

    /// The underlying mlx matrix, for the things that reach past the product
    /// into the layout: FR-Spec gathers `lm_head`'s rows, and a row of a Q4_K
    /// matrix is three tensors with a 12-byte packing rather than a slice.
    ///
    /// Gathering them is possible and not written: the first cut is one
    /// kernel, and pruning a head that is already being measured for bytes is
    /// a second experiment. Saying so here beats a wrong gather.
    pub fn as_mlx(&self) -> &QuantMatrix {
        match self {
            Linear::Mlx(q) => q,
            Linear::Q4k(_) => panic!(
                "this module is q4k; the caller wants mlx's three-tensor \
                 affine layout, which a k-quant does not have"
            ),
        }
    }
}
