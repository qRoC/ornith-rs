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
use crate::ops::kquant::{self, KMatrix};
use crate::ops::quantized::{QuantMatrix, QuantMode, gather_qmm_rows, quantized_matmul};

/// A quantized `[O, I]` matrix that `x @ w.T` can be asked of.
#[derive(Clone)]
pub enum Linear {
    /// One of mlx's modes, held to `max_abs == 0.0` against MLX.
    Mlx(QuantMatrix),
    /// One of ggml's k-quants, held to `max_abs == 0.0` against ggml.
    K(KMatrix),
}

impl Linear {
    /// `x @ w.T`, with `x` `[.., M, I]` and the result `[.., M, O]`.
    pub fn matmul(&self, ctx: &Context, x: &Array) -> Array {
        match self {
            Linear::Mlx(q) => quantized_matmul(ctx, x, q),
            Linear::K(q) => kquant::matmul(ctx, x, q),
        }
    }

    /// `mx.gather_qmm` at M = 1: one output row per (token, expert) pair,
    /// reading `x[lhs[r]]` against expert `rhs[r]`.
    ///
    /// `sorted` is mlx's hint that the rows are grouped by expert, which lets
    /// its kernel stream an expert's weights once for a run of rows. The
    /// k-quant's gemm does the same, by finding the runs inside its own tile,
    /// so the flag now reaches both sides -- and on the k-quant side it is only
    /// a hint about speed: `kquant::gather_qmm_t` is correct for any ordering.
    pub fn gather_matmul(
        &self,
        ctx: &Context,
        x: &Array,
        lhs_indices: &Array,
        rhs_indices: &Array,
        sorted: bool,
    ) -> Array {
        match self {
            Linear::Mlx(q) => gather_qmm_rows(ctx, x, q, lhs_indices, rhs_indices, sorted),
            Linear::K(q) => kquant::gather_matmul(ctx, x, q, lhs_indices, rhs_indices, sorted),
        }
    }

    /// Experts for a routed weight, else 1.
    pub fn num_experts(&self) -> usize {
        match self {
            Linear::Mlx(q) => q.num_experts(),
            Linear::K(q) => q.num_experts(),
        }
    }

    pub fn out_dims(&self) -> usize {
        match self {
            Linear::Mlx(q) => q.out_dims(),
            Linear::K(q) => q.out_dims(),
        }
    }

    pub fn in_dims(&self) -> usize {
        match self {
            Linear::Mlx(q) => q.in_dims(),
            Linear::K(q) => q.in_dims(),
        }
    }

    pub fn mode(&self) -> QuantMode {
        match self {
            Linear::Mlx(q) => q.mode,
            Linear::K(q) => QuantMode::K(q.kind),
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
            Linear::K(_) => panic!(
                "this module is a k-quant; the caller wants mlx's three-tensor \
                 affine layout, which a k-quant does not have"
            ),
        }
    }
}
