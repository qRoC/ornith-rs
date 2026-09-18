//! Batch invariance: every row computed with the arithmetic it would have had
//! on its own.
//!
//! `quantized_matmul` picks a kernel by row count -- `qmv` at M = 1,
//! `qmv_wide` at 2..`qmv_batch_limit`, the split-K gemm above that -- and the
//! three do not agree. They are all correct; they associate the same sum
//! differently, and on this model's own weights that is worth up to 7.8e-3 in
//! an `lm_head` logit (`tests/batch_invariance.rs` measures it). Everywhere a
//! batch is an optimisation over rows that *would* have been computed one at a
//! time, that difference is an artefact of the batching:
//!
//!   - **speculative verification.** The target runs over a window of
//!     `depth + 1` rows where the autoregressive path would have run it one row
//!     at a time. A draft token is rejected when the target's argmax disagrees
//!     with it -- so an argmax that moved because the row was batched is a
//!     rejection charged to the draft. mlx-vlm fixes it the same way and for
//!     the same reason: `mlx_vlm/models/quantized_verifier.py`,
//!     `singleton_quantized_linear`, "fuse a runtime-sized batch while
//!     retaining singleton qmv arithmetic".
//!   - **the probe captures.** Phase 19 found that tiling a prefill moves a
//!     capture's KL, on any quantization family, and answered it with a control
//!     experiment rather than at the root. Under this flag the prefill kernel
//!     is not a degree of freedom: `ORNITH_KQUANT_GEMM` and `ORNITH_AFFINE_GEMM`
//!     stop changing the capture because there is only one arithmetic left.
//!
//! **What it covers is the quantized linears, which is most but not all of a
//! forward.** Both families, dense and routed. It does not touch SDPA, the
//! gated-delta recurrence or the bf16 gemms, each of which also picks a kernel
//! by row count; `src/bin/invariance.rs` measures what is left after this flag
//! and says which op it is.
//!
//! The cost is real and is the reason this is not the default: a batch of rows
//! re-reads the whole weight matrix once per row. At a verify window of four
//! rows that is nothing next to a decode step's traffic; at a 2932-token
//! prefill it is the mat-vec prefill `Q4K.md` Phase 19 replaced.
//!
//! **Two settings, because a prefill chunk is not a batch of rows that would
//! otherwise have run alone.** `ORNITH_INVARIANT=1` covers batches up to
//! [`EXACT_WINDOW`] rows -- every width a decode step or a depth-3 verify
//! produces -- and leaves a prefill on the tile Phase 19 built, so the mode
//! costs nothing at a long prompt. `ORNITH_INVARIANT=all` covers every M,
//! which is what the probe captures want and what makes them expensive.
//!
//! [`set`] moves it at runtime, which is how an A/B runs inside one process --
//! the only way to compare two kernels on a machine whose wall clock swings 3x
//! between sittings.

use std::sync::atomic::{AtomicUsize, Ordering};

/// The widest window this model can honour, and it is not a choice: mlx takes
/// the fused vector SDPA only while `q_len * gqa_factor <= 32`, and this model
/// has 16 query heads over 2 kv heads, so a fifth row drops attention into the
/// unfused batched gemm and the promise ends there. `src/bin/invariance.rs`
/// measures exactly that -- windows of 2, 3 and 4 come back bit-identical and
/// a window of 5 first disagrees at layer 3, the first full-attention layer.
///
/// It is also the largest verify window `--mtp-depth 3` produces, which is the
/// depth `mtplx_runtime.json` recommends. The two numbers agreeing is luck,
/// but it is the luck this artifact has.
pub const EXACT_WINDOW: usize = 4;

/// 0 is off; anything else is the widest batch that computes row by row.
static ROWS: AtomicUsize = AtomicUsize::new(UNREAD);
const UNREAD: usize = usize::MAX;

fn rows() -> usize {
    match ROWS.load(Ordering::Relaxed) {
        UNREAD => {
            let n = match std::env::var("ORNITH_INVARIANT").as_deref() {
                Ok("1") => EXACT_WINDOW,
                Ok("all") => usize::MAX - 1,
                Ok("0") | Err(_) => 0,
                Ok(other) => panic!(
                    "ORNITH_INVARIANT={other}: expected 1 (a verify window) or \
                     all (every M, which is what a probe capture wants)"
                ),
            };
            ROWS.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

/// Should an `m`-row product compute its rows one at a time?
pub fn for_rows(m: usize) -> bool {
    m > 1 && m <= rows()
}

/// Is every batch invariant, whatever its width?
///
/// The capture mode. A prefill chunk is not a batch of rows that would
/// otherwise have run alone, so the window mode leaves it on the tile; this
/// one does not, and that is the point -- under it no kernel anywhere depends
/// on a row count, so `ORNITH_KQUANT_GEMM` and `ORNITH_AFFINE_GEMM` cannot
/// move a capture.
pub fn all() -> bool {
    rows() == usize::MAX - 1
}

/// Turn it on or off for everything dispatched after this call.
pub fn set(on: bool) {
    ROWS.store(if on { EXACT_WINDOW } else { 0 }, Ordering::Relaxed);
}

/// Set the widest invariant batch directly; `usize::MAX - 1` is every M.
pub fn set_rows(n: usize) {
    ROWS.store(n, Ordering::Relaxed);
}

/// Run `f` with invariance set to `on`, and restore what was there.
pub fn with<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let was = rows();
    set(on);
    let out = f();
    ROWS.store(was, Ordering::Relaxed);
    out
}

/// [`with`], for an explicit width.
pub fn with_rows<T>(n: usize, f: impl FnOnce() -> T) -> T {
    let was = rows();
    set_rows(n);
    let out = f();
    ROWS.store(was, Ordering::Relaxed);
    out
}
