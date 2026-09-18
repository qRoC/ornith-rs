//! `generate_step` from `vendor/mlx-lm/mlx_lm/generate.py`.
//!
//! The shape of the loop matters for more than speed:
//!
//! ```python
//! while total_prompt_tokens - prompt_processed_tokens > 1:
//!     n_to_process = min(prefill_step_size, remaining)
//!     _model_call(prompt[:n_to_process][None])      # logits never evaluated
//!     ...
//! y, logprobs = _step(input_tokens=prompt)          # exactly one token left
//! ```
//!
//! so the prompt is consumed in 2048-token chunks *except* its last token,
//! and the step that produces logits is always a single-token step. That
//! keeps `lm_head` on the `qmv` path rather than the split-K gemm, and it is
//! why the prefill chunks can skip `lm_head` entirely: mlx is lazy and only
//! ever evaluates the caches for them.

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::model::cache::LayerCache;
use crate::model::qwen3_5::TextModel;
use crate::vision::splice::{VisionSplice, spliced_chunk_embeddings};

pub const DEFAULT_PREFILL_STEP_SIZE: usize = 2048;

/// `ORNITH_PREFILL` overrides it, for measuring what the chunk width costs.
pub fn prefill_step_size() -> usize {
    std::env::var("ORNITH_PREFILL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PREFILL_STEP_SIZE)
}

/// Picks the next token id from the logits of the last position.
pub trait Sampler {
    fn sample(&mut self, ctx: &Context, logits: &Array) -> u32;

    /// Whether a row's token depends on nothing but that row.
    ///
    /// A stochastic sampler draws from a stream, so how many rows it is asked
    /// for -- and in what order -- decides what it returns. The speculative
    /// loop therefore has to ask row by row and stop at the first mismatch,
    /// which costs a round trip to the host per row for one argmax's worth of
    /// work. `Greedy` has no stream, so the loop can take every row in one
    /// pass and one round trip. Conservative by default: a sampler that does
    /// not say otherwise is treated as having a stream.
    fn rows_are_independent(&self) -> bool {
        false
    }
}

/// `greedy_sampler`: `mx.argmax(logprobs, axis=-1)`.
pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&mut self, ctx: &Context, logits: &Array) -> u32 {
        crate::ops::sample::greedy_token(ctx, logits)
    }

    fn rows_are_independent(&self) -> bool {
        true
    }
}

/// One model call over `tokens`, with the image rows spliced in when the
/// chunk holds `<|image_pad|>`.
///
/// `spliced_chunk_embeddings` returns `None` for a chunk with no pads, which
/// is the plain token-id path and the only one a text prompt ever takes.
pub fn model_call(
    ctx: &Context,
    model: &TextModel,
    caches: &mut [LayerCache],
    tokens: &[u32],
    splice: Option<&mut VisionSplice>,
) -> (Array, Array) {
    let ids = Array::from_slice(ctx, tokens, &[tokens.len()], Dtype::U32);
    let embedded = splice
        .and_then(|s| spliced_chunk_embeddings(ctx, &model.embed_tokens, tokens, &ids, s));
    match embedded {
        Some(h) => model.forward_from_embeddings(ctx, &h, caches),
        None => model.forward_hidden(ctx, &ids, caches),
    }
}

/// Run the prompt through the model, leaving the last token unconsumed.
///
/// Returns nothing: the point is the state left in `caches`.
pub fn prefill(
    ctx: &Context,
    model: &TextModel,
    caches: &mut [LayerCache],
    prompt: &[u32],
    prefill_step_size: usize,
    mut splice: Option<&mut VisionSplice>,
    mut progress: impl FnMut(usize, usize),
) -> usize {
    let total = prompt.len();
    let mut processed = 0;
    progress(0, total);
    while total - processed > 1 {
        let n = prefill_step_size.min(total - processed - 1);
        model_call(
            ctx,
            model,
            caches,
            &prompt[processed..processed + n],
            splice.as_deref_mut(),
        );
        // mlx evaluates the caches here and nothing else; the equivalent is to
        // let the queue drain so the next chunk's host writes are safe.
        ctx.sync();
        processed += n;
        progress(processed, total);
    }
    processed
}

/// One `_step`: run `tokens` and sample from the last position's logits.
pub fn step(
    ctx: &Context,
    model: &TextModel,
    caches: &mut [LayerCache],
    tokens: &[u32],
    sampler: &mut dyn Sampler,
    splice: Option<&mut VisionSplice>,
) -> u32 {
    let (h, _) = model_call(ctx, model, caches, tokens, splice);
    let logits = model.lm_head.matmul(ctx, &h);
    sampler.sample(ctx, &logits)
}

/// `generate_step`: yields token ids until `max_tokens` or `on_token` stops.
///
/// `on_token` returns false to stop, which is where the caller puts the
/// end-of-sequence check -- mlx-lm does the same, in `stream_generate`.
///
/// The step for a token is taken *before* that token is handed out, exactly
/// as mlx-lm pipelines it (`next_y = _step(y)` sits above the `yield`). So
/// when this returns, the caches hold the prompt plus every token `on_token`
/// was shown, the stopping one included.
#[allow(clippy::too_many_arguments)]
pub fn generate_step(
    ctx: &Context,
    model: &TextModel,
    caches: &mut [LayerCache],
    prompt: &[u32],
    sampler: &mut dyn Sampler,
    max_tokens: usize,
    prefill_step_size: usize,
    mut splice: Option<&mut VisionSplice>,
    mut progress: impl FnMut(usize, usize),
    mut on_token: impl FnMut(u32) -> bool,
) {
    assert!(!prompt.is_empty(), "generate_step needs a prompt");
    let processed = prefill(
        ctx,
        model,
        caches,
        prompt,
        prefill_step_size,
        splice.as_deref_mut(),
        &mut progress,
    );
    let mut y = step(
        ctx,
        model,
        caches,
        &prompt[processed..],
        sampler,
        splice.as_deref_mut(),
    );
    progress(prompt.len(), prompt.len());

    let mut n = 0;
    loop {
        // A generated token is never an image pad, so the decode loop keeps
        // the plain token-id path.
        let next = (n != max_tokens).then(|| step(ctx, model, caches, &[y], sampler, None));
        let Some(next) = next else { return };
        if !on_token(y) {
            return;
        }
        y = next;
        n += 1;
    }
}
