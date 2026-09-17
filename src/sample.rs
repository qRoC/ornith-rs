//! `make_sampler` from `vendor/mlx-lm/mlx_lm/sample_utils.py`, for the
//! settings `mtplx_runtime.json` recommends: temperature 0.6, top_p 0.95,
//! top_k 20.
//!
//! ```python
//! sampling_methods = []
//! if 0 < top_p < 1.0:  sampling_methods.append(lambda x: apply_top_p(x, top_p))
//! if top_k > 0:        sampling_methods.append(lambda x: apply_top_k(x, top_k))
//!
//! def sampler(logprobs):
//!     for method in sampling_methods:
//!         logprobs = method(logprobs)
//!     return categorical_sampling(logprobs, temp)
//! ```
//!
//! The order matters: top-p first, then top-k over what top-p left, then the
//! draw. Everything runs in the logits' own dtype -- bfloat16 -- because mlx
//! treats the Python scalars (`1 - top_p`, `1.0 / temp`, `-inf`) as weak types
//! and casts them down to the array's dtype rather than promoting the array.
//! The one place that promotes is `categorical`, which computes its cumulative
//! distribution in float32.
//!
//! The randomness comes from mlx's global key sequence. Each call to the
//! compiled `categorical_sampling` consumes one `split` of it, because
//! `mx.random.categorical` with no key calls `KeySequence::default_().next()`
//! and the `inputs=mx.random.state, outputs=mx.random.state` decoration
//! threads the update back out. `apply_top_p` and `apply_top_k` carry the same
//! decoration but draw nothing, so they leave the state alone.

use crate::array::{Array, Dtype, f32_to_bf16};
use crate::generate::Sampler;
use crate::metal::Context;
use crate::ops::indexing::{put_along_last_axis, take_along_last_axis};
use crate::ops::random::KeySequence;
use crate::ops::reduce::all_reduce;
use crate::ops::sample::log_softmax_last_row;
use crate::ops::scan::cumsum_last_axis;
use crate::ops::searchsorted::searchsorted;
use crate::ops::sort::argsort_last_axis;

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        Dtype::Bool => "bool_",
        d => panic!("no sampler kernel for {}", d.name()),
    }
}

fn flat(n: usize) -> ((usize, usize, usize), (usize, usize, usize)) {
    ((n, 1, 1), (1024.min(n).max(1), 1, 1))
}

/// A Python float landing in a binary op with a bfloat16 array: mlx's
/// `to_arrays` casts it to the array's dtype, so it is rounded to bfloat16
/// *before* the op rather than promoting the array to float32.
fn weak_bf16(v: f64) -> u16 {
    f32_to_bf16(v as f32)
}

fn unary(ctx: &Context, kernel: &str, x: &Array, out_dtype: Dtype) -> Array {
    let out = Array::empty(ctx, &x.shape, out_dtype);
    let (g, t) = flat(x.numel());
    ctx.kernel(&format!("{kernel}_{}", tname(x.dtype))).arr(x).arr(&out).threads(g, t);
    out
}

fn filled(ctx: &Context, shape: &[usize], dtype: Dtype, bits: &[u8]) -> Array {
    let out = Array::empty(ctx, shape, dtype);
    let (g, t) = flat(out.numel());
    ctx.kernel(&format!("sampler_fill_{}", tname(dtype)))
        .arr(&out)
        .bytes_slice(bits)
        .threads(g, t);
    out
}

/// `apply_top_p(logprobs, top_p)`.
pub fn apply_top_p(ctx: &Context, logprobs: &Array, top_p: f64) -> Array {
    let n = logprobs.dim(-1);
    let probs = unary(ctx, "sampler_exp", logprobs, logprobs.dtype);
    let sorted_indices = argsort_last_axis(ctx, logprobs);
    let sorted_probs = take_along_last_axis(ctx, &probs, &sorted_indices);
    let cumulative = cumsum_last_axis(ctx, &sorted_probs, true);

    // sorted_keep = cumulative_probs > 1 - top_p
    let thresh = weak_bf16(1.0 - top_p);
    let sorted_keep = Array::empty(ctx, &logprobs.shape, Dtype::Bool);
    let (g, t) = flat(n);
    ctx.kernel(&format!("sampler_greater_scalar_{}", tname(logprobs.dtype)))
        .arr(&cumulative)
        .arr(&sorted_keep)
        .bytes(&thresh)
        .threads(g, t);

    // keep = put_along_axis(zeros_like(sorted_keep), sorted_indices, sorted_keep)
    let zeros = filled(ctx, &logprobs.shape, Dtype::Bool, &[0u8]);
    let keep = put_along_last_axis(ctx, &zeros, &sorted_indices, &sorted_keep, true);

    // mx.where(keep, logprobs, -inf)
    let neg_inf = weak_bf16(f64::NEG_INFINITY);
    let out = Array::empty(ctx, &logprobs.shape, logprobs.dtype);
    ctx.kernel(&format!("sampler_select_scalar_{}", tname(logprobs.dtype)))
        .arr(&keep)
        .arr(logprobs)
        .arr(&out)
        .bytes(&neg_inf)
        .threads(g, t);
    out
}

/// `apply_top_k(logprobs, top_k)`.
///
/// `mx.argpartition(-logprobs, kth=top_k - 1)[..., top_k:]` is a full
/// ascending argsort of the negated logprobs -- mlx has no partition kernel --
/// so the slice is every index *except* the `top_k` largest, and those get
/// `-inf` scattered into them.
pub fn apply_top_k(ctx: &Context, logprobs: &Array, top_k: usize) -> Array {
    let n = logprobs.dim(-1);
    assert!(top_k > 0 && top_k < n, "top_k must be in (0, {n})");
    let negated = unary(ctx, "sampler_negative", logprobs, logprobs.dtype);
    let order = argsort_last_axis(ctx, &negated);

    // `[..., top_k:]`: with one row this is a contiguous view.
    assert_eq!(order.shape[0], 1, "the sampler only ever sees a single row");
    let mask_idx = order.reshape(&[n]).slice_last(top_k, n - top_k).reshape(&[1, n - top_k]);

    let neg_inf = filled(ctx, &[1], logprobs.dtype, &weak_bf16(f64::NEG_INFINITY).to_le_bytes());
    put_along_last_axis(ctx, logprobs, &mask_idx, &neg_inf, false)
}

/// `categorical_sampling(logits, temp)`: `mx.random.categorical(logits / temp)`
/// down the `categorical_inverse_cdf` path, which is the one a single row of
/// logits takes.
pub fn categorical_sampling(
    ctx: &Context,
    keys: &mut KeySequence,
    logits: &Array,
    temp: f64,
) -> Array {
    let n = logits.dim(-1);

    // logits * (1.0 / temp)
    let scale = weak_bf16(1.0 / temp);
    let scaled = Array::empty(ctx, &logits.shape, logits.dtype);
    let (g, t) = flat(n);
    ctx.kernel(&format!("sampler_mul_scalar_{}", tname(logits.dtype)))
        .arr(logits)
        .arr(&scaled)
        .bytes(&scale)
        .threads(g, t);

    // categorical_inverse_cdf(reshape(logits, {n}), shape={1}, key)
    let x = unary(ctx, "sampler_to_float", &scaled.reshape(&[n]), Dtype::F32);
    let m = all_reduce(ctx, &x, "max");

    let w = Array::empty(ctx, &[n], Dtype::F32);
    ctx.kernel("sampler_cdf_weights").arr(&x).arr(&m).arr(&w).threads(g, t);

    let cdf = cumsum_last_axis(ctx, &w, false);
    let total = all_reduce(ctx, &w, "sum");

    // u = uniform({1}, float32, key) * sum(w)
    let sub = keys.next(ctx);
    let raw = crate::ops::random::bits(ctx, &sub, &[1]);
    let u = Array::empty(ctx, &[1], Dtype::F32);
    // `array(std::numeric_limits<uint32_t>::max(), float32)` is 2^32 exactly
    // once it is a float, and `upper` is `nextafter(1.0f, 0.0f)`.
    let maxval = u32::MAX as f32;
    let upper = f32::from_bits(0x3f7f_ffff);
    ctx.kernel("sampler_uniform_scaled")
        .arr(&raw)
        .arr(&total)
        .arr(&u)
        .bytes(&maxval)
        .bytes(&upper)
        .threads((1, 1, 1), (1, 1, 1));

    let idx = searchsorted(ctx, &cdf, &u, true);
    let out = Array::empty(ctx, &[1], Dtype::U32);
    ctx.kernel("sampler_minus_one").arr(&idx).arr(&out).threads((1, 1, 1), (1, 1, 1));
    out
}

/// The whole chain, as `make_sampler(temp, top_p, top_k)` composes it.
pub fn sample(
    ctx: &Context,
    keys: &mut KeySequence,
    logprobs: &Array,
    temp: f64,
    top_p: f64,
    top_k: usize,
) -> u32 {
    let mut x = logprobs.clone();
    if top_p > 0.0 && top_p < 1.0 {
        x = apply_top_p(ctx, &x, top_p);
    }
    if top_k > 0 {
        x = apply_top_k(ctx, &x, top_k);
    }
    categorical_sampling(ctx, keys, &x, temp).to_vec_u32(ctx)[0]
}

/// The `Sampler` the generation loop drives.
pub struct TopSampler {
    pub temp: f64,
    pub top_p: f64,
    pub top_k: usize,
    keys: Option<KeySequence>,
    seed: u64,
}

impl TopSampler {
    pub fn new(temp: f64, top_p: f64, top_k: usize, seed: u64) -> TopSampler {
        TopSampler { temp, top_p, top_k, keys: None, seed }
    }
}

impl Sampler for TopSampler {
    fn sample(&mut self, ctx: &Context, logits: &Array) -> u32 {
        let logprobs = log_softmax_last_row(ctx, logits);
        let keys = self.keys.get_or_insert_with(|| KeySequence::seed(ctx, self.seed));
        sample(ctx, keys, &logprobs, self.temp, self.top_p, self.top_k)
    }
}

/// `mx.random.uniform(shape, float32, key)` on its own, so the test can pin
/// the scaling that turns threefry bits into a number in [0, 1).
pub fn uniform_for_test(ctx: &Context, raw: &Array) -> Array {
    let one = Array::from_f32(ctx, &[1.0], &[1]);
    let u = Array::empty(ctx, &raw.shape, Dtype::F32);
    ctx.kernel("sampler_uniform_scaled")
        .arr(raw)
        .arr(&one)
        .arr(&u)
        .bytes(&(u32::MAX as f32))
        .bytes(&f32::from_bits(0x3f7f_ffff))
        .threads((raw.numel(), 1, 1), (1, 1, 1));
    u
}
