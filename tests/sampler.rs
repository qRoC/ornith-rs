//! The sampler against mlx-lm's, stage by stage.
//!
//! Reference: tools/dump_sampler.py
//!
//! The pieces here are the ones the model never needed: the multi-block merge
//! sort, the scan, whole-array reductions, gather/scatter along an axis,
//! searchsorted, and threefry. Each is checked on its own before the composed
//! draw, so a failure names the kernel.

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::ops::indexing::{put_along_last_axis, take_along_last_axis};
use ornith_rs::ops::random::{KeySequence, bits};
use ornith_rs::ops::reduce::all_reduce;
use ornith_rs::ops::sample::log_softmax_last_row;
use ornith_rs::ops::scan::cumsum_last_axis;
use ornith_rs::ops::searchsorted::searchsorted;
use ornith_rs::ops::sort::argsort_last_axis;
use ornith_rs::refs::{RefCase, compare};
use ornith_rs::sample::{apply_top_k, apply_top_p, sample};

fn check(case: &RefCase, ctx: &Context, name: &str, got: &Array) {
    let want = case.f32(name);
    let d = compare(&got.to_vec_f32(ctx), &want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

#[test]
fn stages_match_mlx() {
    let Some(case) = RefCase::open_or_skip("sampler") else { return };
    let ctx = Context::new().unwrap();

    let logits = case.array(&ctx, "logits");
    let logprobs = log_softmax_last_row(&ctx, &logits);
    check(&case, &ctx, "logprobs", &logprobs);

    let n = logprobs.dim(-1);
    let (bn, tn, n_blocks) = ornith_rs::ops::sort::sort_tiles(n, Dtype::BF16);
    assert_eq!((bn, tn), (512, 4), "gpu_merge_sort picked a different tile");
    assert!(n_blocks > 1, "a {n}-wide sort should be multi-block");

    // apply_top_p, stage by stage
    let exp = ornith_rs::ops::sample::exp_for_test(&ctx, &logprobs);
    check(&case, &ctx, "exp", &exp);
    let order = argsort_last_axis(&ctx, &logprobs);
    check(&case, &ctx, "argsort", &order);
    let gathered = take_along_last_axis(&ctx, &exp, &order);
    check(&case, &ctx, "gathered", &gathered);
    let cum = cumsum_last_axis(&ctx, &gathered, true);
    check(&case, &ctx, "cumsum", &cum);
    let top_p = apply_top_p(&ctx, &logprobs, 0.95);
    check(&case, &ctx, "top_p", &top_p);

    // apply_top_k
    let negated = ornith_rs::ops::sample::negate_for_test(&ctx, &top_p);
    check(&case, &ctx, "negated", &negated);
    check(&case, &ctx, "argsort_k", &argsort_last_axis(&ctx, &negated));
    let top_k = apply_top_k(&ctx, &top_p, 20);
    check(&case, &ctx, "top_k", &top_k);

    // categorical_inverse_cdf
    let x = case.array(&ctx, "x");
    check(&case, &ctx, "maxval", &all_reduce(&ctx, &x, "max"));
    let w = case.array(&ctx, "weights");
    check(&case, &ctx, "total", &all_reduce(&ctx, &w, "sum"));
    check(&case, &ctx, "cdf", &cumsum_last_axis(&ctx, &w, false));
}

#[test]
fn threefry_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("sampler") else { return };
    let ctx = Context::new().unwrap();
    let seed = case.meta_usize("seed") as u64;

    let want_keys = case.get("rng_keys").to_u32();
    let want_subs = case.get("rng_subs").to_u32();
    let want_uniform = case.f32("rng_uniform");

    let mut ks = KeySequence::seed(&ctx, seed);
    assert_eq!(ks.key.to_vec_u32(&ctx), want_keys[0..2].to_vec(), "seeded key");
    for i in 0..want_subs.len() / 2 {
        let sub = ks.next(&ctx);
        assert_eq!(sub.to_vec_u32(&ctx), want_subs[2 * i..2 * i + 2].to_vec(), "sub key {i}");
        assert_eq!(
            ks.key.to_vec_u32(&ctx),
            want_keys[2 * (i + 1)..2 * (i + 1) + 2].to_vec(),
            "key {} after the split",
            i + 1
        );
        // `mx.random.uniform(shape=(1,), key=sub)` is one `bits` draw scaled.
        let raw = bits(&ctx, &sub, &[1]);
        let u = ornith_rs::sample::uniform_for_test(&ctx, &raw);
        assert_eq!(u.to_vec_f32(&ctx)[0], want_uniform[i], "uniform {i}");
    }
}

#[test]
fn searchsorted_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("sampler") else { return };
    let ctx = Context::new().unwrap();
    // The cdf is exclusive and non-decreasing, so a query at each of a spread
    // of points exercises both the ties and the gaps.
    let cdf = case.array(&ctx, "cdf");
    let total = case.f32("total")[0];
    let host = cdf.to_vec_f32(&ctx);
    let queries: Vec<f32> = (0..64).map(|i| total * (i as f32) / 63.0).collect();
    let q = Array::from_f32(&ctx, &queries, &[queries.len()]);
    let got = searchsorted(&ctx, &cdf, &q, true).to_vec_u32(&ctx);
    for (i, v) in queries.iter().enumerate() {
        // `side="right"`: the first index whose value is strictly greater.
        let want = host.partition_point(|&x| x <= *v) as u32;
        assert_eq!(got[i], want, "searchsorted at {v}");
    }
}

#[test]
fn scatter_along_axis_inverts_the_sort() {
    let Some(case) = RefCase::open_or_skip("sampler") else { return };
    let ctx = Context::new().unwrap();
    // Scattering a sorted row back through its own argsort has to reproduce
    // the row, which pins both the permutation and the scatter's addressing.
    let logprobs = log_softmax_last_row(&ctx, &case.array(&ctx, "logits"));
    let n = logprobs.dim(-1);
    let order = argsort_last_axis(&ctx, &logprobs);
    let sorted = take_along_last_axis(&ctx, &logprobs, &order);
    let zeros = Array::from_f32(&ctx, &vec![0.0f32; n], &[1, n]);
    let sorted_f32 = Array::from_f32(&ctx, &sorted.to_vec_f32(&ctx), &[1, n]);
    let back = put_along_last_axis(&ctx, &zeros, &order, &sorted_f32, true);
    let d = compare(&back.to_vec_f32(&ctx), &logprobs.to_vec_f32(&ctx));
    assert_eq!(d.max_abs, 0.0, "scatter through argsort: {d}");

    // The broadcast-update variant, which `apply_top_k` uses to write one
    // scalar into every masked position.
    let one = Array::from_f32(&ctx, &[7.0], &[1]);
    let idx = Array::from_slice(&ctx, &[3u32, 9, 17], &[1, 3], Dtype::U32);
    let out = put_along_last_axis(&ctx, &zeros, &idx, &one, false);
    let got = out.to_vec_f32(&ctx);
    for i in 0..n {
        let want = if [3, 9, 17].contains(&i) { 7.0 } else { 0.0 };
        assert_eq!(got[i], want, "broadcast scatter at {i}");
    }
}

#[test]
fn draws_match_mlx() {
    let Some(case) = RefCase::open_or_skip("sampler") else { return };
    let ctx = Context::new().unwrap();
    let seed = case.meta_usize("seed") as u64;
    let n_draws = case.meta_usize("n_draws");

    for (logprob_name, token_name) in
        [("logprobs", "tokens"), ("flat_logprobs", "flat_tokens")]
    {
        let logprobs = if logprob_name == "logprobs" {
            log_softmax_last_row(&ctx, &case.array(&ctx, "logits"))
        } else {
            case.array(&ctx, logprob_name)
        };
        if logprob_name == "flat_logprobs" {
            check(&case, &ctx, "flat_top_p", &apply_top_p(&ctx, &logprobs, 0.95));
            let tp = apply_top_p(&ctx, &logprobs, 0.95);
            check(&case, &ctx, "flat_top_k", &apply_top_k(&ctx, &tp, 20));
        }
        let want = case.get(token_name).to_u32();
        let mut keys = KeySequence::seed(&ctx, seed);
        let got: Vec<u32> =
            (0..n_draws).map(|_| sample(&ctx, &mut keys, &logprobs, 0.6, 0.95, 20)).collect();
        assert_eq!(got, want, "{token_name}");
    }
}
