//! rms_norm, rope and the fused elementwise ops against mlx.
//!
//! Reference: tools/dump_ops.py

use ornith_rs::metal::Context;
use ornith_rs::ops::elementwise::{add, precise_swiglu, sigmoid_gate, swiglu};
use ornith_rs::ops::norm::rms_norm;
use ornith_rs::ops::rope::rope;
use ornith_rs::refs::{RefCase, compare};

fn exact(name: &str, got: &[f32], want: &[f32]) {
    let d = compare(got, want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

#[test]
fn rms_norm_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("ops") else { return };
    let ctx = Context::new().unwrap();
    let eps = case.meta().unwrap().get("eps").unwrap().as_f32().unwrap();

    let w = case.array(&ctx, "ln_weight");
    for t in [1, 7, 512] {
        let x = case.array(&ctx, &format!("rms_x_t{t}"));
        let y = rms_norm(&ctx, &x, Some(&w), eps);
        exact(&format!("rms T={t}"), &y.to_vec_f32(&ctx), &case.f32(&format!("rms_y_t{t}")));
    }

    // GatedDeltaNet normalises q and k with no weight: mlx passes a scalar 1
    // and a zero stride rather than taking a separate code path.
    for t in [1, 7] {
        let x = case.array(&ctx, &format!("rms_nw_x_t{t}"));
        let y = rms_norm(&ctx, &x, None, 1e-6);
        exact(&format!("rms weightless T={t}"), &y.to_vec_f32(&ctx), &case.f32(&format!("rms_nw_y_t{t}")));
    }

    // q_norm / k_norm run over head_dim = 256, which is under the looped
    // limit but wide enough to need 2 simdgroups.
    let qn = case.array(&ctx, "q_norm_weight");
    for t in [1, 7] {
        let x = case.array(&ctx, &format!("rms_hd_x_t{t}"));
        let y = rms_norm(&ctx, &x, Some(&qn), eps);
        exact(&format!("rms head_dim T={t}"), &y.to_vec_f32(&ctx), &case.f32(&format!("rms_hd_y_t{t}")));
    }
}

#[test]
fn rope_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("ops") else { return };
    let ctx = Context::new().unwrap();
    let meta = case.meta().unwrap();
    let dims = meta.get("rope_dims").unwrap().as_usize().unwrap();
    let theta = meta.get("rope_theta").unwrap().as_f32().unwrap();

    for (t, off) in [(1, 0), (1, 37), (7, 0), (64, 128)] {
        let x = case.array(&ctx, &format!("rope_x_t{t}_o{off}"));
        let y = rope(&ctx, &x, dims, theta, 1.0, off);
        let want = case.f32(&format!("rope_y_t{t}_o{off}"));
        exact(&format!("rope T={t} offset={off}"), &y.to_vec_f32(&ctx), &want);
    }
}

#[test]
fn elementwise_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("ops") else { return };
    let ctx = Context::new().unwrap();
    let a = case.array(&ctx, "ew_a");
    let b = case.array(&ctx, "ew_b");

    exact("swiglu", &swiglu(&ctx, &a, &b).to_vec_f32(&ctx), &case.f32("ew_swiglu"));
    exact(
        "precise_swiglu",
        &precise_swiglu(&ctx, &b, &a).to_vec_f32(&ctx),
        &case.f32("ew_precise_swiglu"),
    );
    exact("add", &add(&ctx, &a, &b).to_vec_f32(&ctx), &case.f32("ew_add"));

    let g = case.array(&ctx, "ew_gate");
    exact(
        "sigmoid_gate elementwise",
        &sigmoid_gate(&ctx, &g, &b).to_vec_f32(&ctx),
        &case.f32("ew_sigmoid_gate"),
    );
    let g1 = case.array(&ctx, "ew_gate1");
    exact(
        "sigmoid_gate broadcast",
        &sigmoid_gate(&ctx, &g1, &b).to_vec_f32(&ctx),
        &case.f32("ew_sigmoid_gate1"),
    );
}

/// The two transposes at sixteen bytes a thread, against the permutation
/// they are and against the one-element kernel they replace.
///
/// A transpose here swaps `L` and `H`, both of them outside `D`, so it is a
/// permutation of contiguous `D`-runs and a vector access needs only a `D`
/// that divides -- see `kernels/layout.metal`. Two references, because a
/// wide/narrow pair alone would not catch a *shared* addressing mistake:
///
///   * float32 at distinct values, against the permutation computed here.
///     Every index is exactly representable, so no two source positions
///     hold the same number and a misaddressed element cannot look right;
///   * bfloat16 at the model's own head shape, wide against narrow, which
///     is where the guard is actually live.
///
/// And the guard gets its own check in both directions, because a green
/// pair means nothing if the wide path never ran.
#[test]
fn the_wide_transposes_are_the_narrow_ones() {
    use ornith_rs::array::{Array, Dtype};
    use ornith_rs::ops::elementwise::set_wide_elem;
    use ornith_rs::ops::layout::{transpose_hl, transpose_lh, transpose_would_widen};

    let ctx = Context::new().unwrap();

    let (b, l, h, d) = (2usize, 5usize, 3usize, 8usize);
    let n = b * l * h * d;
    let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = Array::from_f32(&ctx, &src, &[b, l, h, d]);
    assert!(
        transpose_would_widen(&x),
        "the wide path is guarded off here: d {d}, dtype {}, offset {}",
        x.dtype.name(),
        x.offset,
    );
    eprintln!("[{b}, {l}, {h}, {d}] {}, the wide path is live", x.dtype.name());

    let mut want_lh = vec![0f32; n];
    for bb in 0..b {
        for hh in 0..h {
            for ll in 0..l {
                for dd in 0..d {
                    want_lh[((bb * h + hh) * l + ll) * d + dd] =
                        src[((bb * l + ll) * h + hh) * d + dd];
                }
            }
        }
    }
    // `transpose_hl` reads the same buffer as [B, H, L, D], so on this shape
    // its H is `l` and its L is `h` -- the axes are positions, not names.
    let mut want_hl = vec![0f32; n];
    for bb in 0..b {
        for hh in 0..l {
            for ll in 0..h {
                for dd in 0..d {
                    want_hl[((bb * h + ll) * l + hh) * d + dd] =
                        src[((bb * l + hh) * h + ll) * d + dd];
                }
            }
        }
    }
    for wide in [true, false] {
        set_wide_elem(wide);
        let got = transpose_lh(&ctx, &x).to_vec_f32(&ctx);
        exact(&format!("transpose_lh wide={wide}"), &got, &want_lh);
        let got = transpose_hl(&ctx, &x).to_vec_f32(&ctx);
        exact(&format!("transpose_hl wide={wide}"), &got, &want_hl);
        // And the round trip, which is the identity for any correct pair.
        let back = transpose_hl(&ctx, &transpose_lh(&ctx, &x)).to_vec_f32(&ctx);
        exact(&format!("lh then hl wide={wide}"), &back, &src);
    }
    set_wide_elem(true);

    // The model's own shape: 16 query heads of 256, bfloat16, a 512-byte run.
    let (l, h, d) = (64usize, 16usize, 256usize);
    let n = l * h * d;
    let vals: Vec<f32> = (0..n).map(|i| (i % 251) as f32).collect();
    let x = Array::from_f32_as_bf16(&ctx, &vals, &[1, l, h, d]);
    assert!(transpose_would_widen(&x), "the head shape should widen");
    set_wide_elem(false);
    let narrow_lh = transpose_lh(&ctx, &x).to_vec_f32(&ctx);
    let narrow_hl = transpose_hl(&ctx, &x).to_vec_f32(&ctx);
    set_wide_elem(true);
    exact("transpose_lh bf16 head shape", &transpose_lh(&ctx, &x).to_vec_f32(&ctx), &narrow_lh);
    exact("transpose_hl bf16 head shape", &transpose_hl(&ctx, &x).to_vec_f32(&ctx), &narrow_hl);

    // A `d` that is not a whole number of vectors: the guard has to decline,
    // and the narrow kernel has to keep being right when it does.
    let (l, h, d) = (3usize, 2usize, 6usize);
    let n = l * h * d;
    let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = Array::from_f32_as_bf16(&ctx, &src, &[1, l, h, d]);
    assert_eq!(x.dtype, Dtype::BF16);
    assert!(!transpose_would_widen(&x), "d = {d} bfloat16 is not a whole number of vectors");
    let mut want = vec![0f32; n];
    for hh in 0..h {
        for ll in 0..l {
            for dd in 0..d {
                want[(hh * l + ll) * d + dd] = src[(ll * h + hh) * d + dd];
            }
        }
    }
    exact("transpose_lh ragged d", &transpose_lh(&ctx, &x).to_vec_f32(&ctx), &want);
}
