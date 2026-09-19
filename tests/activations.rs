//! The sigmoid-family activations over every bfloat16 input.
//!
//! Reference: tools/dump_activations.py
//!
//! mlx has two code paths for the same expression and they do not always
//! agree in the last bit: the precompiled unary kernel (`mx.sigmoid`) and the
//! JIT kernel `mx.compile` builds (`nn.silu`, and mlx-lm's `swiglu` /
//! `precise_swiglu`, which are `@partial(mx.compile, shapeless=True)`). The
//! model uses both, so this test pins both -- exhaustively, because the
//! disagreement shows up on roughly one input in 10^5 and a random sample
//! misses it.

use ornith_rs::metal::Context;
use ornith_rs::ops::elementwise::{precise_swiglu, sigmoid_gate, silu, swiglu};
use ornith_rs::refs::{RefCase, compare};

fn exact(name: &str, got: &[f32], want: &[f32]) {
    let mismatches = got.iter().zip(want).filter(|(a, b)| a != b).count();
    let d = compare(got, want);
    assert_eq!(d.max_abs, 0.0, "{name}: {mismatches} of {} differ, {d}", got.len());
}

#[test]
fn activations_match_mlx_for_every_bfloat16() {
    let Some(case) = RefCase::open_or_skip("activations") else { return };
    let ctx = Context::new().unwrap();
    let x = case.array(&ctx, "x");
    let other = case.array(&ctx, "other");

    // mx.compile'd: nn.silu
    exact("silu", &silu(&ctx, &x).to_vec_f32(&ctx), &case.f32("silu"));
    // mx.compile'd: mlx-lm's swiglu(gate, x) = nn.silu(gate) * x
    exact("swiglu", &swiglu(&ctx, &x, &other).to_vec_f32(&ctx), &case.f32("swiglu"));
    // mx.compile'd, but computed in float32 throughout
    exact(
        "precise_swiglu",
        &precise_swiglu(&ctx, &x, &other).to_vec_f32(&ctx),
        &case.f32("precise_swiglu"),
    );
    // Plain unary sigmoid followed by a multiply, the uncompiled path
    exact(
        "sigmoid * x",
        &sigmoid_gate(&ctx, &x, &other).to_vec_f32(&ctx),
        &case.f32("sigmoid_mul"),
    );
}

/// The same, with the elementwise kernels moving sixteen bytes a thread.
///
/// Every output element is the same expression of the same inputs and a
/// thread doing eight of them evaluates eight independent scalars, so this
/// has to hold to `max_abs == 0.0` against the same reference the narrow
/// form is held to. A width change that rounded anything would show here
/// and nowhere else.
///
/// The case carries 6660 values and a 16-byte access wants a multiple of
/// eight, so the wide arm runs on the first 6656 of them -- the guard
/// declines a ragged tail rather than handling one, and every shape in the
/// model is a multiple of 2048 anyway.
#[test]
fn the_wide_activations_are_the_narrow_ones() {
    let Some(case) = RefCase::open_or_skip("activations") else { return };
    let ctx = Context::new().unwrap();
    let x = case.array(&ctx, "x");
    let other = case.array(&ctx, "other");
    let n = x.numel() - x.numel() % 8;
    let xs = x.reshape(&[x.numel()]).slice_leading(0, n);
    let os = other.reshape(&[other.numel()]).slice_leading(0, n);

    // A green pair means nothing if the guard answered "narrow" to both, so
    // say which path this covers before comparing anything.
    assert!(
        ornith_rs::ops::elementwise::would_widen(&xs, n),
        "the wide path is guarded off here: numel {n}, offset {}",
        xs.offset,
    );
    eprintln!("{n} of {} values, offset {}, the wide path is live", x.numel(), xs.offset);

    let want = &case.f32("swiglu")[..n];
    for wide in [true, false] {
        ornith_rs::ops::elementwise::set_wide_elem(wide);
        exact(&format!("swiglu wide={wide}"), &swiglu(&ctx, &xs, &os).to_vec_f32(&ctx), want);
    }

    // `sigmoid_gate` in the case above runs at `inner = 1`, which the wide
    // path declines -- a thread's sixteen bytes must not straddle two
    // gates. So the broadcast shape the model actually uses is covered
    // separately, against the narrow kernel, which is the one the golden
    // case holds to mlx.
    let inner = 8usize;
    let gate = xs.slice_leading(0, n / inner);
    ornith_rs::ops::elementwise::set_wide_elem(false);
    let narrow = sigmoid_gate(&ctx, &gate, &os).to_vec_f32(&ctx);
    ornith_rs::ops::elementwise::set_wide_elem(true);
    let wide = sigmoid_gate(&ctx, &gate, &os).to_vec_f32(&ctx);
    exact(&format!("sigmoid_gate broadcast inner={inner}"), &wide, &narrow);
    ornith_rs::ops::elementwise::set_wide_elem(true);
}
