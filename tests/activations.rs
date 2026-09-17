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
