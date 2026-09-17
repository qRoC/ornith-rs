//! Our quantized_matmul dispatch against mlx's, on the model's real weights.
//!
//! Both sides run the same kernels (mlx's, compiled into our metallib), so a
//! correct dispatch must agree bit for bit -- any difference means we picked a
//! different kernel, grid or argument order than `quantized.cpp` does.
//!
//! Reference: tools/dump_quantized.py

use ornith_rs::array::Array;
use ornith_rs::metal::Context;
use ornith_rs::ops::quantized::quantized_matmul;
use ornith_rs::refs::{RefCase, compare, model_dir};
use ornith_rs::weights::WeightStore;

const ROWS: [usize; 7] = [1, 2, 4, 8, 16, 64, 256];

#[test]
fn matches_mlx_bit_for_bit() {
    let Some(case) = RefCase::open_or_skip("quantized_matmul") else { return };
    let dir = model_dir();
    if !dir.exists() {
        eprintln!("model not present, skipping");
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();

    let mut checked = 0;
    for tag in ["in_proj_qkv", "out_proj", "gate", "expert0"] {
        let path = case
            .meta()
            .and_then(|m| m.path(&format!("{tag}.path")))
            .and_then(|v| v.as_str())
            .unwrap()
            .to_string();
        let mut qw = ws.quant_matrix(&ctx, &path);
        if tag == "expert0" {
            // The reference took expert 0 out of the [256, ...] switch weight.
            qw.w = slice_expert0(&qw.w);
            qw.scales = slice_expert0(&qw.scales);
            qw.biases = slice_expert0(&qw.biases);
        }

        for m in ROWS {
            let x = case.array(&ctx, &format!("{tag}_x_m{m}"));
            let y = quantized_matmul(&ctx, &x, &qw);
            let got = y.to_vec_f32(&ctx);
            let want = case.f32(&format!("{tag}_y_m{m}"));
            let d = compare(&got, &want);
            assert_eq!(
                d.max_abs, 0.0,
                "{tag} M={m}: {d}\n(same kernels on both sides, so any \
                 difference is a dispatch mismatch)"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 4 * ROWS.len());
}

/// A view of expert 0: the expert is the leading dimension and the array is
/// row-contiguous, so expert 0 is the first slab.
fn slice_expert0(a: &Array) -> Array {
    assert_eq!(a.ndim(), 3);
    a.index(0)
}
