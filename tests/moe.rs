//! The sparse MoE block against mlx-lm, on layer 0's real weights.
//!
//! Reference: tools/dump_moe.py

use ornith_rs::array::Array;
use ornith_rs::metal::Context;
use ornith_rs::ops::moe::{normalize_scores, sparse_moe_block, switch_glu, top_k};
use ornith_rs::ops::quantized::quantized_matmul;
use ornith_rs::ops::softmax::softmax;
use ornith_rs::refs::{RefCase, compare, model_dir};
use ornith_rs::weights::WeightStore;

const TS: [usize; 4] = [1, 7, 24, 512];
const P: &str = "language_model.model.layers.0.mlp";

fn exact(name: &str, got: &[f32], want: &[f32]) {
    let d = compare(got, want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

#[test]
fn router_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let gate = ws.quant_matrix(&ctx, &format!("{P}.gate"));
    let k = case.meta_usize("top_k");

    for t in TS {
        let x = case.array(&ctx, &format!("x_t{t}"));
        let x2 = x.reshape(&[t, x.dim(-1)]);
        let gates = softmax(&ctx, &quantized_matmul(&ctx, &x2, &gate), true);
        exact(&format!("router gates T={t}"), &gates.to_vec_f32(&ctx), &case.f32(&format!("gates_t{t}")));

        let (inds, scores) = top_k(&ctx, &gates, k);
        // The expert *set* and its order both matter: the order fixes the
        // summation order of the mixture below.
        let got: Vec<u32> = inds.to_vec_u32(&ctx);
        let want: Vec<u32> = case.get(&format!("inds_t{t}")).to_u32();
        assert_eq!(got, want, "top-k indices T={t}");
        exact(&format!("scores raw T={t}"), &scores.to_vec_f32(&ctx), &case.f32(&format!("scores_raw_t{t}")));

        let norm = normalize_scores(&ctx, &scores);
        exact(&format!("scores norm T={t}"), &norm.to_vec_f32(&ctx), &case.f32(&format!("scores_t{t}")));
    }
}

#[test]
fn switch_glu_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let g = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.gate_proj"));
    let u = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let dn = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.down_proj"));
    let k = case.meta_usize("top_k");

    for t in TS {
        let x = case.array(&ctx, &format!("x_t{t}"));
        let x2 = x.reshape(&[t, x.dim(-1)]);
        let inds = Array::from_slice(
            &ctx,
            &case.get(&format!("inds_t{t}")).to_u32(),
            &[t, k],
            ornith_rs::array::Dtype::U32,
        );
        let y = switch_glu(&ctx, &x2, &inds, &g, &u, &dn);
        exact(
            &format!("switch_glu T={t} ({})", if t * k >= 64 { "sorted/gather_qmm_rhs" } else { "gather_qmv" }),
            &y.to_vec_f32(&ctx),
            &case.f32(&format!("switch_out_t{t}")),
        );
    }
}

#[test]
fn whole_block_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let gate = ws.quant_matrix(&ctx, &format!("{P}.gate"));
    let sg = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.gate_proj"));
    let su = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let sd = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.down_proj"));
    let hg = ws.quant_matrix(&ctx, &format!("{P}.shared_expert.gate_proj"));
    let hu = ws.quant_matrix(&ctx, &format!("{P}.shared_expert.up_proj"));
    let hd = ws.quant_matrix(&ctx, &format!("{P}.shared_expert.down_proj"));
    let seg = ws.quant_matrix(&ctx, &format!("{P}.shared_expert_gate"));
    let k = case.meta_usize("top_k");

    for t in TS {
        let x = case.array(&ctx, &format!("x_t{t}"));
        let out = sparse_moe_block(
            &ctx, &x, &gate, &sg, &su, &sd, &hg, &hu, &hd, &seg, k, true,
        );
        exact(&format!("moe block T={t}"), &out.to_vec_f32(&ctx), &case.f32(&format!("out_t{t}")));
    }
}

/// Split the block into its parts so a mismatch points at one of them.
#[test]
fn mixture_and_shared_expert_match_mlx() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let k = case.meta_usize("top_k");
    let hg = ws.quant_matrix(&ctx, &format!("{P}.shared_expert.gate_proj"));
    let hu = ws.quant_matrix(&ctx, &format!("{P}.shared_expert.up_proj"));
    let hd = ws.quant_matrix(&ctx, &format!("{P}.shared_expert.down_proj"));
    let seg = ws.quant_matrix(&ctx, &format!("{P}.shared_expert_gate"));

    for t in TS {
        let x = case.array(&ctx, &format!("x_t{t}"));
        let dim = x.dim(-1);
        let x2 = x.reshape(&[t, dim]);

        // The mixture: (y * scores[..., None]).sum(axis=-2)
        let y = case.array(&ctx, &format!("switch_out_t{t}")).reshape(&[t, k, dim]);
        let scores = case.array(&ctx, &format!("scores_t{t}"));
        let mixed = ornith_rs::ops::moe::mix_experts(&ctx, &y, &scores);
        exact(&format!("mixture T={t}"), &mixed.to_vec_f32(&ctx), &case.f32(&format!("mixed_t{t}")));

        // The shared expert
        let sg = quantized_matmul(&ctx, &x2, &hg);
        let su = quantized_matmul(&ctx, &x2, &hu);
        let act = ornith_rs::ops::elementwise::swiglu(&ctx, &sg, &su);
        let sy = quantized_matmul(&ctx, &act, &hd);
        let gs = quantized_matmul(&ctx, &x2, &seg);
        let sy = ornith_rs::ops::elementwise::sigmoid_gate(&ctx, &gs, &sy);
        exact(&format!("shared expert T={t}"), &sy.to_vec_f32(&ctx), &case.f32(&format!("shared_t{t}")));
    }
}

/// The expert-sort permutation against mlx's `_gather_sort`.
#[test]
fn sort_permutation_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let ctx = Context::new().unwrap();
    let k = case.meta_usize("top_k");
    let n_experts = case.meta_usize("n_experts");
    for t in TS {
        let inds = Array::from_slice(
            &ctx,
            &case.get(&format!("inds_t{t}")).to_u32(),
            &[t, k],
            ornith_rs::array::Dtype::U32,
        );
        let (order, inv) = ornith_rs::ops::moe::counting_sort_for_test(&ctx, &inds, n_experts);
        // mlx dumps `indices[order]`, not `order` itself, so compare that.
        let ord = order.to_vec_u32(&ctx);
        let flat = case.get(&format!("inds_t{t}")).to_u32();
        let got: Vec<u32> = ord.iter().map(|o| flat[*o as usize]).collect();
        assert_eq!(got, case.get(&format!("sort_idx_t{t}")).to_u32(), "sorted indices T={t}");
        assert_eq!(inv.to_vec_u32(&ctx), case.get(&format!("sort_inv_t{t}")).to_u32(), "inv_order T={t}");
    }
}

/// Sorting is only a permutation, so switching it off must not change the
/// result. If it does, the sorted plumbing is wrong rather than the kernel.
#[test]
fn sorting_does_not_change_switch_glu() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let g = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.gate_proj"));
    let u = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let dn = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.down_proj"));
    let k = case.meta_usize("top_k");

    for t in [24usize] {
        let x = case.array(&ctx, &format!("x_t{t}"));
        let x2 = x.reshape(&[t, x.dim(-1)]);
        let inds = Array::from_slice(
            &ctx,
            &case.get(&format!("inds_t{t}")).to_u32(),
            &[t, k],
            ornith_rs::array::Dtype::U32,
        );
        let unsorted = ornith_rs::ops::moe::switch_glu_sorted(
            &ctx, &x2, &inds, &g, &u, &dn, Some(false),
        );
        let want = case.f32(&format!("switch_out_t{t}"));
        let d = compare(&unsorted.to_vec_f32(&ctx), &want);
        eprintln!("T={t} unsorted vs mlx: {d}");
        let sorted = ornith_rs::ops::moe::switch_glu_sorted(
            &ctx, &x2, &inds, &g, &u, &dn, Some(true),
        );
        let d2 = compare(&sorted.to_vec_f32(&ctx), &want);
        eprintln!("T={t} sorted   vs mlx: {d2}");
        assert_eq!(d.max_abs, 0.0, "unsorted path differs from mlx");
    }
}

#[test]
fn debug_sorted_gather() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let ctx = Context::new().unwrap();
    let k = case.meta_usize("top_k");
    let n_experts = case.meta_usize("n_experts");
    let t = 24usize;
    let x = case.array(&ctx, &format!("x_t{t}"));
    let dim = x.dim(-1);
    let x2 = x.reshape(&[t, dim]);
    let inds = Array::from_slice(
        &ctx,
        &case.get(&format!("inds_t{t}")).to_u32(),
        &[t, k],
        ornith_rs::array::Dtype::U32,
    );
    let (order, inv) = ornith_rs::ops::moe::counting_sort_for_test(&ctx, &inds, n_experts);
    let ord = order.to_vec_u32(&ctx);
    let invv = inv.to_vec_u32(&ctx);
    eprintln!("order[0..8]   = {:?}", &ord[..8]);
    eprintln!("inv[0..8]     = {:?}", &invv[..8]);

    let xs = ornith_rs::ops::moe::gather_rows_for_test(&ctx, &x2, &order, k);
    let xs_host = xs.to_vec_f32(&ctx);
    let x_host = x2.to_vec_f32(&ctx);
    let mut bad = 0;
    for (r, o) in ord.iter().enumerate() {
        let src = (*o as usize) / k;
        for c in 0..dim {
            if xs_host[r * dim + c] != x_host[src * dim + c] {
                bad += 1;
            }
        }
    }
    eprintln!("gathered rows mismatching elements: {bad}");
    assert_eq!(bad, 0);
}

/// `gather_qmm` on gathered rows must equal the ungathered result permuted.
#[test]
fn debug_gather_qmv_sorted() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let up = ws.quant_matrix(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let k = case.meta_usize("top_k");
    let n_experts = case.meta_usize("n_experts");
    let t = 24usize;
    let x = case.array(&ctx, &format!("x_t{t}"));
    let dim = x.dim(-1);
    let x2 = x.reshape(&[t, dim]);
    let flat_inds = case.get(&format!("inds_t{t}")).to_u32();
    let inds = Array::from_slice(&ctx, &flat_inds, &[t, k], ornith_rs::array::Dtype::U32);
    let rows = t * k;

    // Unsorted: lhs = p / k, rhs = inds[p]
    let lhs_u: Vec<u32> = (0..rows as u32).map(|i| i / k as u32).collect();
    let lhs_u = Array::from_slice(&ctx, &lhs_u, &[rows], ornith_rs::array::Dtype::U32);
    let rhs_u = inds.reshape(&[rows]);
    let y_u = ornith_rs::ops::quantized::gather_qmm_rows(&ctx, &x2, &up, &lhs_u, &rhs_u, false);

    // Sorted: gather the rows first, lhs = arange
    let (order, _inv) = ornith_rs::ops::moe::counting_sort_for_test(&ctx, &inds, n_experts);
    let xs = ornith_rs::ops::moe::gather_rows_for_test(&ctx, &x2, &order, k);
    let ord = order.to_vec_u32(&ctx);
    let idx_s: Vec<u32> = ord.iter().map(|o| flat_inds[*o as usize]).collect();
    let idx_s = Array::from_slice(&ctx, &idx_s, &[rows], ornith_rs::array::Dtype::U32);
    let lhs_s: Vec<u32> = (0..rows as u32).collect();
    let lhs_s = Array::from_slice(&ctx, &lhs_s, &[rows], ornith_rs::array::Dtype::U32);
    let y_s = ornith_rs::ops::quantized::gather_qmm_rows(&ctx, &xs, &up, &lhs_s, &idx_s, true);

    let n = up.out_dims();
    let hu = y_u.to_vec_f32(&ctx);
    let hs = y_s.to_vec_f32(&ctx);
    let mut bad = 0;
    for (j, o) in ord.iter().enumerate() {
        for c in 0..n {
            if hs[j * n + c] != hu[*o as usize * n + c] {
                bad += 1;
            }
        }
    }
    eprintln!("gather_qmv sorted vs permuted-unsorted: {bad} of {} differ", rows * n);
    assert_eq!(bad, 0);
}

#[test]
fn debug_unsort() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let ctx = Context::new().unwrap();
    let k = case.meta_usize("top_k");
    let n_experts = case.meta_usize("n_experts");
    let t = 24usize;
    let dim = 2048usize;
    let rows = t * k;
    let inds = Array::from_slice(
        &ctx,
        &case.get(&format!("inds_t{t}")).to_u32(),
        &[t, k],
        ornith_rs::array::Dtype::U32,
    );
    let (order, inv) = ornith_rs::ops::moe::counting_sort_for_test(&ctx, &inds, n_experts);
    // A recognisable payload: row j holds the value j.
    let payload: Vec<f32> = (0..rows * dim).map(|i| (i / dim) as f32).collect();
    let y = Array::from_f32(&ctx, &payload, &[rows, dim]);
    let un = ornith_rs::ops::moe::unsort_for_test(&ctx, &y, &inv);
    let got = un.to_vec_f32(&ctx);
    let ord = order.to_vec_u32(&ctx);
    // out[order[j]] must be j
    let mut bad = 0;
    for (j, o) in ord.iter().enumerate() {
        if got[*o as usize * dim] != j as f32 {
            bad += 1;
        }
    }
    eprintln!("unsort mismatches: {bad} of {rows}");
    assert_eq!(bad, 0);
}
