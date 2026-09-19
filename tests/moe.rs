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
    let gate = ws.linear(&ctx, &format!("{P}.gate"));
    let k = case.meta_usize("top_k");

    for t in TS {
        let x = case.array(&ctx, &format!("x_t{t}"));
        let x2 = x.reshape(&[t, x.dim(-1)]);
        let gates = softmax(&ctx, &gate.matmul(&ctx, &x2), true);
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
    let g = ws.linear(&ctx, &format!("{P}.switch_mlp.gate_proj"));
    let u = ws.linear(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let dn = ws.linear(&ctx, &format!("{P}.switch_mlp.down_proj"));
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
    let gate = ws.linear(&ctx, &format!("{P}.gate"));
    let sg = ws.linear(&ctx, &format!("{P}.switch_mlp.gate_proj"));
    let su = ws.linear(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let sd = ws.linear(&ctx, &format!("{P}.switch_mlp.down_proj"));
    let hg = ws.linear(&ctx, &format!("{P}.shared_expert.gate_proj"));
    let hu = ws.linear(&ctx, &format!("{P}.shared_expert.up_proj"));
    let hd = ws.linear(&ctx, &format!("{P}.shared_expert.down_proj"));
    let seg = ws.linear(&ctx, &format!("{P}.shared_expert_gate"));
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
    let hg = ws.linear(&ctx, &format!("{P}.shared_expert.gate_proj"));
    let hu = ws.linear(&ctx, &format!("{P}.shared_expert.up_proj"));
    let hd = ws.linear(&ctx, &format!("{P}.shared_expert.down_proj"));
    let seg = ws.linear(&ctx, &format!("{P}.shared_expert_gate"));

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
        let sg = hg.matmul(&ctx, &x2);
        let su = hu.matmul(&ctx, &x2);
        let act = ornith_rs::ops::elementwise::swiglu(&ctx, &sg, &su);
        let sy = hd.matmul(&ctx, &act);
        let gs = seg.matmul(&ctx, &x2);
        let sy = ornith_rs::ops::elementwise::sigmoid_gate(&ctx, &gs, &sy);
        exact(&format!("shared expert T={t}"), &sy.to_vec_f32(&ctx), &case.f32(&format!("shared_t{t}")));
    }
}

/// The two copies the sorted path used to make, against the indexing that
/// replaced them.
///
/// `gather_rows_div` and `unsort_rows` put rows where the next kernel
/// expected them; the gemm and the block's tail both index, so both copies
/// are gone (`REBUILD.md` Phase 36). The claim is not that the answers are
/// close. The same rows meet the same weights and the tail folds the same
/// `k` of them in the same order, so all four combinations of the two
/// controls have to reproduce mlx **exactly**, and a tolerance here would
/// hide the one mistake this can make -- an off-by-one in a permutation,
/// which moves a whole row and nothing else.
#[test]
fn the_indexed_moe_block_is_the_copied_one() {
    let Some(case) = RefCase::open_or_skip("moe") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let gate = ws.linear(&ctx, &format!("{P}.gate"));
    let sg = ws.linear(&ctx, &format!("{P}.switch_mlp.gate_proj"));
    let su = ws.linear(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let sd = ws.linear(&ctx, &format!("{P}.switch_mlp.down_proj"));
    let hg = ws.linear(&ctx, &format!("{P}.shared_expert.gate_proj"));
    let hu = ws.linear(&ctx, &format!("{P}.shared_expert.up_proj"));
    let hd = ws.linear(&ctx, &format!("{P}.shared_expert.down_proj"));
    let seg = ws.linear(&ctx, &format!("{P}.shared_expert_gate"));
    let k = case.meta_usize("top_k");

    // Only the sorted widths exercise either of them, and only a weight
    // whose gathered matmul reads `lhs_indices` exercises the front one --
    // which mlx's own sorted kernel does not. Say which artifact this is, so
    // a green run on an affine one is not read as covering both.
    let sorted: Vec<usize> = TS.iter().copied().filter(|t| t * k >= 64).collect();
    for &t in &sorted {
        eprintln!(
            "T={t}: the expert gemm {} lhs_indices on this artifact",
            if sg.gather_reads_lhs(t * k, true) { "reads" } else { "ignores" },
        );
    }
    // And whether the copy that is left takes its wide path here. The `wide`
    // arm of the loop below is only a control if this says yes -- a guard
    // that answers "narrow" to both arms makes the pair green and empty.
    {
        let x = case.array(&ctx, &format!("x_t{}", sorted[0]));
        let dim = x.dim(-1);
        let x2 = x.reshape(&[sorted[0], dim]);
        assert!(
            ornith_rs::ops::moe::gather_would_widen(&x2, dim),
            "the wide gather is guarded off here, so the `wide` arm below is not a control"
        );
        eprintln!("the copy takes its 16-byte path (offset {}, dim {dim})", x2.offset);
    }

    for &t in &sorted {
        let x = case.array(&ctx, &format!("x_t{t}"));
        let want = case.f32(&format!("out_t{t}"));
        for gather in [true, false] {
            for unsort in [true, false] {
                // And the two widths of the copy that is left when the
                // front is not indexed -- one element a thread and sixteen
                // bytes a thread. A copy may not round anything, so this
                // belongs under the same `max_abs == 0.0`.
                for wide in [true, false] {
                    ornith_rs::ops::moe::set_fuse_moe_gather(gather);
                    ornith_rs::ops::moe::set_fuse_moe_unsort(unsort);
                    ornith_rs::ops::moe::set_gather_wide(wide);
                    let out = sparse_moe_block(
                        &ctx, &x, &gate, &sg, &su, &sd, &hg, &hu, &hd, &seg, k, true,
                    );
                    exact(
                        &format!("moe block T={t} gather={gather} unsort={unsort} wide={wide}"),
                        &out.to_vec_f32(&ctx),
                        &want,
                    );
                }
            }
        }
    }
    ornith_rs::ops::moe::set_fuse_moe_gather(true);
    ornith_rs::ops::moe::set_fuse_moe_unsort(true);
    ornith_rs::ops::moe::set_gather_wide(true);
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
    let g = ws.linear(&ctx, &format!("{P}.switch_mlp.gate_proj"));
    let u = ws.linear(&ctx, &format!("{P}.switch_mlp.up_proj"));
    let dn = ws.linear(&ctx, &format!("{P}.switch_mlp.down_proj"));
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
    let up = ws.linear(&ctx, &format!("{P}.switch_mlp.up_proj"));
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
    let y_u = up.gather_matmul(&ctx, &x2, &lhs_u, &rhs_u, false);

    // Sorted: gather the rows first, lhs = arange
    let (order, _inv) = ornith_rs::ops::moe::counting_sort_for_test(&ctx, &inds, n_experts);
    let xs = ornith_rs::ops::moe::gather_rows_for_test(&ctx, &x2, &order, k);
    let ord = order.to_vec_u32(&ctx);
    let idx_s: Vec<u32> = ord.iter().map(|o| flat_inds[*o as usize]).collect();
    let idx_s = Array::from_slice(&ctx, &idx_s, &[rows], ornith_rs::array::Dtype::U32);
    let lhs_s: Vec<u32> = (0..rows as u32).collect();
    let lhs_s = Array::from_slice(&ctx, &lhs_s, &[rows], ornith_rs::array::Dtype::U32);
    let y_s = up.gather_matmul(&ctx, &xs, &lhs_s, &idx_s, true);

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

/// The parallel gather sort is the one-thread gather sort, permutation for
/// permutation.
///
/// `sort_permutation_matches_mlx` above already pins whichever path runs to
/// mlx's `mx.argsort`, and at T = 512 that is the parallel one. This pins the
/// other direction: the two paths against each other, so the control the
/// timing runs alternate against is the same sort and not merely a sort.
/// Stability is the whole content of the claim -- 16384 equal keys have to
/// keep their input order across 64 blocks.
#[test]
fn the_parallel_sort_is_the_serial_one() {
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
        let was = ornith_rs::ops::moe::set_parallel_sort(false);
        let (so, si) = ornith_rs::ops::moe::counting_sort_for_test(&ctx, &inds, n_experts);
        let (so, si) = (so.to_vec_u32(&ctx), si.to_vec_u32(&ctx));
        ornith_rs::ops::moe::set_parallel_sort(true);
        let (po, pi) = ornith_rs::ops::moe::counting_sort_for_test(&ctx, &inds, n_experts);
        let (po, pi) = (po.to_vec_u32(&ctx), pi.to_vec_u32(&ctx));
        ornith_rs::ops::moe::set_parallel_sort(was);
        assert_eq!(po, so, "order, T={t} ({} rows)", t * k);
        assert_eq!(pi, si, "inv_order, T={t} ({} rows)", t * k);
    }
}
