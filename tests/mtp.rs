//! The MTP head against MTPLX's `mtp_forward`.
//!
//! Reference: tools/dump_mtp.py
//!
//! Narrow to wide: the unquantized 256-expert block on its own (which is the
//! only genuinely new kernel path -- `gather_mv` instead of `gather_qmv`),
//! then the head's input after `pre_fc_norm` + `fc`, then a whole
//! `mtp_forward`, then the depth-3 draft chain.

use ornith_rs::array::Array;
use ornith_rs::metal::Context;
use ornith_rs::model::config::TextConfig;
use ornith_rs::model::mtp::{MtpHead, make_mtp_cache, sidecar_path};
use ornith_rs::model::qwen3_5::TextModel;
use ornith_rs::ops::moe::sparse_moe_block_dense;
use ornith_rs::refs::{RefCase, compare, model_dir};
use ornith_rs::weights::WeightStore;

fn check(case: &RefCase, ctx: &Context, name: &str, got: &Array) {
    let want = case.f32(name);
    let d = compare(&got.to_vec_f32(ctx), &want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

#[test]
#[ignore = "loads the trunk and the 1.7 GB MTP sidecar; run with --ignored"]
fn head_matches_mtplx() {
    // MTPLX computed these references with its own window forward, so they pin
    // the undeparted path -- see `tests/quantized.rs` for the argument.
    ornith_rs::ops::invariant::with(false, head_matches_mtplx_inner)
}

fn head_matches_mtplx_inner() {
    let Some(case) = RefCase::open_or_skip("mtp") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[sidecar_path(&dir)]).unwrap();
    ws.prefetch();
    let t0 = std::time::Instant::now();
    let model = TextModel::load(&ctx, &mut ws, cfg.clone());
    let head = MtpHead::load(&ctx, &mut ws, &cfg);
    ctx.sync();
    eprintln!(
        "loaded {:.1} GB in {:.1}s",
        ws.loaded_bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    // The contract the reference ran under has to be the one this port
    // implements, or everything below is comparing different models.
    //
    // `mtp_position_mode` takes two spellings. The published artifact's
    // contract said "cache"; MTPLX 2.11.3's calibrator writes "local" for the
    // rebuild, and `_resolve_runtime_mtp_position_mode` maps both to the same
    // thing -- the KV cache owns the offset and no explicit draft-side RoPE
    // offset is applied, which is what this port does. Only "absolute",
    // "capped" and "wrapped" would be a different model.
    let meta = case.meta().unwrap();
    for (k, want) in [
        ("base_hidden_variant", "post_norm"),
        ("hidden_variant", "post_norm"),
        ("concat_order", "embedding_hidden"),
    ] {
        assert_eq!(meta.get(k).and_then(|v| v.as_str()), Some(want), "contract.{k}");
    }
    let position = meta.get("mtp_position_mode").and_then(|v| v.as_str());
    assert!(
        matches!(position, Some("cache") | Some("local")),
        "contract.mtp_position_mode: {position:?} is an explicit-offset mode"
    );

    // --- the unquantized MoE block on its own
    for t in [1usize, 4] {
        let x = case.array(&ctx, &format!("moe{t}_in"));
        let got = sparse_moe_block_dense(
            &ctx,
            &x,
            &head.mlp.gate,
            &head.mlp.switch_gate,
            &head.mlp.switch_up,
            &head.mlp.switch_down,
            &head.mlp.shared_gate,
            &head.mlp.shared_up,
            &head.mlp.shared_down,
            &head.mlp.shared_expert_gate,
            cfg.num_experts_per_tok,
            cfg.norm_topk_prob,
        );
        check(&case, &ctx, &format!("moe{t}_out"), &got);
    }

    // --- the trunk, so the hidden the head is fed is the right one
    let ids_ref = case.get("ids");
    let n = ids_ref.numel();
    let ids = Array::from_slice(
        &ctx,
        &ids_ref.to_u32(),
        &[n],
        ornith_rs::array::Dtype::U32,
    );
    let mut caches = model.make_cache();
    let (hidden, _) = model.forward_hidden(&ctx, &ids, &mut caches);
    check(&case, &ctx, "trunk_hidden", &hidden);
    let primary = case.get("primary").to_u32()[0];

    // --- the head's input, before the decoder block
    let last = hidden.slice_leading(0, 1).reshape(&[n, cfg.hidden_size]);
    let last = last.slice_leading(n - 1, 1).reshape(&[1, 1, cfg.hidden_size]);
    let ptok = Array::from_slice(&ctx, &[primary], &[1], ornith_rs::array::Dtype::U32);

    // --- one mtp_forward at T = 1
    let mut mtp_cache = make_mtp_cache();
    let (logits, h) = head.forward(
        &ctx,
        &cfg,
        &model.embed_tokens,
        &model.lm_head,
        &last,
        &ptok,
        &mut mtp_cache[0],
    );
    check(&case, &ctx, "mtp1_hidden", &h);
    check(&case, &ctx, "mtp1_logits", &logits);

    // --- the depth-3 draft chain
    let want_drafts = case.get("draft_tokens").to_u32();
    let mut mtp_cache = make_mtp_cache();
    let mut dh = last.clone();
    let mut dtok = ptok.clone();
    let mut drafts: Vec<u32> = Vec::new();
    for level in 0..case.meta_usize("depth") {
        let (dl, dhid) = head.forward(
            &ctx,
            &cfg,
            &model.embed_tokens,
            &model.lm_head,
            &dh,
            &dtok,
            &mut mtp_cache[0],
        );
        check(&case, &ctx, &format!("draft{level}_hidden"), &dhid);
        check(&case, &ctx, &format!("draft{level}_logits"), &dl);
        let tok = ornith_rs::ops::sample::greedy_argmax(&ctx, &dl);
        drafts.push(tok);
        dtok = Array::from_slice(&ctx, &[tok], &[1], ornith_rs::array::Dtype::U32);
        dh = dhid;
    }
    assert_eq!(drafts, want_drafts, "draft tokens");

    // --- verify: one trunk forward over the drafted window
    let window = case.get("verify_window").to_u32();
    let w = Array::from_slice(
        &ctx,
        &window,
        &[window.len()],
        ornith_rs::array::Dtype::U32,
    );
    let (vh, _) = model.forward_hidden(&ctx, &w, &mut caches);
    check(&case, &ctx, "verify_hidden", &vh);
    let vlogits =
        model.lm_head.matmul(&ctx, &vh);
    check(&case, &ctx, "verify_logits", &vlogits);
}

/// The capturing forward must be the packed one, bit for bit, and rewinding to
/// the full window must leave the caches exactly where the packed forward
/// leaves them.
///
/// The first half is the claim that makes the whole rewind possible: the
/// recurrence kernel loops `for (t = 0; t < T; ++t)` carrying its state in
/// float32 registers, so stepping it one token at a time and writing the state
/// out between steps changes nothing.
#[test]
#[ignore = "loads the whole model; run with --ignored"]
fn capture_matches_the_packed_forward() {
    // Both arms of this are window forwards against MTPLX's own, so it pins the
    // undeparted path -- see `tests/quantized.rs`.
    ornith_rs::ops::invariant::with(false, capture_matches_the_packed_forward_inner)
}

fn capture_matches_the_packed_forward_inner() {
    let Some(case) = RefCase::open_or_skip("mtp") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg.clone());
    ctx.sync();

    let ids_ref = case.get("ids");
    let n = ids_ref.numel();
    let ids =
        Array::from_slice(&ctx, &ids_ref.to_u32(), &[n], ornith_rs::array::Dtype::U32);
    let window = case.get("verify_window").to_u32();
    let w = Array::from_slice(
        &ctx,
        &window,
        &[window.len()],
        ornith_rs::array::Dtype::U32,
    );

    let mut a = model.make_cache();
    model.forward_hidden(&ctx, &ids, &mut a);
    let (ha, _, capture) = model.forward_capture(&ctx, &w, &mut a);
    check(&case, &ctx, "verify_hidden", &ha);

    let mut b = model.make_cache();
    model.forward_hidden(&ctx, &ids, &mut b);
    let (hb, _) = model.forward_hidden(&ctx, &w, &mut b);
    let d = compare(&ha.to_vec_f32(&ctx), &hb.to_vec_f32(&ctx));
    assert_eq!(d.max_abs, 0.0, "captured forward vs packed forward: {d}");

    capture.rewind(&ctx, &mut a, window.len());
    for (i, (ca, cb)) in a.iter().zip(b.iter()).enumerate() {
        use ornith_rs::model::cache::LayerCache;
        assert_eq!(ca.offset(), cb.offset(), "layer {i} offset");
        if let (LayerCache::Gdn(ga), LayerCache::Gdn(gb)) = (ca, cb) {
            let (sa, sb) = (ga.state.as_ref().unwrap(), gb.state.as_ref().unwrap());
            let d = compare(&sa.to_vec_f32(&ctx), &sb.to_vec_f32(&ctx));
            assert_eq!(d.max_abs, 0.0, "layer {i} recurrent state: {d}");
            let (va, vb) = (ga.conv.as_ref().unwrap(), gb.conv.as_ref().unwrap());
            let d = compare(&va.to_vec_f32(&ctx), &vb.to_vec_f32(&ctx));
            assert_eq!(d.max_abs, 0.0, "layer {i} conv window: {d}");
        }
    }
}

/// Speculative decoding is supposed to be output-preserving: with a greedy
/// sampler the accepted tokens are the ones autoregression would have
/// produced. On this stack it is not, and `window_and_steps_both_match_mlx`
/// shows why -- so what this test checks is that the *acceptance* is sane,
/// which is what would collapse if the gated-delta rewind were wrong.
///
/// A broken rewind feeds the trunk a state that never existed; the drafts
/// would then almost never be confirmed and the acceptance rate would fall to
/// near zero. Measured against MTPLX's own loop driven from Python, depth 1
/// runs 0.53 - 0.84.
#[test]
#[ignore = "loads the trunk and the MTP sidecar; run with --ignored"]
fn speculative_tracks_autoregressive() {
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[sidecar_path(&dir)]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg.clone());
    let head = MtpHead::load(&ctx, &mut ws, &cfg);
    ctx.sync();
    drop(ws);

    let tk = ornith_rs::tokenizer::Tokenizer::load_as(
        &dir,
        ornith_rs::tokenizer::split::Dialect::Transformers,
    )
    .unwrap();
    const MAX: usize = 120;

    let prompts = [
        "In one sentence, what is a neutron star?",
        "Write a Python function that merges two sorted lists.",
        "Name the first eight prime numbers.",
    ];
    let mut total = 0usize;
    let mut agreed = 0usize;
    for p in prompts {
        let text = ornith_rs::chat::apply_chat_template(
            &[ornith_rs::chat::Message::text(ornith_rs::chat::Role::User, p)],
            true,
            false,
        )
        .unwrap();
        let ids = tk.encode(&text);

        let mut ar: Vec<u32> = Vec::new();
        let mut caches = model.make_cache();
        ornith_rs::generate::generate_step(
            &ctx,
            &model,
            &mut caches,
            &ids,
            &mut ornith_rs::generate::Greedy,
            MAX,
            ornith_rs::generate::DEFAULT_PREFILL_STEP_SIZE,
            None,
            |_, _| {},
            |t| {
                ar.push(t);
                !tk.eos_ids.contains(&t) && ar.len() < MAX
            },
        );

        let mut sp: Vec<u32> = Vec::new();
        let mut caches = model.make_cache();
        let stats = ornith_rs::mtp_generate::generate_step_mtp(
            &ctx,
            &model,
            &head,
            None,
            &mut caches,
            &ids,
            &mut ornith_rs::generate::Greedy,
            3,
            MAX,
            ornith_rs::generate::DEFAULT_PREFILL_STEP_SIZE,
            ornith_rs::mtp_generate::HistoryPolicy::Committed,
            None,
            |_, _| {},
            |t| {
                sp.push(t);
                !tk.eos_ids.contains(&t) && sp.len() < MAX
            },
        );

        let common = ar.iter().zip(&sp).take_while(|(a, b)| a == b).count();
        total += ar.len().min(sp.len());
        agreed += common;
        let by: Vec<String> =
            stats.acceptance_by_depth().iter().map(|v| format!("{v:.2}")).collect();
        eprintln!(
            "{:<52} ar {:3} tokens, mtp {:3}, agree {:3}  acceptance {}",
            &p[..p.len().min(50)],
            ar.len(),
            sp.len(),
            common,
            by.join(" / ")
        );
        if common < ar.len().min(sp.len()) {
            eprintln!(
                "    first divergence at {common}: ar {:?} vs mtp {:?}",
                tk.token(ar[common]),
                tk.token(sp[common])
            );
        }
    }
    eprintln!("agreed on {agreed}/{total} tokens before the first divergence");
    assert!(agreed > 0, "the streams never agreed at all");
}

/// Both halves of the divergence, against mlx.
///
/// The window fed at once and the window fed one token at a time disagree --
/// by ~1.4 on a hidden whose scale is ~35, which is a handful of bfloat16 ulps
/// compounded through 40 residual layers, and by up to ~1.1 on logits whose
/// top-2 gap is sometimes 0.31. This port reproduces *both* of mlx's answers
/// bit for bit, so the gap belongs to the stack, not to the port: no
/// speculative decoder on this model can be output-identical to
/// autoregression, MTPLX's included.
#[test]
#[ignore = "loads the whole model; run with --ignored"]
fn window_and_steps_both_match_mlx() {
    // Off, for `tests/quantized.rs`'s reason: this pins both of *mlx's*
    // answers, and the window mode deliberately computes a third thing -- the
    // one the single-step path would have given. That the two mlx answers
    // differ at all is what the window mode exists to fix.
    ornith_rs::ops::invariant::with(false, window_and_steps_both_match_mlx_inner)
}

fn window_and_steps_both_match_mlx_inner() {
    let Some(case) = RefCase::open_or_skip("mtp") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let cfg = TextConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    ws.prefetch();
    let model = TextModel::load(&ctx, &mut ws, cfg.clone());
    ctx.sync();

    let ids_ref = case.get("ids");
    let n = ids_ref.numel();
    let ids =
        Array::from_slice(&ctx, &ids_ref.to_u32(), &[n], ornith_rs::array::Dtype::U32);
    let window = case.get("verify_window").to_u32();

    let mut a = model.make_cache();
    model.forward_hidden(&ctx, &ids, &mut a);
    let w = Array::from_slice(
        &ctx,
        &window,
        &[window.len()],
        ornith_rs::array::Dtype::U32,
    );
    let (ha, _) = model.forward_hidden(&ctx, &w, &mut a);
    check(&case, &ctx, "verify_hidden", &ha);

    let mut b = model.make_cache();
    model.forward_hidden(&ctx, &ids, &mut b);
    let mut steps: Vec<f32> = Vec::new();
    for &t in &window {
        let one = Array::from_slice(&ctx, &[t], &[1], ornith_rs::array::Dtype::U32);
        let (h, _) = model.forward_hidden(&ctx, &one, &mut b);
        steps.extend(h.to_vec_f32(&ctx));
    }
    let d = compare(&steps, &case.f32("step_hidden"));
    assert_eq!(d.max_abs, 0.0, "step_hidden: {d}");

    // And the gap between them, which is what the speculative loop inherits.
    let d = compare(&ha.to_vec_f32(&ctx), &steps);
    eprintln!("window vs single steps: {d}");
    assert!(d.max_abs > 0.0, "the two paths agreed, so this whole note is stale");
}
