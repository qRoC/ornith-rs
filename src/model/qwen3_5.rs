//! The decoder: a port of `vendor/mlx-lm/mlx_lm/models/qwen3_5.py` (with
//! `Attention`, `MLP` and `SparseMoeBlock` imported from `qwen3_next.py`) and
//! the weight naming from `qwen3_5_moe.py`.
//!
//! Layer structure, for the 40 layers of this model:
//!
//! ```text
//!   DecoderLayer(i):
//!     r = (linear_attn if (i + 1) % 4 else self_attn)(input_layernorm(x))
//!     h = x + r
//!     out = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! so 30 gated-delta layers and 10 full-attention layers, each followed by the
//! same 256-expert MoE block.

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::elementwise::{add, precise_swiglu, scalar_mul, sigmoid_gate, silu};
use crate::ops::embedding::embedding;
use crate::ops::gdn::{concat_time, depthwise_conv1d, gated_delta_update};
use crate::ops::layout::{slice_inner, transpose_hl, transpose_lh};
use crate::ops::moe::sparse_moe_block;
use crate::ops::norm::rms_norm;
use crate::ops::quantized::{QuantMatrix, quantized_matmul};
use crate::ops::rope::rope;
use crate::weights::WeightStore;

use crate::ops::linear::Linear;

use super::cache::{LayerCache, make_cache};
use super::config::TextConfig;

pub struct GatedDeltaNet {
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    pub in_proj_b: Array,
    pub in_proj_a: Array,
    pub conv1d: Array,
    pub dt_bias: Array,
    pub a_log: Array,
    pub norm: Array,
    pub out_proj: Linear,
}

pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Array,
    pub k_norm: Array,
}

pub struct MoeBlock {
    pub gate: Linear,
    pub switch_gate: Linear,
    pub switch_up: Linear,
    pub switch_down: Linear,
    pub shared_gate: Linear,
    pub shared_up: Linear,
    pub shared_down: Linear,
    pub shared_expert_gate: Linear,
}

pub enum Mixer {
    Linear(GatedDeltaNet),
    Full(Attention),
}

pub struct DecoderLayer {
    pub mixer: Mixer,
    pub input_layernorm: Array,
    pub post_attention_layernorm: Array,
    pub mlp: MoeBlock,
}

pub struct TextModel {
    pub cfg: TextConfig,
    pub embed_tokens: QuantMatrix,
    pub layers: Vec<DecoderLayer>,
    pub norm: Array,
    pub lm_head: Linear,
}

impl TextModel {
    /// Load every tensor the language model needs.
    ///
    /// mlx-lm reaches these names through `qwen3_5_moe.Model.sanitize`, which
    /// rewrites `model.language_model.*` to `language_model.model.*` and splits
    /// a fused `experts.gate_up_proj` into `switch_mlp.gate_proj`/`up_proj`.
    /// This artifact is already stored in the sanitized layout -- its keys are
    /// `language_model.model.layers.N.mlp.switch_mlp.gate_proj.weight` and its
    /// `conv1d.weight` is already `[D, K, 1]` -- so nothing is rewritten here.
    /// That last point also decides the norm weights: `TextModel.sanitize` adds
    /// 1.0 to them only when it sees an unsanitized `conv1d.weight`, which this
    /// export does not have. Adding it anyway is the "double shift" MTPLX's
    /// qwen3_5 patch warns about, and it turns the trunk into gibberish.
    pub fn load(ctx: &Context, ws: &mut WeightStore, cfg: TextConfig) -> TextModel {
        let p = "language_model.model";
        assert!(
            ws.info(&format!("{p}.layers.0.linear_attn.conv1d.weight"))
                .map(|i| i.shape[2] == 1)
                .unwrap_or(false),
            "conv1d.weight is not in the sanitized [D, K, 1] layout; mlx-lm would \
             moveaxis it and shift every norm weight by +1.0"
        );

        let embed_tokens = ws.quant_matrix(ctx, &format!("{p}.embed_tokens"));
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("{p}.layers.{i}");
            let mixer = if cfg.is_linear_layer(i) {
                Mixer::Linear(GatedDeltaNet {
                    in_proj_qkv: ws.linear(ctx, &format!("{lp}.linear_attn.in_proj_qkv")),
                    in_proj_z: ws.linear(ctx, &format!("{lp}.linear_attn.in_proj_z")),
                    in_proj_b: ws.tensor(ctx, &format!("{lp}.linear_attn.in_proj_b.weight")),
                    in_proj_a: ws.tensor(ctx, &format!("{lp}.linear_attn.in_proj_a.weight")),
                    conv1d: ws.tensor(ctx, &format!("{lp}.linear_attn.conv1d.weight")),
                    dt_bias: ws.tensor(ctx, &format!("{lp}.linear_attn.dt_bias")),
                    a_log: ws.tensor(ctx, &format!("{lp}.linear_attn.A_log")),
                    norm: ws.tensor(ctx, &format!("{lp}.linear_attn.norm.weight")),
                    out_proj: ws.linear(ctx, &format!("{lp}.linear_attn.out_proj")),
                })
            } else {
                Mixer::Full(Attention {
                    q_proj: ws.linear(ctx, &format!("{lp}.self_attn.q_proj")),
                    k_proj: ws.linear(ctx, &format!("{lp}.self_attn.k_proj")),
                    v_proj: ws.linear(ctx, &format!("{lp}.self_attn.v_proj")),
                    o_proj: ws.linear(ctx, &format!("{lp}.self_attn.o_proj")),
                    q_norm: ws.tensor(ctx, &format!("{lp}.self_attn.q_norm.weight")),
                    k_norm: ws.tensor(ctx, &format!("{lp}.self_attn.k_norm.weight")),
                })
            };
            layers.push(DecoderLayer {
                mixer,
                input_layernorm: ws.tensor(ctx, &format!("{lp}.input_layernorm.weight")),
                post_attention_layernorm: ws
                    .tensor(ctx, &format!("{lp}.post_attention_layernorm.weight")),
                mlp: MoeBlock {
                    gate: ws.linear(ctx, &format!("{lp}.mlp.gate")),
                    switch_gate: ws.linear(ctx, &format!("{lp}.mlp.switch_mlp.gate_proj")),
                    switch_up: ws.linear(ctx, &format!("{lp}.mlp.switch_mlp.up_proj")),
                    switch_down: ws.linear(ctx, &format!("{lp}.mlp.switch_mlp.down_proj")),
                    shared_gate: ws.linear(ctx, &format!("{lp}.mlp.shared_expert.gate_proj")),
                    shared_up: ws.linear(ctx, &format!("{lp}.mlp.shared_expert.up_proj")),
                    shared_down: ws.linear(ctx, &format!("{lp}.mlp.shared_expert.down_proj")),
                    shared_expert_gate: ws.linear(ctx, &format!("{lp}.mlp.shared_expert_gate")),
                },
            });
        }
        let norm = ws.tensor(ctx, &format!("{p}.norm.weight"));
        let lm_head = ws.linear(ctx, "language_model.lm_head");
        TextModel { cfg, embed_tokens, layers, norm, lm_head }
    }

    pub fn make_cache(&self) -> Vec<LayerCache> {
        make_cache(&self.cfg)
    }

    /// The full forward pass. Returns `(logits, hidden)` where `hidden` is the
    /// pre-final-norm residual stream -- the MTP head's input.
    pub fn forward(
        &self,
        ctx: &Context,
        ids: &Array,
        caches: &mut [LayerCache],
    ) -> (Array, Array) {
        let (h, pre_norm) = self.forward_hidden(ctx, ids, caches);
        let logits = self.lm_head.matmul(ctx, &h);
        (logits, pre_norm)
    }

    /// Everything up to and including the final norm, plus the pre-norm
    /// residual stream alongside it.
    /// `forward_hidden`, keeping every intermediate gated-delta state so a
    /// speculative window can be rewound to any prefix of `ids`.
    pub fn forward_capture(
        &self,
        ctx: &Context,
        ids: &Array,
        caches: &mut [LayerCache],
    ) -> (Array, Array, TrunkCapture) {
        let n = ids.numel();
        let base_offsets: Vec<usize> = caches.iter().map(|c| c.offset()).collect();
        let mut gdn: Vec<Vec<(Array, Array)>> = vec![Vec::new(); caches.len()];
        let mut h = embedding(ctx, &self.embed_tokens, ids).reshape(&[1, n, self.cfg.hidden_size]);
        for (i, layer) in self.layers.iter().enumerate() {
            h = match &layer.mixer {
                Mixer::Linear(_) => {
                    let (out, snaps) = layer.forward_capture(ctx, &self.cfg, &h, &mut caches[i]);
                    gdn[i] = snaps;
                    out
                }
                Mixer::Full(_) => layer.forward(ctx, &self.cfg, &h, &mut caches[i]),
            };
        }
        let normed = rms_norm(ctx, &h, Some(&self.norm), self.cfg.rms_norm_eps);
        (normed, h, TrunkCapture { gdn, base_offsets, tokens: n })
    }

    pub fn forward_hidden(
        &self,
        ctx: &Context,
        ids: &Array,
        caches: &mut [LayerCache],
    ) -> (Array, Array) {
        let n = ids.numel();
        let h = embedding(ctx, &self.embed_tokens, ids).reshape(&[1, n, self.cfg.hidden_size]);
        self.forward_from_embeddings(ctx, &h, caches)
    }

    /// The same forward from an already-embedded prompt.
    ///
    /// mlx-lm's `Model.__call__` takes an `input_embeddings` for exactly this
    /// reason: a vision request replaces the `<|image_pad|>` rows of
    /// `embed_tokens(ids)` with the tower's output before the first layer
    /// sees them (MTPLX's `spliced_chunk_embeddings`).
    pub fn forward_from_embeddings(
        &self,
        ctx: &Context,
        embeddings: &Array,
        caches: &mut [LayerCache],
    ) -> (Array, Array) {
        let n = embeddings.numel() / self.cfg.hidden_size;
        let mut h = embeddings.reshape(&[1, n, self.cfg.hidden_size]);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(ctx, &self.cfg, &h, &mut caches[i]);
        }
        let normed = rms_norm(ctx, &h, Some(&self.norm), self.cfg.rms_norm_eps);
        (normed, h)
    }
}

/// What a partially-accepted speculative window needs to undo.
///
/// A `KVCache` rewinds by moving its offset, which is all `trim` does; a
/// gated-delta layer has no offset to move -- its memory is the conv window
/// and the recurrent state -- so the only way back is to have kept the
/// intermediate ones. `forward_capture` keeps them.
///
/// This is only affordable because the packed recurrence kernel is a strict
/// `for (t = 0; t < T; ++t)` over tokens carrying the state in registers, so
/// stepping it one token at a time is bit-identical to one T-token call -- and
/// each step already allocates its own state and conv window, so a snapshot is
/// a reference, not a copy.
pub struct TrunkCapture {
    /// Per layer, per token: the gated-delta `(conv, state)` after that token.
    /// Empty for a full-attention layer.
    gdn: Vec<Vec<(Array, Array)>>,
    /// Each cache's offset before the forward.
    base_offsets: Vec<usize>,
    pub tokens: usize,
}

impl TrunkCapture {
    /// Rewind every cache to the state it would have had after `keep` of the
    /// `tokens` this forward consumed.
    pub fn rewind(&self, caches: &mut [LayerCache], keep: usize) {
        assert!(keep <= self.tokens);
        for (i, cache) in caches.iter_mut().enumerate() {
            match cache {
                LayerCache::Kv(c) => c.offset = self.base_offsets[i] + keep,
                LayerCache::Gdn(c) => {
                    if keep == 0 {
                        // Nothing of this window survives; the snapshots start
                        // after token 0, so fall back to the entry state only
                        // when the caller kept none -- which the loop never
                        // does, since the primary is always committed.
                        panic!("a speculative window always commits its first token");
                    }
                    let (conv, state) = &self.gdn[i][keep - 1];
                    c.conv = Some(conv.clone());
                    c.state = Some(state.clone());
                    c.offset = self.base_offsets[i] + keep;
                }
            }
        }
    }
}

impl DecoderLayer {
    pub fn forward(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        x: &Array,
        cache: &mut LayerCache,
    ) -> Array {
        let normed = rms_norm(ctx, x, Some(&self.input_layernorm), cfg.rms_norm_eps);
        let r = match &self.mixer {
            Mixer::Linear(g) => g.forward(ctx, cfg, &normed, cache.as_gdn()),
            Mixer::Full(a) => a.forward(ctx, cfg, &normed, cache.as_kv()),
        };
        let h = add(ctx, x, &r);
        let normed2 = rms_norm(ctx, &h, Some(&self.post_attention_layernorm), cfg.rms_norm_eps);
        let m = sparse_moe_block(
            ctx,
            &normed2,
            &self.mlp.gate,
            &self.mlp.switch_gate,
            &self.mlp.switch_up,
            &self.mlp.switch_down,
            &self.mlp.shared_gate,
            &self.mlp.shared_up,
            &self.mlp.shared_down,
            &self.mlp.shared_expert_gate,
            cfg.num_experts_per_tok,
            cfg.norm_topk_prob,
        );
        add(ctx, &h, &m)
    }
}

impl DecoderLayer {
    /// The gated-delta branch with per-token snapshots.
    fn forward_capture(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        x: &Array,
        cache: &mut LayerCache,
    ) -> (Array, Vec<(Array, Array)>) {
        let Mixer::Linear(g) = &self.mixer else { panic!("not a gated-delta layer") };
        let normed = rms_norm(ctx, x, Some(&self.input_layernorm), cfg.rms_norm_eps);
        let (r, snaps) = g.forward_capture(ctx, cfg, &normed, cache.as_gdn());
        let h = add(ctx, x, &r);
        let normed2 = rms_norm(ctx, &h, Some(&self.post_attention_layernorm), cfg.rms_norm_eps);
        let m = sparse_moe_block(
            ctx,
            &normed2,
            &self.mlp.gate,
            &self.mlp.switch_gate,
            &self.mlp.switch_up,
            &self.mlp.switch_down,
            &self.mlp.shared_gate,
            &self.mlp.shared_up,
            &self.mlp.shared_down,
            &self.mlp.shared_expert_gate,
            cfg.num_experts_per_tok,
            cfg.norm_topk_prob,
        );
        (add(ctx, &h, &m), snaps)
    }
}

impl Attention {
    pub fn forward(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        x: &Array,
        cache: &mut super::cache::KvCache,
    ) -> Array {
        let (b, l) = (x.shape[0], x.shape[1]);
        let hd = cfg.head_dim;
        let hq = cfg.num_attention_heads;
        let hkv = cfg.num_key_value_heads;
        let scale = (hd as f32).powf(-0.5);

        // q_proj emits 2 * head_dim per head: the query and its output gate.
        let qg = self.q_proj.matmul(ctx, x).reshape(&[b, l, hq, 2 * hd]);
        let queries = slice_inner(ctx, &qg, 0, hd);
        let gate = slice_inner(ctx, &qg, hd, hd).reshape(&[b, l, hq * hd]);

        let keys = self.k_proj.matmul(ctx, x).reshape(&[b, l, hkv, hd]);
        let values = self.v_proj.matmul(ctx, x).reshape(&[b, l, hkv, hd]);

        let queries = rms_norm(ctx, &queries, Some(&self.q_norm), cfg.rms_norm_eps);
        let keys = rms_norm(ctx, &keys, Some(&self.k_norm), cfg.rms_norm_eps);
        let queries = transpose_lh(ctx, &queries);
        let keys = transpose_lh(ctx, &keys);
        let values = transpose_lh(ctx, &values);

        let offset = cache.offset;
        let dims = cfg.rope_dims();
        let queries = rope(ctx, &queries, dims, cfg.rope_theta, 1.0, offset);
        let keys = rope(ctx, &keys, dims, cfg.rope_theta, 1.0, offset);

        let (k_all, v_all) = cache.update_and_fetch(ctx, &keys, &values);
        let cap = cache.capacity();

        // `create_attention_mask` returns "causal" for a multi-token step and
        // None for a single one, which is exactly `l > 1`.
        let out = crate::ops::sdpa::attention_with_cache(
            ctx, &queries, &k_all, &v_all, cap, scale, l > 1,
        );

        let out = transpose_hl(ctx, &out).reshape(&[b, l, hq * hd]);
        let gated = sigmoid_gate(ctx, &gate, &out);
        self.o_proj.matmul(ctx, &gated)
    }
}

impl GatedDeltaNet {
    pub fn forward(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        x: &Array,
        cache: &mut super::cache::GdnCache,
    ) -> Array {
        let (b, s) = (x.shape[0], x.shape[1]);
        let hk = cfg.linear_num_key_heads;
        let hv = cfg.linear_num_value_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;
        let key_dim = cfg.key_dim();
        let kernel = cfg.linear_conv_kernel_dim;

        let qkv = self.in_proj_qkv.matmul(ctx, x);
        let z = self.in_proj_z.matmul(ctx, x).reshape(&[b, s, hv, dv]);
        // in_proj_b and in_proj_a are unquantized [Hv, hidden] matrices.
        let bb = crate::ops::matmul::matmul(ctx, &x.reshape(&[b * s, x.dim(-1)]), &self.in_proj_b, true)
            .reshape(&[b, s, hv]);
        let aa = crate::ops::matmul::matmul(ctx, &x.reshape(&[b * s, x.dim(-1)]), &self.in_proj_a, true)
            .reshape(&[b, s, hv]);

        // The causal conv sees the cached window ahead of this step's tokens.
        let conv_state = cache.conv_or_zeros(ctx, cfg, b, qkv.dtype);
        let conv_input = concat_time(ctx, &conv_state, &qkv);
        let n_keep = kernel - 1;
        cache.conv = Some(keep_tail(ctx, &conv_input, n_keep));
        let conv_out = silu(ctx, &depthwise_conv1d(ctx, &conv_input, &self.conv1d));

        let q = slice_inner(ctx, &conv_out, 0, key_dim).reshape(&[b, s, hk, dk]);
        let k = slice_inner(ctx, &conv_out, key_dim, key_dim).reshape(&[b, s, hk, dk]);
        let v = slice_inner(ctx, &conv_out, 2 * key_dim, cfg.value_dim()).reshape(&[b, s, hv, dv]);

        // The Dk**-0.5 readout scale, folded into q and k as mlx-lm's
        // gated_delta contract requires.
        let inv_scale = (dk as f32).powf(-0.5);
        let q = scalar_mul(ctx, &rms_norm(ctx, &q, None, 1e-6), inv_scale * inv_scale);
        let k = scalar_mul(ctx, &rms_norm(ctx, &k, None, 1e-6), inv_scale);

        let state = cache.state_or_zeros(ctx, cfg, b);
        let (out, new_state) = gated_delta_update(
            ctx, &q, &k, &v, &aa, &bb, &self.a_log, &self.dt_bias, &state, None,
        );
        cache.state = Some(new_state);
        cache.offset += s;

        // RMSNormGated: normalise, then gate with z in float32.
        let normed = rms_norm(ctx, &out, Some(&self.norm), cfg.rms_norm_eps);
        let gated = precise_swiglu(ctx, &z, &normed);
        self.out_proj.matmul(ctx, &gated.reshape(&[b, s, hv * dv]))
    }
}

impl GatedDeltaNet {
    /// `forward`, stepping the recurrence one token at a time and keeping the
    /// `(conv window, recurrent state)` after each.
    ///
    /// Everything up to the recurrence is unchanged -- the projections and the
    /// causal conv still run over the whole window in one go, which is what
    /// makes this affordable. Only the `gated_delta_update` call is split, and
    /// that split is exact: the kernel's `for (t = 0; t < T; ++t)` carries the
    /// state in float32 registers, so writing it out and reading it back
    /// between tokens changes nothing.
    pub fn forward_capture(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        x: &Array,
        cache: &mut super::cache::GdnCache,
    ) -> (Array, Vec<(Array, Array)>) {
        let (b, s) = (x.shape[0], x.shape[1]);
        let hk = cfg.linear_num_key_heads;
        let hv = cfg.linear_num_value_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;
        let key_dim = cfg.key_dim();
        let kernel = cfg.linear_conv_kernel_dim;

        let qkv = self.in_proj_qkv.matmul(ctx, x);
        let z = self.in_proj_z.matmul(ctx, x).reshape(&[b, s, hv, dv]);
        let bb = crate::ops::matmul::matmul(ctx, &x.reshape(&[b * s, x.dim(-1)]), &self.in_proj_b, true)
            .reshape(&[b, s, hv]);
        let aa = crate::ops::matmul::matmul(ctx, &x.reshape(&[b * s, x.dim(-1)]), &self.in_proj_a, true)
            .reshape(&[b, s, hv]);

        let conv_state = cache.conv_or_zeros(ctx, cfg, b, qkv.dtype);
        let conv_input = concat_time(ctx, &conv_state, &qkv);
        let n_keep = kernel - 1;
        let conv_out = silu(ctx, &depthwise_conv1d(ctx, &conv_input, &self.conv1d));

        let q = slice_inner(ctx, &conv_out, 0, key_dim).reshape(&[b, s, hk, dk]);
        let k = slice_inner(ctx, &conv_out, key_dim, key_dim).reshape(&[b, s, hk, dk]);
        let v = slice_inner(ctx, &conv_out, 2 * key_dim, cfg.value_dim()).reshape(&[b, s, hv, dv]);

        let inv_scale = (dk as f32).powf(-0.5);
        let q = scalar_mul(ctx, &rms_norm(ctx, &q, None, 1e-6), inv_scale * inv_scale);
        let k = scalar_mul(ctx, &rms_norm(ctx, &k, None, 1e-6), inv_scale);

        let mut state = cache.state_or_zeros(ctx, cfg, b);
        let mut snaps: Vec<(Array, Array)> = Vec::with_capacity(s);
        let mut outs: Vec<Array> = Vec::with_capacity(s);
        for t in 0..s {
            let (out_t, next) = gated_delta_update(
                ctx,
                &slice_head(ctx, &q, t),
                &slice_head(ctx, &k, t),
                &slice_head(ctx, &v, t),
                &slice_time(ctx, &aa, t, 1),
                &slice_time(ctx, &bb, t, 1),
                &self.a_log,
                &self.dt_bias,
                &state,
                None,
            );
            // The conv window after token t is the last `kernel - 1` rows of
            // everything consumed up to and including it.
            let conv_t = slice_time(ctx, &conv_input, t + 1, n_keep);
            snaps.push((conv_t, next.clone()));
            outs.push(out_t);
            state = next;
        }
        cache.conv = Some(slice_time(ctx, &conv_input, s, n_keep));
        cache.state = Some(state);
        cache.offset += s;

        let out = concat_all_time(ctx, &outs);
        let normed = rms_norm(ctx, &out, Some(&self.norm), cfg.rms_norm_eps);
        let gated = precise_swiglu(ctx, &z, &normed);
        let y = self.out_proj.matmul(ctx, &gated.reshape(&[b, s, hv * dv]));
        (y, snaps)
    }
}

/// `x[:, t:t + 1]` for a `[B, T, H, D]` array, which is contiguous, so this is
/// a view rather than a copy.
fn slice_head(ctx: &Context, x: &Array, t: usize) -> Array {
    let _ = ctx;
    let (b, tt, h, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    assert_eq!(b, 1, "the speculative window is single-batch");
    assert!(t < tt);
    x.reshape(&[tt, h * d]).slice_leading(t, 1).reshape(&[1, 1, h, d])
}

/// `mx.concatenate(parts, axis=1)`.
fn concat_all_time(ctx: &Context, parts: &[Array]) -> Array {
    assert!(!parts.is_empty());
    let mut it = parts.iter();
    let mut acc = it.next().unwrap().clone();
    for p in it {
        acc = concat_time_4d(ctx, &acc, p);
    }
    acc
}

/// `concat_time` for the `[B, T, H, D]` recurrence output, which is the same
/// move over a flattened inner dimension.
fn concat_time_4d(ctx: &Context, a: &Array, b: &Array) -> Array {
    let (bs, h, d) = (a.shape[0], a.shape[2], a.shape[3]);
    let (ta, tb) = (a.shape[1], b.shape[1]);
    let joined = concat_time(
        ctx,
        &a.reshape(&[bs, ta, h * d]),
        &b.reshape(&[bs, tb, h * d]),
    );
    joined.reshape(&[bs, ta + tb, h, d])
}

/// `x[:, -n:, :]` for a `[B, T, D]` array.
fn keep_tail(ctx: &Context, x: &Array, n: usize) -> Array {
    slice_time(ctx, x, x.shape[1] - n, n)
}

/// `x[:, start:start + n, :]` for a `[B, T, D]` array.
pub fn slice_time(ctx: &Context, x: &Array, start: usize, n: usize) -> Array {
    let (b, t, d) = (x.shape[0], x.shape[1], x.shape[2]);
    assert!(start + n <= t);
    let out = Array::empty(ctx, &[b, n, d], x.dtype);
    let total = b * n * d;
    let tg = 256.min(total.next_power_of_two()).max(1);
    let tname = match x.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no slice kernel for {}", d.name()),
    };
    // One row per (batch, time): take the last n of t.
    ctx.kernel(&format!("slice_inner_{tname}"))
        .arr(x)
        .out(&out)
        .bytes(&((t * d) as u32))
        .bytes(&((n * d) as u32))
        .bytes(&((start * d) as u32))
        .threads((total, 1, 1), (tg, 1, 1));
    out
}
