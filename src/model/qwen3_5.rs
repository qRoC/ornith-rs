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
use crate::ops::elementwise::{add, scalar_mul, sigmoid_gate, silu};
use crate::ops::embedding::embedding;
use crate::ops::gdn::{
    beta, compute_g, concat_time, depthwise_conv1d, gated_delta_kernel, gated_delta_kernel_into,
    gated_delta_update, gdn_front,
};
use crate::ops::layout::{slice_inner, transpose_hl, transpose_lh};
use crate::ops::moe::sparse_moe_block;
use crate::ops::norm::{add_rms_norm, rms_norm, rms_norm_gated};
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

    /// The norm the add closing layer `i` feeds: the next layer's input norm,
    /// or the model's own final norm after the last one.
    fn next_norm(&self, i: usize) -> &Array {
        match self.layers.get(i + 1) {
            Some(l) => &l.input_layernorm,
            None => &self.norm,
        }
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
        let mut gdn: Vec<Option<GdnSnaps>> = (0..caches.len()).map(|_| None).collect();
        let mut x = embedding(ctx, &self.embed_tokens, ids).reshape(&[1, n, self.cfg.hidden_size]);
        let mut normed =
            rms_norm(ctx, &x, Some(&self.layers[0].input_layernorm), self.cfg.rms_norm_eps);
        for (i, layer) in self.layers.iter().enumerate() {
            let (h, m) = match &layer.mixer {
                Mixer::Linear(_) => {
                    let (h, m, snaps) =
                        layer.forward_capture_parts(ctx, &self.cfg, &x, &normed, &mut caches[i]);
                    gdn[i] = Some(snaps);
                    (h, m)
                }
                Mixer::Full(_) => {
                    layer.forward_parts(ctx, &self.cfg, &x, &normed, &mut caches[i])
                }
            };
            let w = self.next_norm(i);
            let (nx, nn) = add_rms_norm(ctx, &h, &m, Some(w), self.cfg.rms_norm_eps);
            x = nx;
            normed = nn;
        }
        (
            normed,
            x,
            TrunkCapture {
                gdn,
                base_offsets,
                tokens: n,
                n_keep: self.cfg.linear_conv_kernel_dim - 1,
            },
        )
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
        let mut x = embeddings.reshape(&[1, n, self.cfg.hidden_size]);
        // Only the first layer's input norm stands on its own; every other
        // one is fused into the add that closes the layer before it, and so
        // is the model's final norm.
        let mut normed =
            rms_norm(ctx, &x, Some(&self.layers[0].input_layernorm), self.cfg.rms_norm_eps);
        for (i, layer) in self.layers.iter().enumerate() {
            // Where one layer's dispatches end and the next begin, for a
            // trace that wants to be read a layer at a time. Free unless a
            // trace is recording.
            ctx.prof_section(
                match layer.mixer {
                    Mixer::Linear(_) => "layer.linear",
                    Mixer::Full(_) => "layer.full",
                },
                i,
            );
            let (h, m) = layer.forward_parts(ctx, &self.cfg, &x, &normed, &mut caches[i]);
            let w = self.next_norm(i);
            let (nx, nn) = add_rms_norm(ctx, &h, &m, Some(w), self.cfg.rms_norm_eps);
            x = nx;
            normed = nn;
        }
        (normed, x)
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
/// each step already allocates its own state, so *that* snapshot is a
/// reference rather than a copy.
///
/// **The conv window was not, until Phase 29.** It used to be sliced out of
/// `conv_input` once per token -- `T` dispatches a layer, of which `rewind`
/// reads exactly one, since a cycle commits one prefix. `conv_input` is
/// already materialised and is 114 KB a layer against the 49 KB a snapshot,
/// so keeping it whole and slicing at rewind is both fewer dispatches and
/// less memory.
pub struct TrunkCapture {
    /// Per gated-delta layer: the whole conv input, and the recurrent state
    /// after each token. `None` for a full-attention layer.
    gdn: Vec<Option<GdnSnaps>>,
    /// Each cache's offset before the forward.
    base_offsets: Vec<usize>,
    pub tokens: usize,
    /// `linear_conv_kernel_dim - 1`: the rows of `conv_input` a cache keeps.
    n_keep: usize,
}

/// What one gated-delta layer hands the capture.
pub struct GdnSnaps {
    /// `[B, T + n_keep, qkv]`: the conv window the layer entered with,
    /// followed by this window's tokens. Any token's conv state is a slice.
    conv_input: Array,
    /// The recurrent state after each token, as the kernel wrote it.
    states: Vec<Array>,
}

impl TrunkCapture {
    /// Rewind every cache to the state it would have had after `keep` of the
    /// `tokens` this forward consumed.
    pub fn rewind(&self, ctx: &Context, caches: &mut [LayerCache], keep: usize) {
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
                    let snaps = self.gdn[i].as_ref().expect("a gdn layer without snapshots");
                    // The conv window after token `keep - 1` is the last
                    // `n_keep` rows of everything consumed up to and including
                    // it -- one slice, taken now that `keep` is known, instead
                    // of `T` slices taken when it was not.
                    c.conv = Some(slice_time(ctx, &snaps.conv_input, keep, self.n_keep));
                    c.state = Some(snaps.states[keep - 1].clone());
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
        let (h, m) = self.forward_parts(ctx, cfg, x, &normed, cache);
        add(ctx, &h, &m)
    }

    /// The layer from an input already normalised by the caller, and without
    /// its last add: `(h, m)` are the residual and the MoE block's output, so
    /// the layer's result is `h + m`.
    ///
    /// The split is here because of what comes next. Every layer ends by
    /// adding `m` to the residual and the next one begins by normalising the
    /// sum, which is one dispatch and one barrier epoch each where it can be
    /// one of each -- see [`ops::norm::add_rms_norm`]. Handing the caller the
    /// two halves is what lets it fuse across the boundary; nothing else
    /// about the layer moves.
    fn forward_parts(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        x: &Array,
        normed: &Array,
        cache: &mut LayerCache,
    ) -> (Array, Array) {
        let r = match &self.mixer {
            Mixer::Linear(g) => g.forward(ctx, cfg, normed, cache.as_gdn()),
            Mixer::Full(a) => a.forward(ctx, cfg, normed, cache.as_kv()),
        };
        let (h, normed2) =
            add_rms_norm(ctx, x, &r, Some(&self.post_attention_layernorm), cfg.rms_norm_eps);
        let m = self.moe(ctx, cfg, &normed2);
        (h, m)
    }

    fn moe(&self, ctx: &Context, cfg: &TextConfig, normed2: &Array) -> Array {
        sparse_moe_block(
            ctx,
            normed2,
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
        )
    }
}

impl DecoderLayer {
    /// The gated-delta branch with per-token snapshots.
    fn forward_capture_parts(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        x: &Array,
        normed: &Array,
        cache: &mut LayerCache,
    ) -> (Array, Array, GdnSnaps) {
        let Mixer::Linear(g) = &self.mixer else { panic!("not a gated-delta layer") };
        let (r, snaps) = g.forward_capture(ctx, cfg, normed, cache.as_gdn());
        let (h, normed2) =
            add_rms_norm(ctx, x, &r, Some(&self.post_attention_layernorm), cfg.rms_norm_eps);
        let m = self.moe(ctx, cfg, &normed2);
        (h, m, snaps)
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
        let n_keep = kernel - 1;
        // The Dk**-0.5 readout scale, folded into q and k as mlx-lm's
        // gated_delta contract requires.
        let inv_scale = (dk as f32).powf(-0.5);
        let state = cache.state_or_zeros(ctx, cfg, b);

        // Everything between the projections and the recurrence, in one
        // dispatch where the shapes allow it and ten where they do not.
        let (out, new_state) = match gdn_front(
            ctx,
            &conv_state,
            &qkv,
            &self.conv1d,
            &aa,
            &bb,
            &self.a_log,
            &self.dt_bias,
            hk,
            dk,
            hv,
            dv,
            inv_scale * inv_scale,
            inv_scale,
            1e-6,
            false,
        ) {
            Some(f) => {
                cache.conv = Some(f.conv_next);
                gated_delta_kernel(ctx, &f.q, &f.k, &f.v, &f.g, &f.beta, &state, None)
            }
            None => {
                let conv_input = concat_time(ctx, &conv_state, &qkv);
                cache.conv = Some(keep_tail(ctx, &conv_input, n_keep));
                let conv_out = silu(ctx, &depthwise_conv1d(ctx, &conv_input, &self.conv1d));

                let q = slice_inner(ctx, &conv_out, 0, key_dim).reshape(&[b, s, hk, dk]);
                let k = slice_inner(ctx, &conv_out, key_dim, key_dim).reshape(&[b, s, hk, dk]);
                let v = slice_inner(ctx, &conv_out, 2 * key_dim, cfg.value_dim())
                    .reshape(&[b, s, hv, dv]);

                let q = scalar_mul(ctx, &rms_norm(ctx, &q, None, 1e-6), inv_scale * inv_scale);
                let k = scalar_mul(ctx, &rms_norm(ctx, &k, None, 1e-6), inv_scale);
                gated_delta_update(
                    ctx, &q, &k, &v, &aa, &bb, &self.a_log, &self.dt_bias, &state, None,
                )
            }
        };
        cache.state = Some(new_state);
        cache.offset += s;

        // RMSNormGated: normalise, then gate with z in float32 -- inside
        // `out_proj` where the shapes allow it, since this is the one norm in
        // the model with exactly one consumer.
        let flat = [b, s, hv * dv];
        let (fout, fz) = (out.reshape(&flat), z.reshape(&flat));
        if let Some(y) =
            self.out_proj.rmsgate_matmul(ctx, &fout, &self.norm, &fz, dv, cfg.rms_norm_eps)
        {
            return y;
        }
        let gated = rms_norm_gated(ctx, &out, Some(&self.norm), &z, cfg.rms_norm_eps);
        self.out_proj.matmul(ctx, &gated.reshape(&flat))
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
    ) -> (Array, GdnSnaps) {
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
        let n_keep = kernel - 1;
        let inv_scale = (dk as f32).powf(-0.5);

        // The same one dispatch the decode path takes, over the window
        // rather than one token -- and it keeps the concatenation, which
        // `rewind` slices a cached conv state out of.
        //
        // The gate and the decay come out of it for the whole window rather
        // than once per token. Both are elementwise over `[B, T, Hv]` --
        // `sigmoid` and `compute_g`'s exp -- so a thread computes the same
        // element from the same inputs whichever shape it is handed, and
        // slicing the results per token is free where recomputing them was
        // two dispatches a token a layer.
        let (conv_input, q, k, v, beta_all, g_all) = match gdn_front(
            ctx,
            &conv_state,
            &qkv,
            &self.conv1d,
            &aa,
            &bb,
            &self.a_log,
            &self.dt_bias,
            hk,
            dk,
            hv,
            dv,
            inv_scale * inv_scale,
            inv_scale,
            1e-6,
            true,
        ) {
            Some(f) => (f.conv_input.unwrap(), f.q, f.k, f.v, f.beta, f.g),
            None => {
                let conv_input = concat_time(ctx, &conv_state, &qkv);
                let conv_out = silu(ctx, &depthwise_conv1d(ctx, &conv_input, &self.conv1d));

                let q = slice_inner(ctx, &conv_out, 0, key_dim).reshape(&[b, s, hk, dk]);
                let k = slice_inner(ctx, &conv_out, key_dim, key_dim).reshape(&[b, s, hk, dk]);
                let v = slice_inner(ctx, &conv_out, 2 * key_dim, cfg.value_dim())
                    .reshape(&[b, s, hv, dv]);

                let q = scalar_mul(ctx, &rms_norm(ctx, &q, None, 1e-6), inv_scale * inv_scale);
                let k = scalar_mul(ctx, &rms_norm(ctx, &k, None, 1e-6), inv_scale);
                let beta_all = beta(ctx, &bb);
                let g_all = compute_g(ctx, &self.a_log, &aa, &self.dt_bias);
                (conv_input, q, k, v, beta_all, g_all)
            }
        };

        let mut state = cache.state_or_zeros(ctx, cfg, b);
        let mut states: Vec<Array> = Vec::with_capacity(s);
        // The `T` one-token outputs, written where a concatenation would have
        // copied them: each step is handed row `t` of this buffer, so the
        // window comes out assembled. Same kernel, same arithmetic, different
        // destination address.
        let out = Array::empty(ctx, &[b, s, hv, dv], x.dtype);
        for t in 0..s {
            let (_, next) = gated_delta_kernel_into(
                ctx,
                &slice_head(ctx, &q, t),
                &slice_head(ctx, &k, t),
                &slice_head(ctx, &v, t),
                &row_view(&g_all, t),
                &row_view(&beta_all, t),
                &state,
                None,
                &out_row(&out, t),
            );
            // The conv window after token `t` is a slice of `conv_input`,
            // and `rewind` needs exactly one of them -- so it is not taken
            // here. See `TrunkCapture`.
            states.push(next.clone());
            state = next;
        }
        cache.conv = Some(slice_time(ctx, &conv_input, s, n_keep));
        cache.state = Some(state);
        cache.offset += s;

        let flat = [b, s, hv * dv];
        let y = match self.out_proj.rmsgate_matmul(
            ctx,
            &out.reshape(&flat),
            &self.norm,
            &z.reshape(&flat),
            dv,
            cfg.rms_norm_eps,
        ) {
            Some(y) => y,
            None => {
                let gated = rms_norm_gated(ctx, &out, Some(&self.norm), &z, cfg.rms_norm_eps);
                self.out_proj.matmul(ctx, &gated.reshape(&flat))
            }
        };
        (y, GdnSnaps { conv_input, states })
    }
}

/// Row `t` of a `[B, T, H]` array, as a `[1, 1, H]` view.
///
/// `slice_time` is a kernel because a general `[B, T, D]` slice is strided;
/// at `B = 1` -- which is every speculative window -- one row is contiguous
/// and this is a view, so it costs nothing. Only for values consumed inside
/// the same forward: a cache keeps the copying form, so that nothing it holds
/// points into a pooled buffer.
fn row_view(x: &Array, t: usize) -> Array {
    let (b, tt, h) = (x.shape[0], x.shape[1], x.shape[2]);
    assert_eq!(b, 1, "the speculative window is single-batch");
    assert!(t < tt);
    x.reshape(&[tt, h]).slice_leading(t, 1).reshape(&[1, 1, h])
}

/// Row `t` of a `[B, T, Hv, Dv]` output buffer, as a `[1, 1, Hv, Dv]` view.
///
/// The buffer is contiguous and an `Array` carries its own offset, so this
/// costs nothing and a kernel handed it writes into the middle of the window.
fn out_row(out: &Array, t: usize) -> Array {
    let (tt, hv, dv) = (out.shape[1], out.shape[2], out.shape[3]);
    out.reshape(&[tt, hv * dv]).slice_leading(t, 1).reshape(&[1, 1, hv, dv])
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

/// (`concat_all_time` and `concat_time_4d` lived here until Phase 29. They
/// put the capture's `T` one-token recurrence outputs back together, which is
/// work that only existed because the outputs arrived apart: the kernel now
/// writes each one into its row of the window buffer, so there is nothing
/// left to join. Nothing else called them.)

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
