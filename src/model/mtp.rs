//! The MTP head: `mtplx/mtp_patch.py`'s `_MTPModule` and `_mtp_core`.
//!
//! One appended NextN predictor, sitting on top of the finished trunk:
//!
//! ```text
//!   e = pre_fc_norm_embedding(embed_tokens(next_token_ids))
//!   h = pre_fc_norm_hidden(trunk_hidden)
//!   x = fc(concat([e, h], axis=-1))              2H -> H
//!   x = layers[0](x, mask, cache=mtp_cache[0])   one FULL-attention block
//!   logits = lm_head(norm(x))
//! ```
//!
//! and `hidden` for the next draft level is `norm(x)` -- the `post_norm`
//! variant. The contract that picks all of this is in `mtplx_runtime.json`:
//!
//! ```json
//!   "mtp_contract": {
//!     "base_hidden_variant": "post_norm",   <- what the trunk feeds in
//!     "hidden_variant":      "post_norm",   <- what the head feeds forward
//!     "concat_order":        "embedding_hidden",
//!     "mtp_position_mode":   "cache",       <- no explicit position offset
//!     "mtp_prequantized":    false
//!   }
//! ```
//!
//! which are `MTPContract`'s defaults, so this artifact takes the generic
//! `inject_mtp_support` path, not `qwen3_5_mtp_patch.py`'s -- that one is
//! gated on `model_type == "qwen3_5_mtp"` and this config says
//! `qwen3_5_moe`. The difference matters: the qwen3_5-specific patch hardcodes
//! `pre_norm`, the generic one takes the contract's `post_norm`.
//!
//! `mtp_prequantized: false` and a sidecar whose 785 tensors are all BF16 mean
//! `_quantize_mtp_module` returns without doing anything, so every projection
//! here is a dense `matmul` and the 256-expert block is an unquantized
//! `SwitchGLU` -- `gather_mv`, not `gather_qmv`.
//!
//! The block itself is `qwen3_5.DecoderLayer(args, layer_idx=interval - 1)`,
//! i.e. the full-attention layer the trunk builds at index 3, 7, 11, ...  so
//! the attention below is the same code as `qwen3_5::Attention::forward` with
//! the quantized projections swapped for dense ones.

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::elementwise::{add, sigmoid_gate};
use crate::ops::embedding::embedding;
use crate::ops::layout::{concat_inner, slice_inner, transpose_hl, transpose_lh};
use crate::ops::matmul::matmul;
use crate::ops::moe::sparse_moe_block_dense;
use crate::ops::norm::rms_norm;
use crate::ops::linear::Linear;
use crate::ops::quantized::QuantMatrix;
use crate::ops::rope::rope;
use crate::weights::WeightStore;

use super::cache::KvCache;
use super::config::TextConfig;

pub struct MtpAttention {
    pub q_proj: Array,
    pub k_proj: Array,
    pub v_proj: Array,
    pub o_proj: Array,
    pub q_norm: Array,
    pub k_norm: Array,
}

pub struct MtpMoe {
    pub gate: Array,
    pub switch_gate: Array,
    pub switch_up: Array,
    pub switch_down: Array,
    pub shared_gate: Array,
    pub shared_up: Array,
    pub shared_down: Array,
    pub shared_expert_gate: Array,
}

pub struct MtpHead {
    pub pre_fc_norm_embedding: Array,
    pub pre_fc_norm_hidden: Array,
    pub fc: Array,
    pub input_layernorm: Array,
    pub post_attention_layernorm: Array,
    pub attn: MtpAttention,
    pub mlp: MtpMoe,
    pub norm: Array,
}

impl MtpHead {
    /// Read `mtp/weights.safetensors`.
    ///
    /// The sidecar stores the MoE as 768 numbered `experts.N.*` tensors;
    /// `stacked_experts` folds them into the `[256, out, in]` shape a
    /// `SwitchLinear` holds, which is what `sanitize` /
    /// `stack_numbered_experts` do on the Python side.
    pub fn load(ctx: &Context, ws: &mut WeightStore, cfg: &TextConfig) -> MtpHead {
        assert_eq!(cfg.mtp_layers, 1, "only a depth-1 MTP predictor is ported");
        let p = "mtp";
        let lp = "mtp.layers.0";
        let e = cfg.num_experts;
        let t = |ws: &mut WeightStore, n: &str| ws.tensor(ctx, n);
        MtpHead {
            pre_fc_norm_embedding: t(ws, &format!("{p}.pre_fc_norm_embedding.weight")),
            pre_fc_norm_hidden: t(ws, &format!("{p}.pre_fc_norm_hidden.weight")),
            fc: t(ws, &format!("{p}.fc.weight")),
            input_layernorm: t(ws, &format!("{lp}.input_layernorm.weight")),
            post_attention_layernorm: t(ws, &format!("{lp}.post_attention_layernorm.weight")),
            attn: MtpAttention {
                q_proj: t(ws, &format!("{lp}.self_attn.q_proj.weight")),
                k_proj: t(ws, &format!("{lp}.self_attn.k_proj.weight")),
                v_proj: t(ws, &format!("{lp}.self_attn.v_proj.weight")),
                o_proj: t(ws, &format!("{lp}.self_attn.o_proj.weight")),
                q_norm: t(ws, &format!("{lp}.self_attn.q_norm.weight")),
                k_norm: t(ws, &format!("{lp}.self_attn.k_norm.weight")),
            },
            mlp: MtpMoe {
                gate: t(ws, &format!("{lp}.mlp.gate.weight")),
                switch_gate: ws.stacked_experts(
                    ctx,
                    &format!("{lp}.mlp.experts"),
                    "gate_proj.weight",
                    e,
                ),
                switch_up: ws.stacked_experts(
                    ctx,
                    &format!("{lp}.mlp.experts"),
                    "up_proj.weight",
                    e,
                ),
                switch_down: ws.stacked_experts(
                    ctx,
                    &format!("{lp}.mlp.experts"),
                    "down_proj.weight",
                    e,
                ),
                shared_gate: t(ws, &format!("{lp}.mlp.shared_expert.gate_proj.weight")),
                shared_up: t(ws, &format!("{lp}.mlp.shared_expert.up_proj.weight")),
                shared_down: t(ws, &format!("{lp}.mlp.shared_expert.down_proj.weight")),
                shared_expert_gate: t(ws, &format!("{lp}.mlp.shared_expert_gate.weight")),
            },
            norm: t(ws, &format!("{p}.norm.weight")),
        }
    }

    /// `_mtp_core` + `mtp_forward`, returning `(logits, hidden)` with
    /// `hidden` the `post_norm` variant the contract asks for.
    ///
    /// `trunk_hidden` is `[1, T, H]`, `next_ids` is the `T` token ids whose
    /// embeddings pair with those hidden rows.
    pub fn forward(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        embed_tokens: &QuantMatrix,
        lm_head: &Linear,
        trunk_hidden: &Array,
        next_ids: &Array,
        cache: &mut KvCache,
    ) -> (Array, Array) {
        let hidden = self.hidden(ctx, cfg, embed_tokens, trunk_hidden, next_ids, cache);
        let logits = lm_head.matmul(ctx, &hidden);
        (logits, hidden)
    }

    /// Everything up to and including `mtp.norm`.
    pub fn hidden(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        embed_tokens: &QuantMatrix,
        trunk_hidden: &Array,
        next_ids: &Array,
        cache: &mut KvCache,
    ) -> Array {
        let t = next_ids.numel();
        let h_size = cfg.hidden_size;
        assert_eq!(trunk_hidden.numel(), t * h_size, "one hidden row per token");

        let emb = embedding(ctx, embed_tokens, next_ids).reshape(&[1, t, h_size]);
        let e = rms_norm(ctx, &emb, Some(&self.pre_fc_norm_embedding), cfg.rms_norm_eps);
        let h = rms_norm(
            ctx,
            &trunk_hidden.reshape(&[1, t, h_size]),
            Some(&self.pre_fc_norm_hidden),
            cfg.rms_norm_eps,
        );
        // concat_order "embedding_hidden".
        let joined = concat_inner(ctx, &e, &h);
        let x = matmul(ctx, &joined.reshape(&[t, 2 * h_size]), &self.fc, true)
            .reshape(&[1, t, h_size]);

        let x = self.layer(ctx, cfg, &x, cache);
        rms_norm(ctx, &x, Some(&self.norm), cfg.rms_norm_eps)
    }

    /// `qwen3_5.DecoderLayer.__call__` for the full-attention branch.
    fn layer(&self, ctx: &Context, cfg: &TextConfig, x: &Array, cache: &mut KvCache) -> Array {
        let normed = rms_norm(ctx, x, Some(&self.input_layernorm), cfg.rms_norm_eps);
        let r = self.attention(ctx, cfg, &normed, cache);
        let h = add(ctx, x, &r);
        let normed2 = rms_norm(ctx, &h, Some(&self.post_attention_layernorm), cfg.rms_norm_eps);
        let m = sparse_moe_block_dense(
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

    /// The same attention as the trunk's full-attention layers, with dense
    /// projections. `mtp_position_mode` is `"cache"`, so the rotary offset is
    /// the MTP cache's own offset and no explicit position is passed.
    fn attention(
        &self,
        ctx: &Context,
        cfg: &TextConfig,
        x: &Array,
        cache: &mut KvCache,
    ) -> Array {
        let (b, l) = (x.shape[0], x.shape[1]);
        let hd = cfg.head_dim;
        let hq = cfg.num_attention_heads;
        let hkv = cfg.num_key_value_heads;
        let scale = (hd as f32).powf(-0.5);
        let x2 = x.reshape(&[b * l, cfg.hidden_size]);

        let qg = matmul(ctx, &x2, &self.attn.q_proj, true).reshape(&[b, l, hq, 2 * hd]);
        let queries = slice_inner(ctx, &qg, 0, hd);
        let gate = slice_inner(ctx, &qg, hd, hd).reshape(&[b, l, hq * hd]);

        let keys = matmul(ctx, &x2, &self.attn.k_proj, true).reshape(&[b, l, hkv, hd]);
        let values = matmul(ctx, &x2, &self.attn.v_proj, true).reshape(&[b, l, hkv, hd]);

        let queries = rms_norm(ctx, &queries, Some(&self.attn.q_norm), cfg.rms_norm_eps);
        let keys = rms_norm(ctx, &keys, Some(&self.attn.k_norm), cfg.rms_norm_eps);
        let queries = transpose_lh(ctx, &queries);
        let keys = transpose_lh(ctx, &keys);
        let values = transpose_lh(ctx, &values);

        let offset = cache.offset;
        let dims = cfg.rope_dims();
        let queries = rope(ctx, &queries, dims, cfg.rope_theta, 1.0, offset);
        let keys = rope(ctx, &keys, dims, cfg.rope_theta, 1.0, offset);

        let (k_all, v_all) = cache.update_and_fetch(ctx, &keys, &values);
        let cap = cache.capacity();
        let out = crate::ops::sdpa::attention_with_cache(
            ctx, &queries, &k_all, &v_all, cap, scale, l > 1,
        );
        let out = transpose_hl(ctx, &out).reshape(&[b * l, hq * hd]);
        let gated = sigmoid_gate(ctx, &gate.reshape(&[b * l, hq * hd]), &out);
        matmul(ctx, &gated, &self.attn.o_proj, true).reshape(&[b, l, cfg.hidden_size])
    }
}

/// `make_mtp_cache()`: one `KVCache`, because the head is one layer deep.
pub fn make_mtp_cache() -> Vec<KvCache> {
    vec![KvCache::new()]
}

/// The sidecar the head lives in, as `mlx_lm_extra_tensors.mtp_file` names it.
pub fn sidecar_path(model_dir: &std::path::Path) -> std::path::PathBuf {
    let raw = std::fs::read(model_dir.join("config.json")).expect("config.json");
    let cfg = crate::json::parse(&raw).expect("config.json");
    let rel = cfg
        .path("mlx_lm_extra_tensors.mtp_file")
        .and_then(|v| v.as_str())
        .unwrap_or("mtp/weights.safetensors");
    model_dir.join(rel)
}

/// Bytes the head occupies, for the loader's report.
pub fn head_dtype_is_bf16(ws: &WeightStore) -> bool {
    ws.dtype_of("mtp.fc.weight") == Some(Dtype::BF16)
}
