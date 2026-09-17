//! `TextModelArgs` from `vendor/mlx-lm/mlx_lm/models/qwen3_5.py`, read out of
//! the artifact's `config.json`.

use std::path::Path;

use crate::json::{self, Value};

#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub full_attention_interval: usize,
    pub attention_bias: bool,

    // Linear attention (GatedDeltaNet)
    pub linear_num_value_heads: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,

    // MoE
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,

    // RoPE
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub rope_type: String,

    pub tie_word_embeddings: bool,
    /// Number of appended MTP predictors (`num_nextn_predict_layers`).
    pub mtp_layers: usize,
}

impl TextConfig {
    pub fn load(model_dir: &Path) -> Result<TextConfig, String> {
        let raw = std::fs::read(model_dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let cfg = json::parse(&raw).map_err(|e| format!("config.json: {e}"))?;
        TextConfig::from_value(&cfg)
    }

    pub fn from_value(cfg: &Value) -> Result<TextConfig, String> {
        // `ModelArgs.from_dict` uses `text_config` when present and treats the
        // top level as the text config otherwise.
        let t = cfg.get("text_config").unwrap_or(cfg);
        let u = |k: &str| -> Result<usize, String> {
            t.get(k).and_then(|v| v.as_usize()).ok_or_else(|| format!("missing {k}"))
        };
        let f = |k: &str, d: f32| t.get(k).and_then(|v| v.as_f32()).unwrap_or(d);
        let b = |k: &str, d: bool| t.get(k).and_then(|v| v.as_bool()).unwrap_or(d);

        let hidden_size = u("hidden_size")?;
        let num_attention_heads = u("num_attention_heads")?;
        let head_dim = t
            .get("head_dim")
            .and_then(|v| v.as_usize())
            .unwrap_or(hidden_size / num_attention_heads);

        // `__post_init__` pulls the rope settings out of `rope_parameters`.
        let rope = t.get("rope_parameters");
        let rope_type = rope
            .and_then(|r| r.get("type").or_else(|| r.get("rope_type")))
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .and_then(|v| v.as_f32())
            .unwrap_or_else(|| f("rope_theta", 100000.0));
        let partial_rotary_factor = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(|v| v.as_f32())
            .unwrap_or_else(|| f("partial_rotary_factor", 0.25));

        Ok(TextConfig {
            hidden_size,
            num_hidden_layers: u("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: u("num_key_value_heads")?,
            head_dim,
            rms_norm_eps: f("rms_norm_eps", 1e-6),
            vocab_size: u("vocab_size")?,
            max_position_embeddings: t
                .get("max_position_embeddings")
                .and_then(|v| v.as_usize())
                .unwrap_or(131072),
            full_attention_interval: t
                .get("full_attention_interval")
                .and_then(|v| v.as_usize())
                .unwrap_or(4),
            attention_bias: b("attention_bias", false),
            linear_num_value_heads: u("linear_num_value_heads")?,
            linear_num_key_heads: u("linear_num_key_heads")?,
            linear_key_head_dim: u("linear_key_head_dim")?,
            linear_value_head_dim: u("linear_value_head_dim")?,
            linear_conv_kernel_dim: u("linear_conv_kernel_dim")?,
            num_experts: t.get("num_experts").and_then(|v| v.as_usize()).unwrap_or(0),
            num_experts_per_tok: t
                .get("num_experts_per_tok")
                .and_then(|v| v.as_usize())
                .unwrap_or(0),
            moe_intermediate_size: t
                .get("moe_intermediate_size")
                .and_then(|v| v.as_usize())
                .unwrap_or(0),
            shared_expert_intermediate_size: t
                .get("shared_expert_intermediate_size")
                .and_then(|v| v.as_usize())
                .unwrap_or(0),
            // mlx-lm's TextModelArgs defaults this to True; qwen3_next's
            // ModelArgs defaults it to False, and qwen3_5 uses the former.
            norm_topk_prob: b("norm_topk_prob", true),
            rope_theta,
            partial_rotary_factor,
            rope_type,
            tie_word_embeddings: b("tie_word_embeddings", false),
            mtp_layers: cfg
                .get("num_nextn_predict_layers")
                .or_else(|| t.get("num_nextn_predict_layers"))
                .and_then(|v| v.as_usize())
                .unwrap_or(0),
        })
    }

    /// `(layer_idx + 1) % full_attention_interval != 0` -- the linear-attention
    /// test from `qwen3_5.DecoderLayer`.
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval != 0
    }

    /// The rotated width: `int(head_dim * partial_rotary_factor)`.
    pub fn rope_dims(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    pub fn key_dim(&self) -> usize {
        self.linear_key_head_dim * self.linear_num_key_heads
    }

    pub fn value_dim(&self) -> usize {
        self.linear_value_head_dim * self.linear_num_value_heads
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_this_artifacts_config() {
        let dir = crate::refs::model_dir();
        if !dir.exists() {
            return;
        }
        let c = TextConfig::load(&dir).unwrap();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.num_hidden_layers, 40);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.vocab_size, 248320);
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 512);
        assert_eq!(c.rope_dims(), 64);
        assert_eq!(c.key_dim(), 2048);
        assert_eq!(c.value_dim(), 4096);
        assert_eq!(c.conv_dim(), 8192);
        assert_eq!(c.mtp_layers, 1);
        assert!(!c.tie_word_embeddings);

        // 10 of the 40 layers are full attention: every 4th.
        let full: Vec<usize> = (0..40).filter(|i| !c.is_linear_layer(*i)).collect();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);

        // And the config's own layer_types must agree with that rule.
        let raw = std::fs::read(dir.join("config.json")).unwrap();
        let cfg = crate::json::parse(&raw).unwrap();
        let types = cfg.path("text_config.layer_types").unwrap().as_arr().unwrap();
        for (i, t) in types.iter().enumerate() {
            let is_linear = t.as_str().unwrap() == "linear_attention";
            assert_eq!(is_linear, c.is_linear_layer(i), "layer {i}");
        }
    }
}
