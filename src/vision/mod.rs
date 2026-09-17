//! The Qwen3-VL vision tower, its preprocessing and the prompt splice.
//!
//! Ported from `vendor/MTPLX/mtplx/vision/`, which is itself mlx-vlm's
//! `qwen3_vl` adapted to this checkpoint's naming. mlx-lm has no vision path
//! at all -- `qwen3_vl.Model.sanitize` drops every `vision_tower.*` tensor --
//! so MTPLX is the only reference for this half of the model.
//!
//! ```text
//!   mod.rs          vision/__init__.py       the config and the token ids
//!   preprocess.rs   vision/processing.py     Qwen2VLImageProcessorFast
//!   resample.rs     Pillow's Resample.c      the bicubic that sits inside it
//!   tower.rs        vision/qwen3_vl_tower.py the 27-block ViT
//!   splice.rs       vision/splice.py         image rows over `<|image_pad|>`
//! ```
//!
//! What the tower adds over the trunk, all of it new in this port: a 3D patch
//! embedding (`mx.conv3d`), an interpolated learned position embedding, a 2D
//! rotary embedding, LayerNorm rather than RMSNorm, two flavours of GELU,
//! biased linears (`mx.addmm`), mlx's *full* fused attention kernel, and a
//! spatial merger that folds 2x2 patches into one language token.

pub mod preprocess;
pub mod resample;
pub mod splice;
pub mod tower;

use std::path::Path;

use crate::json::{self, Value};

/// `Qwen3VLVisionConfig` from `qwen3_vl_tower.py`, read out of
/// `config.json`'s `vision_config`.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

/// `VisionSpec` from `vision/__init__.py`: the token ids the prompt uses to
/// mark where image rows go. `config.json` carries all four explicitly here,
/// so the family defaults never apply.
#[derive(Debug, Clone, Copy)]
pub struct VisionSpec {
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl VisionConfig {
    pub fn load(model_dir: &Path) -> Result<(VisionConfig, VisionSpec), String> {
        let raw = std::fs::read(model_dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let cfg = json::parse(&raw).map_err(|e| format!("config.json: {e}"))?;
        VisionConfig::from_value(&cfg)
    }

    pub fn from_value(cfg: &Value) -> Result<(VisionConfig, VisionSpec), String> {
        let v = cfg.get("vision_config").ok_or("config.json has no vision_config")?;
        let u = |k: &str, d: usize| v.get(k).and_then(|x| x.as_usize()).unwrap_or(d);
        let deepstack: Vec<usize> = v
            .get("deepstack_visual_indexes")
            .and_then(|x| x.as_arr())
            .map(|a| a.iter().filter_map(|x| x.as_usize()).collect())
            .unwrap_or_default();
        // mlx-vlm's qwen3_5 VisionConfig hard-disables deepstack, and this
        // checkpoint ships an empty list and no `deepstack_merger_list`
        // weights, so the per-tap mergers are never constructed.
        assert!(
            deepstack.is_empty(),
            "deepstack taps are not ported; this artifact ships none"
        );

        let config = VisionConfig {
            depth: u("depth", 27),
            hidden_size: u("hidden_size", 1152),
            intermediate_size: u("intermediate_size", 4304),
            out_hidden_size: u("out_hidden_size", 5120),
            num_heads: u("num_heads", 16),
            patch_size: u("patch_size", 16),
            spatial_merge_size: u("spatial_merge_size", 2),
            temporal_patch_size: u("temporal_patch_size", 2),
            in_channels: u("in_channels", 3),
            num_position_embeddings: u("num_position_embeddings", 2304),
            deepstack_visual_indexes: deepstack,
        };

        let t = |k: &str, d: u32| {
            cfg.get(k).and_then(|x| x.as_usize()).map(|x| x as u32).unwrap_or(d)
        };
        let spec = VisionSpec {
            image_token_id: t("image_token_id", 248056),
            video_token_id: t("video_token_id", 248057),
            vision_start_token_id: t("vision_start_token_id", 248045),
            vision_end_token_id: t("vision_end_token_id", 248046),
        };
        Ok((config, spec))
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// `image_pad_token_count`: how many language tokens one image's patch
    /// grid expands to.
    pub fn pad_token_count(&self, grid: (usize, usize, usize)) -> usize {
        let m = self.spatial_merge_size;
        (grid.0 * grid.1 * grid.2) / (m * m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_this_artifacts_vision_config() {
        let dir = crate::refs::model_dir();
        if !dir.exists() {
            return;
        }
        let (c, s) = VisionConfig::load(&dir).unwrap();
        assert_eq!(c.depth, 27);
        assert_eq!(c.hidden_size, 1152);
        assert_eq!(c.intermediate_size, 4304);
        assert_eq!(c.num_heads, 16);
        assert_eq!(c.head_dim(), 72);
        assert_eq!(c.patch_size, 16);
        assert_eq!(c.spatial_merge_size, 2);
        assert_eq!(c.temporal_patch_size, 2);
        assert_eq!(c.num_position_embeddings, 2304);
        // The tower's output is the *text* hidden size, not mlx-vlm's 5120
        // family default.
        assert_eq!(c.out_hidden_size, 2048);
        assert_eq!(s.image_token_id, 248056);
        assert_eq!(s.vision_start_token_id, 248053);
        assert_eq!(s.vision_end_token_id, 248054);
    }
}
