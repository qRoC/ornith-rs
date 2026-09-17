//! Turning safetensors shards into device arrays.
//!
//! mlx's `load_safetensors` builds a lazy `Load` primitive per tensor that
//! reads the bytes into a fresh allocation through a `ParallelFileReader`. This
//! does the same thing eagerly: map the shard, allocate a device buffer per
//! tensor and copy. The copy is not avoidable -- the artifact packs tensors
//! back to back, so most of them start on odd byte offsets that no Metal
//! buffer can be pointed at (see the safetensors tests).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::quantized::QuantMatrix;
use crate::safetensors::{SafeTensors, TensorInfo, shards_from_index};

pub struct WeightStore {
    st: SafeTensors,
    /// Per-module `{bits, group_size}` from config.json's `quantization`.
    quant: HashMap<String, (usize, usize)>,
    default_quant: (usize, usize),
    cache: HashMap<String, Array>,
}

impl WeightStore {
    /// Open the sharded model plus any extra files (the MTP sidecar).
    pub fn open(model_dir: &Path, extra: &[PathBuf]) -> Result<WeightStore, String> {
        let mut paths = shards_from_index(model_dir)?;
        paths.extend_from_slice(extra);
        let st = SafeTensors::open_many(&paths)?;

        let cfg_raw = std::fs::read(model_dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let cfg = crate::json::parse(&cfg_raw).map_err(|e| format!("config.json: {e}"))?;
        let q = cfg.get("quantization").ok_or("config.json has no quantization block")?;
        let default_quant = (
            q.get("group_size").and_then(|v| v.as_usize()).ok_or("quantization.group_size")?,
            q.get("bits").and_then(|v| v.as_usize()).ok_or("quantization.bits")?,
        );
        let mut quant = HashMap::new();
        for (k, v) in q.as_obj().ok_or("quantization is not an object")? {
            if let Some(o) = v.as_obj() {
                let gs = o.get("group_size").and_then(|v| v.as_usize()).unwrap_or(default_quant.0);
                let bits = o.get("bits").and_then(|v| v.as_usize()).unwrap_or(default_quant.1);
                quant.insert(k.clone(), (gs, bits));
            }
        }
        Ok(WeightStore { st, quant, default_quant, cache: HashMap::new() })
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.get(name).is_some()
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.st.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.st.names()
    }

    /// `(group_size, bits)` for a module path, with the same per-module
    /// override lookup mlx-lm's loader does.
    pub fn quant_config(&self, path: &str) -> (usize, usize) {
        *self.quant.get(path).unwrap_or(&self.default_quant)
    }

    /// Copy one tensor to the device. Repeated calls return the same array.
    pub fn tensor(&mut self, ctx: &Context, name: &str) -> Array {
        if let Some(a) = self.cache.get(name) {
            return a.clone();
        }
        let a = self.load_uncached(ctx, name);
        self.cache.insert(name.to_string(), a.clone());
        a
    }

    /// Copy one tensor without caching it (used for the one-shot tests).
    pub fn load_uncached(&self, ctx: &Context, name: &str) -> Array {
        let info = self
            .st
            .get(name)
            .unwrap_or_else(|| panic!("no tensor named {name:?} in the model"));
        let src = self.st.bytes(info);
        let buf = ctx.alloc_exact(info.nbytes);
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents(), info.nbytes);
        }
        Array { buf: Rc::new(buf), offset: 0, shape: info.shape.clone(), dtype: info.dtype }
    }

    /// Load `<path>.weight` / `.scales` / `.biases` as one quantized matrix.
    pub fn quant_matrix(&mut self, ctx: &Context, path: &str) -> QuantMatrix {
        let (group_size, bits) = self.quant_config(path);
        QuantMatrix {
            w: self.tensor(ctx, &format!("{path}.weight")),
            scales: self.tensor(ctx, &format!("{path}.scales")),
            biases: self.tensor(ctx, &format!("{path}.biases")),
            group_size,
            bits,
        }
    }

    /// Stack `<prefix>.<e>.<suffix>` for `e` in `0..experts` into one
    /// `[experts, out, in]` array.
    ///
    /// mlx-lm reaches this layout through `sanitize`, which rewrites a
    /// checkpoint's numbered `experts.N.*` tensors into the single stacked
    /// `switch_mlp.*` weight a `SwitchLinear` holds (MTPLX calls the same thing
    /// `stack_numbered_experts`). The MTP sidecar stores the numbered form --
    /// 768 of its 785 tensors -- so the stacking happens here.
    pub fn stacked_experts(
        &mut self,
        ctx: &Context,
        prefix: &str,
        suffix: &str,
        experts: usize,
    ) -> Array {
        let key = format!("{prefix}.*.{suffix}");
        if let Some(a) = self.cache.get(&key) {
            return a.clone();
        }
        let first = self
            .st
            .get(&format!("{prefix}.0.{suffix}"))
            .unwrap_or_else(|| panic!("no tensor named {prefix}.0.{suffix}"));
        let dtype = first.dtype;
        let per = first.shape.clone();
        let stride = first.nbytes;
        let mut shape = vec![experts];
        shape.extend_from_slice(&per);

        let buf = ctx.alloc_exact(stride * experts);
        for e in 0..experts {
            let name = format!("{prefix}.{e}.{suffix}");
            let info = self
                .st
                .get(&name)
                .unwrap_or_else(|| panic!("no tensor named {name:?} in the model"));
            assert_eq!(info.shape, per, "{name} has a different shape from expert 0");
            assert_eq!(info.dtype, dtype, "{name} has a different dtype from expert 0");
            let src = self.st.bytes(info);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    buf.contents().add(e * stride),
                    stride,
                );
            }
        }
        let a = Array { buf: Rc::new(buf), offset: 0, shape, dtype };
        self.cache.insert(key, a.clone());
        a
    }

    /// Bytes resident on the device so far.
    pub fn loaded_bytes(&self) -> usize {
        self.cache.values().map(|a| a.nbytes()).sum()
    }

    pub fn dtype_of(&self, name: &str) -> Option<Dtype> {
        self.st.get(name).map(|i| i.dtype)
    }

    /// Start faulting the mappings in; call before a bulk load.
    pub fn prefetch(&self) {
        self.st.will_need();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> PathBuf {
        PathBuf::from(std::env::var("ORNITH_MODEL").unwrap_or_else(|_| {
            "/Users/qroc/.mtplx/models/Shiftedx--ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx".into()
        }))
    }

    #[test]
    fn loads_a_quantized_matrix_with_the_right_config() {
        let dir = model_dir();
        if !dir.exists() {
            return;
        }
        let ctx = Context::new().unwrap();
        let mut ws = WeightStore::open(&dir, &[]).unwrap();

        // 8-bit / group 64, per the per-module override in config.json.
        let qkv = ws.quant_matrix(&ctx, "language_model.model.layers.0.linear_attn.in_proj_qkv");
        assert_eq!((qkv.group_size, qkv.bits), (64, 8));
        assert_eq!(qkv.out_dims(), 8192);
        assert_eq!(qkv.in_dims(), 2048);
        assert_eq!(qkv.w.shape, vec![8192, 512]); // 2048 * 8 / 32 = 512 words

        // 4-bit / group 32 comes from the top-level default.
        let e = ws.quant_matrix(&ctx, "language_model.model.layers.0.mlp.switch_mlp.gate_proj");
        assert_eq!((e.group_size, e.bits), (32, 4));
        assert_eq!(e.num_experts(), 256);
        assert_eq!(e.w.shape, vec![256, 512, 256]); // 2048 * 4 / 32 = 256 words
    }
}
