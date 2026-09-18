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
use crate::ops::q4k::Q4kMatrix;
use crate::ops::quantized::{QuantMatrix, QuantMode};
use crate::safetensors::{SafeTensors, TensorInfo, shards_from_index};

pub struct WeightStore {
    st: SafeTensors,
    /// Per-module `{bits, group_size, mode}` from config.json's
    /// `quantization`.
    quant: HashMap<String, QuantSpec>,
    default_quant: QuantSpec,
    cache: HashMap<String, Array>,
}

/// One module's entry in config.json's `quantization` block, as mlx-lm's
/// `quantize_model` writes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuantSpec {
    pub group_size: usize,
    pub bits: usize,
    pub mode: QuantMode,
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
        // `mode` is absent from artifacts forged before mlx grew the fp
        // modes; mlx-lm's own loader defaults it to affine, so this does too.
        let default_quant = QuantSpec {
            group_size: q
                .get("group_size")
                .and_then(|v| v.as_usize())
                .ok_or("quantization.group_size")?,
            bits: q.get("bits").and_then(|v| v.as_usize()).ok_or("quantization.bits")?,
            mode: QuantMode::from_config(
                q.get("mode").and_then(|v| v.as_str()).unwrap_or("affine"),
            ),
        };
        let mut quant = HashMap::new();
        for (k, v) in q.as_obj().ok_or("quantization is not an object")? {
            if let Some(o) = v.as_obj() {
                let spec = QuantSpec {
                    group_size: o
                        .get("group_size")
                        .and_then(|v| v.as_usize())
                        .unwrap_or(default_quant.group_size),
                    bits: o.get("bits").and_then(|v| v.as_usize()).unwrap_or(default_quant.bits),
                    mode: o
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .map_or(default_quant.mode, QuantMode::from_config),
                };
                quant.insert(k.clone(), spec);
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

    /// The quantization spec for a module path, with the same per-module
    /// override lookup mlx-lm's loader does.
    pub fn quant_config(&self, path: &str) -> QuantSpec {
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
    ///
    /// The zero point is affine's alone: `mx.quantize` returns three arrays in
    /// affine mode and two in nvfp4, and `nn.QuantizedLinear` stores
    /// `self.biases = None` for the latter, so the artifact holds no
    /// `.biases` tensor at all. Both are checked here rather than inferred, so
    /// a module whose tensors disagree with the mode config.json claims fails
    /// at load instead of producing numbers.
    pub fn quant_matrix(&mut self, ctx: &Context, path: &str) -> QuantMatrix {
        let spec = self.quant_config(path);
        let biases_name = format!("{path}.biases");
        let has_biases = self.has(&biases_name);
        match spec.mode {
            QuantMode::Affine => assert!(
                has_biases,
                "{path} is affine but the artifact has no {biases_name}"
            ),
            QuantMode::Nvfp4 => assert!(
                !has_biases,
                "{path} is nvfp4 but the artifact has a {biases_name}; nvfp4 \
                 has no zero point, so this is not the layout mlx produces"
            ),
            QuantMode::Q4k => panic!(
                "{path} is q4k, which has no mlx kernel -- load it with \
                 `q4k_matrix` instead of `quant_matrix`"
            ),
        }
        // The optional global scale is the other way a quantized module can
        // carry a third tensor, and the fp kernels take it in the slot the
        // zero point would use. mlx_lm.convert never emits one -- it neither
        // passes `global_scale` to `mx.quantize` nor sets `quantize_input` --
        // so finding one means this loader is reading an artifact it was not
        // written for.
        let gs_name = format!("{path}.global_scale");
        assert!(
            !self.has(&gs_name),
            "{path} carries a {gs_name}; the fp kernels' has_global_scale \
             specialisations are not instantiated (see fp_quantized_inst.metal)"
        );
        QuantMatrix {
            w: self.tensor(ctx, &format!("{path}.weight")),
            scales: self.tensor(ctx, &format!("{path}.scales")),
            biases: has_biases.then(|| self.tensor(ctx, &biases_name)),
            mode: spec.mode,
            group_size: spec.group_size,
            bits: spec.bits,
        }
    }

    /// Load `<path>.weight` / `.scales` / `.dm` as one Q4_K matrix.
    ///
    /// The three-tensor split is `Q4K.md`'s storage decision: ggml keeps one
    /// packed 144-byte `block_q4_K`, and an artifact that did the same would
    /// carry a module whose only tensor is an opaque blob -- `check_rebuild.py`
    /// counts a module as quantized when it has more than one tensor, and the
    /// loader would need a shape it has never seen. Splitting keeps both, at
    /// no cost to the kernel: the bytes inside each tensor are ggml's own.
    ///
    /// The names deliberately reuse `.weight` and `.scales`, which every other
    /// mode here also has; `.dm` is the one new name, and `.biases` must be
    /// absent because Q4_K's minimum lives in the packed 6-bit fields rather
    /// than in a zero-point tensor.
    pub fn q4k_matrix(&mut self, ctx: &Context, path: &str) -> Q4kMatrix {
        let spec = self.quant_config(path);
        assert_eq!(
            spec.mode,
            QuantMode::Q4k,
            "{path} is not q4k in config.json's quantization block"
        );
        assert_eq!(
            spec.group_size,
            crate::ops::q4k::QK_K,
            "{path}: q4k's super-block is {} weights, not {}",
            crate::ops::q4k::QK_K,
            spec.group_size
        );
        for absent in [format!("{path}.biases"), format!("{path}.global_scale")] {
            assert!(
                !self.has(&absent),
                "{path} is q4k but the artifact has a {absent}; the minimum is \
                 one of the packed 6-bit fields, not a separate tensor"
            );
        }
        // `Q4kMatrix::new` checks the three against each other -- dtypes,
        // block counts, row widths -- so a mismatched set fails at load rather
        // than being read as garbage once per dispatch.
        Q4kMatrix::new(
            self.tensor(ctx, &format!("{path}.weight")),
            self.tensor(ctx, &format!("{path}.scales")),
            self.tensor(ctx, &format!("{path}.dm")),
        )
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
            "/Users/qroc/.mtplx/models/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx".into()
        }))
    }

    /// The loaded tensors agree with what `config.json` says about them.
    ///
    /// Deliberately derived rather than hardcoded: the three Phase 4 variants
    /// disagree about which modules are 4-bit and the nvfp4 one disagrees
    /// about the format, so an assertion of the control's own numbers here
    /// fails on a correct artifact -- which is exactly the shape of mistake
    /// `check_rebuild.py` was rewritten to avoid. What is invariant is the
    /// relation between the config and the shapes.
    #[test]
    fn loads_a_quantized_matrix_with_the_right_config() {
        let dir = model_dir();
        if !dir.exists() {
            return;
        }
        let ctx = Context::new().unwrap();
        let mut ws = WeightStore::open(&dir, &[]).unwrap();

        // A 2-D linear and a 3-D stacked expert weight, which between them
        // cover both the per-module override and the top-level default.
        for (path, experts) in [
            ("language_model.model.layers.0.linear_attn.in_proj_qkv", 1),
            ("language_model.model.layers.0.mlp.switch_mlp.gate_proj", 256),
        ] {
            let spec = ws.quant_config(path);
            let q = ws.quant_matrix(&ctx, path);
            assert_eq!((q.group_size, q.bits, q.mode), (spec.group_size, spec.bits, spec.mode));
            assert_eq!(q.num_experts(), experts, "{path}");
            assert_eq!(q.in_dims(), 2048, "{path}");

            // `32 / bits` codes per uint32 word, in both modes: nvfp4 packs 8
            // E2M1 nibbles where affine int4 packs 8 codes.
            let words = q.in_dims() * q.bits / 32;
            assert_eq!(*q.w.shape.last().unwrap(), words, "{path} packed width");
            assert_eq!(
                *q.scales.shape.last().unwrap(),
                q.in_dims() / q.group_size,
                "{path} scale groups"
            );
            // Two tensors or three, decided by the mode and nothing else.
            match q.mode {
                QuantMode::Affine => {
                    let b = q.biases.as_ref().expect("affine carries a zero point");
                    assert_eq!(b.shape, q.scales.shape, "{path}");
                    assert_eq!(q.scales.dtype, b.dtype, "{path}");
                }
                QuantMode::Nvfp4 => {
                    assert!(q.biases.is_none(), "{path} is nvfp4 and must be two tensors");
                    assert_eq!(q.scales.dtype, Dtype::U8, "{path} nvfp4 scales are E4M3 bytes");
                    assert_eq!((q.group_size, q.bits), (16, 4), "{path}");
                }
                // `quant_matrix` refuses a q4k module, so this walk cannot
                // produce one; `tests/q4k.rs` checks that layout instead.
                QuantMode::Q4k => unreachable!("quant_matrix rejects q4k"),
            }
        }
    }
}
