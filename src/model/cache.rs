//! Caches, from `vendor/mlx-lm/mlx_lm/models/cache.py`.
//!
//! Two kinds, one per layer kind:
//!
//!   `KVCache`     for the 10 full-attention layers: keys and values grown in
//!                 256-step blocks and written in place.
//!   `ArraysCache` for the 30 linear-attention layers: two slots holding the
//!                 causal conv window and the recurrent state, both of fixed
//!                 size -- a GatedDeltaNet layer's memory does not grow with
//!                 the sequence, which is the whole point of the architecture.

use crate::array::{Array, Dtype};
use crate::metal::Context;

use super::config::TextConfig;

/// `KVCache`, including its step-256 growth.
pub struct KvCache {
    pub keys: Option<Array>,
    pub values: Option<Array>,
    pub offset: usize,
}

const KV_STEP: usize = 256;

impl KvCache {
    pub fn new() -> KvCache {
        KvCache { keys: None, values: None, offset: 0 }
    }

    /// Capacity along the sequence axis.
    pub fn capacity(&self) -> usize {
        self.keys.as_ref().map(|k| k.shape[2]).unwrap_or(0)
    }

    /// `update_and_fetch`: append `keys`/`values` and return views covering
    /// everything cached so far.
    pub fn update_and_fetch(
        &mut self,
        ctx: &Context,
        keys: &Array,
        values: &Array,
    ) -> (Array, Array) {
        let prev = self.offset;
        let (b, h, l, dk) = (keys.shape[0], keys.shape[1], keys.shape[2], keys.shape[3]);
        let dv = values.shape[3];

        if self.keys.is_none() || prev + l > self.capacity() {
            let n_steps = (KV_STEP + l - 1) / KV_STEP;
            let grow = n_steps * KV_STEP;
            let new_cap = self.capacity() + grow;
            let nk = Array::empty(ctx, &[b, h, new_cap, dk], keys.dtype);
            let nv = Array::empty(ctx, &[b, h, new_cap, dv], values.dtype);
            zero(ctx, &nk);
            zero(ctx, &nv);
            if let (Some(ok), Some(ov)) = (self.keys.as_ref(), self.values.as_ref()) {
                // mlx concatenates the old block with a fresh one; copying the
                // live prefix into the new buffer is the same thing.
                copy_seq(ctx, ok, &nk, self.capacity(), new_cap, prev);
                copy_seq(ctx, ov, &nv, self.capacity(), new_cap, prev);
            }
            self.keys = Some(nk);
            self.values = Some(nv);
        }

        self.offset += l;
        let cap = self.capacity();
        write_seq(ctx, keys, self.keys.as_ref().unwrap(), cap, prev);
        write_seq(ctx, values, self.values.as_ref().unwrap(), cap, prev);

        // `keys_and_values()` returns the [..., :offset, :] view; the caller
        // gets the capacity separately so kernels can be given real strides.
        let k = self.keys.as_ref().unwrap().clone();
        let v = self.values.as_ref().unwrap().clone();
        (
            Array { shape: vec![b, h, self.offset, dk], ..k },
            Array { shape: vec![b, h, self.offset, dv], ..v },
        )
    }
}

impl Default for KvCache {
    fn default() -> Self {
        KvCache::new()
    }
}

/// `ArraysCache(size=2)` as `qwen3_5.TextModel.make_cache` builds it: slot 0 is
/// the conv window, slot 1 the recurrent state.
pub struct GdnCache {
    pub conv: Option<Array>,
    pub state: Option<Array>,
    pub offset: usize,
}

impl GdnCache {
    pub fn new() -> GdnCache {
        GdnCache { conv: None, state: None, offset: 0 }
    }

    /// The zero-filled conv window `GatedDeltaNet` starts from.
    pub fn conv_or_zeros(&self, ctx: &Context, cfg: &TextConfig, b: usize, dtype: Dtype) -> Array {
        match &self.conv {
            Some(c) => c.clone(),
            None => {
                let a = Array::empty(ctx, &[b, cfg.linear_conv_kernel_dim - 1, cfg.conv_dim()], dtype);
                zero(ctx, &a);
                a
            }
        }
    }

    /// The zero recurrent state; float32, per `mamba_ssm_dtype`.
    pub fn state_or_zeros(&self, ctx: &Context, cfg: &TextConfig, b: usize) -> Array {
        match &self.state {
            Some(s) => s.clone(),
            None => {
                let a = Array::empty(
                    ctx,
                    &[
                        b,
                        cfg.linear_num_value_heads,
                        cfg.linear_value_head_dim,
                        cfg.linear_key_head_dim,
                    ],
                    Dtype::F32,
                );
                zero(ctx, &a);
                a
            }
        }
    }
}

impl Default for GdnCache {
    fn default() -> Self {
        GdnCache::new()
    }
}

/// One cache per layer, of the kind that layer needs.
pub enum LayerCache {
    Kv(KvCache),
    Gdn(GdnCache),
}

impl LayerCache {
    pub fn offset(&self) -> usize {
        match self {
            LayerCache::Kv(c) => c.offset,
            LayerCache::Gdn(c) => c.offset,
        }
    }

    pub fn as_kv(&mut self) -> &mut KvCache {
        match self {
            LayerCache::Kv(c) => c,
            _ => panic!("expected a KV cache"),
        }
    }

    pub fn as_gdn(&mut self) -> &mut GdnCache {
        match self {
            LayerCache::Gdn(c) => c,
            _ => panic!("expected a gated-delta cache"),
        }
    }
}

/// `TextModel.make_cache`.
pub fn make_cache(cfg: &TextConfig) -> Vec<LayerCache> {
    (0..cfg.num_hidden_layers)
        .map(|i| {
            if cfg.is_linear_layer(i) {
                LayerCache::Gdn(GdnCache::new())
            } else {
                LayerCache::Kv(KvCache::new())
            }
        })
        .collect()
}

// --- helpers ----------------------------------------------------------------

pub fn zero(ctx: &Context, a: &Array) {
    let n = a.numel();
    let tg = 256.min(n.next_power_of_two()).max(1);
    let name = match a.dtype {
        Dtype::BF16 => "fill_zero_bfloat16",
        Dtype::F32 => "fill_zero_float32",
        Dtype::U32 => "fill_zero_uint32",
        d => panic!("no fill kernel for {}", d.name()),
    };
    ctx.kernel(name).out(a).threads((n, 1, 1), (tg, 1, 1));
}

/// Copy `[B, H, used, D]` out of a `[B, H, src_cap, D]` buffer into the front
/// of a `[B, H, dst_cap, D]` one.
fn copy_seq(ctx: &Context, src: &Array, dst: &Array, src_cap: usize, dst_cap: usize, used: usize) {
    if used == 0 {
        return;
    }
    let (b, h, d) = (src.shape[0], src.shape[1], src.shape[3]);
    let n = b * h * used * d;
    let tg = 256.min(n.next_power_of_two()).max(1);
    ctx.kernel(&format!("cache_copy_{}", tname(src.dtype)))
        .arr(src)
        .out(dst)
        .bytes(&(src_cap as u32))
        .bytes(&(dst_cap as u32))
        .bytes(&(used as u32))
        .bytes(&(d as u32))
        .bytes(&0u32)
        .threads((n, 1, 1), (tg, 1, 1));
}

/// `cache[..., offset:offset + L, :] = src`.
fn write_seq(ctx: &Context, src: &Array, dst: &Array, dst_cap: usize, offset: usize) {
    let (b, h, l, d) = (src.shape[0], src.shape[1], src.shape[2], src.shape[3]);
    let n = b * h * l * d;
    let tg = 256.min(n.next_power_of_two()).max(1);
    ctx.kernel(&format!("cache_copy_{}", tname(src.dtype)))
        .arr(src)
        .out(dst)
        .bytes(&(l as u32))
        .bytes(&(dst_cap as u32))
        .bytes(&(l as u32))
        .bytes(&(d as u32))
        .bytes(&(offset as u32))
        .threads((n, 1, 1), (tg, 1, 1));
}

fn tname(d: Dtype) -> &'static str {
    match d {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no cache kernel for {}", d.name()),
    }
}
