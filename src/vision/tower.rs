//! `Qwen3VLVisionTower` from `vendor/MTPLX/mtplx/vision/qwen3_vl_tower.py`.
//!
//! ```text
//!   hidden = patch_embed(pixel_values)              3D conv, 2x16x16x3 -> 1152
//!   hidden = hidden + fast_pos_embed_interpolate()  bilinear over a 48x48 grid
//!   rotary = rot_pos_emb()                          2D rope over (row, col)
//!   for blk in blocks:                              27 of them
//!       hidden += attn(norm1(hidden), rotary)       16 heads of 72
//!       hidden += mlp(norm2(hidden))                1152 -> 4304 -> 1152
//!   return merger(hidden)                           2x2 patches -> one token
//! ```
//!
//! Every linear here carries a bias, which nothing else in this model does, so
//! all of them are `mx.addmm` rather than `mx.matmul`. The norms are
//! LayerNorm, not RMSNorm. The MLP's activation is `gelu_pytorch_tanh` and the
//! merger's is the exact erf gelu -- the reference really does use two
//! different ones.

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::conv::{pad_conv_weight_channels, patch_embed_conv3d};
use crate::ops::elementwise::{add, add_bias_last, copy_cast, gelu_exact, gelu_tanh, pad_last};
use crate::ops::embedding::gather_rows;
use crate::ops::layout::{concat_inner, slice_inner, transpose_hl, transpose_lh};
use crate::ops::matmul::addmm;
use crate::ops::norm::layer_norm;
use crate::ops::sdpa::steel_attention;
use crate::weights::WeightStore;

use super::VisionConfig;

/// `nn.LayerNorm`'s two parameters.
pub struct LayerNormWeights {
    pub weight: Array,
    pub bias: Array,
}

/// `nn.Linear` with a bias.
pub struct LinearWeights {
    pub weight: Array,
    pub bias: Array,
}

pub struct VisionBlock {
    pub norm1: LayerNormWeights,
    pub norm2: LayerNormWeights,
    pub qkv: LinearWeights,
    pub proj: LinearWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

/// `PatchMerger(use_postshuffle_norm=False)`: the norm runs over the
/// unmerged 1152 channels, and only then are four patches folded into one
/// 4608-wide row.
pub struct PatchMerger {
    pub norm: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

pub struct VisionTower {
    pub cfg: VisionConfig,
    /// `patch_embed.proj.weight`, zero-padded from 3 to 16 input channels --
    /// which is what `pad_and_slice_conv_3D_gpu` does on every call, hoisted
    /// here because the weight is constant.
    pub patch_proj: Array,
    pub patch_proj_bias: Array,
    pub pos_embed: Array,
    pub blocks: Vec<VisionBlock>,
    pub merger: PatchMerger,
    /// `int(num_position_embeddings ** 0.5)`.
    pub num_grid_per_side: usize,
}

/// `nn.LayerNorm(..., eps=1e-6)` -- every norm in the tower.
const LN_EPS: f32 = 1e-6;
/// `VisionRotaryEmbedding(head_dim // 2)`'s default theta.
const ROPE_THETA: f32 = 10000.0;
/// The padded head dim mlx's fused full-attention kernel accepts.
const SDPA_PAD_DIM: usize = 80;
/// `pad_and_slice_conv_3D_gpu` rounds the channel count up to a multiple of
/// 16; at `in_channels = 3` that is 16.
const CONV_PADDED_CHANNELS: usize = 16;

impl VisionTower {
    /// Load the 333 tensors of `model-vision-00001-of-00001.safetensors`.
    ///
    /// They live under `vision_tower.`, one of the two prefixes
    /// `resolve_vision_prefix` accepts. `_conv_weight_in_mlx_layout` finds
    /// this checkpoint's `patch_embed.proj.weight` already in mlx's
    /// `(out, kD, kH, kW, in)` order, so nothing is transposed on the way in.
    pub fn load(ctx: &Context, ws: &mut WeightStore, cfg: VisionConfig) -> VisionTower {
        let p = "vision_tower";
        let conv_shape = ws
            .info(&format!("{p}.patch_embed.proj.weight"))
            .expect("the vision shard is not loaded")
            .shape
            .clone();
        assert_eq!(
            conv_shape.last().copied(),
            Some(cfg.in_channels),
            "patch_embed.proj.weight is not in mlx's (out, kD, kH, kW, in) layout; \
             `_conv_weight_in_mlx_layout` would transpose it"
        );

        let ln = |ws: &mut WeightStore, path: &str| LayerNormWeights {
            weight: ws.tensor(ctx, &format!("{path}.weight")),
            bias: ws.tensor(ctx, &format!("{path}.bias")),
        };
        let lin = |ws: &mut WeightStore, path: &str| LinearWeights {
            weight: ws.tensor(ctx, &format!("{path}.weight")),
            bias: ws.tensor(ctx, &format!("{path}.bias")),
        };

        let raw_proj = ws.tensor(ctx, &format!("{p}.patch_embed.proj.weight"));
        let patch_proj = pad_conv_weight_channels(ctx, &raw_proj, CONV_PADDED_CHANNELS);
        let patch_proj_bias = ws.tensor(ctx, &format!("{p}.patch_embed.proj.bias"));
        let pos_embed = ws.tensor(ctx, &format!("{p}.pos_embed.weight"));

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let b = format!("{p}.blocks.{i}");
            blocks.push(VisionBlock {
                norm1: ln(ws, &format!("{b}.norm1")),
                norm2: ln(ws, &format!("{b}.norm2")),
                qkv: lin(ws, &format!("{b}.attn.qkv")),
                proj: lin(ws, &format!("{b}.attn.proj")),
                fc1: lin(ws, &format!("{b}.mlp.linear_fc1")),
                fc2: lin(ws, &format!("{b}.mlp.linear_fc2")),
            });
        }
        let merger = PatchMerger {
            norm: ln(ws, &format!("{p}.merger.norm")),
            fc1: lin(ws, &format!("{p}.merger.linear_fc1")),
            fc2: lin(ws, &format!("{p}.merger.linear_fc2")),
        };

        let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt() as usize;
        VisionTower {
            cfg,
            patch_proj,
            patch_proj_bias,
            pos_embed,
            blocks,
            merger,
            num_grid_per_side,
        }
    }

    /// `Qwen3VLVisionTower.__call__`.
    ///
    /// `pixel_values` is the preprocessor's `[patches, C * T * P * P]` float32
    /// block and `grid` its `(t, h, w)` patch grid. The result is
    /// `[patches / merge**2, out_hidden_size]` -- one row per `<|image_pad|>`
    /// token.
    ///
    /// MTPLX runs the tower once per image (`_vision_rows_for_image`), so
    /// `grid` always holds exactly one still frame and the per-frame attention
    /// split is always empty; both are asserted rather than half-implemented.
    pub fn forward(
        &self,
        ctx: &Context,
        pixel_values: &Array,
        grid: &[(usize, usize, usize)],
    ) -> Array {
        assert_eq!(grid.len(), 1, "MTPLX calls the tower once per image");
        assert_eq!(grid[0].0, 1, "video frames are not ported");
        let patches: usize = grid.iter().map(|(t, h, w)| t * h * w).sum();
        assert_eq!(pixel_values.shape[0], patches, "pixel rows do not match the grid");

        let hidden = self.patch_embed(ctx, pixel_values);
        let pos = self.fast_pos_embed_interpolate(ctx, grid);
        let mut hidden = add(ctx, &hidden, &pos);

        let (cos, sin) = self.rot_pos_emb(ctx, grid);

        for blk in &self.blocks {
            hidden = self.block_forward(ctx, blk, &hidden, &cos, &sin);
        }
        self.merge(ctx, &hidden)
    }

    /// `PatchEmbed.__call__`, plus `nn.Conv3d`'s separate bias add.
    pub fn patch_embed(&self, ctx: &Context, pixel_values: &Array) -> Array {
        assert_eq!(pixel_values.dtype, Dtype::F32);
        // `pixel_values.astype(self.patch_embed.proj.weight.dtype)`.
        let x = copy_cast(ctx, pixel_values, Dtype::BF16);
        let n = x.shape[0];
        let (c, t, p) = (self.cfg.in_channels, self.cfg.temporal_patch_size, self.cfg.patch_size);
        let x = x.reshape(&[n, c, t, p, p]);
        let y = patch_embed_conv3d(ctx, &x, &self.patch_proj, c);
        add_bias_last(ctx, &y, &self.patch_proj_bias)
    }

    /// One `VisionBlock`: pre-norm attention and MLP, both residual.
    pub fn block_forward(
        &self,
        ctx: &Context,
        blk: &VisionBlock,
        x: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Array {
        let h1 = layer_norm(ctx, x, &blk.norm1.weight, &blk.norm1.bias, LN_EPS);
        let a = self.attention(ctx, blk, &h1, cos, sin);
        let x = add(ctx, x, &a);

        let h2 = layer_norm(ctx, &x, &blk.norm2.weight, &blk.norm2.bias, LN_EPS);
        let m1 = addmm(ctx, &h2, &blk.fc1.weight, &blk.fc1.bias);
        let act = gelu_tanh(ctx, &m1);
        let m2 = addmm(ctx, &act, &blk.fc2.weight, &blk.fc2.bias);
        add(ctx, &x, &m2)
    }

    /// `Attention.__call__`.
    pub fn attention(
        &self,
        ctx: &Context,
        blk: &VisionBlock,
        x: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Array {
        let seq = x.shape[0];
        let heads = self.cfg.num_heads;
        let hd = self.cfg.head_dim();
        let dim = self.cfg.hidden_size;

        // `self.qkv(x).reshape(seq, 3, heads, -1).transpose(1, 0, 2, 3)`.
        let qkv = addmm(ctx, x, &blk.qkv.weight, &blk.qkv.bias);
        let qkv = transpose_lh(ctx, &qkv.reshape(&[1, seq, 3, dim])).reshape(&[3, seq, dim]);

        let rope = |i: usize| {
            let part = qkv.index(i).reshape(&[1, seq, heads, hd]);
            self.apply_rotary(ctx, &part, cos, sin)
        };
        // `_apply_rotary_pos_emb_vision` for q and k; v is left alone.
        let q = transpose_lh(ctx, &rope(0));
        let k = transpose_lh(ctx, &rope(1));
        let v = transpose_lh(ctx, &qkv.index(2).reshape(&[1, seq, heads, hd]));

        // `self.scale = self.head_dim ** -0.5`, which Python evaluates in
        // double and hands to the kernel as a float -- over the *unpadded* 72.
        let scale = (hd as f64).powf(-0.5) as f32;
        // `_fused_sdpa`: pad the head dim up to the nearest size mlx's fused
        // kernel supports, run it, slice back.
        let qp = pad_last(ctx, &q, SDPA_PAD_DIM);
        let kp = pad_last(ctx, &k, SDPA_PAD_DIM);
        let vp = pad_last(ctx, &v, SDPA_PAD_DIM);
        let o = steel_attention(ctx, &qp, &kp, &vp, scale);
        let o = slice_inner(ctx, &o, 0, hd);

        // `output.transpose(0, 2, 1, 3).reshape(seq, -1)`, then the projection.
        let o = transpose_hl(ctx, &o).reshape(&[seq, dim]);
        addmm(ctx, &o, &blk.proj.weight, &blk.proj.bias)
    }

    /// `_apply_rotary_pos_emb_vision(tensor, freqs)` for a `[1, seq, H, D]`
    /// tensor, with the cosine and sine tables precomputed.
    pub fn apply_rotary(&self, ctx: &Context, x: &Array, cos: &Array, sin: &Array) -> Array {
        let (seq, heads, hd) = (x.shape[1], x.shape[2], x.shape[3]);
        let out = Array::empty(ctx, &x.shape, x.dtype);
        let n = out.numel();
        let tg = 256.min(n.next_power_of_two()).max(1);
        assert_eq!(cos.shape, vec![seq, hd / 2]);
        ctx.kernel(&format!(
            "vision_rope_{}",
            match x.dtype {
                Dtype::BF16 => "bfloat16",
                Dtype::F32 => "float32",
                d => panic!("no vision rope for {}", d.name()),
            }
        ))
        .arr(x)
        .arr(cos)
        .arr(sin)
        .arr(&out)
        .bytes(&(heads as u32))
        .bytes(&(hd as u32))
        .threads((n, 1, 1), (tg, 1, 1));
        out
    }

    /// `PatchMerger.__call__` with `use_postshuffle_norm=False`.
    pub fn merge(&self, ctx: &Context, x: &Array) -> Array {
        let merged = self.cfg.hidden_size * self.cfg.spatial_merge_size.pow(2);
        let normed = layer_norm(ctx, x, &self.merger.norm.weight, &self.merger.norm.bias, LN_EPS);
        let rows = normed.numel() / merged;
        let folded = normed.reshape(&[rows, merged]);
        let y = addmm(ctx, &folded, &self.merger.fc1.weight, &self.merger.fc1.bias);
        let y = gelu_exact(ctx, &y);
        addmm(ctx, &y, &self.merger.fc2.weight, &self.merger.fc2.bias)
    }

    // --- position embeddings -------------------------------------------------

    /// `rot_pos_emb(grid)` followed by the `cos`/`sin` of
    /// `_apply_rotary_pos_emb_vision`.
    ///
    /// mlx recomputes the two tables inside every call, i.e. twice per block;
    /// they are a pure function of the grid, so computing them once here is
    /// the same arithmetic evaluated once instead of 54 times.
    pub fn rot_pos_emb(&self, ctx: &Context, grid: &[(usize, usize, usize)]) -> (Array, Array) {
        let merge = self.cfg.spatial_merge_size;
        // `VisionRotaryEmbedding(head_dim // 2)`: the table is half that wide
        // again, and the row and column halves concatenate back to head_dim/2.
        let rot_dim = self.cfg.head_dim() / 2;
        let max_hw = grid.iter().map(|(_, h, w)| (*h).max(*w)).max().unwrap();

        let freq_table = self.rotary_freq_table(ctx, max_hw, rot_dim);

        // `pos_ids`: for each patch, its (row, col) inside the frame, in the
        // spatial-merge order the patches themselves are in.
        let mut rows: Vec<u32> = Vec::new();
        let mut cols: Vec<u32> = Vec::new();
        for &(frames, h, w) in grid {
            let (mh, mw) = (h / merge, w / merge);
            let mut frame_rows = Vec::with_capacity(mh * mw * merge * merge);
            let mut frame_cols = Vec::with_capacity(mh * mw * merge * merge);
            for br in 0..mh {
                for bc in 0..mw {
                    for rin in 0..merge {
                        for cin in 0..merge {
                            frame_rows.push((br * merge + rin) as u32);
                            frame_cols.push((bc * merge + cin) as u32);
                        }
                    }
                }
            }
            for _ in 0..frames {
                rows.extend_from_slice(&frame_rows);
                cols.extend_from_slice(&frame_cols);
            }
        }
        let seq = rows.len();
        let row_ids = Array::from_slice(ctx, &rows, &[seq], Dtype::U32);
        let col_ids = Array::from_slice(ctx, &cols, &[seq], Dtype::U32);
        let h_emb = gather_rows(ctx, &freq_table, &row_ids, rot_dim / 2);
        let w_emb = gather_rows(ctx, &freq_table, &col_ids, rot_dim / 2);
        let freqs = concat_inner(ctx, &h_emb, &w_emb);

        // `mx.cos(freqs)` / `mx.sin(freqs)`, float32.
        let cos = Array::empty(ctx, &freqs.shape, Dtype::F32);
        let sin = Array::empty(ctx, &freqs.shape, Dtype::F32);
        let n = freqs.numel();
        let tg = 256.min(n.next_power_of_two()).max(1);
        ctx.kernel("vision_cos_sin")
            .arr(&freqs)
            .arr(&cos)
            .arr(&sin)
            .threads((n, 1, 1), (tg, 1, 1));
        (cos, sin)
    }

    /// `VisionRotaryEmbedding.__call__(seqlen)`: `outer(arange(seqlen),
    /// 1 / theta ** (arange(0, dim, 2) / dim))`, all float32.
    pub fn rotary_freq_table(&self, ctx: &Context, seqlen: usize, dim: usize) -> Array {
        let half = dim / 2;
        let inv_freq = Array::empty(ctx, &[half], Dtype::F32);
        ctx.kernel("vision_inv_freq")
            .arr(&inv_freq)
            .bytes(&ROPE_THETA)
            .bytes(&(dim as u32))
            .threads((half, 1, 1), (half.min(256).max(1), 1, 1));

        let table = Array::empty(ctx, &[seqlen, half], Dtype::F32);
        let n = table.numel();
        let tg = 256.min(n.next_power_of_two()).max(1);
        ctx.kernel("vision_freq_table")
            .arr(&inv_freq)
            .arr(&table)
            .bytes(&(half as u32))
            .threads((n, 1, 1), (tg, 1, 1));
        table
    }

    /// `fast_pos_embed_interpolate(grid)`.
    ///
    /// The learned table is a 48x48 grid of 1152-wide embeddings; a patch grid
    /// of any other size samples it bilinearly. The four corner indices and
    /// their weights are built on the host in float32, exactly as the
    /// reference builds them (`mx.linspace`, a truncating cast, and four
    /// products), then rounded to bfloat16 -- which is what
    /// `mx.array(weight_list, dtype=self.pos_embed.weight.dtype)` does.
    pub fn fast_pos_embed_interpolate(
        &self,
        ctx: &Context,
        grid: &[(usize, usize, usize)],
    ) -> Array {
        let (idx, weights, total) = self.interpolation_table(grid);
        let dims = self.cfg.hidden_size;

        let ids = Array::from_slice(ctx, &idx, &[idx.len()], Dtype::U32);
        let wts = Array::from_slice(ctx, &weights, &[weights.len()], Dtype::BF16);
        let gathered = gather_rows(ctx, &self.pos_embed, &ids, dims);

        // `pos_embeds * weight[:, :, None]` then the three adds.
        let blended = Array::empty(ctx, &[total, dims], Dtype::BF16);
        let n = blended.numel();
        let tg = 256.min(n.next_power_of_two()).max(1);
        ctx.kernel("pos_embed_blend_bfloat16")
            .arr(&gathered)
            .arr(&wts)
            .arr(&blended)
            .bytes(&(total as u32))
            .bytes(&(dims as u32))
            .threads((n, 1, 1), (tg, 1, 1));

        // Split per image, permute each into spatial-merge order, concatenate.
        // A multi-frame grid tiles its table over the frames first, so the
        // output is `sum(t * h * w)` rows even though the gather was `h * w`.
        let out_rows: usize = grid.iter().map(|(t, h, w)| t * h * w).sum();
        let out = Array::empty(ctx, &[out_rows, dims], Dtype::BF16);
        let (mut src_at, mut dst_at) = (0usize, 0usize);
        for &(t, h, w) in grid {
            let src = blended.slice_leading(src_at, h * w);
            let dst = out.slice_leading(dst_at, t * h * w);
            let n = dst.numel();
            let tg = 256.min(n.next_power_of_two()).max(1);
            ctx.kernel("pos_embed_permute_bfloat16")
                .arr(&src)
                .arr(&dst)
                .bytes(&(h as u32))
                .bytes(&(w as u32))
                .bytes(&(self.cfg.spatial_merge_size as u32))
                .bytes(&(dims as u32))
                .threads((n, 1, 1), (tg, 1, 1));
            src_at += h * w;
            dst_at += t * h * w;
        }
        out
    }

    /// The host half of `fast_pos_embed_interpolate`: the four gather indices
    /// and their bilinear weights, flattened as `[4 * total]`.
    pub fn interpolation_table(
        &self,
        grid: &[(usize, usize, usize)],
    ) -> (Vec<u32>, Vec<u16>, usize) {
        let side = self.num_grid_per_side;
        let total: usize = grid.iter().map(|(_, h, w)| h * w).sum();
        let mut idx = vec![0u32; 4 * total];
        let mut wts = vec![0u16; 4 * total];

        let mut base = 0usize;
        for &(_, h, w) in grid {
            let h_idxs = linspace(0.0, (side - 1) as f32, h);
            let w_idxs = linspace(0.0, (side - 1) as f32, w);
            let floor_ceil = |v: &[f32]| -> (Vec<i32>, Vec<i32>, Vec<f32>) {
                let fl: Vec<i32> = v.iter().map(|x| *x as i32).collect();
                let ce: Vec<i32> = fl.iter().map(|x| (*x + 1).min(side as i32 - 1)).collect();
                let d: Vec<f32> = v.iter().zip(&fl).map(|(x, f)| *x - *f as f32).collect();
                (fl, ce, d)
            };
            let (h_floor, h_ceil, dh) = floor_ceil(&h_idxs);
            let (w_floor, w_ceil, dw) = floor_ceil(&w_idxs);

            for y in 0..h {
                let bh = h_floor[y] * side as i32;
                let bh_c = h_ceil[y] * side as i32;
                let one_dh = 1.0f32 - dh[y];
                for x in 0..w {
                    let k = base + y * w + x;
                    let one_dw = 1.0f32 - dw[x];
                    let corners = [
                        (bh + w_floor[x], one_dh * one_dw),
                        (bh + w_ceil[x], one_dh * dw[x]),
                        (bh_c + w_floor[x], dh[y] * one_dw),
                        (bh_c + w_ceil[x], dh[y] * dw[x]),
                    ];
                    for (c, (i, wgt)) in corners.into_iter().enumerate() {
                        idx[c * total + k] = i as u32;
                        wts[c * total + k] = crate::array::f32_to_bf16(wgt);
                    }
                }
            }
            base += h * w;
        }
        (idx, wts, total)
    }
}

/// `mx.linspace(start, stop, num)` in float32: mlx evaluates
/// `(1 - t) * start + t * stop` with `t = arange(num) / (num - 1)`.
fn linspace(start: f32, stop: f32, num: usize) -> Vec<f32> {
    if num == 1 {
        return vec![start];
    }
    let denom = (num - 1) as f32;
    (0..num)
        .map(|i| {
            let t = i as f32 / denom;
            (1.0f32 - t) * start + t * stop
        })
        .collect()
}
