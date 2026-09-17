//! `mx.conv3d`, narrowed to the vision tower's patch embedding.
//!
//! Port of `conv_3D_gpu` -> `dispatch_conv_3D_gpu` ->
//! `pad_and_slice_conv_3D_gpu` -> `implicit_gemm_conv_3D_gpu`
//! (vendor/mlx/mlx/backend/metal/conv.cpp) over mlx's own
//! `steel/conv/kernels/steel_conv_3d.h`.
//!
//! `PatchEmbed` is the only convolution in the vision tower and the only 3D
//! convolution anywhere in this model:
//!
//! ```text
//!   nn.Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)
//! ```
//!
//! over an `[N, 2, 16, 16, 3]` input, so each patch produces one `1x1x1`
//! output pixel of 1152 channels -- an implicit GEMM whose M is the patch
//! count, N is 1152 and K is the whole 2x16x16x16 padded filter volume.
//!
//! The channel padding is not an optimisation, it is the branch mlx takes:
//! `mod16_channels` is false at 3 input channels, so `dispatch_conv_3D_gpu`
//! pads both operands to 16 channels with zeros and runs the *padded* problem
//! through the implicit gemm. Doing the unpadded thing instead would be a
//! different kernel with a different reduction order.

use crate::array::{Array, Dtype};
use crate::metal::Context;

/// `MLXConvParams<3>` (kernels/steel/conv/params.h), field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams3 {
    n: i32,
    c: i32,
    o: i32,
    i_s: [i32; 3],
    w_s: [i32; 3],
    o_s: [i32; 3],
    str_: [i32; 3],
    pad: [i32; 3],
    kdil: [i32; 3],
    idil: [i32; 3],
    in_strides: [i64; 5],
    wt_strides: [i64; 5],
    out_strides: [i64; 5],
    groups: i32,
    flip: bool,
}

/// `mlx::steel::ImplicitGemmConv3DParams`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ImplicitGemmConv3DParams {
    m: i32,
    n: i32,
    k: i32,
    gemm_k_iterations: i32,
    inp_jump_w: i32,
    inp_jump_h: i32,
    inp_jump_d: i32,
    inp_jump_c: i32,
    tiles_n: i32,
    tiles_m: i32,
    swizzle_log: i32,
}

/// Zero-pad a convolution weight's channel axis, `[O, kD, kH, kW, C]` ->
/// `[O, kD, kH, kW, Cp]`.
///
/// mlx redoes this on every `conv3d` call (`pad_and_slice_conv_3D_gpu`'s
/// `pad_array`); the tower hoists it to load time because the weight is a
/// constant and the padding writes exact zeros either way.
pub fn pad_conv_weight_channels(ctx: &Context, w: &Array, padded_channels: usize) -> Array {
    super::elementwise::pad_last(ctx, w, padded_channels)
}

/// The patch-embedding convolution.
///
/// `x` is `[N, C, kD, kH, kW]` -- the layout `PatchEmbed.__call__` reshapes
/// the preprocessed pixels into, *before* its `moveaxis(1, 4)`. `w` is the
/// already channel-padded weight, `[O, kD, kH, kW, Cp]`. The result is
/// `[N, O]`, which is the `reshape(-1, hidden_size)` the caller wants; mlx
/// gets there through a `[N, 1, 1, 1, O]` output whose spatial extents are all
/// 1 because stride equals kernel size.
pub fn patch_embed_conv3d(ctx: &Context, x: &Array, w: &Array, channels: usize) -> Array {
    assert_eq!(x.ndim(), 5);
    assert_eq!(w.ndim(), 5);
    let n = x.shape[0];
    assert_eq!(x.shape[1], channels);
    let (kd, kh, kw) = (x.shape[2], x.shape[3], x.shape[4]);
    let o = w.shape[0];
    assert_eq!([w.shape[1], w.shape[2], w.shape[3]], [kd, kh, kw]);
    let padded_channels = w.shape[4];
    assert_eq!(padded_channels % 16, 0, "the implicit gemm needs mod-16 channels");
    assert_eq!(o % 16, 0, "extra_o is 0 for this model; a pad-and-slice output is not ported");
    assert_eq!(x.dtype, w.dtype);

    // `moveaxis(1, 4)` and `pad_gpu`'s channel pad in one pass: mlx runs them
    // as a strided view followed by a copy, and both are exact.
    let spatial = kd * kh * kw;
    let padded_in = Array::empty(ctx, &[n, kd, kh, kw, padded_channels], x.dtype);
    {
        let total = padded_in.numel();
        let tg = 256.min(total.next_power_of_two()).max(1);
        let tname = match x.dtype {
            Dtype::BF16 => "bfloat16",
            Dtype::F32 => "float32",
            d => panic!("no patch embed for {}", d.name()),
        };
        ctx.kernel(&format!("patch_moveaxis_pad_{tname}"))
            .arr(x)
            .out(&padded_in)
            .bytes(&(channels as u32))
            .bytes(&(padded_channels as u32))
            .bytes(&(spatial as u32))
            .threads((total, 1, 1), (tg, 1, 1));
    }

    let out = Array::empty(ctx, &[n, 1, 1, 1, o], x.dtype);

    // `conv_3D_gpu` builds the params from *contiguous* strides of the
    // unpadded shapes, then `with_padded_channels` rewrites them.
    let c = channels;
    let mut in_strides = [0i64; 5];
    let mut wt_strides = [0i64; 5];
    in_strides[4] = 1;
    wt_strides[4] = 1;
    let in_shape = [n, kd, kh, kw, c];
    let wt_shape = [o, kd, kh, kw, c];
    for i in (0..4).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1] as i64;
        wt_strides[i] = wt_strides[i + 1] * wt_shape[i + 1] as i64;
    }
    let out_strides_full: Vec<i64> = out.strides();

    // `MLXConvParams<3>::with_padded_channels(params, extra_o = 0, extra_c)`.
    let pad_in = (padded_channels - c) as i64;
    let mut params = ConvParams3 {
        n: n as i32,
        c: c as i32,
        o: o as i32,
        i_s: [kd as i32, kh as i32, kw as i32],
        w_s: [kd as i32, kh as i32, kw as i32],
        o_s: [1, 1, 1],
        str_: [kd as i32, kh as i32, kw as i32],
        pad: [0, 0, 0],
        kdil: [1, 1, 1],
        idil: [1, 1, 1],
        in_strides,
        wt_strides,
        out_strides: [
            out_strides_full[0],
            out_strides_full[1],
            out_strides_full[2],
            out_strides_full[3],
            out_strides_full[4],
        ],
        groups: 1,
        flip: false,
    };
    for i in 0..4 {
        params.in_strides[i] = (params.in_strides[i] / c as i64) * (c as i64 + pad_in);
        params.wt_strides[i] = (params.wt_strides[i] / c as i64) * (c as i64 + pad_in);
        params.out_strides[i] = (params.out_strides[i] / o as i64) * o as i64;
    }
    params.in_strides[4] = 1;
    params.wt_strides[4] = 1;
    params.out_strides[4] = 1;
    params.c = padded_channels as i32;

    // `implicit_gemm_conv_3D_gpu`, on the padded problem.
    let c_per_group = padded_channels;
    let implicit_m = n;
    let implicit_n = o;
    let implicit_k = kd * kh * kw * c_per_group;

    let (mut wm, mut wn) = (2usize, 2usize);
    let bm = if implicit_m >= 8192 && c_per_group >= 64 { 64usize } else { 32 };
    let mut bn = if bm == 64 || implicit_n >= 64 { 64usize } else { 32 };
    let bk = 16usize;
    if implicit_n <= 16 {
        bn = 8;
        wm = 4;
        wn = 1;
    }
    assert_eq!((bm, bn, bk, wm, wn), (32, 64, 16, 2, 2), "only this block shape is instantiated");

    let tn = implicit_n.div_ceil(bn);
    let tm = implicit_m.div_ceil(bm);
    let swizzle_log = 0i32;

    let small_filter = kd <= 16 && kh <= 16 && kw <= 16;
    assert!(small_filter, "only the small-filter loader is instantiated");

    let channel_k_iters = c_per_group.div_ceil(bk);
    let gemm_k_iters = kd * kh * kw * channel_k_iters;

    // The input jumps, with `sign = 1` because the filter is not flipped.
    let ijw = params.in_strides[3] * params.kdil[2] as i64;
    let ijh = params.in_strides[2] * params.kdil[1] as i64;
    let ijd = params.in_strides[1] * params.kdil[0] as i64;
    let inp_jump_w = ijw;
    let inp_jump_h = ijh - (kw as i64 - 1) * ijw;
    let inp_jump_d = ijd - (kh as i64 - 1) * ijh - (kw as i64 - 1) * ijw;
    let inp_jump_c = bk as i64 - (kd as i64 - 1) * ijd - (kh as i64 - 1) * ijh - (kw as i64 - 1) * ijw;

    let gemm_params = ImplicitGemmConv3DParams {
        m: implicit_m as i32,
        n: implicit_n as i32,
        k: implicit_k as i32,
        gemm_k_iterations: gemm_k_iters as i32,
        inp_jump_w: inp_jump_w as i32,
        inp_jump_h: inp_jump_h as i32,
        inp_jump_d: inp_jump_d as i32,
        inp_jump_c: inp_jump_c as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
    };

    let tname = match out.dtype {
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        d => panic!("no conv3d for {}", d.name()),
    };
    let name = format!(
        "implicit_gemm_conv_3d_{tname}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}_filter_{}",
        if small_filter { 's' } else { 'l' }
    );

    let tile = 1usize << swizzle_log;
    let grid_y = tm.div_ceil(tile);
    let grid_x = tn * tile;
    ctx.kernel(&name)
        .arr(&padded_in)
        .arr(w)
        .out(&out)
        .bytes(&params)
        .bytes(&gemm_params)
        .groups((grid_x, grid_y, 1), (32, wn, wm));

    out.reshape(&[n, o])
}
