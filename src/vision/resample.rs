//! Pillow's bicubic resampler, `src/libImaging/Resample.c`.
//!
//! `preprocess_images` calls `Image.resize(..., Image.BICUBIC)` whenever
//! `smart_resize` asks for different dimensions, so an image that is not
//! already a multiple of `patch_size * merge_size` goes through this before
//! the model ever sees it. It is part of the vendored
//! `mtplx/vision/processing.py` pipeline, so it is ported like the rest --
//! the image *decoder* is the thing that stays outside (see
//! [`super::preprocess::Image::read`]).
//!
//! The arithmetic is not "bicubic interpolation" in the abstract; it is
//! Pillow's, and three details decide the last bit of every pixel:
//!
//!   * the filter coefficients are computed in `double` and then **quantized**
//!     to 22-bit fixed point (`normalize_coeffs_8bpc`), so the convolution
//!     itself is integer;
//!   * it is **two passes**, horizontal then vertical, and the intermediate is
//!     rounded back to `uint8` in between -- one fused pass would not agree;
//!   * a pass is **skipped** when that axis does not change size, which is not
//!     the same as running an identity pass through the quantizer.
//!
//! `tests/resample.rs` pins all of it against Pillow itself.

use super::preprocess::Image;

/// `#define PRECISION_BITS (32 - 8 - 2)`.
const PRECISION_BITS: i32 = 32 - 8 - 2;

/// `static struct filter BICUBIC = {bicubic_filter, 2.0}`.
const BICUBIC_SUPPORT: f64 = 2.0;

/// `bicubic_filter`, with Pillow's `a = -0.5`. Written in its own order of
/// operations, because in double precision the order is the value.
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = if x < 0.0 { -x } else { x };
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
    }
    0.0
}

/// The per-output-pixel filter support, as `precompute_coeffs` returns it:
/// `ksize` taps each, the first source index and how many taps are live.
struct Coeffs {
    ksize: usize,
    /// `(xmin, xmax)` per output pixel, where `xmax` is a *count*.
    bounds: Vec<(i32, i32)>,
    /// `out_size * ksize` coefficients, already in 22-bit fixed point.
    kk: Vec<i32>,
}

/// `precompute_coeffs` followed by `normalize_coeffs_8bpc`.
///
/// Pillow keeps them apart because the same buffer is reused in place; here
/// they are one step, because nothing else reads the floating-point form.
fn precompute_coeffs(in_size: usize, in0: f64, in1: f64, out_size: usize) -> Coeffs {
    let scale = (in1 - in0) / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = BICUBIC_SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;

    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];

    for xx in 0..out_size {
        let center = in0 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // `(int)` truncates toward zero; a negative xmin is clamped away, and
        // Pillow rounds by adding 0.5 before the cast rather than after.
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size as i32 {
            xmax = in_size as i32;
        }
        xmax -= xmin;

        let mut ww = 0.0f64;
        for (x, kx) in k.iter_mut().enumerate().take(xmax as usize) {
            let w = bicubic_filter((x as f64 + xmin as f64 - center + 0.5) * ss);
            *kx = w;
            ww += w;
        }
        for kx in k.iter_mut().take(xmax as usize) {
            if ww != 0.0 {
                *kx /= ww;
            }
        }
        for kx in k.iter_mut().skip(xmax as usize) {
            *kx = 0.0;
        }

        // `normalize_coeffs_8bpc`: round half away from zero into 22-bit fixed
        // point, by adding +-0.5 and truncating.
        for (x, kx) in k.iter().enumerate() {
            let scaled = kx * (1i32 << PRECISION_BITS) as f64;
            kk[xx * ksize + x] =
                if *kx < 0.0 { (scaled - 0.5) as i32 } else { (scaled + 0.5) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    Coeffs { ksize, bounds, kk }
}

/// `clip8`: an arithmetic shift down out of fixed point, then Pillow's
/// saturating lookup table.
fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `ImagingResampleHorizontal_8bpc` for a 3-band image.
///
/// `offset` is the first source row the vertical pass will want, so the
/// intermediate holds only the rows that pass actually reads.
fn resample_horizontal(
    src: &Image,
    offset: usize,
    out_w: usize,
    out_h: usize,
    c: &Coeffs,
) -> Image {
    let mut rgb = vec![0u8; out_w * out_h * 3];
    for yy in 0..out_h {
        let row = (yy + offset) * src.width * 3;
        for xx in 0..out_w {
            let (xmin, xmax) = c.bounds[xx];
            let k = &c.kk[xx * c.ksize..];
            let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
            for x in 0..xmax as usize {
                let p = row + (x + xmin as usize) * 3;
                for (b, s) in ss.iter_mut().enumerate() {
                    *s = s.wrapping_add((src.rgb[p + b] as i32).wrapping_mul(k[x]));
                }
            }
            let d = (yy * out_w + xx) * 3;
            for b in 0..3 {
                rgb[d + b] = clip8(ss[b]);
            }
        }
    }
    Image { width: out_w, height: out_h, rgb }
}

/// `ImagingResampleVertical_8bpc` for a 3-band image.
fn resample_vertical(src: &Image, out_h: usize, c: &Coeffs) -> Image {
    let w = src.width;
    let mut rgb = vec![0u8; w * out_h * 3];
    for yy in 0..out_h {
        let (ymin, ymax) = c.bounds[yy];
        let k = &c.kk[yy * c.ksize..];
        for xx in 0..w {
            let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
            for y in 0..ymax as usize {
                let p = ((y + ymin as usize) * w + xx) * 3;
                for (b, s) in ss.iter_mut().enumerate() {
                    *s = s.wrapping_add((src.rgb[p + b] as i32).wrapping_mul(k[y]));
                }
            }
            let d = (yy * w + xx) * 3;
            for b in 0..3 {
                rgb[d + b] = clip8(ss[b]);
            }
        }
    }
    Image { width: w, height: out_h, rgb }
}

/// `Image.resize((out_w, out_h), Image.BICUBIC)` over the whole image.
///
/// This is `ImagingResampleInner` with `box = (0, 0, width, height)`, which is
/// what `Image.resize` passes when `box` is `None`. With that box the two
/// `need_*` tests collapse to "does this axis change size".
pub fn resize_bicubic(src: &Image, out_w: usize, out_h: usize) -> Image {
    assert!(out_w > 0 && out_h > 0, "resize to {out_w}x{out_h}");
    if (out_w, out_h) == (src.width, src.height) {
        // `Image.resize` returns a copy before it reaches the resampler.
        return Image { width: src.width, height: src.height, rgb: src.rgb.clone() };
    }

    let need_horizontal = out_w != src.width;
    let need_vertical = out_h != src.height;

    let horiz = precompute_coeffs(src.width, 0.0, src.width as f64, out_w);
    let mut vert = precompute_coeffs(src.height, 0.0, src.height as f64, out_h);

    // The first and last source rows the vertical pass reads.
    let ybox_first = vert.bounds[0].0;
    let last = vert.bounds[out_h - 1];
    let ybox_last = last.0 + last.1;

    let temp;
    let mid: &Image = if need_horizontal {
        for b in vert.bounds.iter_mut() {
            b.0 -= ybox_first;
        }
        temp = resample_horizontal(
            src,
            ybox_first as usize,
            out_w,
            (ybox_last - ybox_first) as usize,
            &horiz,
        );
        &temp
    } else {
        src
    };

    if need_vertical {
        resample_vertical(mid, out_h, &vert)
    } else {
        Image { width: mid.width, height: mid.height, rgb: mid.rgb.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_filter_is_pillows() {
        // A cubic through (0, 1) with zeros at every other integer, which is
        // what makes an identity resize exact.
        assert_eq!(bicubic_filter(0.0), 1.0);
        assert_eq!(bicubic_filter(1.0), 0.0);
        assert_eq!(bicubic_filter(2.0), 0.0);
        assert_eq!(bicubic_filter(2.5), 0.0);
        assert_eq!(bicubic_filter(-1.0), 0.0);
        // Negative lobe between 1 and 2.
        assert!(bicubic_filter(1.5) < 0.0);
    }

    #[test]
    fn identity_is_a_copy() {
        let src = Image { width: 2, height: 2, rgb: (0..12).collect() };
        let out = resize_bicubic(&src, 2, 2);
        assert_eq!(out.rgb, src.rgb);
    }
}
