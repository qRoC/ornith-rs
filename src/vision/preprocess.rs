//! `Qwen2VLImageProcessorFast`, i.e. `preprocess_images` from
//! `vendor/MTPLX/mtplx/vision/processing.py`.
//!
//! What is *not* here is the image decoder: this port reads already-decoded
//! pixels (see [`Image::read`]), because decoding is open-ended -- formats,
//! ICC profiles, EXIF orientation -- and macOS already ships one. The
//! resample is a different matter and is ported, in [`super::resample`]:
//! `Image.resize(..., Image.BICUBIC)` is inside this pipeline, it decides
//! which pixels the model sees, and it is 150 lines of closed integer
//! arithmetic.
//!
//! Everything else is float32 throughout, exactly as numpy evaluates it:
//! rescale by 1/255, subtract the mean, divide by the standard deviation,
//! duplicate the frame along the temporal axis, and reorder into one row per
//! patch with the 2x2 spatial-merge blocks already contiguous.

use std::path::Path;

/// Decoded 8-bit RGB pixels.
pub struct Image {
    pub width: usize,
    pub height: usize,
    /// `height * width * 3` bytes, row major, R then G then B.
    pub rgb: Vec<u8>,
}

/// `preprocessor_config.json`, the fields `preprocess_images` reads.
#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub do_rescale: bool,
    pub rescale_factor: f32,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl PreprocessorConfig {
    pub fn load(model_dir: &Path) -> Result<PreprocessorConfig, String> {
        let raw = std::fs::read(model_dir.join("preprocessor_config.json"))
            .map_err(|e| format!("preprocessor_config.json: {e}"))?;
        let v = crate::json::parse(&raw).map_err(|e| format!("preprocessor_config.json: {e}"))?;
        let u = |k: &str, d: usize| v.get(k).and_then(|x| x.as_usize()).unwrap_or(d);
        // `_resolve_pixel_bounds`: `size.shortest_edge` / `size.longest_edge`,
        // overridden by top-level `min_pixels` / `max_pixels` when present.
        let size = v.get("size");
        let mut min_pixels =
            size.and_then(|s| s.get("shortest_edge")).and_then(|x| x.as_usize()).unwrap_or(56 * 56);
        let mut max_pixels = size
            .and_then(|s| s.get("longest_edge"))
            .and_then(|x| x.as_usize())
            .unwrap_or(14 * 14 * 4 * 1280);
        if let Some(x) = v.get("min_pixels").and_then(|x| x.as_usize()) {
            min_pixels = x;
        }
        if let Some(x) = v.get("max_pixels").and_then(|x| x.as_usize()) {
            max_pixels = x;
        }
        let triple = |k: &str, d: f32| -> [f32; 3] {
            match v.get(k).and_then(|x| x.as_arr()) {
                Some(a) if a.len() == 3 => {
                    [a[0].as_f32().unwrap_or(d), a[1].as_f32().unwrap_or(d), a[2].as_f32().unwrap_or(d)]
                }
                _ => [d, d, d],
            }
        };
        Ok(PreprocessorConfig {
            patch_size: u("patch_size", 16),
            merge_size: u("merge_size", 2),
            temporal_patch_size: u("temporal_patch_size", 2),
            min_pixels,
            max_pixels,
            do_rescale: v.get("do_rescale").and_then(|x| x.as_bool()).unwrap_or(true),
            rescale_factor: v
                .get("rescale_factor")
                .and_then(|x| x.as_f32())
                .unwrap_or(1.0 / 255.0),
            do_normalize: v.get("do_normalize").and_then(|x| x.as_bool()).unwrap_or(true),
            image_mean: triple("image_mean", 0.5),
            image_std: triple("image_std", 0.5),
        })
    }

    /// `patch_size * merge_size`, the multiple both sides must land on.
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }
}

/// Python's `round`, which is round-half-to-even -- not Rust's
/// `f64::round`, which rounds halves away from zero. `smart_resize` calls it
/// on `height / factor`, so a 1.5-block image (48 px at factor 32) rounds
/// down, not up.
fn py_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }
}

/// `smart_resize` from processing.py.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let f = factor as f64;
    let mut h_bar = py_round(h / f) * f;
    let mut w_bar = py_round(w / f) * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// One image's preprocessed patches plus its `(t, h, w)` grid.
pub struct Patches {
    /// `[grid_t * grid_h * grid_w, C * T * P * P]`, float32.
    pub values: Vec<f32>,
    pub rows: usize,
    pub row_len: usize,
    pub grid: (usize, usize, usize),
}

/// `preprocess_images([image], cfg)` for one image.
pub fn preprocess(image: &Image, cfg: &PreprocessorConfig) -> Result<Patches, String> {
    let (rh, rw) = smart_resize(
        image.height,
        image.width,
        cfg.factor(),
        cfg.min_pixels,
        cfg.max_pixels,
    )?;
    let resized;
    let image = if (rw, rh) != (image.width, image.height) {
        resized = super::resample::resize_bicubic(image, rw, rh);
        &resized
    } else {
        image
    };

    let ps = cfg.patch_size;
    let ms = cfg.merge_size;
    let tps = cfg.temporal_patch_size;
    let c = 3usize;
    let (h, w) = (image.height, image.width);
    let grid_t = 1usize;
    let grid_h = h / ps;
    let grid_w = w / ps;

    // `img = transpose(asarray(image, uint8), (2, 0, 1)).astype(float32)`,
    // then rescale and normalise -- three float32 array ops in numpy.
    let mut chw = vec![0f32; c * h * w];
    for ch in 0..c {
        let mean = cfg.image_mean[ch];
        let std = cfg.image_std[ch];
        for y in 0..h {
            for x in 0..w {
                let mut v = image.rgb[(y * w + x) * c + ch] as f32;
                if cfg.do_rescale {
                    v *= cfg.rescale_factor;
                }
                if cfg.do_normalize {
                    v = (v - mean) / std;
                }
                chw[ch * h * w + y * w + x] = v;
            }
        }
    }

    // `patches = np.repeat(img[None, None, ...], tps, axis=1)` then the
    // reshape/transpose/reshape into one row per patch. The frame axis is a
    // straight duplicate for a still image, so the two temporal halves of
    // every row are equal.
    let row_len = c * tps * ps * ps;
    let rows = grid_t * grid_h * grid_w;
    let mut values = vec![0f32; rows * row_len];
    // Output row order: (grid_t, grid_h/ms, grid_w/ms, ms, ms); column order:
    // (c, tps, ps, ps).
    let bh = grid_h / ms;
    let bw = grid_w / ms;
    let mut row = 0usize;
    for _t in 0..grid_t {
        for br in 0..bh {
            for bc in 0..bw {
                for rin in 0..ms {
                    for cin in 0..ms {
                        let py = (br * ms + rin) * ps;
                        let px = (bc * ms + cin) * ps;
                        let base = row * row_len;
                        for ch in 0..c {
                            for t in 0..tps {
                                let dst = base + ((ch * tps + t) * ps) * ps;
                                for yy in 0..ps {
                                    let src = ch * h * w + (py + yy) * w + px;
                                    values[dst + yy * ps..dst + yy * ps + ps]
                                        .copy_from_slice(&chw[src..src + ps]);
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
    }

    Ok(Patches { values, rows, row_len, grid: (grid_t, grid_h, grid_w) })
}

// --- reading pixels ---------------------------------------------------------

impl Image {
    /// Read an already-decoded image: binary PPM (`P6`), an uncompressed
    /// 24/32-bit BMP, or raw `width * height * 3` RGB bytes.
    ///
    /// There is no image decoder in this port on purpose. macOS ships `sips`,
    /// which writes BMP:
    ///
    /// ```text
    ///     sips -s format bmp photo.heic --out photo.bmp
    /// ```
    pub fn read(path: &Path, raw_size: Option<(usize, usize)>) -> Result<Image, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.starts_with(b"P6") {
            return Image::from_ppm(&bytes);
        }
        if bytes.starts_with(b"BM") {
            return Image::from_bmp(&bytes);
        }
        let (w, h) = raw_size.ok_or_else(|| {
            format!(
                "{}: not a P6 PPM or a BMP; for raw RGB pass --image-size WxH",
                path.display()
            )
        })?;
        if bytes.len() != w * h * 3 {
            return Err(format!(
                "{}: {} bytes is not {w}x{h}x3",
                path.display(),
                bytes.len()
            ));
        }
        Ok(Image { width: w, height: h, rgb: bytes })
    }

    /// Binary PPM: `P6`, then width, height and maxval as ASCII with `#`
    /// comments allowed, then one whitespace byte, then the pixels.
    pub fn from_ppm(bytes: &[u8]) -> Result<Image, String> {
        let mut pos = 2usize;
        let mut fields = [0usize; 3];
        for field in fields.iter_mut() {
            // Skip whitespace and comments.
            loop {
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos < bytes.len() && bytes[pos] == b'#' {
                    while pos < bytes.len() && bytes[pos] != b'\n' {
                        pos += 1;
                    }
                } else {
                    break;
                }
            }
            let start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            if start == pos {
                return Err("malformed PPM header".into());
            }
            *field = std::str::from_utf8(&bytes[start..pos]).unwrap().parse().unwrap();
        }
        let (w, h, maxval) = (fields[0], fields[1], fields[2]);
        if maxval != 255 {
            return Err(format!("PPM maxval {maxval} is not 255"));
        }
        pos += 1; // exactly one whitespace byte separates header and data
        let need = w * h * 3;
        if bytes.len() < pos + need {
            return Err(format!("PPM is short: {} bytes, want {}", bytes.len() - pos, need));
        }
        Ok(Image { width: w, height: h, rgb: bytes[pos..pos + need].to_vec() })
    }

    /// Uncompressed BMP, which is what `sips -s format bmp` writes: a 14-byte
    /// file header, an info header, then bottom-up (or top-down, for a
    /// negative height) rows of BGR or BGRA padded to 4 bytes.
    pub fn from_bmp(bytes: &[u8]) -> Result<Image, String> {
        let u32le = |o: usize| -> u32 {
            u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        };
        if bytes.len() < 54 {
            return Err("BMP is too short".into());
        }
        let data_off = u32le(10) as usize;
        let header_size = u32le(14) as usize;
        if header_size < 40 {
            return Err(format!("BMP header size {header_size} is not supported"));
        }
        let w = u32le(18) as i32 as usize;
        let raw_h = u32le(22) as i32;
        let h = raw_h.unsigned_abs() as usize;
        let top_down = raw_h < 0;
        let bpp = u16::from_le_bytes([bytes[28], bytes[29]]) as usize;
        let compression = u32le(30);
        if compression != 0 && !(compression == 3 && bpp == 32) {
            return Err(format!("BMP compression {compression} is not supported"));
        }
        if bpp != 24 && bpp != 32 {
            return Err(format!("BMP with {bpp} bits per pixel is not supported"));
        }
        let stride = (w * bpp / 8).div_ceil(4) * 4;
        if bytes.len() < data_off + stride * h {
            return Err("BMP pixel data is short".into());
        }
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            let src_row = if top_down { y } else { h - 1 - y };
            let src = data_off + src_row * stride;
            for x in 0..w {
                let p = src + x * (bpp / 8);
                let d = (y * w + x) * 3;
                rgb[d] = bytes[p + 2];
                rgb[d + 1] = bytes[p + 1];
                rgb[d + 2] = bytes[p];
            }
        }
        Ok(Image { width: w, height: h, rgb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_round_is_half_to_even() {
        assert_eq!(py_round(1.5), 2.0);
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(-1.5), -2.0);
        assert_eq!(py_round(-2.5), -2.0);
        assert_eq!(py_round(0.5), 0.0);
        assert_eq!(py_round(1.4999), 1.0);
    }

    #[test]
    fn smart_resize_matches_the_python() {
        let (min_p, max_p) = (65536usize, 16777216usize);
        // Already a multiple of 32 and inside the bounds: a no-op.
        assert_eq!(smart_resize(256, 256, 32, min_p, max_p).unwrap(), (256, 256));
        assert_eq!(smart_resize(224, 320, 32, min_p, max_p).unwrap(), (224, 320));
        // Rounded to the nearest multiple of 32, halves to even.
        assert_eq!(smart_resize(600, 800, 32, min_p, max_p).unwrap(), (608, 800));
        // Under min_pixels: scaled up.
        let (h, w) = smart_resize(64, 64, 32, min_p, max_p).unwrap();
        assert!(h * w >= min_p, "{h}x{w}");
        assert_eq!((h % 32, w % 32), (0, 0));
    }

    #[test]
    fn ppm_roundtrip() {
        let mut p = b"P6\n# a comment\n2 2\n255\n".to_vec();
        p.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let img = Image::from_ppm(&p).unwrap();
        assert_eq!((img.width, img.height), (2, 2));
        assert_eq!(img.rgb, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }
}
