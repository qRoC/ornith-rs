"""Reference for Pillow's bicubic resampler.

`preprocess_images` resizes with `Image.resize(..., Image.BICUBIC)` whenever
`smart_resize` asks for different dimensions, so `src/vision/resample.rs` has
to reproduce `src/libImaging/Resample.c` exactly: the coefficients quantized to
22-bit fixed point, two passes with a uint8 intermediate, and a pass skipped
when that axis keeps its size.

The pairs below cover every branch of that: both axes down, both up, one axis
only, a huge downscale (where `filterscale` grows the kernel), a tiny upscale
(where the support clamps at the edges), and the real (in -> out) pairs
`smart_resize` produces for ordinary photographs.

Unlike every other reference here, this one pins the internals of a *library
version* rather than of the model, so the version is recorded and checked.

    cd tools && python3 dump_resample.py
"""

import numpy as np
import PIL
from PIL import Image

import ref

# (in_w, in_h, out_w, out_h)
CASES = [
    (64, 48, 32, 32),      # down in both, different ratios per axis
    (32, 32, 64, 48),      # up in both
    (40, 30, 40, 64),      # height only: the horizontal pass is skipped
    (40, 30, 96, 30),      # width only: the vertical pass is skipped
    (100, 70, 96, 64),     # a small downscale, filterscale just above 1
    (333, 211, 320, 192),  # odd sizes, non-integer ratios
    (5, 7, 32, 32),        # tiny source: xmin/xmax clamp at both edges
    (300, 300, 7, 5),      # huge downscale: ksize grows with filterscale
    (700, 500, 704, 512),  # what smart_resize asks of a 700x500 photo
    (1280, 853, 1280, 864),  # ... and of a 1280x853 one: height only
]


def make_image(width, height, seed):
    """Structured pixels: a filter bug that only smears edges must show."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    r = (xx * 255 // max(width - 1, 1)).astype(np.uint8)
    g = (yy * 255 // max(height - 1, 1)).astype(np.uint8)
    # Hard edges every few pixels, so ringing lands on exact pixel values.
    b = (((xx // 3) + (yy // 5)) % 2 * 255).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    noise = rng.integers(0, 64, size=img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int16) // 2 + noise, 0, 255).astype(np.uint8)


def main():
    case = ref.Case("resample")
    case.note(pillow_version=PIL.__version__, cases=[list(c) for c in CASES])

    import mlx.core as mx

    for seed, (iw, ih, ow, oh) in enumerate(CASES):
        tag = f"{iw}x{ih}_to_{ow}x{oh}"
        arr = make_image(iw, ih, seed)
        src = Image.fromarray(arr, mode="RGB")
        out = src.resize((ow, oh), resample=Image.BICUBIC)
        assert out.size == (ow, oh)
        case.put(f"{tag}_in", mx.array(arr.reshape(-1).astype(np.uint8)))
        case.put(
            f"{tag}_out",
            mx.array(np.asarray(out, dtype=np.uint8).reshape(-1)),
        )
        print(f"{tag}: ok", flush=True)

    case.write()


if __name__ == "__main__":
    main()
