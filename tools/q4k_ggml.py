"""ggml's own Q4_K quantizer, called through ctypes. `Q4K.md` step 1.

    python3 tools/q4k_ggml.py [<ggml build dir>]

`q4k.quantize` is a faithful port of `quantize_row_q4_K_ref`'s *shape* and is
0.77 points worse than ggml, because ggml runs a search. Nothing that ships
should use it. This wraps the real one, which is exported from `libggml-base`
and -- the part that matters for the next phase -- **takes an importance
vector**:

    size_t quantize_q4_K(const float *src, void *dst,
                         int64_t nrows, int64_t n_per_row,
                         const float *imatrix);

`diag(H)` from `refs/gptq-parent/hessians/` is exactly that input, which makes
importance weighting a one-argument change rather than the super-block
coordination GPTQ x Q4_K would need.

Build ggml first; see the header of `tools/q4k_oracle.c` for the cmake line.
"""

import ctypes
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aq       # noqa: E402
import q4k      # noqa: E402

BUILD = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                     os.environ.get("ORNITH_GGML_BUILD", "../.ggml-build"))
PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"


def load():
    lib = ctypes.CDLL(str(BUILD / "src/libggml-base.dylib"))
    lib.quantize_q4_K.restype = ctypes.c_size_t
    lib.quantize_q4_K.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                  ctypes.c_int64, ctypes.c_int64,
                                  ctypes.POINTER(ctypes.c_float)]
    return lib


def quantize(lib, w, imatrix=None):
    """[rows, cols] float32 -> [rows * cols / 256, 144] uint8, ggml's own bytes."""
    w = np.ascontiguousarray(np.asarray(w, np.float32))
    rows, cols = w.shape
    assert cols % q4k.QK_K == 0, cols
    dst = np.zeros(rows * cols // q4k.QK_K * q4k.BLOCK_BYTES, np.uint8)
    im = None
    if imatrix is not None:
        im = np.ascontiguousarray(np.asarray(imatrix, np.float32))
        im = im.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.quantize_q4_K(w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      dst.ctypes.data_as(ctypes.c_void_p), rows, cols, im)
    return dst.reshape(-1, q4k.BLOCK_BYTES)


def main():
    import mlx.core as mx
    lib = load()
    idx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    cases = [("in_proj_qkv", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
             ("out_proj", "model.language_model.layers.0.linear_attn.out_proj.weight"),
             ("lm_head", "lm_head.weight")]
    print(f"{'module':<14}{'ggml Q4_K 4.50b':>17}{'affine g32 5.00b':>19}")
    for lbl, k in cases:
        t = mx.load(str(PARENT / idx[k]))[k]
        mx.eval(t)
        w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False))
        del t
        if lbl == "lm_head":
            w = w[:8192]                     # a slice; the head is 0.5 GB
        rows, cols = w.shape
        r = q4k.dequantize(quantize(lib, w)).reshape(rows, cols)
        q, s, b = aq.quantize(w, 32, 4)
        aff = (q.reshape(rows, cols).astype(np.float32)
               * np.repeat(aq.bf16(s).reshape(rows, -1), 32, 1)
               + np.repeat(aq.bf16(b).reshape(rows, -1), 32, 1))
        print(f"{lbl:<14}{100 * np.linalg.norm(w - r) / np.linalg.norm(w):>16.3f}%"
              f"{100 * np.linalg.norm(w - aff) / np.linalg.norm(w):>18.3f}%")


if __name__ == "__main__":
    main()
