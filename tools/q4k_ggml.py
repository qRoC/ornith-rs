"""ggml's own k-quant quantizers, called through ctypes. `Q4K.md` step 1.

    python3 tools/q4k_ggml.py [<ggml build dir>]          # Q4_K vs affine 5.00b
    python3 tools/q4k_ggml.py --q6k                       # Q6_K vs affine 8.50b
    python3 tools/q4k_ggml.py --imatrix                   # Q4_K with diag(H)
    python3 tools/q4k_ggml.py --groups                    # where each format spends
    python3 tools/q4k_ggml.py --matvec                    # error in the output, not the weights


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

# The env var wins: other tools import this one and their own argv[1] is a
# model directory, not a ggml build.
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
BUILD = pathlib.Path(
    os.environ.get("ORNITH_GGML_BUILD")
    or (_args[0] if _args and __name__ == "__main__" else "../.ggml-build")
)
PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"


# name -> (bytes per 256-weight block, bits per weight)
KINDS = {"q4_K": (144, 4.50), "q6_K": (210, 6.5625)}


def load():
    lib = ctypes.CDLL(str(BUILD / "src/libggml-base.dylib"))
    for kind in KINDS:
        q = getattr(lib, f"quantize_{kind}")
        q.restype = ctypes.c_size_t
        q.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                      ctypes.c_int64, ctypes.c_int64,
                      ctypes.POINTER(ctypes.c_float)]
        d = getattr(lib, f"dequantize_row_{kind}")
        d.restype = None
        d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    return lib


def quantize(lib, w, imatrix=None, kind="q4_K"):
    """[rows, cols] float32 -> [rows * cols / 256, block bytes] uint8, ggml's own."""
    w = np.ascontiguousarray(np.asarray(w, np.float32))
    rows, cols = w.shape
    assert cols % q4k.QK_K == 0, cols
    nbytes = KINDS[kind][0]
    dst = np.zeros(rows * cols // q4k.QK_K * nbytes, np.uint8)
    im = None
    if imatrix is not None:
        im = np.ascontiguousarray(np.asarray(imatrix, np.float32))
        im = im.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    getattr(lib, f"quantize_{kind}")(
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.c_void_p), rows, cols, im)
    return dst.reshape(-1, nbytes)


def dequantize(lib, blocks, n, kind="q4_K"):
    """ggml's own dequantizer -- the oracle, rather than a second reader.

    `q4k.dequantize` is the numpy port and is exact against this (`Q4K.md`
    step 1); for a pricing number there is no reason to go through it, and
    Q6_K has no numpy port at all because nothing has to read it yet."""
    blocks = np.ascontiguousarray(blocks)
    out = np.zeros(n, np.float32)
    getattr(lib, f"dequantize_row_{kind}")(
        blocks.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
    return out


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


# The 169 modules the shipping artifact leaves at affine int8 g64 = 8.50 bits:
# the ten full-attention layers' projections, the last eight layers' experts
# and shared experts, every router, and the embedding. `Q4K.md` asks what Q6_K
# (6.5625 bits) costs on them, which is 22.8% of those bytes and 6.1% of a
# decode step.
EIGHT_BIT = [
    ("self_attn.q_proj", "model.language_model.layers.3.self_attn.q_proj.weight", "plain"),
    ("self_attn.k_proj", "model.language_model.layers.3.self_attn.k_proj.weight", "plain"),
    ("self_attn.o_proj", "model.language_model.layers.3.self_attn.o_proj.weight", "plain"),
    ("mlp.gate (router)", "model.language_model.layers.3.mlp.gate.weight", "plain"),
    ("shared_expert.gate", "model.language_model.layers.39.mlp.shared_expert.gate_proj.weight", "plain"),
    ("shared_expert.down", "model.language_model.layers.39.mlp.shared_expert.down_proj.weight", "plain"),
    ("experts.gate_up", "model.language_model.layers.39.mlp.experts.gate_up_proj", "experts"),
    ("experts.down", "model.language_model.layers.39.mlp.experts.down_proj", "experts"),
    ("embed_tokens", "model.language_model.embed_tokens.weight", "head"),
]
EXPERTS = 8       # weight_error.py's slice: enough rows, not the whole 1.0 GiB
HEAD_ROWS = 8192


def _load(mx, idx, name, kind):
    t = mx.load(str(PARENT / idx[name]))[name]
    mx.eval(t)
    w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False))
    del t
    if kind == "experts":
        w = w[:EXPERTS]
        w = w.reshape(-1, w.shape[-1])
    elif kind == "head":
        w = w[:HEAD_ROWS]
    return w


def _affine(w, group, bits):
    rows, cols = w.shape
    q, s, b = aq.quantize(w, group, bits)
    return (q.reshape(rows, cols).astype(np.float32)
            * np.repeat(aq.bf16(s).reshape(rows, -1), group, 1)
            + np.repeat(aq.bf16(b).reshape(rows, -1), group, 1))


def rel(w, r):
    return 100 * np.linalg.norm(w - r) / np.linalg.norm(w)


def hrel(w, r, h):
    """The same error under `diag(H)`: the objective the imatrix optimizes."""
    d = (w - r).astype(np.float64) ** 2 @ h.astype(np.float64)
    n = w.astype(np.float64) ** 2 @ h.astype(np.float64)
    return 100 * np.sqrt(d.sum() / n.sum())


def q6k():
    """Q6_K against the affine int8 g64 those 169 modules carry today."""
    import mlx.core as mx
    lib = load()
    idx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    print("the 8-bit side of the artifact, against the bf16 parent\n")
    print(f"{'module':<20}{'ggml Q6_K':>12}{'affine g64':>12}{'ggml Q4_K':>12}   shape")
    print(f"{'':<20}{'6.5625b':>12}{'8.50b':>12}{'4.50b':>12}")
    for lbl, key, kind in EIGHT_BIT:
        w = _load(mx, idx, key, kind)
        if w.shape[1] % q4k.QK_K:
            print(f"{lbl:<20}{'--':>12}{'--':>12}{'--':>12}   {list(w.shape)} not a multiple of 256")
            continue
        q6 = dequantize(lib, quantize(lib, w, kind="q6_K"), w.size, kind="q6_K").reshape(w.shape)
        q4 = dequantize(lib, quantize(lib, w, kind="q4_K"), w.size, kind="q4_K").reshape(w.shape)
        a8 = _affine(w, 64, 8)
        print(f"{lbl:<20}{rel(w, q6):>11.3f}%{rel(w, a8):>11.3f}%{rel(w, q4):>11.3f}%   {list(w.shape)}")
        del w


def imatrix():
    """`quantize_q4_K`'s importance vector, fed the GPTQ Hessian's diagonal.

    `refs/gptq-parent/hessians/<module>.npy` is `diag(H)` over the calibration
    set -- one number per input column, which is exactly the shape
    `quantize_q4_K` wants. Weighting is not error compensation, so this is not
    expected to reach GPTQ's 27%; the question is whether it is free.
    """
    import mlx.core as mx
    lib = load()
    idx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    hdir = pathlib.Path(__file__).resolve().parent.parent / "refs/gptq-parent/hessians"
    print("Q4_K with and without diag(H) as the importance vector\n")
    print(f"{'':<38}{'  unweighted rel RMS':>28}{'  H-weighted rel RMS':>32}")
    print(f"{'module':<38}{'plain':>9}{'+imatrix':>9}{'change':>9}   "
          f"{'plain':>9}{'+imatrix':>9}{'change':>9}")
    for f in sorted(hdir.glob("*.npy"))[:6] + sorted(hdir.glob("*lm_head*.npy")):
        mod = f.stem
        key = ("lm_head.weight" if mod.endswith("lm_head")
               else "model." + mod.replace("language_model.model.", "language_model.") + ".weight")
        if key not in idx:
            print(f"{mod:<38}  (no parent tensor {key})")
            continue
        # The file is the whole `H`, [in, in]. `quant_weights` wants one
        # number per input column, and the one that means "importance" is the
        # diagonal -- H's first *row* has negative entries, which the
        # quantizer turns into 38589% error rather than a finding.
        h = np.ascontiguousarray(np.diag(np.asarray(np.load(f), np.float32)))
        w = _load(mx, idx, key, "head" if "lm_head" in mod else "plain")
        assert h.shape == (w.shape[1],), (h.shape, w.shape)
        assert (h >= 0).all(), f"{mod}: diag(H) has negative entries"
        a = dequantize(lib, quantize(lib, w), w.size).reshape(w.shape)
        b = dequantize(lib, quantize(lib, w, imatrix=h), w.size).reshape(w.shape)
        ra, rb = rel(w, a), rel(w, b)
        wa, wb = hrel(w, a, h), hrel(w, b, h)
        print(f"{mod.split('language_model.')[-1]:<38}"
              f"{ra:>9.3f}%{rb:>9.3f}%{100*(rb-ra)/ra:>+8.2f}%   "
              f"{wa:>9.3f}%{wb:>9.3f}%{100*(wb-wa)/wa:>+8.2f}%")
        del w


def matvec():
    """The error in the *product*, which is the object a KL is about.

    `measure-the-same-object`: every table in this rebuild ranks formats by
    `||W - W_hat|| / ||W||`, and the thing a distribution sees is
    `||(W - W_hat) x|| / ||W x||`. The two agree only when the weight error is
    uncorrelated across a row. A quantizer that leaves a systematic offset
    inside each group has that offset add *coherently* over the 2048 terms of
    a dot product, where an unbiased error adds as its square root -- so a
    format can win the norm and lose the product by a wide margin.

    Reported per module, over random gaussian `x` and over the calibration
    activations when they are on disk:

        rel out   ||(W - W_hat) x|| / ||W x||
        bias      the mean error inside a group of 32, as a share of the
                  group's RMS error. An unbiased quantizer sits near 1/sqrt(32)
                  = 0.177 by chance alone; well above that is a real offset.
    """
    import mlx.core as mx
    lib = load()
    idx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    cases = [("lm_head", "lm_head.weight", 4096),
             ("in_proj_qkv", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 8192),
             ("in_proj_z", "model.language_model.layers.0.linear_attn.in_proj_z.weight", 4096),
             ("out_proj", "model.language_model.layers.0.linear_attn.out_proj.weight", 2048)]
    rng = np.random.default_rng(3)
    print(f"{'module':<14}{'weight err':>12}{'  ':2}{'rel out err':>12}{'':4}"
          f"{'group bias':>12}")
    print(f"{'':<14}{'Q4_K':>5}{'affine':>7}{'  ':2}{'Q4_K':>5}{'affine':>7}{'':4}"
          f"{'Q4_K':>5}{'affine':>7}")
    for lbl, key, rows in cases:
        t = mx.load(str(PARENT / idx[key]))[key]
        mx.eval(t)
        w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False)[:rows])
        del t
        r, c = w.shape
        q4 = dequantize(lib, quantize(lib, w), w.size).reshape(r, c)
        qa, sa, ba = aq.quantize(w, 32, 4)
        aff = (qa.reshape(r, c).astype(np.float32)
               * np.repeat(aq.bf16(sa).reshape(r, -1), 32, 1)
               + np.repeat(aq.bf16(ba).reshape(r, -1), 32, 1))

        x = rng.standard_normal((c, 64)).astype(np.float32)
        ref = w.astype(np.float64) @ x
        def out_err(h):
            return 100 * np.linalg.norm((w - h).astype(np.float64) @ x) / np.linalg.norm(ref)
        def wt_err(h):
            return 100 * np.linalg.norm(w - h) / np.linalg.norm(w)
        def bias(h):
            e = (w - h).reshape(-1, 32)
            return float(np.abs(e.mean(1)).mean() / np.sqrt((e ** 2).mean(1)).mean())
        print(f"{lbl:<14}{wt_err(q4):>5.2f}{wt_err(aff):>7.2f}{'  ':2}"
              f"{out_err(q4):>5.2f}{out_err(aff):>7.2f}{'':4}"
              f"{bias(q4):>5.3f}{bias(aff):>7.3f}")
        del w, q4, aff
    print("\n(an unbiased quantizer's group bias sits near 1/sqrt(32) = 0.177)")


def groups():
    """Where each format spends its resolution, by sub-block magnitude.

    Q4_K measures *better* total weight error than affine g32 and *worse* KL,
    which is the third time in this rebuild that the norm has ranked backwards.
    The norm is an L2 over the whole tensor, so it is dominated by the groups
    with the largest weights; this splits the same error by how big a group is
    relative to its own 256-wide super-block.

    The mechanism to look for: affine gives every group of 32 its own bfloat16
    scale, so its *relative* resolution is the same everywhere. Q4_K gives a
    group a 6-bit integer against one fp16 super-scale shared by eight, so a
    group far below its super-block's maximum is coarse in relative terms and
    a group at the maximum is finer than affine can be. If that is the story,
    the error split by magnitude decile crosses over.
    """
    import mlx.core as mx
    lib = load()
    idx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    cases = [("lm_head", "lm_head.weight", 4096),
             ("in_proj_qkv", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 8192),
             ("out_proj", "model.language_model.layers.0.linear_attn.out_proj.weight", 2048)]
    print("relative error by sub-block magnitude, against the super-block's own maximum")
    print("(each row is a decile of `group max / super-block max`; 10 = the loudest)\n")
    for lbl, key, rows in cases:
        t = mx.load(str(PARENT / idx[key]))[key]
        mx.eval(t)
        w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False)[:rows])
        del t
        r, c = w.shape
        q4 = dequantize(lib, quantize(lib, w), w.size).reshape(r, c)
        qa, sa, ba = aq.quantize(w, 32, 4)
        aff = (qa.reshape(r, c).astype(np.float32)
               * np.repeat(aq.bf16(sa).reshape(r, -1), 32, 1)
               + np.repeat(aq.bf16(ba).reshape(r, -1), 32, 1))

        g = w.reshape(-1, 32)
        gmax = np.abs(g).max(1)
        smax = np.repeat(np.abs(w.reshape(-1, 256)).max(1), 8)
        share = np.where(smax > 0, gmax / np.maximum(smax, 1e-30), 0.0)
        e4 = np.linalg.norm((w - q4).reshape(-1, 32), axis=1)
        ea = np.linalg.norm((w - aff).reshape(-1, 32), axis=1)
        nrm = np.linalg.norm(g, axis=1)

        edges = np.quantile(share, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, share, "right") - 1, 0, 9)
        print(f"{lbl}  [{r}, {c}]")
        print(f"  {'decile':>6} {'share':>12}  {'Q4_K':>8} {'affine g32':>11}  {'ratio':>7}"
              f"  {'% of total L2':>14}")
        tot = (e4 ** 2).sum()
        for k in range(10):
            m = b == k
            if not m.any():
                continue
            r4 = 100 * np.linalg.norm(e4[m]) / np.linalg.norm(nrm[m])
            ra = 100 * np.linalg.norm(ea[m]) / np.linalg.norm(nrm[m])
            print(f"  {k + 1:>6} {share[m].mean():>12.3f}  {r4:>7.3f}% {ra:>10.3f}%"
                  f"  {r4 / ra:>7.2f}  {100 * (e4[m] ** 2).sum() / tot:>13.1f}%")
        print()
        del w, q4, aff


if __name__ == "__main__":
    if "--matvec" in sys.argv:
        matvec()
    elif "--groups" in sys.argv:
        groups()
    elif "--q6k" in sys.argv:
        q6k()
    elif "--imatrix" in sys.argv:
        imatrix()
    else:
        main()
