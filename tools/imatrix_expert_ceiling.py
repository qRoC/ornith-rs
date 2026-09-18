"""Is `down_proj`'s importance vector the wrong object too?

    ORNITH_GGML_BUILD=../.ggml-build python3 tools/imatrix_expert_ceiling.py [layers] [n]

`forge_q4k.py` converts the routed experts with one importance vector a layer
and says why in the file: "One vector serves all 256 experts of a layer,
because the input distribution is the layer's and not the expert's."

That is true of `gate_proj` and `up_proj` -- they all read one
`post_attention_layernorm` -- and **false of `down_proj`**, whose input is
`silu(gate) * up` and is expert-specific by construction. `REBUILD.md` Phase 22
measured how false: the pooled per-channel statistic has 1.1-1.7x spread where
the per-expert one has 2.0-4.7x, so pooling 256 experts cancels 256 patterns
rather than averaging one. Every k-quant artifact on disk was forged on the
pooled vector.

The right vector now exists -- `refs/awq-experts/`, 21 MB, from
`dump_awq_experts.py` -- and `quantize_q4_K` takes an importance vector
natively, so spending it costs a forge and no kernel. This prices it first,
the way Phase 22 priced AWQ, in the currency that is the imatrix's own
objective: relative output error under that expert's own diagonal.

Three quantizations a tensor, and the middle one is what is on disk:

    plain       no importance vector at all
    pooled      diag of the layer's shared Hessian -- the artifacts' own
    per-expert  that expert's E[h_c**2]
"""

import os
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import q4k_ggml  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"
HESS = ROOT / "refs/gptq-experts-q0/hessians"
SAL = ROOT / "refs/awq-experts"
IDX = __import__("json").load(open(PARENT / "model.safetensors.index.json"))["weight_map"]


def load(name):
    t = mx.load(str(PARENT / IDX[name]))[name]
    mx.eval(t)
    return np.array(t.astype(mx.float32), copy=False)


def kind_of(layer):
    """The k-quant artifacts convert layers 32-39's experts to Q6_K."""
    return "q6_K" if layer >= 32 else "q4_K"


def main():
    layers = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "0,5,20,35,39").split(",")]
    nexp = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    lib = q4k_ggml.load()

    # Three currencies, because a weighting scheme is *supposed* to lose on
    # the other two and reporting only the one it optimises would be measuring
    # the wrong object -- `Q4K.md`'s rule when the imatrix was first measured.
    print(f"down_proj of {nexp} experts a layer. Relative output error in three "
          f"currencies: unweighted, under the layer's pooled diagonal, and "
          f"under that expert's own.\n")
    print(f"{'layer':>5} {'fmt':>6} {'currency':>12} {'plain':>9} {'pooled':>9} "
          f"{'per-expert':>11} {'pooled/plain':>13} {'per-exp/pooled':>15}")
    for L in layers:
        kind = kind_of(L)
        pooled = np.ascontiguousarray(
            np.diag(np.asarray(np.load(HESS / f"layer{L}.inter.npy"), np.float32)))
        assert (pooled >= 0).all(), f"layer {L}: the pooled diagonal has negatives"
        z = np.load(SAL / f"layer{L}.npz")
        h2 = z["s2"] / np.maximum(z["n"], 1)[:, None]
        dn = load(f"model.language_model.layers.{L}.mlp.experts.down_proj")[:nexp]

        ones = np.ones(dn.shape[2], np.float32)
        rows = {c: [] for c in ("unweighted", "pooled", "per-expert")}
        for e in range(dn.shape[0]):
            w = dn[e]
            wt = np.ascontiguousarray(h2[e].astype(np.float32))
            deq = {}
            for tag, im in (("plain", None), ("pooled", pooled), ("per-expert", wt)):
                b = q4k_ggml.quantize(lib, w, imatrix=im, kind=kind)
                deq[tag] = q4k_ggml.dequantize(lib, b, w.size, kind=kind).reshape(w.shape)
            for cur, weight in (("unweighted", ones), ("pooled", pooled), ("per-expert", wt)):
                den = float(((w * w) * weight[None, :]).astype(np.float64).sum())
                rows[cur].append([
                    float(np.sqrt((((w - deq[t]) ** 2) * weight[None, :])
                                  .astype(np.float64).sum() / max(den, 1e-30)))
                    for t in ("plain", "pooled", "per-expert")
                ])
        for cur in ("unweighted", "pooled", "per-expert"):
            r = np.array(rows[cur])
            print(f"{L:>5} {kind:>6} {cur:>12} {r[:,0].mean():>9.4f} "
                  f"{r[:,1].mean():>9.4f} {r[:,2].mean():>11.4f} "
                  f"{100*(r[:,1]/r[:,0]-1).mean():>12.1f}% "
                  f"{100*(r[:,2]/r[:,1]-1).mean():>14.1f}%", flush=True)
        print()


if __name__ == "__main__":
    main()
