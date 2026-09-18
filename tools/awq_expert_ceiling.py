"""What an activation-aware scale could buy on the routed experts, priced.

    python3 tools/awq_expert_ceiling.py [spread|down|oracle|gateup|all] [layers] [n experts]

`REBUILD.md` Phase 22, and `rotation_ceiling.py`'s sibling: the point is to
price a lever in the currency Phase 12 reported -- `||(W - Wq) X^T||_F /
||W X^T||_F` -- *before* spending a forge and a probe capture on it.

Phase 12 named AWQ on `down_proj` as GPTQ's complement: the fold is exact
because `h_c = silu(gate)_c * up_c` puts the nonlinearity on the gate path
alone, and AWQ needs no anisotropic Hessian. What it does need is spread in
the diagonal, and the object that decides it is per-expert.

Four measurements, and they are meant to be read in this order:

  spread   the diagonal of the Hessians already on disk, both input spaces,
           against the per-expert saliency `dump_awq_experts.py` records. The
           pooled and per-expert numbers differ by an order of magnitude, and
           that is the phase's transferable finding.
  down     the AWQ family on `down_proj`, continuous and power-of-two, best
           alpha per expert, scored on the per-expert diagonal.
  oracle   an unconstrained per-channel scale by coordinate descent. Uniformly
           scaling a group changes nothing, so the objective decomposes over
           groups and 32 coordinates per group is the whole search space --
           this is what says whether the alpha family or the tensor is the
           limit.
  gateup   the same family on the other side of the block, where the spread
           is, folding into `post_attention_layernorm`.
"""

import json
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aq  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"
IDX = json.load(open(PARENT / "model.safetensors.index.json"))["weight_map"]
HESS = ROOT / "refs/gptq-experts-q0/hessians"
SAL = ROOT / "refs/awq-experts"
ALPHAS = np.linspace(0.0, 1.5, 31)
STEPS = np.array([2.0 ** e for e in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)])


def load(name):
    t = mx.load(str(PARENT / IDX[name]))[name]
    mx.eval(t)
    return np.array(t.astype(mx.float32), copy=False)


def rate(L):
    """The experts' bit width: layers 32-39 are 8-bit/g64, the rest 4-bit/g32."""
    return (8, 64) if L >= 32 else (4, 32)


def rtn(w, gs, bits):
    q, s, b = aq.quantize(w, gs, bits)
    return (q.astype(np.float32).reshape(-1, gs) * s[:, None] + b[:, None]).reshape(w.shape)


def per_expert(L):
    z = np.load(SAL / f"layer{L}.npz")
    n = np.maximum(z["n"], 1)[:, None]
    return z["s1"] / n, z["s2"] / n


def guard(sal):
    """A dead channel and a never-routed one must not drive the normalisation.

    Layer 0's experts are 41% dead in the parent (Phase 20), so this is not a
    hypothetical: without the floor the geometric mean goes to zero and every
    live channel's scale goes to infinity.
    """
    live = sal[sal > 0]
    return np.maximum(sal, np.median(live) * 1e-3 if live.size else 1.0)


def search(w, sal, weight, gs, bits):
    """Best AWQ scale, continuous and power-of-two, by output error."""
    den = float(((w * w) * weight[None, :]).astype(np.float64).sum())

    def err(wq):
        return float(np.sqrt(
            (((w - wq) ** 2) * weight[None, :]).astype(np.float64).sum() / max(den, 1e-30)))

    e0 = err(rtn(w, gs, bits))
    best = {False: (e0, 0.0), True: (e0, 0.0)}
    for a in ALPHAS:
        s = sal ** a
        s = s / np.exp(np.log(s).mean())
        for po2 in (False, True):
            sc = 2.0 ** np.round(np.log2(s)) if po2 else s
            v = err(rtn(w * sc[None, :], gs, bits) / sc[None, :])
            if v < best[po2][0]:
                best[po2] = (v, a)
    return e0, best[False], best[True]


def cmd_spread(layers, _n):
    print(f"{'layer':>5} {'hidden pooled':>14} {'inter pooled':>13} "
          f"{'inter per-expert: median':>25} {'p90':>7} {'max':>7}")
    for L in layers:
        row = [L]
        for name in ("hidden", "inter"):
            g = np.sqrt(np.maximum(np.diag(np.load(HESS / f"layer{L}.{name}.npy")), 0))
            g = g[g > g.max() * 1e-6]
            row.append(g.max() / np.median(g))
        sal, _ = per_expert(L)
        r = []
        for e in range(sal.shape[0]):
            x = sal[e][sal[e] > sal[e].max() * 1e-6]
            if x.size >= 32:
                r.append(x.max() / np.median(x))
        r = np.array(r)
        print(f"{row[0]:>5} {row[1]:>14.2f} {row[2]:>13.2f} "
              f"{np.median(r):>25.2f} {np.percentile(r, 90):>7.2f} {r.max():>7.2f}")


def cmd_down(layers, nexp):
    print(f"{nexp} experts a layer, best alpha per expert, scored on that "
          f"expert's own diagonal")
    print(f"{'layer':>5} {'RTN':>8} {'continuous':>11} {'power of 2':>11} "
          f"{'alpha c/p':>11} {'best expert':>12}")
    for L in layers:
        bits, gs = rate(L)
        sal_all, h2_all = per_expert(L)
        dn = load(f"model.language_model.layers.{L}.mlp.experts.down_proj")[:nexp]
        rows = []
        for e in range(dn.shape[0]):
            e0, (c, ac), (p, ap) = search(dn[e], guard(sal_all[e].copy()),
                                          h2_all[e], gs, bits)
            rows.append((e0, c, p, ac, ap))
        r = np.array(rows)
        g = 100 * (r[:, 1] / r[:, 0] - 1)
        print(f"{L:>5} {r[:,0].mean():>8.4f} {g.mean():>+10.1f}% "
              f"{100*(r[:,2]/r[:,0]-1).mean():>+10.1f}% "
              f"{r[:,3].mean():>5.2f}/{r[:,4].mean():<5.2f} {g.min():>+11.1f}%", flush=True)


def cmd_oracle(layers, nexp):
    print("unconstrained per-channel scale, coordinate descent per group")
    for L in layers:
        bits, gs = rate(L)
        _, h2_all = per_expert(L)
        dn = load(f"model.language_model.layers.{L}.mlp.experts.down_proj")[:nexp]
        for e in range(min(nexp, dn.shape[0])):
            w, hd = dn[e], h2_all[e]

            def gerr(wg, s, hg):
                q, sc, b = aq.quantize(wg * s[None, :], gs, bits)
                deq = (q.astype(np.float32).reshape(-1, gs) * sc[:, None]
                       + b[:, None]).reshape(wg.shape)
                d = wg - deq / s[None, :]
                return float(((d * d) * hg[None, :]).astype(np.float64).sum())

            base = tot = 0.0
            for g in range(w.shape[1] // gs):
                sl = slice(g * gs, (g + 1) * gs)
                wg, hg = w[:, sl], hd[sl]
                s = np.ones(gs, np.float32)
                cur = base_g = gerr(wg, s, hg)
                base += base_g
                for _ in range(3):
                    moved = False
                    for c in range(gs):
                        for m in STEPS:
                            t = s.copy()
                            t[c] *= m
                            v = gerr(wg, t, hg)
                            if v < cur * (1 - 1e-6):
                                cur, s, moved = v, t, True
                    if not moved:
                        break
                tot += cur
            print(f"  L{L} expert {e}: {100 * (np.sqrt(tot / base) - 1):+.1f}%", flush=True)


def cmd_gateup(layers, nexp):
    print(f"gate+up of {nexp} experts, one scale for the whole folding group")
    print(f"{'layer':>5} {'RTN':>8} {'continuous':>11} {'power of 2':>11} {'alpha c/p':>11}")
    for L in layers:
        bits, gs = rate(L)
        H = np.load(HESS / f"layer{L}.hidden.npy").astype(np.float32)
        gu = load(f"model.language_model.layers.{L}.mlp.experts.gate_up_proj")[:nexp]
        mid = gu.shape[1] // 2
        W = np.concatenate([gu[:, :mid, :], gu[:, mid:, :]], axis=1).reshape(-1, gu.shape[2])
        del gu
        sal = guard(np.sqrt(np.maximum(np.diag(H), 0)))

        def quad(A):
            return float(((A @ H) * A).astype(np.float64).sum())

        den = quad(W)
        e0 = float(np.sqrt(quad(W - rtn(W, gs, bits)) / den))
        best = {False: (e0, 0.0), True: (e0, 0.0)}
        for a in ALPHAS:
            s = sal ** a
            s = s / np.exp(np.log(s).mean())
            for po2 in (False, True):
                sc = 2.0 ** np.round(np.log2(s)) if po2 else s
                v = float(np.sqrt(quad(W - rtn(W * sc[None, :], gs, bits) / sc[None, :]) / den))
                if v < best[po2][0]:
                    best[po2] = (v, a)
        (c, ac), (p, ap) = best[False], best[True]
        print(f"{L:>5} {e0:>8.4f} {100*(c/e0-1):>+10.1f}% {100*(p/e0-1):>+10.1f}% "
              f"{ac:>5.2f}/{ap:<5.2f}", flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    layers = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "0,5,20,35,39").split(",")]
    nexp = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    cmds = {"spread": cmd_spread, "down": cmd_down, "oracle": cmd_oracle, "gateup": cmd_gateup}
    for name, fn in cmds.items():
        if what in (name, "all"):
            print(f"\n--- {name} ---")
            fn(layers, 1 if name == "oracle" else nexp)


if __name__ == "__main__":
    main()
