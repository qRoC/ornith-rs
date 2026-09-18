"""Damping sweep: the effect, and the noise floor it has to clear.

    python3 tools/sweep_report.py

`REBUILD.md` Phases 10 and 11. Two families of draw, and they answer different
questions:

  full     two draws a damping value, from the `A` Hessians and the parent's.
           Phase 9 showed those differ by nothing systematic (27 of 58,
           p = 0.69), so the pair is a replicate. Two of them is one degree of
           freedom, which is why the quarters exist.

  quarter  four draws, from four disjoint quarters of the calibration corpus
           (`ORNITH_GPTQ_SPLIT=4`, one pass). Each is a quarter of the data, so
           its Hessian is a noisier estimate than production -- **the spread
           these measure is an upper bound on the spread at full corpus size,
           not an estimate of it.** What they can settle is whether one damping
           value is reproducibly better than another.

Everything is `KL(bf16 parent || variant)` over the live probes; the reference
is `refs/artifact/parent`, captured by `dump_parent_probes.py`.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART, SWEEP = ROOT / "refs/artifact", ROOT / "refs/sweep"

FULL = {  # damping -> the two full-corpus draws
    "0.003": [SWEEP / "d0.003-gptq", SWEEP / "d0.003-gptq-parent"],
    "0.01": [ART / "gptq-head-linattn", ART / "gptqp-head-linattn"],
    "0.03": [SWEEP / "d0.03-gptq", SWEEP / "d0.03-gptq-parent"],
    "0.05": [SWEEP / "d0.05-gptq", SWEEP / "d0.05-gptq-parent"],
    "0.10": [SWEEP / "d0.10-gptq", SWEEP / "d0.10-gptq-parent"],
}
QUARTER = {d: [SWEEP / f"d{d}-q{q}" for q in range(4)] for d in ("0.01", "0.03")}


def read(d):
    m = json.loads((d / "probes.json").read_text())
    return np.fromfile(d / "probes.bin", np.float32).reshape(
        tuple(m["probes"]["shape"])).astype(np.float64)


def main():
    P = read(ART / "parent")

    def kl(d):
        v = read(d)
        return (np.exp(P) * (P - v)).sum(1)

    series, K = {}, {}
    for fam, table in (("full", FULL), ("quarter", QUARTER)):
        for d, dirs in table.items():
            got = [p for p in dirs if (p / "probes.json").exists()]
            if got:
                series[(fam, d)] = got
                for p in got:
                    K[p] = kl(p)
    K["rtn"] = kl(ART / "int4-head-linattn")
    K["A"] = kl(ART / "new")

    live = np.stack(list(K.values())).max(0) >= 1e-3
    rtn = np.median(K["rtn"][live])
    print(f"{int(live.sum())} live probes; reference = bf16 parent")
    print(f"round-to-nearest {rtn:.4f}   control A {np.median(K['A'][live]):.4f}\n")

    for fam in ("full", "quarter"):
        rows = [(d, ps) for (f, d), ps in series.items() if f == fam]
        if not rows:
            continue
        n = max(len(ps) for _, ps in rows)
        print(f"{fam}-corpus draws")
        print(f"{'percdamp':>9}{'n':>3}" + "".join(f"{'draw ' + str(i):>10}"
                                                  for i in range(n))
              + f"{'mean':>9}{'sd':>8}{'vs RTN':>9}{'sd/effect':>11}")
        for d, ps in sorted(rows, key=lambda r: float(r[0])):
            v = np.array([np.median(K[p][live]) for p in ps])
            m, sd, eff = v.mean(), v.std(ddof=1), rtn - v.mean()
            cells = "".join(f"{x:>10.4f}" for x in v) + "".join(
                f"{'':>10}" for _ in range(n - len(v)))
            r = sd / eff if eff > 0 else float("inf")
            print(f"{d:>9}{len(v):>3}{cells}{m:>9.4f}{sd:>8.4f}{eff:>+9.4f}{r:>10.2f}x")
        print()

    if ("quarter", "0.01") in series and ("quarter", "0.03") in series:
        a = np.array([np.median(K[p][live]) for p in series[("quarter", "0.01")]])
        b = np.array([np.median(K[p][live]) for p in series[("quarter", "0.03")]])
        print(f"0.03 vs 0.01 on four quarter draws each:")
        print(f"  0.01 {np.sort(a).round(4)}   0.03 {np.sort(b).round(4)}")
        print(f"  separation: max(0.03) {b.max():.4f} vs min(0.01) {a.min():.4f}"
              f"  -> {'no overlap' if b.max() < a.min() else 'OVERLAP'}")
        w = sum(1 for x in b for y in a if x < y)
        print(f"  rank sum: 0.03 lower in {w} of {len(a) * len(b)} pairings "
              f"(Mann-Whitney, exact p = {mw_p(w, len(a), len(b)):.4f} one-sided)")


def mw_p(u, n, m):
    """Exact one-sided Mann-Whitney tail for small samples, by enumeration."""
    from itertools import combinations
    tot = list(range(n + m))
    cnt = hit = 0
    for c in combinations(tot, m):
        s = set(c)
        uu = sum(1 for x in c for y in tot if y not in s and x < y)
        cnt += 1
        hit += uu >= u
    return hit / cnt


main()
