"""Judge variants on the wide probe set, and judge the probe set itself.

    python3 tools/compare_probes.py [<control capture> [<variant capture>...]]

`REBUILD.md` Phase 5. `compare_baselines.py` asks "is this the same model" of
eight contexts; this asks "how far did the distribution move, and does that
order the variants" of 64 -- the question Phase 4 answered with one context
out of eight.

**A median printed here is a property of the run, not of the variant**, and
two medians from two runs may not be compared. A probe is *live* when some
variant **in this run** pays >= 0.001 nats, and each median is taken over the
live set -- so adding a variant changes every other variant's number. One
unchanged capture read in three compositions (`REBUILD.md` Phase 44):
8 variants -> 59 live -> 0.0334, 2 variants -> 56 live -> 0.0362, alone ->
55 live -> 0.0371. **11% of spread on a file that did not change**, which is
more than most of the gaps this file exists to rank. Compare within one run,
put every arm you mean to rank into it, and re-run the whole table rather
than quoting a row from an older one.

**What makes a probe worth having, measured rather than assumed.** The first
version of this file gated probes on control entropy, on the theory that a
confident prediction is where quantization damage shows. The old eight say
otherwise, and they are the only labelled data there is:

  - Entropy correlates *positively* with KL across those eight (+0.55), so an
    entropy gate at 2 nats would have dropped `Дважды два равно` (H = 3.51),
    the one probe with demonstrated power. Mechanically obvious in hindsight:
    KL against a near-delta distribution is near zero because there is nowhere
    for mass to go.
  - But entropy does not explain it either. `The Rust borrow checker prevents`
    has *higher* entropy (4.40) and does not discriminate -- all four variants
    land within a factor of 2.4. `Дважды два равно` spreads them over 9.4x.

So the property is neither absolute KL nor entropy: it is **spread across
variants**. A probe that moves a lot under every quantization measures that
quantization is lossy, which was never in doubt. A probe that moves under
three and not the fourth is the one that can rank them, and ranking them is
the entire job. This file reports spread as a first-class number and ranks
the probes by it.

**Live probes.** A probe is live when the largest KL any variant pays on it
reaches `FLOOR`. Below that the distribution moved by less than a thousandth
of a nat and its ordering of the variants is a comparison between numbers
that round to zero. Liveness is measured, not predicted -- it falls out of
the captures instead of out of a guess about the text.
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "refs/artifact"
FLOOR = 1e-3


def read(d):
    d = pathlib.Path(d).expanduser()
    meta = json.loads((d / "probes.json").read_text())
    shape = tuple(meta["probes"]["shape"])
    return d, meta, np.fromfile(d / "probes.bin", dtype=np.float32).reshape(shape)


def kl(a, b):
    """KL(a || b) per row, nats, from log-probabilities."""
    return (np.exp(a) * (a - b)).sum(axis=-1)


def main():
    args = sys.argv[1:]
    control = args[0] if args else ARTIFACTS / "new"
    # Whatever is the control is not also a variant. Until Phase 8 the control
    # was always `new` and the exclusion could be spelled as its name; with the
    # bf16 parent captured (`dump_parent_probes.py`), `new` becomes an ordinary
    # variant and is the one that says whether the streamed capture is sane --
    # the least quantized artifact has to come out nearest the parent.
    cname = pathlib.Path(control).expanduser().resolve().name
    variants = args[1:] or sorted(
        p for p in ARTIFACTS.iterdir()
        if p.is_dir() and p.resolve().name != cname and (p / "probes.json").exists()
    )
    if not variants:
        sys.exit("no variant captures with probes.bin; run dump_probes.py first")

    cdir, cmeta, ca = read(control)
    ids = [f"{r['group']}/{r['id']}" for r in cmeta["set"]]
    groups = [r["group"] for r in cmeta["set"]]
    H = np.array([r["entropy"] for r in cmeta["set"]])

    names, K, moved = [], [], []
    for v in variants:
        vdir, vmeta, va = read(v)
        if [f"{r['group']}/{r['id']}" for r in vmeta["set"]] != ids:
            sys.exit(f"{vdir.name} captured a different probe set")
        names.append(vdir.name)
        K.append(kl(ca, va))
        moved.append(int((ca.argmax(-1) != va.argmax(-1)).sum()))
    K = np.array(K)                       # [variant, probe]
    live = K.max(axis=0) >= FLOOR
    spread = K.max(axis=0) / np.maximum(K.min(axis=0), 1e-12)
    best = K.argmin(axis=0)

    print(f"control: {cdir.name}  ({cmeta['model_dir']})")
    print(f"{len(ids)} probes, {len(dict.fromkeys(groups))} groups; "
          f"{int(live.sum())} live (some variant pays >= {FLOOR} nats)\n")

    w = max(max(len(n) for n in names), 7)
    print("KL(control || variant), nats, over the live probes")
    print(f"{'variant':<{w}}  {'median':>8} {'p90':>8} {'max':>8}   "
          f"{'best on':>8}  {'worst on':>8}  top-1 moved")
    for i, n in enumerate(names):
        d = K[i][live]
        wins = int((best[live] == i).sum())
        worst = int((K.argmax(axis=0)[live] == i).sum())
        print(f"{n:<{w}}  {np.median(d):>8.4f} {np.quantile(d,.9):>8.4f} "
              f"{d.max():>8.4f}   {wins:>3} of {int(live.sum())}  "
              f"{worst:>3} of {int(live.sum())}  {moved[i]:>3} of {len(ids)}")

    print("\nmedian KL by group (live probes; n = live of 8)")
    print(f"{'group':<10} {'n':>2}  " + "  ".join(f"{n[:17]:>17}" for n in names))
    for g in dict.fromkeys(groups):
        m = np.array([x == g for x in groups]) & live
        if not m.any():
            print(f"{g:<10} {0:>2}  " + "  ".join(f"{'--':>17}" for _ in names))
            continue
        print(f"{g:<10} {int(m.sum()):>2}  "
              + "  ".join(f"{np.median(K[i][m]):>17.4f}" for i in range(len(names))))

    idx = np.where(live)[0]
    rank = idx[np.argsort(-spread[idx])]
    print(f"\nprobes that rank the variants, by spread (max KL / min KL)")
    print(f"{'#':>3} {'probe':<24} {'H':>5} {'spread':>7}  "
          + "  ".join(f"{n[:17]:>17}" for n in names))
    for k, i in enumerate(rank[:15], 1):
        print(f"{k:>3} {ids[i]:<24} {H[i]:>5.2f} {spread[i]:>7.1f}  "
              + "  ".join(f"{K[j][i]:>17.4f}" for j in range(len(names))))

    if len(names) >= 2:
        order = np.argsort(K[:, live].sum(axis=1))
        a, b = int(order[0]), int(order[1])
        ka, kb = K[a][live], K[b][live]
        print(f"\nhead to head, the two least-damaged: "
              f"{names[a]} vs {names[b]}")
        print(f"  {names[a]} lower on {int((ka < kb).sum())} of {len(ka)} live probes")
        print(f"  median ratio {names[b]}/{names[a]}  "
              f"{np.median(kb / np.maximum(ka, 1e-12)):.2f}x")
        print(f"  largest single gap  {np.max(kb - ka):+.4f} nats on "
              f"{ids[idx[int(np.argmax(kb - ka))]]}")

    dead = [ids[i] for i in range(len(ids)) if not live[i]]
    if dead:
        print(f"\ndead probes ({len(dead)}, no variant reaches {FLOOR} nats): "
              + ", ".join(dead))
    return 0


sys.exit(main())
