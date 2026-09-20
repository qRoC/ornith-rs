"""Turn the port's captured logits into a `probes.bin`, and check the port
against mlx while doing it.

`src/bin/probes.rs` writes raw logits and stops there on purpose: the only
arithmetic between a model's logits and a KL is
`lp = logits - mx.logsumexp(logits)`, and reimplementing that in Rust would
put a second reduction order into the comparison for no reason. So mlx does
it here, to both sides' logits, and the port is only ever asked for the thing
it is bit-exact at.

    python3 tools/probes_from_logits.py <port capture> [<mlx capture to check>]

Writes `probes.bin` + `probes.json` into the port capture, in exactly
`dump_probes.py`'s layout, so `compare_probes.py` reads it without knowing
which instrument produced it. With a second argument it also reports
`max_abs` between the two, which is the number that says whether the port may
stand in for mlx on an artifact mlx cannot load.

The `instrument` block `probes.rs` writes is carried into `probes.json`,
because that is the file a verdict is read from. Captures taken before
Phase 44 have no such block, and that absence is information too: they were
taken with the prefill kernel as a free parameter of the instrument.
"""

import json
import pathlib
import sys

import mlx.core as mx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = pathlib.Path(sys.argv[1]).expanduser()
    meta = json.loads((d / "logits.json").read_text())
    n, vocab = meta["logits"]["shape"]
    logits = np.fromfile(d / "logits.bin", dtype=np.float32).reshape(n, vocab)

    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(pathlib.Path(meta["model_dir"]).expanduser())

    rows, out = [], []
    for r, probe in zip(logits, meta["set"]):
        lg = mx.array(r)                       # already float32, as mlx's astype leaves it
        lp = lg - mx.logsumexp(lg)
        p = mx.exp(lp)
        top = int(mx.argmax(lp).item())
        mx.eval(lp, p)
        rows.append(lp)
        out.append({
            "group": probe["group"],
            "id": probe["id"],
            "text": probe["text"],
            "tokens": probe["tokens"],
            "entropy": float(-(p * lp).sum().item()),
            "top1_p": float(p[top].item()),
            "top1_id": top,
            "top1_text": tokenizer.decode([top]),
        })

    dist = mx.stack(rows)
    (d / "probes.bin").write_bytes(np.array(dist.astype(mx.float32), copy=False).tobytes())
    # Carry the instrument through. A verdict is read from `probes.json`, so
    # that is where the capture has to say which instrument produced it --
    # `logits.json` alone is a file nobody downstream opens. Absent for every
    # capture taken before Phase 44, and absent is itself the information.
    carried = {
        "model_dir": meta["model_dir"],
        "source": meta.get("source", "port"),
    }
    if "instrument" in meta:
        carried["instrument"] = meta["instrument"]
    (d / "probes.json").write_text(json.dumps({
        **carried,
        "probes": {
            "file": "probes.bin",
            "dtype": "float32",
            "shape": list(dist.shape),
            "chunk": meta["logits"]["chunk"],
        },
        "set": out,
    }, indent=1, ensure_ascii=False))
    print(f"wrote {d}/probes.json and probes.bin {tuple(dist.shape)}")

    if len(sys.argv) < 3:
        return 0
    ref = pathlib.Path(sys.argv[2]).expanduser()
    rmeta = json.loads((ref / "probes.json").read_text())
    rshape = tuple(rmeta["probes"]["shape"])
    ra = np.fromfile(ref / "probes.bin", dtype=np.float32).reshape(rshape)
    got = np.array(dist.astype(mx.float32), copy=False)
    if [f"{x['group']}/{x['id']}" for x in rmeta["set"]] != [f"{x['group']}/{x['id']}" for x in out]:
        sys.exit("the two captures are not the same probe set")

    bad_tok = [(a["id"], a["tokens"], b["tokens"])
               for a, b in zip(out, rmeta["set"]) if a["tokens"] != b["tokens"]]
    print(f"\ntokenization: {len(out) - len(bad_tok)} of {len(out)} probes agree on token count"
          + ("" if not bad_tok else f"\n  disagree: {bad_tok}"))

    diff = np.abs(got - ra)
    per = diff.max(axis=-1)
    print(f"\n{d.name} vs {ref.name}   max_abs = {diff.max():.6g}   "
          f"exact = {bool((got == ra).all())}")
    print(f"  rows bit-identical: {int((got == ra).all(axis=-1).sum())} of {len(out)}")
    if diff.max() > 0:
        worst = int(np.argmax(per))
        print(f"  worst probe: {out[worst]['group']}/{out[worst]['id']}  {per[worst]:.6g}")
        kl = float((np.exp(ra) * (ra - got)).sum(axis=-1).max())
        print(f"  largest KL(mlx || port) over the 64: {kl:.3e} nats")
        print(f"  top-1 moved on {int((ra.argmax(-1) != got.argmax(-1)).sum())} of {len(out)}")
    return 0


sys.exit(main())
