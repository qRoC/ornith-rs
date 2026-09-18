"""Forge the GPTQ variant: `int4-head-linattn`, byte for byte, re-rounded.

    PYTHONPATH=../vendor/mlx-lm python3 tools/forge_gptq.py [--off]

`REBUILD.md` Phase 7. Phase 4 closed the *format* (four mlx modes plus the
best non-mlx 4.50-bit one, all measured), Phase 5 closed the *coverage* (the
knee is here, 44 of 60 probes, p = 0.0004), Phase 6 closed the *scaling*
(activation-aware, built, does not pay). That leaves one degree of freedom in
the artifact that nothing has touched: **which integer each weight rounds
to**. This file spends it.

Same 91 modules, same affine int4 g32, same three tensors, same shapes, same
shards, same kernels, same bytes a token. `tools/gptq.py` has the algorithm
and the four decisions inside it.

**`--off` is the control and it is not a formality.** It runs the identical
code path with the compensation term zeroed, and asserts the result is the
base artifact's own tensors, bit for bit -- codes, scales and biases. That is
what makes "the only variable is the rounding" a checked statement rather
than a claim about intent. It is the same role `alpha = 0` played in
`forge_awq.py`.

**What this cannot do, and it is the honest half of the pitch.** GPTQ's
objective is `||(W - Wq) X^T||`, which is *precisely* the calibration output
error Phase 6 measured and found wanting: the activation-aware family bought
six points of it and no KL at all. So the prior here is lower than the
literature's, and the verdict belongs to the 64 probes and the median KL, not
to the objective this file minimises. `gptq.py` reports the objective anyway,
because a result is worth explaining even when it cannot be worth choosing by.

**Where it differs from Phase 6, and why it is worth one build.** AWQ had a
fold point: the inverse of any scale had to be absorbed by a preceding op,
which put `out_proj` and `lm_head` out of reach and left one family of three.
GPTQ folds nothing into anything -- it only moves integers inside the grid
they were already going to be written on -- so all 91 modules are reachable,
including the two AWQ could not touch.
"""

import json
import os
import pathlib
import shutil
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aq       # noqa: E402
import gptq     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
HESS = ROOT / os.environ.get("ORNITH_GPTQ_OUT", "refs/gptq")
MODELS = pathlib.Path.home() / ".mtplx/models"
PARENT = MODELS / "ornith-ai--Ornith-1.5-35B-A3B"
BASE = MODELS / "ornith-1.5-35b-a3b-int4-head-linattn"
OUT = MODELS / os.environ.get("ORNITH_GPTQ_ARTIFACT",
                              "ornith-1.5-35b-a3b-gptq-head-linattn")
GROUP_SIZE, BITS = 32, 4
PERCDAMP = float(os.environ.get("ORNITH_GPTQ_DAMP", "0.01"))
FAMILIES = ("in_proj_qkv", "in_proj_z", "out_proj")


def hessian_name(base_key):
    """The module whose input this weight is multiplied by.

    `in_proj_z` has no Hessian of its own: it reads the same tensor as
    `in_proj_qkv`, which `dump_gptq_stats.py` asserts bit for bit rather than
    assuming.
    """
    m = base_key.rsplit(".", 1)[0]
    return m.replace("in_proj_z", "in_proj_qkv")


def parent_key(base_key):
    if base_key.startswith("language_model.lm_head."):
        return base_key.replace("language_model.lm_head.", "lm_head.")
    return base_key.replace("language_model.model.", "model.language_model.")


def main():
    off = "--off" in sys.argv
    if not off and OUT.exists():
        sys.exit(f"{OUT} exists; remove it or set ORNITH_GPTQ_ARTIFACT")
    calib = json.loads((HESS / "calib.json").read_text())
    print(f"calibration: {calib['samples']} samples, {calib['tokens']} tokens, "
          f"{calib['model_dir'].split('/')[-1]}")

    bidx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]

    targets = [k for k in bidx
               if k.endswith(".weight")
               and (k == "language_model.lm_head.weight"
                    or any(k.endswith(f"linear_attn.{f}.weight") for f in FAMILIES))]
    assert len(targets) == 91, f"expected 91 modules, matched {len(targets)}"

    writes = {}
    for k in targets:
        writes.setdefault(bidx[k], []).append(k)
    print(f"{len(targets)} modules across {len(writes)} shards"
          + (" -- CONTROL RUN, nothing written" if off else ""))

    if not off:
        OUT.mkdir(parents=True)
        # `ORNITH_FORGE_LINK=1` hard-links the shards this run does not rewrite
        # instead of copying them. Only for sweeps, where the artifacts are
        # transient: a hard link means editing the file in one artifact edits it
        # in the other, which is exactly wrong for anything kept.
        link = os.environ.get("ORNITH_FORGE_LINK") == "1"
        for f in sorted(BASE.iterdir()):
            if not f.is_file() or f.name in writes:
                continue
            try:
                os.link(f, OUT / f.name) if link else shutil.copy2(f, OUT / f.name)
            except OSError:
                shutil.copy2(f, OUT / f.name)

    Hcache, report, t0, done = {}, {}, time.time(), 0
    for shard in sorted(writes):
        tensors = dict(mx.load(str(BASE / shard)))
        for key in sorted(writes[shard]):
            hn = hessian_name(key)
            if hn not in Hcache:
                Hcache.clear()          # one Hessian live at a time: 67 MB each
                Hcache[hn] = np.load(HESS / "hessians" / f"{hn}.npy")
            H = Hcache[hn]

            pk = parent_key(key)
            t = mx.load(str(PARENT / pidx[pk]))[pk]
            mx.eval(t)
            w = np.array(t.astype(mx.float32), copy=False)
            del t
            assert w.shape[1] == H.shape[0], (key, w.shape, H.shape)

            Hinv, dead = gptq.prepare(H, PERCDAMP)
            q, S, B = gptq.quantize(w, Hinv, dead, GROUP_SIZE, BITS,
                                    compensate=not off)

            out, cols = w.shape
            gshape = [out, cols // GROUP_SIZE]
            wshape = [out, cols * BITS // 32]
            packed = aq.pack4(q.reshape(-1), wshape[1]).reshape(wshape)

            if off:
                base_w = np.array(tensors[key], copy=False)
                base_s = np.array(tensors[key[:-6] + "scales"].astype(mx.float32),
                                  copy=False)
                base_b = np.array(tensors[key[:-6] + "biases"].astype(mx.float32),
                                  copy=False)
                assert np.array_equal(packed, base_w), f"{key}: codes differ"
                assert np.array_equal(S, base_s), f"{key}: scales differ"
                assert np.array_equal(B, base_b), f"{key}: biases differ"
                done += 1
                print(f"  [{done:2d}/91] {key.replace('language_model.', ''):<48} "
                      f"identical to base", flush=True)
                del w, q, S, B
                continue
            else:
                tensors[key] = mx.array(packed)
                tensors[key[:-6] + "scales"] = mx.array(S).astype(mx.bfloat16)
                tensors[key[:-6] + "biases"] = mx.array(B).astype(mx.bfloat16)

            # What it bought, in the objective's own currency and in weight error.
            deq = (q.astype(np.float32)
                   * np.repeat(S, GROUP_SIZE, axis=1)
                   + np.repeat(B, GROUP_SIZE, axis=1))
            rtn_q, rtn_s, rtn_b = aq.quantize(w, GROUP_SIZE, BITS)
            rtn = (rtn_q.reshape(out, cols).astype(np.float32)
                   * np.repeat(aq.bf16(rtn_s).reshape(gshape), GROUP_SIZE, axis=1)
                   + np.repeat(aq.bf16(rtn_b).reshape(gshape), GROUP_SIZE, axis=1))
            report[key] = {
                "out_err_gptq": gptq.relative_output_error(w, deq, H),
                "out_err_rtn": gptq.relative_output_error(w, rtn, H),
                "w_err_gptq": float(np.linalg.norm(w - deq) / np.linalg.norm(w)),
                "w_err_rtn": float(np.linalg.norm(w - rtn) / np.linalg.norm(w)),
                "dead": int(dead.sum()),
            }
            r = report[key]
            done += 1
            print(f"  [{done:2d}/91] {key.replace('language_model.', ''):<48} "
                  f"out {r['out_err_rtn']:.4f}->{r['out_err_gptq']:.4f} "
                  f"({100 * (r['out_err_gptq'] / r['out_err_rtn'] - 1):+.1f}%)  "
                  f"w {r['w_err_rtn']:.4f}->{r['w_err_gptq']:.4f}", flush=True)
            del w, q, S, B, deq, rtn

        if not off:
            mx.eval(*tensors.values())
            mx.save_safetensors(str(OUT / shard), tensors, metadata={"format": "mlx"})
        del tensors

    if off:
        print(f"\nCONTROL: all {done} modules reproduce the base bit for bit "
              f"in {time.time() - t0:.0f}s")
        return

    g = np.array([r["out_err_gptq"] for r in report.values()])
    n = np.array([r["out_err_rtn"] for r in report.values()])
    print(f"\noutput error: {100 * (g / n - 1).mean():+.2f}% mean, "
          f"{100 * (g / n - 1).min():+.2f}% best, {100 * (g / n - 1).max():+.2f}% worst")

    (OUT / "gptq.json").write_text(json.dumps({
        "base": BASE.name, "parent": PARENT.name,
        "modules": len(targets), "group_size": GROUP_SIZE, "bits": BITS,
        "percdamp": PERCDAMP, "act_order": False,
        "calibration": calib,
        "per_module": report,
    }, indent=1))
    print(f"\n{done} modules re-rounded in {time.time() - t0:.0f}s -> {OUT}")
    print("\nalign_rebuild.py is NOT needed -- the base was already aligned. Next:")
    print(f"  python3 tools/check_rebuild.py {OUT}")
    print(f"  cd tools && ORNITH_BASELINE_OUT=../refs/artifact/gptq-head-linattn \\")
    print(f"    PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py {OUT}")
    print(f"  cd tools && ORNITH_PROBES_OUT=../refs/artifact/gptq-head-linattn \\")
    print(f"    PYTHONPATH=../../vendor/mlx-lm python3 dump_probes.py {OUT}")


main()
