"""Re-round the routed experts -- 93.1% of the artifact, and nothing has touched them.

    PYTHONPATH=../vendor/mlx-lm python3 tools/forge_gptq_experts.py [--off]

`REBUILD.md` Phase 12. Phases 7 to 11 established that GPTQ buys **23%** of
round-to-nearest's damage, byte for byte, and measured it properly: 14
independent forges, sign test p = 6e-05, per-forge spread 39% of the effect.
All of it came from 91 modules that are **1.52 B parameters, 3.7%** of the
model. The 256 experts of 40 layers are **38.65 B, 93.1%**, quantized by
round-to-nearest since the parent's own recipe, and their damage lives in the
0.0185 nats the *control* pays -- a pool no phase has attacked.

This file spends the same lever on them. Same affine int4 g32, same shapes,
same shards, same kernels, same bytes a token; only which integer each expert
weight rounds to changes.

**Base is `int4-head-linattn`, not the GPTQ'd variant, deliberately.** That
isolates the experts: whatever this buys is comparable to the 23% the 91
modules bought, measured against the same round-to-nearest. Combining the two
is the next build, not this one.

**One Hessian a layer, shared by its 256 experts** -- see
`dump_gptq_experts.py` for why the exact per-expert object is 172 GB and what
the pooling costs. `gate_proj` and `up_proj` share the 2048-dimensional one;
`down_proj` has its own at 512.

**The layout.** The parent stores the experts fused, `gate_up_proj` as
`[256, 1024, 2048]`, which `sanitize` splits at the middle of the second axis.
Quantization groups run along the last axis and every row is independent, so a
stack of 256 experts is just `[256 * out, in]` to the quantizer -- which is
also what makes one column walk serve all 256 at once.

**`--off` is the control**: identical path, compensation zeroed, asserts the
result is the base artifact's own expert tensors bit for bit. It is what makes
"the only variable is the rounding" checked rather than intended, and here it
also checks the fused split, the reshape and the packing.
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
HESS = ROOT / os.environ.get("ORNITH_GPTQ_OUT", "refs/gptq-experts")
MODELS = pathlib.Path.home() / ".mtplx/models"
PARENT = MODELS / "ornith-ai--Ornith-1.5-35B-A3B"
BASE = MODELS / "ornith-1.5-35b-a3b-int4-head-linattn"
OUT = MODELS / os.environ.get("ORNITH_GPTQ_ARTIFACT",
                              "ornith-1.5-35b-a3b-xgptq-head-linattn")
GROUP_SIZE, BITS = 32, 4          # the default; layers 32-39 override it
PERCDAMP = float(os.environ.get("ORNITH_GPTQ_DAMP", "0.01"))


def main():
    off = "--off" in sys.argv
    if not off and OUT.exists():
        sys.exit(f"{OUT} exists; remove it or set ORNITH_GPTQ_ARTIFACT")
    calib = json.loads((HESS / "calib.json").read_text())
    print(f"calibration: {calib['tokens']} tokens, split {calib.get('split')}, "
          f"{calib['model_dir'].split('/')[-1]}")

    # Per-tensor quantization, because the experts are **not uniform**: the
    # last eight layers hold theirs at 8 bits / group 64. Read it rather than
    # assume it; `--off` is what found this.
    cfg = json.loads((BASE / "config.json").read_text())
    qcfg = cfg.get("quantization") or cfg["text_config"]["quantization"]

    def qof(key):
        o = qcfg.get(key[:-7]) if isinstance(qcfg.get(key[:-7]), dict) else None
        return ((o or qcfg)["bits"], (o or qcfg)["group_size"])

    bidx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in bidx
                     if "switch_mlp.gate_proj.weight" in k})
    assert len(layers) == 40, layers

    writes = {}
    for i in layers:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            k = f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}.weight"
            writes.setdefault(bidx[k], set()).add(i)
    print(f"{len(layers)} layers x 3 projections across {len(writes)} shards"
          + (" -- CONTROL RUN, nothing written" if off else ""))

    if not off:
        OUT.mkdir(parents=True)
        link = os.environ.get("ORNITH_FORGE_LINK") == "1"
        for f in sorted(BASE.iterdir()):
            if not f.is_file() or f.name in writes:
                continue
            try:
                os.link(f, OUT / f.name) if link else shutil.copy2(f, OUT / f.name)
            except OSError:
                shutil.copy2(f, OUT / f.name)

    t0, done, report = time.time(), 0, {}
    for shard in sorted(writes):
        tensors = dict(mx.load(str(BASE / shard)))
        for i in sorted(writes[shard]):
            Hh = np.load(HESS / "hessians" / f"layer{i}.hidden.npy")
            Hi = np.load(HESS / "hessians" / f"layer{i}.inter.npy")
            inv = {"hidden": gptq.prepare(Hh, PERCDAMP),
                   "inter": gptq.prepare(Hi, PERCDAMP)}

            pre = f"model.language_model.layers.{i}.mlp.experts."
            gu = mx.load(str(PARENT / pidx[pre + "gate_up_proj"]))[pre + "gate_up_proj"]
            dn = mx.load(str(PARENT / pidx[pre + "down_proj"]))[pre + "down_proj"]
            mx.eval(gu, dn)
            mid = gu.shape[-2] // 2
            parts = {"gate_proj": (np.array(gu[..., :mid, :].astype(mx.float32),
                                            copy=False), "hidden"),
                     "up_proj": (np.array(gu[..., mid:, :].astype(mx.float32),
                                          copy=False), "hidden"),
                     "down_proj": (np.array(dn.astype(mx.float32), copy=False),
                                   "inter")}
            del gu, dn

            for proj, (w3, space) in parts.items():
                key = f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}.weight"
                ne, out, cols = w3.shape
                bits, gs = qof(key)
                Hinv, dead = inv[space]
                q, S, B = gptq.quantize(w3.reshape(ne * out, cols), Hinv, dead,
                                        gs, bits, compensate=not off)
                gshape = [ne, out, cols // gs]
                wshape = [ne, out, cols * bits // 32]
                packed = aq.pack(q.reshape(-1), bits, wshape[-1]).reshape(wshape)
                S3, B3 = S.reshape(gshape), B.reshape(gshape)

                if off:
                    for got, name in ((packed, "weight"), (S3, "scales"), (B3, "biases")):
                        k = key[:-6] + name
                        ref = np.array(tensors[k] if name == "weight"
                                       else tensors[k].astype(mx.float32), copy=False)
                        assert np.array_equal(got, ref), f"L{i}.{proj}: {name} differ"
                else:
                    tensors[key] = mx.array(packed)
                    tensors[key[:-6] + "scales"] = mx.array(S3).astype(mx.bfloat16)
                    tensors[key[:-6] + "biases"] = mx.array(B3).astype(mx.bfloat16)
                    w2 = w3.reshape(ne * out, cols)
                    deq = (q.astype(np.float32) * np.repeat(S, gs, axis=1)
                           + np.repeat(B, gs, axis=1))
                    rq, rs, rb = aq.quantize(w2, gs, bits)
                    rtn = (rq.reshape(w2.shape).astype(np.float32)
                           * np.repeat(aq.bf16(rs).reshape(S.shape), gs, axis=1)
                           + np.repeat(aq.bf16(rb).reshape(B.shape), gs, axis=1))
                    H = Hh if space == "hidden" else Hi
                    report[f"L{i}.{proj}"] = {
                        "out_err_gptq": gptq.relative_output_error(w2, deq, H),
                        "out_err_rtn": gptq.relative_output_error(w2, rtn, H),
                        "w_err_gptq": float(np.linalg.norm(w2 - deq) / np.linalg.norm(w2)),
                        "w_err_rtn": float(np.linalg.norm(w2 - rtn) / np.linalg.norm(w2)),
                    }
                    del deq, rtn, rq, rs, rb, w2
                del q, S, B, packed, S3, B3
                done += 1
            del parts, inv, Hh, Hi
            # Per projection, not averaged: `down_proj` reads the SwiGLU output,
            # which is nearly isotropic (off-diagonal mass 0.15 against 0.93 for
            # the gate/up input), so GPTQ degenerates to round-to-nearest there
            # and would drag a mean down for a reason that is not about the
            # experts at all.
            msg = "identical to base" if off else "  ".join(
                f"{p[0]}{100 * (report[f'L{i}.{p}_proj']['out_err_gptq'] / report[f'L{i}.{p}_proj']['out_err_rtn'] - 1):+.1f}%"
                for p in ("gate", "up", "down"))
            print(f"  [{done:3d}/120] layer {i:>2}  {msg}", flush=True)

        if not off:
            mx.eval(*tensors.values())
            mx.save_safetensors(str(OUT / shard), tensors, metadata={"format": "mlx"})
        del tensors

    if off:
        print(f"\nCONTROL: all {done} expert tensors reproduce the base bit for "
              f"bit in {time.time() - t0:.0f}s")
        return

    g = np.array([r["out_err_gptq"] for r in report.values()])
    n = np.array([r["out_err_rtn"] for r in report.values()])
    print(f"\noutput error: {100 * (g / n - 1).mean():+.2f}% mean, "
          f"{100 * (g / n - 1).min():+.2f}% best, {100 * (g / n - 1).max():+.2f}% worst")
    (OUT / "gptq.json").write_text(json.dumps({
        "base": BASE.name, "parent": PARENT.name, "scope": "routed experts",
        "layers": len(layers), "group_size": GROUP_SIZE, "bits": BITS,
        "percdamp": PERCDAMP, "shared_hessian_per_layer": True,
        "calibration": calib, "per_module": report}, indent=1))
    print(f"\n{done} expert tensors re-rounded in {time.time() - t0:.0f}s -> {OUT}")


main()
