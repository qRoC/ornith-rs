"""Forge the real Q4_K artifact: the 91 modules, in ggml's own bytes.

    ORNITH_GGML_BUILD=../../.ggml-build python3 tools/forge_q4k.py [options]

      --imatrix          feed the quantizer diag(H) as its importance vector
      --experts          also convert the 96 routed-expert modules
      --q6k              convert the *8-bit* side to Q6_K instead
      --resume           keep an existing output directory and its copies
      --from <name>      base artifact to build on (default int4-head-linattn)
      --out <name>       output artifact name

`Q4K.md`, after the kernel. Phase 14 measured a Q4_K-*shaped* grid as a
stand-in -- power-of-two super-scales and magnitude-plus-sign fields, so it
could be written into the affine layout and run without a kernel -- and read
the same median KL as the 5.00-bit base for 7.3% fewer bytes. Both constraints
were scaffolding, and dropping them is what makes llama.cpp the oracle. This
writes the format itself.

**The quantizer is ggml's.** `tools/q4k.py`'s writer is a faithful port of
`quantize_row_q4_K_ref`'s shape and is 0.77 points worse, because ggml runs a
search rather than plain min/max. Nothing that ships should use it;
`q4k_ggml.quantize` calls the real one through ctypes.

**The 91 modules** are the ones `int4-head-linattn` moved to 4 bits: 30
gated-delta layers' `in_proj_qkv`, `in_proj_z` and `out_proj`, plus `lm_head`.
They are 57% of a decode step's traffic and they are what every number since
Phase 4 is measured on.

**`--experts` adds the other 96** -- `switch_mlp.gate_proj` / `up_proj` /
`down_proj` for the 32 layers `int4-head-linattn` leaves at 4 bits. They are
0.7172 GB of a decode step, 34.5%, and the reason they were out of the first
cut is that they need the gathered kernel (`q4k_gather_qmv`, ggml's
`kernel_mul_mv_id`). Their importance vectors are the *expert* Hessians in
`refs/gptq-experts-q0/hessians/`: `layerN.hidden` for the two that read the
hidden state, `layerN.inter` for `down_proj`, which reads the intermediate.
One vector serves all 256 experts of a layer, because the input distribution
is the layer's and not the expert's. Everything else is copied from the base artifact byte for
byte, so the only difference between the two is the thing under test.

**Storage**, `Q4K.md`'s three tensors per module -- ggml's bytes, split so the
loader and `check_rebuild.py` see shapes they already know:

    <mod>.weight   uint32  [O, I / 8]        the 4-bit codes
    <mod>.scales   uint8   [O, I / 256, 12]  8 x 6-bit scale + 8 x 6-bit min
    <mod>.dm       float16 [O, I / 256, 2]   d and dmin

and `.biases` is dropped: Q4_K's minimum is one of the packed 6-bit fields.
`config.json` gets `mode: "q4k"`, `group_size: 256` per converted module.

`--imatrix` passes `diag(H)` from `refs/gptq-parent/hessians/` as
`quantize_q4_K`'s importance vector, which measured 8 to 80% off the
H-weighted weight error for 1.3 to 5.2% of the unweighted one. Only 61 of the
91 have a Hessian of their own; `in_proj_z` borrows `in_proj_qkv`'s, which is
the same matrix because the two read the same tensor, so the coverage is all
91 at no calibration cost. Measured on four layers: H-weighted error down
14-30% for 1-4% of unweighted.
"""

import json
import pathlib
import shutil
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import q4k        # noqa: E402
import q4k_ggml   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = pathlib.Path.home() / ".mtplx/models"
PARENT = MODELS / "ornith-ai--Ornith-1.5-35B-A3B"
HESSIANS = ROOT / "refs/gptq-parent/hessians"

def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


IMATRIX = "--imatrix" in sys.argv
EXPERTS = "--experts" in sys.argv
# Q6_K takes the other end of the artifact: the 169 modules held at affine
# int8 g64 = 8.50 bits, where 6.5625 is 22.8% fewer bytes. `Q4K.md` priced it
# at 0.68 error-doublings a bit against the 0.96 the 4-bit side already pays.
Q6K = "--q6k" in sys.argv
KIND = "q6_K" if Q6K else "q4_K"
MODE = "q6k" if Q6K else "q4k"
SRC_BITS = 8 if Q6K else 4
BASE = MODELS / _opt("--from", "ornith-1.5-35b-a3b-int4-head-linattn")
OUT = MODELS / _opt(
    "--out",
    "ornith-1.5-35b-a3b-"
    + MODE
    + ("-imatrix" if IMATRIX else "")
    + ("-experts" if EXPERTS else ""),
)
EXPERT_HESSIANS = ROOT / "refs/gptq-experts-q0/hessians"

# `Q4K.md`'s first cut. A module is in when `int4-head-linattn` moved it to 4
# bits, which is exactly these three suffixes plus `lm_head`.
SUFFIXES = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
EXPERT_SUFFIXES = (
    "mlp.switch_mlp.gate_proj",
    "mlp.switch_mlp.up_proj",
    "mlp.switch_mlp.down_proj",
)

# The 8-bit side, minus two that the kernel cannot take whatever the config
# says: `embed_tokens` is read by a gather and never by a matvec, and
# `shared_expert_gate` is `[1, 2048]` where the kernel covers four rows at a
# time. That is 128 of the 169, and the 41 left out are 0.0% and 0.8% of a
# decode step.
Q6K_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
) + EXPERT_SUFFIXES


def parent_of(mod):
    """Artifact module path -> (parent tensor name, which half or None).

    `forge_q4k_twin.py`'s map. The 256 experts' `gate_proj` and `up_proj` are
    the two halves of one fused `mlp.experts.gate_up_proj` in the parent.
    """
    if mod == "language_model.lm_head":
        return "lm_head.weight", None
    assert mod.startswith("language_model.model.layers."), mod
    n, rest = mod[len("language_model.model.layers.") :].split(".", 1)
    base = f"model.language_model.layers.{n}."
    if rest in SUFFIXES or rest in Q6K_SUFFIXES[:8]:
        return base + rest + ".weight", None
    if rest == "mlp.switch_mlp.gate_proj":
        return base + "mlp.experts.gate_up_proj", "lo"
    if rest == "mlp.switch_mlp.up_proj":
        return base + "mlp.experts.gate_up_proj", "hi"
    if rest == "mlp.switch_mlp.down_proj":
        return base + "mlp.experts.down_proj", None
    raise AssertionError(mod)


def hessian_for(mod):
    """`diag(H)` for a module, or None when no calibration run covered it."""
    # `in_proj_z` never had a Hessian of its own and does not need one: it
    # reads the *same* tensor `in_proj_qkv` does -- `qkv = in_proj_qkv(inputs)`
    # and `z = in_proj_z(inputs)`, one line apart in both mlx-lm's
    # `qwen3_5.py` and `qwen3_5.rs` -- so `X^T X` is the same matrix and so is
    # its diagonal. That takes the imatrix from 61 of the 91 to all of them at
    # no calibration cost.
    if mod.endswith("linear_attn.in_proj_z"):
        mod = mod[: -len("in_proj_z")] + "in_proj_qkv"
    # Everything that reads the post-attention normed hidden shares one
    # Hessian: the router, the shared expert's two input projections and the
    # routed experts' two. `down_proj` reads the routed experts' intermediate,
    # which is what `layerN.inter` was captured on -- and the *shared*
    # expert's intermediate is a different tensor, so it gets none.
    hidden_readers = (".switch_mlp.gate_proj", ".switch_mlp.up_proj", ".mlp.gate",
                      ".shared_expert.gate_proj", ".shared_expert.up_proj")
    if mod.endswith(".switch_mlp.down_proj") or mod.endswith(hidden_readers):
        n = mod.split(".layers.")[1].split(".")[0]
        which = "inter" if mod.endswith(".switch_mlp.down_proj") else "hidden"
        f = EXPERT_HESSIANS / f"layer{n}.{which}.npy"
    else:
        f = HESSIANS / f"{mod}.npy"
    if not f.exists():
        return None
    # The files hold the whole `H`, [in, in]; the importance vector is its
    # diagonal. Passing the array as it loads hands `quantize_q4_K` H's first
    # row, negatives and all, and the answer is 38589% error.
    h = np.ascontiguousarray(np.diag(np.asarray(np.load(f), np.float32)))
    assert (h >= 0).all(), f"{mod}: diag(H) has negative entries"
    return h


RESUME = "--resume" in sys.argv


def main():
    if OUT.exists() and not RESUME:
        sys.exit(f"{OUT} exists; remove it, pick another name, or pass --resume")
    lib = q4k_ggml.load()
    acfg = json.loads((BASE / "config.json").read_text())
    aidx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]

    want = Q6K_SUFFIXES if Q6K else SUFFIXES + (EXPERT_SUFFIXES if EXPERTS else ())
    mods = sorted({k.rsplit(".", 1)[0] for k in aidx if k.endswith(".scales")})
    already = {m for m, v in acfg["quantization"].items()
               if isinstance(v, dict) and v.get("mode") in ("q4k", "q6k")}
    q = acfg["quantization"]

    def bits_of(m):
        o = q.get(m)
        return (o or q).get("bits", q["bits"]) if isinstance(o, (dict, type(None))) else q["bits"]

    convert = [
        m
        for m in mods
        if m not in already
        and bits_of(m) == SRC_BITS
        and ((m == "language_model.lm_head" and not Q6K)
             or m.split(".layers.")[-1].split(".", 1)[-1] in want)
    ]
    # **Only what is already 4-bit.** The eight last layers keep their experts
    # at int8 g64 because Phase 4 put the knee there, and `EXPERT_SUFFIXES`
    # matches them by name. Converting those would be a *coverage* change
    # wearing a format change's clothes, and the comparison would measure the
    # wrong thing -- it is 24 of the 120 modules the name filter alone selects.
    if already:
        print(f"{len(already)} modules are already q4k in {BASE.name}; leaving them alone")
    print(f"{len(mods)} quantized modules, converting {len(convert)} "
          f"({SRC_BITS}-bit) to {KIND}" + (" with diag(H)" if IMATRIX else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    touched = {aidx[m + ".weight"] for m in convert}
    # The untouched files are ~20 GB and copying them takes longer than the
    # quantization does on a busy disk, so a re-run skips what is already
    # there byte count for byte count. Restarting a forge is otherwise the
    # most expensive way to change one line.
    for f in sorted(BASE.iterdir()):
        if not f.is_file() or f.name in touched:
            continue
        dst = OUT / f.name
        if dst.exists() and dst.stat().st_size == f.stat().st_size:
            continue
        shutil.copy2(f, dst)
    print(f"copied {len(list(OUT.iterdir()))} files unchanged; "
          f"rewriting {len(touched)} shards")

    by_shard = {}
    for m in convert:
        by_shard.setdefault(aidx[m + ".weight"], []).append(m)

    t0 = time.time()
    done = with_h = 0
    bytes_before = bytes_after = 0
    for shard in sorted(touched):
        tensors = dict(mx.load(str(BASE / shard)))
        for mod in by_shard[shard]:
            tm = [time.time()]
            pname, half = parent_of(mod)
            t = mx.load(str(PARENT / pidx[pname]))[pname]
            if half:
                h2 = t.shape[1] // 2
                t = t[:, :h2, :] if half == "lo" else t[:, h2:, :]
            mx.eval(t)
            w = np.ascontiguousarray(np.array(t.astype(mx.float32), copy=False))
            del t
            # A routed module is [E, O, I]; `quantize_q4_K` takes rows of `I`
            # and the importance vector is per input column, so all E * O rows
            # go in one call and the vector serves every expert.
            tm.append(time.time())
            experts = w.shape[0] if w.ndim == 3 else 1
            w = w.reshape(-1, w.shape[-1])
            rows, cols = w.shape
            assert cols % q4k.QK_K == 0, f"{mod}: {cols} is not a multiple of 256"
            assert rows % 4 == 0, f"{mod}: {rows} rows, kernel covers 4 at a time"
            nb = cols // q4k.QK_K

            h = hessian_for(mod) if IMATRIX else None
            with_h += h is not None
            tm.append(time.time())
            blocks = q4k_ggml.quantize(lib, w, imatrix=h, kind=KIND)
            tm.append(time.time())
            qs, scales, dm = q4k.split(blocks, rows, nb, KIND)
            del blocks
            tm.append(time.time())
            if experts > 1:
                qs = qs.reshape(experts, rows // experts, -1)
                scales = scales.reshape(experts, rows // experts, nb, scales.shape[-1])
                dm = dm.reshape(experts, rows // experts, nb, dm.shape[-1])

            for suffix in ("weight", "scales", "biases", "dm"):
                tensors.pop(f"{mod}.{suffix}", None)
            # affine int4 g32 is 5.00 bits, affine int8 g64 is 8.50
            bytes_before += int(rows * cols * (8.5 if Q6K else 5.0) / 8)
            bytes_after += rows * nb * q4k_ggml.KINDS[KIND][0]
            tensors[f"{mod}.weight"] = mx.array(qs)
            tensors[f"{mod}.scales"] = mx.array(scales)
            tensors[f"{mod}.dm"] = mx.array(dm)
            tm.append(time.time())
            del w, qs, scales, dm
            # No `mx.clear_cache()` here, and that is a measured decision
            # rather than an omission: adding one took the loop from 13 s a
            # module to over twelve minutes. The per-module timings below stay
            # flat while the wall clock does not, which is what says the cost
            # is outside the work -- releasing ~1 GB of device buffers and
            # re-acquiring them, on a machine whose swap is full, is page
            # faults rather than housekeeping.
            done += 1
            d = [tm[i + 1] - tm[i] for i in range(len(tm) - 1)]
            print(f"  [{done:3d}/{len(convert)}] {mod}"
                  + ("  +imatrix" if h is not None else "")
                  + f"  [load {d[0]:.1f} hess {d[1]:.1f} quant {d[2]:.1f} "
                    f"split {d[3]:.1f} to-mx {d[4]:.1f}]", flush=True)
        mx.eval(*tensors.values())
        mx.save_safetensors(str(OUT / shard), tensors, metadata={"format": "mlx"})
        del tensors

    # The index has to lose `.biases` and gain `.dm`, or the loader looks for a
    # tensor no shard holds.
    idx = json.loads((BASE / "model.safetensors.index.json").read_text())
    for m in convert:
        shard = idx["weight_map"].pop(m + ".biases")
        idx["weight_map"][m + ".dm"] = shard
    (OUT / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))

    for m in convert:
        # `bits` is informational for a k-quant -- the real rate is 4.50 or
        # 6.5625 and `mode` is what decides the layout -- but mlx-lm's schema
        # wants an integer, so it carries the code width.
        acfg["quantization"][m] = {
            "group_size": q4k.QK_K, "bits": 6 if Q6K else 4, "mode": MODE
        }
    (OUT / "config.json").write_text(json.dumps(acfg, indent=1))

    print(f"\n{done} modules in {time.time() - t0:.0f}s"
          + (f", {with_h} of them with diag(H)" if IMATRIX else ""))
    print(f"those modules: {bytes_before / 1e9:.4f} GB -> {bytes_after / 1e9:.4f} GB "
          f"({100 * (bytes_after / bytes_before - 1):+.1f}%)")
    print(f"wrote {OUT}\n")
    print("Next -- and note mlx_lm cannot load this, so the port is the instrument:")
    print(f"  python3 tools/check_rebuild.py {OUT}")
    print(f"  ORNITH_MODEL={OUT} ORNITH_PROBES_OUT=refs/artifact/{OUT.name.split('a3b-')[-1]} \\")
    print("    cargo run --release --bin probes")


if __name__ == "__main__":
    main()
