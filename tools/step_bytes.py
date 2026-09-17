"""Weight bytes one decode step reads, derived from an artifact on disk.

    python3 tools/step_bytes.py <artifact dir> [<artifact dir> ...]

`REBUILD.md` Phase 4 judges a variant on three numbers -- perplexity, KL, and
then bytes and GPU busy. The first two have tools (`dump_artifact_baseline.py`
and `compare_baselines.py`); the third was arithmetic done by hand from the
module inventory, which is fine until a format changes the tensor count and
the arithmetic has to be redone. This reads the safetensors headers instead,
so the number is a property of the artifact rather than of a spreadsheet, and
it does not care how many tensors a module has -- which is the thing nvfp4
changes.

**What a decode step reads, and what it does not.** Every weight in the trunk,
once, with two exceptions:

  * the MoE experts. 256 per layer, `num_experts_per_tok` of them per token,
    so `switch_mlp.*` counts at k/256. This is the one place the inventory is
    not the traffic, and it is 7.6% of a step in the control.
  * `embed_tokens`, which is one row of a 248320-row table. Counted as that
    one row; it rounds to 0.0% either way.

The vision tower and the MTP sidecar are excluded: the tower runs once per
image and not per token, and the sidecar only runs under `--mtp`. Norms,
`conv1d`, `A_log` and `dt_bias` are included -- they are small but they are
read.

This is the *weight* traffic, which at batch 1 is what a step is bound by. It
is not the whole of it: activations, the KV cache for the 10 full-attention
layers and the gated-delta states are on top, and at short context they are
the small part (`HANDOFF.md`: 20 KB a token of cache against 2.83 GB of
weights).
"""

import json
import pathlib
import struct
import sys

DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1,
}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def tensors(art):
    """Every trunk tensor as `name -> bytes`, skipping vision and MTP."""
    index = json.loads((art / "model.safetensors.index.json").read_text())
    out = {}
    for shard in sorted(set(index["weight_map"].values())):
        for name, meta in header(art / shard).items():
            if name == "__metadata__":
                continue
            low = name.lower()
            if "vision" in low or "visual" in low or low.startswith("mtp."):
                continue
            n = 1
            for d in meta["shape"]:
                n *= d
            out[name] = n * DTYPE_BYTES[meta["dtype"]]
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = []
    for arg in sys.argv[1:]:
        art = pathlib.Path(arg).expanduser()
        cfg = json.loads((art / "config.json").read_text())
        text = cfg.get("text_config", cfg)
        experts = text["num_experts"]
        top_k = text["num_experts_per_tok"]

        by = tensors(art)
        total = 0
        groups = {}
        for name, nbytes in by.items():
            if ".switch_mlp." in name:
                # top_k of `experts` slabs, and the tensor holds all of them.
                nbytes = nbytes * top_k // experts
                key = "switch_mlp (top-k)"
            elif name.endswith(("embed_tokens.weight", "embed_tokens.scales",
                                "embed_tokens.biases")):
                # One row of the table, not the table.
                nbytes = nbytes // text["vocab_size"]
                key = "embed_tokens (1 row)"
            elif "lm_head" in name:
                key = "lm_head"
            elif "linear_attn" in name:
                key = "linear_attn"
            elif "self_attn" in name:
                key = "self_attn"
            elif "shared_expert" in name:
                key = "shared_expert"
            elif name.endswith(("mlp.gate.weight", "mlp.gate.scales", "mlp.gate.biases")):
                key = "mlp.gate (router)"
            else:
                key = "other (norms, conv1d, A_log, dt_bias)"
            groups[key] = groups.get(key, 0) + nbytes
            total += nbytes

        q = cfg.get("quantization") or {}
        rows.append((art.name, q.get("mode", "affine"), q.get("bits"),
                     q.get("group_size"), total, groups))

    base = rows[0][4]
    for name, mode, bits, gs, total, groups in rows:
        print(f"\n{name}")
        print(f"  body {bits}-bit g{gs} {mode}")
        for k in sorted(groups, key=lambda k: -groups[k]):
            print(f"    {k:<38}{groups[k] / 1e9:8.4f} GB  {100 * groups[k] / total:5.1f}%")
        delta = "" if total == base else f"  ({100 * (total / base - 1):+.1f}% vs {rows[0][0].split('-')[-1]})"
        print(f"    {'TOTAL':<38}{total / 1e9:8.4f} GB{delta}")


main()
