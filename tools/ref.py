"""Golden-reference dumping: run mlx, write tensors the Rust tests read back.

Every reference lives in `refs/<case>/`: raw little-endian tensor files plus a
`manifest.json` describing them. The format is deliberately trivial so the Rust
side needs no parser beyond the one it already has for config.json.

    refs/qmm_8bit/
      manifest.json   {"tensors": {"x": {"dtype": "bfloat16", "shape": [...],
                                         "file": "x.bin"}, ...}}
      x.bin
      y_m1.bin
      ...

Run with the vendored mlx-lm on the path so the reference is the code this port
was read from:

    PYTHONPATH=../vendor/mlx-lm python3 tools/dump_<case>.py
"""

import json
import os
import sys

import mlx.core as mx

MODEL = os.environ.get(
    "ORNITH_MODEL",
    "/Users/qroc/.mtplx/models/"
    "ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx",
)
REFS = os.environ.get(
    # The Rust side has read this since the port began (`refs::refs_root`); the
    # writer did not, so regenerating references for a forged artifact meant
    # overwriting the control's. Same variable, both sides.
    "ORNITH_REFS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refs"),
)

_DTYPE_NAMES = {
    mx.bfloat16: "bfloat16",
    mx.float16: "float16",
    mx.float32: "float32",
    mx.uint32: "uint32",
    mx.int32: "int32",
    mx.uint8: "uint8",
    mx.bool_: "bool",
}


class Case:
    """One reference directory."""

    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(REFS, name)
        os.makedirs(self.dir, exist_ok=True)
        self.tensors = {}
        self.meta = {}

    def put(self, name, arr):
        mx.eval(arr)
        dtype = _DTYPE_NAMES.get(arr.dtype)
        if dtype is None:
            raise ValueError(f"{name}: unhandled dtype {arr.dtype}")
        fname = f"{name}.bin"
        # mx arrays expose the buffer protocol for every dtype except bfloat16,
        # which numpy has no equivalent for; view it as uint16 to get the bits.
        raw = arr
        if arr.dtype == mx.bfloat16:
            raw = arr.view(mx.uint16)
        with open(os.path.join(self.dir, fname), "wb") as f:
            f.write(bytes(memoryview(raw)))
        self.tensors[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "file": fname,
        }
        return arr

    def note(self, **kwargs):
        self.meta.update(kwargs)

    def put_json(self, name, obj):
        """A side file for things that are not tensors: text, token lists."""
        fname = f"{name}.json"
        with open(os.path.join(self.dir, fname), "w") as f:
            json.dump(obj, f, ensure_ascii=True, separators=(",", ":"))
        self.tensors[name] = {"dtype": "json", "shape": [], "file": fname}
        return obj

    def write(self):
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump({"tensors": self.tensors, "meta": self.meta}, f, indent=1)
        total = sum(
            os.path.getsize(os.path.join(self.dir, t["file"]))
            for t in self.tensors.values()
        )
        print(
            f"{self.name}: {len(self.tensors)} tensors, {total / 1e6:.1f} MB -> {self.dir}",
            file=sys.stderr,
        )


def load_tensors(names, shard_hint=None):
    """Load specific tensors out of the sharded model without reading it all.

    `model.safetensors.index.json` maps names to shards; only the shards that
    actually hold a requested tensor get read.
    """
    with open(os.path.join(MODEL, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard = {}
    for n in names:
        if n not in weight_map:
            raise KeyError(f"{n} is not in the index")
        by_shard.setdefault(weight_map[n], []).append(n)
    out = {}
    for shard, wanted in by_shard.items():
        loaded = mx.load(os.path.join(MODEL, shard))
        for n in wanted:
            out[n] = loaded[n]
    return out


def config():
    with open(os.path.join(MODEL, "config.json")) as f:
        return json.load(f)


def quant_config(path, cfg=None):
    """Bits/group_size for one module path, honouring per-module overrides."""
    cfg = cfg or config()
    q = cfg["quantization"]
    entry = q.get(path)
    if isinstance(entry, dict):
        return entry["group_size"], entry["bits"]
    return q["group_size"], q["bits"]
