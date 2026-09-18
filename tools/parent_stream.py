"""Streamed access to the bf16 parent: one layer resident at a time.

`REBUILD.md` Phase 8. `mlx_lm.load` dies on this checkpoint -- 67.0 GiB of
weights against a 51.8 GiB working set -- but that is about **residency, not
about the forward**. A layer only has to be resident while it runs, so the
loop is inverted: push every input through layer *i*, drop it, load *i+1*.

    heaviest layer of 40   3.13 GiB      active between layers   0.00 GiB
    lm_head / embed        0.95 GiB      measured peak          ~1.7 GiB

**How the release works, and it is the whole mechanism.** An `mx.array` that
has been evaluated stays evaluated, so a layer is freed by overwriting its
parameters with **freshly loaded, unevaluated** ones. `mx.load` is lazy, so
calling it again is cheap -- but its result must never be cached, because the
arrays in a cached dict are the ones that get evaluated.

**This file exists so that `sanitize` has one implementation.** The parent is
an HF checkpoint carrying `mtp.*` and a torch-layout `conv1d`, so
`qwen3_5.TextModel.sanitize` **adds 1.0 to the trunk norm gains** -- true of a
parent, false of a converted artifact. Streaming sanitizes per layer instead of
once over the whole dict, so `has_unsanitized_conv1d` has to be decided
globally, from the full key list, before any layer is touched. Two callers
copying that reasoning is how the double shift comes back.
"""

import json

import mlx.core as mx

NORM_KEYS = (".input_layernorm.weight", ".post_attention_layernorm.weight",
             "model.norm.weight", ".q_norm.weight", ".k_norm.weight")


def gib(n):
    return n / 2**30


def mem():
    return (f"active {gib(mx.get_active_memory()):5.2f} GiB, "
            f"peak {gib(mx.get_peak_memory()):5.2f}")


class Parent:
    """Lazy, per-tensor access to the parent, with `sanitize` applied by hand.

    Every accessor returns **fresh** lazy arrays; that is what makes a layer
    releasable. See the module docstring.
    """

    def __init__(self, path):
        self.path = path
        idx = json.loads((path / "model.safetensors.index.json").read_text())
        self.map = idx["weight_map"]
        self._by_prefix = {}
        # The global decision, taken once over the full key list. Reading a
        # header is not reading a tensor.
        self.shift_norms = False
        for k, shard in self.map.items():
            if "conv1d.weight" in k:
                self.shift_norms = mx.load(str(path / shard))[k].shape[-1] != 1
                break
        assert self.shift_norms, (
            "parent conv1d is already in [D, K, 1] layout, so sanitize would not "
            "shift the norm gains -- this is not the HF parent this file expects")

    def _rename(self, key):
        if key.startswith("model.language_model"):
            return key.replace("model.language_model", "language_model.model")
        if key.startswith("language_model."):
            return key
        return "language_model." + key

    def keys_under(self, prefix):
        if prefix not in self._by_prefix:
            self._by_prefix[prefix] = [
                k for k in self.map
                if k.startswith(prefix)
                and not k.startswith(("vision_tower", "model.visual"))
                and "mtp." not in k]
        return self._by_prefix[prefix]

    def raw(self, key):
        return mx.load(str(self.path / self.map[key]))[key]

    def get(self, prefix):
        """Sanitized {mlx key: lazy array} for every parent key under `prefix`."""
        out = {}
        for key in self.keys_under(prefix):
            name, v = self._rename(key), self.raw(key)
            if "conv1d.weight" in name and v.shape[-1] != 1:
                v = v.moveaxis(2, 1)
            elif self.shift_norms and any(name.endswith(s) for s in NORM_KEYS):
                if v.ndim == 1:
                    v = v + 1.0
            out[name] = v
        return out

    def layer(self, i):
        """One decoder layer's parameters, relative to the layer module."""
        pre = f"model.language_model.layers.{i}."
        w = self.get(pre)
        mlx_pre = f"language_model.model.layers.{i}."
        rel = {}
        for k, v in w.items():
            r = k[len(mlx_pre):]
            # The MoE experts arrive fused; `sanitize` splits gate_up in two.
            if r == "mlp.experts.gate_up_proj":
                mid = v.shape[-2] // 2
                rel["mlp.switch_mlp.gate_proj.weight"] = v[..., :mid, :]
                rel["mlp.switch_mlp.up_proj.weight"] = v[..., mid:, :]
            elif r == "mlp.experts.down_proj":
                rel["mlp.switch_mlp.down_proj.weight"] = v
            else:
                rel[r] = v
        return rel

    def check_norm_shift(self):
        """The double-shift trap, checked on the result rather than the intent.

        A delta-encoded gain sits near 0 and an absolute one near 1. Missing
        the +1.0 gives a model that loads and is wrong, which is the whole
        failure mode -- and both quantities being wrong by the same 1.0 is why
        comparing the parent's own tensors cannot catch it.
        """
        g = self.get("model.language_model.layers.0.")[
            "language_model.model.layers.0.input_layernorm.weight"]
        mx.eval(g)
        m = float(mx.abs(g).mean())
        assert 0.5 < m < 2.0, (
            f"layer 0 input_layernorm mean |gain| is {m:.4f}; shifted gains sit "
            "near 1.0, delta-encoded ones near 0.0")
        mx.clear_cache()
        return m
