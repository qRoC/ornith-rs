"""Reference outputs for the sparse MoE block, on layer 0's real weights.

Covers both of mlx's gather paths: with 1 token the router produces 8
(token, expert) pairs, below `SwitchGLU`'s sort threshold of 64, so it takes
`gather_qmv`; with 512 tokens there are 4096 pairs over 256 experts, which
satisfies `GatherQMM::eval_gpu`'s `M == 1 && B >= 16 && sorted && B/E >= 4`
and takes `gather_qmm_rhs`.
"""

import mlx.core as mx
import mlx.nn as nn

import ref
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU, _gather_sort

L = 0
P = f"language_model.model.layers.{L}.mlp"


def quant_linear(t, base, gs, bits, dims):
    lin = nn.QuantizedLinear(dims[1], dims[0], bias=False, group_size=gs, bits=bits)
    lin.weight, lin.scales, lin.biases = (
        t[f"{base}.weight"], t[f"{base}.scales"], t[f"{base}.biases"]
    )
    lin.group_size, lin.bits = gs, bits
    return lin


def main():
    cfg = ref.config()["text_config"]
    k = cfg["num_experts_per_tok"]
    n_experts = cfg["num_experts"]
    dim = cfg["hidden_size"]
    moe_dim = cfg["moe_intermediate_size"]
    shared_dim = cfg["shared_expert_intermediate_size"]

    names = []
    for sub in ["gate", "shared_expert_gate",
                "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
                "switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj"]:
        for suf in ["weight", "scales", "biases"]:
            names.append(f"{P}.{sub}.{suf}")
    t = ref.load_tensors(names)

    case = ref.Case("moe")
    case.note(top_k=k, n_experts=n_experts, dim=dim, moe_dim=moe_dim,
              shared_dim=shared_dim,
              norm_topk_prob=cfg.get("norm_topk_prob", True))
    mx.random.seed(4)

    # Quantization is per module: config.json overrides the 4-bit/g32 default
    # with 8-bit/g64 for the router gates everywhere, and for the experts and
    # shared experts of only 8 of the 40 layers.
    def q(path):
        return ref.quant_config(path)

    def lin(sub, dims):
        gs, bits = q(f"{P}.{sub}")
        return quant_linear(t, f"{P}.{sub}", gs, bits, dims)

    gate = lin("gate", (n_experts, dim))
    seg = lin("shared_expert_gate", (1, dim))
    sh_gate = lin("shared_expert.gate_proj", (shared_dim, dim))
    sh_up = lin("shared_expert.up_proj", (shared_dim, dim))
    sh_down = lin("shared_expert.down_proj", (dim, shared_dim))
    case.note(**{f"quant_{k2}": list(q(f"{P}.{k2}"))
                 for k2 in ["gate", "shared_expert_gate", "shared_expert.gate_proj",
                            "switch_mlp.gate_proj"]})

    switch = SwitchGLU(dim, moe_dim, n_experts)
    for name in ["gate_proj", "up_proj", "down_proj"]:
        gs, bits = q(f"{P}.switch_mlp.{name}")
        ql = QuantizedSwitchLinear(dim, moe_dim, n_experts, False, gs, bits)
        ql.weight = t[f"{P}.switch_mlp.{name}.weight"]
        ql.scales = t[f"{P}.switch_mlp.{name}.scales"]
        ql.biases = t[f"{P}.switch_mlp.{name}.biases"]
        ql.group_size, ql.bits = gs, bits
        setattr(switch, name, ql)

    for T in [1, 7, 24, 512]:
        x = (mx.random.normal((1, T, dim)) * 0.5).astype(mx.bfloat16)
        case.put(f"x_t{T}", x)

        gates = gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        case.put(f"gates_t{T}", gates)

        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        case.put(f"inds_t{T}", inds.astype(mx.uint32))
        case.put(f"scores_raw_t{T}", scores)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        case.put(f"scores_t{T}", scores)

        # mlx's own sort permutation, to check ours against
        flat_x = mx.expand_dims(x.reshape(-1, dim), (-2, -3))
        xs, idx, inv = _gather_sort(flat_x, inds.reshape(-1, k))
        case.put(f"sort_idx_t{T}", idx.astype(mx.uint32))
        case.put(f"sort_inv_t{T}", inv.astype(mx.uint32))

        y = switch(x, inds)
        case.put(f"switch_out_t{T}", y)
        y = (y * scores[..., None]).sum(axis=-2)
        case.put(f"mixed_t{T}", y)

        shared_y = sh_down(nn.silu(sh_gate(x)) * sh_up(x))
        shared_y = mx.sigmoid(seg(x)) * shared_y
        case.put(f"shared_t{T}", shared_y)
        case.put(f"out_t{T}", y + shared_y)

    case.write()


if __name__ == "__main__":
    main()
