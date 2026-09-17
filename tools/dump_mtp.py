"""Reference for the MTP head: `mtp_forward` and the depth-3 draft chain.

The head is bound the way the runtime binds it -- MTPLX's generic
`inject_mtp_support` with a default `MTPContract`, which is what
`mtplx_runtime.json`'s `mtp_contract` block spells out:

    base_hidden_variant  post_norm      what the trunk feeds in
    hidden_variant       post_norm      what the head feeds forward
    concat_order         embedding_hidden
    mtp_position_mode    cache          no explicit position offset
    mtp_prequantized     false          the sidecar is BF16 throughout

Note this artifact's `model_type` is `qwen3_5_moe`, not `qwen3_5_mtp`, so
`is_qwen3_5_mtp_config` is False and the qwen3_5-specific patch -- which
hardcodes `pre_norm` -- never runs.

Dumped, narrow to wide:

    trunk_hidden     the trunk's post-norm hidden for the prompt
    mtp_fc           the head's input after pre_fc_norm + fc
    mtp_logits/hidden  one `mtp_forward` at T = 1 and again at T = 4
    moe_*            the unquantized 256-expert block on its own
    draft_*          a depth-3 greedy draft chain
    verify_*         one trunk forward over the drafted window, with the
                     per-row gated-delta states a rollback needs

Run it as

    cd tools && PYTHONPATH=../../vendor/mlx-lm:../../vendor/MTPLX \\
        python3 dump_mtp.py
"""

import json
import pathlib

import mlx.core as mx

import ref

DEPTH = 3
PROMPT = "The capital of France is Paris, and the capital of Japan is"


def main():
    from mlx_lm.utils import load
    from mtplx.mtp_patch import MTPContract, inject_mtp_support, validate_mtp_support

    path = pathlib.Path(ref.MODEL)
    config = json.loads((path / "config.json").read_text())
    model, tokenizer = load(path)
    contract = MTPContract()
    assert inject_mtp_support(model, path, config, contract)
    assert validate_mtp_support(model)
    lm = model.language_model

    case = ref.Case("mtp")
    case.note(depth=DEPTH, **contract.to_dict())

    ids = mx.array(tokenizer.encode(PROMPT, add_special_tokens=False))
    case.put("ids", ids.astype(mx.uint32))

    cache = lm.make_cache()
    logits, hidden = lm(ids[None], cache=cache, return_hidden=True)
    case.put("trunk_logits", logits)
    case.put("trunk_hidden", hidden)
    primary = mx.argmax(logits[:, -1, :], axis=-1)
    case.put("primary", primary.astype(mx.uint32))

    # --- one mtp_forward, T = 1, off a fresh cache
    mtp_cache = lm.make_mtp_cache()
    d_logits, d_hidden = lm.mtp_forward(
        hidden[:, -1:, :],
        primary.reshape(1, 1),
        mtp_cache=mtp_cache,
        return_hidden=True,
        mtp_hidden_variant=contract.hidden_variant,
    )
    case.put("mtp1_logits", d_logits)
    case.put("mtp1_hidden", d_hidden)
    case.put("mtp1_token", mx.argmax(d_logits[:, -1, :], axis=-1).astype(mx.uint32))

    # The head's input after pre_fc_norm + fc, so a mismatch upstream of the
    # decoder block is told apart from one inside it.
    e = lm.mtp.pre_fc_norm_embedding(lm.model.embed_tokens(primary.reshape(1, 1)))
    h = lm.mtp.pre_fc_norm_hidden(hidden[:, -1:, :])
    case.put("mtp_e", e)
    case.put("mtp_h", h)
    case.put("mtp_fc", lm.mtp.fc(mx.concatenate([e, h], axis=-1)))

    # --- the unquantized MoE block on its own, at T = 1 and T = 4
    block = lm.mtp.layers[0].mlp
    for t in (1, 4):
        mx.random.seed(17 + t)
        xin = (mx.random.normal((1, t, config["text_config"]["hidden_size"])) * 0.5).astype(
            mx.bfloat16
        )
        case.put(f"moe{t}_in", xin)
        case.put(f"moe{t}_out", block(xin))

    # --- a depth-3 greedy draft chain, the shape `chain_fn` runs
    mtp_cache = lm.make_mtp_cache()
    dh, dtok = hidden[:, -1:, :], primary.reshape(1, 1)
    drafts = []
    for level in range(DEPTH):
        dl, dhid = lm.mtp_forward(
            dh,
            dtok,
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=contract.hidden_variant,
        )
        case.put(f"draft{level}_logits", dl)
        case.put(f"draft{level}_hidden", dhid)
        dtok = mx.argmax(dl[:, -1, :], axis=-1).reshape(1, 1)
        drafts.append(int(dtok))
        dh = dhid[:, -1:, :]
    case.put("draft_tokens", mx.array(drafts, dtype=mx.uint32))
    print("primary", int(primary), "drafts", drafts, flush=True)

    # --- verify: one trunk forward over [primary] + drafts, which is what the
    #     speculative loop feeds, plus the per-row recurrent state a partial
    #     acceptance rolls back to.
    window = mx.array([int(primary)] + drafts, dtype=mx.uint32)
    case.put("verify_window", window)
    v_logits, v_hidden = lm(window[None], cache=cache, return_hidden=True)
    case.put("verify_logits", v_logits)
    case.put("verify_hidden", v_hidden)
    case.put("verify_targets", mx.argmax(v_logits[0], axis=-1).astype(mx.uint32))

    # --- the same window fed one token at a time, which is what autoregression
    #     would have done.
    #
    # These do NOT agree with the rows above, and the gap is mlx's own: a
    # T-token forward and T single-token forwards pick different matmul
    # kernels (M = 4 takes the split-K gemm where M = 1 takes qmv), with
    # different reduction orders, and 40 layers of bfloat16 residual turn that
    # into O(1) on a hidden whose scale is ~35. Dumping both lets the Rust side
    # show it reproduces mlx on each path, so the divergence between
    # speculative and autoregressive decoding is a property of the stack rather
    # than of the port.
    step_cache = lm.make_cache()
    lm(ids[None], cache=step_cache, return_hidden=True)
    step_logits, step_hidden = [], []
    for tok in [int(t) for t in window]:
        sl, sh = lm(mx.array([tok])[None], cache=step_cache, return_hidden=True)
        step_logits.append(sl)
        step_hidden.append(sh)
    case.put("step_logits", mx.concatenate(step_logits, axis=1))
    case.put("step_hidden", mx.concatenate(step_hidden, axis=1))

    case.write()


if __name__ == "__main__":
    main()
