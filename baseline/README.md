# Baseline of the artifact this port was built against

`REBUILD.md` Phase 0. **None of this can be recreated once the artifact is
deleted**, which is why it lives here and not in `refs/` -- everything in
`refs/` is regenerable from a `tools/dump_*.py`, and nothing here is.

The artifact: `Shiftedx/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx`,
forged from `ornith-ai/Ornith-1.5-35B-A3B@10fbf86f` by the recipe in
`BUILD_RECIPE.json`.

**Why it exists.** Every test in this port compares against MLX running on the
same artifact, which validates the port and not the artifact. A subtly wrong
rebuild would have MLX and this port agreeing on the wrong thing, with the
whole suite green. These are the measurements that would notice.

| | |
|---|---|
| `BUILD_RECIPE.json` | the recipe, and the **input** to a rebuild -- `mtplx forge build --recipe "$(cat ...)"` |
| `.mtplx-source.json` | provenance, including a **per-file sha256** of every shard, which is what makes a byte-identical rebuild checkable |
| `conversion_receipt.json` | the parent and the revision actually converted from |
| `SHA256SUMS`, `RELEASE_MANIFEST.json` | the published manifests |
| `config.json` and the rest | the artifact's configuration, including the per-module quantization block |
| `ppl_corpus.txt` | the text the perplexity below is over. A perplexity without its corpus is not a measurement |
| `baseline.json` | the numbers, below |
| `logprobs.bin` | float32 `[8, 248320]`, the full next-token distribution for the 8 contexts named in `baseline.json` |

## The numbers

| | |
|---|---|
| perplexity | **6.3327** over 6620 tokens, mean NLL 1.845725 |
| logprobs | 8 contexts, full distribution, float32 |
| anchors | 4 prompts, 96 greedy tokens each, as ids and text |
| MTP acceptance | 4 prompts at depth 1 and depth 3 |
| the port | commit, decode and prefill figures at the time |

## How to use it against a rebuild

- **Perplexity** is the headline: rerun `tools/dump_artifact_baseline.py`
  against the new artifact over the same corpus. A requantization that holds
  quality lands close; one that does not, does not.
- **`logprobs.bin` is the sensitive one.** Perplexity averages over thousands
  of tokens and hides a lot. A KL divergence between these distributions and
  the rebuild's, on the same 8 contexts, says how far the model itself moved.
- **Anchors will not match** and are not meant to. Different weights give
  different tokens eventually; what matters is that a rebuild diverges late
  and stays coherent, not that it is identical.
- **MTP acceptance** shifts with requantization because draft and target move
  together. Without a before, a format regression is indistinguishable from
  noise.
