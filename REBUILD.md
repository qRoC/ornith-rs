# Rebuilding the artifact from upstream

The artifact this port runs on was forged from a bf16 parent, and the forging
is recorded in the artifact itself. Rebuilding it is therefore a reproducible
operation rather than a reverse-engineering one, and it is what unlocks the
only large lever left: finding out whether a better 4-bit format holds quality
where affine int4 did not, which is what would let the 260 modules currently
held at 8 bits drop to 4 and take ~35% of a decode step's traffic with them.

**The one thing to understand before starting.** Every test in this port
compares against MLX running on the same artifact. That validates *the port*,
not *the artifact* -- if a rebuild is subtly wrong, MLX and this port will
agree on the wrong thing and the whole suite will pass. So the artifact needs
a different kind of check, and the material for it exists only while the
current artifact does. Phase 0 is not optional and it is not reversible.

---

## What the current artifact says about itself

From `conversion_receipt.json` and `BUILD_RECIPE.json`, both in the artifact
root:

| | |
|---|---|
| parent | `ornith-ai/Ornith-1.5-35B-A3B` |
| recipe | `attention8-bf16-recurrence`, version 1 |
| adapter | `qwen35-moe-attention8-bf16recurrence-v1` |
| body | 4-bit affine, group 32 |
| overrides | 19 rules, e.g. `linear_attn.in_proj_a` -> bf16, not quantized |
| expected | 192 modules at q4, 260 at q8, 60 float exclusions, 512 quantizable |
| MTP | `same-parent-bf16-sidecar` -- 785 tensors, kept bf16 |
| vision | `same-parent-bf16-preserve` -- 333 tensors, kept bf16 |

So the MTP head and the vision tower **come from the parent**; MTPLX preserves
them rather than adding them. Nothing in this artifact is unavailable upstream.

**The two recorded revisions disagreed; it is resolved.**

    conversion_receipt.json   source: ...A3B@10fbf86fed7ecee4a061f8b499a618f46001cac1
    BUILD_RECIPE.json         source_revision: fbb995a79eedd569a5edc5f2af9644c0fa1124fc

The HF API reports the repo's `sha` as **`10fbf86f...`**, the receipt's, which
is also current `main`. The recipe's field matches nothing in the repo -- treat
it as the recipe's own pin rather than the parent, and build from
`10fbf86f...`, which is what the receipt says the conversion actually used.

**And the parent is already verified, remotely, before downloading a byte.**
`model.safetensors.index.json` at that revision against `source_signature` in
the recipe:

| | recipe | upstream |
|---|---|---|
| indexed tensors | 1811 | 1811 |
| `mtp.*` | 785 | 785 |
| vision | 333 | 333 |
| recurrent `a` / `b` | 30 / 30 | 30 / 30 |

Exact on every count, with top-level prefixes `model` (1025), `mtp` (785) and
`lm_head` (1). **The MTP head and the vision tower are in the parent**, which
is the fact the whole rebuild rests on.

Worth knowing because it is confusing: **the model card never mentions either
of them.** It documents a reasoning MoE for coding and agentic work and stops
there. The evidence they exist is the tensor index and the three processor
configs in the repo (`preprocessor_config.json`, `processor_config.json`,
`video_preprocessor_config.json`), not the prose. A future reader who checks
the card and concludes the vision tower is an MTPLX addition will be wrong.

There is also an empty marker file, `.model_only_pruned`, at the repo root. It
carries no content and nothing explains it; note it and move on unless a
rebuild comes out short.

## The mechanism

`mtplx forge build` does the whole path -- download, convert, verify, brand --
and takes the recipe as JSON on the command line:

```
mtplx forge build \
  --repo ornith-ai/Ornith-1.5-35B-A3B \
  --model-root ~/.mtplx/models \
  --out <progress dir> --run-id <id> \
  --recipe "$(cat BUILD_RECIPE.json)" \
  --branded-name <name>
```

(`mtplx/cli.py:3269`. `--dtype` overrides the recipe's `body_dtype`;
`--allow-degraded-mtp` is the guard against recipes that requantize the MTP
head, which this one does not.)

Inside, `body_bits`, `body_group_size` and `body_mode` become
`mlx_lm convert --q-bits/--q-group-size/--q-mode`
(`mtplx/commands/forge.py:1448`), and mlx accepts exactly four modes
(`mlx/python/mlx/nn/layers/quantized.py:11`):

| mode | default group | default bits |
|---|---|---|
| `affine` | 64 | 4 |
| `mxfp4` | 32 | 4 |
| `nvfp4` | 16 | 4 |
| `mxfp8` | 32 | 8 |

**So the experiment this whole rebuild exists for is one word in the recipe.**
`module_overrides` additionally require a bf16/native source
(`forge.py:1404`), which is exactly why it could not be done on the artifact
we have.

---

## What the parent's model card tells us, and what it does not

Verbatim from the card (MIT licence, ~70 GB in bf16, 262144 native context):

- **It is a reasoning model**: "by default the assistant turn opens with a
  `<think> … </think>` block before the final answer." This port already
  renders that -- the reasoning-content split is in `chat.rs`.
- **Recommended sampling is `temperature=0.6, top_p=0.95, top_k=20`**, which
  is exactly what `main.rs` defaults to. The card adds that its published
  benchmarks were reproduced at `temperature=1.0`.
- **On YaRN**: "Open-source runtimes implement YaRN *statically*: the same
  scaling factor is applied to every request regardless of its length, which
  can slightly hurt quality on ordinary-length inputs. Only enable
  `rope_scaling` when your workload genuinely needs the longer window." This
  port implements no rope scaling and the artifact's `rope_parameters.type` is
  `"default"`, so it is on the right side of that note by construction -- and
  a rebuild must not turn scaling on to "match upstream".
- Runtime floors are given as Transformers >= 5.8.1, vLLM >= 0.19.1,
  SGLang >= 0.5.9. Only the first matters here, and only for reference
  generation: see the tokenizer section in `HANDOFF.md`, where transformers 5.x
  ignoring `tokenizer.json`'s pre-tokenizer is already a known divergence.

What the card does **not** contain: anything about multi-token prediction,
about the vision tower, or about quantization. No guidance exists upstream on
which modules tolerate 4 bits -- which is precisely the question Phase 4 sets
out to answer experimentally, because there is nobody to ask.

## Phase 0 -- before anything is deleted -- **DONE**

Captured and committed under `baseline/`; see `baseline/README.md`. It is out
of `refs/` on purpose: everything in `refs/` regenerates from a
`tools/dump_*.py`, and nothing here does.

| | |
|---|---|
| metadata | 13 files, including `BUILD_RECIPE.json` (the rebuild's input) and `.mtplx-source.json` (per-file sha256) |
| perplexity | **6.3327** over 6620 tokens of `baseline/ppl_corpus.txt`, mean NLL 1.845725 |
| logprobs | float32 `[8, 248320]`, full next-token distributions for 8 fixed contexts |
| anchors | 4 prompts x 96 greedy tokens, ids and text |
| MTP acceptance | 4 prompts at depth 1 and 3 |

`tools/dump_artifact_baseline.py` is the writer, and it is the one writer in
this repository that **cannot be re-run** -- its input is the artifact about
to be deleted. Point it at the rebuild to get the other side of the
comparison.

The original checklist, for the record:

1. **Copy the artifact's own metadata out.** These are the *inputs* to the
   rebuild, not just records:

       BUILD_RECIPE.json  conversion_receipt.json  .mtplx-source.json
       config.json  generation_config.json  chat_template.jinja
       SHA256SUMS  RELEASE_MANIFEST.json

   Put them in `refs/artifact/old/` (outside git, like the rest of `refs/`).
   `.mtplx-source.json` carries a **per-file sha256 of every shard**, which is
   what makes Phase 2 checkable.

2. **Capture the quality baseline.** This is the number the whole exercise is
   judged against, and it cannot be recomputed later. Perplexity over a fixed
   held-out text, via the vendored mlx-lm, on the current artifact. Keep the
   text file alongside the number -- a perplexity without its corpus is not a
   measurement.

3. **Capture behavioural anchors.** Greedy continuations for the prompt set in
   `tests/generate.rs`, as token ids, from *this* artifact. They will **not**
   match a requantized rebuild -- different weights, different tokens -- and
   that is fine. They are there to answer "is the new one still the same
   model" by eye and by agreement length, which is the only cross-artifact
   check available.

4. **Capture the MTP acceptance profile.** `--mtp --mtp-depth 3` over the same
   prompts: cycles, accepted, acceptance by depth. Requantizing changes the
   draft/target agreement, and without a before we cannot tell a format
   regression from ordinary noise.

5. **Record the port's current numbers** -- already done, they are the
   headline table in `HANDOFF.md`. Note the commit.

Only then delete.

## Phase 1 -- the parent

1. Resolve the revision question above. `huggingface-cli` or the HF API on
   both shas; if only one exists, that settles it.
2. Pull it with **MTPLX's own pull, not `huggingface-cli`**:

       mtplx pull ornith-ai/Ornith-1.5-35B-A3B \
         --revision 10fbf86fed7ecee4a061f8b499a618f46001cac1 \
         --download-backend auto

   ~70 GB. Three reasons it has to be this and not the HF CLI:

   - **Destination.** `pull_model` writes to
     `<model root>/ornith-ai--Ornith-1.5-35B-A3B` (`hf_loader.py:1617`,
     `safe_model_name` turning `/` into `--`, the same shape as the artifact
     directory we have). `huggingface-cli download` without `--local-dir` goes
     to `~/.cache/huggingface/hub`, where forge will not find it and will pull
     its own copy -- 140 GB against 84 free.
   - **The provenance marker.** Every successful pull writes
     `.mtplx-source.json` with a per-file sha256. That marker is what makes
     Phase 2's byte-identical check possible at all, and forge uses it
     (`_source_marker_matches`) to recognise an already-downloaded source
     instead of re-fetching. The HF CLI leaves none.
   - **The revision.** `main` is `10fbf86f` today; pinning costs nothing and
     protects against the repo moving mid-run.

   No `--include`: everything but two PNG assets is needed, the 70 GB is the
   weights themselves, and a filter that misses a config fails late.
   `--download-backend auto` picks up `aria2c` when it is installed, which is
   worth having at this size; the default is the built-in Python downloader.
3. Re-run the inventory check locally after the pull -- the remote one above
   says it should pass, so a failure here means the download is incomplete
   rather than the revision being wrong.

## Phase 2 -- reproduce what we had, before changing anything

**Start here in a new session.** Phase 0 is captured in `baseline/`, the old
artifact is deleted, and the parent is pulled.

Build with the saved recipe **unmodified**:

    cd ~/Projects/ai/ornith-rs
    mtplx forge build \
      --repo ornith-ai/Ornith-1.5-35B-A3B \
      --model-root ~/.mtplx/models \
      --out /tmp/forge --run-id phase2 \
      --recipe "$(cat baseline/BUILD_RECIPE.json)" \
      --branded-name ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx

One thing to watch: `BUILD_RECIPE.json` is the artifact's *record* of its
recipe, so it carries keys that are history rather than input --
`artifact_name`, `source_model`, `source_revision`, `release_repo`,
`architecture_signature`, `source_signature`. `_read_recipe` only parses JSON
and the builder reads the keys it knows (`forge.py:1448` and around), so the
extras should be inert; if forge objects, strip them to the ones the recipe
actually sets: `body_bits`, `body_group_size`, `body_mode`, `body_dtype`,
`module_overrides`, `mtp_policy`, `vision_policy`, `recipe_id`,
`recipe_version`, `adapter_id`, and the `expected_*` counts.

Then check it:

    python3 tools/check_rebuild.py ~/.mtplx/models/<branded name>

That script encodes every condition found while writing this plan, split into
INVARIANT (must hold for any rebuild, including the two the golden tests are
blind to) and RECIPE (must hold in Phase 2, expected to break in Phase 4).
Its inventory half was verified against the parent once the pull landed: all
six counts pass (1811 / 785 / 333 / 30 / 30 / 30), so the download is complete
and that half of the script works. The rest has still never run against a real
rebuilt artifact, so treat a first failure there as possibly the script's.

Pointing it at the **parent** is a category error and it now refuses, because
everything else fails for the right reasons and reads like a disaster: the
parent's `conv1d.weight` is still the torch `[8192, 1, 4]` that conversion
transposes into `[D, K, 1]`, and its config has no `rope_parameters` at all
until mlx writes one. Both differences are conversion doing its job -- and
they confirm from the other side that the sanitisation this port depends on
happens during the forge, which is exactly why it has to be re-checked on the
output.

Compare per-file sha256 against `.mtplx-source.json`.

- **Byte-identical:** the mechanism is proven, the parent is right, and every
  later variant differs only by what the recipe says. This is the outcome to
  hope for and it is worth a day to get.
- **Not identical but structurally equal** (same tensor names, shapes, dtypes,
  same 192/260/60 module split): usable, but the difference has to be
  explained before going on -- an engine version, a dtype default, a scrub
  step. `engine_version` in `.mtplx-source.json` was 2.11.2.
- **Structurally different:** stop. Everything downstream is measuring a
  different model.

Then re-anchor the port against this rebuild, which is Phase 3. Do that before
any experiment, so that a later failure is attributable to the format and not
to the rebuild.

## Phase 3 -- re-anchoring the port

1. **Regenerate every golden reference.** They are tied to the artifact's exact
   weights. All the writers are in `tools/` and `HANDOFF.md`'s working notes
   list the commands; `refs/` is ~510 MB.
2. **Run the whole suite**, including the `--ignored` model, generate, mtp and
   vision cases. The contract does not move: `max_abs == 0.0`.
3. **Check the two artifact properties this port depends on**, which no test
   covers because they were true of the artifact rather than of the code:
   - `conv1d.weight` must be `[D, K, 1]` -- already sanitized, so mlx-lm's
     `TextModel.sanitize` does **not** add 1.0 to the norm weights. If a
     rebuild produces `[D, 1, K]` or similar, the port takes the "double
     shift" `qwen3_5_mtp_patch.py` warns about, and *the golden tests will not
     catch it*, because MLX would shift the same way.
   - `rope_parameters.type` must still be `"default"`, or the text path stops
     being plain rotary.
4. **Re-run the Phase 0 anchors** and compare: perplexity within noise of the
   baseline, continuations diverging late rather than immediately, MTP
   acceptance in the same band.

## Why any of this -- the arithmetic the whole rebuild rests on

A decode step reads **2.83 GB of weights** and nothing else is close. Over the
session that measured this port, every other lever was tried and priced:

| lever | verdict |
|---|---|
| barriers, epochs, encode order | ~7% total, half of it taken (MoE interleave), the rest needs a deferred op layer |
| pipelining the host round trip | measured *worse*; the 128-dispatch commit threshold already does what `async_eval` does |
| a better small-M matmul (`qmm_mma4`) | not applicable -- this port's kernels already amortize where it would pay |
| handing the CPU a share of the rows | physics favourable (516-543 GB/s together vs 417-425 GPU alone), bit-exactness achievable, **shut by plumbing**: a split costs a command-buffer boundary at ~60 us and only `lm_head` is big enough to pay for one |
| speculative decoding | done, and FR-Spec took 14% of the GPU work -- but it will not convert to wall clock until the stall is understood |

**Bytes are what is left.** And the split of those 2.83 GB, read off the
per-kernel table in `HANDOFF.md` (the big kernels take time in proportion to
the bytes they read):

| | share of traffic |
|---|---|
| modules at 8-bit g64 -- 260 of them | **~73%** |
| modules at 4-bit g32 | ~27% |

`linear_attn.in_proj` alone is 1.07 GB of the 2.83 and `lm_head` another 0.54,
both 8-bit. So the prize is not the 4-bit body; it is the 8-bit 73%.

Per-weight costs, from mlx's own formats:

| format | bits/weight |
|---|---|
| affine int4 g32 (the body today) | 5.00 |
| **mxfp4** | **4.25** |
| nvfp4 g16 | 4.50 |
| affine int8 g64 (the 260 today) | 8.50 |
| mxfp8 | 8.25 |

Which prices the two questions:

- Re-cutting only the 4-bit body to mxfp4 moves 27% of traffic by 15% --
  **~4%**. Barely worth the kernels on its own.
- Re-cutting the 260 8-bit modules to a 4-bit format halves 73% of the traffic
  -- **~36% fewer bytes a token**. That is the experiment.

**How much of 36% becomes speed is not 36%.** `HANDOFF.md` is explicit that
bytes convert to time only for reads already at the bandwidth roof: `lm_head`
sustains 83% of peak and would convert nearly fully, while `in_proj_qkv` sits
at 24% and is limited by its own ramp, so halving it buys much less than half
its time. The honest band is 15-36%, and the top of it would put a step at
~6.6 ms and decode near 150 tok/s against today's 92.

**Why it was not simply done before:** the 260 modules are at 8 bits because
whoever wrote the recipe found 4-bit affine insufficient there, and the
artifact we had was already quantized -- there were no bf16 weights to re-cut.
That is the entire reason for this rebuild.

## Phase 4 -- the experiments, in order

One variant at a time; see the disk note below. For each: build, regenerate
refs, run the suite, then measure perplexity, bytes a token, and GPU busy a
token.

**Not all 260 modules, and probably not mxfp4.** Two things, worked out from
`baseline/config.json` and the architecture, reshape the obvious plan.

*Where the bytes are.* The 260 are not uniform. Per token, at 8.5 bits a
weight:

| family | count | MB/token | of all traffic |
|---|---|---|---|
| `lm_head` | 1 | 540 | 19.1% |
| `linear_attn.in_proj_qkv` | 30 | 535 | 18.9% |
| `linear_attn.in_proj_z` | 30 | 267 | 9.4% |
| `linear_attn.out_proj` | 30 | 267 | 9.4% |
| `switch_mlp.*`, top-8 read | 8 | 214 | 7.6% |
| `self_attn.q_proj` + `o_proj` | 10 | 178 | 6.2% |
| `shared_expert.*` | 8 | 27 | 0.9% |
| **`mlp.gate` (the router)** | **40** | **22** | **0.8%** |
| `self_attn.k_proj` + `v_proj` | 10 | 22 | 0.8% |
| `shared_expert_gate`, `embed_tokens` | 41 | ~0 | 0.0% |

Four families are **57% of everything a decode step reads**. The 80 gate
modules -- the router and the shared-expert gate, which decide *which* experts
run and are the most quality-critical thing in a MoE -- are 0.8%. Requantizing
them is all risk and no reward, and the first version of this plan would have
done it, because it moved all 260 at once.

*What each format actually costs in precision.* `tools/quant_error.py`
quantizes a million draws and measures the relative RMS weight error. Scales
and biases are rounded to bfloat16 as the artifact stores them, so affine is
not flattered:

| format | bits/w | gaussian | laplace | student-t(4) |
|---|---|---|---|---|
| affine int8 g64 -- **today** | 8.50 | **0.54%** | 0.67% | 0.70% |
| affine int4 g32 -- the body today | 5.00 | 7.82% | 9.18% | 9.37% |
| nvfp4 g16 | 4.50 | 9.51% | 9.28% | 9.18% |
| mxfp4 g32 | 4.25 | 20.32% | 21.19% | 22.00% |

Three things fall out, and they reorder everything.

**mxfp4 is not a contender.** 20-22% against affine int4's 8-9% -- 2.3x the
error for 0.75 fewer bits. The power-of-two scale is that expensive. Drop it
from the ladder rather than demote it.

**nvfp4 is not better than affine int4, it is equal.** Slightly worse on
gaussian, slightly better on heavy tails, 0.5 bits cheaper. So it buys bytes,
not quality -- and that matters, because it means the experiment is not "does
a better format succeed where int4 failed".

**The first experiment needs no new kernels at all.** If nvfp4 is merely equal
to affine int4, then the question -- do these modules tolerate 4 bits? -- can
be asked with the format this port already runs and already verifies. Moving
the four families from 8-bit g64 to the body's own 4-bit g32 is 8.50 -> 5.00
bits over 57% of traffic: **-23.5% of all bytes**, against nvfp4's -26.8%.
Three points for a whole `fp_quantized` port, and only worth paying once 4
bits is known to work at all.

**And it names what is being asked honestly:** those 260 modules would go from
~0.6% weight error to ~8-9%, which is what the other 192 already live with.
Not "a better format", but "the same precision the rest of the model already
has".

*Why mxfp4 looked like the first choice, and was not.* Against affine int4 g32 it trades a
bf16 scale **and a zero-point** for a power-of-two-only E8M0 scale and **no
zero-point**, to save 0.75 bits. A power-of-two scale wastes up to 2x of a
group's dynamic range and the missing zero-point hurts asymmetric
distributions; against that it offers a non-uniform grid, which does suit
weights. The expected net for weights is **slightly worse than affine int4**,
which is why nvfp4 exists. So asking mxfp4 to succeed where affine int4
already failed -- which is why these 260 are at 8 bits -- is betting against
the odds.

**nvfp4 costs almost nothing over mxfp4 here.** On those four families:
8.5 bits -> 4.50 saves 26.8% of all traffic, 8.5 -> 4.25 saves 28.4%. One and
a half points apart, for a real E4M3 scale instead of a power of two and a
group of 16 instead of 32. Start there.

| # | change | what it answers | bytes | new kernels |
|---|---|---|---|---|
| A | none (Phase 2) | control -- reproduces what we had | -- | no |
| **B** | the four byte-heavy families from 8-bit g64 to the body's **affine int4 g32** | **the whole question**: do these modules tolerate 4 bits? | -23.5% | **no** |
| C | B, then the same four to **nvfp4 g16** | three more points of bytes at equal error, once 4 bits is known to work | -26.8% | yes, `fp_quantized` |
| D | extend to `switch_mlp` and `q/o_proj` | another 13.8% of traffic | -34% | -- |
| -- | the 80 gate modules, `embed_tokens` | **never**: 0.8% of bytes, the highest sensitivity in the model | -- | -- |
| -- | mxfp4, anywhere | **dropped**: 20-22% weight error | -- | -- |

**B is the experiment, and it is nearly free.** It changes `module_overrides`
in a recipe and nothing else -- no new Metal, no new oracle, the existing
`tests/quantized.rs` covers the format already. If B fails on quality, C
cannot save it, because nvfp4 is not the more precise format; and if B holds,
C is a bytes-only follow-up that can be judged on its own.

**Judge each variant on three numbers, in this order.** Perplexity against
`baseline/`'s 6.3327 is the coarse gate. The KL against `baseline/logprobs.bin`
on the same 8 contexts is the sensitive one -- perplexity averages over
thousands of tokens and hides a format that is wrong in a narrow place. Only
then bytes and GPU busy, because a variant that is fast and worse is not a
result. `tools/dump_artifact_baseline.py` produces all of it, pointed at the
new artifact.

Expect C to be the one that fails on quality if any does -- it is the variant
betting that mxfp4 at 4 bits beats affine int4 at 4 bits by enough to replace
8 bits. If it fails, D (nvfp4 g16, 4.50 bits, a finer grid at 16) is the next
rung, and a partial C -- moving some of the 260 and not others -- is the
fallback, since the recipe's `module_overrides` are per-module by suffix.

### Formats and ideas considered, and where they stand

- **mxfp4 / mxfp8 / nvfp4.** In mlx, kernels in `vendor/mlx`
  (`fp_quantized.metal`), oracle available, one word in the recipe. This is the
  route, for that reason and not because the method is the best known.

  Two things Rigel (arXiv 2606.12765) settles about it on this chip. The fp4
  (E2M1) representable set **matches the OCP specification exactly**, across
  all 256 bit patterns, so the element format holds no surprises and a numpy
  reference would agree with the hardware. And block-scaled MXFP4 in the
  *tensor* path is **absent** on M4 despite the Metal 4.1 feature table
  claiming it -- which does not touch this route, since `fp_quantized.metal`
  is a software path, but does mean the faster `fp_quantized_nax.metal`
  variant is not an alternative here. See `M5.md`.
- **A codebook derived from the weights** -- an exponent table plus a mantissa
  in the weight, the table small enough to live in registers. Worth knowing
  that this *is* microscaling: mxfp4 is 4-bit elements with a shared 8-bit
  exponent per block of 32, which is the same idea with the table removed. The
  richer version, a non-uniform grid fitted to the weights (NF4, SqueezeLLM),
  buys quality per bit rather than bytes -- a global table could drop the
  per-group scale, 5.00 -> 4.00 bits, but per-channel variation is what the
  group scale exists for. **Note the trap in the framing**: the lookup argument
  optimises arithmetic, and dequantization arithmetic here is already free --
  it hides under the weight read, which is why the interleaved-layout idea was
  killed earlier. Only bytes move the number.
- **HIGGS** (Hadamard incoherence, MSE-optimal grids, data-free). One of its
  three blockers is now gone -- the parent gives bf16 weights to quantize. The
  rest stand: the runtime is CUDA (FLUTE), the format is not affine and rotates
  the activations too, so it changes the op rather than the weight layout, and
  MLX has no HIGGS, so there is no oracle and `max_abs == 0.0` has nothing to
  compare against. Its own benchmark table is also a caution: 4-bit HIGGS is
  *slower* than bf16 without `torch.compile` and only wins with it, which says
  the speed comes from fusion rather than from the bytes.
- **TurboQuant** (KV cache). Implemented for MLX on Metal already, in
  `vendor/mlx-vlm/mlx_vlm/turboquant.py`, so the runtime exists. Wrong profile
  for this model: 30 of 40 layers are gated-delta and have no KV cache at all,
  so the cache is 20 KB a token of context -- 1.9% of a step's traffic at 2600
  tokens, 24% at 32k. A long-context lever, and this port's bottleneck is
  weights at short context. If it ever matters, port mlx-lm's own quantized KV
  first, because that one has an oracle.
- **The framing behind all of these** -- fusing the model and the algorithm,
  specialising to weights a generic runtime cannot assume -- is already in this
  port and already paid: FR-Spec is exactly that, a ranked sub-vocabulary baked
  in, and it took 14% of the GPU work. The same framing covers a permanent
  `lm_head` prune and reordering experts by usage. What it costs is the oracle,
  which is why a bespoke format is the last resort and mlx's own formats are
  the first.

**The port needs `fp_quantized` kernels for B onwards** -- it carries only the
`affine_*` family today. mlx has `fp_quantized.metal` and `fp_quantized.h` in
`vendor/mlx`, and the oracle for them is MLX itself, so the usual contract
applies. Budget that as real work; it is the reason B and C are separate
steps.

---

## Disk

84 GB free of 1.9 TB at the time of writing, so the choreography is tight:

**This constraint is gone.** It was written against 84 GB free; the disk now
has **592 GB**, which holds the parent and several variants at once. Variants
*can* be compared side by side, so build them all and keep them until the
numbers are in.

What the old table forced is still worth recording, because it is why Phase 0
happened when it did: with 84 GB free, the order had to be captures, then
delete, then pull -- and deleting before capturing would have thrown away the
only baseline the experiment can be judged against. That ordering is no longer
forced by disk, but the reason for Phase 0 has nothing to do with disk.

That leaves room for the parent plus **one** artifact at a time, and forge
needs scratch on top. So variants cannot be compared side by side on this
disk: measure one, write the numbers down, delete, build the next. Keeping the
parent is what makes that cheap -- deleting it would mean a 70 GB download per
variant.

If an external volume is available, keeping the parent there and building
locally removes the whole constraint.

## How this goes wrong, and how we would notice

- **Wrong parent revision.** Caught in Phase 1 by the tensor counts, and in
  Phase 2 by the shas. Not caught later by anything.
- **A sanitization difference.** Caught only by the explicit `conv1d` shape
  check in Phase 3; the golden tests cannot see it, because MLX reads the same
  artifact and makes the same mistake. This is the failure mode to fear.
- **A quality regression from the format.** Caught by perplexity against the
  Phase 0 baseline -- which is why Phase 0 cannot be skipped.
- **A port bug in the new kernels.** Caught by `max_abs == 0.0`, as usual.
  This is the one failure the existing machinery already handles.
