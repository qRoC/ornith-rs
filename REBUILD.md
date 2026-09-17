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

## Phase 0 -- before anything is deleted

Everything here is unrecoverable once the artifact is gone. It is small and
quick; do all of it.

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

Build with the saved recipe **unmodified**, and compare per-file sha256
against `.mtplx-source.json`.

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

## Phase 4 -- the experiments, in order

One variant at a time; see the disk note below. For each: build, regenerate
refs, run the suite, then measure perplexity, bytes a token, and GPU busy a
token.

| # | recipe change | what it answers |
|---|---|---|
| A | none (Phase 2) | control |
| B | `body_mode: "mxfp4"`, `body_group_size: 32` | does MXFP4 match affine int4 at 4 bits? 5.00 -> 4.25 bits a weight |
| C | B, plus moving the 260 q8 overrides to 4-bit mxfp4 | **the actual prize**: does a better 4-bit format hold where affine int4 did not? ~35% of traffic |
| D | `nvfp4` group 16, if C is close but not close enough | 4.50 bits, a finer grid |

C is the one worth doing all of this for. B is the step that proves the format
works at all before betting the 8-bit half on it.

**The port needs `fp_quantized` kernels for B onwards** -- it carries only the
`affine_*` family today. mlx has `fp_quantized.metal` and `fp_quantized.h` in
`vendor/mlx`, and the oracle for them is MLX itself, so the usual contract
applies. Budget that as real work; it is the reason B and C are separate
steps.

---

## Disk

84 GB free of 1.9 TB at the time of writing, so the choreography is tight:

| step | free after |
|---|---|
| delete the current artifact (27 GB) | 111 GB |
| pull the parent (~70 GB) | ~41 GB |
| build one artifact (~27 GB) | ~14 GB |

The order is forced by that table: Phase 0's captures, then the delete, then
the pull. Pulling first leaves 14 GB and no room for forge's output, and
deleting before Phase 0 throws away the only baseline the experiment can be
judged against.

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
