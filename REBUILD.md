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

## Phase 2 -- reproduce what we had, before changing anything -- **DONE**

**The trunk is the same model, exactly.** Every one of the 1986560 float32
log-probabilities in `baseline/logprobs.bin` reproduces bit for bit, mean NLL
matches to twelve decimals, and all four greedy anchors are identical token
for token. The mechanism is proven and every Phase 4 variant now differs only
by what the recipe says.

**The MTP head is not the same head**, and that is the one substantive finding
of this phase -- see "The MTP sidecar" below. It changes nothing about Phase
4, whose subject is the quantized body, but it does retire one of the four
Phase 0 captures.

    perplexity    6.332690  ->  6.332690     delta 0.000e+00
    mean NLL      1.845725035884 -> 1.845725035884
    logprobs      max_abs 0.000000e+00, 1986560 of 1986560 identical
    anchors       4 of 4 identical

It is **not** byte-identical, and it cannot be on this toolchain. That is the
middle outcome the plan allowed for, so here is the explanation it asked for.

### What was run

    mtplx forge build \
      --repo ornith-ai/Ornith-1.5-35B-A3B \
      --model-root ~/.mtplx/models \
      --out /tmp/forge --run-id phase2 \
      --recipe "$(cat baseline/BUILD_RECIPE.json)" \
      --branded-name ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx

It found the pulled parent rather than downloading again ("using the copy
already on disk"), classified it `bf16_native`, and took the mixed-precision
lane because the recipe has `module_overrides`. The recipe's history keys
(`artifact_name`, `source_model`, `source_revision`, `release_repo`,
`architecture_signature`, `source_signature`) were inert as predicted -- no
stripping was needed.

| stage | |
|---|---|
| quantize body | **38 s**, "Quantized model with 5.889 bits per weight" |
| vision tower | 333 tensors, 893142496 bytes, restored from the parent |
| MTP sidecar | 785 tensors, gains delta-decoded back to the +1.0 convention |
| calibrate | 64 contract candidates, `exact_agreement` |
| verify | AR 63.7 tok/s, depth 1 **88.3** (acceptance 0.93), depth 2 **123.5** (0.83) |

The speed gate passed comfortably, which is worth noting against
`HANDOFF.md`'s finding that MTP does not pay off *in this port*: MTPLX's own
runtime gets 1.4x at depth 1 on the same weights, so the gap is in the
driving, not in the head.

### Why byte-identity is off the table

The published artifact records its own builder: `config.json`'s
`mtplx_wrapper.mtplx_version` is **2.9.1**. This machine has 2.11.3, with an
mlx-lm newer than whatever 2.9.1 pulled in. Six differences follow, none of
them a weight -- and then a seventh that is, which has its own section below:

| | published (2.9.1) | rebuild (2.11.3) |
|---|---|---|
| body shards | 17, largest 1.96 GiB | **5**, `make_shards(MAX_FILE_SIZE_GB = 5)` |
| MTP sidecar | `mtp/weights.safetensors` | `mtp.safetensors` |
| vision file | `model-vision-00001-of-00001.safetensors` | `model-vision.safetensors` |
| tokenizer | the parent's four files, verbatim | re-serialised by transformers 5.14.1 |
| config extras | `mtp_num_hidden_layers`, `num_nextn_predict_layers`, `publisher`, `mtplx_wrapper` | `mtplx_mtp_contract`, `mtplx_mtp_payload_audit` |
| MTP contract | `mtp_position_mode: "cache"` | `"local"` |

The two sidecar names are read out of `mlx_lm_extra_tensors.mtp_file` and the
index, by both the port and MTPLX, so they are naming and nothing else. The
shard count is immaterial for the same reason -- nothing reads a shard by
name, and `model.safetensors.index.json` maps every tensor. `"local"` and
`"cache"` are two spellings of one behaviour --
`_resolve_runtime_mtp_position_mode` maps the first onto the second, and only
`absolute`, `capped` and `wrapped` would be a different model. `tests/mtp.rs`
now accepts both spellings instead of the literal `"cache"`.

The tokenizer and the missing config keys are **not** cosmetic, and they are
what `tools/align_rebuild.py` exists for.

### The tokenizer, which is the trap in this phase

mlx-lm's `save()` ends with `tokenizer.save_pretrained`, so the artifact's
tokenizer comes back out through transformers rather than being copied. The
published artifact's four tokenizer files are the **parent's, byte for byte**
(sha256-checked against `baseline/SHA256SUMS`); the re-serialised ones are
not, and the round trip is lossy in five ways:

- the Split pattern loses `\p{M}` from both character classes -- it is
  literally the older transformers pattern from `HANDOFF.md`'s "two
  behaviours" section, now written *into the artifact*, which would delete the
  `Dialect::TokenizerJson` behaviour by leaving nothing to distinguish;
- ByteLevel flips `trim_offsets` false -> true, and the decoder flips
  `add_prefix_space` and `use_regex` false -> true;
- `merges` is rewritten from `"a b"` to `["a", "b"]`;
- `added_tokens` goes 26 -> 33, folding in the 7 that only the config carried;
- `vocab.json` and `merges.txt` are **not written at all** -- and those two are
  exactly what `Qwen2Tokenizer` rebuilds itself from, which is the default
  dialect of this port and what `mlx_lm.generate` runs.

`tokenizer_config.json` also shrinks 16718 -> 1161 bytes, losing
`added_tokens_decoder`, which `src/tokenizer/mod.rs` reads.

**None of this is visible to a golden test.** `dump_tokenizer.py` reads the
same `tokenizer.json` the port does, so both would agree on the wrong pattern
with `max_abs == 0.0` throughout. It is the same shape as the `conv1d` trap
and it was not on the list.

And `config.json` loses `num_nextn_predict_layers`, which 2.9.1 wrote and
2.11.3 does not. `TextConfig::load` falls back to **0** for it, so the MTP
head would be configured away; `MtpHead::load` asserts, so this one fails
loudly rather than silently, but only if something loads the head.

### So the procedure is now three steps, not two

    mtplx forge build ... --branded-name <name>
    python3 tools/align_rebuild.py ~/.mtplx/models/<name>
    python3 tools/check_rebuild.py ~/.mtplx/models/<name>

`align_rebuild.py` restores the parent's tokenizer files and the config keys
2.11.3 drops, and is idempotent; its docstring justifies every line it
touches. **Run it after every forge build, including each Phase 4 variant**,
or the variants get compared to each other through different tokenizers.

`check_rebuild.py` was itself wrong and has been fixed. Its inventory half had
been verified against the *parent* and never against an artifact, and two of
its six counts can never hold on one: an artifact's index has 1970 entries
(1637 text + 333 vision) where the parent's has 1811, and the MTP head is a
sidecar that the index does not list at all. The invariant that does hold is
the **module** count -- 733, which is the parent's 693 non-MTP non-vision
tensors plus one per layer for mlx-lm splitting the fused
`mlp.experts.gate_up_proj` -- because a module is one tensor or three
depending on whether the recipe quantized it, which is exactly what Phase 4
changes. It now also checks the MTP sidecar through the config key, the three
tokenizer properties above, `num_nextn_predict_layers`, and the sidecar's
seven norm gains; the per-file sha256 comparison is INFO, and the ten
published *content* files are compared by sha256 from `baseline/SHA256SUMS`
instead. Six of its invariants are now marked (blind) where there were two.

`baseline/SHA256SUMS` turns out to be the better manifest: it covers all 35
files where `.mtplx-source.json` carries a sha256 for only the 20 LFS ones,
and it was computed *before* publish scrubbed `config.json` and
`conversion_receipt.json` -- so it describes the local build, which is what a
rebuild is.

### The check that survived

Per-file sha256 cannot answer "are these the same numbers" once the file
layout moves. `baseline/logprobs.bin` can, and does: 8 full next-token
distributions, 248320 wide, float32. MLX is deterministic and file layout is
not an input to a matmul, so identical weights give identical bits --
`max_abs == 0.0`, the same contract the port is held to. Capture the other
side with

    cd tools && ORNITH_BASELINE_OUT=../refs/artifact/new \
      PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py <artifact>
    python3 tools/compare_baselines.py

In Phase 4 those three numbers stop being a check and become the measurement.

### The MTP sidecar, which is a different head

The trunk is bit-identical and the port's own `--mtp --greedy --raw` output is
character-for-character the Phase 0 anchor -- so tokenizer, config, index,
body and sidecar are all being read correctly, and speculative decoding is
exact as it should be. What changed is only how often the draft head *guessed
right*:

| prompt, depth 1 | Phase 0 | rebuild |
|---|---|---|
| merges two sorted lists | 51 cycles, 45 accepted, **0.88** | 49, 47, **0.96** |
| memory-bound vs compute-bound | 59, 36, **0.61** | 54, 41, **0.76** |
| first eight primes | 50, 45, **0.90** | 48, 47, **0.98** |
| what is a neutron star | 58, 38, **0.66** | 54, 42, **0.78** |

Higher every time, and not noise: the same prompt re-run gives 49/47/0.96
again, to the cycle. MTPLX's own build-time verification points the same way
-- 0.93 at depth 1 where the old artifact's `mtplx_runtime.json` recorded 0.78
-- though those two ran different prompt suites, so take it as corroboration
and not as a second measurement.

**What the sidecar is.** 785 tensors, of which **778 are byte-identical to the
parent's `mtp.*`**. The seven that differ are the RMSNorm gains, each exactly
1.0 higher: forge runs `shift_delta_mtp_norms` over the sidecar on the way
out.

**The shift is right, and proving it took a wrong turn worth recording.** The
parent's MTP norms sit inside the distribution of the parent's *trunk* norms
-- same means, same standard deviations -- which reads as "they are already
absolute and forge is double-shifting them". Restoring the parent's seven
tensors byte for byte and re-running gives a head that drafts **0 of 95**
tokens correctly. The comparison was between two delta-encoded sets: the
parent stores *both* the trunk's and the head's gains as deltas, and the
trunk's get their +1.0 from `qwen3_5.TextModel.sanitize`, which does it
whenever the checkpoint still has `mtp.*` weights or a torch-layout `conv1d`
-- both true of the parent, neither true of an artifact. That is the same
`should_shift_norm_weights` flag `HANDOFF.md` describes from the other
direction. Two quantities that are both wrong by the same 1.0 agree with each
other perfectly; the only reference that settles it is the model running.

**Why the head is different, and this part is measured rather than argued.**
The search is finite: 778 of the 785 tensors are the parent's, the recipe is
`same-parent-bf16-sidecar`, so the published sidecar's 778 are the same 778
and the only thing that can differ is which of the seven gains were shifted.
That is at most 128 configurations, and the port will evaluate any of them --
patch the sidecar, run the prompt, read the acceptance off `--mtp`.

Three of them, and the third is the answer:

| excluded from the shift | prompt 1, depth 1 |
|---|---|
| nothing -- the 2.11.3 rebuild | 49 cycles, 47 accepted, 0.96 |
| `mtp.norm` | 49, 47, 0.96 -- **no effect at all** |
| **`post_attention_layernorm`** | **51, 45, 0.88 -- Phase 0, to the cycle** |
| `pre_fc_norm_embedding` + `pre_fc_norm_hidden` | 95, 0, **dead** |
| all seven -- the parent's bytes | 95, 0, **dead** |

Leaving `post_attention_layernorm` at the parent's value and shifting the
other six reproduces the Phase 0 profile on **all eight runs** -- four prompts
at depth 1 and four at depth 3, exact on cycles, on accepted drafts, and on
every per-level rate including 0.87 / 0.53 / 0.47 and 0.52 / 0.48 / 0.23.
Eight independent numbers landing on eight targets is not a coincidence.

**So the published artifact was built by a rule that decided per tensor, and
got one of the seven wrong.** All seven parent gains are delta-encoded, but
their means differ, and `post_attention_layernorm` is the only one of the low
set whose mean (0.869) is above `LOW_DELTA_MEAN_MAX = 0.5`. A per-tensor gate
reads that as "already absolute" and skips it. Six norms got their +1.0 and
the seventh stayed a delta, so the head ran on one layer whose gains were a
full 1.0 low. 2.11.3's comment beside the suffix lists is the fix, and it says
so outright: the convention is decided per sidecar "and then applied to all
seven norms -- **never per tensor**, and never *always*".

Two things that fell out of the search and were not expected. `mtp.norm` --
the final norm feeding the shared `lm_head` -- is **inert**: shifting it or
not changes no draft decision in 96 tokens. And `pre_fc_norm_embedding` and
`pre_fc_norm_hidden` are not merely important but load-bearing; leaving those
two unshifted does not degrade the head, it kills it, 0 accepted out of 95.
So the seven are in three regimes, not on a spectrum. The guess going in --
that the compounding-with-depth signature pointed at the norms in the
recursion path -- was wrong: `mtp.norm` is in that path and means nothing.
The depth signature has a duller explanation. One layer's gains are wrong, and
that layer runs three times per cycle at depth 3.

Depth 3 says it louder, and says something more:

| prompt, depth 3 | Phase 0 | rebuild |
|---|---|---|
| merges two sorted lists | 38 cycles, 58 of 112, 0.87 / 0.53 / **0.47** | 28, 67 of 84, 0.96 / 0.85 / **0.74** |
| memory-bound vs compute-bound | 52, 43 of 154, 0.52 / 0.48 / **0.23** | 36, 59 of 108, 0.72 / 0.73 / **0.74** |
| first eight primes | 32, 64 of 95, 0.94 / 0.83 / **0.38** | 29, 67 of 86, 0.93 / 0.89 / **0.70** |
| what is a neutron star | 47, 49 of 141, 0.57 / 0.52 / **0.57** | 33, 63 of 99, 0.82 / 0.81 / **0.64** |

**The gap widens with depth.** At level 1 the two heads are close enough to
argue about -- on the primes prompt the old head is even a hair ahead, 0.94 to
0.93. By level 3 there is nothing to argue about: 0.23 becomes 0.74, 0.38
becomes 0.70. That is one decoder layer carrying gains a full 1.0 too low and
being run three times per cycle instead of once.

Worth noting for its own sake: these were captured at 10 - 32 tok/s against
the 70 tok/s the same prompts decode at depth 1, on a machine at 15 GB of
swap. That is `HANDOFF.md`'s "MTP does not pay off here" showing up again, and
the acceptance counts are unaffected by it -- they are deterministic. Only the
wall clock is, which is why the counts are what this table records.

**What that retires, and what it does not.**

- `baseline/baseline.json`'s `mtp_profile` is **no longer a valid before**.
  It measures a head that no rebuild on this toolchain will reproduce. The
  replacement is captured and lives at `refs/artifact/new/mtp_profile.json`
  -- the eight runs above, depth 1 and depth 3, in the same shape.
- Nothing else moves. The sidecar is bf16 and no recipe variant touches it,
  so the new profile is as stable a control as the old one was, and the
  quality baseline that Phase 4 actually turns on -- perplexity and
  `logprobs.bin`, both trunk-only -- is bit-identical and fully comparable.
- `HANDOFF.md`'s acceptance tables were all measured on the old head and now
  read low.
- `check_rebuild.py` gained an invariant for it: the seven gains must *not*
  equal the parent's. A sidecar left delta-encoded gives a head that drafts
  nothing while every golden test stays green, because MLX reads the same
  file. That is a fourth member of the blind family. Note it would **not**
  have caught the published artifact, whose six-of-seven sidecar is neither
  the parent's nor correct; catching that needs the acceptance profile, which
  is the argument for keeping one per variant.

## Phase 3 -- re-anchoring the port -- **DONE**

Every reference regenerated, every test green, and one more difference between
the two toolchains found -- the first one that is a weight layout.

    fast suite       72 passed, 0 failed
    --ignored         8 passed, 0 failed
      model           full_forward_matches_mlx
      generate        greedy and sampled, token for token against mlx_lm
      mtp             capture, head_matches_mtplx, speculative, window vs steps
      vision_generate vision_greedy_matches_mtplx

1. **Every golden reference regenerated**, 18 writers, all of them clean. The
   stale 518 MB was moved out of `refs/` first rather than overwritten in
   place: a writer that fails silently leaves a gap that way, instead of a
   test comparing the new artifact against the deleted one's tensors and
   passing. Only `refs/artifact/` was kept -- it holds the rebuild's own
   baseline capture, which is current.
2. **The whole suite runs.** `cargo test --release -- --ignored
   --nocapture --test-threads=1` covers all eight model-loading cases in one
   pass; cargo runs the test binaries sequentially, so the "only one model
   fits" rule holds without arranging it by hand.
3. **The artifact properties are checked by `tools/check_rebuild.py`**, which
   passes -- see Phase 2 for what each one is and why the golden tests cannot
   see it.
4. **The Phase 0 anchors were re-run in Phase 2** and are exact.

### The vision tower's patch embedding, and why the port caught it

`tools/dump_vision_ops.py` failed where nothing else did:

    ValueError: [conv] Expect the input channels in the input and weight
    array to match -- input: (256,2,16,16,3) weight: (1152,3,2,16,16)

mlx's `Conv3d` wants `(out, kD, kH, kW, in)`; a torch checkpoint stores
`(out, in, kD, kH, kW)`. **2.9.1 transposed the patch embedding when it
grafted the tower in, and 2.11.3 copies the parent's layout verbatim.** Both
the writer and `src/vision/tower.rs` were written against the published
artifact and say so in their comments -- "already `[O, kD, kH, kW, C]` in this
checkpoint".

MTPLX itself never notices: `_conv_weight_in_mlx_layout` transposes at load,
so `dump_vision.py` and `dump_vision_generate.py` both succeeded. This port
reads the tensor as it lies, and `VisionTower::load` asserts on the shape --
so it **fails closed**, which is the one thing that keeps this off the blind
list. A silent version of this bug would have convolved the wrong axes.

`tools/align_rebuild.py` applies the transpose, and the result is
**byte-identical to the published tower**: 893179493 bytes, sha256
`ce387e11510de90772c482840c6d2bb8df1fe0f2d280b02cfea8484d04007f6b`. That is
the first of the twenty large files to match exactly, and it settles that the
transpose is precisely what 2.9.1 did rather than a plausible substitute.

Two ways to get this wrong, both of which I did before getting it right:

- `mx.load` is **lazy** and its arrays stay bound to the file. Writing
  `mx.save_safetensors` back to that same path truncates it before the reads
  happen, and the sidecar ends up empty. Materialise with `mx.eval` first,
  write beside it, then rename.
- `mx.save_safetensors` **appends `.safetensors`** when the name lacks it, so
  a temp file called `...safetensors.new` is written as
  `...safetensors.new.safetensors` and the rename finds nothing.

Recovery, if it happens again: the tower is 333 tensors grafted verbatim from
the parent, so `mtplx.vision_graft.graft_vision_tower(parent, artifact)`
rebuilds it exactly. It is idempotent on an index that still resolves a vision
prefix, so strip those 333 keys from `model.safetensors.index.json` first --
and put `metadata.total_parameters` back afterwards, because the graft adds
the vision parameters to whatever it finds.

### One test asserted a file layout

`safetensors::tests::reads_the_index_and_headers` required exactly 18 shards,
17 body plus vision. That is `mlx_lm`'s `MAX_FILE_SIZE_GB` packing tensors
greedily, not a property of the model, and it moved to 6. The test now asks
what the index actually owes: that it resolves every shard it names, that the
vision tower is among them, and that the set carries all 1970 tensors.

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

**B is also the only variant that can fail on quality.** C does not re-ask
B's question: nvfp4 g16 measures 9.51 / 9.28 / 9.18% against affine int4
g32's 7.82 / 9.18 / 9.37%, which is the same error to within the noise of the
measurement, half a bit cheaper. So C cannot rescue a failed B and cannot
spoil a successful one; it is a bytes-only follow-up, judged on bytes. D is
not a rung below C either -- it is the same 4-bit question asked of two more
families.

If B fails, the fallback is a **partial B**: move some of the four families
and not others, which `module_overrides` supports directly, being per-module
by suffix. `lm_head` and `linear_attn.in_proj_qkv` are 38% of decode traffic
between them, so even half of B is most of the prize.

**mxfp4 is not a fallback and not a rung.** `tools/quant_error.py` puts it at
20.32 / 21.19 / 22.00% -- two and a half times affine int4's error to save
0.75 bits. Measured against the best any code can do at its own rate it is
worse still: the gaussian floor is 5.26% at 4.25 bits and 3.12% at 5.00, so
affine int4 lands 2.5x above its floor and mxfp4 nearly 4x above its own.
That is the power-of-two scale and the missing zero-point, not the bit rate,
and no ordering of the experiments makes it the right thing to try.

### What the experiments measured -- **DONE**

Four artifacts, built from the same verified parent by the same toolchain,
measured against `baseline/` on the same corpus and the same eight contexts.

Named for what each one puts at 4 bits, not for the tier the publisher sold
it as -- `int8-everywhere` is the artifact this port was built on, and the
rest are what the experiment moves.

| what is at 4 bits | 8-bit left | ppl | vs control | max KL | GB/token | GPU busy |
|---|---:|---:|---:|---:|---:|---:|
| **`int8-everywhere`** -- the control | 260 | 6.3327 | -- | 0 | 2.739 | 10.40 ms |
| **`int4-head-linattn`** -- `lm_head` and the three `linear_attn` projections | 169 | 6.5022 | +2.7% | **0.048** | 2.076 (-24.2%) | 9.17 ms (-11.8%) |
| `nvfp4-head-linattn` -- the same four families, in nvfp4 g16 | 169 | 6.6010 | +4.2% | **0.429** | 1.925 (-29.7%) | 8.55 ms (-17.8%) |
| **`int4-except-gates`** -- everything but the 80 router gates | 80 | 6.6422 | +4.9% | **0.323** | 1.858 (-32.2%) | 8.29 ms (-20.3%) |

The GB/token column is now `tools/step_bytes.py` reading the safetensors
headers rather than arithmetic off the module inventory; it moves the three
older absolutes by <= 0.2% and leaves their deltas unchanged. GPU busy is
`ORNITH_PROF=1`, greedy, 256 tokens, a 27-token prompt, two runs each and
within 1% of each other -- the wall clock is not reported because this machine
was carrying a 12-17 GB swap file throughout and `HANDOFF.md` has the reason
that number cannot be trusted. The earlier AR / D1 tok/s columns were MTPLX's
own verification figures and are in the forge logs.

**The answer to Phase 4's question is yes, and `int4-head-linattn` is where
to stop.** These
modules do tolerate 4 bits: it costs 2.7% of perplexity and 0.048 nats of KL
for a quarter of the bytes and 18% of the decode rate. Going further does not
scale:

    control -> head-linattn    24.2 points of traffic for 0.048 nats   0.0020 a point
    head-linattn -> all-but-gates 8.0 points of traffic for 0.275 nats   0.0344 a point

**Seventeen times the distribution cost per point of traffic past
`int4-head-linattn`.** The last eight points are paid for at a rate the first
twenty-four were not, and they buy 4.3% of decode. That is the knee.

**And the knee held when a fourth variant went looking for a way round it.**
`nvfp4-head-linattn` buys its extra points with a format rather than with
more modules, and pays 0.0693 a point -- twice `int4-except-gates`'s rate, at
*fewer* points. Two different ways past `int4-head-linattn`, one changing
coverage and one changing precision, and both land on the wrong side of the
same knee. See "nvfp4, built and measured" below.

**Perplexity would have hidden this, exactly as the plan warned.** It moves
+2.7% -> +4.9%, a factor of 1.8, while the KL moves 0.048 -> 0.323, a factor
of 6.7. Averaging 6620 tokens flattens a distribution that has moved much
further in places; the eight full 248320-wide distributions do not. Judging
these two on perplexity alone would have made `int4-except-gates` look like a
reasonable next step. It is not.

### The row that is missing, and why it cannot be filled here

The honest reference for all of this is the bf16 parent: not "how far did a
variant move from the control" but "how far is each from the model the control
was cut from". Nobody has that number -- the publisher's own card says "Full
BF16 parent parity was not run" -- and it cannot be taken on this host either.

    parent weights on disk              67.0 GiB
    max_recommended_working_set_size    51.8 GiB
    max_buffer_length                   38.9 GiB
    memory_size                         64.0 GiB

`mlx_lm.load` ends in `mx.eval(model.parameters())`, which asks for all 67.0
GiB resident at once against a 51.8 GiB working set. It went through once, by
luck, and answered a five-token prompt correctly in 170 s; the next attempt
died in the loader with `[METAL] Command buffer execution failed: Caused GPU
Timeout Error`. Unified memory means the weights need not all be resident,
but the GPU working set is not the whole 64 GiB, and 15 GiB over is not a
margin luck closes twice.

`load(..., lazy=True)` skips that eval and would page through layer by layer,
which is the one avenue left. At ~170 s a forward and 26 chunks of 256 tokens
for the perplexity alone -- where a 256-token chunk touches nearly all 256
experts per layer, not the 40 a 5-token prompt reaches -- it is over an hour
with no guarantee the long chunk does not hit the same timeout. It is a
measurement to take on a bigger host, not to grind out here.

### The publisher shipped this experiment already, and it sharpens the result

The artifact this port runs on is one of a matrix -- the sibling recipes carry
`adapter_id: qwen35-moe-ornith15-quant-matrix-v1` and
`profile: control-candidate`:

| repo suffix | label | 8-bit | bf16 |
|---|---|---:|---:|
| `attention8-bf16recurrence-vision` | "Flagship / Highest Quality" | 260 | 60 |
| `affine4-router8-bf16recurrence-vision` | "Balanced" | 80 | 60 |
| `affine4-router8-vision` | "Most Compact" | 80 | 0 |

So `int4-except-gates` above *is* their "Balanced", rebuilt here from their
own recipe,
and they independently landed on the plan's rule that the router gates never
move. Two things their cards do not give, and this measurement does.

**Their benchmark cannot rank the variants.** Balanced and Compact both carry
a Shiftedx Bench table; the Flagship carries none, so the tier this port
actually runs on has no row to compare against. And the two tables disagree in
the wrong direction: Compact, the *more* quantized one, scores 7/10 on Quality
against Balanced's 6/10, while losing Long context 9/15 to 11/15 and Vision
1/4 to 2/4. At ten pass/fail items around p = 0.65 one standard deviation is
about 1.5 items, so a 60-vs-70 gap is one item and well inside noise. The two
runs are not even comparable -- Balanced's effective tested context is 131072
and Compact's is 4096. The card says as much itself: "the benchmark does not
produce a composite intelligence score".

**Most Compact was never built here, and the arithmetic is why.** It buys
0.2 percentage points of traffic over Balanced -- 1.850 against 1.855 GB a
token -- by quantizing the recurrence inputs the other two recipes keep in
bf16 deliberately. Near-zero prize against real risk, the same shape as the
argument for leaving the 80 gates alone. Seconds of arithmetic, one build and
one qualification run saved.

**What this leaves for nvfp4.** It measures the same weight error as affine
int4 g32 at half a bit less, so applied to `int4-head-linattn`'s four families
it is a bytes-only follow-up: roughly three more points of traffic at that
variant's quality, not `int4-except-gates`'s. That is now the only remaining rung worth building, and it needs the
`fp_quantized` port. Whether three points justifies a new kernel family is a
separate decision from the one this phase settled.

*That prediction was wrong, and the section below is what it cost.* It was
right about the bytes to the decimal and wrong about the quality by a factor
of nine, because it priced the format on synthetic draws.

### nvfp4, built and measured -- and it does not pay

The rung named above is built: `recipes/nvfp4-head-linattn.json` is
`int4-head-linattn` with `body_mode` nvfp4 and `body_group_size` 16, which
moves all 283 four-bit modules to nvfp4 g16 and leaves the 169 eight-bit ones
on affine. The kernels are in the port (`kernels/fp_quantized_inst.metal`,
`src/ops/quantized.rs`), and the whole language model runs on them at
`max_abs == 0.0` against MLX -- see below. The prize and the price:

| against `int4-head-linattn` | |
|---|---|
| bytes a decode step | 2.076 -> 1.925 GB, **-7.3%**, exactly the predicted figure |
| GPU busy a token | 9.17 -> 8.55 ms, **-6.7%** |
| perplexity | 6.5022 -> 6.6010, +1.5% |
| max KL against the control | 0.048 -> **0.429 nats**, a factor of **8.9** |

**The bytes arrived and the quality did not.** The float-pair table below
priced this at "9.4% -> 9.5% weight error, which is nothing"; the measured
distribution cost is nine times `int4-head-linattn`'s.

**It is strictly dominated by a variant already rejected.** Against
`int4-except-gates`, which Phase 4 turned down for being past the knee, nvfp4
is worse on every axis that matters:

| | GB/token | GPU busy | max KL | ppl |
|---|---:|---:|---:|---:|
| `nvfp4-head-linattn` | 1.925 | 8.55 ms | 0.429 | 6.6010 |
| `int4-except-gates` | **1.858** | **8.29 ms** | **0.323** | 6.6422 |

More bytes, more GPU time, *and* a third more KL. The one column it wins is
perplexity -- and **that is the same trap this phase already documented,
firing a second time and pointing the other way.** Phase 4 warned that
perplexity would make `int4-except-gates` look reasonable; here it makes
nvfp4 look like the better of the two, and the KL says it is the worse. A
format comparison is no safer on perplexity than a coverage comparison was.

Cost per point of the control's traffic, on the same ladder as above:

    control -> head-linattn      24.2 points for 0.048 nats   0.0020 a point
    head-linattn -> except-gates  8.0 points for 0.275 nats   0.0344 a point
    head-linattn -> nvfp4         5.5 points for 0.381 nats   0.0693 a point

**Twice the rate of the rung that was already too expensive.**

#### Why the prediction missed: synthetic weights are not these weights

`tools/quant_error.py` prices a format on a million gaussian, laplace and
student-t draws. `tools/weight_error.py` is new and asks the same question of
the model: dequantize a module out of each artifact and compare it to the
same module in the bf16 parent. The parent cannot be *loaded* on this host,
but `mx.load` is lazy and one tensor is at most 1.0 GiB, so one module at a
time is free -- this is the one thing the missing bf16 row can still give.

Relative RMS weight error against the parent:

| module | affine int8 g64 | affine int4 g32 | nvfp4 g16 |
|---|---:|---:|---:|
| `lm_head` | 0.75% | 8.33% | **10.39%** |
| `linear_attn.in_proj_qkv` | 0.72% | 8.11% | **10.87%** |
| `linear_attn.in_proj_z` | 0.73% | 8.17% | **10.64%** |
| `linear_attn.out_proj` | 0.78% | 8.52% | **10.80%** |
| `switch_mlp.gate_proj` | -- | 8.21% | **13.11%** |
| `mlp.gate` (the router) | 0.86% | -- | -- |
| *predicted, gaussian / laplace / t(4)* | *0.54 / 0.67 / 0.70* | *7.82 / 9.18 / 9.37* | *9.51 / 9.28 / 9.18* |

**affine int4's prediction transferred and nvfp4's did not.** Affine lands at
8.1-8.5%, between its gaussian and laplace figures, which is what a
heavier-tailed real weight should do. nvfp4 lands at 10.4-10.9% on the dense
projections -- above *all three* of its predicted numbers -- and at 13.11% on
the expert slabs, 40% above the worst of them. So nvfp4 is not "equal error
half a bit cheaper" on these weights; it is 1.3x the error on the dense
projections and 1.6x on the experts.

`weight_error.py --decompose` splits nvfp4's share of that in two, by
reconstructing fp4 by hand over the same parent tensors with the shared scale
at three precisions. The affine column below is mlx's own, from the table
above; the E4M3-round-to-nearest column is the hand reconstruction, and it
reproduces **mlx's nvfp4 to the last digit on all five modules** -- that
agreement is what says the exact-scale column is measuring the E2M1 grid and
not a reimplementation of the format.

| | affine int4 g32 (mlx) | fp4 g16, exact scale | nvfp4 g16, E4M3 rn | clipped | E4M3 rounded up |
|---|---:|---:|---:|---:|---:|
| `lm_head` | 8.33% | 9.34% | 10.39% | 4.2% | 11.61% |
| `in_proj_qkv` | 8.11% | 9.41% | 10.87% | 4.6% | 11.47% |
| `in_proj_z` | 8.17% | 9.39% | 10.64% | 4.4% | 11.62% |
| `out_proj` | 8.52% | 9.33% | 10.80% | 4.5% | 11.62% |
| `switch_mlp.gate_proj` | 8.21% | 9.38% | **13.11%** | 4.2% | **11.29%** |

- **The E2M1 grid is the smaller half.** Even with a perfect scale and half
  the group size it is 9.3-9.4% against affine's 8.1-8.5% -- 12-16% more
  error before the scale is quantized at all. A non-uniform grid suits weights
  in principle; four exponent levels and one mantissa bit is not enough of
  one, and the missing zero point costs on top. But that is a point and a
  half, not the gap.
- **Everything past that is the shared scale, and the cause is not what it
  first looked like.** The proximate mechanism is clipping -- a scale rounded
  *down* puts the group's own maximum past the E2M1 clip at 6.0, which
  affine's min/max scale cannot do by construction, and 4.2-4.6% of elements
  land there. Rounding the scale up instead removes the clipping entirely and
  takes the experts from 13.11% to 11.29% while costing every other module
  0.7-1.2 points.

#### The root cause: 99.6% of these group scales are subnormal

Why is a scale with three mantissa bits rounding down by enough to clip? It
is not the mantissa. **E4M3's smallest normal is 2**-6, and this model's
weights have sigma 0.015**, so a group's `max/6` lands around 2**-7.7 and
99.6% of them are *subnormal* -- encoded on a fixed grid of 2**-9, with two
or three steps to work with. That is a ~20% relative error on the scale, not
the ~6% three mantissa bits suggest, and it multiplies the whole group.

Two consequences, and the second is the useful one.

**`tools/quant_error.py` had a bug that no synthetic draw could reveal.** Its
`fp8()` took `floor(log2(a))` unclamped, so it gave every scale a full 3-bit
mantissa and never modelled a subnormal. On a unit-variance draw that is
correct -- a group max near 3 gives a scale near 0.4, comfortably normal -- so
**fixing it changes not one digit of either synthetic table**, byte for byte.
It is wrong only in the regime real weights live in, and the study never
entered that regime because its inputs were unit-variance. A self-consistent
study, an invisible defect, and a conclusion off by a factor of nine once it
met the model. `fp8()` now calls `e4m3()`, which had modelled subnormals
correctly all along for mxfp8's *elements*; the two encodings are the same
encoding and the file had one right and one wrong.

**And nvfp4 has an input for exactly this, which `mlx_lm.convert` does not
pass.** One fp32 a tensor, `global_scale`, folded in as
`inv_scale_enc = global_scale / (F8E4M3_MAX * F4E2M1_MAX)` -- it exists to
move the group scales into E4M3's normal range before they are encoded. With
it, on these same weights (`tools/weight_error.py --schemes`):

| | as forged | + global scale | affine int4 g32 |
|---|---:|---:|---:|
| `lm_head` | 10.38% | **9.45%** | 8.03% |
| `in_proj_qkv` | 10.86% | **9.53%** | 7.81% |
| `in_proj_z` | 10.65% | **9.50%** | 7.90% |
| `out_proj` | 10.79% | **9.44%** | 8.25% |
| `switch_mlp.gate_proj` | 13.07% | **9.48%** | 7.94% |

(The left column reproduces mlx's own measurements on all five to within
0.04 points, which is what says the model is the format.)

**So the built variant pays a penalty the format does not require.** The
global scale recovers 1 to 3.6 points and flattens the expert anomaly
completely -- the experts were never a different distribution, they were
further into the subnormals. What is left at 9.5% against affine int4's
7.8-8.3% is the E2M1 grid, which is the real and irreducible part.

**Does that overturn the verdict? Not on the measurements in hand, and it is
not settled either.** The rung that was built is the rung `mlx_lm.convert`
can forge, and it lost. A global-scaled nvfp4 would sit at 9.5% rather than
10.4-13.1%, which is most of the way back to affine int4's error at half a
bit less -- so its KL would land somewhere well below 0.429 and above 0.048,
and whether that beats the knee is an open question rather than a predicted
one. This project has now mispriced that prediction twice, so it stays open
until measured.

What it would cost, honestly: a quantization pass outside `mlx_lm.convert`
(neither it nor `nn.QuantizedLinear.from_linear` passes `global_scale`), the
`_hgs` kernel specialisations this port deliberately did not instantiate, and
a loader path for the third tensor -- which `WeightStore::quant_matrix`
currently rejects by design. Perhaps half the work of the port that is
already here, for a question worth asking only if the byte lever is still
the one being pulled.

**The lesson for `IDEAS.md`, measured rather than feared.** That file's two
leading ideas both spend a coarser shared scale to buy a smaller group, which
is what nvfp4 just lost on, so they were run on real weights too:

| scheme | bits/w | real weights | predicted (gaussian) |
|---|---:|---:|---:|
| affine g32, bf16 scale+bias -- the body today | 5.00 | 7.8-8.3% | 7.82% |
| **#1 affine g16, fp8 scale+bias, with a global scale** | 5.00 | **7.0-7.2%** | 7.02% |
| #1 without one | 5.00 | 11.8-18.6% | -- |
| #2 Lloyd-Max g16, fp8 scale, with a global scale | 4.50 | 7.5-18.7% | 7.50% |
| Lloyd-Max g32, bf16 scale | 4.50 | 8.2-9.0% | 8.21% |

- **#1 survives, and only with the global scale.** 7.0-7.2% against the
  body's 7.8-8.3% at the same 5.00 bits -- the predicted 10% holds on real
  weights, though the extra 15-17% the heavy-tailed synthetic columns
  promised does not. Without the per-tensor fp32 the same scheme is 11.8-18.6%
  and far worse than what it replaces. **The idea was never "fp8 scales are
  fine", it was "fp8 scales are fine once something tells them the
  magnitude", and the study could not see the difference.**
- **#2 does not reproduce and is not usable as stated.** 7.5-7.7% on three
  modules, 9.4% on the experts and **18.7% on `in_proj_qkv`** -- against a
  predicted 7.50%. A single codebook fitted across a tensor is evidently
  sensitive to something that tensor has and the others do not; whether that
  is the fit or the scheme is unresolved, and it is a prerequisite rather
  than a detail.

### The other 4.50-bit format, and what it exposed about the measurement

`weight_error.py --schemes` said the best thing left at 4.50 bits was not an
mlx mode at all: **affine int4 g32 with the scale and bias stored as E4M3**
measures 8.3-9.0% weight error on real weights against nvfp4's 9.4-9.5% at
the same rate, and the body's 7.8-8.3% at 5.00 bits. It needs no new grid and
no new packing -- only a different encoding of the two numbers a group
carries.

It was measured without writing a kernel. An E4M3 value times a power of two
keeps four significant bits and bfloat16 holds eight, so a scale and bias cut
to the candidate format **can be written out in the ordinary affine layout
with no loss at all**. `tools/forge_fp8scale_twin.py` builds that stand-in
from the bf16 parent: a plain `affine int4 g32` artifact that reconstructs,
bit for bit, what the 4.50-bit format would, and that unmodified mlx-lm and
the unmodified port both run. `tools/aq.py` is mlx's `affine_quantize` in
numpy and reproduces `mx.quantize` bit for bit with the encoding left alone,
which is what says the substitution is the only thing being measured.

| against `int4-head-linattn` | |
|---|---|
| bytes a decode step | 2.076 -> 1.925 GB, -7.3% (arithmetic; the twin is 5.00 bits on disk) |
| weight error | 8.11-8.71% -> 8.34-8.96%, **+3% relative** |
| perplexity | 6.5022 -> **6.5035**, +0.02% |
| max KL against the control | 0.048 -> **0.443 nats**, a factor of **9.2** |

**It fails, and it fails exactly as hard as nvfp4** -- 0.443 against 0.429 --
while sitting four percentage points closer to `int4-head-linattn` in weight
error. Both 4.50-bit routes are also dominated by `int4-except-gates`, which
reads fewer bytes (1.858 GB) at a lower KL (0.323).

**So weight error does not predict the thing that decides.** This is worth
stating flatly, because `tools/weight_error.py` was built earlier the same
day and this format was chosen over nvfp4 *on its numbers*:

| | weight error vs `int4-head-linattn` | max KL |
|---|---:|---:|
| `int4-head-linattn` | -- | 0.048 |
| the fp8-scale twin | +3% relative | 0.443 |
| `nvfp4-head-linattn` | +34% relative | 0.429 |

A tenfold difference in how far the weights moved, no difference in where the
distribution ended up. `weight_error.py` remains good at *explaining* a
result -- it found nvfp4's subnormal group scales, and that diagnosis held --
and it is not a selection criterion. Nothing cheap has yet been found that
is.

#### Seven of the eight contexts cannot tell these variants apart

The headline is a **max** over eight contexts, and unpacking it changes what
the number is:

| context | `int4-head-linattn` | `int4-except-gates` | fp8-scale twin | `nvfp4` |
|---|---:|---:|---:|---:|
| 0 `The capital of France is` | 0.0145 | 0.0116 | 0.0226 | 0.1156 |
| 1 `def quicksort(arr):` ... | 0.0002 | 0.0007 | 0.0011 | 0.0023 |
| 2 `In a memory-bound decode step` ... | 0.0291 | 0.0481 | 0.0522 | 0.0878 |
| 3 `1, 1, 2, 3, 5, 8, 13,` | 0.0076 | 0.0040 | 0.0384 | 0.0437 |
| 4 `The Rust borrow checker prevents` | 0.0481 | 0.0909 | 0.1052 | 0.1154 |
| 5 `SELECT name FROM users WHERE` | 0.0277 | 0.0278 | 0.0286 | 0.0632 |
| **6 `Дважды два равно`** | **0.0472** | **0.3232** | **0.4433** | **0.4294** |
| 7 `def fib(n):` ... | 0.0001 | 0.0001 | 0.0003 | 0.0002 |
| **max** | 0.0481 | 0.3232 | 0.4433 | 0.4294 |
| mean | 0.0218 | 0.0633 | 0.0865 | 0.1072 |
| median | 0.0211 | 0.0197 | 0.0335 | 0.0755 |

**Every verdict this phase has reached rests on context 6, the one Russian
prompt.** On the other seven the four variants are within 0.1 nats of each
other and of the control, and on two of them within 0.001. Change the
statistic and the ranking changes with it: by median, `int4-except-gates`
*ties* `int4-head-linattn`.

Two readings, and the second is the right one.

- It could be that a max over eight samples is a noisy statistic and the
  phase has been reading noise. That is the worry, and it deserves the
  weight.
- But the same context separates the variants in **four independently built
  artifacts**, by a factor of ten, in the same order every time -- and the
  one variant that holds on it is the one every other measure also prefers.
  That is not noise; it is the only probe in the set with any discriminating
  power, and the other seven are too easy to be doing anything.

So the conclusion survives -- `int4-head-linattn` is where to stop, and it is
the only variant that holds where it is hard -- while the confidence in it
does not. **Four format experiments have now been decided by one four-word
Russian prompt**, which is a thin foundation whichever way the answer comes
out. The corpus behind the perplexity is 6620 tokens and cannot see any of
this at all: it puts the twin at 6.5035 against 6.5022, a difference of two
parts in ten thousand, for a variant that moved a distribution nine times
further.

**That is the next thing to fix, and it comes before the next format.**
Whatever is tried next will be judged on this set, and right now the set has
about one bit of discriminating power. It wants more contexts that are hard
in the way context 6 is hard -- other scripts, rare tokens, long-range
dependencies -- and enough of them that a maximum is not a single sample.

### The float pair, asked and priced

"nvfp4 with nvfp8 instead of int4 with int8" is the obvious next thought and
it is half right. **There is no nvfp8** -- mlx takes exactly four modes,
`affine`, `mxfp4`, `nvfp4`, `mxfp8` -- so the float pair on offer is nvfp4 and
mxfp8. `tools/quant_error.py` now measures the 8-bit side too:

| format | bits/w | gaussian | laplace | student-t(4) |
|---|---|---|---|---|
| affine int8 g64 -- where the 169 sit | 8.50 | **0.54%** | 0.67% | 0.70% |
| mxfp8 g32 | 8.25 | **2.94%** | 3.17% | 3.23% |

Applied to `int4-head-linattn`, whose read is 2421M weights at 4 bits and
520M at 8:

| formats | GB/token | vs it | 4-bit err | 8-bit err |
|---|---:|---:|---:|---:|
| affine int4 g32 + affine int8 g64 | 2.066 | -- | 9.4% | 0.70% |
| **nvfp4 g16** + affine int8 g64 | 1.915 | **-7.3%** | 9.5% | 0.70% |
| affine int4 g32 + **mxfp8 g32** | 2.050 | -0.8% | 9.4% | **3.23%** |
| nvfp4 g16 + mxfp8 g32 | 1.898 | -8.1% | 9.5% | 3.23% |

**nvfp4 for the 4-bit side: yes**, and it is the rung already named -- 7.3%
of bytes for 9.4% -> 9.5% error, which is nothing. It needs the
`fp_quantized` port and nothing else.

**mxfp8 for the 8-bit side: no.** It buys 0.8% of bytes for 4.6 times the
weight error, and the reason is the one that killed mxfp4 one rung down: a
power-of-two shared scale, no zero point, three mantissa bits. The same shape
at 8 bits as at 4.

The deeper reason there is nothing to win there: after
`int4-head-linattn` only **17.7% of the read is still 8-bit**, so even a free
0.25 bits is under a point. And those 169 modules are at 8 bits *because they
need the precision* -- trading 0.70% error for 3.23% on exactly them, to save
0.8% of bytes, is the trade backwards.

*A note on measuring it.* The first mxfp8 number this tool produced was 16.1%,
which would have made the format look catastrophic rather than merely a bad
trade. That was the harness, not the format: the shared scale was taken as
`floor(log2(max / 448))`, which puts every group's largest element at or past
the clip. The OCP rule is `floor(log2(max)) - emax_elem`, with `emax_elem = 8`
for E4M3. An independent check -- a bare E4M3 round trip with no shared scale
at all -- lands at 2.65% against the ~3.6% three mantissa bits predict, which
is what says the corrected path is sound.

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

## Phase 5 -- a probe set with power -- **DONE**

Phase 4 judged five artifacts on eight contexts and 6620 tokens of
perplexity, and then it turned out that **one** of the eight was carrying the
verdict: `Дважды два равно` spread the variants over a factor of ten, while
on the other seven all four sat within 0.1 nats of the control and two within
0.001. The conclusions were right; the instrument behind them was one sample
wide. Phase 5 replaces the instrument and re-runs every Phase 4 judgement
through it.

`tools/probe_set.py` holds 64 contexts in eight groups, `tools/dump_probes.py`
captures their full next-token distributions from one artifact, and
`tools/compare_probes.py` compares captures. It is a **second** set, not a
wider first one: `dump_artifact_baseline.py`'s `CONTEXTS` stays frozen because
`baseline/logprobs.bin` -- the Phase 2 check that the rebuilt trunk is
bit-identical to the deleted artifact -- is eight distributions over exactly
those eight strings, and it cannot be regenerated. The probes live beside it
in `probes.bin` and touch nothing.

    cd tools && ORNITH_PROBES_OUT=../refs/artifact/<capture> \
      PYTHONPATH=../../vendor/mlx-lm python3 dump_probes.py <artifact>
    python3 tools/compare_probes.py

### What the 64 say, against the four Phase 4 variants

60 of 64 probes are live -- some variant moves the distribution by at least
0.001 nats. KL(control || variant) over those, and the count of probes each
variant is *least* damaged on:

| variant | median | p90 | max | best on | worst on |
|---|---:|---:|---:|---:|---:|
| `int4-head-linattn` | **0.0272** | 0.1953 | 0.5029 | **28** of 60 | 6 of 60 |
| `fp8scale-twin` | 0.0340 | 0.2612 | 0.4890 | 14 of 60 | 11 of 60 |
| `int4-except-gates` | 0.0549 | 0.2462 | 1.5938 | 8 of 60 | 13 of 60 |
| `nvfp4-head-linattn` | 0.0649 | 0.3172 | 6.6391 | 10 of 60 | 30 of 60 |

**The Phase 4 ordering holds, and for the first time it is significant rather
than anecdotal.** A sign test over the live probes, row lower than column:

| | vs `int4-head-linattn` | vs `fp8scale-twin` | vs `int4-except-gates` |
|---|---|---|---|
| `int4-head-linattn` | -- | 40/60, p = 0.014 | 44/60, p = 0.0004 |
| `fp8scale-twin` | 20/60 | -- | 31/60, p = 0.90 |
| `nvfp4-head-linattn` | 17/60 | 22/60, p = 0.052 | 21/60, p = 0.027 |

Three results change:

1. **"By median `int4-except-gates` ties `int4-head-linattn`" is refuted.**
   That was a tie over eight contexts. Over sixty it loses 44 of 60,
   p = 0.0004, at 2.0x the median KL. The knee is where Phase 4 put it, and
   now it is a measurement and not a maximum.
2. **The magnitudes were badly inflated by the single context.** Phase 4 read
   the fp8-scale twin as paying **9.2x** the KL of `int4-head-linattn`
   (0.443 against 0.048). Over 64 probes the geometric-mean ratio is
   **1.46x**, median 1.43x, and the twin is the *better* of the two rejected
   4.50-bit routes rather than the worse. `int4-head-linattn` still wins, by
   a margin an order of magnitude smaller than reported.
3. **`nvfp4` is worse than Phase 4 could see**, not better: its max goes from
   0.429 to **6.64 nats**, on `arith/ru-7x8`, where it abandons the
   arithmetic continuation entirely and predicts a paragraph break. Its
   rejection stands on much firmer ground.

And one thing the old eight had no probe for at all: **two variants lose an
in-context recall outright**. On `longrange/russian-decl` -- a name stated
once and asked for ~350 tokens later -- `int4-except-gates` and `nvfp4` both
answer the wrong name (` Ю` for ` Чел`), at 1.59 and 2.19 nats, while
`int4-head-linattn` and the twin keep it. That is a behaviour change, not a
distribution wobble, and no amount of perplexity would have shown it.

### Where damage lands, and what to probe next time

Median KL by group, across the four variants:

| group | range | |
|---|---|---|
| `arith` | 0.089 - 0.288 | forced numbers, non-English -- the strongest family |
| `morph` | 0.048 - 0.109 | agreement-driven inflection |
| `longrange` | 0.056 - 0.081 | in-context recall; hides the worst single failures |
| `script` | 0.027 - 0.114 | non-Latin, forced factual |
| `code` | 0.012 - 0.086 | |
| `rare` | 0.008 - 0.040 | |
| `mixed` | 0.020 - 0.035 | |
| `format` | 0.012 - 0.024 | syntax-locked and nearly free; 3 of the 4 dead probes |

`Дважды два равно` was **representative, not a fluke** -- `arith` is the
highest-damage group for three of the four variants, so the family it belongs
to is real. What it was not is uniquely powerful: it ranks 12th of 64 by
spread, and eleven probes separate the variants harder.

### The instrument's own lesson

This set was built on a theory that the captures then refuted, which is worth
recording because the theory is the intuitive one. It assumed damage shows
where the model is *confident by a narrow path*, so probes were chosen to be
low-entropy. Both halves are wrong:

- Entropy correlates **positively** with KL (+0.55 over the old eight). An
  entropy gate at 2 nats -- which the first version of `compare_probes.py`
  had -- would have dropped `Дважды два равно` itself. Against a near-delta
  distribution there is nowhere for mass to go.
- Entropy does not select for discrimination either. `The Rust borrow checker
  prevents` has the highest entropy of the old eight and cannot rank the
  variants at all; every one of them lands within 2.4x.

The property that matters is **spread across variants**, and it is not
visible in the text -- it has to be measured, which is why `compare_probes.py`
ranks probes by it and why liveness is a floor on observed KL rather than a
guess about the prompt. The set survived its own bad theory because breadth
was the real requirement.

---

## Phase 6 -- activation-aware scaling, built and it does not pay -- **DONE**

`IDEAS.md` ranked this first of everything left, and the ranking was sound on
its own terms: it needs no new Metal, it produces an ordinary affine artifact
so the MLX oracle survives, and it attacks *output* error, the axis nothing in
Phase 4 touched. Built, measured on Phase 5's instrument, and it is a trade
rather than a win.

`W diag(s)` quantized against `x diag(1/s)` is the same product, so a channel
the activations actually use can be given more of the 4-bit grid for free --
provided something upstream absorbs `1/s`. Three tools:
`tools/dump_awq_stats.py` (per-channel input statistics and sampled input
vectors over a calibration corpus), `tools/awq_search.py` (choose `s`), and
`tools/forge_awq.py` (apply it).

### The fold point decided the scope, not the statistics

Three families were candidates. Only one survived, and activation spread had
nothing to do with which:

| family | spread, max/median | folds into | outcome |
|---|---:|---|---|
| `in_proj_qkv`, `in_proj_z` | 20-57x | `input_layernorm` | **built**, 28.3% of traffic |
| `out_proj` | **1319x** | `linear_attn.norm` | unreachable |
| `lm_head` | 9.4x | `model.norm` | unreachable |

`out_proj` has by far the most saliency to exploit and none of it is
reachable: its only fold point is `linear_attn.norm`, which is per-head-dim
and shared across all 32 value heads, while the outliers sit at individual
(head, dim) pairs -- taking out the per-dim mean leaves 98-114% of the
log-variance behind. The gate is the other candidate and it is nonlinear, so
nothing folds through it.

`lm_head` folds into `model.norm`, which also feeds the MTP head through
`pre_fc_norm_hidden` -- and an RMSNorm cannot absorb a per-channel rescale,
because its own denominator moves with the channel pattern. The search
independently priced `lm_head` at +1.5%, the smallest of the three, so
nothing was lost.

**The generalisation: spread is necessary and not sufficient.** What decides
is whether the preceding op is indexed the same way the saliency is. The dead
family here has 140x the spread of the live one.

### Powers of two, so that nothing but the quantizer moves

Every fold is a bf16 tensor divided or multiplied by `s`: the norm gain by
`1/s`, and the two bf16 recurrence projections `in_proj_a`/`in_proj_b`, which
read the same tensor, by `s`. At an arbitrary scale each of those rounds --
~0.2% of uncontrolled relative error applied per channel to a whole channel,
which is the shape of error that killed nvfp4, one order down. Restricted to
powers of two every fold is exact, asserted tensor by tensor in
`forge_awq.py`. The search prices the resolution this costs: **+26.2% mean
output-error reduction against +32.6% unrestricted.**

Two checks make the experiment interpretable. `alpha = 0` reproduces
`int4-head-linattn` bit for bit -- weight, scales and biases -- so the scaling
is the only variable. And the port's `full_forward_matches_mlx` passes on the
forged artifact at `max_abs == 0.0` through all 40 layers, which is the
"keeps the oracle" half of the idea, and it held exactly as claimed.

### Two artifacts, and perplexity disagrees with the distribution

Both carry **2.0761 GB a token, identical to the base** -- same format, same
modules, same group size. This is a pure quality intervention at zero cost.

| | calibration | ppl | median KL | p90 | max | vs base |
|---|---|---:|---:|---:|---:|---|
| `int4-head-linattn` | -- | 6.5022 | **0.0268** | 0.1928 | 0.5029 | -- |
| `awq-linattn` | `ppl_corpus.txt`, all Latin | 6.4670 | 0.0300 | **0.1771** | 0.8389 | 27/60, p = 0.52 |
| `awq-ml` | held-out, 36% non-Latin | **6.4512** | 0.0329 | 0.2374 | 0.4181 | 25/60, p = 0.25 |

**Perplexity improves and the distribution does not.** The first run's
perplexity gain was in-sample -- it calibrated on `ppl_corpus.txt` and was
then scored on it -- which is why `tools/awq_calib.txt` exists: 10 KB of
held-out text sharing no line with the perplexity corpus and 36% non-Latin by
letter. On that corpus perplexity improves *more* (6.4512, 30% of the gap
back to the control) and out of sample. Meanwhile the 61 live probes move the
other way: geometric-mean KL ratios of 1.11x and 1.16x against the base,
neither significant.

### Where it goes wrong, and why the multilingual corpus did not fix it

The loss is concentrated in `arith`, the group Phase 5 measured as carrying
the most signal:

| | `int4-head-linattn` | `awq-linattn` | `awq-ml` |
|---|---:|---:|---:|
| median KL, `arith` (8) | 0.0887 | 0.1634 | 0.2133 |
| median KL, other 53 | 0.0228 | 0.0242 | 0.0298 |
| `arith/ru-2x2` | 0.0507 | 0.1623 | 0.3643 |
| `arith/ru-7x8` | 0.1361 | 0.2131 | 0.4181 |

Excluding `arith` the experiment is a wash (24/52, p = 0.68 and 23/52,
p = 0.49). Including it, AWQ makes Russian arithmetic 3-7x worse, and the
multilingual calibration made it **worse again**, monotonically, which is the
opposite of the hypothesis that motivated it.

The reason is visible in the scales rather than the outputs: recalibrating on
a corpus with 11.8% Cyrillic kept the same alpha on **21 of 30 layers** and
moved **1.9% of channels** (max 3.3%). The saliency map `mean|x|` produces is
substantially language-independent, so a calibration set cannot reach what is
going wrong. The channels that carry a rare forced continuation are not the
channels with large mean magnitude, and mean magnitude is the whole of AWQ's
saliency model.

### The verdict, and why the continuous variant was not built

**Not adopted.** Identical bytes, no measurable distribution gain, and a
concentrated regression on the probes with the most power. `int4-head-linattn`
remains the artifact.

The unrestricted-scale family is the obvious next thing to try, and it is
deliberately not built. The only evidence for it is +6 points of calibration
output error over the power-of-two family -- and calibration output error is
precisely the predictor this phase just falsified, in the same way weight
error was falsified in Phase 4 and perplexity in Phase 5. Buying it would
cost the exactness of every fold. **That is four predictors of the
distribution now measured and all four found wanting: weight error,
perplexity, calibration output error, and activation magnitude itself.**

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
  Phase 2 by `logprobs.bin` -- *not* by the shas, which cannot match on this
  toolchain at all. Not caught later by anything.
- **A sanitization difference.** Caught only by the explicit `conv1d` shape
  check in `check_rebuild.py`; the golden tests cannot see it, because MLX
  reads the same artifact and makes the same mistake. This is the failure mode
  to fear.
- **A re-serialised tokenizer.** Found in Phase 2 and it is the same shape as
  the one above: mlx-lm's `save()` writes the tokenizer back out through
  transformers, which silently swaps the artifact's own Split pattern for the
  older one without `\p{M}`, flips three ByteLevel flags, and drops
  `vocab.json` and `merges.txt` entirely. `dump_tokenizer.py` reads the file
  the port reads, so every tokenizer test stays green on the wrong pattern.
  Caught by the three tokenizer invariants in `check_rebuild.py`, fixed by
  `tools/align_rebuild.py`, and it was not on this list before.
- **A quality regression from the format.** Caught by perplexity against the
  Phase 0 baseline -- which is why Phase 0 cannot be skipped.
- **A port bug in the new kernels.** Caught by `max_abs == 0.0`, as usual.
  This is the one failure the existing machinery already handles.
