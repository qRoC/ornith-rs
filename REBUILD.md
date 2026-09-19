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

## Phase 7 -- the rounding, which is the last degree of freedom -- **DONE; VERDICT WEAKENED BY PHASE 8**

> **Read Phase 8 before this section.** Everything below is measured against
> the control artifact `A`, because at the time that was the only reference
> there was. The bf16 parent has since been captured
> (`tools/dump_parent_probes.py`), and against it this phase's headline does
> not hold at its stated strength: the sign test goes from 38/58, p = 0.025 to
> **31/57, p = 0.60**, and the median per-probe ratio from 1.59x to **1.10x**.
> The artifact is still not worse and is kept. The claim that it *pays*, at
> that margin, was partly an artifact of the reference. Phase 8 blamed the
> Hessians having been captured on `A`; **Phase 9 refuted that** and found
> the real cause -- two draws of this method disagree with each other by
> 56% of their own distance from the parent, so this phase and the next
> were each reading one sample. The group table below is noise.

Phase 4 closed the **format**: mlx has exactly four modes, affine int4 is the
answer, and the best non-mlx 4.50-bit format was built as a stand-in and lost
by the same margin. Phase 5 closed the **coverage**: the knee is
`int4-head-linattn`, 44 of 60 probes, p = 0.0004. Phase 6 closed the
**scaling**: activation-aware, built, keeps the oracle, buys perplexity and no
KL.

Between them those three exhaust everything about the artifact *except which
integer each weight rounds to*. Round-to-nearest picks the closest grid point
for every weight independently. GPTQ (Frantar et al., arXiv:2210.17323) picks
a point that costs the layer's **output** less and pays for the deviation by
pushing the error into the columns it has not reached yet, along the direction
the inverse Hessian of the calibration input says is cheapest. The output is
an ordinary affine artifact: same bits, same group size, same three tensors,
same shards, same kernels.

**It is the first lever since the MTP fix that pays.** Median KL over the 58
live probes falls 0.0285 -> 0.0187, a factor of **1.52**; the median of the
per-probe ratios is **1.59x**. Those are two statistics, not one.

| | `int4-head-linattn` | `gptq-head-linattn` |
|---|---:|---:|
| GB a token | 2.0761 | **2.0761**, line for line |
| dispatch profile | 1078 epochs, 37 kernels | **identical**, count for count |
| perplexity (6620 tokens) | 6.5022 | **6.4238** |
| max KL, the frozen eight | 0.048 | **0.0417** |
| **median KL, 58 live probes** | 0.0285 | **0.0187** |
| p90 KL | 0.2003 | **0.0985** |
| max KL | **0.5029** | 0.9229 |
| least damaged on | 20 of 58 | **38 of 58** |
| relative weight error | **8.1-8.7%** | 9.3-12.2% |
| calibration output error | -- | **-33.2% mean** (-76.8% best, -20.7% worst) |
| `full_forward_matches_mlx` | `max_abs == 0.0` | `max_abs == 0.0` |

Head to head it is lower on **38 of 58** live probes, median ratio **1.59x**,
sign test p = 0.025 two-sided. That is a weaker p than Phase 5's 0.0004 and
the effect is larger; both facts are the same fact, because this comparison
has two variants where that one had five.

### What it cost, because it is not free

    group       n  gptq-head-linattn  int4-head-linattn
    arith       8             0.0699             0.0887
    script      8             0.0157             0.0379
    morph       8             0.0540             0.0481   <- worse
    longrange   7             0.0512             0.1059
    format      6             0.0098             0.0119
    rare        8             0.0152             0.0083   <- worse
    code        6             0.0278             0.0747
    mixed       7             0.0135             0.0203

It wins the four groups Phase 5 measured as carrying the signal -- `arith`
most of all -- and **loses `rare` by 1.8x and `morph` slightly**. The single
worst probe in the whole set is now GPTQ's: `code/metal-kernel` goes 0.5029 ->
**0.9229**, and `rare/cauchy` 0.0034 -> 0.0632.

The obvious suspect is the corpus, and it is named rather than cleared:
**624 samples, 254028 tokens** -- omlx's built-in calibration set (code 200,
en 150, ko 60, zh 50, ja 60, tool_calling 40, reasoning 40) plus the 24
paragraphs of `awq_calib.txt`. That mix has **six Cyrillic characters in
744k**, so nearly all the Russian in the calibration comes from this project's
own file. A rare-token probe is exactly what an underrepresented tail looks
like when `lm_head`'s rounding is fitted to a distribution that does not
contain it. Untested; the corpus is the first thing to move if anyone wants
the `rare` row back.

### The finding that outlasts the artifact: weight error ranks *backwards* here

Phase 4 established that `tools/weight_error.py` ranks formats only within a
fixed coverage, by order and never by magnitude. This is a sharper result and
it comes from the one direction Phase 4 could not produce:

**Coverage is fixed, the format is fixed, weight error went up by 13%, and the
KL went down by 1.6x.** Every variant before this one moved both currencies
the same way, so "more weight error, more damage" survived Phase 5 as a
weakened but intact ordering. It does not survive a variant built to trade one
against the other. `weight_error.py` cannot rank a *rounding* change at all,
not even by order.

And the objective GPTQ actually minimises is `||(W - Wq) X^T||` -- **the
calibration output error Phase 6 measured and rejected**. Phase 6 bought six
points of it and no KL; this bought 33 and 1.6x. So the currency was not the
problem in Phase 6; the **reachable set** was. AWQ had a fold point -- the
inverse of every scale had to be absorbed by a preceding op, which put
`out_proj` and `lm_head` out of reach and left one family of three. GPTQ folds
nothing into anything, so all 91 modules are reachable, including the two AWQ
could not touch. **A predictor that fails under a constrained search is not
thereby falsified; it may be the constraint that failed.** That is the fifth
predictor and the first one to survive.

### The control, which is the reason any of this can be believed

`forge_gptq.py --off` runs the identical code path with the compensation term
zeroed and asserts the result **is the base artifact's own tensors -- codes,
scales and biases, all 91 modules, bit for bit**. It passes. So "the only
variable is the rounding" is a checked statement, not a description of intent;
it is the same role `alpha = 0` played in Phase 6.

Two more checks behind it:

- `gptq.py`'s grid is `aq.quantize`'s grid, verified against both `aq.quantize`
  and the base artifact's stored tensors on a real module before anything was
  built.
- `dump_gptq_stats.py` asserts that `in_proj_qkv` and `in_proj_z` see the
  **identical** input tensor before letting one Hessian serve both -- 30
  layers, 4 calls each, `max_abs == 0.0`. `dump_awq_stats.py` asserted the same
  thing in prose.

### What was built

    tools/dump_gptq_stats.py   H = sum x x^T for the 61 input groups, 2.36 GB
                               fp32, 254028 tokens, 272 s. Not the 256 `rows`
                               `dump_awq_stats.py` keeps -- that is a rank-256
                               estimate of a matrix about to be inverted at
                               dimension 4096.
    tools/gptq.py              the algorithm, in numpy, over aq.py's exact grid
    tools/forge_gptq.py        the forge and the --off control, 142 s for 91
                               modules
    tools/ref.py               now honours ORNITH_REFS, which the Rust side has
                               read since the port began

### What is still open, and the big one is not small

- **The experts were not touched.** These 91 modules are 57% of a decode
  step's *traffic* but a small share of its *parameters*: the 256 experts of
  40 layers are 90%+ of the model and are quantized by round-to-nearest, as
  they have been since the parent. They need a routed-activation capture (each
  expert sees only the tokens routed to it), which is the one piece of
  machinery this phase did not need. omlx's own documentation describes a
  batched expert GPTQ that shares one Hessian across a layer's experts,
  because they all read the same hidden states; the **implementation is not in
  the vendored copy** (0.7.0.dev3 ships oQe, an imatrix weighting, not GPTQ),
  so `docs/oQ_Quantization.md` there describes a version this tree does not
  have.
- **No act-order.** Quantizing columns in order of decreasing `diag(H)` is the
  standard companion trick and needs a `g_idx` per column, which mlx's affine
  layout has no room for. Permuting the weight instead needs the input
  permuted, and the input comes out of an RMSNorm shared with modules that are
  not being touched -- the same fold-point wall Phase 6 hit, from the other
  side.
- **The corpus has no Russian**, and `rare` regressed. See above.
- **Damping was not searched.** `percdamp` is 0.01, the literature's default,
  and `ORNITH_GPTQ_DAMP` exists so that it can be moved. Nothing here says
  0.01 is right for this model.

---

## Phase 8 -- the bf16 parent, measured at last -- **DONE, AND IT MOVES A VERDICT**

Every KL in Phases 4 to 7 is `KL(A || variant)`, where the control `A` is
itself a quantized artifact: 260 modules at 8 bits. The unquantized parent was
never measured, and this file and `recipes/README.md` both carried the row as
missing, for a stated reason -- 67.0 GiB of weights against a 51.8 GiB working
set, so `mlx_lm.load` dies.

**That reason was about residency, not about the forward.** A layer only has to
be resident while it runs. `tools/dump_parent_probes.py` inverts the loop:
instead of walking 40 layers for one probe, it walks 64 probes for one layer,
then drops the layer and loads the next.

    heaviest layer of 40   3.13 GiB        measured peak   1.73 GiB
    lm_head / embed        0.95 GiB each   working set    51.8 GiB
    whole 64-probe capture                 **24 seconds**

Twenty-four seconds, against a row that stood open for four phases. The
`--check` mode exists because the assumption the whole thing rests on --
that a dropped layer is actually released -- is worth testing before spending
a pass: active memory is **0.00 GiB between layers**, flat.

**The capture is sane, and one number says so.** With the parent as control,
the least quantized artifact has to come out nearest it. It does: `new` (`A`
itself, 260 modules at 8 bits) is the least damaged on **34 of 61** live
probes at a median of 0.0179, well clear of everything forged from it.

### What it says about Phase 7

    KL(parent || variant), 57 live probes      gptq      int4-head-linattn
    median                                   0.0303          0.0427
    p90                                      0.2510          0.2076   <- worse
    max                                      1.0122          0.7394   <- worse
    least damaged on                        31 of 57        26 of 57
    sign test                          31/57, p = 0.60  (was 38/58, p = 0.025)
    median per-probe ratio                    1.10x         (was 1.59x)

**The direction survives and the margin does not.** GPTQ is still ahead on the
median and is still the least damaged of the forged variants -- but probe by
probe the two are not separable, and GPTQ's *tail* is worse than the artifact
it replaces. Phase 7's "it pays, 1.52x" was measured from `A` and reads
differently from the parent.

Two group-level conclusions from Phase 7 were wrong, **in opposite directions**:

    group      vs A: gptq / base     vs parent: gptq / base
    arith       0.0699 / 0.0887       0.1771 / 0.1526   <- looked best, is worst
    rare        0.0152 / 0.0083       0.0239 / 0.0231   <- looked worst, is a tie
    script      0.0157 / 0.0379       0.0217 / 0.0438   holds
    longrange   0.0512 / 0.1059       0.0472 / 0.1077   holds

So the `rare` regression this file spent a paragraph explaining, and blamed on
six Cyrillic characters in 744k, **was not a regression at all** -- it was the
reference. And `arith`, the group Phase 5 measured as carrying the most signal,
inverts: GPTQ is further from the parent there than the artifact it replaces.

### The mechanism, and it is a mistake worth writing down

`dump_gptq_stats.py` captured the Hessians **from `A`**, because that was the
best model that would load. GPTQ minimises `||(W - Wq) X^T||` over exactly
those `X`. So it does not reconstruct the parent's layer outputs -- it
reconstructs **`A`'s**, including `A`'s own damage, and then it was scored on
its distance from `A`. Part of the Phase 7 margin was the measurement agreeing
with itself.

**Calibrate on the thing you are trying to reproduce, and score against it
too.** The streaming path makes both possible now and neither was before, so
this is a correction the earlier phase could not have made.

> **REFUTED BY PHASE 9.** The mechanism above was tested by re-capturing the
> Hessians from the parent and re-forging, and it does not hold: the two
> Hessians differ by **0.58%** and the parent-Hessian variant comes out
> *slightly worse*, 27 of 58, p = 0.69. What Phase 9 found instead is that
> GPTQ here is a **high-variance procedure**, so Phase 7 and Phase 8 were each
> reading a single draw. Read Phase 9 before building on anything in this
> section, including its group table.

### What this opens

- **Re-capture the Hessians from the parent and re-forge.** Same streaming
  harness, same `gptq.py`, same 91 modules. This is the direct fix for the
  mechanism above and it is now cheap.
- **`A` is not uniformly the nearest thing to the parent.** On `arith`,
  `awq-linattn` (0.0976), `fp8scale-twin` (0.0988) and `int4-except-gates`
  (0.1256) all sit closer to the parent than `A` does (0.1376). Treat that as
  a caution rather than a finding -- eight probes to a group makes a group
  median noisy -- but the `A`-referenced instrument could not see it at all.
- **Every earlier capture can be re-scored for nothing.** The eight
  `refs/artifact/*/probes.bin` are log-probabilities; the parent's is one more
  file. `compare_probes.py` now takes any capture as the control and excludes
  only that one from the variant list, so Phases 4, 5, 6 and 7 all re-rank
  against the parent without a rebuild.
- **The parent's perplexity is still missing**, and it is the same pass: keep
  hidden states for the 6620-token corpus and run `norm` + `lm_head` per chunk
  at the end. Not done here.

### The trap this had to clear

The parent is an HF checkpoint carrying `mtp.*` and a torch-layout `conv1d`,
so `qwen3_5.TextModel.sanitize` **adds 1.0 to the trunk norm gains**. Streaming
sanitizes per layer instead of once over the whole dict, so
`has_unsanitized_conv1d` is computed globally from the full key list before any
layer is touched, and asserted. A second assertion checks the result rather
than the intent: layer 0's mean `|gain|` is **1.0312**, which is an absolute
gain; a delta-encoded one sits near 0.0 and would have produced a model that
loads and is wrong. That is the double shift, from the side that is easy to
miss.

---

## Phase 9 -- the same method twice, and it disagrees with itself -- **DONE**

Phase 8 blamed Phase 7's inflated margin on the Hessians having been captured
from `A` rather than from the parent. Phase 8's own streaming path makes that
testable, so it was tested: `tools/dump_gptq_stats_parent.py` accumulates the
same 61 Hessians over the same 254028-token corpus with `X` taken from the
parent, and `forge_gptq.py` re-forges from them.

**The two calibrations are the same calibration.**

    ||H_parent - H_A|| / ||H_A||        0.58%  (in_proj_qkv)   0.12%  (out_proj)
    off-diagonal correlation          0.999968               0.999998
    GPTQ objective, mean improvement    -33.20%   against  -33.22%

**And the two artifacts are not the same artifact.** A 0.58% perturbation of
`H` flips **9.9%** of the codes by a mean of 1.8 grid levels, because the
compensation is sequential -- every column's error is pushed into all the
columns after it, so a microscopic difference in direction compounds over a
2048-column walk.

    KL(parent || variant), 58 live probes    median      p90      max   best on
    gptq  (A Hessians)                       0.0300   0.2501   1.0122  22 of 58
    gptq  (parent Hessians)                  0.0354   0.1788   1.4685  18 of 58
    int4-head-linattn  (round-to-nearest)    0.0422   0.2040   0.7394  18 of 58

    parent-H vs A-H, head to head: 27 of 58, p = 0.69

**The number that matters is not in that table.** The median KL *between the
two GPTQ variants* is **0.0196** against their median distance from the parent
of 0.0348 -- **0.56x**. Two artifacts produced by the same method, from
calibrations that agree to three decimal places, are **more than half as
different from each other as either is from the truth**.

In the currency the phases have been arguing in: the claimed GPTQ-over-RTN
effect is 0.0122 nats of median KL, and the spread between two draws of GPTQ
is 0.0054 -- **44% of it**.

### What that retires

- **Phase 8's mechanism is refuted.** The Hessians were never the problem;
  `A` and the parent produce inputs whose second moments agree to a fraction of
  a percent.
- **Both phases' group tables are noise.** Phase 7 blamed a `rare` regression
  on six Cyrillic characters in 744k; Phase 8 blamed an `arith` inversion on
  the reference. Neither survives a second draw: between these two runs
  `longrange` moves 0.0472 -> 0.1397 (3x), `arith` 0.1771 -> 0.1342, `script`
  0.0217 -> 0.0350. Those are two samples of one procedure, not two findings.
- **The tail was never a property of GPTQ either.** Phase 8 recorded that
  GPTQ's p90 and max come out worse than RTN's. On the second draw the p90 is
  *better* than RTN's (0.1788 against 0.2040) and the max is much worse
  (1.4685). Erratic in both directions is what the variance predicts.

### What survives

**A systematic gain, smaller than claimed.** Both draws beat round-to-nearest
on the median -- 0.0300 and 0.0354 against 0.0422 -- so the direction is real
and it is not a draw artifact. Call it **~0.008 nats, roughly 20%**, not the
29% one draw showed and not the 1.52x Phase 7 announced against the wrong
reference. Both variants remain byte-identical at 2.0761 GB a token with an
identical dispatch profile, so a ~20% median gain for nothing is still worth
having; it is the *confidence* that was never there.

### The lesson, and it is this project's recurring one

Phase 5 found a verdict resting on one context of eight. Phase 7's
[[speculation]] correction found a verdict resting on a stale measurement.
This one rests on **one draw of a stochastic procedure**, and the procedure
was never examined for being stochastic -- it is deterministic given `H`, which
is not the same thing as stable in `H`.

### What to do instead, and the knob is already there

The variance is not in the Hessian estimate -- that is stable to 0.58% across
two very different models -- it is in the algorithm's response to it. More
calibration data will not fix that. **Damping will**: `percdamp` controls how
aggressive the compensation is, it is at the literature default of 0.01, and
`ORNITH_GPTQ_DAMP` has existed since Phase 7 without ever being searched.
Phase 7 listed that as open and had no reason to care; now there is one.

Anything measured about GPTQ here from now on needs **several draws**, not one.
Two forges cost 280 s and a probe capture each, so this is affordable and was
simply never done.

---

## Phase 10 -- the damping sweep -- **DONE; ITS HEADLINE DOES NOT REPLICATE**

> **Read Phase 11 first.** This phase concluded that `percdamp = 0.03` beats
> the default, on two draws a point that did not overlap. With **four** draws a
> point they overlap completely (Mann-Whitney p = 0.56) and the optimum is not
> there. What survives is the finding this phase was not looking for: GPTQ
> beats round-to-nearest by **23%**, 14 of 14 forges, p = 6e-05. The method and
> the tooling below are sound; the ranking of damping values is not.

Phase 9 left GPTQ with a real effect and a noise floor half its size, and named
the only available knob: `percdamp` adds `lambda * mean(diag H)` to the
diagonal, so it interpolates continuously between full GPTQ (chaotic) and
round-to-nearest (deterministic, no gain). Act-order needs a `g_idx` the affine
layout has no room for and the block already equals the group; there is nothing
else to turn.

**The design uses Phase 9's negative result as its control.** The `A` Hessians
and the parent's differ by nothing systematic (27 of 58, p = 0.69), so forging
from each at the same damping gives two draws of one procedure -- a replicate,
not a second effect. Four damping values, two draws each; `percdamp = 0.01` was
already on disk from Phases 7 and 9.

    59 live probes, reference = bf16 parent, RTN = 0.0418, control A = 0.0185

    percdamp   draw A-H   draw P-H     mean   spread   vs RTN   spread/effect
       0.003     0.0355     0.0352   0.0354   0.0003  +0.0064          0.04x
        0.01     0.0297     0.0343   0.0320   0.0046  +0.0098          0.47x
     -> 0.03     0.0285     0.0293   0.0289   0.0008  +0.0129          0.06x
        0.10     0.0394     0.0419   0.0406   0.0025  +0.0012          2.14x

**`percdamp = 0.03` is better than the default on both axes at once.** The
median gain over round-to-nearest goes from **23% to 31%**, and the two draws
do not overlap with the two at 0.01 -- max(0.03) = 0.0293 against
min(0.01) = 0.0297. `p90` improves too, 0.1801 and 0.1673 against RTN's 0.2005,
where at 0.01 one draw was 0.2491. Still byte-identical, still the same kernels.

At the other end 0.10 does what the mechanism predicts: 0.0406 against RTN's
0.0418, effect +0.0012, noise twice the signal. The curve is a U with its
minimum somewhere between 0.01 and 0.10, and 0.03 is the best point tested.

### Two things this does not say

**The artifacts did not become stable; the statistic did.** KL between the two
draws at 0.03 is 0.0190 -- **0.66x their own distance from the parent**, against
0.58x at 0.01 and 0.50x at 0.10. Two very different sets of rounding decisions
arrive at the same aggregate quality. That is enough to decide whether GPTQ
beats RTN and says nothing about which artifact to keep, where it happens not
to matter because both are equally good.

**Two replicates is a spread estimate with one degree of freedom.** The spread
column is nearly worthless on its own -- 0.0003 at `percdamp` 0.003 is most
likely luck, and that same point has the *largest* disagreement between its
artifacts (0.81x). What carries the result is the non-overlap of two pairs and
the coherent shape of the curve, not that column.

### Tooling

`ORNITH_FORGE_LINK=1` hard-links the shards a forge does not rewrite instead of
copying them -- six builds at 26 GB each moved the disk by 3 GB total. It is
opt-in and only for sweeps: a hard link means editing one artifact edits the
other, which is wrong for anything kept. Captures live in `refs/sweep/` rather
than `refs/artifact/` so they stay out of `compare_probes.py`'s default variant
list; `tools/sweep_report.py` reads them. The artifacts themselves were deleted
after each capture, which is the whole point of `probes.bin` being the product.

### Open

- **More replicates.** `H` is additive over samples, so one streaming pass with
  four accumulators (samples modulo 4) yields four independent Hessians in the
  same ~290 s. Four draws at 0.01 and 0.03 would make the spread column mean
  something instead of decorating.
- **`percdamp = 0.05`**, to bracket the minimum.
- **The experts**, still 256 x 40 layers at round-to-nearest, still 90%+ of the
  parameters. This phase is what that job was waiting on, and the answer is
  now "probably yes, after the replicates".

---

## Phase 11 -- four draws instead of two, and the optimum evaporates -- **DONE**

Phase 10 ranked damping values on two draws each and found `percdamp = 0.03`
better than the default, on the strength of two pairs not overlapping. Phase 10
also wrote down why that was weak: two replicates is a spread estimate with one
degree of freedom. So the replicates were made properly.

`H` is additive over samples, so `ORNITH_GPTQ_SPLIT=4` keeps four accumulators
and routes sample *s* to *s* mod 4 -- **four independent calibrations in one
streaming pass**, whose sum is the single one. Each is a quarter of the data, so
each Hessian is noisier than production and the spread they show is an **upper
bound** on the spread at full corpus size. What they can settle is a ranking.

    four quarter draws each, median KL(parent || variant), 59 live probes

    percdamp 0.01   0.0277  0.0324  0.0337  0.0373     mean 0.0328
    percdamp 0.03   0.0251  0.0335  0.0338  0.0369     mean 0.0323

    separation: max(0.03) 0.0369  against  min(0.01) 0.0277  -- OVERLAP
    Mann-Whitney: 0.03 lower in 8 of 16 pairings, exact p = 0.56

**A coin flip.** Phase 10's non-overlap was luck. And `percdamp = 0.05`, forged
to bracket the minimum, lands at 0.0371 with a spread of 0.0060 -- worse than
both and its own noise is larger than its effect. There is no usable optimum in
0.003 to 0.05; at 0.10 the method degenerates to round-to-nearest as predicted,
with one of the two draws landing *above* it.

### What survives, and it is now the solid part

Pooling every forge at `percdamp <= 0.03` -- **14 independent artifacts** across
two Hessian sources, four corpus quarters and three damping values:

    median KL     0.0324 +- 0.0010 (sem)     round-to-nearest  0.0418
    effect        +0.0094 nats = **23%** of the damage RTN pays
    every draw below RTN                     14 of 14, sign test p = 6.1e-05
    per-forge sd  0.0037 = **39% of the effect**

So GPTQ's gain over round-to-nearest is **real, and it is 23%** -- byte
identical, same kernels, same dispatch profile. What the damping knob cannot do
is shrink the run-to-run spread: it sits at 39% of the effect and no value
tested moves it. Phase 9 called the variance irreducible by anything but this
knob; the knob has now been searched and it does not reduce it either.

**Keep `percdamp` at 0.01.** Not because it won -- 0.01 and 0.03 are
indistinguishable -- but because nothing argues for moving off the default.

### The lesson, for the third time in this rebuild

Phase 5 found a verdict resting on one context of eight. Phase 9 found two
phases resting on one draw each. This one found a phase that *named* its own
one-degree-of-freedom problem in writing and drew a conclusion from it anyway.
**Two samples is not a measurement**, and writing "two samples is not a
measurement" next to the conclusion does not make it one.

Anything measured about GPTQ here takes **four draws minimum**, which costs one
streaming pass plus four forges -- about 25 minutes, now scripted.

### What this settles for the experts

The gain is real at p = 6e-05, so extending GPTQ to the 256 experts of 40
layers -- 90%+ of the parameters, still round-to-nearest -- is justified on the
evidence. What it will need is the same four-draw discipline, because a single
expert forge will land anywhere in a +-39% band.

---

## Phase 12 -- the experts, 93.1% of the model, and they buy ~3.5% -- **DONE, two draws**

The 91 modules of Phases 7 to 11 are **1.52 B parameters, 3.7%** of the
artifact. The 256 experts of 40 layers are **38.65 B, 93.1%**, round-to-nearest
since the parent's own recipe, and their damage sits in the 0.0185 nats the
*control* pays -- a pool no phase has touched.

`tools/dump_gptq_experts.py` and `tools/forge_gptq_experts.py` spend the same
lever on them. One Hessian a layer shared by its 256 experts (the exact
per-expert object is 172 GB), pooled over **(token, expert) pairs** so it is
routing-weighted; `gate_proj` and `up_proj` share the 2048-dimensional one and
the identity is asserted at `max_abs == 0.0`, `down_proj` has its own at 512.

### The result

    KL(parent || variant), 58 live probes    median      p90      max   vs RTN
    RTN `int4-head-linattn`                  0.0422   0.2040   0.7394     0.0%
    91 modules GPTQ, draw A-H                0.0300   0.2501   1.0122   +29.0%
    91 modules GPTQ, draw P-H                0.0354   0.1788   1.4685   +16.2%
    experts GPTQ, draw 1                     0.0412   0.3116            +1.4%
    experts GPTQ, draw 2                     0.0395   0.2919            +5.5%
    **experts GPTQ, mean of two**            0.0403                     **+3.5%**

(Draw 1 reads -0.6% against 58 live probes and +1.4% against the 59 that draw 2
brings in; same capture, different denominator.)

Byte-identical at 2.0761 GB a token. GPTQ's own objective fell **13.26%** on
average and as much as 36% on the best `gate_proj`, and **none of it reached
the distribution**. The tail is slightly *worse*: p90 0.3175 against RTN's
0.2040.

Two draws, and they bracket the effect rather than confirm it: the spread
between them (0.0017) is **larger than the effect itself** (0.0015 against RTN).
So +3.5% is not distinguishable from zero, and both draws make the **tail
worse** -- p90 0.3116 and 0.2919 against round-to-nearest's 0.2005.

Against the 91 modules' +17.9% and +28.9%, on 3.7% of the parameters. The
expert line is closed.

### Parameter share does not predict damage share, and this phase was built on assuming it did

**3.7% of the parameters carried the damage; 93.1% carried almost none.** The
motivation written at the top of this section -- that the experts must be the
big prize because they are most of the model -- is refuted by the measurement
it motivated.

Four candidate reasons, ranked, none of them measured here:

1. **`lm_head` is the output.** 508 M of the 91 modules' 1.52 B, and its error
   *is* the logit error. Nothing downstream dilutes it; every other module's
   error passes through a norm, a residual and more layers.
2. **The linear-attention path accumulates.** `in_proj_qkv`, `in_proj_z` and
   `out_proj` feed a float32 recurrent state carried across 30 layers. An error
   injected there persists; a feed-forward error perturbs the residual stream
   once.
3. **MoE averages its experts.** A token sums 8 of 256, router-weighted. If
   their quantization errors are independent the sum suppresses them by roughly
   sqrt(8), and each expert is seen by ~3% of tokens.
4. **The shared Hessian may be the limit, not the experts.** One pooled `H` a
   layer serves 256 experts whose inputs are disjoint token subsets. That is
   the approximation this phase chose for tractability, and it is the honest
   alternative explanation: exact per-expert Hessians for one layer would
   separate "the experts do not pay" from "this way of paying them does not
   work".

### The control earned its keep again

`forge_gptq_experts.py --off` failed at layer 32, and the reason is a fact
about this artifact that nothing had written down: **the experts are not
uniform.** Layers 0-31 hold theirs at 4 bits / group 32 and **layers 32-39 at
8 bits / group 64**. The forge had assumed one rate. Fixed by reading
`config.json`'s per-tensor overrides and by generalising `aq.pack4` into
`aq.pack(q, bits, cols)`; all 120 expert tensors then reproduce the base bit
for bit, at both rates.

### The experts split in two, and only one half is reachable

On a 16-expert slice, which answers in a minute rather than forty:

    L0  gate_proj    out 0.0373 -> 0.0238   **-36.1%**
    L20 gate_proj    out 0.0587 -> 0.0483   **-17.8%**
    L0  down_proj    out 0.0731 -> 0.0729         -0.3%
    L20 down_proj    out 0.0813 -> 0.0810         -0.4%

`gate_proj` and `up_proj` behave like the 91 modules. `down_proj` does not move
at all, and the Hessians say why:

    Hessian                          off-diag   eff. rank        top-1% var
    in_proj_qkv (the -33% baseline)  0.72-0.87   9-23 / 2048      38-48%
    experts gate/up                  0.93-0.96  12-91 / 2048      21-45%
    experts down (SwiGLU output)     **0.15-0.17**  **468-490 / 512**   **2.2%**

**`down_proj` reads `silu(gate) * up`** -- an elementwise product of two
near-independent projections, which decorrelates the channels. The Hessian goes
near-isotropic, `H^-1` near-diagonal, and GPTQ has nowhere to push the error:
it degenerates to round-to-nearest *by construction*, not by accident.

No `down_proj` was in the 91 modules, so nothing had looked. It is a third of
expert parameters and a third of the forge time, and it should be excluded from
any future GPTQ scope.

### The complementarity that falls out of it

The one module GPTQ cannot touch is one AWQ **can**. `h_c = silu(W_gate x)_c *
(W_up x)_c`, and the nonlinearity is only on the gate path, so scaling channel
`c` of `h` is dividing **row `c` of `W_up`**, with `W_down diag(s)` restoring
the product. An exact fold, over ~12.9 B parameters, that Phase 6 never
considered because expert `down_proj` was never in its scope.

Untested, and it is the first thing to try if this phase's KL comes back thin.

**Tested in Phase 22 and rejected**, on the objective rather than on an
artifact: 0.3% against the pooled Hessian this phase used, 6.3% against the
per-expert saliency it could not have -- and an unconstrained oracle says 0.4%,
so the family is not the limit, the tensor is. Phase 22 also confirms this
phase's alternative explanation #4: pooling 256 experts hides an order of
magnitude of per-channel structure, and for a *diagonal* statistic the pooling
was never necessary.

---

## Phase 13 -- IDEAS #1, priced out rather than left untested -- **DONE**

`IDEAS.md`'s surviving row -- `affine g16, fp8 scale+bias, with a per-tensor
fp32` at 5.00 bits -- was the only scheme in that table that held on real
weights and had never been built. Reproduced exactly:

    scheme                                   bits   lm_head  in_proj_qkv  out_proj  switch_mlp
    affine g32, bf16 scale+bias (the body)   5.00     8.03%       7.81%     8.25%       7.94%
    IDEAS #1  g16, fp8 scale+bias + global   5.00     7.11%       7.01%     7.24%       7.05%

**And decomposed, which is new.** Two rows were added to
`tools/weight_error.py --schemes` to ask whether the win is the granularity or
the encoding:

    decomp  affine g16, bf16 scale+bias (rate-free)  6.00   6.60%  6.50%  6.71%  6.55%
    decomp  affine g32, fp8 scale+bias + global      4.50   8.31%  8.10%  8.56%  8.22%

So: g32 -> g16 with a perfect encoding buys **-1.31 points** and costs a whole
bit; paying that bit back with fp8 scales instead of bf16 returns **+0.51**.
Net at equal 5.00 bits, **-0.80 points, -10% relative**. The idea is real and
it is the best 5.00-bit scheme measured here.

### It cannot be built inside the contract

    ValueError: [quantize] The requested group size 16 is not supported.
                The supported group sizes are 32, 64, and 128.

That is `mlx/ops.cpp:5007`, an API-level validation, so **there is no MLX
oracle at g16** -- nothing to compare against. The kernel side is no friendlier:
`QuantizedBlockLoader` carries `static_assert(BCOLS <= group_size)`, which the
gemm tiles may not satisfy at 16.

**And the twin trick does not transfer.** `forge_fp8scale_twin.py` worked
because an E4M3 value times a power of two is exactly representable in bf16 --
a different *encoding* inside the *same layout*. Here the layout itself
differs: g16 needs twice as many (scale, bias) pairs as g32 and the affine
layout has nowhere to put them.

Unblocking means patching the vendored reference, rebuilding mlx from source,
instantiating `(16, 4)` in the port, and then verifying against a **modified**
mlx. The contract survives that in form and not in spirit.

### Why it is not worth it, in the currency that has failed twice

0.80 points of relative weight error, and weight error is the predictor this
rebuild has now falsified twice:

- Phase 4: `int4-except-gates` has the **lower** weight error and the **higher**
  median KL.
- Phase 7: GPTQ moved weight error **up 13%** and median KL **down 23%**.

The prize here is smaller than what GPTQ delivered while moving the same
quantity the wrong way. **Priced out, not untested** -- and the decomposition
is the part worth keeping, because it says the granularity is where the gain
lives, which is useful to anyone who later has a g16 kernel.

---

## Phase 14 -- a Q4_K-shaped grid, and the 4.50-bit space reopens -- **DONE**

Phase 4 closed the 4.50-bit space on two routes -- mlx's `nvfp4` and the best
non-mlx format then known, the fp8-scale twin -- and both lost. Phase 13 priced
out IDEAS #1 because its win lives at a group size mlx will not run. What was
never tried is the structure llama.cpp actually ships: **a sub-block's scale and
min as 6-bit integers against two super-scales per 256 weights.**

An integer has **no subnormal regime**, which is precisely what cost nvfp4 its
quality here: 99.6% of this model's group scales sit below E4M3's smallest
normal, so a format predicted at 9.5% measured 10.4-13.1%. The trade runs the
other way -- an integer has uniform *absolute* resolution against the
super-block maximum, so a sub-block far below it is coarse in relative terms.
Which failure costs more on these weights was the question.

**Priced first, then built.** `weight_error.py --schemes`:

    affine g32, bf16 scale+bias -- the body   5.00   8.03  7.81  7.90  8.25  7.94
    **Q4_K-shaped**                           4.50   8.07  7.86  7.93  8.30  7.98
    fp8-scale twin                            4.50   8.31  8.10  8.18  8.56  8.22
    Lloyd-Max g32, bf16 scale                 4.50   8.42  8.20  8.28  8.57  8.98
    nvfp4 g16, + global scale                 4.50   9.45  9.53  9.50  9.44  9.48

**And it is standable-in for the same reason the fp8-scale twin was**: the
super-scales are powers of two, so a 6-bit integer times one is exactly
representable in bfloat16, and the sub-block stays 32 wide -- mlx's own group
size, unlike IDEAS #1's 16. `aq.q4k_encode`, `tools/forge_q4k_twin.py`, 283
modules in 195 s, `check_rebuild.py` PASS, zero underflows.

    KL(parent || variant), 60 live probes   median      p90      max   vs RTN
    control A                     2.739     0.0182   0.1677   1.0479   +55.0%
    int4-head-linattn RTN         2.076     0.0404   0.1969   0.7394     0.0%
    int4-head-linattn + GPTQ      2.076     0.0295   0.2482   1.0122   +27.1%
    **Q4_K twin**        stands for 1.925   0.0403   0.2447   0.7218    +0.2%
    fp8-scale twin       stands for 1.925   0.0432   0.2395   0.6521    -7.0%
    nvfp4                          1.925    0.0521   0.4613   4.7839   -29.0%

**The same KL for 7.3% fewer bytes a decode step**, and a slightly *better*
maximum. It strictly dominates `int4-head-linattn` on the (bytes, quality)
pair, which neither earlier 4.50-bit route did. **"The 4.50-bit space is
closed" was true of the two routes tested and is not true of the space.**

### Three things this is not

- **The bytes are arithmetic, not measured.** On disk the artifact is 5.00
  bits; 4.50 is the rate of the format it stands for. Realising it needs a
  Metal kernel for a layout mlx has no mode for -- it takes `affine`, `mxfp4`,
  `nvfp4`, `mxfp8`. So this is a **pricing** result: it says the kernel is now
  worth considering, where before it was not, because before both routes lost.
- **The tail is worse**: p90 0.2447 against 0.1969.
- **Weight error happened to predict correctly this time** -- 7.86% against the
  twin's 8.10%, same order as the KL. After Phase 4 and Phase 7 both falsified
  it, treat that as a coincidence rather than a rehabilitation.

### The combination nobody has built

GPTQ is +27.1% at 2.076 GB a token; this is +0.2% at 1.925. They are orthogonal
-- one picks the grid, the other picks which point on it -- so together they
would be GPTQ's quality at Q4_K's bytes.

The obstacle is structural rather than conceptual: `gptq.py` derives a group's
scale from the weights it is about to quantize, one group at a time, while a
super-scale needs all eight of a super-block's groups at once. Either derive
the super-scale from the pre-compensation weights (an approximation) or make
the walk's block the whole 256-column super-block (exact, but reintroduces the
staleness Phase 7's "the block is the group" decision removed).

### The kernel, designed before it is written

`Q4K.md` is the plan, and it turns on one decision that is not obvious: **build
real ggml Q4_K, not the twin's variant.** The twin's power-of-two super-scales
and magnitude-plus-sign fields exist only so it could be measured without a
kernel. Dropping them buys the thing a bespoke format normally costs --
**llama.cpp becomes the oracle.** mlx has no k-quant mode, so a Q4_K kernel has
no MLX reference and `max_abs == 0.0` against MLX does not exist for it; adding
the mode to vendored mlx would only make the port agree with code from the same
hand. `ggml`'s `dequantize_row_q4_K` did not learn the format from us.

The hidden cost, which `Q4K.md` says to price before writing Metal: an artifact
in a mode mlx cannot load also cannot be captured by `dump_probes.py`, so the
measuring instrument needs a port-side equivalent.

### A bug worth recording, because the number lied convincingly

The first measurement read **18.8-25.4%** and looked like a verdict on the
format. It was `floor` where `ceil` belonged in the power-of-two super-scale:
with `floor`, a super-block's own largest field overflows its 6-bit range in up
to half of all super-blocks and gets clipped. A format that everyone else ships
successfully measuring three times worse than plain affine is a bug report, not
a finding.

---

## Phase 15 -- real Q4_K, built and measured, and the bytes are free -- **DONE**

Phase 14 measured a Q4_K-*shaped* grid as a stand-in and read the same median
KL as the 5.00-bit base for fewer bytes. `Q4K.md` said the stand-in's two
constraints were scaffolding and that dropping them buys llama.cpp as the
oracle. This is the format itself: ggml's own `quantize_q4_K` over the
parent's weights, a Metal kernel ported from ggml's own
`kernel_mul_mv_q4_K_f32_impl`, and an artifact mlx cannot load, measured by
the port because nothing else can.

**Everything was checked before anything was believed.** `check_rebuild.py`
PASSes and the 452/281 split survives; the 91 modules' weight error against
the parent is 7.14-7.76% where ggml's own measurement said 7.12-7.55%; and
`tests/q4k.rs`'s ignored case walks the forged artifact module by module and
finds the port's dequantize agreeing with ggml's, **digest for digest, on all
91**. So what follows is a property of the format, not of a loader.

### What it bought

    artifact                bytes/token    decode    median KL(parent || .)
                                           best of 4   all 64      live 58
    int4-head-linattn        2.0761 GB    102.6 t/s    0.0350      0.0422
    **q4k**                  1.9814 GB    106.8 t/s    0.0303      0.0486
    **q4k + imatrix**        1.9814 GB    104.8 t/s    0.0260      0.0305
    gptq-head-linattn        2.0761 GB         --      0.0273      0.0300

**-4.6% of a decode step, and +4% of the rate**, both measured rather than
arithmetic -- `step_bytes.py` off the headers, and four draws because two of
the twelve timing runs came back at half speed.

**Not the -7.3% Phase 14 advertised, and the difference is coverage, not
format.** The twin re-encoded all **283** of the artifact's 4-bit modules; this
cut converts the **91** that `int4-head-linattn` moved to 4 bits, because the
experts need `q4k_gather_qmv_fast` and that is a second kernel. The experts are
0.7172 GB of a step, so the remaining 2.7 points are sitting there.

### The quality, and the median is the wrong statistic here

The two medians disagree about the ordering, which is the first thing to
report rather than the last. `compare_probes.py` drops probes no variant moves
by 0.001 nats; those six are the six smallest KLs, and removing them shifts
`q4k`'s median from 0.0303 to 0.0486 while barely moving the others'. A
statistic that unstable is not the one to read a verdict off.

The rank test is stable, and it says **nothing here is distinguishable**:

    q4k         vs int4-head-linattn    lower on 32 of 64    p = 1.0
    q4k+imatrix vs int4-head-linattn    lower on 38 of 64    p = 0.17
    q4k+imatrix vs gptq-head-linattn    lower on 37 of 64    p = 0.26

So the honest verdict is **the bytes are free**: 4.6% of a decode step and 4%
of the rate for no quality difference this instrument can see. That is what a
k-quant is supposed to deliver, and it is the first format in this rebuild to
deliver it. What is *not* demonstrated is `Q4K.md`'s prize -- "GPTQ's quality
at Q4_K's bytes". The imatrix has the best point estimate of the four, at
fewer bytes than GPTQ, and 37 of 64 is a coin flip with a story attached.

### Four predictors, all of them wrong, and the last one was supposed to be right

This is the fourth time the norm has been falsified here (Phases 4, 7, 14 and
now), and it is the strongest instance, because the measure that failed was
built to be activation-aware. Against the parent, real Q4_K beats affine int4
g32 on:

  - **total weight error** -- 7.48% median against 8.45%, over all 91;
  - **weight error in every magnitude decile** -- the obvious mechanism for a
    format with a shared super-scale is that a sub-block far below its
    super-block's maximum goes coarse, so the error was split by
    `group max / super-block max`. There is no crossover: Q4_K is 0.87-0.91x
    affine's error in all ten deciles of all three modules tried;
  - **the error in the product**, on random gaussian `x`: 7.13% against 8.07%
    for `lm_head`, and the per-group bias that would make a weight error and
    an output error disagree is *lower* for Q4_K (0.123 against 0.145, where
    an unbiased quantizer sits at 1/sqrt(32) = 0.177);
  - **the H-weighted weight error**, `diag(H)` from the GPTQ calibration --
    the quantity the activations say matters. Over all 61 modules that have a
    Hessian, Q4_K is better in **59**.

Four measures, one of them the activation-aware one, all pointing the same
way, and 64 probes saying the two artifacts are level. `L0.out_proj` -- 21.62%
H-weighted against affine's 18.80% while unweighted it is 7.55% against 8.52%
-- looked like the mechanism until the other 60 modules were measured and it
turned out to be one of two exceptions.

### What the imatrix did, and it is one argument

`quantize_q4_K` takes an importance vector and `diag(H)` is one, so the
combination `Q4K.md` described as needing "no super-block coordination, no
changes to `gptq.py`" cost exactly one argument. Only 61 of the 91 modules
have a Hessian (`in_proj_z` never had one), so this is a partial application.

It moves the H-weighted error a long way -- `in_proj_qkv` 7.97% -> 6.03%,
`out_proj` 8.12% -> 6.91%, medians over 30 modules each -- and costs 0.2-0.3
points of unweighted error, which is what a weighting scheme is supposed to
do. On the probes its clearest effect is the tail: `arith/ru-7x8`, the probe
that has broken every fragile format in this rebuild, goes from **1.9555 to
0.2470 nats**, and the count of probes whose top-1 token moves falls from 10
to 6, the lowest of any variant here.

### The trap this cleared, and it is the one the file warned about

The first imatrix reading was **38589% error**. `refs/gptq-parent/hessians/`
holds the whole `H`, `[in, in]`, not its diagonal; handing the array to
`quantize_q4_K` as it loads gives the quantizer `H`'s first *row*, which has
negative entries. `Q4K.md`'s own rule -- "a format the entire llama.cpp
ecosystem ships cannot measure three times worse than plain affine" -- caught
it, for the second phase running.

### What is left

  - **`q4k_gather_qmv_fast`** for the 256-expert modules: 0.7172 GB of a step,
    the other 2.7 points of the -7.3%.
  - **`q4k_qmm_t`** for prefill. The M > 1 path here is ggml's token axis, a
    mat-vec per token, which re-reads the weights once each; it exists so
    there can be a probe capture at all.
  - **A Hessian for `in_proj_z`**, which would take the imatrix from 61 of 91
    to all of them.
  - **Q6_K on the 8-bit side**, priced in `Q4K.md`: 2.5x the weight error for
    22.8% of those bytes, which is 0.68 error-doublings a bit against the 0.96
    the 4-bit side already pays. Given this phase, price it and expect the
    probes to say nothing.

---

## Phase 16 -- the experts, and the full -7% arrives -- **DONE**

Phase 15 converted the 91 modules `int4-head-linattn` moves to 4 bits and got
-4.6% of a decode step where `Q4K.md` had advertised -7.3%. The gap was
coverage: the twin it was compared against re-encoded all 283 four-bit modules,
and the 96 **routed-expert** ones were out of the first cut because they need a
gathered kernel. This is that kernel and those modules.

`kernel_mul_mv_id` turns out to be a *wrapper* rather than a kernel -- ggml
offsets the weights by the expert and the activations by the token and calls
the matvec already ported -- so the port does the same: the body moved into an
inline `qmv_impl` and two entry points resolve their own pointers.

**Checked before believed**, as in Phase 15. `check_rebuild.py` PASSes with 187
q4k modules and the 452/281 split intact; weight error against the parent is
6.93-8.20%, median 7.33%; and the port's dequantize agrees with ggml's on all
**187 modules, digest for digest**. The kernel itself is pinned twice over
eleven routed rows whose experts repeat and whose tokens do not: 2.8e-7 against
a float64 reference, and **exactly zero** with a one-hot activation, where each
output row must be one column of one expert.

### What it bought

    artifact              bytes/token   decode      median KL, all 64 probes
                                        best of 4   median     p90      max
    int4-head-linattn      2.0761 GB    102.6 t/s   0.0350   0.1933   0.7394
    gptq-head-linattn      2.0761 GB         --     0.0273   0.2412   1.0122
    q4k          (91)      1.9814 GB    106.8 t/s   0.0303   0.2113   1.9555
    q4k-imatrix  (91)      1.9814 GB    104.8 t/s   0.0260   0.2336   1.6524
    **q4k-all   (187)**    **1.9311 GB** **107.9**  **0.0237** **0.1697** 4.6466

**-7.0% of a decode step and +5.2% of the rate**, which is the number
`Q4K.md` promised and Phase 15 could not reach. The experts are 0.6669 GB of
the step after conversion against 0.7172 before.

**And the experts cost nothing measurable in quality: 31 of 64 against
`q4k-imatrix` is a coin flip.** That is the same verdict Phase 15 reached for
the first 91 and it is the one that matters -- the bytes are free. Against the
others, still nothing significant: 36 of 64 over GPTQ (p = 0.38), 39 of 64 over
the affine int4 it replaces (p = 0.10).

By point estimate `q4k-all` is now the best artifact this rebuild has produced
on median *and* p90 *and* on the count of probes whose top-1 token moves (6 of
64, tied with `q4k-imatrix` and better than every affine variant). None of it
is significant, and the honest headline stays the one Phase 15 wrote: at
7% fewer bytes and 5% more tokens a second, the quality is where it was.

### The one probe that is not fine

    probe                    q4k-all   q4k-imatrix   int4    gptq
    arith/ru-7x8              4.6466      0.2470    0.1570  1.0122
    code/metal-kernel         1.8749      1.6524    0.2890  0.4771
    everything else            <=0.26

`arith/ru-7x8` is 19x what the same artifact pays without the experts, and it
is the probe that has broken every fragile format in this rebuild -- nvfp4 paid
6.64 nats on it, and Phase 5 recorded it as the one context that carried
Phase 4's whole verdict. Drop it and `q4k-all`'s mean falls from 0.1476 to
0.0761 and its max to 1.8749, which is `code/metal-kernel`, a probe
`q4k-imatrix` already pays 1.65 on.

So the tail is **one probe, not a pattern** -- and it is explained, by the
probe rather than by the kernel. `arith/ru-7x8` is `Семью восемь равно`, the
parent's own entropy on it is **4.53 nats**, and across the fifteen artifacts
this rebuild has captured its KL runs from 0.0949 to 4.7839, a **50x spread**
on one context. The distribution is not a gradient but a **cliff**, and which
side an artifact lands on is visible in one token:

    capture              KL      top-1 (p)
    int4-head-linattn  0.1570    'бе'   0.344     the arithmetic continues
    q4k-imatrix        0.2470    'бе'   0.133     still there, already wobbling
    gptq-head-linattn  1.0122    'бе'   0.820
    q4k (91, plain)    1.9555    ' '    0.126     half off
    **q4k-all**        **4.6466** '\n\n' 0.260    over
    nvfp4-head-linattn 4.7839    '\n\n' 0.240    over, and rejected in Phase 4

**`q4k-all` lands exactly where nvfp4 landed** -- same token, same magnitude,
the paragraph break Phase 5 recorded as nvfp4's signature failure on this
context. That is a known cliff being fallen off, not a kernel reading the
wrong bytes: the routed gather is pinned at exactly zero on a one-hot, the
experts' weight error is in line with every other module, and all 187 modules
agree with ggml digest for digest.

What it costs is real all the same -- a top-1 that changes from an arithmetic
continuation to a line break is a behaviour change, not a distribution wobble,
which is the distinction Phase 5 drew when nvfp4 did it. It is one probe of
64, `q4k-all` still moves the fewest top-1 tokens of any variant (6), and the
probe is the one this rebuild has watched break four formats.

### What is left

  - **`qmm_t`**, still. M > 1 is ggml's token axis, a mat-vec per token, and
    now the experts go through it too: the probe capture is the slowest step of
    the whole pipeline.
  - **Q6_K on the 8-bit side**: the kernel is written and pinned
    (`tests/q6k.rs`, dequantize bit-exact against ggml, one-hot exactly zero),
    `forge_q4k.py --q6k` selects its 128 of the 169, and the artifact is not
    built. Expect what this phase found: bytes, and nothing in the probes.

---

## Phase 17 -- Q6_K on the eight-bit side, and the first measurable cost -- **DONE**

`Q4K.md` priced Q6_K for the 169 modules the artifact holds at affine int8 g64
and argued, per bit, that they were the artifact's *cheapest* bits to give up:
0.68 error-doublings a bit against the 0.96 the 4-bit side already pays. This
phase built it. **The probes say the opposite, and it is the first thing this
instrument has called significant.**

`kernels/q6k.metal` is ggml's `kernel_mul_mv_q6_K_f32_impl` and
`dequantize_row_q6_K`. Two traps of its own, neither of them Q4_K's: the
sub-block scale is **signed** and the code is **biased by 32**, so
`w = d*sc*(q - 32)` with no minimum anywhere. `forge_q4k.py --q6k` converts
**128 of the 169** -- not `embed_tokens`, read by a gather and never by a
matvec, and not `shared_expert_gate`, which is `[1, 2048]` where the kernel
covers four rows at a time.

Built on top of `q4k-all`, so the artifact is 315 k-quantized modules and the
only variable against it is the 8-bit side. `check_rebuild.py` PASSes, 452/281
holds, and **all 315 modules agree with ggml digest for digest**.

### What it bought, and what it cost

    artifact            bytes/token  decode      median KL, all 64 probes
                                     best of 4   median     p90      max    top-1
    int4-head-linattn    2.0761 GB   102.6 t/s   0.0350   0.1933   0.7394   8/64
    gptq-head-linattn    2.0761 GB        --     0.0273   0.2412   1.0122   9/64
    q4k-imatrix  (91)    1.9814 GB   104.8 t/s   0.0260   0.2336   1.6524   6/64
    q4k-all     (187)    1.9311 GB   107.9 t/s   0.0237   0.1697   4.6466   6/64
    **q6k-all   (315)**  **1.8051**  **124.2**   0.0342   **0.1572** 4.8855  **5/64**

**-13.1% of a decode step against the baseline**, and the rate is where this
machine stops resolving: 124.2 best-of-four here, 110.4 best-of-four for
`q6k-full` in Phase 18 -- *identical traffic, identical kernels* -- with
single draws as low as 79.9. So the honest reading is **+8% to +21%** against
the baseline's 102.6, and the spread is the machine rather than the artifact.

    q6k-all vs q4k-all            lower on 22 of 64   p = 0.017
    q6k-all vs int4-head-linattn  lower on 36 of 64   p = 0.38
    q6k-all vs gptq-head-linattn  lower on 33 of 64   p = 0.90

**22 of 64 is the first significant result this instrument has produced.**
Every comparison in Phases 15 and 16 was a coin flip; this one is not, and it
says the last step *hurt* -- which, once the two steps are set side by side
below, is the expected answer rather than a surprising one. The whole package is still indistinguishable from
the artifact that ships -- 36 of 64 against `int4-head-linattn`, at 13% fewer
bytes -- but the marginal move from `q4k-all` to `q6k-all` is a measured loss
rather than a free byte.

Two honesties about that p-value. It is uncorrected, and four comparisons were
run, so Bonferroni would put it at 0.068. And it is in the *expected*
direction -- more quantization, more damage -- so it is a confirmation that
the instrument can see something rather than a surprise it dug up.

### Why one side was free and the other was not, which is not what was written first

The first version of this section called the per-bit argument a fifth
falsified predictor. **That was wrong, and the reason is worth more than the
claim.** The two conversions are not the same kind of move:

    the 4-bit side   affine int4 g32   5.00 bits   8.45%  ->  Q4_K  4.50 bits  7.48%
    the 8-bit side   affine int8 g64   8.50 bits   0.74%  ->  Q6_K  6.56 bits  1.85%

**Fewer bits and *less* error on one; fewer bits and 2.5x the error on the
other.** The 4-bit step was never a trade -- Q4_K is a better grid than affine
int4, so it bought bytes and accuracy at once, which is the whole reason a
k-quant is interesting. The 8-bit step is an ordinary purchase along the
rate-distortion curve.

`Q4K.md`'s per-bit arithmetic compared two movements *down that curve*
(8.50 -> 6.5625 at 0.68 doublings a bit, 6.5625 -> 4.50 at 0.96) and then
applied the ranking to a step that was not on the curve at all: affine 5.00 ->
Q4_K 4.50 is **-0.27** doublings a bit, because the error falls. So the
argument was not refuted by the probes; it was pointed at the wrong step, by
this file, and the probes did what they should -- the free lunch measured
free, and the purchase measured like a purchase.

What that does *not* license is rehabilitating weight error as a selector.
Phase 15 stands: at fixed coverage it ranked Q4_K above affine int4 on four
separate measures and the probes said level. What these two steps show is only
that the *sign* of a large weight-error change carried through, which is a
much weaker claim than ranking two formats that differ by a point.

### The shape of the trade is not uniform

`q6k-all` is worse in the middle and better in the tail: the **best p90 of
every artifact here (0.1572)** and the **fewest top-1 tokens moved (5 of
64)**, against the worst median of the three k-quant artifacts. Where the
8-bit conversion shows most:

    code/metal-kernel      -0.8404    q6k 1.0345   q4k-all 1.8749
    arith/ru-7x8           +0.2389    q6k 4.8855   q4k-all 4.6466
    longrange/json-field   +0.1502    q6k 0.3961   q4k-all 0.2460

So it is not one probe this time -- it is a broad small loss in the middle
with one large gain, which is what a median moving against a p90 looks like.

---

## Phase 18 -- the same conversion, weighted, and Phase 17's loss disappears -- **DONE**

Phase 17 measured the 8-bit side's conversion as the first significant quality
cost this instrument has found: 22 of 64 probes, p = 0.017. **It was measured
on a conversion that quantized 52% of that side blind.**

`forge_q4k.py --q6k --imatrix` weighted 80 of its 128 modules. The 48 it could
not weight were every `self_attn` projection and the shared expert's
`down_proj`, because nothing in `refs/` saw what they see: the gated-delta
Hessians are the *linear* layers' input, and `layerN.hidden` is the
post-attention hidden the MoE reads. A full attention layer's projections read
`input_layernorm`, and that tensor had never been captured.

So capture it. `ORNITH_GPTQ_FAMILIES=attn dump_gptq_stats.py` adds three
groups -- `q_proj` (2048x2048, shared with `k_proj` and `v_proj`), `o_proj`
(4096x4096, its own) and `shared_expert.down_proj` (512x512, all 40 layers) --
and the sharing is **asserted rather than assumed**: the first four inputs of
each of q, k and v are compared element by element, `max_abs == 0.0` over ten
groups. 60 Hessians, 841 MB, 218 s over the same 254028-token corpus, taken on
`int4-head-linattn` rather than the parent because Phase 9 measured the two at
0.58% apart.

`q6k-full` is `q6k-all` with one variable changed: **128 of 128 weighted
instead of 80**. The audit confirms the change landed where it should --
exactly 48 modules' digests move, and their *unweighted* error rises
(`self_attn.q_proj` 1.867% -> 1.920%, `o_proj` 1.867% -> 1.942%,
`shared_expert.down_proj` 2.028% -> 2.260%), which is the signature of a
weighting scheme doing its job.

### The loss is gone

    variant              median      p90      max     mean   top-1
    q6k-full             0.0244   0.2042   4.5238   0.1426   5 of 64
    q6k-all              0.0342   0.1572   4.8855   0.1454   5 of 64
    q4k-all              0.0237   0.1697   4.6466   0.1476   6 of 64
    int4-head-linattn    0.0350   0.1933   0.7394   0.0865   8 of 64
    gptq-head-linattn    0.0273   0.2412   1.0122   0.0862   9 of 64

    q6k-full vs q6k-all            lower on 40 of 64   p = 0.060
    q6k-full vs q4k-all            lower on 31 of 64   p = 0.901
    q6k-full vs int4-head-linattn  lower on 38 of 64   p = 0.169
    q6k-full vs gptq-head-linattn  lower on 35 of 64   p = 0.532

**31 of 64 against `q4k-all` is a coin flip**, where Phase 17's unweighted
version was 22 of 64 at p = 0.017. The 8-bit side's bytes are free after all
-- *but only weighted*. The imatrix is not a refinement here; it is what makes
the conversion viable, and the difference between the two runs is 40 of 64 in
its favour (p = 0.060).

### What Phase 17 should have said

Not "the 8-bit side's bits cost", but **"quantizing 52% of a side without an
importance vector costs, and the vector was missing because nobody had
captured that input"**. The per-bit correction already in Phase 17 stands --
a grid change and a rate change are different moves -- and this is a second,
independent reason its headline was too strong.

It also puts a number on the imatrix that the 4-bit side could not: there, the
comparison was 37 of 64 with and without (p = 0.26) on top of a conversion
that was already free. Here it is the difference between a measured loss and a
coin flip.

### Where it helped and where it did not

    helped                                    hurt
    arith/ru-7x8      4.8855 -> 4.5238        longrange/russian-decl  0.2877 -> 0.4912
    longrange/json-field 0.3961 -> 0.2330     arith/ar-2p2            0.0964 -> 0.2169
    script/hi-capital 0.1220 -> 0.0617        mixed/ja-mlx-eval       0.1466 -> 0.2539

p90 moves the wrong way (0.1572 -> 0.2042) while the median moves the right
way (0.0342 -> 0.0244), so this is a redistribution and not a uniform gain --
the same shape Phase 17 saw, pointing the other way.

---

## Phase 19 -- the prefill debt paid, and the captures were never on one instrument -- **DONE**

Phases 15-18 built five k-quant artifacts and every one of them was measured
through a **mat-vec per token** at prefill: ggml's `kernel_mul_mv_*` carries a
token axis on the grid's y, so M > 1 works, and it works by re-reading the
whole weight matrix once for each row. `HANDOFF.md` put a number on it --
level with affine at 600 prompt tokens, where L2 absorbs a 9 MB matrix read
599 times, and -8% / -21% in tok/s at 3060. This is the tile that replaces it:
ggml's `kernel_mul_mm`, 64 output features by 32 rows by 32 of K, in
`kernels/kquant_mm.h`, shared by both formats through a traits struct, plus the
routed twin.

`Q4K.md` step 7 has the design and the three departures from ggml's kernel.
The one that decided everything else: **the tile scalar is `float`, not ggml's
`half`**, because a gemm that replaces a checked kernel owes agreement with it,
and rounding a dequantized weight to eleven significand bits is a 5e-4
perturbation where reassociation is 2e-6.

### What it bought

Read GPU seconds here and not the wall. Five configurations, two prompts, two
or three draws each -- and within a configuration the **GPU seconds hold to
1%** (`q4k-all` on the mat-vec: 6.85, 6.90, 6.84) while the wall swings by
**3x** (`int4` at 2932: 3.78, 8.30, 11.46 s for 2.49, 2.57, 2.64 s of GPU) and
by **4.4x** at worst (`q4k-all` on the tile at 599: 0.72 and 3.16 s, for 0.68
and 0.76). The wall is measuring the swap file; the GPU is measuring the
kernel.

    prompt tokens        affine    q4k-all              q6k-full
                         int4      mat-vec    tile      mat-vec    tile
    599                   0.69      1.43       0.68      1.79       0.69   s GPU
    2932                  2.49      6.84       2.71      8.82       2.75   s GPU

**The debt is paid.** At 2932 tokens the k-quant prefill was 2.75x and 3.54x
the affine artifact's GPU work and is now **1.09x and 1.10x**; at 599 it was
2.1x and 2.6x and is now **1.00x** -- level, to the digit, on both formats.

The wall follows wherever the host keeps up. Quoting only the draws that ran at
93% GPU busy or better, so the number is the kernel and not the queue:

    prompt tokens     q4k-all               q6k-full
                      mat-vec    tile       mat-vec    tile
    599                399.8     836.9       310.4     815.3   tok/s
    2932               411.9       --        325.7    1035.5   tok/s

**and where it does not keep up, the gain stays in the GPU column.** Every
`q4k-all` draw on the tile at 2932 came back at 34-41% busy: 2.71 s of GPU
inside a 6.7-8.2 s wall, against the mat-vec's 6.84 s inside 7.1-9.2. Same
wall, a third of the work. That is `HANDOFF.md`'s "prefill's half-idle GPU"
showing through from underneath -- the tile removed GPU work that had been
hiding inside a host-bound wall, and what is left is a scheduling problem that
this phase does not touch.

Decode is untouched by construction: M = 1 and 8 routed rows are both below the
crossover, so both take the mat-vec they always took.

One number that is *not* a win and is worth leaving visible: at 2932 tokens the
affine artifact still does 2.49 s of GPU where `q6k-full` does 2.75, and it is
reading **more** bytes a token. mlx's steel gemm is a tuned kernel and this is
ggml's untuned one; ~10% is what that costs.

**That last sentence is not supported by that measurement, and Phases 23 and 25
both aimed at it as if it were.** 2.49 against 2.75 compares *two artifacts'
whole prefills*, which read different bytes and convert different modules --
not two kernels on one shape. Phase 23 closed 1.3-4.3% of the gap by tuning the
tile and Phase 25 showed the rest is not in the tile's knowledge of its own
shape; whether any of it belongs to the kernel at all is unmeasured. To be an
item it has to be re-posed kernel against kernel at one shape.

### The crossover, measured rather than picked

`src/bin/kquant_mm.rs` times both kernels at M = 1..64 on the four shapes the
formats are used at. The tile's own time is **flat** from M = 1 to 32 -- it
does a 32-row tile's work whether the rows are there or not -- so the ratio is
the mat-vec's straight line over a constant, and the crossing is at 4 to 8
rows depending on width. `MM_ROW_LIMIT` is 8: the conservative end, and it
leaves the speculative verify window (4 rows) exactly where it was.

At M = 1 the tile is 1.8x to 4.6x *worse*, so the guard is load-bearing rather
than decorative.

### The instrument moved, and the control says it is the tile

`Q4K.md` and this file both said the probe capture had to come back with the
same KL table or the kernel was wrong. It came back with `q6k-full`'s median at
**0.0327 -> 0.0354** and a head-to-head of "lower on 19 of 57", which read like
the first significant loss since Phase 17.

Both halves of that reading were wrong, and the second half is the finding.

**The sign test counted ties as losses.** Restricted to the probes that
actually moved, all three A/Bs are coin flips:

    pair                             moved   tile lower on      p     median
    q4k-all      -> tile                43        18        0.360   0.0320 -> 0.0315
    q6k-full     -> tile                44        19        0.451   0.0327 -> 0.0354
    int4 mat-vec -> **mlx's own gemm**  34        19        0.608   0.0447 -> 0.0427

**The third row is a control and it is the point.** `ORNITH_AFFINE_GEMM=0` --
a measurement escape hatch in `ops::quantized`, inert by default -- runs an
all-affine artifact's prefill through mlx's mat-vec instead of mlx's tiled
gemm. Same move, no k-quant anywhere in it, code nobody here wrote. It moves
the capture by *more* than either k-quant tile does:

    pair                            max_abs   bit-identical rows   largest KL
    int4 mat-vec vs mlx's gemm        3.116         24 of 64          0.348
    q6k-full mat-vec vs this tile     3.062         13 of 64          0.152
    q4k-all  mat-vec vs this tile     2.671         13 of 64          0.062

So the movement is what tiling a prefill does to a capture, on any
quantization family -- and this tile moves it less than mlx's own, because
mlx's `qmm_t` dequantizes into threadgroup memory as *bfloat16* and this one
uses float.

**What that costs to know: the k-quant verdicts in Phases 15-18 were taken on a
mat-vec prefill while every affine verdict in this rebuild was taken on a tiled
one. The captures were never on the same instrument.** The orderings survive
it, which is why those phases stand:

    variant           median      p90      max   top-1 moved
    q4k-all-gemm      0.0301   0.1449   4.6466     8 of 64
    q6k-full-gemm     0.0344   0.2116   4.5238     5 of 64
    int4-gemm         0.0422   0.2040   0.7394     8 of 64

Same order as Phase 18's table, and `max` unchanged to four decimals on both
k-quant artifacts -- `arith/ru-7x8` lands on the same side of its cliff. But it
was worth checking rather than assuming, and it is exactly the sort of
difference that sits undetected under a table.

### And one recorded verdict does not survive it: Phase 17's

Phase 17 produced **the only significant result this instrument has ever
returned** -- `q6k-all` lower than `q4k-all` on 22 of 64 probes, p = 0.017 --
and Phase 18 explained it as 48 modules quantized without an importance
vector. `q6k-all` was therefore worth re-capturing on the tile, so that both
arms of that comparison sit on one instrument. They now do:

    comparison                        instrument   sign test          p
    q6k-all vs q4k-all   (Phase 17)   mat-vec      20 of 57       0.033
    q6k-all vs q4k-all   (Phase 17)   **tile**     **23 of 58**   **0.148**
    q6k-full vs q6k-all  (Phase 18)   mat-vec      35 of 57       0.111
    q6k-full vs q6k-all  (Phase 18)   tile         35 of 58       0.148

(The first row recomputes Phase 17's claim on its own captures over the live
set rather than all 64, which is why it reads 0.033 where the phase recorded
0.017. Unlike the same-artifact A/Bs above, these have no ties to argue about:
two different artifacts differ on every live probe.)

**Phase 18's finding is unchanged and Phase 17's headline is not.** The
weighted conversion is better than the unweighted one by the same margin on
either instrument, which is the claim Phase 18 rests on. But the loss it was
explaining -- the only p below 0.05 in nineteen phases -- reads as a coin flip
once both artifacts are prefilled the same way. Phase 17 already carried two
corrections (the per-bit argument was pointed at the wrong step; the missing
imatrix, not the format, was the cost). This is a third, and it is the one that
takes the result rather than the explanation: **there may have been nothing
there to explain.**

What that does *not* license is reading the tile as the truer instrument. It is
the one the port now runs and the one every affine number was taken on, which
is the argument for using it; it is not independently more correct than a
mat-vec, and the two disagree by less than either disagrees with the parent.

### The routed side, and why it is mlx's design and not ggml's

ggml solves "which rows belong to which expert" with a separate pass that
builds a per-expert row list and a grid dispatched over experts. mlx's
`gather_qmm_rhs` solves it inside the tile: walk the tile's own indices, find
each maximal run of one expert, run the K loop once per run, store only that
run's rows. This takes mlx's, for three reasons -- no second kernel and no
scratch buffers, the port's own output grid is preserved, and it is **correct
for any ordering**. Unsorted rows are slow, not wrong, and that is a test
rather than a comment. `switch_glu_sorted` sorts at every batch of 64 rows or
more, so at prefill the runs are ~64 rows against a 32-row tile.

That also closes the "kquant_gather_qmm_rhs" item `Q4K.md` listed as a
follow-on: the streaming behaviour mlx gets from sorted rows is what the run
loop does, so there is no second kernel to write.

### What is left

  - ~~A Hessian for `in_proj_z`~~ -- **already done**, and this list was wrong
    to carry it. `forge_q4k.py:162` rewrites the module name to
    `in_proj_qkv`, whose Hessian is the same matrix (the two read the same
    tensor one line apart), so the imatrix has covered all 91 since the commit
    before Phase 16 and every artifact on disk was forged with it. `Q4K.md`
    step 9 has the measurement; Phase 15's "what is left" predates it and is
    left as written.
  - **Re-capturing the last two k-quant artifacts**, if their numbers are ever
    going to be compared against anything again -- `q4k-imatrix` and `q4k`, the
    91-module first cut. **Not "on the tile", which is what this item used to
    say**: a capture taken on whatever kernel happened to ship goes stale the
    next time one changes, which is the debt this very item is. Phase 21 has
    the fix -- `ORNITH_INVARIANT=all` makes a capture independent of every
    kernel choice, and costs 31 s for 64 probes.
  - ~~**The tile is ggml's shape, not a tuned one.**~~ **Phase 23 did it**, and
    the guess in this item was backwards: flat across M = 1..32 means the
    fixed per-block cost is a third to three fifths of the time, so the tile
    wanted to be *wider*, not narrower. 64 rows dense, 16 or 32 routed by
    `rows / experts`, -1.3% to -4.3% of prefill GPU and byte-identical output.

---

## Phase 20 -- reversible rotations, priced out before building -- **DONE**

Rotation is the last untried lever in this rebuild. The taxonomy the earlier
phases built has five axes and four of them have been walked:

    coverage     which module gets how many bits      Phases 4-5, knee mapped
    grid         the representable set                Phases 4, 14-18
    rounding     which grid point                     GPTQ, Phases 7-11
    weights      a diagonal rescale folded in         AWQ, Phase 6, rejected
    weighting    the objective                        imatrix, Phases 15, 18
    **weights**  **an orthogonal rotation folded in**  **this phase**

**And the recorded reason for not trying it was pointed at the wrong thing.**
Phase 4 surveyed HIGGS -- "Hadamard incoherence, MSE-optimal grids, data-free"
-- and parked it on three blockers: a CUDA runtime, a non-affine format, and no
MLX oracle. All three belong to HIGGS's *grid*. QuaRot and SpinQuant are the
rotation without the grid: an orthogonal `Q` folded offline into the
neighbouring weight matrices, with plain affine or Q4_K afterwards. The
artifact stays in a format mlx has, so **the oracle survives** -- and the one
piece that cannot be folded offline, the online Hadamard, mlx already
implements (`mlx::hadamard_transform`, `kernels/hadamard.h`, `n = m*2^k` for
`m` in 1, 12, 20, 28; every input dim here is a power of two). So the blocker
on record did not apply, and the question was open rather than closed.

`tools/rotation_ceiling.py` answers it without building anything, which is
Phase 13's move.

### The premise is already satisfied, so there is nothing to spread

A rotation exists to kill outliers. What binds here is not a tensor's outliers
but a **group's**: affine g32 fits a scale and a bias to each run of 32 input
channels, so the quantity is `(max - min) / rms` per group, and the yardstick
is what 32 Gaussian draws give. Simulated, not quoted: **median 4.14**, p99
5.35.

    module                  spread med   p99      (the 4-bit side, g32)
    lm_head                    4.13     5.60     <- already below the floor
    L0.in_proj_qkv             4.16     5.46
    L0.out_proj                4.26     5.85
    L20.in_proj_qkv            4.31     6.00     <- the furthest from it
    L20.experts.gate_up        4.16     5.41
    L20.experts.down           4.14     5.35
    L11.self_attn.q_proj       4.85     6.45     (8-bit side, g64)
    L11.self_attn.o_proj       4.84     7.28

**At the granularity the quantizer actually sees, these weights are already
Gaussian.** There is no incoherence to remove.

### Rotate and re-quantize, which is the direct answer

`W' = W @ Q` along the *input* axis -- the axis the groups run along and the
one QuaRot rotates, because `x' = Q.T x` undoes it in the previous op. A
Hadamard, which is what an implementation would use, and a Haar-random
orthogonal, which is the best case incoherence theory offers. Ratios
rotated/unrotated, below 1.00 means the rotation helps:

    module                  spread  affine  k-quant   out(aff)  out(kq)
    lm_head                   1.00    1.00     0.99      0.99     0.99
    L0.in_proj_qkv            0.99    0.99     0.99      1.00     1.00
    L0.out_proj               0.97    0.95     0.95    **1.23**   0.91
    L20.in_proj_qkv           0.96    0.94     0.93      0.95     0.91
    L11.self_attn.q_proj      0.96  **1.12**   0.96      0.90     0.95
    L11.self_attn.o_proj      0.96  **1.11**   0.96      0.96     0.84
    L20.experts.gate_up       0.99    0.99     0.99       --       --
    L20.experts.down          1.00    1.00     1.00       --       --

(Hadamard rows; the Haar rows differ by at most 0.03 except where noted below.)

The best weight-error gain anywhere is **6-7%**, on `L20.in_proj_qkv`, the
module that started furthest from the floor. On the 8-bit side affine gets
**11-12% worse**. Haar is never better than Hadamard, which is the tell: if the
mechanism were incoherence, the Haar-random rotation would be the ceiling and
it is not.

### And in the currency that has a link to the output, it is a wash with a tail

Weight error has been falsified as a selector four times here (Phases 4, 7, 14,
15), so the column that decides is `out`: the error in the layer's *product*
under the real activation second moment,

    rel = sqrt( tr(dW H dW.T) / tr(W H W.T) ),   H' = Q.T H Q

which is the same output error in two bases and therefore directly comparable.
`refs/gptq-parent/hessians/` and `refs/gptq-attn/hessians/` hold the full `H`.

It is a wash -- and `L0.out_proj`, the module `Q4K.md` already flagged for
carrying 21.6% H-weighted error against 7.5% unweighted, comes out **23% worse
with a Hadamard and 17% worse with Haar.**

The mechanism is clean. A rotation makes the **weights** uniform; the
**activations** do not become uniform. Mixing the input channels destroys
whatever alignment existed between the quantizer's error and the directions the
activations do not use, and the more concentrated the Hessian, the more there
was to destroy.

### The prize does not exist in this artifact

QuaRot's gain is **W4A4**. It exists so that *activations* can be quantized to
four bits, and the rotation is what makes activation outliers survivable. This
artifact is **W4A16**: every activation on every path is bfloat16 and nothing
here quantizes one. So the cost is paid and the thing it buys is not on the
table.

Which means the fold-point question -- the one AWQ's Phase 6 says decides scope,
and which looked hard here because 30 of 40 layers are gated-delta rather than
q/k/v/o -- never had to be asked. **Rejected, on measurement, before a line of
it was written.**

One honesty. SpinQuant *learns* the rotation against a task loss rather than
taking a fixed one, and a learned rotation optimising the H-weighted objective
could avoid `L0.out_proj`'s regression by construction. What this phase kills
is the argument from incoherence, which is measured and dead. The argument from
a learned rotation is untested and costs training, which is a different class
of expense than anything in this rebuild so far.

### What it found on the way: layer 0's experts are 41% dead in the parent

`L0.experts.gate_up` came back with a median group spread of **0.00**, which is
not a number a weight matrix produces. Read straight out of the safetensors
bytes, past mlx:

**16% to 82% of each layer-0 expert's intermediate channels are dead** -- a
whole row of `gate_up_proj` at the single bf16 pattern `0x01D4` = 7.7877e-38,
thirty-five orders below the mean weight of 0.0032, and the matching column of
`down_proj`. Over all 256 experts: **mean 41.4%, median 43.9%, min 0.4%, max
81.8%, and not one expert is fully alive; 71 of 256 are more than half dead.**

    layer      gate_up            down_proj
             all    whole rows   all    whole rows
       0    62.5%     62.5%     54.7%      0.0%   <- columns, not rows
       1     0.0%      0.0%      0.4%      0.0%
    2..39    0.0%      0.0%      0.0%      0.0%

No other layer, and no non-expert tensor of layer 0 (`in_proj_qkv`,
`shared_expert`, the router gate). The correspondence is exact and explains
itself: a dead `gate_up` row is a dead channel of `h`, which is a dead
`down_proj` column.

**It is worth about 0.38% of a decode step**, so it is not a lever. It is two
other things:

  - **a measurement hazard.** Any per-module statistic taken on layer 0's
    experts is computed on a tensor that is 41% constant, and the first run of
    this phase tripped over exactly that.
  - **a second defect in a published artifact from this publisher**, after the
    unshifted RMSNorm gain in 2.9.1's MTP head. It was found by a number that
    was too strange to be a result -- the same rule that caught Phase 14's
    `floor`-for-`ceil`.

`IDEAS.md`'s "Weight sparsity. Dense training; nothing to skip." is corrected
rather than deleted: for layer 0 there *is* something to skip, and there is
very little of it.

---

## Phase 21 -- batch invariance, and speculative decoding becomes exact -- **DONE**

A quantized linear picks its kernel by row count: `qmv` at M = 1, `qmv_wide`
at 2..`qmv_batch_limit`, the split-K gemm above that, and on the k-quant side
ggml's mat-vec below `MM_ROW_LIMIT` and Phase 19's tile above it. All of them
are correct and **none of them agrees with the others**, because they
associate the same sum differently.

Everywhere a batch is an optimisation over rows that *would* have been
computed one at a time, that disagreement is an artefact of the batching. The
speculative loop is exactly that: the target runs over `depth + 1` rows where
the autoregressive path would have run it one row at a time. mlx-vlm fixes it
and says why in one line -- `quantized_verifier.py`,
`singleton_quantized_linear`, "fuse a runtime-sized batch while retaining
singleton qmv arithmetic".

### The check that costs minutes, and it answered no

`tests/quantized.rs` pins M = 1/2/4/8/16/64/256 against MLX at
`max_abs == 0.0`, but that is port-against-mlx, not M = 1 against M = 4.
`tests/batch_invariance.rs` asks the second question, and it needs no
reference: both arms are this machine's own kernels on the same input.

    module                    M = 2..12 against the same row alone, max_abs
                              int4-head-linattn        q6k-full
    in_proj_qkv  2048->8192      4.9e-4 .. 3.9e-3      0 to M=7, 7.8e-3 at M>=8
    out_proj     4096->2048      3.8e-6 (0 at M=2)     --
    mlp.gate     2048->256       0 to M=10, 6.0e-8     --
    shared.gate  2048->512       2.3e-2 .. 3.5e-2      --
    lm_head      2048->248320    **7.8e-3**            0 to M=7, **1.6e-2** at M>=8
    routed, 2048 rows sorted     1.8e-2                7.8e-3

So the answer is no, and the shape of the no is informative. The k-quant
modules are *already* invariant up to `MM_ROW_LIMIT = 8` -- ggml's mat-vec
carries the token axis on the grid's y, which is M independent matvecs -- and
lose it exactly where Phase 19's tile takes over. The affine modules lose it
at M = 2, where `qmv_wide` starts sharing one weight read between five rows.

### The flag, and what it does not need

`ORNITH_INVARIANT` in `src/ops/invariant.rs`, off by default. It does not need
a kernel. The port owns its dispatcher, so "every row a mat-vec" is a branch:
`qmv_fast_impl` reads row `tid.x` and reduces over K the same way whatever is
beside it, so `qmv` at M rows **is** M copies of the M = 1 answer. That is
also why mlx-vlm needs `gather_qmm(rhs_indices=0)` and this does not -- from
Python there is no way to make `mx.quantized_matmul` take the mat-vec at
M = 4; from inside the dispatcher there is nothing else to do.

Two settings, because a prefill chunk is *not* a batch of rows that would
otherwise have run alone:

  - `=1` -- batches of at most `EXACT_WINDOW` rows compute row by row. Prefill
    keeps Phase 19's tile, so the mode is free at a long prompt.
  - `=all` -- every M, which is what the probe captures want.

**`EXACT_WINDOW` is 4 and it is not a choice.** mlx takes the fused vector
SDPA only while `q_len * gqa_factor <= 32`, and this model has 16 query heads
over 2 kv heads. `src/bin/invariance.rs` measures the boundary rather than
quoting it: windows of 2, 3 and 4 come back bit-identical, and a window of 5
first disagrees at **layer 3, the first full-attention layer**, by 3.9e-3. It
is also the largest window `--mtp-depth 3` produces, which is the depth
`mtplx_runtime.json` recommends. The two agreeing is luck.

### One thing the quantized flag did not cover, and the bisect found it

With the quantized linears pinned, three windows in sixty still moved -- and
the first layer to disagree was a *linear-attention* one, a different one each
time. `in_proj_a` and `in_proj_b` are bf16, and `Matmul::eval_gpu` sends a
1-row product to `gemv` and a 4-row product to the steel gemm. So the flag now
row-splits an unbatched bf16 product too. The batched ones are the unfused
attention's two, which mlx reaches only past the fused limit -- never at a
decode step or a four-row window.

That is the general lesson of this phase: **batch invariance is a property of
every kernel a forward touches, not of the quantized ones**, and the only way
to know which ones noticed is to bisect the layers.

### What it bought, on a whole forward

`src/bin/invariance.rs`, 45-token prompt, 240 decode steps, windows of 4, each
window measured against the same tokens one at a time *from the same cache
state*:

    arm                         bit-identical rows   top-1 moved   logit max
    pinned (same state)  off          0 of 240        7 of 240       2.72
    pinned (same state)  on         **240 of 240**   **0**           0.0
    rewound (partial accept) off      0 of 48         4 of 48        1.62
    rewound (partial accept) on     **48 of 48**     **0**           0.0
    drift (own caches)   off          0 of 240        8 of 240       2.57
    drift (own caches)   on         **240 of 240**   **0**           0.0

**2.9% of verify rows had an argmax that moved because the row was batched.**
A logit differing by 2.7 sounds impossible for a reassociated sum and is not:
the error enters at the first layer and compounds through forty, which is the
same mechanism the six divergent probes have in `HANDOFF.md`. Those rows are
drafts rejected because the target changed its mind.

### And the output is now the autoregressive output

`speculative_tracks_autoregressive` measures how far the two streams stay
together. It was **36 of 214 tokens** and is **221 of 221** -- all three
prompts identical to the last token, where two of them used to part company at
token 8 and 18. The printed text of `--mtp --greedy` is byte-identical to
`--greedy` on the same prompt, with and without FR-Spec, *without the flag on
the autoregressive side* -- the window mode does not touch prefill, so there is
nothing to match.

### What it costs, which is the surprise

300 tokens, one prompt, greedy, GPU seconds because the wall is not a
measurement on this machine:

    config                      acceptance      by depth          GPU ms/token
    autoregressive                  --             --             10.49
    `--mtp`                     168/393  42.7%  0.66/0.56/0.65     9.65  9.66
    `--mtp` + invariant         180/360  **50.0%**  0.71/0.71/0.58     9.66  9.74
    `--mtp --frspec`            167/396  42.2%  0.66/0.55/0.67     8.55  8.46
    `--mtp --frspec` + inv      179/363  **49.3%**  0.70/0.69/0.59     8.54  8.75

**Acceptance up 17% relative, GPU work unchanged, wall unchanged** (89.7/91.1
against 90.6/90.4; 102.0/102.2 against 102.5/102.6).

Unchanged GPU is not what the arithmetic predicts. A verify window's dense
linears now read the weight matrix once per row instead of once per five, and
a decode step is 95% GPU busy waiting on weight reads. The reason it is free
is that **the grid gets wider, not the traffic**: `qmv`'s grid is
`(M, O/bn, 1)`, so the four rows of one output tile are adjacent threadgroups
reading the same lines, and the cache serves three of the four. It is the same
effect Phase 19 measured from the other side -- "L2 absorbs a 9 MB matrix read
599 times".

So what the flag buys is exactness and eight points of acceptance; the
acceptance pays for the extra reads and no more. **It does not make
speculation faster. It makes it correct.**

### The instrument, fixed at the root this time

Phase 19 found that tiling a prefill moves a probe capture on any
quantization family, and answered it with a control experiment
(`ORNITH_AFFINE_GEMM=0`: same move, code nobody here wrote). Under
`ORNITH_INVARIANT=all` there is nothing to control for, because no kernel
anywhere depends on a row count:

    pair of captures, same artifact           without the flag     with it
    int4-head-linattn, mlx gemm vs mat-vec    max_abs 3.116     **byte-identical**
    q6k-full, tile vs mat-vec                 max_abs 3.062     **byte-identical**

That is `cmp` on `logits.bin`, not a tolerance. A capture in this mode is a
**third instrument, and the only one with no free parameter in it**:

    variant                 median      p90      max
    int4-gemm               0.0422   0.2040   0.7394
    int4-matvec             0.0437   0.2612   0.9139
    **int4-invariant**      0.0460   0.2571   0.9139
    q6k-full-gemm           0.0344   0.2116   4.5238
    q6k-full mat-vec        0.0325   0.2109   4.5238
    **q6k-full-invariant**  0.0332   0.1898   5.2520

It agrees with both of the others on the ordering -- `q6k-full` below `int4`
on the median on all three -- and differs from each by about as much as they
differ from each other. Which is the honest reading of Phase 19's finding
restated: the instrument has a spread of this size, and no capture in this
rebuild was ever outside it.

### What is left

  - **Windows past four.** `--mtp-depth 5` drops attention into the unfused
    batched gemm and the promise ends. Row-splitting a *batched* product is
    possible and not written; it is also pointless until a depth past 3 looks
    like it pays, and Phase 21's own acceptance table does not say it does.
  - **`=all` at a long prompt** is Phase 19's debt taken back on purpose: a
    mat-vec per token plus one `gemv` per token for `in_proj_a/b`. 64 probes
    cost 31 s, which is the only place it is used.
  - **The MTP head is not covered and does not need to be.** Its arithmetic
    decides which drafts are proposed, not which are accepted.

### The default, taken 2026-09-19

**The window mode is on by default now**, and the argument that decided it is
not the one this phase was built on. By Phase 26 the shipped configuration is
`--mtp-depth 1 --frspec`, where the verify window is two rows and the flag is
worth *one point* of acceptance (0.72 -> 0.73) and nothing on the clock
(7.22/7.31 against 7.43/7.31 ms of GPU). What it is worth is the property:
without it `--mtp --greedy` silently prints something other than `--greedy`,
and `src/frspec.rs`'s own argument for why a draft cannot change what the model
emits is true of the draft and **false of the target**, whose arithmetic moves
with the window. A port whose contract is `max_abs == 0.0` should not ship the
side of that where the two disagree.

**Scoped by call site, not by width, and that is what makes it free.** A
five-token prompt prefills four rows, so a width rule would have changed the
arithmetic of short prompts for nothing. `generate::prefill` runs inside
`invariant::without_window`, so a prefill chunk is never invariant and `all`
passes through untouched. Three checks, all byte-for-byte:

    --mtp --greedy against --greedy, depth 1 and depth 3   identical
    a five-token prompt, default against ORNITH_INVARIANT=0  identical
    64 probes through q6k-full, before the flip and after    identical

So the blast radius is exactly the speculative path.

**And it makes the test suite say where the departure is**, which is the part
worth keeping. Four cases pin *mlx's* or MTPLX's dispatch rather than this
port's -- `quantized.rs`, `quantized_nvfp4.rs` and two in `mtp.rs` -- and they
now ask for the undeparted path explicitly, with the reason written where they
ask. `ORNITH_INVARIANT=0` is that path.

---

## Phase 22 -- AWQ on the experts, priced on the right object and rejected -- **DONE**

Phase 12 closed GPTQ on the experts and named the successor in one line:
"`down_proj` reads `silu(gate) * up` ... the one module GPTQ cannot touch is
one AWQ **can**. Untested, and it is the first thing to try if this phase's KL
comes back thin." The KL came back thin. This is that test, and it is a
pricing rather than a build -- the same shape as Phase 20.

### The fold is exact, and it can only ever reach one side of the block

`h_c = silu((W_gate x)_c) * ((W_up x)_c)` puts the nonlinearity on the gate
path alone, so dividing **row `c` of `W_up`** divides `h_c` exactly, and
`W_down diag(s)` restores the product. Phase 12 has it right.

What Phase 12 did not say is that the same structure bounds what the fold can
buy, and the bound is one-sided. **A per-channel scale can only move the
rounding of a tensor whose quantization groups run along that channel index.**
`W_down`'s groups are 32 consecutive input channels, so `s` varies inside a
group and changes its rounding -- there is a lever. `W_up`'s groups run along
its *input* axis, so scaling row `c` scales all 64 of that row's groups
uniformly, and affine quantization is **exactly equivariant** under that:

    per-row factor       codes identical    stored scale == factor * scale
    2**-3, 2**-1, 2, 2**4      yes                    yes
    1.37                       no                 no (max 14.5% off)

Measured on `L20.up_proj`, expert 0, through `aq.quantize`. So at a power of
two the compensation side is not merely accurate, it is a **no-op**: the same
integer codes, the scales and biases scaled exactly, `h'_c = h_c / 2**k` bit
for bit. At an arbitrary factor it is not a rounding effect but a discrete one
-- the quantizer's `q0 == 0` branch flips -- which is a stronger reason for
Phase 6's power-of-two restriction than Phase 6 had.

**The consequence: the exact fold protects `down_proj` and nothing else.** The
side of the expert that can be helped is fixed by the layout, not chosen.

### On the object Phase 12 had, there is nothing to take

Priced in Phase 12's own currency, `||(W - Wq) X^T||_F / ||W X^T||_F` through
the Hessian already on disk, 8-16 experts a layer, best alpha per expert:

    layer   saliency        RTN   AWQ continuous   AWQ power-of-two
      0     pooled       0.0730     -0.3%              -0.0%
     20     pooled       0.0813     -0.1%              +0.0%
     39     pooled       0.0071     -0.0%              +0.0%

The RTN column reproduces Phase 12's own (`L0` 0.0731, `L20` 0.0813), which is
how this is known to be measuring the same object. The search **chooses
`alpha = 0`** -- no scaling -- on most experts.

And it is not AWQ's parametric family that is the limit. Uniformly scaling a
group changes nothing, so the objective decomposes over groups and a
32-coordinate descent per group is an unconstrained oracle for what *any*
per-channel scale could do:

    L0  expert 0   unconstrained per-channel scale   -0.4%
    L20 expert 0                                     -0.1%

### But that object was the wrong one, and the diagonal was never 172 GB

`dump_gptq_experts.py` pools over (token, expert) pairs because the exact
per-expert Hessian is 172 GB, and Phase 12 listed that as the honest
alternative explanation for its own null result. **A diagonal is not 172 GB.**
One expert's saliency is 512 numbers; the whole model's is 21 MB.
`tools/dump_awq_experts.py` records it -- `mean|h_c|` and `E[h_c**2]` per
(layer, expert, channel), same corpus, same streamed parent, 241 s, 4 GiB
peak, every expert seeing at least 1432 rows.

It says the pooled object had hidden an order of magnitude:

    spread of mean|h_c| across channels     pooled    per expert (median)   (max)
    layer  0                                 1.29          3.56             22.5
    layer  5                                 1.09          2.02              6.9
    layer 20                                 1.12          2.21             12.8
    layer 35                                 1.52          3.80             25.6
    layer 39                                 1.68          4.69             47.5

**Pooling 256 experts does not average a shared pattern, it cancels 256
different ones.** That is Phase 12's alternative explanation #4, measured and
real -- and it retires a weight-side proxy this phase tried first, row norms of
`|gate_c| * |up_c|`, which said the per-expert spread was *smaller* than the
pooled one (1.4-1.8x). It ignores the input covariance and `silu`'s gating,
which is exactly where the per-expert structure lives.

### Re-priced on the right object, and still a reject

Saliency `mean|h_c|` and scorer `E[h_c**2]`, both per expert, 16 experts a
layer, best alpha per expert:

    layer      RTN   AWQ continuous   AWQ power-of-two   best single expert
      0     0.0713       -6.3%             -3.8%             -26.0%
      5     0.0809       -1.6%             -0.3%              -6.2%
     20     0.0813       -2.8%             -1.6%             -17.7%
     35     0.0070       -0.0%             -0.7%              -0.0%
     39     0.0072       -0.7%             -3.0%              -7.1%

Per-expert granularity is worth an order of magnitude over the pooled
statistic -- 0.3% becomes 6.3% -- and it is still not enough, because of what
this same currency has already bought elsewhere:

    intervention, same proxy                 objective   delivered on 64 probes
    GPTQ, expert gate/up (Phase 12)          -36%/-18%   +3.5%, inside its own draw spread
    AWQ, in_proj, power-of-two (Phase 6)     -26.2%      nothing, and `arith` 3-7x worse
    **AWQ, expert down_proj, per-expert**    **-2%..-6%**   not built

Two interventions with three to six times more of this objective both came back
unreadable on the instrument. Phase 12's +3.5% was already smaller than the gap
between its own two draws (0.0017 against 0.0015). An intervention a further
level down is **below what 64 probes and a sign test can resolve**, which is
not a prediction about AWQ -- it is a statement about the instrument.

And the scorer used here is *diagonal*, which flatters AWQ: AWQ is a diagonal
method, so scoring it diagonally ignores whatever it does in the off-diagonal
directions. The number above is an optimistic one.

### The other side of the expert, priced while the tools were out

The side of the block with spread is the one `gate_proj` and `up_proj` read.
Its Hessian is on disk too, and it is not close:

    diagonal spread, max/median        layer 0   5    10   20   30   35   39
    hidden  (gate/up input, 2048)       14.5   10.3  8.6  5.8  9.7  6.9  10.4
    inter   (down_proj input, 512)       1.7    1.4  1.3  1.6  1.4  2.4   3.0

That side folds into `post_attention_layernorm`, whose readers are exactly the
router gate, the 256 experts' `gate`/`up` and the shared expert's -- all
quantized, no bf16 reader but the gain itself, so the folding group is clean.
Priced the same way, one scale for the group, gate+up of 8 experts:

    layer      RTN    AWQ continuous   AWQ power-of-two
      0     0.0467      -15.5%            -13.7%
     20     0.0656       -4.0%             -2.7%
     39     0.0050       -0.0%             -2.0%

Better than `down_proj` and still weaker than the GPTQ attempt on the same
tensors that delivered nothing. **Not built either**, for the same reason.

### What would change this

  - **An instrument that can read a 2-6% objective.** Everything here is a
    statement about a lever being smaller than the noise floor of 64 probes.
    More probes, or a paired design that removes between-draw variance, moves
    the floor rather than the lever.
  - **A saliency model that is not `mean|x|`.** Phase 6 measured that
    `mean|x|` is substantially language-independent and that the damage it
    misses is concentrated in rare forced continuations. That diagnosis
    applies unchanged here; a saliency that predicted *those* channels would
    be a different experiment, not a different alpha.
  - **Not a bigger alpha grid, and not the continuous family.** The
    unconstrained oracle is 0.4%, and the power-of-two restriction costs
    little of what is there (6.3% -> 3.8%).

`refs/awq-experts/` is kept: 21 MB, and it is the first per-expert statistic
this rebuild has, which is the part of this phase most likely to be wanted
again.

---

## Phase 23 -- the tile tuned, and the narrow one was the wrong direction -- **DONE**

Phase 19 built ggml's tile and said what it had not done: "the tile is ggml's
shape, not a tuned one ... the flat 98.4 us across M = 1..32 says a narrower
tile would help the shapes that cross late." The flatness is the right
diagnosis and the conclusion drawn from it is backwards.

### What flat actually means

Time two widths and the tile separates into a fixed part and a part per row:

    shape                     tile 32   tile 16   fixed   per row   fixed share
    in_proj_qkv [8192, 2048]   100 us     66 us   32 us   2.13 us      32%
    out_proj    [2048, 4096]    56         38     21      1.09         38%
    q_proj      [2048, 2048]    31         22     13      0.56         42%
    down_proj   [2048,  512]    11.5        9.2    6.9     0.14         60%

The fixed part is the A side: 64 weight rows dequantized once per block
whatever the rows are. Flat from M = 1 to the width is that fixed part being
*all there is* below a full block -- and above one it is still a third to
three fifths of the time, paid once per block. **A narrower tile pays it twice
as often; a wider one halves it.**

### The narrow tile, built and measured

`kquant_mm.h`'s `NR1` is a template parameter now, and 16, 32 and 64 are all
instantiated. `tests/q4k.rs::the_tile_widths_agree` holds them to `max_abs ==
0.0` against each other at nine row counts, which is what the coupling needs:
`NR1` moves the accumulator count, `lsmb`'s offset and stride, the staging
offset and the B-side thread mapping together.

    M                         1      4      8     16     32     64    256
    tile 16 / tile 32       0.66   0.67   0.68   0.67   1.26   1.27   1.33

**33% better below 16 rows and 26-33% worse above**, which end to end is
**7.6% worse** at 3402 prompt tokens. The shapes it helps are not shapes this
model has: below 16 rows the mat-vec is still better than either tile, and
above them prefill lives. Kept as an instantiation, not as a default.

### The wide tile, and a 20x cliff that is not about the shape

64 rows is the direction the fixed part points, and the first build of it was
**12 to 23 times slower**, uniformly, at every shape and width. Not register
spilling in the sense one would guess from occupancy -- `maxTotalThreadsPerThreadgroup`
reports 1024 for every one of these kernels. The Metal compiler stops fully
unrolling the accumulator loop somewhere between 8 and 16 iterations, and an
array of `simdgroup_matrix` that is indexed dynamically is not in registers at
all. `#pragma clang loop unroll(full)` on the three `mc` loops takes the
64-row tile from 2238 us to 179 us.

**Worth recording as a trap rather than a fix**: a shape parameter that
changes an accumulator count also changes whether the compiler unrolls, and
the failure mode is a cliff, not a slope. Nothing about the arithmetic
changed -- the bit-identical test passed at 2238 us and at 179 us alike.

Unrolled, the wide tile does what the decomposition predicts:

    M                        64    128    256
    tile 64 / tile 32      0.94   0.93   0.94   (in_proj_qkv)
                           0.95   0.94   0.95   (out_proj)
                           0.97   0.97   0.94   (q_proj)
                           0.98   0.98   0.93   (down_proj)

### The routed tile wants a different width, and the quantity is mlx's own

The routed tile runs the K loop once per maximal run of one expert inside its
own rows, so a tile wider than the runs straddles boundaries. Timed against
synthetic runs:

    run length              16     32     48     64     96
    tile 16 / tile 32     0.71   1.35   0.95   1.35   1.29
    tile 64 / tile 32     1.89   1.89   1.37   0.97   1.30

A 32-row tile takes **2.1x its aligned time** when the runs are 16 long. So
the width should follow the average run, which is `rows / experts` -- the same
quantity `GatherQMM::eval_gpu` already thresholds on. End to end, GPU seconds,
two draws each:

    prompt   rows an expert   routed 16   routed 32   routed 64
      748          23          **0.77**      0.80
     1887          59            1.72      **1.70**
     3402          64            3.16      **3.13**       3.43

**64 is never right even when the mean run is 64**, because the distribution
has short runs in it. The rule is 16 below 32 rows an expert and 32 above.

### What it bought, and what it did not

Shipped: dense tile 64, routed tile 16 or 32 by `rows / experts`,
`MM_ROW_LIMIT` 8 -> 16 (a 64-row tile costs 1.7x a 32-row one below a full
block, so the crossover moved with it). GPU seconds, two draws a cell:

    prompt tokens     q6k-full           q4k-all
                    before   after     before   after
       748           0.81     0.78
      1887           1.73     1.70
      3402           3.18     3.13      3.09     3.05
    decode          8.78     8.69  ms/token -- M = 1, untouched by construction

**-4.3% at 748 tokens, -1.7% at 1887, -1.3% at 3402.** Against Phase 19's
~10% gap to mlx's steel gemm, that closes a small part of it; the rest is not
in the tile's shape.

### The scalar was free, and nobody had checked

`kquant_mm.h` instantiates the tile at `float` where ggml uses `half`, and
says why: a gemm replacing a checked kernel owes agreement with it, and eleven
significand bits is a 5e-4 perturbation where reassociation is 2e-6. That was
a decision taken on a contract and never priced. Priced now --
`ORNITH_KQUANT_TILE_HALF=1` instantiates ggml's own scalar -- and it is
**0 to 4% on every shape**, inside the measurement. So the exactness cost
nothing, and the flag stays so the question does not have to be asked again.

### And the whole phase is quality-neutral by construction

Not "the sign tests are coin flips", which is what every other kernel change
in this rebuild has had to settle for. All three widths are bit-identical to
each other, and the probe capture proves it end to end: 64 probes through
`q6k-full` on the old tile and the tuned one are the **same 63,569,920 bytes**.
`cmp`, not a tolerance.

That is the difference between tuning a shape and changing arithmetic, and it
is the reason Phase 19's instrument question does not reappear here.

### What is left, and it is the lever this phase did not spend

**The tile is still compiled for an unknown shape.** `kquant_qmm_t` and
`kquant_gather_qmm_t` take `K`, `O` and `M` as `constant int&` (buffers 5, 6
and 7), so at compile time the kernel knows none of them. Three consequences,
all of them paid on every tile:

  - `nb = K / Q::QK` is a runtime divide, and
    `for (loop_k = 0; loop_k < K; loop_k += KQ_NK)` cannot be unrolled at all.
  - `nr0`, `nr1`, `lr0` and `lr1` clamp every load against the tail, on every
    block, including the overwhelming majority that are full.
  - the store's `i < nr0` bound is runtime.

**And this model has six shapes, all known before the metallib is built**:
`[8192, 2048]`, `[2048, 4096]`, `[2048, 2048]` and `[248320, 2048]` dense,
`[512, 2048]` and `[2048, 512]` routed. `O % 64 == 0` on every one of them,
and `K % 32 == 0` on every one of them, so both clamps are provably dead code
for the shapes that exist.

This is the second lever the brief for this phase named and the one it did not
reach: `vendor/accelerated-qwen-3.8`'s `patches/metal-combination.patch` does
exactly this to ggml's *mat-vec*, moving shape parameters into function
constants and instantiating narrow row counts separately, and mlx does it to
its own gemm (`alN_{aligned}` in `qmm_t`, function constants 200/201/202 in
`gather_qmm_rhs_nt`). The port already lives that way elsewhere --
`affine_qmv_fast` against `qmv_wide` is the same idea one level up.

**The reason to expect it to pay is this phase's own 20x cliff.** An unroll
pragma -- one line, no arithmetic touched -- was worth 12 to 23 times on the
wide tile. A kernel whose time moves by that much on a codegen decision is a
kernel where telling the compiler what it is compiling is not a micro-
optimisation.

**Built in Phase 25, and that reasoning was wrong.** The lever is spent and it
buys nothing; the paragraph above is left as written because the argument was
the best available before the measurement, and because the shape of its error
is the phase's finding.

---

## Phase 24 -- the per-expert importance vector, built and not adopted -- **DONE**

`forge_q4k.py` converted the routed experts with one importance vector a layer
and said why: "One vector serves all 256 experts of a layer, because the input
distribution is the layer's and not the expert's." That is true of `gate_proj`
and `up_proj`, which all read one `post_attention_layernorm`, and **false of
`down_proj`**, whose input is `silu(gate) * up`. Phase 22 measured the cost of
the pooling and Phase 12's alternative explanation #4 with it; this spends the
correction.

### Priced first, and the price said build it

`tools/imatrix_expert_ceiling.py`, three currencies because a weighting scheme
is *supposed* to lose on the ones it does not optimise, 16 experts a layer:

    layer  currency        plain   pooled   per-expert   pooled/plain  per-exp/pooled
      0    per-expert     0.0630   0.0636     0.0518         +0.9%         -18.7%
      5    per-expert     0.0712   0.0706     0.0682         -0.8%          -3.4%
     20    per-expert     0.0717   0.0711     0.0678         -0.9%          -4.6%
     39    per-expert     0.0187   0.0184     0.0169         -1.7%          -8.1%
      0    unweighted     0.0695   0.0683     0.0895         -1.8%         +31.7%

**The vector the five artifacts carry is nearly inert on this tensor** -- it
moves every currency by 1-4%, which is what a statistic with 1.1-1.7x spread
across channels can do. The per-expert one is worth 3.4% to 18.7%. Layer 0 is
the extreme of both because 41.4% of its channels are dead in the parent
(Phase 20), and the per-expert vector is the only one that knows a channel is
dead *for this expert*.

### One variable, checked by digest

`forge_q4k.py --redo --per-expert --only mlp.switch_mlp.down_proj --from
q6k-full`: 40 modules, 427 s, one `quantize_q4_K` call an expert. The point of
`--redo` is that it makes a one-variable experiment cost 40 modules instead of
315 from `int4-head-linattn` twice over.

`q4k_audit.py` confirms it landed where it should: **exactly 40 of 315 digests
move, and all 40 are `down_proj`.** Their unweighted error rises -- 6.035% ->
6.159% mean, layer 0's 6.93% -> 8.72% -- which is Phase 18's own signature of a
weighting scheme doing its job. Bytes are unchanged to the digit.

### And the instrument reads a coin flip

64 probes, both arms captured on the same tile:

    variant            median      p90      max    best on   top-1 moved
    q6k-full           0.0362   0.2126   4.5238   26 of 56       5 of 64
    q6k-perexpert      0.0347   0.2110   3.3834   30 of 56       6 of 64

    sign test: per-expert lower on 30 of 56 live probes, p = 0.69
    median ratio 1.04x

**Not a reduction of damage, a redistribution of it.** Five groups improve and
three do not:

    group        q6k-full   per-expert          group        q6k-full   per-expert
    longrange      0.0810     **0.0468**        arith          0.0578       0.0743
    format         0.0127     **0.0072**        script         0.0190       0.0331
    rare           0.0350     **0.0274**        morph          0.0350       0.0387
    code           0.0560     **0.0477**
    mixed          0.0175     **0.0123**

and the individual probes move in both directions by more than the medians do
(`script/he-capital` 0.1947 -> 0.0324, `script/ru-capital` 0.0164 -> 0.0461).

The one column that is not a wash is `max`: **4.5238 -> 3.3834**, and the
largest single gap in the head-to-head is +1.14 nats on `arith/ru-7x8` in the
new artifact's favour. That probe is this rebuild's known cliff, and 3.38 nats
is still on it -- the variant moved along the cliff, not off it. Worth one line,
not a verdict.

### What it settles

  - **The premise is corrected by measurement rather than by argument.** The
    comment in `forge_q4k.py` was wrong for one of its three modules, and the
    file now says so and can do the other thing.
  - **An imatrix where there was none is a bigger step than a better imatrix
    where there was a weak one.** Phase 18 went from *no vector at all* on 48
    modules to a vector and Phase 17's loss disappeared. This goes from a
    nearly-inert vector to a good one and nothing moves. The lever has
    diminishing returns and this is the far end of them.
  - **The pricing predicted the null exactly**, which is the part worth
    keeping: 3-18% of the objective, against an instrument whose own draw
    spread swallowed Phase 12's +3.5%. Spending the forge was still right --
    it cost 427 s and it is the difference between a prediction and a result --
    but the prediction was not wrong.

`ornith-1.5-35b-a3b-q6k-perexpert` stays on disk (296 GB free) and
`refs/artifact/q6k-perexpert` is its capture. **`q6k-full` remains the
artifact.**

---

## Phase 25 -- the tile told its own shape, and it changes nothing -- **DONE**

Phase 23's open item, built. `K`, `O` and `M` reach the tile as `constant int&`,
so it is compiled for a shape it does not know; all 13 of this artifact's
k-quant shapes satisfy `O % 64 == 0` and `K % 256 == 0`, and `K` takes three
values. Three function constants say so: `kq_aligned_o`, `kq_aligned_m` and
`kq_k`, with `kq_k == 0` meaning "read it from the buffer" so that the
specialisation has a control.

Bit-identical, as a claim about dead code must be -- a test held the two arms
to `max_abs == 0.0` at seven row counts. And then, in one process, alternating:

    unshaped / shaped        M = 64   128    256
    in_proj_qkv [8192, 2048]   1.01  1.02   1.03
    out_proj    [2048, 4096]   0.99  1.01   0.99
    q_proj      [2048, 2048]   1.00    --   0.96
    down_proj   [2048,  512]   1.01    --   1.05

**Scattered around 1.00.** Reverted.

### Why it was never going to pay, which the code says if you read it for cost

Two things were claimed and only one of them is true.

  - "The tail clamps run on every block" -- true, and **they run once per
    block, not once per K step.** `nr0`, `nr1`, `lr0` and `lr1` are all
    computed *before* the K loop; the loop body never re-evaluates them. A
    handful of instructions against a tile that runs `K / 32` iterations of a
    body carrying two `threadgroup_barrier`s.
  - "The K loop cannot be unrolled" -- true and worthless. The bound test is
    one compare against that same body. Unrolling across a barrier buys
    nothing, which is why the compiler's refusal costs nothing here and its
    refusal to unroll **the accumulator loop** cost 12-23x: that one is inside
    the innermost loop and its operands are registers.

The general shape: **a codegen fact is only worth what the thing it applies to
costs.** Phase 23's 20x cliff made codegen look like the live lever in this
kernel, and it is -- in the innermost loop, not at the tile's edges.

### And the measurement corrected itself, which is the other finding

The first reading of this said **1 to 5%**, and it was taken by comparing a
run against numbers recorded an hour earlier. In one process, alternating the
two arms, the same shapes give 0.96 to 1.05. Across-run deltas of a few percent
on this machine are the machine -- which is exactly why every flag this
rebuild has added is a *runtime* switch rather than a build one, and this phase
is the first time that rule caught something rather than merely being obeyed.

`HANDOFF.md` carried this as an open item; it does not any more.

---

## Phase 26 -- FR-Spec on a k-quant head, and the depth knee moves to 1 -- **DONE**

Measuring where the port stood after Phase 25 turned up something no document
carried: **`--frpec` could not run on a k-quant artifact at all.** It panicked
in `linear.rs`, deliberately -- "this module is a k-quant; the caller wants
mlx's three-tensor affine layout" -- because pruning `lm_head` is a row gather
and nobody had written one for three tensors.

So the two largest wins in this project were mutually exclusive. `q6k-full`
buys -13.1% of a decode step; FR-Spec bought -27% of one on the affine
artifact; and no artifact could have both.

### The gather is a row copy, and the layout makes it exact

A Q4_K row at this model's 2048-wide head is **eight whole super-blocks**, so
the pruning never has to understand the 12-byte scale packing or the nibble
interleave -- they are inside the blocks being copied. Three row gathers, of
which the port already had two:

    affine    weight + scales + biases
    nvfp4     weight + scales (uint8 E4M3, no zero point)
    k-quant   qs (uint32) + scales (uint8) + dm (float16)

`gather_rows_by_id_float16` is the one new instantiation. `frspec::prune`
replaces the panic, and `tests/q4k.rs` holds it to the contract that matters:
a pruned row's logit is **the same number** the full head gives for that token,
not a close one. Anything else is a wrong token wearing a fast one's clothes.

And the property FR-Spec rests on survives the move, measured rather than
argued: `--mtp --greedy` prints **byte-identical** text with the pruned head
and without it, at depth 1 and at depth 3. Every committed token is the verify
forward's own argmax, so the draft decides how many tokens a cycle commits and
never which.

### What it bought, and the knee is not where it was

`q6k-full`, 300 tokens, greedy, same prompt, GPU seconds:

    config                        wall tok/s   GPU ms/token   acceptance
    autoregressive                   108.9         8.70          --
    `--mtp` (depth 3)                 94.5        10.08     0.69 / 0.57 / 0.53
    `--mtp --frspec` (depth 3)       102.1         9.49     0.68 / 0.56 / 0.54
    `--mtp-depth 2 --frspec`         114.9         8.40     0.68 / 0.55
    **`--mtp-depth 1 --frspec`**   **127.4**     **7.51**       **0.72**

**-13.7% of a decode step's GPU work and +17% on the wall against
autoregression**, reproducible to 0.4% across draws. Without the gather the
best available on this artifact was autoregression itself, and every
speculative configuration lost.

### Why the knee moved from 3 to 1, which is the finding

On the affine artifact Phase 19's numbers put the knee at depth 3
(`--mtp-depth 3 --frspec`, 8.55 GPU ms/token against autoregression's 10.49).
Here depth 3 is the *worst* speculative configuration and depth 1 is the best.

The mechanism is arithmetic. **The trunk got 13.1% cheaper and the draft did
not.** The MTP sidecar is 1.7 GB of bfloat16 and no phase of this rebuild has
ever quantized a byte of it -- Phase 4's coverage map does not include it, and
`q6k-full` converts the trunk alone. So every draft level costs what it always
cost while the thing it is amortised against shrank, and the depth at which one
more level stops paying comes down with it.

Which looked like it named the next lever -- the MTP head is the only part of
this artifact still at bfloat16 that a decode step reads every cycle -- and
**that reading is wrong, measured the same day.**

### The draft is 4.4x off its byte budget, so its bytes are not the lever

The cycle decomposes, and the decomposition checks itself:

    cycle at depth 3   21.57 ms = 3 drafts + one verify
    cycle at depth 1   12.95 ms = 1 draft  + one verify
    => one draft        4.31 ms
       one verify       8.64 ms   -- against autoregression's 8.70, which is
                                     the check: a verify reads the trunk once

And a draft's bytes are small. The sidecar is 1.689 GB of which 1.611 GB is its
256 experts, so a one-token forward reads **129 MB** of it (top-8 of 256 plus
the 78.7 MB that is not experts), and the pruned `lm_head` adds **75.5 MB** at
Q4_K. 204 MB against a decode step's 1805.

    a draft reads   11.3% of a decode step's bytes
    a draft costs   50%  of a decode step's time
                    ---- 4.4x its byte budget

**So the draft is not bandwidth-bound and quantizing the MTP head is worth at
most a tenth of it**: 129 MB at 4.5 bits is 36, saving 93 MB, which at the
trunk's own byte-to-time rate is 0.45 ms of the 4.31.

The lever is wherever the other 3.9 ms goes. `ORNITH_COMMIT_EVERY=1` attributes
GPU time per kernel and shows it **spread thin rather than concentrated** --
the MTP head's own bf16 gemvs and its expert gather come to ~0.6 ms a token
across 82 dispatches -- which is the signature of a forward that is latency and
occupancy bound, not one waiting on weights. A 256-expert MoE routing a single
token is the worst shape this model has, and the draft runs one every level.

**Naming a lever from an arithmetic sketch and then measuring it the same day
is the only reason this is a paragraph and not a phase spent building it.**

### The correction this phase makes to `HANDOFF.md`

That file's speculative-decoding section is written against the affine
artifact and says depth 3 is the knee. It is, there. On the artifact this
project actually ships it is depth 1, and the two disagree for a reason that
belongs in the file rather than in a table: quantizing the trunk changes what
the draft is worth.

---

## Phase 27 -- the 3.9 ms was never in the draft -- **DONE**

Phase 26 decomposed a speculative cycle by differencing two depths:

    cycle at depth 3   21.57 ms = 3 drafts + one verify
    cycle at depth 1   12.95 ms = 1 draft  + one verify
    => one draft        4.31 ms

and then priced the draft's bytes at 204 MB against a decode step's 1805 --
11.3% of the bytes for 50% of the time -- and wrote down that the lever was
"wherever the other 3.9 ms goes", with a 256-expert MoE routing one token as
the suspect.

**That subtraction charges the whole difference to the draft levels, and it is
only right if a verify costs the same over a four-token window as over a
two-token one.** Nothing had measured that, and nothing could: `ORNITH_PROF`
attributes GPU time per kernel over a whole run, and a speculative run is a
draft *and* a verify sharing every kernel name they use.

`src/bin/draft_cost.rs` runs each half on its own, in one process, on one set
of weights. `q6k-full`, 44-token prompt, eight committed tokens first, GPU
seconds. One run, and the four-run spread beside it -- the machine moves the
level of all of these together by a few percent between sittings and moves
none of the ratios:

    one draft level (depth 1, with FR-Spec)      0.587 ms   0.53 - 0.61
    an autoregressive decode step                8.342 ms   7.93 - 8.34

    a verify forward, by window width                      dispatches
      width 1                                    8.217 ms       1756
      width 2                                   11.368 ms       2156
      width 3                                   14.733 ms       2426
      width 4                                   18.149 ms       2696
    each extra row                             3.15 - 3.42      3.0 - 3.9

    Phase 26's subtraction, on these numbers:
      (cycle(3) - cycle(1)) / 2 = 3.977 ms (3.98 - 4.03 over four runs),
      against a draft level of 0.587 measured directly

**So the arithmetic reproduces -- Phase 26 read 4.31 and this reads 3.98-4.03
-- and the attribution was wrong by about sevenfold.** A draft level is
0.587 ms. The other 3.4 ms is the verify forward, and it is paid *per row of
the window*.

The reason the subtraction lands where it does is arithmetic once it is
written out: `cycle(3) - cycle(1)` is two draft levels **plus the window's
third and fourth rows**, so halving it gives `draft + (row3 + row4) / 2` --
0.587 + 3.39 here. Phase 26 read that whole quantity as the draft.

### The draft is not slow, and it is not 4.4x its byte budget

Its blocks, each run on its own with the head's own weights (best of three
passes; a block is a few hundred microseconds with a sync on each side, which
is exactly the size this machine's interference comes in):

    whole level                            0.587 ms   100%
      head.hidden                          0.375       64%
        prefix: embed, two norms, fc       0.029        5%
        MoE block, 256 experts             0.167       28%
          router gate + softmax            0.006
          top-k: argsort 256, slice, norm  0.014
          switch_glu: 3 gather_mv, 8 experts 0.105
          shared expert, mix               0.011
        attention + residuals              0.179       31%
      lm_head, pruned to 64K rows          0.141       24%
      (lm_head, full, for comparison       0.593)

204 MB in 0.587 ms is **347 GB/s**, against the 219 GB/s a decode step
sustains (1805 MB in 8.22) and the 465 this machine gives a long read. So the
draft reads **11.3% of a decode step's bytes in 7.1% of its time** -- it runs
at 1.6x the trunk's byte rate, not 4.4x under it. Phase 26's ratio divided a
wrong numerator by a right denominator. Quantizing the MTP head would save at
most 93 MB of 204, i.e. ~0.27 ms of 0.587 -- of a 12.0 ms cycle, **2%**.

**The hypothesis this phase was opened on is refuted with it.** A 256-expert
MoE routing one token is indeed the worst shape this model has, and it costs
0.167 ms: 1.4% of a depth-1 cycle. Making the draft's MoE free would not be
measurable in a run.

### Where the window's rows go, per kernel

`ORNITH_COMMIT_EVERY=1`, so each command buffer holds one dispatch and its
timestamps belong to one kernel. That inflates a verify from 8.22 to 11.47 ms
and inflates small kernels most; read the growth, not the totals.

    ms per verify           width 1   width 2   width 4   dispatches/verify
    q4k_qmv                   2.124     3.010     6.014   91, unchanged
    q4k_gather_qmv            1.652     2.516     4.257   96, unchanged
    q6k_qmv                   1.164     1.446     2.183   104, unchanged
    q6k_gather_qmv            0.443     0.734     1.468   24, unchanged
    slice_inner               0.503     1.170     1.549   140 -> 320 -> 500
    gated_delta_packed        0.293     0.596     1.267   30 -> 60 -> 120
    gemv (in_proj_a/b)        0.265     0.483     0.921   60 -> 120 -> 240
    concat_time               0.117     0.242     0.417   30 -> 60 -> 120
    compute_g                 0.107     0.243     0.380   30 -> 60 -> 120
    whole verify             11.469    15.338    23.468

The +12.0 ms splits three ways, and the first column of the split is the one
that matters:

    dense k-quant mat-vecs      +4.91   41%   same dispatch, same weights,
                                              read once per row
    routed expert gathers       +3.63   30%   8 rows a token -> 32; ~23 of
                                              those 32 are distinct experts
    gated-delta, per token      +3.43   29%   `forward_capture` steps the
                                              recurrence one token at a time

**The dense half is a pure re-read.** `q4k_qmv` is dispatched 91 times whatever
the width, and takes 2.8x as long at four rows as at one. ggml's mat-vec
carries the token axis on the grid, so an M-row call is M independent
mat-vecs: the weight matrix is read M times. `src/bin/window_mv.rs` measures
it one module at a time, and the shape is unmistakable:

    module (q6k-full)          M=1     M=2     M=4     M=8    M=16
    out_proj, 4.7 MB          1.00    1.95    2.99    7.88   15.26
    q_proj, 13.8 MB           1.00    1.92    3.79    7.53    7.58
    lm_head, 286 MB           1.00    1.68    3.22    6.57   10.23
    in_proj_qkv, 9.4 MB       1.00    0.45    0.87    1.72    2.72

(The last row is not monotone and is the exception that marks the instrument:
9.4 MB stays resident across the sixteen repetitions this bin makes, so its
extra rows are served from cache, and its M=1 point is slower than its M=2 one
because a single row does not fill the machine. In a real forward each module
is read once and evicted by the next, which is why the end-to-end sweep above
is the measurement of record and this one is the optimistic end of it.)

### Three things that are not the fix

- **The grid axis order.** mlx's affine `qmv` dispatches `(rows, output
  tiles)` and ggml's k-quant mat-vec dispatches `(output tiles, rows)`;
  threadgroups are scheduled x-fastest, so the first has the rows of one tile
  adjacent and reading the same lines, and the second sweeps the whole matrix
  once per row. That is a two-line change -- a function constant picking which
  of `tgpig.x`/`tgpig.y` is the row -- and it moves no arithmetic at all.
  Built, held bit-identical (the outputs compared element for element at every
  width), measured in one process on both instruments: **1.00 on every module,
  and a verify of 8.217 / 11.368 ms at widths 1 and 2 against 8.273 / 11.399
  with the rows on x.** Reverted. The order threadgroups are issued in is not
  the order they run in at this occupancy.

- **The tile.** `kquant::qmm_t` reads the matrix once whatever M is, and the
  measurement says so exactly -- q_proj 0.173, 0.173, 0.174, 0.173 ms at M =
  1, 2, 4, 16. But its floor is 7.5x the mat-vec's M=1, so it only crosses
  over at 16 rows, which is where `MM_ROW_LIMIT` already puts it. At a verify
  window it is 2x *slower* than the re-reads it would replace. The routed tile
  is worse: it runs its K loop once per run of one expert, and a window's runs
  are one row long.

- **mlx's `qmv_wide`**, which is the kernel written for exactly this ("several
  input vectors share one weight read") and is what the affine artifact takes
  with `ORNITH_INVARIANT=0`. On `int4-head-linattn`'s `q_proj`, M=4 costs
  0.051 ms against row-by-row `qmv`'s 0.089 and one row's 0.020 -- 0.57x the
  re-reads, and still 2.6x one row. Sharing a weight read between five rows
  is not the same as reading it once, and `qmv_wide` is also a different
  association (eight lanes reducing K instead of thirty-two), which is why
  the window mode turns it off in the first place.

### What the lever is, and what it is worth

A **multi-row k-quant mat-vec**: read each weight block once and accumulate it
into M separate row accumulators, leaving every row's accumulation order
exactly as the one-row kernel has it. That last clause is what makes it worth
writing rather than reaching for the tile -- Phase 21's window mode is on by
default precisely because a verify row must compute what it would have
computed alone, and a tile or a `qmv_wide` breaks that. A per-block weight
load hoisted across rows does not: the `ib` loop order, the lane split and the
`simd_sum` are untouched, so the two agree bit for bit by construction.

Priced by subtracting the dense re-read from the **measured** cycles in the
next section, which is the honest form of this arithmetic: 41% of a row's
3.31 ms is 1.36 ms saved per extra window row, and nothing else moves. The
third column takes the other two thirds as far as they go -- the expert
gather charging for the ~23 distinct experts a four-token window names rather
than its 32 rows, and one recurrence dispatch writing all `T` snapshots
instead of one per token, call it half of that share, which brings a row to
~1.2 ms.

    on this prompt, GPU ms a token   today   dense re-read back   all of it
    autoregressive                    8.56        8.56              8.56
    --mtp-depth 1 --frspec            7.53        6.73 (-21%)       6.28 (-27%)
    --mtp-depth 2 --frspec            8.37        7.05              6.31
    --mtp-depth 3 --frspec            9.67        7.86  (-8%)       6.84 (-20%)

**Depth 3 stops losing, and the knee does not move.** That is worth stating
plainly, because the session that opened this phase predicted the opposite --
"if it confirms, it also takes the knee deeper than one". It does not, and the
reason is not the price of a row but the decay of acceptance. At 0.69 / 0.57 /
0.52 the marginal draft levels are worth 0.69, 0.39 and 0.20 of a token, i.e.
5.91, 3.37 and 1.75 ms of avoided autoregression, against a marginal cost of
one draft level plus one window row plus the sampler's extra row:

    marginal row        worth     costs today   costs with the lever
    second window row    5.91       4.49            3.04  /  2.28
    third                3.37       4.49            3.04  /  2.28
    fourth               1.75       4.49            3.04  /  2.28

A depth pays while `cost / extra tokens` is below the current ms a token. The
third row crosses only in the most optimistic column, and by a hair (depth 2
at 6.31 against depth 1's 6.28); the fourth is never worth 2.28 ms whatever
the kernel does. **A cheaper window widens the win at the depth that already
works rather than buying a deeper one** -- to move the knee, acceptance has to
move, which is the head's business and not the trunk's.

### The loop agrees with the decomposition

The check that these two halves are the same object the runtime runs. Same
prompt, 200 tokens, greedy, `q6k-full`:

    config                     GPU ms/token   acceptance   tokens/cycle
    autoregressive                 8.56           --            1.00
    --mtp-depth 1 --frspec         7.53          0.69           1.695
    --mtp-depth 2 --frspec         8.37        0.67/0.57        2.062
    --mtp-depth 3 --frspec         9.67   0.67/0.55/0.52        2.247

    measured cycle = ms/token x tokens/cycle   12.76    17.26    21.73
    draft + verify, from the decomposition     11.96    15.91    19.91
    unaccounted                                 0.80     1.35     1.82

Depth 2 was measured after the rest of this phase was written, as the check
the two flanking depths could not be: the decomposition predicted its cycle at
15.91 ms of draft and verify plus an overhead interpolated between the other
two depths (1.31), i.e. 17.22 ms, and it came back **17.26**.

The residual is the loop itself and grows with the window as it should: the
sampler takes `depth + 1` rows through `log_softmax` and `argmax` over 248320,
and the `Committed` history policy re-encodes `head.hidden` for every accepted
row. ~0.5 ms a window row, which is worth knowing next to the verify's 3.4.

### The lesson, and it is the same one twice

Phase 26 named a lever from a subtraction and checked the arithmetic rather
than the assumption inside it. The assumption -- "a verify costs the same at
any width" -- was the whole content of the claim, and it was cheaper to
measure than the sentence was to write: one bin, one process, four widths.

`HANDOFF.md` already carries the general form of this ("price the options
before choosing between them"), and the specific form is worth adding: **when
a quantity is obtained by differencing two configurations, the thing it
measures is every difference between them, not the one in the name.** Depth 3
and depth 1 differ by two draft levels *and* by two verify rows, and the rows
cost 5.8x what the levels do.

---

## Phase 28 -- the re-read was free and the dequantize was not -- **DONE**

Phase 27 named the lever and priced it: a multi-row k-quant mat-vec, one read
of a weight block accumulated into M row accumulators, `-21%` a token at depth
1 and depth 3 stopping losing. It is written -- `q4k_qmv_m{2,3,4}` and
`q6k_qmv_m{2,3,4}`, `kquant::qmv_rows` -- and it is **bit-exact and worth
nothing**:

    the verify forward, by window width (GPU ms, `q6k-full`)

    width   a row each   one read for all   per-row cost
      1        7.749          7.534
      2       10.467         10.122
      4       17.164         17.041
    each extra row            3.14  vs  3.14

The contract held exactly as predicted -- `tests/q4k.rs` and `tests/q6k.rs`
compare every row of an M-row call against that row computed alone and measure
`max_abs == 0.0` at M = 2, 3, 4, 5 and 8, and a 200-token run accepts the same
82 of 118 drafts under both kernels, byte for byte. So the part of Phase 27
that was about *invariance* was right. The part that was about *time* was not.

### Why: past the first row the kernel is not reading-bound

`src/bin/window_mv.rs` grew an `ORNITH_SERIAL=1` mode -- drain the queue
between repetitions, so the mat-vecs run in a chain the way a verify runs them
rather than sixteen at a time serving each other out of cache. `lm_head`,
286 MB of Q4_K, one dispatch at a time:

    M    a row each    one read for all    bytes read   GB/s at that time
    1       0.550           0.538            286 MB          520
    2       0.828           0.767            286 MB          373
    3       1.240           1.124            286 MB          254
    4       1.651           1.488            286 MB          192
    8       3.321           2.963            286 MB           97

**A single row already runs at 520 GB/s, which is this machine's ceiling**
(546 theoretical, 465 on a long read). Four rows through a kernel that reads
the matrix *once* take 2.7x as long as one row while reading the same bytes.
The shape of the whole table is `max(one read, M x arithmetic)`: 0.55 ms of
traffic against 0.365 ms of arithmetic a row, so the read wins at M = 1 and
nothing else. The three reads the new kernel saves were already hidden
underneath the arithmetic that consumes them -- and the old kernel's re-reads
were partly served from cache besides (at M = 4 it sustains 694 GB/s of
apparent traffic, which no DRAM here can do).

So Phase 27's 41% was real and its explanation was not. What grows with the
window in that kernel is not the traffic; it is the work.

### What does pay, and only on one of the two formats

The other thing that repeats per row is the **dequantize**, and unlike the
read it is not hidden. So: unpack a block's codes into registers once and
spend them M times. Both formats got it (`q6k_qmv_mh*`, `q4k_qmv_mh*`,
`kquant::qmv_hoist`), both are bit-exact, and they do not agree. Per kernel
inside one width-4 verify, `ORNITH_COMMIT_EVERY=1`, two draws:

    ms a verify        a row each    one read     + unpacked once
    q6k dense mat-vec   2.118/2.112  2.505/2.475  **1.677/1.653**
    q4k dense mat-vec   5.297/5.372  **5.220/5.192**  5.789/5.322

Q6_K unpacks five operations a weight -- two masks, a shift, an or and the
bias of 32 -- against the one multiply-add that spends it, so hoisting it is
most of the inner loop: **-21%**. Q4_K unpacks one mask and one conversion,
and the hoist costs more than it saves: holding 32 unpacked weights and 32
activations at once does not fit, so the output row has to move outside the
row-of-`x` loop, which doubles the activation reads to one per multiply-add.
**+9%.** `ORNITH_QMV_HOIST` therefore defaults to Q6_K alone, and `=all`
keeps the Q4_K kernel reachable so the next session can re-measure it instead
of re-deriving it.

### What it is worth

Two draws each, 200 tokens, greedy, GPU ms a token. This machine reads 7.56
for autoregression today where Phase 27 read 8.56 -- it is shared, and it
moves every level together; the ratios are what carry.

    config                   a row each    this phase
    autoregressive              7.56          7.56
    --mtp-depth 1 --frspec   6.78 / 6.80   **6.60 / 6.60**    -2.8%
    --mtp-depth 3 --frspec   8.72 / 8.72   **8.28 / 8.30**    -5.0%

Against a prediction of -12% -> -21% at depth 1 and +13% -> -8% at depth 3.
The knee stays at depth 1, which Phase 27 did get right and for the right
reason: a cheaper row cannot outrun a geometric decay in acceptance.

### Which compiler these numbers came out of

Every table above was measured on Xcode 26.3's `metal`. Xcode **27.0** landed
on this machine an hour later (a background install, and it also moved the
Metal compiler into a separately downloaded component), so the suite and the
A/B were re-run on `metalfe-32023.921.6`. **The bit-exact floor did not move**
-- the whole suite is green, including `q4k_dequantize` against ggml's own
bytes at `max_abs == 0.0`, and a 200-token run still accepts the same 82 of
118 drafts. Neither did the ordering the code rests on, measured under a
neighbour's VM at 350% CPU, so read the ranks and not the levels:

    ms a verify, width 4     a row each   one read   + unpacked once
    q6k dense mat-vec           2.136       2.529      **1.839**
    q4k dense mat-vec           5.959     **5.228**      5.406

The magnitudes are a different question from the ranks and this machine could
not answer it that hour: `--mtp-depth 1 --frspec` read 7.78 GPU ms a token
against the 6.78 of an hour before, with both kernels inside the noise of each
other. The `-2.8%` and `-5.0%` above stand as measured on 26.3 on a quieter
machine, and they are the numbers to re-take first if this phase is ever
doubted. `target/kernels-xcode26.3.metallib` is the old compiler's output,
kept for exactly that comparison.

### The lesson, and it is Phase 27's own, one level down

Phase 27 caught Phase 26 charging a difference between two configurations to
the one thing in its name, when two things differed. Then it wrote:

> `q4k_qmv` is dispatched 91 times whatever the width, and takes 2.8x as long
> at four rows as at one. [...] **The dense half is a pure re-read.**

A fixed dispatch count says the work *inside* a dispatch grew. It does not say
which work, and **both** the traffic and the arithmetic grow with M. The
sentence picked one of the two and the next paragraph priced it. The only
instrument that separates them is a kernel that removes one and leaves the
other -- which took a day to write, a minute to run, and says the removed one
was not the cost.

Generalised, and it is worth carrying: **a quantity that grows with a
parameter is not evidence about *which* term grows with it.** Phase 26
differenced two configurations, Phase 27 differenced two widths; the same
correction applies both times, and the second time it was made by the phase
that named the first.

What survives from Phase 27 unchanged: the draft is 0.587 ms and not 4.31, its
MoE is 1.4% of a cycle, the tile and `qmv_wide` and the grid axis order are
refuted, and the window's other two thirds -- the expert gather paying for 32
rows where ~23 are distinct, and `forward_capture` stepping the recurrence per
token -- have not been touched. Those two are now the whole of what is left in
a window row, and nothing here says what *their* growth is made of either.

---

## Disk

84 GB free of 1.9 TB at the time of writing, so the choreography is tight:

**This constraint came back, 2026-09-18.** It was written against 84 GB free,
then relaxed when the disk had **592 GB** -- and after Phases 7-15 forged six
more artifacts at ~26 GB each there are **103 GB** left. Variants still *can*
be compared side by side, which is what made Phase 15 possible, but building
two more now means deleting one first. `ornith-1.5-35b-a3b-q4k-twin` is the
cheapest to lose: it was a stand-in for a format that now exists for real.

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
- **A defect in the parent itself**, which no check here looks for: every test
  compares the port against MLX on the same weights, and `check_rebuild.py`
  checks shapes and counts. Two have turned up anyway, both because a number
  came back too strange to be a result -- 2.9.1's unshifted MTP norm gain, and
  layer 0's 41%-dead experts in Phase 20. Neither was caught by machinery.
- **A quality regression from the format.** Caught by perplexity against the
  Phase 0 baseline -- which is why Phase 0 cannot be skipped.
- **A port bug in the new kernels.** Caught by `max_abs == 0.0`, as usual.
  This is the one failure the existing machinery already handles.
