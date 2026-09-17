# ornith-rs — state of the port

A from-scratch Rust + Metal port of the MLX inference path for
`~/.mtplx/models/Shiftedx--ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx`.

The point of the exercise is to **understand how the model works**, so the port
stays 1:1 with the sources in `../vendor/` (mlx, mlx-lm, MTPLX), keeps every
optimization those take, and is verified against MLX numerically rather than by
reading the generated text.

**That 1:1 state is pinned at `261fe0c`, and is no longer a requirement past
it.** Everything up to that commit does what mlx does; from there a departure
is an ordinary engineering decision, taken when it pays and written down here
with what it bought. What does *not* move is the verification: `max_abs ==
0.0` against MLX is how every one of these changes is shown to be a
rearrangement and not a new model. A reordering that changes a number is a
bug, not a trade-off.

**The model talks, and it looks.** `cargo run --release` opens a chat; greedy
decoding and sampled decoding both reproduce `mlx_lm.generate` token for token
and character for character. `--mtp` adds speculative decoding off the native
MTP head. `--image` runs the vision tower and reproduces MTPLX's description of
the same picture, likewise token for token.

---

## The model

Qwen3.5-MoE (`qwen3_5_moe`), 35B total / ~3B active:

| | |
|---|---|
| Layers | 40: 30 `linear_attention` (GatedDeltaNet) + 10 `full_attention`, every 4th is full |
| Hidden | 2048, head_dim 256, 16 query heads, 2 kv heads (GQA factor 8) |
| Attention | partial RoPE over the first 64 of 256 channels, theta 1e7, plus a sigmoid output gate (`attn_output_gate`) |
| Linear attention | conv kernel 4, 16 key heads x 128, 32 value heads x 128, float32 recurrent state |
| MoE | 256 experts, top-8, `moe_intermediate_size` 512, plus a shared expert (512) with its own sigmoid gate |
| Quantization | 4-bit/group 32 by default; 8-bit/group 64 for attention, `linear_attn` projections, both router gates, `embed_tokens`, `lm_head`, and the experts + shared experts of 8 of the 40 layers. Per-module, read from `config.json`'s `quantization` block |
| BF16 (unquantized) | `conv1d.weight`, `A_log`, `dt_bias`, `in_proj_a/b`, every norm, the whole MTP head (1.7 GB) and the whole vision tower (0.9 GB) |
| MTP | one appended NextN predictor: `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `fc` (2H -> H), one full-attention decoder layer with its own 256-expert MoE, `norm`; lm_head is shared with the trunk. Sidecar `mtp/weights.safetensors` |
| Vision | 27-block ViT, hidden 1152, 16 heads of 72, intermediate 4304, 3D patch embed (2x16x16x3), interpolated 48x48 position embedding, 2D RoPE, LayerNorm, two GELUs, spatial merge 2, out 2048. Every linear carries a bias, which nothing else in this model does |
| Tokenizer | BPE, 248044 entries + 33 added tokens, NFC, the GPT-4 split regex, ByteLevel |

Artifact quirks that matter:

- The checkpoint is **already sanitized**: `conv1d.weight` is `[D, K, 1]`, so
  mlx-lm's `TextModel.sanitize` does **not** add 1.0 to the norm weights. Doing
  it anyway is the "double shift" MTPLX's `qwen3_5_mtp_patch.py` warns about.
- `rope_parameters.type` is `"default"`, so despite the `mrope_section` next to
  it the text path takes plain rotary, not mrope. MTPLX agrees from the other
  side: `openai.py` builds an M-RoPE table only when
  `spec.model_type == "qwen4_exp"`, and this artifact says `qwen3_5_moe`. So
  image tokens are roped sequentially, like any other token.
- Tensor offsets in the safetensors are **not** aligned (odd byte offsets), so
  weights must be copied into device buffers; zero-copy is impossible.
- **The tokenizer has two behaviours and they disagree.** See below.

## This machine

`applegpu_g16s` (M4 Max, 64 GB). Two consequences, both asserted in code so a
different Mac fails loudly instead of silently diverging:

- `metal::is_nax_available()` is false (needs gen >= 17), so none of mlx's
  Metal 4 / `mpp::tensor_ops` kernels run. They are not ported.
- `GEMM_TPARAM_MACRO` classes this as a "medium device", so the steel gemm tile
  is always 64x64x16 / wm=wn=2. Only that tile is instantiated.

### Metal 4.1, fp8/fp4, and what changes on an M5

`M5.md` is the working sheet for the move, with mlx's routing conditions
quoted line by line. Its headline is worth having here: **every route into the
`*_nax` kernels is guarded by a width a decode step does not have**, so they
are a prefill lever. This section is the other half, the low-precision
formats.

Metal 4.1 does expose fp8 (E4M3), fp4 and block-scaled MXFP4 -- in the tensor
path, the MPP `matmul2d` operation over `cooperative_tensor` fragments. Two
gates keep it off this machine, and both are worth knowing before the move:

- **The toolchain.** `xcrun metal -std=metal4.1` answers `invalid value
  'metal4.1'` under the Xcode 26.5 SDK; `build.rs` compiles at
  `-std=metal4.0`, and the 4.1 low-precision formats are absent under the 4.0
  toolchain.
- **The OS.** Metal 4.1 needs Xcode 27 *and* macOS 27.0, not the "Xcode 26.1+"
  the specification claims, and a 4.1 binary refuses to load below that. This
  machine runs macOS 26.6.2.

And on an M4 the formats are **emulated rather than accelerated**: fp8
`matmul2d` sustains 0.94x the throughput of fp16 while reading half the
operand bytes, `matmul2d` executes on the GPU shader cores through the legacy
`simdgroup_matrix` path with no dedicated matrix datapath and no sign of ANE
routing, and it accumulates in at least fp32. Those are measurements on this
exact part -- M4 Max, 40 GPU cores, 64 GB -- from *Rigel: Reverse-Engineering
the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU* (arXiv 2606.12765),
run on macOS 27.0.

**The inversion to remember.** "A memory-footprint feature, not a performance
feature" is a verdict from compute-bound GEMM. A decode step here is the
opposite: 95-96% GPU busy, waiting on 2.83 GB of weight reads at ~272 GB/s of
a possible 546. Halving operand bytes for 6% of the arithmetic is, at batch 1,
close to a win proportional to the bytes saved. Footprint *is* performance
here, which is exactly why the paper's dismissal does not transfer.

The same paper's optimization study is the other way round and confirms
something this port measured independently: a hand-fused GEMM+bias+GELU beats
the decomposed path by 6.5-12.9% in the cache-resident regime, the win decays
as the O(n^3) compute dominates, and it "does not extend to operations that
forgo the matrix units". Decode forgoes them entirely -- see the reverted
router fusion below, which bought half a percent.

**None of that is needed for the byte win, though.** mlx already carries
`fp_quantized.metal` for `Mxfp4` / `Mxfp8` / `Nvfp4` (uint8 E8M0 scales, no
biases, group size fixed at 32, or 16 for nvfp4) as a software path whose own
source note reads "fp benefits on every gen"; `fp_quantized_nax.metal` is the
Metal 4 tensor variant. Per weight: affine int4 g32 costs 5.00 bits, MXFP4
4.25, nvfp4 g16 4.50, affine int8 g64 8.50, MXFP8 8.25. Against this model's
mix that is ~5% fewer bytes a token -- worth little on its own, and worth a
lot only if MXFP4 holds quality at 4 bits where affine int4 did not, since
that is what would let the 8-bit half of the artifact drop to 4 and take ~35%
of the traffic with it. This port carries only the `affine_*` family, so going
there means porting `fp_*` whole.

**The checklist for the M5, in order.**

1. `nax_available()` flips true (gen >= 17, or >= 18 on a 'p' part) and three
   asserts fire first, by design: the gemm and the gather gemm in
   `src/ops/quantized.rs`, and `dispatch_choices_for_this_model`. That is the
   signal that mlx would now pick kernels this port does not carry -- porting
   the `*_nax` family becomes a decision rather than an accident.
2. The steel gemm tile. `GEMM_TPARAM_MACRO` keys off the last letter of the
   architecture name; if an M5 classes differently from "medium", a new tile
   has to be instantiated in `kernels/steel_gemm_inst.metal`.
3. `qmv_batch_limit` already carries the gen-17 branch (33/25/13 where gen 16
   gives 13/15/13). `tests/quantized.rs::dispatch_choices_for_this_model`
   skips itself off gen 16 and says so -- re-pin those numbers there.
4. The two-pass SDPA routing and `sdpa_blocks` both key off the architecture's
   last letter, so a different device class changes when attention splits.
5. Re-measure everything in "Where a decode step goes". The Rigel numbers are
   M4 numbers; the M4 predates the Neural Accelerators that arrived with the
   A19 and M5, so the emulation verdict in particular has to be re-measured on
   the new part rather than carried over.

---

## Layout

```
kernels/mlx/...            mlx's Metal sources, verbatim, original paths kept
kernels/*.metal            our instantiation lists + the kernels mlx JITs
src/metal/                 device, pipeline cache, buffer pool, encoding
src/{json,safetensors,array,weights}.rs   loading
src/ops/                   kernel dispatch, one module per mlx primitive family
src/model/                 config, caches, layers, the MTP head
src/vision/                the tower, its preprocessing (incl. Pillow's
                           bicubic resampler) and the prompt splice
src/tokenizer/             NFC, the split regex, BPE, the added vocabulary
src/chat.rs                chat_template.jinja, hand-compiled
src/sample.rs              make_sampler: top-p, top-k, categorical
src/generate.rs            generate_step: prefill chunking and the decode loop
src/mtp_generate.rs        the MTP draft/verify loop
src/frspec.rs              the pruned draft head (FR-Spec)
src/main.rs                the console chat
src/refs.rs                golden-reference reader
tools/dump_*.py            golden-reference writers (run against MLX)
tools/epoch_schedule.py    how few barrier epochs a step's dependencies allow
M5.md                      what changes on a GPU where nax_available() is true
REBUILD.md                 rebuilding the artifact from its bf16 parent
IDEAS.md                   what designing the artifact buys, with the arithmetic
tools/quant_error.py       every quantization number in those two, reproducible
tools/gen_unicode.py       the Unicode tables the tokenizer needs
refs/<case>/               the references themselves (not in git)
tests/                     one test file per reference case
```

`build.rs` compiles every `.metal` under `kernels/` into one metallib with
mlx's own flags plus `-fmetal-math-mode=safe`, and `src/metal/mod.rs` embeds it
with `include_bytes!`.

---

## Running it

```
cargo run --release                          # chat, temp 0.6 / top_p 0.95 / top_k 20
cargo run --release -- --greedy              # temperature 0
cargo run --release -- -p "..." -m 200       # one shot
cargo run --release -- --raw -p "..."        # no chat template
cargo run --release -- --tokenizer-json ...  # tokenizer.json's split regex
cargo run --release -- --mtp                 # speculative decoding, depth 3
cargo run --release -- --mtp --mtp-depth 1 --frspec   # the fast pairing
cargo run --release -- --mtp-depth 1         # shallower draft, see the table below
cargo run --release -- --image p.bmp -p "..." # one image (repeat for more)
cargo run --release -- --image p.rgb --image-size 768x448
```

There is no image *decoder* in Rust, on purpose: `--image` takes a binary PPM,
an uncompressed BMP, or raw RGB. macOS ships the decoder:

```
sips -s format bmp photo.heic --out photo.bmp
```

Any size is fine -- `smart_resize` and Pillow's bicubic resampler are both
ported, so the picture is resized exactly as MTPLX would have resized it.

Loading is ~16 s warm (mmap + per-tensor copy into Metal buffers), ~70 s cold.
Decode runs at ~92 tok/s, greedy and sampled alike -- the sampler's two
248320-wide argsorts cost about a percent, which is well inside the noise.

---

## What is done and verified

All tests assert **`max_abs == 0.0`** against MLX -- bit-exact, not a
tolerance. Both sides run the same kernels, so any difference means the
dispatch diverged.

| Test | Covers |
|---|---|
| `quantized.rs` | `quantized_matmul` on real weights, 4-bit/g32 and 8-bit/g64, M = 1/2/4/8/16/64/256 (qmv_fast, qmv_wide, qmm_t_splitk + reduce) |
| `ops.rs` | rms_norm (weighted, weightless, head_dim), rope (decode + prefill, offsets), fused elementwise |
| `activations.rs` | silu / swiglu / precise_swiglu / sigmoid over **every** bfloat16 input |
| `attn.rs` | matmul (incl. broadcast batch), softmax, attention at 6 shapes across all three SDPA branches |
| `gdn.rs` | conv concat, depthwise conv + silu, `compute_g`, beta, packed and unpacked recurrence, T = 1/7/512 |
| `moe.rs` | router, top-k, score normalisation, SwitchGLU on all three gather paths, sort permutation, whole block, T = 1/7/24/512 |
| `layer0.rs` | layer 0's gated-delta path step by step on real weights |
| `model.rs` | **the whole 40-layer model**: 24-token prefill + 3 decode steps, checked at the embeddings, layers 0/1/2/3/4/7/8/39, pre-norm residual, final hidden and logits. `#[ignore]`d because it loads 25.5 GB |
| `tokenizer.rs` | the split regex's character classes over **all 0x110000 codepoints**, NFC over every codepoint plus ~15k combining sequences, and `encode`/`decode` over a ~1200-case corpus -- in **both** tokenizer dialects |
| `chat.rs` | 120 conversations through the chat template, compared as rendered text *and* as token ids |
| `resample.rs` | Pillow's bicubic resampler, **byte for byte** against Pillow, at ten (in -> out) size pairs: both axes down, both up, one axis only, a huge downscale, a tiny upscale and the pairs `smart_resize` produces for real photographs |
| `vision_ops.rs` | the ops only the tower reaches: `layer_norm`, both GELUs over **every** bfloat16 input, `addmm` at ten shapes across split-K and the regular steel gemm, the 3D patch convolution, and mlx's *full* fused attention at head dim 72 -> 80 |
| `vision.rs` | **the whole 27-block tower** at three image sizes (256x256, 320x224, 512x512), checked at the patch embedding, the interpolated position embedding, the 2D rotary tables, block 0's norm and attention, blocks 0/1/26 and the merger -- plus the preprocessing, from the file's bytes to the patch rows, including two sizes that `smart_resize` changes and so go through the resampler |
| `vision_generate.rs` | **pixels in, text out against MTPLX**: two photographs (one the processor resamples) and a synthetic image, one and two images per prompt, through the tower, the pad expansion, the splice and greedy decoding. `#[ignore]`d because it loads 26.4 GB |
| `sampler.rs` | every stage of `make_sampler(0.6, 0.95, 20)`: the multi-block merge sort, the bfloat16 and float32 scans, whole-array max/sum, gather/scatter along an axis, searchsorted, threefry, and the drawn tokens off a fixed seed |
| `generate.rs` | **end to end against `mlx_lm.generate`**: 8 greedy prompts (raw and chat, one crossing the 2048-token prefill step) and one sampled run, token ids and printed text. `#[ignore]`d |
| `mtp.rs` | the MTP head against MTPLX's `mtp_forward`: the unquantized 256-expert block, the head's input after `pre_fc_norm` + `fc`, a whole `mtp_forward`, the depth-3 draft chain and the verify window. Then the capturing trunk forward against the packed one, and speculative decoding against autoregressive. `#[ignore]`d |

### Bugs the golden tests caught

None of these would have been visible from output text:

1. mlx sums bfloat16 **in bfloat16**; a float32 accumulator is wrong.
2. Reduction order differs by kernel (`col_reduce_small` sequential vs
   `col_reduce_looped`'s `simd_sum` butterfly) — an ULP apart.
3. mlx's sigmoid is the stable `1/(1+exp(|x|))` form.
4. mlx's JIT (`mx.compile`) and precompiled kernels disagree: `fast::exp` vs
   `precise::exp`. One bfloat16 input in 6660 differs.
5. Host writes raced with encoded-but-unexecuted dispatches through the buffer
   pool.
6. Rounded-up dispatch grids wrote past buffer ends (only visible at T = 24).
7. `in_proj_a/b` take the split-K steel gemm at prefill, not the regular one.
8. The fused SDPA has a `q_len * gqa_factor <= 32` limit on top of the head-dim
   rules, so only 4 query rows stay fused here.
9. `transformers` 5.x does not use the artifact's tokenizer (see below).
10. A bfloat16 expression written as one line is not the same as mlx's chain of
    bfloat16 arrays: Metal keeps `a * b` at float precision unless it is
    assigned to a `bfloat16_t` variable, and mlx materialises every
    intermediate. Nine of the 65536 bfloat16 inputs to `gelu_pytorch_tanh`
    disagree between the two forms.
11. `sync` did not wait when a dispatch landed exactly on a commit boundary,
    so the host could read a buffer whose kernel had not run. Latent at the
    shipped commit size; see below.

### The bug the profiler found first

`Context::sync` used to return early when neither an encoder nor a command
buffer was open -- but "nothing open" is not "nothing running". A dispatch
that lands exactly on a commit boundary leaves both empty while the GPU is
still working, so `sync` waited for nothing and the host read a buffer whose
kernel had not run. It reads as zeros.

At the shipped `max_dispatches_per_commit = 128` a decode step ends 95
dispatches into a buffer, so the boundary never lands on the last dispatch of
a step and nothing ever went wrong -- the whole test suite passes on both
sides of this fix. It is shape-dependent luck: any prompt, image or draft
whose dispatch count is a multiple of the commit size would have read
garbage. `ORNITH_COMMIT_EVERY=1` makes every dispatch land on a boundary, and
`tests/model.rs` then fails on the first thing it checks, the embedding
gather, with `got 0 want -0.287`.

`sync` now keeps the last buffer `flush` committed and waits on that when
nothing is open. The queue is serial, so waiting on the newest buffer means
every earlier one has completed too.

---

## The tokenizer has two behaviours

`tokenizer.json` describes a Split pre-tokenizer with the newer Qwen pattern:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

`transformers` 5.x **ignores it**. `Qwen2Tokenizer` there is a
`TokenizersBackend` that rebuilds the tokenizer from `vocab.json` and
`merges.txt` and installs its own module-level `PRETOKENIZE_REGEX`
(`transformers/models/qwen2/tokenization_qwen2.py`), which is the older Qwen2
pattern with **no `\p{M}`** in either class. It also ignores the
`pretokenize_regex` key `tokenizer_config.json` carries, which holds the
`\p{M}` version.

The two disagree on anything with a combining mark: Devanagari, Arabic
diacritics, Hangul jamo, an emoji with a variation selector. For
`यह हिन्दी में` it is 11 tokens against 19.

mlx-lm loads the tokenizer through `AutoTokenizer`, so **the transformers
behaviour is what `mlx_lm.generate` runs**, and it is what this port defaults
to (`Dialect::Transformers`). `Dialect::TokenizerJson` runs the artifact's own
pattern; `--tokenizer-json` selects it. Both are verified against their own
reference on the same corpus. Which one the model was *trained* with is a
question for the artifact's authors -- the `\p{M}` form is the one Qwen ships.

Two other details of the pre-tokenizer are pinned by probing rather than
assumed, because `tokenizers` runs the pattern through oniguruma:

- `\s` is the Unicode `White_Space` property, not Ruby's ASCII-only `\s`.
  U+00A0 and U+3000 are `\s`; U+200B and U+001C are not.
- `(?i:)` folds with simple case folding, so `'s` also matches `'\u{17f}`
  (LATIN SMALL LETTER LONG S). No other character folds onto `s t r e v m l d`.

`tools/dump_tokenizer.py` re-derives the character classes from the live
pre-tokenizer, one probe per codepoint, and `tests/tokenizer.rs` compares them
against `src/tokenizer/tables.rs`.

---

## MTP speculative decoding

The head is one full-attention decoder layer plus its own 256-expert MoE, and
the whole 1.7 GB sidecar is BF16 -- `mtp_prequantized: false`, so
`_quantize_mtp_module` returns without doing anything and every projection is a
dense `matmul`. That makes the MoE an **unquantized** `SwitchGLU`, which is the
one genuinely new kernel path: `mx.gather_mm` with M == 1 routes to
`gather_mv`, i.e. mlx's `gemv_gather`, where the trunk's quantized blocks route
to `gather_qmv`.

Three things about the wiring are not what the file names suggest:

- **`qwen3_5_mtp_patch.py` does not run.** `is_qwen3_5_mtp_config` needs
  `model_type == "qwen3_5_mtp"` and this artifact says `qwen3_5_moe`, so the
  generic `inject_mtp_support` in `mtp_patch.py` handles it. That matters:
  the qwen3_5-specific patch hardcodes `hidden_variant = "pre_norm"`, the
  generic one takes the contract's `post_norm`.
- **The contract lives in `mtplx_runtime.json`, not `config.json`** -- and it
  is exactly `MTPContract`'s defaults: `post_norm` in and out,
  `embedding_hidden`, `mtp_position_mode: cache`, not prequantized.
- **The history policy that matters is `committed`, not `cycle`.** Several
  dataclasses in `generation.py` default to `cycle`, but `cli.py` passes
  `mtp_history_policy="committed"` and `_resolve_mtp_history_policy` calls
  that "the product default". Measured over three prompts:

  | policy | depth 1 | depth 2 | depth 3 |
  |---|---|---|---|
  | `cycle` | 0.43 - 0.71 | 0.12 - 0.24 | 0.00 - 0.18 |
  | `committed` | 0.53 - 0.84 | 0.31 - 0.57 | 0.00 - 0.27 |
  | `committed` + prompt seeding | 0.54 - 0.79 | 0.31 - 0.55 | 0.00 - 0.25 |

  against the 0.78 / 0.35 / 0.13 the artifact records. `committed` is the
  default here. Seeding the head's cache over the prompt -- what
  `_append_mtp_history(..., phase="prefill")` does -- measurably does not
  help, which is convenient: it is the only thing that would have needed
  mlx's steel *gather* gemm, since a prompt-length append crosses
  `SwitchGLU`'s `indices.size >= 64` sort threshold.

### Rewinding a cache that cannot be trimmed

mlx-lm's own `speculative_generate_step` refuses this model outright: it needs
`can_trim_prompt_cache`, and the 30 gated-delta layers' `ArraysCache` reports
`is_trimmable() -> False`. There is nothing to trim -- their memory is a conv
window and a recurrent state, not a sequence.

`TrunkCapture` is the way round, and it is exact rather than approximate
because of one property of the recurrence kernel: it is a strict
`for (t = 0; t < T; ++t)` loop carrying the state in float32 registers. So
stepping it one token at a time and writing the state out between steps is
**bit-identical** to one T-token call (checked at T = 4, 7, 24 and 64, and
again through all 40 layers in `tests/mtp.rs`). The capturing forward therefore
keeps every intermediate `(conv window, recurrent state)` at no numerical cost
and almost no bandwidth cost -- each step already allocates its own outputs, so
a snapshot is a reference rather than a copy.

### It does not pay off here, and the reason is not the one it looks like

Same prompt, 200 tokens, greedy. The wall column was how this was first read;
the GPU column is what it should have been read by, for the reason in "Where a
decode step goes":

| | wall tok/s | GPU busy ms/token | GPU busy share | acceptance by depth |
|---|---|---|---|---|
| autoregressive | 84.5 | 10.35 | 95% | -- |
| `--mtp-depth 1` | 83.7 | **9.56** | 80% | ~0.53 |
| `--mtp-depth 2` | 72.3 | 10.88 | -- | ~0.53 / 0.34 |
| `--mtp-depth 3` | 64.8 | 12.10 | -- | ~0.53 / 0.34 / 0.35 |

Acceptance is in the band the artifact records (0.78 / 0.35 / 0.13), so the
head and the loop are doing their job. And **depth 1 is already 8% less GPU
work per generated token than autoregression** -- 1466 dispatches against
1669, 863 barriers against 1064 -- while being no faster on the wall, because
it leaves the GPU idle 2.58 ms a token where autoregression leaves it 0.50.

**It stalls, it does not compute too much.** Where the stall is, though, is
not settled -- and the obvious answer turned out to be wrong, which is worth
recording so it is not tried a third time.

The obvious answer was host round trips, of which there were 1.66 a token
against autoregression's 1.00. Two were removed:

- **The draft chain stayed on the host.** Each step ran `greedy_argmax`, which
  reads the token back, and then `Array::from_slice` sent it straight to the
  device again for the next step -- a round trip for a value only the GPU ever
  wanted. `draft` now builds the verify window `[primary, d_0, ...]` in place
  on the device, step `i` reading slot `i` as a view while its argmax writes
  slot `i + 1`, and nothing comes back until the cycle needs it.
- **The accept loop asked one row at a time.** It wants row 0, then row 1 only
  if row 0 agreed, and each of those was a round trip for one argmax over the
  vocabulary -- all latency, no work. When `Sampler::rows_are_independent`
  (`Greedy` does; a stochastic sampler does not, because how many rows it is
  asked for decides its draws) every row is now taken in one pass into one
  buffer and read back once.

Round trips a token fell from 1.66 / 2.20 / 2.69 to 1.31 / 1.21 / 1.14 at
depths 1 / 2 / 3, all of it bit-exact -- `tests/mtp` reproduces the baseline's
acceptance and divergence numbers exactly. **And the idle barely moved.**
Depth 2 gained ~5% of wall, depths 1 and 3 nothing, and the GPU busy share is
still ~80% against autoregression's 96%.

So the stall is real, it is worth ~9% at depth 1, and it is *not* the number
of round trips.

**The gap profiler now says where it is**, if not yet what it is. `ORNITH_PROF`
measures `GPUStartTime` of each buffer against `GPUEndTime` of the one before
-- the queue is serial, so that stretch is idle -- and charges it to the first
kernel of the buffer that ended it, which is what the GPU was waiting to be
handed. Autoregression reads cleanly: 0.46 ms a token, 4% of wall, and
essentially all of it is one gap of 0.43 ms once a token, waiting for
`gather_rows_by_id` -- the embedding that starts the next forward, after the
sampler's round trip. That is the price of the one unavoidable round trip and
nothing else is wrong.

MTP at depth 1 idles 2.0-2.5 ms a token, ~20% of wall, and it is the *same*
gap: waiting for `gather_rows_by_id`, ~0.7 times a token, at **~2.0 ms each
against autoregression's 0.43**. One event, 4.6x the price. Four explanations
have been tried and measured away:

- **The number of round trips.** Halved (above); the idle did not move.
- **The `Committed` history policy**, which encodes a `head.hidden` between
  the round trip and the next draft. Real but small: `--mtp-cycle` is ~7% of
  wall faster at identical GPU busy, so the policy costs ~0.4 ms a token. It
  buys the head an authoritative history, and that is the trade.
- **The commit threshold**, on the theory that a short draft phase never
  reaches 128 dispatches so the GPU waits for the verify's encoding to push it
  over. `ORNITH_COMMIT_EVERY` of 64 and 32 both made the idle slightly
  *worse*, and the draft is ~250-500 dispatches anyway.
- **Host work in the loop.** Timed per phase: the CPU spends ~108 us a cycle
  encoding the draft, ~3.6 ms encoding the verify, 4 us rewinding, 62 us on
  `head.hidden` and 15 us emitting. Everything else is blocking on the GPU.
  There is no 2 ms of host work to find.

- **Waiting for a core.** `ORNITH_PROF` reports the time between the GPU
  finishing the buffer that was waited on and `waitUntilCompleted` returning,
  by comparing `GPUEndTime` with `mach_absolute_time` on the same clock --
  neither side's own timer can see that stretch, since the wait timer starts
  before the GPU finishes and the GPU timer stops when it does. It is 88 us a
  round trip on the autoregressive path and 74 on the MTP one: a tenth of the
  gap, and the smaller of the two belongs to the slower path.
- **Memory pressure.** This one was real, and it explained the *variance*
  rather than the gap. See below.

**The four-fold swing was memory pressure, and intervention settled it.** The
machine was 21 GB into a 21.5 GB swap file while MTP's working set is 27.2 GB
against the trunk's 25.5 -- the extra being the bf16 head, touched once a
cycle while the trunk sweeps 2.83 GB past it every token, which is exactly the
profile of a page that gets evicted and faulted back. Memory was handed back
to the host, swap fell to 10.7 GB of 12.3, and the same interleaved series was
run again:

| | wall tok/s | GPU busy | idle |
|---|---|---|---|
| autoregressive, before | 90.4, 92.4, 89.4, 89.1 | 10.4-10.7 | 0.45-0.50 ms |
| `--mtp-depth 1`, before | 78.8, **20.9**, **44.3**, 74.6 | 9.75-9.98 | 2.68-35.54 ms |
| autoregressive, after | 90.4, 90.6, 90.8, 90.1 | 10.50-10.52 | 0.51-0.58 ms |
| `--mtp-depth 1`, after | 75.0, 79.5, 77.6, 87.0 | 9.59-9.76 | 1.78-3.03 ms |

The swing is gone and the 20 tok/s runs do not come back. Autoregression never
showed it at all, because 25.5 GB fits where 27.2 does not. **So no MTP
wall-clock number taken while this machine is swapping means anything** -- and
that, not the load average, is what invalidated the original "slower at every
depth" table.

**What it did not explain is the gap**: still ~2.5 ms a token against
autoregression's 0.5, still on the same `gather_rows_by_id`. Which also argues
against mlx's residency sets, the lead this looked like -- the note at the top
of `src/metal/mod.rs` calls them something a single-stream loop does not need,
and they would have helped most exactly where the swap was worst, which is the
part already fixed.

So the stall is a wall-clock problem on the one machine whose wall clock
cannot be trusted, and seven explanations have now been measured away. The way
forward on MTP was what GPU busy *can* see, and that is what FR-Spec did.

### FR-Spec, and speculative decoding finally paying

`--frspec` drafts through a pruned `lm_head` -- 65536 of the 248320 rows,
26.4% of its bytes -- while the verify pass keeps reading all of them. Port of
MTPLX's `frspec_draft.py`, and `src/frspec.rs` says why the two agree.

**It cannot change what the model emits**, and this is the property that makes
it worth having. Every committed token is the verify forward's own argmax: the
loop accepts draft `i` only where it already equals target `i` and takes
target `k` itself otherwise, so the draft decides how many tokens a cycle
commits and never which. Measured rather than argued -- the same prompt at
depth 3, with and without, gives byte-identical output *and* identical
acceptance, 53 of 199 drafts at 0.57 / 0.27 / 0.50. Over those 199 drafts the
64K subset never once failed to hold the token the full head would have
proposed, which is what a 0.997 coverage figure looks like from close up.
`tests/mtp` is unchanged by it.

GPU busy a token, two runs each, against the same binary with the lane off:

| | off | on | |
|---|---|---|---|
| `--mtp-depth 1` | 9.63, 9.72 | 9.03, 9.07 | -6.5% |
| `--mtp-depth 2` | 11.17, 11.20 | 10.36, 10.04 | -8.7% |
| `--mtp-depth 3` | 12.54, 12.64 | 11.07, 11.35 | -11% |

And against autoregression, interleaved, four runs each:

| | wall tok/s | GPU busy |
|---|---|---|
| autoregressive | 89.5, 90.1, 91.2, 90.8 | 10.50-10.67 |
| `--mtp-depth 1 --frspec` | 85.1, **32.0**, **38.5**, 91.1 | 9.10-9.24 |

**Depth 1 with FR-Spec does 13.5% less GPU work a token than autoregression**,
with no overlap between the two sets, and its best wall run ties. Depth 1 is
also the best depth now: 9.0 against 10.0 and 11.1.

The two collapsed runs are the swap signature from the section above -- FR-Spec
adds 142 MB of gathered rows to a 27.2 GB working set, and the machine was
back to 16.4 GB of a 17.4 GB swap file by then. Which is the whole state of
this path: **the speculative win is real and measured on the metric that
survives this machine, and the stall is what stands between it and the wall
clock.**

Only after that is it worth making the draft cheaper, and there the number is:
a draft step is ~2.4 ms of GPU, about half of which is the `lm_head` below.

The two structural costs, which are why deeper is worse:

1. **Every draft step runs the full `lm_head`.** It is 8-bit over a 248320
   vocabulary, so ~540 MB read per draft -- ~1.6 GB for a depth-3 chain,
   comparable to a whole trunk forward, for three tokens that are then mostly
   rejected. This is exactly what MTPLX's FR-Spec (`frspec_ids`, a pruned
   draft head over a ranked sub-vocabulary) exists to avoid, and it is not
   ported.
2. **The verify forward multiplies MoE traffic**, and nothing can amortise it.
   Its `depth + 1` tokens each pick their own top-8 of 256 experts, so a
   4-token window reads ~30.5 distinct experts for its 32 (token, expert)
   pairs -- 1.05 rows per expert, which is nothing to group. Sorting is the
   mechanism that would group them and it does not apply: `GatherQMM` takes
   the batched rhs-sorted path only at `rows / experts >= 4`, which at 256
   experts means 1024 rows, i.e. 128 tokens. `switch_glu_sorted` has a
   `force_sort` hook and using it here would buy a counting sort and two
   gathers and still land in `gather_qmv`. This was once written down as
   "forbidden because it would stop matching mlx-lm"; it is simply not
   available.

A dense model would win here; a 3B-active MoE with an unpruned draft head does
not. FR-Spec is the fix for the first of those, and it is in -- see below.

### How exact it is

The acceptance rule and the rewind are exact. The output is not quite: the
verify forward runs `depth + 1` tokens at once, and `quantized_matmul` picks a
different kernel for M = 4 than for M = 1, with a different reduction order and
therefore a different last bit. Two logits an ulp apart can swap and the
speculative and autoregressive streams part company.
`window_and_steps_both_match_mlx` pins both halves: this port reproduces mlx's
window forward *and* its single-step forward bit for bit, and the two disagree
with each other by up to 1.44 on a hidden whose scale is 35 -- a few bfloat16
ulps compounded through 40 residual layers, against top-2 logit gaps that are
sometimes 0.31. So the streams part company after 8 - 18 tokens, and no
speculative decoder on this stack can do better, MTPLX's included.
`speculative_tracks_autoregressive` measures how far they stay together.

---

## The vision tower

333 tensors, all bfloat16, under `vision_tower.` in
`model-vision-00001-of-00001.safetensors`. **mlx-lm is not the reference
here**: `qwen3_5_moe.Model.sanitize` drops every `vision_tower.*` key and
`qwen3_vl.Model.sanitize` pops the whole submodule, so mlx-lm runs this
artifact text-only. MTPLX's `mtplx/vision/` is the only source, and
`tests/vision*.rs` check against it rather than against `mlx_lm.generate`.

```
  pixels  -> Qwen2VLImageProcessorFast  -> [patches, 3*2*16*16] float32
          -> conv3d 2x16x16x3 -> 1152   + bias
          +  fast_pos_embed_interpolate  bilinear over a learned 48x48 grid
          -> 27 x (LayerNorm, 16-head attention with 2D RoPE, LayerNorm, MLP)
          -> PatchMerger                 2x2 patches -> one 2048-wide token
  rows    -> spliced over the <|image_pad|> run of the embedded prompt
```

A 768x448 photo is 48x28 patches, 336 language tokens, and the tower takes
~0.25 s. Nothing about it is quantized.

Six dispatch decisions are worth writing down, because none of them is what
the Python looks like:

- **The patch embedding is an implicit-GEMM convolution over *padded*
  channels.** `dispatch_conv_3D_gpu` tests `C_per_group % 16 == 0`, which at 3
  input channels is false, so it takes `pad_and_slice_conv_3D_gpu`: both the
  input and the weight are zero-padded to 16 channels and the implicit gemm
  runs on the padded problem, M = patches, N = 1152, K = 2*16*16*16 = 8192.
  Convolving the unpadded thing instead would be a different kernel with a
  different reduction order.
- **Every linear is `mx.addmm`, and the two gemms disagree about what
  "axpby" means.** `nn.Linear` with a bias is `mx.addmm(bias, x, w.T)`, alpha
  = beta = 1. In `steel_matmul_regular_axpby` that sets `use_out_source` but
  leaves `do_axpby` *off* (`alpha != 1 || beta != 1`); in
  `steel_gemm_splitk_axpby` the accumulate kernel's test is `alpha != 1 ||
  beta != 0`, so it *does* take the `_axbpy` variant. Both paths are live:
  `proj`, the MLP's second layer and both merger layers fall into split-K for
  small images and into the regular gemm for large ones, and
  `tests/vision_ops.rs` pins ten shapes across the boundary.
- **The attention runs mlx's *full* fused kernel, which the trunk never
  reaches.** MTPLX's `_fused_sdpa` pads head dim 72 up to 80 (`_FUSED_SDPA_DIMS
  = (64, 80, 128)`) and slices back; an image is always more than 8 patches, so
  `has_fused_kernel` takes its `query_sequence_length > 8` branch, where 80 is
  supported, and `use_fallback`'s tail only rejects 192 and 256. So
  `steel_attention_bfloat16_bq32_bk32_bd80_wm4_wn1` runs -- a kernel the text
  path has no use for, since at head_dim 256 prefill goes unfused.
- **Two different GELUs.** The block MLP is `nn.GELU(approx="tanh")`, the
  `gelu_pytorch_tanh` the config names; the merger is a bare `nn.GELU()`, which
  is the exact erf form. The reference really does use both, and they differ by
  more than an ULP. `tests/vision_ops.rs` compares each over all 65536
  bfloat16 inputs.
- **LayerNorm, not RMSNorm**, at eps 1e-6, with both a weight and a bias --
  mlx's `layer_norm.metal`, whose mean, variance and normalizer are float32
  whatever the array dtype is.
- **The position embedding is interpolated on the host.**
  `fast_pos_embed_interpolate` builds the four bilinear corner indices and
  weights in Python with `mx.linspace`, a truncating cast and four products,
  then `.tolist()`s them into `mx.array(..., dtype=bfloat16)`. So the weights
  round to bfloat16 *before* the gather, and the port computes the same table
  in float32 on the CPU and rounds it the same way
  (`tests/vision.rs::interpolation_table_matches_mlx`).

Three things are hoisted rather than recomputed. Each moves only exact values,
and each is called out where it happens: the convolution weight's channel
padding (mlx redoes it per call), the `moveaxis(1, 4)` fused into that padding
(mlx does a strided view plus a copy), and the `cos`/`sin` tables, which mlx
rebuilds inside `_apply_rotary_pos_emb_vision` twice per block and which are a
pure function of the patch grid.

### Where the line is: the resampler is in, the decoder is out

`preprocess_images` calls `smart_resize` and, when it asks for different
dimensions, `Image.resize(..., Image.BICUBIC)`. Both are ported -- the second
one in `vision/resample.rs`, from Pillow's `src/libImaging/Resample.c`.

That is a different call from the image *decoder*, which stays outside.
Decoding is open-ended (formats, ICC profiles, EXIF orientation, chroma
upsampling) and its output is exactly what this port asks to be handed. The
resampler is closed: one filter, ~150 lines of deterministic integer
arithmetic, sitting *inside* a vendored file, and it decides which pixels the
model sees. Leaving it out meant the port could only reproduce MTPLX on
images pre-sized by something else, which is a hole in the one property
everything here is checked for.

Three details of it are the whole game, and `tests/resample.rs` pins them byte
for byte:

- the `double` filter coefficients are **quantized to 22-bit fixed point**
  (`normalize_coeffs_8bpc`), so the convolution itself is integer;
- it is **two passes**, horizontal then vertical, with the intermediate
  rounded back to `uint8` in between -- one fused pass does not agree;
- a pass is **skipped** when that axis keeps its size, which is not the same
  as running an identity pass through the quantizer.

`smart_resize` itself has one trap: it rounds with Python's `round`, which is
round-half-to-even, and `f64::round` is not. A 48-pixel side at factor 32 has
to round down.

The one thing to keep an eye on: this pins the internals of a Pillow *version*
(12.3.0 here), not a model contract. `dump_resample.py` records the version and
`tests/resample.rs` prints it.

### The splice

The chat template renders one `<|image_pad|>` (248056) per image, between
`<|vision_start|>` (248053) and `<|vision_end|>` (248054). The request then
expands it to one pad per tower row (`_expand_image_pads`) and prefill replaces
those rows' *embeddings* with the tower's output, so an image never goes
through `embed_tokens` at all -- the pad id only marks positions. Prefill
consumes chunks strictly left to right, so the splice is a cursor:
`generate.rs`'s `model_call` embeds a chunk, scatters the next rows into it and
hands the result to `forward_from_embeddings`. A chunk with no pads returns
`None` and keeps the plain token-id path, which is every chunk of a text
prompt.

---

## Where a decode step goes

`ORNITH_PROF=1` turns on counters in the Metal layer and a summary at the end
of a run: wall, GPU busy (summed `GPUEndTime - GPUStartTime` over the step's
command buffers), time blocked in `waitUntilCompleted`, dispatches and commits
per token, and a dispatch count per kernel. `ORNITH_COMMIT_EVERY=N` overrides
the commit size; at `N=1` every command buffer holds one dispatch, so its
timestamps attribute to a single kernel and the summary breaks GPU time down
per kernel. `cargo run --release --bin dispatch_cost` measures what a dispatch
costs with no work in it.

The numbers below are greedy, 256 tokens, a 25-token prompt, on a quiet
machine. Quiet matters for the wall clock and only for the wall clock, and it
matters more than first recorded. Five consecutive runs of one binary at load
average 12:

| wall | 58.8, 85.8, 47.0, 79.6, 82.5 tok/s | 82% apart |
|---|---|---|
| GPU busy | 11.01, 11.03, 11.22, 11.13, 11.10 ms/token | 1.9% apart |
| GPU busy share | 65%, 95%, 53%, 89%, 92% | |

**So measure GPU busy, not tok/s.** The mechanism is in the third row: the
CPU-bound jobs do not compete for the GPU, they compete for the thread that
encodes, `blocked in wait` goes from 6.3 ms to 18.0, and the GPU idles. What
it executes takes the same time either way -- 11.0 to 11.2 ms whatever else
the machine is doing, which is also what it measured when the machine was
quiet. Waiting for a quiet machine buys nothing that reading the right column
does not.

The caveat that comes with it: GPU busy is summed `GPUEndTime - GPUStartTime`
over command buffers, so it cannot see the gaps *between* them. It is the
right metre for a change on the GPU's side -- kernels, barriers, epochs -- and
the wrong one for a change in what the encoding costs the CPU. For those, and
for the wall clock, a quiet machine is still the only way.

(`mlx_lm`'s CLI renders the chat template its own way and gets 23 tokens from
the same text, so the two generate different essays. `tests/generate.rs` is
what pins them token for token, on identical ids.)

| | |
|---|---|
**Read the GPU column.** Wall figures below are the best of five interleaved
runs and still move with what else the machine is doing; GPU busy holds to 2%
under any load and is the number to compare against.

| | wall tok/s | GPU busy ms/token |
|---|---|---|
| ornith, before the barrier work below | 82 | -- |
| ornith at the 1:1 commit, `261fe0c` | 88 | 11.3 |
| **ornith now** | **92** | **10.3-10.5** |
| ornith `--mtp-depth 1 --frspec` | 94 at best, and it swings | **9.0** |
| `mlx_lm.generate`, same prompt text | 86 | -- |

So the port is now ahead of the reference it was written against rather than
level with it, by the two changes mlx does not have: the MoE block issued
interleaved and the buffer pool declining to recycle inside a barrier epoch.
And the speculative pairing does 14% less GPU work a token than plain decoding
while only tying on the wall, for the reason in "It does not pay off here".

| a decode step, now | |
|---|---|
| GPU busy | 10.3-10.5 ms of a 10.85-11.34 ms wall -- 95-96% |
| idle between command buffers | 0.47-0.56 ms a token, essentially one round trip |
| dispatches | 1669 per token, in 14 command buffers (1759 before single-row slices became views) |
| barriers | 1064 per token (1216 at the 1:1 commit, 1669 before the barrier work) |
| epochs | 1078 per token, 1.55 dispatches each |
| host round trips | 1.00 per token |
| weights read | ~2.83 GB per token, i.e. **272 GB/s** against a measured ceiling of 465 and a sticker of 546 |

| the prompt, at chunk 2048 | |
|---|---|
| GPU busy | 2.05-2.34 s for 2600 tokens -- ~1230 tokens a second of GPU work |
| wall | 380-614 tok/s, i.e. 33-49% busy, the rest lost to memory pressure |
| `mlx_lm.generate`, same prompt | 809 tok/s |

So the step is GPU-bound, and the earlier guess -- that the per-step `sync` and
the ~800 dispatches were the problem -- was wrong on both counts: the sync
costs nothing measurable (the GPU is busy 95-96% of the wall) and there are twice
as many dispatches as that.

What the dispatch count costs is the barrier between each pair.
`dispatch_cost` puts an empty dispatch at **1.0-1.25 us** in a serial encoder
and at **0.15-0.25 us** in a concurrent one with no barriers -- so ~1 us of
every dispatch is the barrier, and 1669 of them would be **~1.6 ms of the
12.2 ms step**. This port used to pay all of it, where mlx pays only for the
dependencies it actually has; that was the one real divergence in the dispatch
path, and it was about the size of the gap to `mlx_lm`. It is ported now, and
the rest of this section is what that took.

### The barrier between dispatches

The encoder is `MTLDispatchTypeConcurrent`, and a barrier goes in only where
the data says it must -- mlx's rule, ported from `CommandEncoder`'s
`set_input_array` / `register_output_array` / `maybeInsertBarrier`. A barrier
splits the encoder into epochs; within one, dispatches may overlap, so the one
being encoded needs a barrier before it if it reads a buffer this epoch wrote,
writes one this epoch read, or writes one this epoch wrote.

The tracking keys on the Metal buffer, not the array, so two views of one
allocation are a hazard and so is a buffer the pool has handed out again --
both want the barrier, and both get it.

What this needs from an op is to say which of its bindings it writes:
`.out(a)` instead of `.arr(a)`. An op that says nothing is treated as writing
everything it binds, which forces the barrier it used to get from the serial
encoder, so an unmigrated op is slow rather than wrong.

Getting that marking wrong in the other direction -- a write bound as `arr` --
is a race, and a race is not something a bit-exact test can be trusted to
catch, since it may simply win. So the Metal layer tracks which buffers have
been written and asserts that nothing reads a pooled buffer no dispatch has
written yet. That is exactly the shape of a missed `out`, and it found three
of them on the first run (the multi-block sort's `dev_vals_out`/`dev_idxs_out`,
which reach the binding under an alias the migration script could not see).

It cannot see the other shape: an in-place write into a buffer something has
already written. Those were found by hand and checked against the kernels'
own signatures, where a written buffer is `device T*` and a read one is
`const device T*` -- `causal_mask`, `zero`, `qmm_t`, the `sum_*` reductions,
`cache_copy`'s `dst` and `splice_rows`' `flat`.

The mirror-image mistake -- an input bound as `out` -- is safe, since it only
asks for a barrier that was not needed. It is also invisible, and it eats
exactly what the change was for. They hide where one array is bound by two
dispatches in the same function, written by the first and read by the second:
`take_along_last`'s indices, `div_by_row`'s denominators, the split-K
accumulate's partials, the second SDPA pass's `partials`/`sums`/`maxs`, the
patch-conv gemm's padded input and the vision rotary table's `inv_freq` all
started out marked as writes. Unmarking them did not move the barrier count on
a text decode -- those dispatches had a real dependency behind them anyway --
but they are the shape to look for if the count is ever worth attacking again.

What it bought: **1216 barriers a token instead of 1669**, a quarter of them
gone, and a step of **11.3 ms against the serial encoder's 12.7** run back to
back. That is the gap to `mlx_lm` closed, which is what the ceiling said to
expect.

`ORNITH_SERIAL=1` puts the serial encoder back, as the control to measure
against and the fallback if the tracking is ever in doubt. `ORNITH_PROF=1`
counts the barriers actually inserted, which is the one number here that does
not care what else the machine is doing.

Commit size matters too, and not in the direction the comment on
`max_dispatches_per_commit` assumed: each command buffer costs ~60 us of GPU
time, so committing more often makes it slower, not smoother.

| commits/token | tok/s |
|---|---|
| 55 (`ORNITH_COMMIT_EVERY=32`) | 68 |
| 14 (128, the default) | 82 |

| 4 (512) | 79 -- now the GPU waits for the CPU to fill a buffer |

And the barrier's real price is not its fixed cost but the overlap it forbids.
`ORNITH_NO_BARRIER=1` runs the same dispatches in a concurrent encoder
with no barriers at all: the output is wrong, but the step drops from 12.2 ms
to **6.9 ms**. That is an unreachable bound -- the dependency chain in a decode
step is mostly real, and mlx, which does insert the necessary barriers, gets
88 tok/s rather than 146. So porting mlx's rule is worth closing the gap to
mlx, ~7%, not the 77% the bound suggests.

What the bound does say is that the kernels themselves are fine. 2.83 GB at
this machine's 546 GB/s is 5.2 ms, and unserialised they take 6.9 -- about 75%
of peak bandwidth. The 12.2 ms step is 5.2 ms of unavoidable weight reading
plus ~5.3 ms of serialisation plus ~1.7 ms of dispatch and commit overhead.

---

### The prompt, which nothing here had ever measured

`ORNITH_PROF=1` now reports the prefill phase before it resets the counters
for decode, and `ORNITH_PREFILL=<n>` overrides the chunk width. The first look
at it says the same thing the rest of this document has been converging on.

2600 raw tokens, chunk 2048, three runs: **2.05-2.34 s of GPU busy against a
wall of 4.19-7.38 s** -- 33% to 49% busy, where a decode step holds 95%. The
GPU work alone corresponds to ~1230 tokens a second; the wall gives 380-614.

`mlx_lm.generate` on the same prompt and the same machine gets **809 tok/s**.
So the port is not doing more work at prefill than mlx-lm -- by GPU busy it is
doing it faster -- it is waiting, and the waiting is what the gap is made of.

What the chunk does, paired runs:

| chunk | GPU busy | pooled peak | note |
|---|---|---|---|
| 256 | 3.17 s | 631 MB | too narrow to amortize the weight reads |
| 512 | 2.48 s | 658 MB | still narrow |
| 1024 | 2.11, 2.11, 2.11 s | 881 MB | level, and the steadiest thing measured |
| 2048 | 2.05, 2.09, 2.34 s | 1307 MB | least work when undisturbed, and the one that swings |

The default stays at 2048, which is mlx-lm's `prefill_step_size`, because
2048 does marginally less GPU work and the instability is the machine's rather
than the code's -- 1307 MB of transients on a machine sitting 16.35 GB into a
17.4 GB swap file. 1024 is the setting to reach for when memory is tight: a
third less peak for 3% more GPU work, and it did not swing once.

**Which makes this the third face of one problem.** MTP's four-fold wall swing,
FR-Spec's collapsed runs and prefill's half-idle GPU are all the same thing,
and in all three the GPU busy figure stayed flat while the wall moved. Memory
pressure is the dominant error term in every wall-clock number this port
produces on this machine, and the swap file is the first thing to read before
believing one.

### Which kernels the time is in

`ORNITH_PROF=1 ORNITH_COMMIT_EVERY=1` gives one dispatch per command buffer,
so each buffer's timestamps belong to a single kernel. That inflates the total
(14.0 ms against 10.9) and inflates small kernels most, since each pays for its
own command buffer -- roughly 1.5 us of the cost of every line below. The
proportions are what to read.

| ms/token | share | count | kernel |
|---|---|---|---|
| 5.361 | 38.2% | 195 | `affine_qmv_fast` 8-bit g64 |
| 1.768 | 12.6% | 96 | `affine_gather_qmv_fast` 4-bit g32, the routed experts |
| 0.770 | 5.5% | 191 | `rms_norm` |
| 0.693 | 4.9% | 96 | `affine_qmv_fast` 4-bit g32 |
| 0.596 | 4.2% | 24 | `affine_gather_qmv_fast` 8-bit g64 |
| 0.587 | 4.2% | 40 | `affine_qmv` 8-bit g64 |
| 0.481 | 3.4% | 40 | `carg_block_sort`, the router's argsort |
| 0.318 | 2.3% | 120 | `add_elem` |
| 0.215 | 1.5% | 50 | `slice_inner` |
| 0.189 | 1.4% | 80 | `swiglu` |
| 0.186 | 1.3% | 40 | `col_reduce_small` |
| 0.182 | 1.3% | 50 | `sigmoid_gate` |
| 0.173 | 1.2% | 60 | `scalar_mul` |
| 0.173 | 1.2% | 40 | `row_reduce_small` |

**The big kernels take time in proportion to the bytes they read**, so there is
no slow one to find. The 8-bit `qmv_fast` is 38% of the time and reads about
38% of the traffic -- the gated-delta `in_proj` alone is 1.07 GB of the 2.83.
A guess that the expert gather was the anomaly did not survive the measurement.

**The small kernels look like the anomaly from the other side**: `rms_norm`,
`add`, `slice`, `swiglu`, `sigmoid_gate`, `scalar_mul` and the two reductions
come to ~2.2 ms, 16% of the time here, while moving ~0.2% of the bytes -- the
hidden state is 2048 values, 4 KB, and all of them together push ~4 MB a token
against 2.83 GB of weights.

**That 16% is not 16% of a real step, and this table cannot tell you what is.**
One dispatch per command buffer is one dispatch at a time; a concurrent encoder
runs the small kernels *underneath* the big ones, which are waiting on memory
and leave the arithmetic idle. Fusing five of the router's dispatches into one
removed 160 dispatches a token and moved the wall clock by about half a
percent, against the 5-6% this table was read to promise. Read the table for
which kernels exist and what they cost in isolation; for what removing one
*buys*, there is no substitute for removing it and measuring.

### What the machine actually sustains

546 GB/s is the number on the box. `cargo run --release --bin bandwidth` asks
what a kernel can have: one that only reads, at several sizes, and then the
real `quantized_matmul` at M = 1 on real weights.

| a kernel that only reads | GB/s | of peak |
|---|---|---|
| 64 MB | 202 | 37% |
| 256 MB | 401 | 73% |
| 1024 MB | 465 | 85% |
| 2048 MB | 452 | 83% |

| `quantized_matmul`, M = 1 | size | GB/s | of peak |
|---|---|---|---|
| `lm_head`, 8-bit g64 | 540 MB | 455 | 83% |
| `in_proj_qkv`, 8-bit g64 | 17.8 MB | 116 | 21% |
| `o_proj`, 8-bit g64 | 8.9 MB | 61 | 11% |

So the practical ceiling is **~465 GB/s**, 85% of the sticker, and it is only
reached by a long read. The remaining 15% is the machine's, not ours.

**The kernels are already at that ceiling.** `qmv` over `lm_head` sustains 455
against the 465 a bare read gets. Which kills a plausible-sounding idea before
it cost anything: that the affine layout hurts because `weight`, `scales` and
`biases` are three separate arrays, so a group of 32 weights is 16 bytes from
one region plus 2 and 2 from two others. Repacking them interleaved, the way
llama.cpp's block formats do, would have been a week of kernels and a new
artifact -- and it would have fixed something that is not broken.

**What is broken is the size of the reads.** The same kernel that holds 83% on
540 MB gets 21% on 17.8 MB and 11% on 8.9 MB: a short kernel never gets enough
requests in flight to saturate the memory system, pays its ramp, and stops. A
decode step is a few hundred small reads and one big one, which is exactly why
it averages ~272 GB/s.

**And the same fact says a verify window is nearly free.** The matmuls again,
across the M a speculative window of `depth + 1` tokens produces, as a multiple
of what one row costs:

| | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 | M=7 | M=8 |
|---|---|---|---|---|---|---|---|---|
| `lm_head` 8-bit g64, 540 MB | 1.00 | 1.02 | 1.09 | 1.34 | 1.43 | 1.87 | 2.04 | 1.84 |
| `in_proj_qkv` 8-bit g64, 17.8 MB | 1.00 | 1.05 | 0.80 | 1.10 | 1.12 | 1.15 | 1.18 | 1.21 |
| `gate_up` 4-bit g32 | 1.00 | 0.95 | 1.18 | 0.97 | 0.94 | 1.22 | 1.31 | 1.26 |

The small matrices are flat: eight rows for 21-26% more than one. That is the
short-read story from the other side -- starved at M = 1, they have bus left
over, and the extra rows ride on it. Only `lm_head` climbs, because it is the
one already at the roof at M = 1 with nothing spare to ride on, and even it is
2.0x at M = 7 rather than 7x.

**Which rules out a kernel before porting it.** mlx-dspark's `small_m_qmm.py`
carries `qmm_mma4`, an 8x8 `simdgroup_matrix` tile that reads each weight group
once and reuses it across rows, against the claim that `quantized_matmul`
"grows ~linearly in M for M in 2..8". That is not what this port does on these
shapes -- the amortization the kernel exists to provide is already happening
wherever it would pay. Its own table agrees from the other end: 0.85x at M = 4,
1.05x at M = 5, so it only wins from M >= 6, and a window of `depth + 1` at
the depths that are worth running never gets there.

**It also confirms where a window's cost really is.** If the dense projections
are flat in M, the growth in MTP's GPU busy with depth -- 9.6, 10.9, 12.1 ms a
token at depths 1, 2, 3 -- is not the matmul shape. It is the draft steps
themselves, and the expert gather, where four tokens pick ~30.5 distinct
experts instead of one token's 8. That is the one part of a window that really
does scale with its width, and it is already written down under "It does not
pay off here".

**That resets where the prize is.** If every read behaved like `lm_head`'s, the
step would be 2.74 GB / 465 GB/s = **5.9 ms, about 170 tok/s** against today's
88. The barrier-free bound measured earlier says the same thing from the other
side: 6.87 ms is 399 GB/s. The lever is **overlap**, not layout and not, in the
first instance, bytes.

It also corrects the arithmetic for a smaller artifact. "35% fewer bytes gives
35% more speed" only holds for reads already limited by bandwidth -- `lm_head`.
A read sitting at 21% of peak is limited by its own ramp, so halving its bytes
buys less than half its time, and a smaller read ramps worse still.

**The open question was how many of the 1216 barriers a token are real.** It
is measured now, and the section below is the answer.

### Where the 1216 barriers come from, and what overlap is left

`ORNITH_PROF=1` attributes every barrier to the pair of dispatches that forced
it, the hazard it was (RAW / WAR / WAW), and what the two shared:

- **real** -- overlapping bytes of one allocation. A dependency the data has.
- **disjoint** -- one allocation bound at two ranges that do not overlap: two
  halves of a fused projection, two experts of a stacked weight.
- **recycled** -- two different allocations that share an address, because the
  pool handed the freed one straight back out. Nothing flows between them.

The last two cost a barrier all the same, because the tracking keys on the
buffer -- which is what mlx does. A barrier is counted spurious only if *no*
cause behind it was real.

| per token | |
|---|---|
| barriers | 1216 |
| with a real dependency behind them | 1186, 97.5% |
| invented by the buffer pool | 30, 2.5% -- all one pair, `rms_norm` writing where `scalar_mul` had read |
| invented by two views of one allocation | 0 |

So the suspicion was right in kind and small in size. These counts do not
depend on what else the machine is doing, and they reproduced exactly across
runs.

**The pool now declines to hand back a buffer the current epoch has touched**,
which removes exactly those 30 and nothing else. Measured by GPU busy against
its own control, four runs each: 10.43 -> 10.35 ms a token, **-0.8%**, which
is what the calibrated model predicted to the second decimal (-1.9% priced,
divided by 2.4). It costs no memory at all -- the buffer is reused in the next
epoch instead, and the pool's peak is 76.8 MB either way -- and `tests/model`
is bit-exact under it. `ORNITH_RECYCLE=1` puts mlx's behaviour back as the
control.

**The number that matters is not the barrier but the epoch.** A Metal barrier
is encoder-wide -- it orders everything before it against everything after --
so what can overlap is the run of dispatches between two barriers. 1669
dispatches in 1230 epochs is **1.36 dispatches per epoch, and 80% of epochs
hold exactly one**. The concurrent encoder is concurrent in name.

`ORNITH_TRACE=<path>` writes one step's dispatches with every binding, and
`tools/epoch_schedule.py` levelises them as-soon-as-possible to ask how few
epochs the dependencies actually permit. That floor is reachable: no two
dispatches on a level collide, and every collision is on an earlier level, so
encoding the levels in order is a legal encode order.

| | epochs/token | dispatches each |
|---|---|---|
| as encoded today | 1230 | 1.36 |
| best order, tracking whole buffers as now | 1007 | 1.66 |
| best order, byte ranges as well | 1007 | 1.66 |
| best order, and no recycling inside an epoch | 947 | 1.76 |

Three things fall out of it.

- **223 epochs a token are encode order and nothing else** -- same tracking,
  same semantics, just issuing independent dispatches next to each other. The
  MoE block is where they are: the router's tail, the routed experts and the
  shared expert are mutually independent and today run one after another. The
  best order puts `affine_qmv` 8-bit with both shared-expert `qmv`s and an
  8-bit `qmv_fast` in one epoch 32 times a token -- four weight-heavy reads at
  once -- and hides the argsort and `take_along_last` under the expert gathers.
- **Byte ranges buy nothing.** Tracking views precisely moves the floor by
  zero epochs, and the attribution counts zero barriers caused by disjoint
  views. Refining the tracking that way is ruled out before it costs anything,
  which is the same shape of answer as the interleaved-affine-layout idea
  above.
- **And what those 223 epochs are worth is 12%.** Given the per-kernel times
  from `ORNITH_COMMIT_EVERY=1`, charge every epoch its slowest member and let
  the rest run underneath. That model prices the schedule as encoded at 8.88
  ms against a step measured at 10.9-11.8 ms of GPU busy -- optimistic by
  about a quarter, as it must be, since concurrent kernels share the memory
  bus. Read the ratio, not the number:

  | | epochs | priced |
  |---|---|---|
  | as encoded | 1230 | 8.88 ms |
  | best order | 1007 | 7.81 ms, **-12%** |
  | best order, no recycling | 947 | 7.38 ms, **-17%** |

  So a perfect encode order is worth ~12%, ~17% with the pool change, and that
  is the whole of what scheduling has left. Half of it is now banked (the MoE
  block below), so what remains is ~3.4%: 92 tok/s would become about 95.

**The model is calibrated on the one experiment that has an answer.** Feed it
the router-tail fusion from the section below -- merge those five dispatches
into one carrying their summed time -- and it reproduces the build exactly,
1669 dispatches to 1509, and prices it at nothing: -0.7% where the built
version measured +0.5%. It predicted the failure it should have predicted,
which is the reason to believe the 12% it predicts for reordering. Both come
out of the same arithmetic.

**Why it is not more, since two other numbers say 6 ms.** The critical path
costs 5.95 ms of kernel time, and 2.83 GB at the machine's 465 GB/s is 6.09
ms; the barrier-free run measured 6.9. All three describe a step with
everything off the critical path hidden underneath it, and the schedule is
1.76 wide, so there is nowhere to hide it. Half the kernel time is off the
path -- the routed experts' 1.9 ms is almost all of it -- but an epoch that
holds one big read and one small one cannot absorb another big read. The 6 ms
step needs a decomposition with fewer, longer reads, not a better order of
this one. Which puts what is left where "What the machine actually sustains"
put it: bytes, or more than one token in flight.

The floor is a property of *this decomposition into kernels*, not of the
model: fusing kernels shortens the chain. It is also optimistic in two ways,
both deliberate -- it ignores the 14 command-buffer boundaries a step has to
have, and a binding's range is what the binding declares, so a kernel that
reads past its array would add dependencies this does not see. Both push the
same way: the real ceiling on overlap is no higher than 1.76.

### Interleaving the MoE block, and what it bought

The first thing done with the departure allowed above, and the measurement
that calibrates the model in the section before it.

The shared expert runs on every token and depends on `x2` alone -- nothing in
it waits on the router. It was nevertheless issued after the whole router
chain, which is a chain of small dispatches each holding a barrier epoch on
its own. `sparse_moe_block` now issues the two together: the shared expert's
three projections go out with the router's gate, its swiglu with the softmax,
its down projection with the argsort. No kernel changes, no weight changes, no
reduction order changes -- only the order the dispatches are issued in, so the
result is bit-exact by construction, and `tests/moe`, `tests/model` and
`tests/generate` all say so.

| | |
|---|---|
| epochs | 1230 -> 1108 |
| barriers | 1216 -> 1094 |
| dispatches per epoch | 1.36 -> 1.51 |
| GPU busy | 11.07 -> 10.71 ms/token, **-3.3%** |

The GPU busy figures are the mean of four runs each, alternating, at load
average 17: control 10.92 / 11.11 / 11.21 / 11.04, interleaved 10.61 / 10.80 /
10.82 / 10.59. Every run of one is below every run of the other, which is what
makes 3.3% a result at a spread of 2%.

**And it is what says the schedule model is optimistic by 2.4x.** The model
priced this change at -7.9% and it measured -3.3%. That is the first
non-zero prediction it has made -- the router fusion it priced at nothing, and
nothing is what that measured -- so 2.4 is the number to divide by. Its
ranking is right and its magnitude is not, for the reason built into it: an
epoch is charged its slowest member, as though the rest ran underneath for
free, and bandwidth-bound kernels in one epoch share the bus instead.

Applying that to what is left: the best possible order is another -8.1% in the
model, so **~3.4% in reality**, and `ORNITH_NO_RECYCLE=1` on top of it another
~0.8%. The whole of scheduling was ~7%, of which 3.3% is now banked.

**Where the remaining ~100 epochs are, and why they cost more than these did.**
Nowhere in particular: the slack is spread over 191 `qmv_fast`, 96 expert
gathers, 186 `rms_norm` -- one epoch each, everywhere, rather than one more
block issued in the wrong order. Taking it means reordering across layer and
block boundaries, which the eager op layer cannot express; it needs something
to defer with, which is mlx's lazy graph. That is a different size of change
from the twelve lines this section is about.

### Pipelining the decode loop against the host read, and why it is not here

Tried and reverted, like the fusion below, and for a better reason than it
looked like it would have.

mlx-lm's `generate_step` encodes step `n + 1` *before* the host reads step
`n`: `_step(y)` is called with `y` still an unevaluated array, `mx.async_eval`
submits it, and only then does `y.item()` go to the host. The round trip is
paid while the GPU already has the next step. This port is eager, so `step`
ended at a host value and the loop was strictly serial -- which looked like a
real 1:1 divergence, and the gap profiler had just priced it exactly: 0.43-0.50
ms a token, one gap, always before the embedding gather that starts the next
forward.

It was built. `Sampler` grew a `sample_device` returning the token as an array,
the loop fed that array straight into the next step, and the Metal layer grew
`Context::mark` / `wait` -- the piece `async_eval` really needs, since `sync`
waits for the whole queue and would have waited for the step being hidden. The
queue is serial, so a command buffer's completion means everything before it
and nothing about what came after, which is exactly the primitive.

It measured **worse**. Best of five runs each, idle a token: 0.47 ms serial
against 0.82 pipelined, with 90.6 tok/s against 85.8, and two of the five
pipelined runs collapsed to 41 and 55 tok/s where no serial run did.

The reason is in the number it was aiming at. `max_dispatches_per_commit` is
128, so the GPU starts as soon as the first 128 dispatches of the next step
are encoded -- about 0.3 ms in -- and that already hides almost all of the
round trip. What is left is 0.47 ms, of which 88 us is the wake from the GPU,
and pipelining spends more than that: two steps' intermediates are live at
once instead of one, which is also what the collapsed runs look like.

**So the port already has what `async_eval` buys, by a different route**, and
the eager encoder plus a commit threshold is that route. Worth knowing before
anyone reads mlx-lm's loop and sees a divergence, which is what it looks like.

### Fusing the router's tail, and why it is not here

Tried and reverted, so that it is not tried again for the same reasons.

The router's tail is five dispatches a layer -- a full argsort of all 256
experts, the slice of its last k, a gather, a row sum and a divide -- two
hundred a token, the argsort alone at 12 us to order 256 numbers. Only the
last k of that order is ever read, so one kernel can replace the sort with k
rounds of selection, and it is bit-exact: mlx's block sort is stable ascending
(`LessThan` compares values alone, `ThreadSort` swaps only on a strict `<`,
`merge_step` takes from the right run only when `b < a`), so its last k are
exactly the k largest under "by value, then by index"; the sum matches
`row_reduce_small`, which accumulates in the output dtype and so rounds to
bfloat16 after every addition, in index order; the divide matches
`div_by_row`. It was written, it passed `tests/moe.rs` against MLX's own
indices and weights at every T, and it took a step from 1669 dispatches to
1509 and from 1216 barriers to 1098.

It bought **half a percent** -- 88.9 tok/s against 88.4 -- and cost the one
property this port has, which is that it does what mlx does. The reason is in
the paragraph above: at 96% GPU and bound by weight reads, those five small
dispatches were never on the critical path. They ran underneath the `qmv`
kernels that were waiting on memory.

The lever here is bytes, not launches, and no rearrangement of the small
kernels changes that.


## What is left

In order:

1. **A machine that is not swapping.** This is first because it is what the
   next two are waiting on, and it is not something the code can fix. Every
   GPU-side number in this document is now current and trustworthy -- they
   were all re-measured, and GPU busy holds to 2% under any load. Every
   wall-clock number is not: the same binary has come out 82% apart on the
   wall while its GPU busy did not move, and the swing tracks the swap file,
   not the load average. MTP's four-fold swing, FR-Spec's collapsed runs and
   prefill's half-idle GPU are one problem with three faces. Read
   `sysctl vm.swapusage` before believing a wall figure.
2. **Close the gap from 272 GB/s to the 465 the machine can give.** The
   kernels are already at the ceiling on a long read and nowhere near it on a
   short one. Overlap was the obvious lever and it has now been measured:
   "Where the 1216 barriers come from" above says the dependency chain is 947
   dispatches deep and at best 1.76 wide, so scheduling cannot get there. Two
   things are still worth doing, in this order.
   - **Encode order. Half of it is done and the rest is not worth it yet.**
     Interleaving the shared expert with the router took 122 of the 223
     available epochs and 3.3% of GPU busy -- see "Interleaving the MoE block"
     above. What is left is ~3.4%, spread one epoch at a time over every
     kernel in the step, and taking it means reordering across block and layer
     boundaries, which needs a deferred op layer rather than another dozen
     lines. Worth doing only if something else wants that layer anyway.
   - **Then bytes, which is the only lever left at batch 1.** `REBUILD.md`
     is the plan and `IDEAS.md` is what the freedom to design the artifact
     buys -- measured, not asserted: the grid has about one bit a weight left
     in it and vector quantization does not collect it, the cheap win is in
     the group metadata, and the one structural idea is a two-stage `lm_head`
     worth 14% of a step and exact for greedy decoding.
3. **Rebuild the artifact from its parent**, which is what unblocks the byte
   lever above. `REBUILD.md` is the plan: the artifact records the recipe that
   forged it, the parent is `ornith-ai/Ornith-1.5-35B-A3B@10fbf86f` and its
   tensor inventory has been checked against that recipe remotely, and
   `body_mode` is one word in the recipe -- mlx quantizes in `affine`,
   `mxfp4`, `nvfp4` or `mxfp8`. The prize is the 260 modules held at 8 bits:
   if a better 4-bit format holds quality where affine int4 did not, they drop
   to 4 and take ~35% of a step's traffic with them.

   Read Phase 0 before deleting anything. Every test here compares against MLX
   on the same artifact, which validates the port and not the artifact, so a
   bad rebuild passes the whole suite -- and the material for the only check
   that would catch it exists only while the current artifact does.
4. **The MTP stall**, which is now the only thing between speculative
   decoding and a wall-clock win. FR-Spec is done and `--mtp-depth 1
   --frspec` does 14% less GPU work a token than autoregression; what it
   does not do is convert that into tok/s, because the loop leaves the GPU
   idle ~2.5 ms a token against autoregression's 0.5. Seven explanations are
   measured away in "It does not pay off here"; the eighth needs a machine
   that is not swapping, since this is a wall-clock question and this
   machine's wall clock is the one number that cannot be trusted.

Smaller gaps, each with an assert or a comment where it bites:

- The chat template's `tools` system block and the assistant `tool_calls`
  block are not ported. Both render through jinja's `tojson`, which
  `transformers` replaces with `json.dumps(..., ensure_ascii=False)`, so they
  need a JSON *writer* over an order-preserving value -- nothing else in this
  port wants one yet. Every other branch of the template is there, including
  `tool` messages and the reasoning-content split.
- MTPLX's runtime adds a lot on top of the draft/verify loop that is not here:
  the acceptance EMA with its suspend/backoff and probation, the copy-blocks,
  FR-Spec, the `last_window` history policy, and probability-ratio acceptance
  for sampled drafts. Acceptance here is by token equality, which is exact for
  a greedy sampler and only approximate for a stochastic one -- `--mtp` says so
  when it is paired with sampling.
- The REPL keeps one cache across turns and reuses it only when the new prompt
  extends what the cache already holds; anything else starts over. A
  gated-delta state cannot be trimmed, so that is not an optimization to fix
  cheaply. A prompt carrying an image never reuses it at all: every image
  expands to the same `<|image_pad|>` id, so a token-id prefix match cannot
  tell two pictures apart. MTPLX guards the same hazard from the other side,
  with a content digest (`vision_bank_key_ids`) and `vision_image_spans`;
  neither is ported, because this REPL has no session bank to key.
- Video is not ported. `chat.rs` renders `Part::Video` as `<|video_pad|>`
  because the template does, but nothing expands or splices it, and
  `VisionTower::forward` asserts a single still frame -- which is also all
  MTPLX ever hands it, since it runs the tower once per image.

Not ported, deliberately, with `assert!`s guarding each: the `*_nax` Metal 4
kernels (M5 only, including the 72 -> 96 head-dim pad the vision attention
would otherwise take), `qmv_quad` (no 64/128-wide input in this model), `qvm`
(every linear here is transposed), `gemv_wide` and `gemv_axbpy` (the vision
tower's M is the patch count, never the <= 15 rows those want), the deepstack
taps (this `vision_config` ships none), and mlx's graph-level fusion engine
(the fused forms it produces are written out by hand instead, in
`kernels/ornith_elementwise.metal`, `kernels/sampler_ops.metal` and
`kernels/vision_ops.metal`).

### The Neural Engine, and why it is not a lever here

Looked at and turned down, so it does not have to be looked at twice. The ANE
is a graph-execution engine reached through private frameworks (`_ANEClient`,
`_ANECompiler`, MIL programs, IOSurface for tensors); there is a Rust toolkit
on the same objc2 stack as this port, `djphoenix/ane-rs`, already carrying
Qwen3/Qwen3.5. Three things rule it out, in order of how final they are.

- **It is fp16, and this trunk is quantised.** 260 tensors at 4-bit g32 and
  8-bit g64 with group scales, and no affine quantised matmul is exposed on
  that path. Dequantising is 70 GB of weights instead of 25.5, or 2-4x the
  bytes per token if done on the fly -- the opposite of what a memory-bound
  decode wants.
- **Decode is bound by memory, not arithmetic.** 2.83 GB a token, a 5.2 ms
  floor at this machine's 546 GB/s against a 10.4 ms step. The ANE's case is
  TOPS, and it shares the same unified memory, so there is nothing for it to
  win at batch 1.
- **It would end the verification.** The published figure for that path is
  MAE 1e-5 against fp32; the check here is `max_abs == 0.0`, and it is what
  caught all ten bugs in the list above.

The one piece that could physically run there is the **vision tower** -- its
333 tensors are unquantised bf16, and a ViT is the ANE's home ground. It still
is not worth it: the tower runs once per image rather than once per token, so
it is not on the decode path at all, and moving it would trade the token-for-
token agreement with MTPLX for fp16.

AMX was dismissed for the same reasons and one more -- "it is CPU-side, on the
same memory" -- and that last clause was wrong. See below.

### The CPU on the same memory, which is not the objection it looked like

`--bin bandwidth` measures it now: a 1 GB streaming read from each side,
alone and at once, on this part.

| | GB/s |
|---|---|
| GPU alone | 417.3, 419.2, 425.0, 393.2 |
| CPU alone, 12 threads | 180.9-236.0 |
| GPU while the CPU reads | 248-344 |
| CPU while the GPU reads | 148-236 |
| **together** | **542.9, 516.3, 516.9**, 396.3 |

Three runs of four put the pair at 516-543 GB/s against 417-425 for the GPU
alone -- 23-30% more, with the top of it within a whisker of the 546 on the
box. **The GPU cannot pull this machine's memory on its own**, and the CPU
makes up part of the difference rather than taking it away. "Same memory" is
not a reason; it was assumed here and it does not hold.

That reopens exactly one idea, and it is the largest one left: hand the CPU a
slice of a decode step's *output rows* while the GPU takes the rest. Decode is
the path with somewhere to put it -- it gets ~272 GB/s of a possible 465 --
and prefill is not, since prefill's idle is memory pressure rather than a
bandwidth ceiling.

Two things stand between the measurement and a win, and they are not small.

- **The kernel.** Accelerate has no quantised matmul, so this is a hand-written
  NEON dequantise-and-dot over the affine layout, and the CPU would be limited
  by that arithmetic rather than by the streaming figure above. A back of the
  envelope on 4-bit g32 -- 20 bytes read per 32 FMAs, ~8 fp32 FMAs a cycle a
  core, 12 cores -- lands near 60 GB/s of weight consumption, against the
  GPU's 250. So the honest expectation is something like +25%, not +30% of
  542.
- **The verification**, which was the gate, and which has now been measured
  rather than argued. See below.

#### The verification gate, measured

`--bin cpu_row` transcribes `qmv_fast`'s order for 8-bit / group 64 straight
out of `quantized.h` -- 32 lanes, 8 consecutive values each per 256-wide block,
float accumulation, one `simd_sum`, a round to bfloat16 -- and computes real
rows of `in_proj_qkv` on the CPU against what the GPU produced for the same
activation. Two things in that kernel are written down nowhere: whether the
Metal compiler contracts `accum += x[i] * w[i]` and `scale * accum + sum *
bias` into fused multiply-adds, and what order `simd_sum` folds 32 lanes in.
So the probe tries both contractions against three plausible trees --
butterfly, halving, lane order -- six combinations in all.

The activation and the expected output are MLX's own, out of
`refs/quantized_matmul`, so this is the same `max_abs == 0.0` against MLX that
every other test here makes rather than this port's GPU standing in for it.
Three real matrices, 10496 rows:

| | butterfly | halving | lane order |
|---|---|---|---|
| `in_proj_qkv`, 8192 x 2048 | **8192/8192** | 8191 without fma | 8191 with fma |
| `out_proj`, 2048 x 4096 | 2048/2048 | 2048/2048 | 2048/2048 |
| `gate`, 256 x 2048 | 256/256 | 256/256 | 256/256 |

**The butterfly fold reproduces MLX exactly on every row under either
contraction** -- 10496 of 10496. The other two trees each miss one row, both in
`in_proj_qkv`, by 1.2e-7 and 2.4e-4. So the fold is not free and has to be the
butterfly, which is also what a `shuffle_xor` all-reduce would do; whether the
compiler contracts into fused multiply-adds then does not matter, which leaves
the faster NEON form available.

It is worth noting what the synthetic version of this probe said first: all
six orderings passed, at a worst spread-to-tie ratio of 2.0. On MLX's real
activation the ratio is 14.0 and two orderings break. A made-up activation
would have licensed the wrong conclusion here, which is the same lesson
`dump_sampler.py` already carries about drawing from a peaked distribution.

The probe also validates the layout reading as a side effect: a wrong
transcription of the affine indexing would be wildly off, not exact.

This is a strong measurement and not a proof -- one activation per matrix, and
3 rows of 8192 sit close enough to a tie that another could land the other
way. What would catch a regression is already in place: `tests/quantized.rs`
compares against MLX on real weights at `max_abs == 0.0`, and a CPU-split path
would live or die by it.

#### And the gate that is shut: the CPU cannot get its input in time

The verification gate is open. A different one is not, and it closes the idea
for decode.

**The CPU needs the matmul's input, and every matmul's input in a decode step
was produced by the GPU tens of microseconds earlier.** The activation sits in
shared memory, but a GPU write only becomes visible to the host when the
command buffer carrying it completes -- reading earlier is not something
Metal's memory model promises, whatever the bytes happen to say. So splitting a
matmul costs a command-buffer boundary in front of it, and a command buffer
costs ~60 us of GPU time (the number `max_dispatches_per_commit` is set by).

That prices the split per matmul, at M = 1:

| | time | boundary worth paying? |
|---|---|---|
| `lm_head` | 1257 us | yes -- 60 us to enable saving ~314 |
| `in_proj_qkv` | 151 us | no |
| `o_proj` | 122 us | no |
| a routed expert gather, x96 | ~18 us | no |

A decode step has 195 8-bit `qmv` dispatches and **exactly one of them is
`lm_head`**. So the reachable share is that one matmul: 1257 us of a ~10500 us
step, and a CPU taking a quarter of its rows nets ~254 us after the boundary --
**2.4% of a step**, not the 25% the bandwidth measurement suggested.

The earlier estimate was wrong because it priced the CPU against all 2.83 GB a
step reads. The CPU can only be handed matmuls big enough to amortize a
handoff, and there is one of those per step.

**Where it does work is where mlx-dspark put it.** Prefill's matmuls are a
chunk wide and take milliseconds each, so a boundary disappears into them --
which is why their CPU co-prefill is a prefill feature and measures 1.41x
there. This port's prefill does not want it either, for an unrelated reason:
its GPU work already corresponds to ~1230 tok/s and what it loses is memory
pressure, not arithmetic.

So the honest end of this line: the physics is favourable, the verification is
achievable, and the *plumbing* is what rules it out. Recorded here with the
arithmetic so it is not re-derived from the bandwidth number alone, which
reads much more invitingly than it should.

---

## Working notes

- Generate references with the **vendored** mlx-lm, not the installed one:
  `cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_<case>.py`.
  Vendored is 0.32.0 (has the packed gated-delta kernel); installed is 0.31.3.
- `refs/` is not in the repository -- 510 MB of raw tensors, and every case
  has a writer. A fresh checkout has to run the `dump_*.py` above before
  `cargo test` has anything to compare against.
- `cargo test --release` runs everything but the full-model and end-to-end
  tests. Those need `--ignored` *and* `--test-threads=1`, because each one
  loads the whole 25.5 GB model and two at once do not fit:

      cargo test --release --test model    -- --ignored --nocapture
      cargo test --release --test generate -- --ignored --nocapture --test-threads=1
      cargo test --release --test mtp      -- --ignored --nocapture --test-threads=1

  The MTP references also need MTPLX on the path:
  `PYTHONPATH=../../vendor/mlx-lm:../../vendor/MTPLX python3 dump_mtp.py`.
  So do the vision ones, and there MTPLX is the *only* reference -- mlx-lm has
  no vision path:

      python3 dump_resample.py                 # Pillow only, no model
      PYTHONPATH=../../vendor/mlx-lm:../../vendor/MTPLX python3 dump_vision.py
      PYTHONPATH=../../vendor/mlx-lm:../../vendor/MTPLX python3 dump_vision_generate.py

  `dump_vision.py` writes its three test images into `refs/vision/` and
  `dump_vision_generate.py` writes a `sips`-converted photograph into
  `refs/vision_generate/`, so both runtimes read the same bytes and neither
  decodes anything. Run `dump_vision.py` first: the generate case copies one of
  its images.

      cargo test --release --test vision_generate -- --ignored --nocapture --test-threads=1
- `cargo run --release --bin bandwidth` measures what this machine sustains:
  a read-only kernel at several sizes, then `quantized_matmul` at M = 1 on real
  weights. Run it before believing any "% of peak" claim.
- `ORNITH_PROF=1` also attributes each barrier to the pair of dispatches that
  forced it, counts host round trips, and prints the epoch histogram and the
  pool's peak; `ORNITH_RECYCLE=1` lets the pool hand back a buffer the current
  barrier epoch has touched, as mlx does, which is worth 30 invented barriers
  a token and 0.8% of GPU busy; `ORNITH_TRACE=<path>` writes one step's
  dispatches with every binding, which `tools/epoch_schedule.py <path>`
  levelises into the fewest epochs the dependencies permit. The barrier and
  epoch counts do not care what else the machine is running -- they are the
  numbers to reach for when it is busy.
- `ORNITH_PROF=1` prints the per-token breakdown at the end of a run;
  `ORNITH_COMMIT_EVERY=N` sets the commit size (`N=1` attributes GPU time per
  kernel, and is ~3x slower); `ORNITH_SERIAL=1` puts back the serial encoder
  that barriers between every pair, as the control for the selective one;
  `ORNITH_NO_BARRIER=1` drops every barrier, which is wrong but bounds what
  they cost. `cargo run --release --bin dispatch_cost` times an empty
  dispatch.
- `tools/dump_frspec.py` writes `refs/frspec/ids_64k.bin`, the ranked
  sub-vocabulary `--frspec` drafts from. It is MTPLX's artifact
  (`mtplx/data/qwen38_code_ranked_64k.npy`, same tokenizer, 65536 ids) and is
  not copied into this repository for the same reason `refs/` is not.
  `ORNITH_FRSPEC=<path>` overrides where it is read from.
- `tools/gen_unicode.py` regenerates `src/tokenizer/tables.rs`. It is checked
  in; only re-run it when the Python that produced it moves to a new Unicode
  version, and re-run `tests/tokenizer.rs` afterwards.
- **Price the options before choosing between them, not after.** The first
  version of `REBUILD.md`'s Phase 4 moved all 260 eight-bit modules at once and
  put mxfp4 first. Both were wrong, and both were computable from files already
  in the repository: `baseline/config.json` plus the architecture says four
  families carry 57% of a step's traffic while the 80 gate modules carry 0.8%,
  and comparing mxfp4's power-of-two scale with no zero-point against affine
  int4's bf16 scale *with* one says it is the weaker format for weights -- so
  it was the wrong thing to ask to succeed where affine int4 had already
  failed. Neither needed a measurement, only arithmetic that was skipped
  because a plausible mechanism felt like a conclusion.

  The same shape recurred all session: CPU co-execution estimated at +25% and
  priced at 2.4% once the command-buffer boundary was counted; "the original
  weights are gone" asserted without opening the artifact's own
  `conversion_receipt.json`; scheduling called closed qualitatively before it
  was priced at 12% and then measured at 3.3%. Every time, the data was
  already on disk. **When about to recommend one option over another, first
  compute how the quantity being optimized is distributed across them** -- in
  this project that is usually `config.json`, the per-kernel table, or
  `--bin bandwidth`.
- When a composed test fails and its parts pass, suspect the plumbing between
  them — and write a narrower dump rather than reasoning about it. Both the
  buffer-pool race and the dispatch-grid overrun were found that way.
- Reference shapes must cover every dispatch branch. T = 1, 7 and 512 all
  passed while T = 24 was broken, because only 24 hits sorted + `gather_qmv`.
- The same lesson bit the sampler: after top-k and top-p the model's own
  distribution is peaked enough that every draw lands on the same token, so
  `dump_sampler.py` also draws from a flat synthetic distribution where the
  random number actually decides.
- Only one model fits in memory at a time. Two `ornith` processes, or two
  `--ignored` tests in parallel, thrash 64 GB and look like a hang. With the
  vision tower it is 26.4 GB rather than 25.5.
- A bfloat16 expression in Metal is not a chain of bfloat16 arrays. `T c = a *
  b + d;` lets the compiler keep `a * b` at float precision, while mlx writes
  `a * b` to a bfloat16 array and reads it back. Give every intermediate its
  own `T` variable; `gelu_pytorch_tanh` disagreed on 9 of 65536 inputs until it
  did.
- `zsh` does not word-split an unquoted `$flags`, so a benchmark loop that
  passes `"--mtp-depth 1"` through a variable hands the binary one argument and
  it panics on an unknown flag. Use `${=flags}` or write the runs out.
