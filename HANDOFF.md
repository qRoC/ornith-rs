# ornith-rs — state of the port

A from-scratch Rust + Metal port of the MLX inference path for
`~/.mtplx/models/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx`.

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
`fp_quantized.metal` for `Mxfp4` / `Mxfp8` / `Nvfp4` (uint8 scales -- E8M0 for
the mx pair, E4M3 for nvfp4 -- no biases, group size fixed at 32, or 16 for
nvfp4) as a software path whose own source note reads "fp benefits on every
gen"; `fp_quantized_nax.metal` is the Metal 4 tensor variant. Per weight:
affine int4 g32 costs 5.00 bits, MXFP4 4.25, nvfp4 g16 4.50, affine int8 g64
8.50, MXFP8 8.25.

**The nvfp4 half of that is now ported, run and measured, and the answer is
no.** `kernels/fp_quantized_inst.metal` instantiates nvfp4 g16 for bf16 and
f32; `src/ops/quantized.rs` carries both halves of mlx's dispatcher behind a
`QuantMode`; `tests/quantized_nvfp4.rs` pins every branch at `max_abs == 0.0`
and `tests/model.rs` runs the whole 40-layer model on an nvfp4 artifact at the
same bar. FR-Spec works on it too, which is not free: nvfp4's scales are uint8
E4M3 rather than bf16, so pruning `lm_head` needs a byte row gather
(`gather_rows_by_id_uint8`). The check that it is right is the acceptance
rate, not the output -- a wrong gather would still emit the verify forward's
argmax, just from garbage drafts -- and acceptance does not move at all,
0.94 with the pruned head and 0.94 with the full one. It delivers its bytes -- 2.076 -> 1.925 GB a token against
`int4-head-linattn`, and 9.17 -> 8.55 ms of GPU busy, which is the byte saving
converting almost one for one, as a memory-bound step should. It does not
deliver its quality: the max KL goes 0.048 -> 0.429 nats, which leaves it
**strictly dominated by `int4-except-gates`** on bytes, GPU busy *and* KL. The
measured reason is that nvfp4's error on this model's own weights is
10.4-13.1%, not the 9.2-9.5% a synthetic study predicted -- and the root
cause is that E4M3's smallest normal is 2**-6 where this model's group scales
sit near 2**-7.7, so **99.6% of them are subnormal**. nvfp4's optional
`global_scale` exists to fix precisely that and `mlx_lm.convert` never passes
one; with it the same weights measure 9.4-9.5%. So the verdict is on the rung
the toolchain can forge, not on the ceiling of the format. `REBUILD.md`,
"nvfp4, built and measured", has the whole thing including what a
global-scaled build would cost; `recipes/README.md` has the table.

mxfp4 and mxfp8 are **not** ported and the instantiation file says why: mxfp4
measures 20-22% weight error against affine int4's 8-9%, and mxfp8 buys 0.8%
of a step's bytes for 4.6x the error of the affine int8 it would replace.
`QuantMode::from_config` names them rather than failing on a missing kernel,
so an artifact forged in either mode says so instead of crashing.

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
kernels/q4k.metal          ggml's Q4_K, ported from llama.cpp rather than mlx
kernels/q6k.metal          ggml's Q6_K, for the 8-bit side
src/ops/kquant.rs          both k-quants' dispatch; the only modes here mlx
                           does not have, with ggml as the oracle in its place
src/ops/linear.rs          one quantized linear, mlx's family or ggml's
src/ops/invariant.rs       the batch-invariant mode: a row of a batch
                           computes what that row would compute alone
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
src/bin/probes.rs          the 64 probe contexts, captured from the port --
                           the instrument for artifacts mlx cannot load
src/bin/invariance.rs      a window against the same tokens one at a time,
                           from the same cache state, and which layer moved
src/bin/draft_cost.rs      a speculative cycle's two halves measured apart --
                           a draft level alone and a verify at each window
                           width, which is what Phase 26's subtraction could
                           not see
src/bin/window_mv.rs       what a window of rows costs one module: the
                           mat-vec as dispatched against the tile, and
                           against mlx's own choice on an affine module
src/bin/step_inventory.rs  a decode step's dispatches with their bytes, their
                           grids, their time and their time with the work
                           clamped away -- and what one class of them costs
                           where it actually runs
kernels/ornith_norm.metal  the residual add and the norm that reads it, in
                           one dispatch and two outputs
tools/epoch_slack.py       which single dispatch is the slowest thing in its
                           own epoch, how far it may move, and what that
                           would save
tools/dump_*.py            golden-reference writers (run against MLX)
tools/export_probes.py     probe_set.py -> refs/probe_set.json, for the port
tools/probes_from_logits.py  the port's logits -> probes.bin, and the
                           port-against-mlx check that says it may stand in
tools/epoch_schedule.py    how few barrier epochs a step's dependencies allow
M5.md                      what changes on a GPU where nax_available() is true
REBUILD.md                 rebuilding the artifact from its bf16 parent
IDEAS.md                   what designing the artifact buys, with the arithmetic
tools/quant_error.py       every quantization number in those two, reproducible
tools/weight_error.py      what a variant's format actually cost, against the
                           bf16 parent, one module at a time
tools/rotation_ceiling.py  what a QuaRot/SpinQuant rotation could buy, priced
                           against the parent before writing one
tools/awq_expert_ceiling.py  the same for an activation-aware scale on the
                           experts, both sides of the block
tools/dump_awq_experts.py  per-(layer, expert, channel) saliency -- the
                           per-expert object the Hessians could not be
tools/step_bytes.py        bytes one decode step reads, from the safetensors
                           headers rather than by hand
tools/aq.py                mlx's affine_quantize in numpy, bit-exact, with the
                           (scale, bias) encoding swappable
tools/forge_fp8scale_twin.py  a 4.50-bit format's stand-in, written in the
                           5.00-bit layout so it needs no kernel to measure
tools/gen_unicode.py       the Unicode tables the tokenizer needs
refs/awq-experts/          that saliency, 21 MB, the first per-expert
                           statistic this rebuild has
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
ORNITH_INVARIANT=1 cargo run --release -- --mtp      # ... and exactly what
                                                     # --greedy would print
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
| `quantized.rs` | `quantized_matmul` on real weights, **affine** 4-bit/g32 and 8-bit/g64, M = 1/2/4/8/16/64/256 (qmv_fast, qmv_wide, qmm_t_splitk + reduce) |
| `q4k.rs` | ggml's **Q4_K**, one of two cases with no MLX behind it: `q4k_dequantize` against ggml's own `dequantize_row_q4_K` at `max_abs == 0.0`, the matvec against a float64 reference and against a one-hot activation, the routed gather over four experts, the three-tensor loader against a forged one-module artifact, and (`--ignored`) every module of a real artifact against ggml digest for digest. Then the **tile**: the same float64 and one-hot checks at M = 37 and at a 100-row matrix (both partial-tile bounds), the routed tile over runs of 5/40/3/22 rows that straddle the 32-row tile three different ways, the same rows shuffled (unsorted is slow, not wrong), and -- the check a replacement owes -- agreement with the matvec it replaces, at 2.0e-6 |
| `q6k.rs` | the same checks for **Q6_K**, whose traps are its own: a *signed* sub-block scale and a code biased by 32, so `w = d*sc*(q - 32)` with no minimum |
| `quantized_nvfp4.rs` | the same, in **nvfp4 g16**, plus both gathered branches and `dequantize`. Carries its own weights -- the control artifact's, recut -- so it runs whatever model is on disk and could check the kernels before a build was spent on them |
| `batch_invariance.rs` | the one case whose oracle is *this machine*: row `r` of an M-row quantized matmul against row `r` computed alone, both families, dense and routed. Without `ops::invariant` they differ -- 7.8e-3 on an `lm_head` logit -- and with it `max_abs == 0.0`. Plus what the window mode promises and where mlx's fused-SDPA limit ends it |
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

### The divergence the probe set found, and it is not a bug

Every test here asserts `max_abs == 0.0`, and every one of them passes. Run
the port over the *64 probe contexts* instead -- `src/bin/probes.rs`, below --
and **58 of 64 come back bit-identical to mlx and six do not.** Worth
following, because the answer says something about what bit-exactness can
mean at all.

The six are `arith/ar-2p2`, `script/el-capital`, `script/hi-capital`,
`morph/ru-prep`, `longrange/password` and `format/sql-order`. They share no
length (8 to 395 tokens) and no shape. Everything cheap was excluded first:

- **Not the tokenizer.** All 64 probes encode to identical ids against
  `mlx_lm`'s tokenizer, not merely identical counts.
- **Not nondeterminism.** `dump_probes.py` run twice on the same artifact is
  bit-identical (`max_abs == 0`, 64 of 64 rows), and so is the port. Both
  sides are reproducible; they reproduce different numbers.
- **Not the compile flags.** Rebuilt without `-fmetal-math-mode=safe
  -fmetal-math-fp32-functions=precise` and again without `-std=metal4.0` --
  mlx's precompiled metallib is built with neither. The divergent element does
  not move.

`ORNITH_MODEL_PROMPT` makes `dump_model.py` dump the probe's own walk, and
`ORNITH_MODEL_LAYERS=all` keeps all forty layers instead of the frozen eight,
so `tests/model.rs` bisects it. On `morph/ru-prep`: layers 0-14 are exact,
**layer 15 -- a full-attention layer -- differs in 1196 of 14336 elements,
all of them in the last of the seven token rows.** Inside it, `q @ k^T`
differs in exactly **one of 784 scores**: query row 6 against key 0 of kv head
1, where mlx has 9.375 and the port has 9.3125. Those are adjacent bfloat16
values.

The dot product in float64 is **9.343749961815774**, and the midpoint between
those two neighbours is **9.34375**. The true value sits 3.8e-8 *below* a
rounding tie. mlx's float32 accumulator lands exactly on the tie and rounds to
even, which is up; the port's lands one float32 ULP below it and rounds down,
which is what the exact value does. **On this element the port is the more
accurate of the two**, which is not a defence of anything -- it is the point.
Two correct implementations of the same gemm, differing by one float32 ULP in
the accumulation order, cannot agree on a value this close to a tie. The same
kernel source compiled by a different toolchain is enough, and the runtime
here is mlx **0.32.2** against `vendor/mlx` at **0.32.3**.

What it costs, and this is the number that matters: that one ULP passes
through 25 more layers and comes out as **0.98 in a log-probability** and
**3.8e-3 nats of KL** on the worst of the six probes, and a different top-1
token on one probe of 64. What it does *not* cost is a verdict -- against the
bf16 parent the port's capture of `gptq-head-linattn` gives median 0.0297,
p90 0.2491, max 1.0122, which is mlx's own capture to four decimals.

So `max_abs == 0.0` against mlx remains the right contract for an op on fixed
inputs, and it is not available for a 40-layer forward over arbitrary text.
Between the two sits a band -- six probes in 64, a thousandth of a nat -- that
is a property of bfloat16 and the toolchain rather than of this port.

### Capturing the probe set from the port

```
cd tools && python3 export_probes.py              # probe_set.py -> refs/probe_set.json
ORNITH_MODEL=<artifact> ORNITH_PROBES_OUT=refs/artifact/<capture> \
  cargo run --release --bin probes                # logits.bin + logits.json
python3 tools/probes_from_logits.py refs/artifact/<capture> [<mlx capture>]
```

`Q4K.md` needs this: an artifact in a mode mlx has no quantization mode for
cannot be loaded by `mlx_lm.load`, so `dump_probes.py` stops working exactly
when the format under test is the interesting one. The port runs it. The
capture writes raw logits and leaves `lp = logits - mx.logsumexp(logits)` to
mlx, so the one arithmetic step outside the model is the same code on both
sides. With a second argument the last step also reports `max_abs` against an
mlx capture, which is how the paragraph above was measured.

---

## The k-quant artifacts, and what they cost

Five artifacts built on ggml's k-quants, all of them unreadable by `mlx_lm`
and all measured by `src/bin/probes.rs`. `REBUILD.md` Phases 15-18 have the
method; this is the table.

    artifact              modules          bytes/token   median KL over 64 probes
    int4-head-linattn     the baseline      2.0761 GB          0.0350
    gptq-head-linattn     ships today       2.0761 GB          0.0273
    q4k-imatrix           91 Q4_K           1.9814 GB          0.0260
    q4k-all               187 Q4_K          1.9311 GB          0.0237
    q6k-all               + 128 Q6_K        1.8051 GB          0.0342
    q6k-full              same, all weighted 1.8051 GB         0.0244

Those six were captured through the mat-vec prefill, which is no longer the
prefill (Phase 19). Re-captured on the tile, the four that were re-run keep
their order -- `q4k-all` 0.0301, `q6k-full` 0.0344, `q6k-all` 0.0370,
`int4-head-linattn` 0.0422 -- and `max` is unchanged to four decimals on the
k-quants. The median is the unstable statistic here, as Phase 15 found; the
sign tests are coin flips, and mlx's own affine gemm moves a capture by more
than this tile does.

**One recorded verdict does not survive the move.** Phase 17's `q6k-all` vs
`q4k-all`, the only significant result in nineteen phases, is 20 of 57
(p = 0.033) on the mat-vec and **23 of 58 (p = 0.148) on the tile**. Phase
18's finding -- that weighting the conversion is what makes it viable -- is
unchanged on either instrument; the loss it explained is not.

**-13.1% of a decode step at no quality difference this instrument can see.**
Every pairwise sign test over the 64 probes is a coin flip except one, and
that one -- `q6k-all`, 22 of 64 -- was 48 modules quantized without an
importance vector rather than a property of the format. Decode runs 110-124
tok/s against the baseline's 102.6, best of four, and two artifacts with
identical traffic measured 110.4 and 124.2, so read that range as the
machine's spread.

Regenerating the pieces, none of which are in git:

    cd tools && python3 export_probes.py                   # the probe list
    ORNITH_GGML_BUILD=../../.ggml-build python3 dump_q4k.py # refs/q4k
    ...                                       python3 dump_q6k.py # refs/q6k
    ORNITH_GPTQ_FAMILIES=attn ORNITH_GPTQ_OUT=../refs/gptq-attn \
      python3 dump_gptq_stats.py <artifact>   # the 8-bit side's Hessians, 218 s

and `vendor/ggml` must be built first -- see the header of `tools/q4k_oracle.c`
for the cmake line. `tools/forge_q4k.py --help` is its own docstring.

Two measurement escape hatches, both inert by default, both there because a
kernel that replaces a working kernel has to be compared against it in one
sitting on a machine whose wall clock swings 3x between sittings:

    ORNITH_KQUANT_GEMM=0    k-quants take the mat-vec at every M
    ORNITH_AFFINE_GEMM=0    mlx's affine modules take *its* mat-vec at every M
    ORNITH_INVARIANT=0      *off*; the window mode is the default, and this
                            is the way back to what mlx dispatches
    ORNITH_INVARIANT=all    every M, not just a verify window's rows, which is
                            what a probe capture wants
    ORNITH_KQUANT_TILE_ROWS=16|32|64       the dense tile's width
    ORNITH_KQUANT_TILE_ROWS_ID=16|32|64    the routed tile's, which otherwise
                                           follows `rows / experts`
    ORNITH_KQUANT_TILE_HALF=1              ggml's `half` tile scalar

The second is the control for the first: it makes the same move on a path
nobody here wrote, which is how Phase 19 established that a tiled prefill
perturbs a probe capture regardless of the format, and that this tile perturbs
it less than mlx's own does.

    cargo run --release --bin kquant_mm     # where the two kernels cross, per M

**The debt was prefill, and `REBUILD.md` Phase 19 paid it.** M > 1 used to go
through ggml's token axis -- a mat-vec per token, so the weights were re-read
once each instead of being tiled, which is invisible on a short prompt and not
on a long one. `kernels/kquant_mm.h` is ggml's `kernel_mul_mm` for both
formats, and `kquant::MM_ROW_LIMIT` (16, measured) is where the dispatcher
switches.

**Phase 23 then tuned its shape**, which Phase 19 had left as ggml's. The
tile's width is a template parameter now and three are instantiated: the dense
path takes 64 rows, the routed path 16 or 32 by `rows / experts`, because a
routed tile wider than the runs straddles expert boundaries and pays a whole
pass over K for each. Worth **-4.3% of prefill GPU at 748 prompt tokens, -1.3%
at 3402**, and the output is *byte-identical* -- all three widths agree at
`max_abs == 0.0` and 64 probes come back the same 63,569,920 bytes. The guess
Phase 19 left behind ("a narrower tile would help") was backwards: flat time
across M = 1..32 means the fixed per-block cost is 32-60% of the total, so the
tile wanted to be wider. ggml's `half` tile scalar, which this port declined
for exactness, is priced in the same phase at 0-4%: the exactness was free.

**And Phase 25 spent the lever Phase 23 named, with nothing to show.** The
tile takes `K`, `O` and `M` as `constant int&`, and all 13 of this artifact's
k-quant shapes satisfy `O % 64 == 0` and `K % 256 == 0`, so function constants
can tell it both. Built, held bit-identical, and measured in one process
alternating the arms: **0.96 to 1.05, scattered around 1.00.** Reverted.

The reason is in the code rather than in the hardware: the tail clamps are
computed *once per tile*, before the K loop, not once per K step, and the K
loop's bound test is one compare against a body carrying two
`threadgroup_barrier`s. Phase 23's 12-23x unroll cliff was real and was about
the **innermost** loop, whose operands are registers. A codegen fact is worth
what the thing it applies to costs. In GPU seconds, which is the number to read on this machine:

    prompt tokens        affine    q4k-all              q6k-full
                         int4      mat-vec    tile      mat-vec    tile
    599                   0.69      1.43       0.68      1.79       0.69
    2932                  2.49      6.84       2.71      8.82       2.75

**2.75x and 3.54x the affine artifact's GPU work, down to 1.09x and 1.10x** at
2932, and to exactly 1.00x at 599. On the wall, taking only draws that ran at
93% GPU busy or better: `q6k-full` 325.7 -> 1035.5 tok/s at 2932 and
310.4 -> 815.3 at 599; `q4k-all` 399.8 -> 836.9 at 599. Decode is untouched,
M = 1 being far below the crossover.

Read GPU seconds and not the wall here. Within a configuration the GPU holds to
1% across draws (6.85, 6.90, 6.84) while the wall swings by 3x (3.78, 8.30,
11.46 s for 2.49, 2.57, 2.64 s of GPU) and by 4.4x at worst. And at 2932 the
`q4k-all` tile draws
all came back at 34-41% busy -- a third of the GPU work inside the same wall,
which is "prefill's half-idle GPU" below showing through from underneath.

The one thing it did change is the instrument. A tiled prefill perturbs a
probe capture -- and so does mlx's own, by more than this one; Phase 19 has
the control and the sign tests, which are coin flips. The consequence worth
carrying: **the k-quant captures in Phases 15-18 were taken on a mat-vec
prefill and every affine capture in this rebuild on a tiled one.** The
orderings survive it.

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

**A rebuild will quietly take the artifact's own pattern away.** `mlx_lm`'s
`save()` ends in `tokenizer.save_pretrained`, so a forged artifact gets its
tokenizer written back out by whatever `transformers` is installed instead of
carrying the parent's files through. transformers 5.14.1 writes the *older*
pattern -- no `\p{M}` in either class -- into `tokenizer.json`, flips
ByteLevel's `trim_offsets` and the decoder's `add_prefix_space` and
`use_regex` from false to true, rewrites `merges` from `"a b"` to
`["a", "b"]`, folds the config's 7 extra added tokens into the file's 26, and
writes no `vocab.json` or `merges.txt` at all -- which is what
`Qwen2Tokenizer` rebuilds itself from, so the default dialect loses its
source too. The published artifact carried the parent's four files byte for
byte; `tools/align_rebuild.py` puts them back, and `tools/check_rebuild.py`
has three invariants that notice if it was not run. No golden test can see
this, because `dump_tokenizer.py` reads the same file the port does.

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
  default here.

  **Those numbers are the deleted artifact's head, and it had a bug.** Its
  sidecar carried six of its seven RMSNorm gains shifted to the absolute
  convention and left `post_attention_layernorm` delta-encoded, a full 1.0
  below where it belonged -- the build rule that made it decided per tensor,
  and that gain's mean sits on the wrong side of the threshold. Reproducing
  exactly that on the rebuild reproduces the Phase 0 acceptance profile on all
  eight runs, to the cycle. The rebuilt sidecar shifts all seven and drafts
  better at every depth. The trunk is unaffected and bit-identical.
  `REBUILD.md` Phase 2, "The MTP sidecar", has the table; the short version is
  that every acceptance figure in this section was measured on the broken head
  and reads higher on the fixed one. Seeding the head's cache over the prompt -- what
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

### It does pay off, and the section that said otherwise was measuring a bug

> **OVERTURNED 2026-09-18.** Everything below was measured on the **published**
> MTP head, the one shipping `post_attention_layernorm` still delta-encoded
> (`REBUILD.md` Phase 2, and the norm bug it found). That head's acceptance
> collapses with depth -- 0.53 / 0.34 / 0.35 -- so every draft past the first
> was nearly free work thrown away, and the honest conclusion at the time was
> that speculation did not pay. The rebuilt head holds **~0.72 all the way to
> depth 3**, and nobody re-ran this section after the rebuild. Re-run now, same
> prompt, 200 tokens, greedy:
>
> | | wall tok/s | GPU busy ms/token | idle ms/token | acceptance by depth |
> |---|---:|---:|---:|---|
> | autoregressive | 84.6 - 87.1 | 10.54 - 10.80 | 0.81 - 0.93 | -- |
> | `--mtp-depth 1` | 84.0 | 8.43 | 2.81 | 0.78 |
> | `--mtp-depth 2` | 95.4 | 8.45 | 1.81 | 0.74 / 0.65 |
> | `--mtp-depth 3` | 97.0 | **8.38** | 1.84 | 0.72 / 0.71 / 0.74 |
> | `--mtp-depth 4` | 79.0 | 9.99 | 2.57 | 0.63 / 0.73 / 0.69 / 0.74 |
> | `--mtp-depth 5` | 83.3 | 9.98 | 1.92 | 0.78 / 0.76 / 0.68 / 0.46 / 0.58 |
> | **`--mtp-depth 3 --frspec`** | **101.2** | **7.76** | 2.02 | 0.71 / 0.71 / 0.72 |
> | `--mtp-depth 5 --frspec` | 91.8 | 8.65 | 2.15 | 0.79 / 0.75 / 0.66 / 0.48 / 0.50 |
>
> **Depth 3 with FR-Spec does 27-28% less GPU work a token than
> autoregression and is 16-20% faster on the wall**, on a machine carrying
> 18.5 GB of swap. The knee is depth 3: depths 4 and 5 give the deep levels
> back (0.46-0.58) and regress on both columns.
>
> What survives from below, unchanged: **the stall is real and still costs**.
> Idle is ~2.0 ms a token against autoregression's 0.9, so ~10% of the wall is
> still being left on the floor, and the diagnosis of *where* -- the
> `gather_rows_by_id` gap -- is untouched by any of this. Fixing it would put
> depth 3 + FR-Spec near 114 tok/s, ~33% over autoregression.
>
> **The lesson is not about MTP.** A conclusion was drawn, written down as
> settled, and then the thing it measured was replaced by a bug fix in another
> file. Nothing re-ran it, because nothing connected "the MTP sidecar had one
> norm wrong" to "speculative decoding does not pay here". When a component is
> fixed, the measurements that condemned it are stale, not confirmed.
>
> **And a second thing was inflating the rejections, found the same way.**
> `REBUILD.md` Phase 21: a verify row and the autoregressive step it stands in
> for were not the same arithmetic, so 2.9% of verify rows came back with a
> different argmax and the drafts that matched the *autoregressive* answer
> were rejected for it. `ORNITH_INVARIANT=1` closes that -- acceptance
> 42.7% -> 50.0% at depth 3 on the prompt measured there, at no GPU cost, and
> the emitted text becomes byte-identical to `--greedy`. The depth knee does
> not move: a window of five leaves mlx's fused SDPA, so depth 3 is both the
> fastest depth and the deepest exact one.

The historical record, on the published head. The wall column was how this was
first read; the GPU column is what it should have been read by, for the reason
in "Where a decode step goes":

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
(On `q6k-full` with FR-Spec, Phase 27 measures a draft level at **0.587 ms**
directly -- `src/bin/draft_cost.rs`. Making the draft cheaper is not where
this goes.)

The two structural costs, which are why deeper is worse:

1. **Every draft step runs the full `lm_head`.** It is 8-bit over a 248320
   vocabulary, so ~540 MB read per draft -- ~1.6 GB for a depth-3 chain,
   comparable to a whole trunk forward, for three tokens that are then mostly
   rejected. This is exactly what MTPLX's FR-Spec (`frspec_ids`, a pruned
   draft head over a ranked sub-vocabulary) exists to avoid, and it is not
   ported.
2. **The verify forward multiplies MoE traffic**, and nothing can amortise it.
   Its `depth + 1` tokens each pick their own top-8 of 256 experts. The
   sentence that used to stand here said a 4-token window reads ~30.5
   distinct experts for its 32 (token, expert) pairs; that is
   `256 * (1 - (1 - 8/256)^4)`, the expectation under *independent* routing,
   and it was never measured. **Routing is not independent**:
   `tools/router_stats.py` measures **20.85** distinct, 1.53 rows per expert,
   and adjacent tokens share 2.84 of their 8 experts against 0.25 under
   independence -- shuffling the token order lands back at 28.3, which is the
   control that says the structure is temporal and real.

   **The conclusion survives the correction anyway, and this is the
   measurement that settles it:** `cargo run --release --bin moe_reuse` holds
   the row count fixed and varies only how many distinct experts those rows
   name. At `d = 4` the gather moves 3% between 32 distinct and 8 -- 0.334 ms
   against 0.324 -- at ~221 GB/s, which is this machine's sustained
   bandwidth. **It pays for rows, not for experts**, so the reuse is real and
   unreachable with these kernels. Sorting is the mechanism that would group
   them, it costs 22-40% more when forced, and it does not apply: `GatherQMM` takes
   the batched rhs-sorted path only at `rows / experts >= 4`, which at 256
   experts means 1024 rows, i.e. 128 tokens. `switch_glu_sorted` has a
   `force_sort` hook and using it here would buy a counting sort and two
   gathers and still land in `gather_qmv`. This was once written down as
   "forbidden because it would stop matching mlx-lm"; it is simply not
   available.

   **Phase 27 sizes it against everything else the window costs**, which is
   what this item never had: the expert gather is **30%** of what a verify
   pays per extra row on `q6k-full`. The larger share, 41%, is the dense
   k-quant mat-vecs -- which Phase 27 read as a re-read that could be taken
   back exactly, and Phase 28 measured as arithmetic that cannot.

   **And Phase 29 measured this 30% the same way, on the shipping artifact,
   and it is arithmetic too.** Hold the rows fixed and vary the distinct
   experts behind them and the time does not move: 32 rows cost the same
   whether they name 26 experts or 8, 48 rows the same at 48 or 8, and sorting
   them into runs so each expert is read once costs **23-44% more**. The
   gather pays for the arithmetic of its rows, not for the bytes of its
   experts, so there is nothing here to group.

A dense model would win here; a 3B-active MoE with an unpruned draft head does
not. FR-Spec is the fix for the first of those, and it is in -- see below.

### On a k-quant artifact the knee is depth 1, not depth 3

Everything in this section was measured on the affine artifact. `REBUILD.md`
Phase 26 re-measured it on `q6k-full`, and the ordering is not the same --
because FR-Spec could not run there at all until that phase wrote the row
gather a k-quant `lm_head` needs, and because quantizing the trunk changes what
a draft is worth.

| `q6k-full`, 300 tokens, greedy | wall tok/s | GPU ms/token | acceptance |
|---|---:|---:|---|
| autoregressive | 108.9 | 8.70 | -- |
| `--mtp` (depth 3) | 94.5 | 10.08 | 0.69 / 0.57 / 0.53 |
| `--mtp --frspec` | 102.1 | 9.49 | 0.68 / 0.56 / 0.54 |
| `--mtp-depth 2 --frspec` | 114.9 | 8.40 | 0.68 / 0.55 |
| **`--mtp-depth 1 --frspec`** | **127.4** | **7.51** | **0.72** |

**-13.7% of a decode step and +17% on the wall**, reproducible to 0.4%. The
mechanism is that **the trunk got 13.1% cheaper and the draft did not**: the
MTP sidecar is 1.7 GB of bfloat16 that no phase of the rebuild has quantized,
so each draft level costs what it always cost while the thing it is amortised
against shrank.

**Phase 26 then decomposed the cycle by differencing those two depths, and
that decomposition was wrong.** It read a 4.31 ms draft and an 8.64 ms verify
out of `(cycle(3) - cycle(1)) / 2`, which charges the whole difference to the
draft levels -- right only if a verify costs the same over a four-token window
as over a two-token one. **Phase 27 measured each half on its own and it does
not.** `src/bin/draft_cost.rs`, one process, GPU seconds:

| | GPU ms | four-run spread |
|---|---:|---|
| one draft level, depth 1, with FR-Spec | **0.587** | 0.53 - 0.61 |
| an autoregressive decode step | 8.342 | 7.93 - 8.34 |
| a verify forward, window of 1 / 2 / 3 / 4 | 8.217 / 11.368 / 14.733 / 18.149 | |
| each extra window row | 3.15 - 3.42 | 3.0 - 3.9 |
| Phase 26's subtraction, on these numbers | 3.977 | 3.98 - 4.03 |

So a draft level is **0.587 ms, not 4.31**, and the missing 3.4 ms is the
verify forward, paid **per row of the window**. Written out, the subtraction
is `draft + (row3 + row4) / 2`, and Phase 26 read all of it as the draft.
Three consequences, all of which replace something this file used to say:

- **The draft is inside its byte budget, not 4.4x over it.** 204 MB in 0.587
  ms is 347 GB/s against a decode step's 219. Quantizing the MTP head saves
  93 MB, i.e. ~0.27 ms of a 12.0 ms cycle -- 2%, not the 0.45 of 4.31 that
  was written here.
- **The draft's 256-expert MoE is 0.167 ms**, 1.4% of a cycle. It is the worst
  shape this model has and it is not a lever; making it free would not be
  measurable in a run.
- **What scales with the window is the trunk doing the work again per row.**
  41% of the width cost is dense k-quant mat-vecs; 30% is the expert gather
  paying for 32 rows where ~23 are distinct experts; 29% is `forward_capture`
  stepping the gated-delta recurrence one token at a time. `REBUILD.md`
  Phase 27 has the per-kernel table and the three fixes that are *not* it (the
  grid axis order, the tile, `qmv_wide` -- all three built or measured and all
  three refuted).

  **Phase 27 read the 41% as the weights being re-read once per row, priced
  taking it back at -21% a token, and Phase 28 built it and measured -2.8%.**
  The re-read is not what a window row costs: on `lm_head` one row already
  sustains 520 GB/s, the machine's ceiling, and four rows through a kernel
  that reads the matrix once run at 192 -- past the first row the mat-vec is
  arithmetic-bound and the saved reads were hidden under the arithmetic. What
  did pay is hoisting the **dequantize** out of the row loop, and only where
  it is expensive: Q6_K -21% on that kernel, Q4_K +9% and left alone. Whole
  run: **-2.8% a token at depth 1 and -5.0% at depth 3**, with the acceptance
  byte-identical.

- **And Phase 29 priced the other two thirds, which leaves the row nearly
  closed.** The expert gather is arithmetic as well (above). Of the
  recurrence's 29%, the part that is *stepping* rather than work is **11% of
  a row** and no more: `forward_hidden` already runs the same trunk with one
  recurrence dispatch over the whole window, no snapshot slices and no output
  concat, and against `forward_capture` on the same window from the same cache
  state it is 1.09 ms cheaper at width 4 and exactly equal at width 1. A fused
  capture cannot take all of that -- it still writes `W` states where the
  plain form writes one. So the accounting is: 41% arithmetic (5% recovered),
  30% arithmetic (nothing), 29% of which at most 11% of a row is loose.

- **Phase 30 took 60% of that 11% without writing a kernel, and closed the
  rest as not worth one.** A captured token cost five dispatches a layer more
  than an uncaptured one; three of the five had nothing to do with the
  recurrence -- a conv snapshot taken per token of which `rewind` reads one,
  a concatenation of outputs that the kernel can write in place, and a gate
  recomputed per token on a slice of its own input. Removing them takes a
  width-4 verify from 2695 dispatches to 2065 and the cost of stepping from
  1.109 ms to **0.417**, i.e. 11% of a row to 4.4%, with both depths still
  printing byte-identical text. What is left is three recurrence calls a
  layer, and a fused kernel could only take back their state *reads* -- 2% of
  a row. **The lever this file listed as item 5's sibling is closed by having
  become unnecessary.**

### How exact it is: exactly, and it is the default

The acceptance rule and the rewind were always exact. The output was not, and
the reason was the batching: the verify forward runs `depth + 1` tokens at
once, `quantized_matmul` picks a different kernel for M = 4 than for M = 1,
and two logits an ulp apart swap. `window_and_steps_both_match_mlx` pins both
halves -- this port reproduces mlx's window forward *and* its single-step
forward bit for bit, and the two disagree with each other by up to 1.44 on a
hidden whose scale is 35, against top-2 logit gaps that are sometimes 0.31. So
the streams parted company after 8 - 18 tokens.

**This file used to conclude "and no speculative decoder on this stack can do
better, MTPLX's included". That was wrong, and mlx-vlm had the fix in the
tree the whole time** -- `models/quantized_verifier.py`,
`singleton_quantized_linear`, "fuse a runtime-sized batch while retaining
singleton qmv arithmetic". The window's rows do not have to be batched
arithmetic; they only have to be batched *dispatch*. `ops::invariant` makes
every batch of at most four rows compute row by row -- **on by default since
2026-09-19**, scoped out of prefill by call site so that its blast radius is
exactly the speculative path, and `ORNITH_INVARIANT=0` is the way back.
`REBUILD.md` Phase 21 has the whole measurement, and the headline is:

  - `speculative_tracks_autoregressive` went from **36 of 214** tokens agreed
    to **221 of 221**;
  - `--mtp --greedy` prints text byte-identical to `--greedy`, with and
    without FR-Spec -- and the autoregressive side needs no flag, because the
    window mode does not touch prefill;
  - acceptance rose **42.7% -> 50.0%** at depth 3 (0.71/0.71/0.58 by depth)
    for **no measurable GPU cost**, because `qmv` at M rows widens the grid
    rather than the traffic and the cache serves the repeated weight reads.

What the flag cannot fix is a window of five or more: mlx takes the fused
vector SDPA only while `q_len * gqa_factor <= 32`, and at 16 query heads over
2 kv heads that is four rows. `src/bin/invariance.rs` measures it -- a window
of 5 first disagrees at layer 3, the first full-attention layer. Depth 3 is
therefore the deepest draft this model can verify exactly, and it is also the
depth `mtplx_runtime.json` recommends.

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
level with it, by the changes mlx does not have: the MoE block issued
interleaved, the buffer pool declining to recycle inside a barrier epoch,
and -- Phase 31 -- the residual add fused into the norm that reads it, the
shared expert's one-row gate issued where a gather can hide it, and the
router's top-k selected rather than sorted.
And the speculative pairing does 14% less GPU work a token than plain decoding
while only tying on the wall, for the reason in "It does not pay off here".

| a decode step, now | affine artifact | `q6k-full`, after Phase 33 |
|---|---|---|
| GPU busy | 10.3-10.5 ms of a 10.85-11.34 ms wall -- 95-96% | **7.19-7.21** ms of a 7.65-7.68 ms wall, 130 tok/s |
| idle between command buffers | 0.47-0.56 ms a token, essentially one round trip | 0.49 ms |
| dispatches | 1669 per token, in 14 command buffers (1759 before single-row slices became views) | **1009** in 8 (1509 in 12 after Phase 31, 1309 in 11 after Phase 32) |
| barriers | 1064 per token (1216 at the 1:1 commit, 1669 before the barrier work) | **609** (908, then 787) |
| epochs | 1078 per token, 1.55 dispatches each | 617, 1.64 each (922, then 798) |
| host round trips | 1.00 per token | 1.00 |
| weights read | ~2.83 GB per token, i.e. **272 GB/s** against a measured ceiling of 465 and a sticker of 546 | **1.81 GB** |

**And half of that step is not weight traffic at all.** `REBUILD.md` Phase 31
measured it rather than estimating it: the same dispatch stream with every
grid clamped to a single threadgroup -- same barriers, same command buffers,
no work -- costs **3.4 ms of a 7.5 ms trunk, 46%**, and what is left moves
1.62 GB at ~400-440 GB/s, which is this machine's ceiling. Three changes came
out of that inventory and took **-8.9% a token** between them; the share of
the step that is launches was 46% before and after. Phases 32 and 33 then
took the launches themselves -- two block-shaped kernels, **-500 dispatches,
-299 barriers, -305 epochs, -4 command buffers** and 117 tok/s to 130 -- and
the share is now **38%**. The section "What the dispatches cost" below is the
whole of it.

**The prompt, re-measured 2026-09-19 on one text and one set of ids** -- 6394
raw tokens (no chat template, so both sides tokenise it identically), chunk
2048, three draws a cell:

| the prompt, 6394 tokens | affine parent | `q6k-full` |
|---|---|---|
| GPU busy | **4.14-4.21 s**, 1528 tok/s of GPU work (4.50-4.61 before Phase 35) | **4.65 s** (5.16-5.19 before) |
| wall | 6.26-12.46 s, **37-72% busy** | 5.29-5.71 s, **90-97% busy** |
| `mlx_lm.generate`, same artifact, same ids | **3.68-3.87 s, 1654-1740 tok/s** | mlx cannot load a k-quant artifact |

**Phase 36 re-measured this on a different text and the row above is not
comparable to the one below.** The prompt there is `baseline/ppl_corpus.txt`
at **6621** raw tokens -- the 6394-token text is not in the repository, and
`--tokenizer-json` does not produce it from this one. Same protocol, three
interleaved pairs of whole runs, and `mlx_lm` re-run in the same sitting:

| the prompt, 6621 tokens | affine parent | `q6k-full` |
|---|---|---|
| GPU busy, Phase 35's stream | 4.22-4.27 s | 4.88-5.01 s |
| GPU busy, after Phase 36 | **3.78-3.84 s** | **4.43-4.53 s** (its width term is still in front of it) |
| `mlx_lm.generate`, same ids, same sitting | 3.98-4.30 s of **wall**, 1541-1666 tok/s | mlx cannot load a k-quant artifact |

The affine figure is two interleaved pairs sharing an arm (the copies
indexed, then widened), and the shared arm reproduced across them at 3.97
and 3.99. **The port's GPU busy is now below mlx's best wall on this text,
which is not the same as being faster than mlx**: its GPU time can only be
under its wall, so what this says is that the remaining difference is inside
the spread of mlx's own three draws.

**Read that last row as a floor on the gap, not a measurement of it.**
`mlx_lm` reports a wall clock and the port's row is GPU busy; a wall swings
with load where GPU busy does not, and re-running mlx in a later sitting
gave 1650 *and* 1192 tok/s within one minute, so the best of its draws is
the estimator. Its own GPU time can only be below its wall, so the port's
remaining 12% is the smallest the difference can be.

Two things in that table, and the first one is the harder news.

**The port was behind mlx on prefill by 22% of *GPU work*, not just on the
wall, and is now behind by 12%.** 4.54 s against mlx's 3.72 s when this was
first measured; 4.17 s after Phase 35 built the missing instrument and the
first thing it found paid. It is the one object where the port loses to the
reference it was written against -- decode is 130 tok/s against mlx's 86-92.
The older row here read "2600 tokens, ~1230 tok/s of GPU work, mlx_lm 809"
and both sides have roughly doubled since; the *sign* of the gap is what did
not change.

**What `step_inventory prefill` says that the decode bin does not**, and
most of it is about reading an instrument on a second object.

- **The clamp is not a launch floor at this width.** A clamped dispatch
  costs 50.5 us against a decode step's 2.55, because the surviving
  threadgroup of a *tile* still walks the whole K. So the launch cost is the
  dispatch count times ~2.09 us, **3.1 ms of 1408, 0.22%**: there is nothing
  here for a *launch* fusion to remove. The fusions still pay, and more than
  at decode -- the gated-delta front end is **-5.3% of a chunk against -4.3%
  of a step** and the MoE tail **-4.7% against -3.6%** -- because what they
  remove at this width is intermediates, not launches. `rms_gated`, which
  measures exactly zero at decode, reads -17.8 ms here.
- **Divide a per-dispatch row by its own run's level**, and take the level
  from the same run. One dispatch a command buffer inflates a chunk by 8%
  against a decode step's 29%, so the table is nearly a decomposition here
  and explicitly is not one there -- but two prefill runs read 1522.8 and
  1752.7 ms for the same chunk, and crossing them inflates every share
  by 15%.
- **The machine throttles across a prefill run, and the boundary section
  does not survive it.** A cell is 16 chunks of 1.4 s at full ALU; over
  eleven cells the *base* arm climbed **1407 ms to 2100-2200**,
  monotonically. Interleaved pairs inside a cell survive that; comparing
  cells does not, and the partition check said so at once -- three disjoint
  buckets at **466%** of the whole. `border` stays a decode-shaped mode
  until it has a protocol that does not ask for four minutes of full-ALU
  work in one sitting.
- **And a ceiling's garbage is not free once it feeds a gemm.** Dropping
  `rms_gated` at a chunk read **+82 ms, 1 of 8** -- a removal cannot make a
  step slower unless what it left behind changed the arithmetic's cost.
  Phase 31's rule was that removal cannot price a dispatch whose output is
  an *address*; at this width it also cannot price one whose output is a
  number a gemm will multiply.
- **Read the bandwidth column, not only the milliseconds.** Phase 35 left
  two kernels named at 78.9 and 78.2 ms and the plan that followed was to
  remove them. The same rows said 37 and 68 GB/s on a 465 GB/s machine, and
  that was the larger finding: the copies were **two bytes a thread**, and
  a 16-byte access was worth four times what removing them was
  (`REBUILD.md` Phase 36). A kernel an order of magnitude under the
  machine's ceiling is not yet a kernel you know the cost of.

**And the idle is sized by the artifact, not by the era.** The same machine
within one hour ran the 25.5 GB affine artifact at 37-72% busy and the 21.5
GB k-quant one at 90-97%, with GPU busy holding to 2% in both. So "prefill's
GPU is half idle" is a memory-pressure statement about a particular
artifact's resident size, not a property of the prefill path -- and the
wall figure swings 2x between draws while the GPU figure does not, which is
the usual instruction to read the GPU column.

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

### What the dispatches cost, measured instead of counted

`REBUILD.md` Phase 31. Three knobs, and the one idea behind all three is to
run the *same* dispatch stream with one term taken out of it.

- `ORNITH_GRID1=1` / `Context::set_grid1` issues every dispatch with a single
  threadgroup: it launches, it pays its barrier, and it does none of its
  work. Its control is free -- a kernel already dispatched over one
  threadgroup is not clamped, so its two times are one measurement twice, and
  the 553 such dispatches differ by a median 0.13 us on a mean of 4.5.
- `Context::set_neuter` does the same to one *class* of kernel, or drops that
  class's dispatches outright while still counting them and still handing
  their bindings to the tracker -- so barriers, epochs and commits are
  untouched and only the dispatch is gone. Its zero is a pattern that matches
  nothing: 0.02-0.06 ms.
- `Context::set_neuter_many` takes a **list** of patterns, which is how a
  whole chain of kernels is priced at once rather than one class at a time --
  see "The boundary of the fusion project" below. `Context::prof_dropped`
  counts what the list actually removed, because a list that matches nothing
  is a measurement of nothing and reads as a boundary of zero.
- `cargo run --release --bin step_inventory` joins `ORNITH_COMMIT_EVERY=1`
  per-dispatch times to `ORNITH_TRACE`'s per-dispatch bindings, grids and
  layer markers, so every dispatch in a layer has its bytes, its grid, its
  time and its time with the work clamped away.

What they say, on `q6k-full`:

| the trunk (`model_call`) | before Phase 31 | after |
|---|---|---|
| as it runs | 7.54 | 6.74 |
| every grid clamped to one threadgroup | **3.48, 46%** | **3.09, 46%** |
| what is left, moving 1.616 GB | 4.05, i.e. 397 GB/s | 3.64, i.e. 443 GB/s |

1062 of the 1665 dispatches move under 64 KB each -- 7.4 MB between them --
and 95% of their isolated time is the floor. There is no slow kernel to
find; there is a very large number of launches.

**The floor and the marginal dispatch are different numbers.** 3.48 ms over
1665 is 2.09 us each, but that is the cost of a stream with nothing to
overlap: removing dispatches one class at a time returns **0.85 us** each,
which is Phase 30's 0.72 us measured again. Price a fusion with the second
number and a decomposition with the first.

**A command buffer costs ~6.6 us, not the ~60 the code said.** Sweeping the
commit size with the work clamped away puts 52 buffers a token 0.34 ms above
none. The default stays at 128: between it and 1024 there is ~0.1 ms of GPU
time and the wall takes it straight back.

**What removal cannot price.** Dropping a dispatch whose output is an
*address* rather than a number makes the step slower, not faster: without the
router's argsort the expert gather reads eight experts at random and loses
0.58 ms. Those need a **stand-in** rather than a hole -- `ORNITH_FAKE_SORT=1`
writes a valid order cheaply in place of the sort, and the 0.385 ms that
measured is what justified replacing it.

### The boundary of the fusion project, and what it costs to reach

`REBUILD.md` Phase 32. One class at a time prices one fusion; the whole
elementwise chain dropped at once prices the *project*, and the two are
different numbers, because removing one dispatch lets its neighbours overlap
into the hole and removing all of them cannot.

**The whole chain is 811 of 1505 dispatches and 1.35 ms of a 6.70 ms trunk --
20%.** That is the ceiling on every fusion that could be written here. Three
processes, 8 interleaved pairs a cell; the partition into three disjoint
buckets sums to 92-99% of the whole, which is what says the garbage output
did not change what the step reads.

**Per dispatch the chain is not uniform, and the category is not the
object.** `swiglu` is 110 dispatches worth 0.6 us each -- already hidden
under the expert gathers, so the textbook fusion, an activation into its
consumer matvec, is the worst-paid one here. `rms` and `add_rms` are worth
2.5 us each. The MoE block's tail was 240 dispatches worth 1.6 us each, and
that is the one that got a kernel.

**A fusion returns about half of what removing its dispatches measures.**
Phase 31's add+rms: 0.068 of 0.145, 47%. Phase 32's `moe_combine`: 0.21 of
0.375, 56%. Price the next one at half.

**That coefficient is about launches, and it does not hold for a pass over
memory.** Phase 36 removed two whole copies whose isolated cost was 150.6 ms
of a 1410 ms chunk and measured **-135 ms, 90%**, not 50%. The half comes
from neighbours overlapping into the hole a removed *launch* leaves; a
kernel that is 134 MB of traffic and 1.9 ms long has nothing small enough
nearby to fill it. Price a launch at half and a pass at most of itself --
and check which one is in front of you before applying either.

**What is left after `moe_combine`**: 611 dispatches and 1.14 ms of ceiling,
of which the gated-delta chain is 200 and 0.29, and everything a general
deferred op layer would be for is 361 and 0.63 -- so ~0.3 ms realized, 4.5%
of the trunk. That is the number the layer has to be worth.

**Phase 33 then wrote the gated-delta one and re-measured the rest.** Ten
dispatches a layer into one (`gdn_front`: the conv window's concatenation and
its tail, the depthwise conv, the silu, q and k's norms and scales, beta and
the decay gate), 300 a token into 30, **-0.330 and -0.346 ms on the trunk in
two processes, 16 of 16 pairs**. What is left is **311 dispatches and 0.89 ms
of ceiling**, of which a deferred op layer's own share is **211 and 0.49 --
~0.25 ms realized, 3.8% of a 6.5 ms trunk.** Phase 32 forecast ~0.3 and 4.5%
and was right.

**Two warnings came out of that kernel and both are about the set, not the
coefficient.** The bucket said 200 dispatches and -0.32, the half rule said
~0.15 ms, and the kernel returned 0.34 -- because `slice_inner` also names
the attention layers' q/gate split (20 dispatches that are not this kernel's)
while the 120 dispatches of `rms` and `scalar_mul` that *are* its own sit in
a different bucket. **A pattern serves a call site; a kernel serves a block;
for a rewrite those are different sets**, and nothing in this instrument can
name the second. Against the object that can -- isolated time in the
inventory -- it is ordinary: 861.7 us for the ten, 227.0 for the one, and
0.34 realized of the 0.635 saved is 53%.

And the second kernel of that phase, `rms_gated` (`RMSNormGated`, 60
dispatches into 30), **measures zero**: +0.012 (3 of 8) and -0.021 (5 of 8)
in two processes, against a zero-control band of -0.023 .. +0.054. It is kept
-- exact, at its floor, 30 fewer dispatches and barriers, never worse -- but
it did not pay. What the same run shows is that the 0.145 ms sitting on those
two dispatches (**4.8 us each, three times the chain's average**, because
they are the serial spine between the recurrence and `out_proj`) is not
reachable by gluing two elementwise ops together at all.

**Phase 34 then folded the norm into the matvec that reads it -- the
deferred op layer's own mechanism, on the one norm in this model with
exactly one consumer -- and it lost.** Four widths, two processes, each
paired against the unfused arm in the same draw: **+0.304/+0.376 at ggml's
512 threadgroups, +0.079/+0.146 at the best of the four, +0.188/+0.266 and
+0.468/+0.537 wider still**; 2 of 16 pairs down at the minimum, against a
0.145 ms prize. Bit-exact throughout (`max_abs == 0.0`, and 200 greedy
tokens byte-identical at every width, 1009 dispatches to 979).

The reason generalises and is the end of the question: **a mat-vec
parallelises over output rows and every threadgroup reads the whole input**,
so folding a producer into it is not computing it once, it is computing it
once per threadgroup. The producer is O(K) and gets multiplied by the
threadgroup count; the consumer does not. Fusing an elementwise op into its
consumer is a *gemm* idea and a decode step has no gemms. Every other norm
here feeds four or six matvecs, which multiplies the same redundancy again.
The kernel is kept off by default as the control that says so
(`ORNITH_FUSE_NORM_PROJ=1`, `ORNITH_RMSGATE_NR0`).

**And the fusion pays outside the decode step, which nothing had measured.**
A 6406-token prompt: **5.47 -> 5.17 GPU seconds, -5.7%, 3 of 3 interleaved
pairs** -- and not because of launches. At the 2048-token chunk the ten
kernels move ~250 MB a layer between them where the one moves ~67, since
eight of the ten exist only to write an intermediate the next one reads.
Every remaining elementwise op has that shape at prefill. The MTP window pays
too: `--mtp --mtp-depth 1 --frspec` goes 965 dispatches a token to 753, 561
barriers to 455, 3 of 3 pairs down on GPU busy.

**And `tools/epoch_slack.py` asks the reordering question of every dispatch
at once**, since the one reorder that paid was found by eye: which dispatches
are the slowest thing in their own epoch, how far they may legally move, and
what the cheapest reachable epoch would cost with them. It reproduces the
move that has an answer (0.225 ms priced against 0.252 measured) and says
what is left after it: **0.196 ms in total, nothing over 0.085**. Reordering
is spent; re-run it after anything that moves the epoch structure.

**It has now proposed the same losing move three times.** Phase 32 priced the
shared expert's gate one epoch later at 0.235 and measured +0.104; Phase 33
re-ran it on a 613-epoch trace, where the whole 0.256 ms it finds is that one
move again, and measured **+0.145, 0 of 8 pairs down**. The model ranks; it
does not predict. Positions 0 and 1 have also separated again (+0.046, 2 of
8) after Phase 32 had measured them level -- a null result expires with its
artifact exactly as a positive one does.

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

**And since Phase 37 there is an arithmetic ceiling beside it**, which
prefill needs and decode does not: `cargo run --release --bin arithmetic`,
no model load. Scalar fp32 FMA sustains **14.1 TFLOP/s** and a simdgroup
8x8x8 `multiply_accumulate` sustains **14.5** (float32 fragments), **14.9**
(float16), **15.1** (bfloat16). Three things follow and all three matter:

- **`simdgroup_multiply_accumulate` is not a separate unit on this machine.**
  Matrix is 1.03x scalar and sixteen-bit fragments buy 3%, not 2x. That is
  why `ORNITH_KQUANT_TILE_HALF` measured free -- the machine gives 3%, the
  synthetic says 1.03x and the real kernel says 1.036x.
- **Sweep the accumulator count or the number is a fifth of the truth.** One
  dependent FMA chain reads 2.6 TFLOP/s; thirty-two independent ones read
  14.1. The same kernel, and only one of those is the machine.
- **The ceiling is validated by a kernel nobody here wrote**: mlx's dense
  bf16 steel gemm reaches 94-96% of it at this model's own shapes. At one
  shape (M=16384, N=2048, K=2048) the whole stack reads: dense bf16
  **94%**, mlx's affine 4-bit gemm **84%**, ggml's tile with its unpack
  removed **76%**, ggml's tile as it ships **75%**. So unpacking costs
  anyone 10 points, **ggml's tiling costs 8 more**, and the k-quant unpack
  itself costs 1-3 -- predicted at 4.7% from an op count before it was
  measured. Four fifths of what first looked like "the price of the
  format" is the price of *that tile*.

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

### Fusing the router's tail, and why it is not here -- **and why it is, now**

Tried and reverted, so that it is not tried again for the same reasons. Then
tried again in Phase 31, where the same kernel is worth **5.5%**. Both
measurements are below, and the second one is the current state; read the
first for the reasons, which were good ones and were all about *this
artifact at that time*.

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

**Three things about that paragraph were wrong, and `REBUILD.md` Phase 31
took them apart one at a time.**

- **The artifact moved out from under it.** That was the affine artifact at
  10.9 ms a step and 2.83 GB a token; `q6k-full` reads 1.8 GB and runs at
  7.5, so the same small dispatch is a larger share of a step whose big
  kernels got shorter. A conclusion of the form "it hides under the weight
  reads" has a shelf life exactly as long as the weight reads.
- **It was priced on the wall clock**, 88.9 tok/s against 88.4, which is
  inside this machine's swing. The in-process pair the port uses now
  separates 0.3% cleanly.
- **And the sort could not be priced by removing it at all**, which is why
  no number in that paragraph is of the sort itself. It was measured in
  Phase 31 against a stand-in that writes a valid order cheaply:
  **0.385 ms a token, 5.5% of the step**, and a rewritten `router_top_k`
  takes all of it. The kernel is the same one, down to the tie rule; what
  changed is that the first version of it was 6x off its own budget by
  re-reading the row every round, which is a fine way to conclude that a
  kernel is not worth writing.

The lever is launches *and* bytes, and which one a given dispatch is on is a
question with an instrument attached -- see "What the dispatches cost" above.


## What is left

In order:

0. **Prefill, and the port is no longer measurably behind mlx on it.**
   Phase 35 built `step_inventory prefill` and the first thing it found --
   a sort on one thread -- took the gap from 22% to 12%. Phase 36 took the
   two kernels it pointed at next, `gather_rows_div` and `unsort_rows`, and
   on a 6621-token prompt the affine parent went **4.26 -> 3.80 GPU
   seconds, -10.8%**, against `mlx_lm`'s 3.98-4.30 s of *wall* on the same
   ids in the same sitting. Since mlx's own GPU time can only be below its
   wall, the honest reading is "inside the spread of mlx's draws", not
   "ahead".
   **But read Phase 36 before reusing its mechanism, because its headline
   is not its plan.** Both copies are gone where the kernels allow it, and
   that is worth **-0.6% of a chunk**. What was worth -4.6% of the prompt
   was noticing that both kernels moved 67 and 134 MB **two bytes a
   thread** -- 37 and 68 GB/s against this machine's 465 -- and giving them
   a 16-byte access. The first A/B of the removals read -8.3% and was
   measuring the control's access width; it had to be run again once the
   control was widened. **A control has to be the best version of what it
   replaces.**
   What is left of it: `gather_rows_div_wide` still runs 40 times a chunk
   on the affine parent, because mlx's sorted kernel `gather_qmm_rhs` takes
   no `lhs_indices` at all -- the k-quant tile does, which is why the front
   removal engages on one artifact and not the other
   (`Linear::gather_reads_lhs`). Handing mlx's kernel a permutation is ~200
   lines of Metal for the 1% the k-quant side measured, and is **priced and
   not built**. The memory-pressure idle sits on top of all of it and is
   item 1.
   **And Phase 37 then measured how much room is left at all, which is not
   much.** After Phase 36 a chunk is **89.4% quantized gemm**, 3.8% data
   movement, 6.7% everything else -- so what is left is an arithmetic
   question and nothing else. At one shape the gemm's distance from the
   ceiling decomposes: dense bf16 94%, mlx's own quantized gemm 84%,
   ggml's k-quant tile 75-76%. Of the ~19 points the k-quant gemm is down,
   **10 are what unpacking costs anyone, 8 are ggml's tiling against
   mlx's, and 1-3 are the k-quant unpack itself**. So the reachable
   headroom is **~9% of a chunk on a k-quant artifact and ~0 on the affine
   parent**, none of it reachable by scheduling, fusing or removing
   anything, and **both candidates the phase first named are now
   closed.** The routed gemm's 8 points are neither `N` nor routing nor
   the weight footprint -- each was moved on its own and none matters --
   they are the tile width, and the wider tile is **14% worse on the real
   router** (0 of 3 pairs), so they are the price of routing over 256
   unevenly loaded experts rather than headroom. And the unpack is **2-3%
   of the gemm**, not the 20% the dense comparison suggested: an op count
   predicted 4.7% and a stand-in with the arithmetic removed
   (`Q4KBareFmt`) measured 2.0-3.2%.
   **What is left is one thing, and only on a k-quant artifact.** At one
   shape the stack reads dense bf16 94%, mlx's affine 4-bit gemm 84%,
   ggml's tile 75-76%. **mlx's quantized gemm is 13% faster than ggml's
   tile while doing more work**, so ~8 points -- ~6% of a chunk -- are
   ggml's `kernel_mul_mm` tiling against steel's `BlockMMA`. Reaching it
   means carrying Q4_K and Q6_K into a steel-tiled gemm: a real kernel
   project, with a measured prize and a ready oracle (`max_abs == 0.0`
   against the tile it replaces). **On the affine parent there is nothing
   here** -- its gemms already run on mlx's kernel at 84%. It also
   explains why `q6k-full` prefills 18% slower than the parent while
   being 4 GB smaller: the k-quants buy memory and pay in prefill
   arithmetic, mostly for the tiling.
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
   - **Encode order, and it was not spent.** Interleaving the shared expert
     with the router took 122 of the 223 available epochs and 3.3% of GPU
     busy -- see "Interleaving the MoE block" above -- and the schedule model
     said ~3.4% was left, spread one epoch at a time, needing a deferred op
     layer to reach. **Phase 31 took another 3.3% by moving one dispatch.**
     The model was averaging over dispatches; the inventory says they are not
     interchangeable, and the shared expert's one-row gate was 12.6 us alone
     in an epoch with six epochs of slack in front of it. So the remaining
     question is not "how many epochs" but "which dispatch is the longest
     thing in its own epoch, and does anything nearby have room for it" --
     which `step_inventory` answers directly and the model cannot.
   - **The launch floor, which was half the step, and is now 38% of it.**
     Phase 31: the same dispatch stream with its work clamped away is 46% of
     a decode step, and what is left runs at the machine's ceiling.
     **Phase 32 measured the ceiling on taking it back**: the entire
     elementwise chain gone is 1.35 ms, 20% of the trunk. Two block-shaped
     kernels have since taken 0.55 of it -- the MoE block's tail, six
     dispatches into one (0.21), and the gated-delta layer's front end, ten
     into one (0.34) -- for **-500 dispatches, -299 barriers, -305 epochs,
     -4 command buffers and 117 tok/s to 130** between them.

     **The deferred op layer is closed on this step, and by its mechanism
     rather than by its prize.** The prize was 211 dispatches and 0.49 ms of
     ceiling, ~0.25 ms realized, 3.8% of a 6.5 ms trunk -- small. Phase 34
     then tried the mechanism itself on the one norm in the model with a
     single consumer and it **cost +0.08 to +0.15 ms against a 0.145 ms
     prize**, because a mat-vec parallelises over output rows and every
     threadgroup reads the whole input, so the producer gets computed once
     per threadgroup rather than once. Fusing an op into its consumer is a
     gemm idea; a decode step has no gemms.

     **Prefill is the shape where it might still hold**, and it is
     unmeasured: at `M >= 8` the k-quant tile takes over, a threadgroup owns
     a block of the input, and the repeat count falls from `O / (NSG * nr0)`
     to `O / BN` -- 32 rather than 512 for `out_proj`. Phase 33 already found
     the elementwise chain costing *bytes* there rather than launches (one
     block kernel, **-5.7% on a 6406-token prompt**). See item 0.

     Price a fusion at ~half of what removing its dispatches measures (twice
     measured), and a rewrite at ~half of the isolated time it removes (once)
     -- but only over **the kernel's own set**: Phase 33's bucket named 20
     dispatches that were not its and missed 120 that were, and the forecast
     came out 2.3x low.
   - **The bytes are spent, but the *rounding* was not, and it paid.**
     `REBUILD.md` Phase 7: GPTQ over the same 91 modules, same bits, same
     coverage, same kernels, **same 2.0761 GB a token and the same 1078-epoch
     dispatch profile count for count**. Median KL over the 58 live probes
     0.0285 -> **0.0187**, least damaged on 38 of 58, perplexity 6.5022 ->
     6.4238, and `full_forward_matches_mlx` still `max_abs == 0.0`. The
     artifact is `~/.mtplx/models/ornith-1.5-35b-a3b-gptq-head-linattn`.
     It costs nothing at runtime because nothing at runtime changed.

     **Phase 8 then measured the bf16 parent and the margin shrank.** Every
     number above is against `A`, which is quantized. Against the parent the
     sign test goes 38/58, p = 0.025 -> **31/57, p = 0.60**, the median
     per-probe ratio 1.59x -> **1.10x**, and GPTQ's p90 and max come out
     *worse* than the artifact it replaces. It is still ahead on the median
     and still the least damaged of the forged variants; it is no longer
     separable probe by probe. The mechanism is named: the Hessians were
     captured on `A`, so GPTQ reconstructs `A`'s layer outputs and was then
     scored on its distance from `A`.

     **Phase 9 tested that mechanism and refuted it, and found a worse
     problem.** The parent's Hessians differ from `A`'s by **0.58%** and the
     re-forge is no better (27 of 58, p = 0.69) -- but it is a *different
     artifact*: a 0.58% perturbation of `H` flips **9.9% of the codes**,
     because the compensation is sequential over 2048 columns. The median KL
     **between the two GPTQ variants** is 0.0196 against their 0.0348 from the
     parent -- **two draws of the same method are 56% as far apart as either
     is from the truth**, and the claimed GPTQ-over-RTN effect is only twice
     that spread. Both phases' group tables are noise; the `rare` regression
     was neither the corpus nor the reference. What survives is a real but
     smaller systematic gain, **~20% of median KL**, for identical bytes.
     Anything measured here from now on needs several draws.

     **Phases 10 and 11 searched the one knob; it does not pay, and the
     search is what finally measured the effect.** Phase 10 read
     `percdamp = 0.03` as better on two draws a point. On **four** draws a
     point (`ORNITH_GPTQ_SPLIT=4`, four disjoint corpus quarters in one
     streaming pass) 0.01 and 0.03 overlap completely -- Mann-Whitney
     p = 0.56 -- and 0.05 is worse than both. **Keep the 0.01 default.**

     What the sweep did establish, pooling **14 independent forges**:
     GPTQ beats round-to-nearest by **0.0094 nats, 23% of its damage**,
     14 of 14 below it, sign test **p = 6.1e-05** -- byte-identical, same
     kernels, same dispatch profile. Per-forge sd is **39% of that effect**
     and no damping value reduces it, so any single forge lands in a +-39%
     band. Four draws minimum for anything measured here from now on; that is
     one streaming pass plus four forges, ~25 minutes, scripted. The 256 experts of 40 layers -- 90%+ of
     the parameters -- were never touched at all.

   - **The bf16 parent is measurable on this machine, and was not.**
     `tools/dump_parent_probes.py` streams it one layer at a time: measured
     peak **1.73 GiB** against a 51.8 GiB working set, whole 64-probe capture
     in **24 seconds**. `mlx_lm.load` dies on 67.0 GiB, but that is residency,
     not the forward. `compare_probes.py` now takes any capture as the control,
     so all eight artifacts re-rank against the parent with no rebuild. The
     same trick sizes up: the 397B parent is 12.5 GiB a layer, which also
     fits.

   - **Then bytes -- and the byte lever is now spent, in both senses.**
     `REBUILD.md` Phase 4 is done. Five artifacts built from the verified
     parent and judged on perplexity, KL, bytes and GPU busy; the answer is
     `int4-head-linattn`, 24.2% of a step's traffic and 11.8% of its GPU busy
     for 0.048 nats. **Three ways past it were tried and all three failed.**

     | | GB/token | max KL, 8 | median KL, 64 | |
     |---|---:|---:|---:|---|
     | `int4-head-linattn` | 2.076 | **0.048** | **0.027** | the answer |
     | `nvfp4-head-linattn` | 1.925 | 0.429 | 0.065 | mlx's 4.50-bit mode |
     | fp8-scale twin | 1.925 | 0.443 | 0.034 | the best non-mlx 4.50-bit format |
     | `int4-except-gates` | 1.858 | 0.323 | 0.055 | more modules, not a cheaper format |

     The third column is Phase 5 and is the one to read: the second is a
     maximum over eight contexts, seven of which were asleep.

     mlx has exactly four modes: affine is the answer, nvfp4 was built and
     lost, mxfp4 and mxfp8 were priced out without building. The best format
     outside mlx at the same 4.50 bits was built as a stand-in (no kernel
     needed -- see `tools/forge_fp8scale_twin.py`) and lost by the same
     margin. The 4.50-bit space is closed.

   - **Two findings from that which outlast the formats, and both change how
     the next thing should be judged.**

     **Weight error predicts the distribution weakly, and the evidence that
     it does not at all was one context.** Phase 4 read the fp8-scale twin at
     +3% relative weight error paying 9.2x the KL and nvfp4 at +34% paying
     8.9x -- the same penalty for a tenfold difference in the predictor,
     which looked decisive. Over 64 probes those become **1.46x and 2.08x**,
     and the order is right: more weight error, more KL. What survives is the
     narrower claim, and it is still fatal to using the tool as a chooser --
     `int4-except-gates` has the *lower* weight error of the two (~8.2%
     against the twin's 8.3-9.0%) and the higher median KL, because a
     coverage change and a format change are not comparable in that currency
     at all. `tools/quant_error.py` and `tools/weight_error.py` explain a
     result -- the second correctly diagnosed nvfp4's subnormal group scales
     -- and rank formats only within a fixed coverage, by order and never by
     magnitude. **`IDEAS.md` is entirely denominated in that currency**, and
     carries the warning; its two headline ideas were both built today and
     both are dead (the two-stage exact `lm_head` prunes nothing: a 2-bit
     first pass leaves all 248320 rows alive, because Cauchy-Schwarz over
     2048 dimensions runs 49-68x loose and that looseness *is* the
     exactness).

     **Every Phase 4 verdict rested on one context of eight -- and Phase 5
     re-ran all of them on 64.** `REBUILD.md` Phase 5 is the record;
     `tools/probe_set.py`, `dump_probes.py` and `compare_probes.py` are the
     instrument, and it is a second set beside the frozen eight, never a
     replacement for them. The ordering held and is now significant instead
     of anecdotal -- `int4-head-linattn` is the least damaged on 28 of 60
     live probes and beats each rival by a sign test (44/60 over
     `int4-except-gates`, p = 0.0004). Three things changed:

     - "By median `int4-except-gates` ties `int4-head-linattn`" is **refuted**.
       That tie was eight samples; over sixty it loses at 2.0x the median KL.
     - The magnitudes were inflated ninefold by the single context. The twin
       pays **1.46x**, not 9.2x.
     - `nvfp4` is **worse** than was visible: max KL 0.429 -> **6.64 nats**,
       on Russian arithmetic it answers with a paragraph break.

     And a failure mode the old eight had no probe for: on a name stated once
     and asked for 350 tokens later, `int4-except-gates` and `nvfp4` both
     answer the *wrong name*, at 1.59 and 2.19 nats. Perplexity and the eight
     contexts are both blind to it.

     **What to probe, if the set is ever extended.** Damage lands in
     `arith` (0.089-0.288 median), `morph` and `longrange`, and barely at all
     in `format` (0.012-0.024, three of the four dead probes). The pattern:
     it lands where the answer is forced by something learned narrowly --
     non-English numerals, agreement, in-context recall -- and not where it
     is forced by syntax. Do **not** select on entropy: it correlates
     positively with KL, and a 2-nat gate would have dropped `Дважды два
     равно` itself. Spread across variants is the property, and it is only
     visible after capture.

   - **Activation-aware scaling was the next thing, and it is now done and
     rejected** -- `REBUILD.md` Phase 6, the last idea `IDEAS.md` had ranked
     above the noise. Two artifacts built, both byte-identical to
     `int4-head-linattn` at 2.0761 GB a token, so it was quality at zero cost
     or nothing. It is nothing: perplexity improves out of sample (6.5022 ->
     6.4512) while the 61 live probes move the other way (geometric-mean KL
     ratio 1.16x, p = 0.25), and the loss lands 3-7x on Russian arithmetic,
     the highest-signal group there is. The oracle half of the pitch held
     exactly as claimed -- no Metal written, `full_forward_matches_mlx` passes
     on the forged artifact at `max_abs == 0.0`.

     Two findings outlast it. **The fold point decides, not the spread**:
     `out_proj` has 1319x max/median saliency and is unreachable because its
     only fold point is shared across 32 heads, while `lm_head` is unreachable
     because `model.norm` also feeds the MTP head through an RMSNorm, which
     cannot absorb a per-channel rescale. And **`mean|x|` saliency is
     substantially language-independent** -- recalibrating on a 36% non-Latin
     held-out corpus kept the same alpha on 21 of 30 layers and moved 1.9% of
     channels, so no calibration set reaches what goes wrong.

   - **So both levers are spent, and what is left is the runtime.** Bytes
     closed in Phase 4, quality in Phase 6, and four predictors of the
     distribution have now been measured and found wanting -- weight error,
     perplexity, calibration output error, and activation magnitude. What
     remains on this list is items 1, 2 and 4 above: a machine that is not
     swapping, the ~3.4% of encode order that needs a deferred op layer, and
     the MTP stall. All three are wall-clock questions, which is why the first
     one gates the other two.

3. **Rebuild the artifact from its parent -- DONE**, Phases 0 through 4, and
   it is what unblocked the byte lever above. `REBUILD.md` is the record. The
   parent is `ornith-ai/Ornith-1.5-35B-A3B@10fbf86f`, it is kept on disk
   because deleting it would mean a 70 GB download per variant, and
   `body_mode` turned out to be exactly the one word in the recipe the plan
   said it was.

   Read Phase 0 before deleting anything. Every test here compares against MLX
   on the same artifact, which validates the port and not the artifact, so a
   bad rebuild passes the whole suite -- and `baseline/` holds the only
   material that would catch it.
4. **The MTP stall** -- and speculation is no longer waiting on it to be
   worth running. Re-measured on the rebuilt head (see the box at the top of
   "It does pay off"), `--mtp-depth 3 --frspec` already does **27-28% less
   GPU work a token** than autoregression and runs 16-20% faster on the wall.
   That is the largest single win in this document and it was sitting behind
   a stale measurement, not behind the stall. The stall is still real and
   still costs: idle ~2.0 ms a token against autoregression's 0.9, ~10% of
   the wall, and the eighth explanation still needs a machine that is not
   swapping. Fixing it is worth another ~13 tok/s on top, not the whole win.

5. **A multi-row k-quant mat-vec -- built in Phase 28, worth an eighth of
   what this item predicted.** Everything below is what was predicted; the
   measurement is `REBUILD.md` Phase 28 and it is `-2.8%` a token at depth 1,
   not `-21%`, because the re-read this item is about was never the binding
   cost. Kept as written because the *reasoning* is the thing to learn from:
   every clause of it is sound except the one that was never checked.
   `REBUILD.md` Phase 27: a verify forward costs **3.4 ms per row of the
   window** on `q6k-full`, against a draft level's 0.587, and 41% of that is
   the dense k-quant mat-vecs reading the same weight matrix once per row --
   ggml's kernel carries the token axis on the grid, so an M-row call is M
   independent mat-vecs. Neither existing kernel fixes it at a window's width:
   the tile reads the matrix once but its floor is 7.5x a mat-vec, so it only
   crosses over at 16 rows, and mlx's `qmv_wide` recovers 43% and still pays
   2.6x at four rows. What does is a mat-vec with an M axis *inside* it --
   read each weight block once, accumulate into M row accumulators, leave the
   `ib` loop, the lane split and the `simd_sum` exactly as they are, so every
   row keeps the arithmetic it would have had alone and the window mode's
   contract survives by construction. Priced by subtracting it from the
   measured cycles: `--mtp-depth 1 --frspec` goes from -12% to **-21%** of a
   token against autoregression and depth 3 from +13% to -8%. Taking the
   other two thirds as far as they go -- the expert gather charging for the
   ~23 distinct experts a four-token window names rather than its 32 rows,
   and one recurrence dispatch writing all `T` snapshots instead of one per
   token -- gives **-27% at depth 1 and -20% at depth 3**.

   **It does not move the knee, and that was the other half of the
   prediction.** Depth 2 measures 8.37 GPU ms a token today and 7.05 / 6.31
   under the two columns above, against depth 1's 7.53 / 6.73 / 6.28 -- the
   knee stays at 1 and only ties at 2 in the most optimistic case. The binding
   constraint there is acceptance, not the price of a row: 0.69 / 0.57 / 0.52
   makes the marginal levels worth 5.91, 3.37 and 1.75 ms against a marginal
   cost of 4.49 today and ~2.3 with everything taken. A cheaper window widens
   the win at the depth that already works; only a better head buys a deeper
   one.

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

Three more on the quantized side, each with its reason in
`kernels/fp_quantized_inst.metal`: the **mxfp4 and mxfp8** modes (priced and
rejected in `REBUILD.md` Phase 4 rather than measured on hardware -- 20-22%
and 2.94% weight error respectively), the **`_hgs` global-scale**
specialisations of the fp kernels (only `mx.quantize(global_scale=...)` and
`nn.quantize(quantize_input=True)` produce one, and `mlx_lm.convert` calls
neither, so `WeightStore::quant_matrix` rejects an artifact that carries one),
and **`_r_2`**, the narrow-output `fp_qmv_fast` mlx picks only at
`architecture_gen() == 17` on an 's' part in nvfp4. That last one is on the
M5 checklist above: it is the one nvfp4 kernel this machine cannot run and so
the one number the nvfp4 verdict does not cover.

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
- **One unit test is artifact-specific and fails on a k-quant model.**
  `weights::tests::loads_a_quantized_matrix_with_the_right_config` derives a
  packed width from `bits` the way mlx's affine layout does, so it fails
  under `ORNITH_MODEL=...-q6k-full` and passes on every affine artifact
  including the default. It is not a regression and nothing else in the
  suite depends on it; run the suite on the default model, and read a
  failure of that one name under a k-quant `ORNITH_MODEL` as expected.
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
- **`ORNITH_MODEL` points both sides at a variant**, `tools/ref.py` and
  `src/refs.rs` alike, which is how a non-default artifact gets checked. The
  references are anchored on one artifact at a time, so verifying the port on
  another one is: move the case out, regenerate it under `ORNITH_MODEL`, run
  the test under the same `ORNITH_MODEL`, move the original back. That is how
  the nvfp4 build was shown to run at `max_abs == 0.0` through all 40 layers
  without re-anchoring the whole tree onto a variant that was then rejected:

      mv refs/model /tmp/refs-model-control
      cd tools && ORNITH_MODEL=~/.mtplx/models/<variant> \
        PYTHONPATH=../../vendor/mlx-lm python3 dump_model.py
      cd .. && ORNITH_MODEL=~/.mtplx/models/<variant> \
        cargo test --release --test model -- --ignored --nocapture
      rm -rf refs/model && mv /tmp/refs-model-control refs/model

  Move rather than overwrite, for the reason Phase 3 gives: a writer that
  fails silently leaves a gap that way, instead of a test comparing the new
  artifact against the old one's tensors and passing.
- **Two measurements need no model load at all**, which makes them the cheap
  first look at a fresh build: `python3 tools/step_bytes.py <artifact>` reads
  the safetensors headers and reports what a decode step reads by family, and
  `PYTHONPATH=../vendor/mlx-lm python3 tools/weight_error.py` dequantizes a
  module out of each variant and compares it to the bf16 parent. The second
  is the only thing the parent can still be asked on this host -- `mx.load`
  is lazy, so one tensor is 1.0 GiB where the model is 67.
- `cargo run --release --bin bandwidth` measures what this machine sustains:
  a read-only kernel at several sizes, then `quantized_matmul` at M = 1 on real
  weights. Run it before believing any "% of peak" claim.
- **`cargo run --release --bin arithmetic` is its sibling for FLOPs**, and
  needs no model load, which makes it the one performance bin that can be
  run on a busy machine (it reproduced to 1% at load average 11, because
  `best_of` takes a minimum and nothing in it competes for DRAM). Scalar
  FMA and simdgroup 8x8x8, each swept over how many independent
  accumulators a thread carries, then the real gemms at a chunk's shapes
  against the result. **Sweep, always:** one dependent FMA chain reads 2.6
  TFLOP/s where thirty-two independent ones read 14.1, so an unswept
  "peak" here is wrong by 5x and looks perfectly reasonable.
- **`--bin draft_cost` and `--bin window_mv` are the speculative pair**, both
  `ORNITH_PROF=1` and both in GPU seconds. The first runs a draft level and a
  verify forward apart from each other, at every window width, and reports
  per-kernel growth under `ORNITH_COMMIT_EVERY=1`; the second asks the same
  question of one module, against the tile and against mlx's own choice.
  `window_mv` repeats one module sixteen times, so its weights stay resident
  and its ratios are the optimistic end of `draft_cost`'s -- when the two
  disagree, `draft_cost` is the one that is measuring a forward.
- `ORNITH_PROF=1` also attributes each barrier to the pair of dispatches that
  forced it, counts host round trips, and prints the epoch histogram and the
  pool's peak; `ORNITH_RECYCLE=1` lets the pool hand back a buffer the current
  barrier epoch has touched, as mlx does, which is worth 30 invented barriers
  a token and 0.8% of GPU busy; `ORNITH_TRACE=<path>` writes one step's
  dispatches with every binding, which `tools/epoch_schedule.py <path>`
  levelises into the fewest epochs the dependencies permit. The barrier and
  epoch counts do not care what else the machine is running -- they are the
  numbers to reach for when it is busy.
- **Every fusion and every reorder has a control, and they are how a change
  is ranked against the code it replaced**: `ORNITH_FUSE_ADD_RMS=0` issues
  the residual add and the norm that reads it separately,
  `ORNITH_FUSE_MOE_TAIL=0` issues the MoE block's tail as the six dispatches
  it was, `ORNITH_FUSE_GDN_FRONT=0` issues the gated-delta layer's front end
  as the ten it was, `ORNITH_FUSE_GDN_NORM=0` issues `RMSNormGated` as two,
  `ORNITH_MOE_GATE_POS=0|1|2` puts the shared expert's gate beside
  the router, beside the gathers (the default) or after them,
  **`ORNITH_FUSE_NORM_PROJ=1`** folds `RMSNormGated` into `out_proj` and is
  the one switch here that is off by default, because it lost -- it is the
  control that says why, with `ORNITH_RMSGATE_NR0=2|4|8|16` for the width
  sweep that found its floor; `ORNITH_PARALLEL_SORT=0` puts the one-thread
  expert sort back in place of the block-parallel one, and is the control
  for the only change so far that a decode step cannot see at all;
  `ORNITH_QMV_SHAPE=<nr0>x<nsg>` and
  `ORNITH_GQMV_SHAPE` set the k-quant mat-vec's rows a simdgroup and
  simdgroups a threadgroup, where ggml's `2x2` is the default **and is
  measured rather than inherited**: 4x2, 2x4, 4x4 and 8x4 are all level or
  worse, in two processes, on both mat-vecs (`REBUILD.md` Phase 34);
  `ORNITH_ROUTER_SORT=1` puts mlx's argsort back in place of the selection
  kernel, and `ORNITH_FAKE_SORT=1` swaps the argsort for a stand-in that
  keeps its contract and does not sort (**the output is wrong while it is
  on**; it exists because removal cannot price a dispatch whose output is an
  address); `ORNITH_FUSE_MOE_GATHER=0` puts `gather_rows_div` back in front
  of the expert gemm and `ORNITH_FUSE_MOE_UNSORT=0` puts `unsort_rows` back
  in front of the block's tail, and **`ORNITH_MOE_GATHER_WIDE=0` narrows
  both of those copies back to one element a thread** -- which is the
  control that has to be *off* for the other two to be ranked against
  anything but a kernel's access width (`REBUILD.md` Phase 36).
  `step_inventory` A/Bs all of them in one process, and it has two
  single-cell modes for the reason prefill's thermal drift gives -- a cell
  placed eleventh is measured in a state the first one was not:
  `step_inventory prefill only` runs the Phase 36 gather/unsort cell alone
  with eight interleaved pairs and stops, and `step_inventory prefill
  addrms` does the same for the residual-add fusion, which needed exactly
  that protocol after three readings of it straddled zero (it is **zero**;
  `REBUILD.md` Phase 35).
- **Two stand-ins produce wrong answers on purpose**, and both exist
  because a removal cannot price what they price. `ORNITH_FAKE_SORT=1` is
  the older one. `Q4KBareFmt` / `kquant::qmm_t_bare` is the newer: the
  k-quant tile with only `Q4KFmt`'s arithmetic removed, same loads, same
  registers, which is what separates "the unpack costs 2-3%" from "this
  tile costs 8 points". Nothing in the model dispatches it; it is reached
  from `bin/arithmetic` alone.
- **A short generation puts the prompt on the denominator.** `ORNITH_PROF`'s
  per-token figures divide by the tokens actually generated, not by `-m`, and
  the prefill call's dispatches and GPU time are in the numerator. A run that
  stopped after 9 tokens reported **1121 dispatches and 11.0 GPU ms a token**
  where the same build on a 44-token prompt for 200 tokens reports 1009 and
  7.2 -- `(1000 + 9 * 1009) / 9 = 1120`, and nothing had changed. Compare
  per-token figures only across runs that generated a comparable number of
  tokens.
- **Check that a control actually applied, and check it by a count.** Phase
  33 ran an MTP pair whose two arms were both fused -- `zsh` does not
  word-split an unquoted `$VAR`, so two environment variables reached the
  process as one and neither matched -- and the run reported "byte-identical"
  because it was comparing a build against itself. Nothing in the output said
  so except the dispatch count, which was the same 753 in both arms. Every
  control here changes a *counter*, so read the counter before reading the
  result; it is the same reflex `Context::prof_dropped` exists for.
- **And check that the output exists.** Phase 36 renamed `gather_u32`,
  which had a second call site in `src/frspec.rs`; the whole suite passed,
  because nothing covers frspec, and the `--mtp --frspec` byte-identity
  check then reported both arms identical. Both had panicked and written
  **zero bytes**, and `md5` of nothing equals `md5` of nothing. Print the
  byte count beside the hash -- `md5 -q f` alone cannot tell "the same" from
  "neither ran". The same applies to any diff-based check whose inputs a
  crash can empty.
- **Grep a kernel's name before renaming it.** The metallib is looked up by
  string at dispatch time, so a missed call site is not a compile error; it
  is a panic the first time that path runs, which may be a path no test
  takes.
- **A failed command buffer panics rather than timing as zero.** An
  `MTLCommandBuffer` that fails reports `GPUStartTime == GPUEndTime == 0`,
  so by the only clock here it costs **0 ms of GPU** and the arrays it owned
  hold whatever was in the pool. Nothing checked `status()` until Phase 35,
  when a prefill run under a GPU reset printed a whole boundary table of
  zeros with correct dispatch counts beside them. `Context::check_completed`
  now panics with the driver's own message, from `sync` and from the
  profiling harvest. If it fires, nothing measured after the failure means
  anything -- the usual cause here is another process faulting the GPU
  ("Discarded (victim of GPU error/recovery)").
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
- The vision tower's `patch_embed.proj.weight` must be in mlx's Conv3d order,
  `(out, kD, kH, kW, in)`. A forge build on 2.11.3 leaves it in the parent's
  torch order and `VisionTower::load` asserts; `tools/align_rebuild.py`
  transposes it, which reproduces the published tower byte for byte. MTPLX
  transposes at load and never notices, so its own dumps succeed either way --
  the failure surfaces in `dump_vision_ops.py` and in the port, both of which
  read the raw tensor.
- After a `mtplx forge build`, run **`tools/align_rebuild.py <artifact>`**
  before anything else: MTPLX 2.11.3 does not write the artifact the way 2.9.1
  did, and two of the differences (the re-serialised tokenizer, a missing
  `num_nextn_predict_layers`) change what this port reads. Then
  `tools/check_rebuild.py <artifact>` for the invariants, and
  `tools/compare_baselines.py` for the one comparison that survived the old
  artifact's deletion -- `baseline/logprobs.bin` against a fresh capture from
  `dump_artifact_baseline.py`, where identical weights must give `max_abs ==
  0.0` exactly as they must inside the port. `REBUILD.md` Phase 2 has the
  whole story.
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
- **A quantity obtained by differencing two configurations measures every
  difference between them, not the one in its name.** `REBUILD.md` Phase 26
  read "a draft level costs 4.31 ms" out of `(cycle(3) - cycle(1)) / 2` and
  spent the next paragraph hunting for where 3.9 ms of it went. Depth 3 and
  depth 1 differ by two draft levels *and* by two verify rows, and the rows
  cost 5.8x the levels: a draft level is 0.587 ms. What settled it was
  running each half on its own in one process -- `src/bin/draft_cost.rs`,
  four command-line minutes -- which is always available and was never done
  because the subtraction looked like a measurement.
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
