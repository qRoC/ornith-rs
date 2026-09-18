# A k-quant kernel for this port: the design, and the one decision that changes it

`REBUILD.md` Phase 14 measured a Q4_K-shaped grid at **the same median KL as
the 5.00-bit base for 7.3% fewer bytes** -- the first 4.50-bit format here that
does not lose. It was measured as a *stand-in*: on disk it is still 5.00-bit
affine, and the 4.50 is arithmetic. Realising it needs a kernel.

This file is what that kernel should be, decided before any of it is written.

---

## The decision: build **real Q4_K**, not the twin's variant

The twin carries two constraints that exist only so it could be measured
without a kernel:

- **power-of-two super-scales**, because `2**k * s_i` with a 6-bit `s_i` is
  exactly representable in bfloat16 and therefore writes into the ordinary
  affine layout;
- **magnitude-plus-sign 6-bit fields**, because `aq.quantize`'s scale carries a
  sign.

Neither is a property of the format. Both were scaffolding. **With a kernel,
drop them** -- and doing so buys something the twin cannot have:

> **llama.cpp becomes the oracle.**

That is the whole argument for this file. mlx has four quantization modes and
none of them is a k-quant, so a Q4_K kernel written here has **no MLX
reference**, and `max_abs == 0.0` against MLX -- the check that has caught every
bug in this port -- simply does not exist for it. Writing the mode into
vendored mlx as well does not fix that; it makes the port agree with code from
the same hand.

**And it is not going to exist.** ml-explore/mlx#1934, "Q4_K_M support?",
closed by the lead maintainer an hour after it was opened, 2025-03-06:

> "We don't quite have exactly Q4_K_M and **aren't planning to add exactly
> that**. But we do support mixed quants with a similar strategy as Q4_K_M
> (e.g. using higher precision for the same weights). You can try e.g. the
> mixed 3_6 ... Another option is to use 4-bit with a smaller group size
> `--q-group-size 32` or to use 6-bit."

**Read that precisely, because it answers a different question.** Q4_K_M is a
*recipe* -- which bit width each tensor role gets. Q4_K is a *grid* -- integer
sub-block scales against a super-scale. The alternatives offered are all the
former: `mixed_quant_predicate_builder` in `mlx_lm/convert.py` sets
`mode = "affine"` and assigns low/high bits per module by layer position and by
`down_proj`. That is the coverage axis, which Phases 4 and 5 mapped here and put
a knee on, and it is **orthogonal to what Phase 14 measured** -- the twin used
uniform 4 bits at the base's own coverage, so every point of its gain came from
the grid and none from bit allocation.

So the position is settled and the grid question was never put. Treat "mlx will
not have a k-quant grid" as sourced rather than assumed.

`ggml` is a mature, independent, widely-deployed implementation of exactly this
format. If the port's kernel reproduces `dequantize_row_q4_K` **bit for bit** on
bytes produced by `llama-quantize`, the contract is intact in the form that
matters: two implementations that did not learn it from each other agree
exactly. The twin's variant forfeits that, because ggml cannot read it.

**MEASURED 2026-09-18, and it beats the prediction.** `vendor/ggml` built,
`quantize_q4_K` called through ctypes on the parent's own weights, read back
with `tools/q4k.py`:

    module         real ggml Q4_K   affine g32 bf16   the Phase 14 twin
                       4.50 bits        5.00 bits        (stand-in)
    in_proj_qkv          7.144%           8.111%           7.86%
    out_proj             7.547%           8.523%           8.30%
    lm_head              7.121%           8.087%           7.86%

**~12% better relative weight error than the 5.00-bit body, at half a bit
less**, and 0.7 points better than the stand-in. Two causes: fp16 super-scales
are finer than powers of two, and `quantize_q4_K` runs a search rather than
plain min/max -- worth 0.77 points on synthetic draws on its own.

The comparison that stings: **IDEAS #1**, the best 5.00-bit scheme in the table
and priced out in Phase 13 for having no oracle, measures 7.01-7.24%. Real
Q4_K very nearly matches it **at 4.50 bits**, and has one.

So Phase 14's `0.0396` median KL is a bound on the damage, not an estimate --
and a loose one. The real number needs the kernel, because the real format
cannot be stood in: `d * sc` is fp16 times a 6-bit integer, which needs 17
significand bits and bfloat16 has 8.

### Step 1 is done, and the oracle works

`tools/q4k.py` ports the format from `vendor/ggml/src/ggml-quants.c`. Against a
C harness calling ggml's own `dequantize_row_q4_K` on ggml's own blocks:

    DEQUANT vs ggml   max_abs = 0   exact = True

**That is the check a kernel needs** -- a kernel reads blocks, it does not write
them -- and it covers all three traps at once: the 12-byte scale packing, the
nibble interleave, and the subtraction. `q4k.quantize` reproduces none of
ggml's blocks (0 of 64) and is 0.77 points worse, which is the expected gap
between plain min/max and ggml's search. **So do not write a quantizer**: call
`quantize_q4_K` through ctypes, which is what produced the table above.

### The imatrix argument, which simplifies the combination

`quantize_q4_K(src, dst, nrows, n_per_row, imatrix)` takes an importance vector
natively. **`diag(H)` is exactly that**, and `refs/gptq-parent/hessians/` already
holds one for all 61 input groups. That is a far cheaper combination than the
GPTQ-plus-Q4_K problem described below -- no super-block coordination, no
changes to `gptq.py`, one extra argument. It is weighting rather than error
compensation, so expect less than GPTQ's 27%; measure it before assuming it
composes.

## The format, as ggml defines it

```c
#define QK_K 256
typedef struct {
    ggml_half d;         // super-scale for the sub-block scales
    ggml_half dmin;      // super-scale for the sub-block minima
    uint8_t   scales[12];// 8 x 6-bit scale + 8 x 6-bit min, packed
    uint8_t   qs[128];   // 256 x 4-bit codes
} block_q4_K;            // 144 bytes / 256 weights = 4.50 bits
```

Reconstruction, per sub-block of 32: `w = d * sc_i * q - dmin * m_i`, with
`sc_i`, `m_i` unsigned 6-bit.

Note the shape difference from affine: `bias = -dmin * m_i` is **subtracted**,
and both fields are unsigned. A port of the affine dequantizer that keeps its
sign conventions will be wrong in a way that still produces plausible text.

## Storage in a safetensors artifact

Keep three tensors per module rather than one packed blob, so the loader keeps
the shape it already has:

    <mod>.weight   uint32, [out, in * 4 / 32]      the 4-bit codes, as today
    <mod>.scales   uint8,  [out, in / 256, 12]     the packed 6-bit fields
    <mod>.dm       float16,[out, in / 256, 2]      d and dmin

`config.json`'s quantization block gets `"mode": "q4k"` with `group_size: 256`.
`check_rebuild.py` counts a module as quantized when it has more than one
tensor, so its 452/281 split survives -- the same way it survived nvfp4's
two-tensor modules.

## What to write, in order

1. **`tools/q4k.py`** -- **DONE.** The format in numpy, ported from
   `ggml-quants.c`. Its reader is exact against ggml (`max_abs == 0`); its
   writer is not and should not be used -- `tools/q4k_ggml.py` calls ggml's.
2. **The oracle harness -- built, and simpler than planned.** `vendor/ggml` is
   vendored; `tools/q4k_oracle.c` carries its own cmake line. `llama-quantize`
   turned out to be unnecessary: `quantize_q4_K` is exported from
   `libggml-base` and `tools/q4k_ggml.py` calls it through ctypes, so reference
   bytes come from three lines of Python. `tests/q4k.rs` still has to compare
   the Metal kernel against those bytes at `max_abs == 0.0`; that part is not
   written.
3. **`kernels/q4k.metal` -- ported, not invented.** ggml already ships
   `kernel_mul_mv_q4_K_f32_impl` for Metal, MIT, in production for years. Start
   from that rather than from the format description; the same relationship the
   port already has with mlx's kernels. Start with the **M = 1 path only** --
   `affine_qmv_fast` is 38% of a step, and that one kernel is enough to measure
   GB/token and GPU busy for real. Add `q4k_dequantize` for the tests.
4. **`q4k_gather_qmv_fast`** for the routed experts, if the first cut pays.
5. **Prefill** (`q4k_qmm_t`) last, and only if the whole thing is kept.

Scope the first cut to the **91 modules** `int4-head-linattn` moves to 4 bits.
They are 57% of a decode step's traffic, they are what every number since
Phase 4 is measured on, and leaving the experts at affine means one new kernel
rather than two.

## A second reference, and it is tuned for this hardware

`vendor/accelerated-qwen-3.8` (Apache-2.0, upstream MIT in `licenses/`) is a
pinned llama.cpp patch that makes **Q4_K decode projections faster on Apple
Silicon**. `patches/metal-combination.patch` adds
`kernel_mul_mv_q4_K_f32_qwen_accel_{rows1,rows4,vector2}` -- direct derivatives
of ggml's own kernel, same `kmask1/2/3` and the same `ix/it/iq/ir`
decomposition, **specialised**: shape parameters moved into function constants
and narrow row counts given their own instantiations. That is exactly the
pattern this port already uses (`affine_qmv_fast` against `qmv_wide`,
`instantiate_quantized_batched`), so the tuning transfers conceptually even
though none of the code does.

It also pins the exact upstream to vendor: **llama.cpp b10434**, with a
published sha256 in `upstream.lock.json`. Take that revision for step 2.

**Do not read its headline as evidence for this phase.** Its 11.0% / 10.2%
decode gain on an M1 Pro is *its Q4_K kernel against ggml's Q4_K kernel* -- a
kernel-tuning result at a fixed format. Phase 14's question is the format
against affine at fewer bytes, and it is already answered. The two numbers
multiply rather than compete, and neither substitutes for the other.

## Q6_K for the 8-bit side, which this repo also priced without meaning to

The patch carries `q6_K` variants alongside `q4_K`, which is a reminder that
the other end of the artifact is unpriced. The shipping artifact's traffic:

    4-bit    1.5132 GB   72.9%
    8-bit    0.5527 GB   26.6%     <- 169 modules at affine int8 g64 = 8.50 bits
    bf16     0.0102 GB    0.5%

**Q6_K is 6.5625 bits.** Moving those 169 modules off 8.50 would cut them by
22.8%, which is **6.1% of the whole decode step** -- on top of Q4_K's 7.3%.
Phase 4 priced `mxfp8` for this slot and rejected it (power-of-two scale, no
zero point, 0.54% -> 2.94% error); Q6_K has neither defect. It has never been
measured here, and `weight_error.py --schemes` can price it in minutes, the way
Phases 13 and 14 priced their candidates. **Do that before writing the second
kernel, not after.**

## What this does not need

- **No mlx change.** The artifact carries a mode mlx does not know, so
  `mlx_lm` cannot load it -- which costs the *MLX* oracle and nothing else. The
  golden tests for every other op are untouched, and `dump_probes.py` will need
  a port-side equivalent to capture distributions. That is the real hidden cost
  of this phase and it should be priced before step 3, not after.
- **No change to `gptq.py`'s grid logic** for a first cut. GPTQ and the format
  are orthogonal; combine them only once the kernel is measured alone.

## The prize, and the honest version of it

    int4-head-linattn + GPTQ    2.076 GB/token   median KL 0.0292   <-- ships today
    Q4_K twin                   1.925 (arith)    median KL 0.0396
    **Q4_K + GPTQ, unbuilt**    **1.925 real**   **~0.029 expected**

GPTQ is +27.1% of quality at fixed bytes; Q4_K is -7.3% of bytes at fixed
quality; they touch different things -- one picks the grid, the other picks the
point on it. Together: **-29.7% of a decode step against the control at the
quality GPTQ already delivers.**

The combination has one structural snag, and it is worth knowing before
starting: a super-scale needs all eight of a super-block's sub-block scales at
once, while `gptq.py` derives a group's scale from the weights it is about to
quantize, one group at a time. Either derive the super-scale from the
pre-compensation weights (an approximation) or make the walk's block the whole
256-column super-block (exact, but reintroduces the staleness that Phase 7's
"the block is the group" decision removed on purpose).

## How this goes wrong

- **The sign convention.** `w = d*sc*q - dmin*m` subtracts. Affine adds. Both
  produce fluent text when wrong.
- **The 6-bit packing.** ggml's 12-byte layout is not the obvious one; six of
  the bytes carry the low bits of all sixteen fields. Port it from the source,
  do not re-derive it.
- **Believing a number that is too bad.** Phase 14's first reading was
  18.8-25.4% and looked like a verdict on the format; it was `floor` where
  `ceil` belonged. A format the entire llama.cpp ecosystem ships cannot measure
  three times worse than plain affine.
