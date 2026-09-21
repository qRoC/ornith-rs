# ornith-rs

A from-scratch Rust + Metal implementation of the MLX inference path for
**Ornith 1.5 35B-A3B** (Qwen3.5-MoE: 30 gated-delta + 10 full-attention
layers, 256 experts, a native MTP head, a Qwen3-VL vision tower), verified
against MLX tensor by tensor — and a k-quant artifact forged from the same
parent.

It started from
[`Shiftedx/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx`](https://huggingface.co/Shiftedx/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx),
the published MLX artifact, run under MTPLX. This document says what came out
of that and, in numbers, where it is better and where it is not.

---

## The short version

**The port is the product; the artifact is an option.**

- **Stack against stack, one quiet machine, one sitting: 101.0 tok/s → 133.7,
  and 154.1 with speculation** — that is `mlx_lm.generate` on the published
  artifact against this port on ours, **+32%** and **+53%**.
- Taken apart, that is three separate things and only two of them are ours:
  on the **same published artifact** this runtime is **5.5% faster at decode**
  than `mlx_lm.generate` and **level at prefill**; our artifact adds another
  **25%**; speculation at the measured depth adds the rest.
- Its **speculative decoding is exact**: `--mtp --greedy` emits *byte-identical*
  text to `--greedy`. That is a correctness property, not a speed one, and it
  is worth **+7 points of acceptance** because it stops charging the draft for
  the target changing its mind.
- Its **defaults are measured rather than inherited**: the depth knee here is
  **1**, not the recommended 3, which is worth up to **+26%** of wall.
- Our own artifact (`q6k-full`, k-quant) is **15% smaller and 25% faster at
  decode** than the published one — and **measurably less accurate**. That
  trade is stated in full below, not buried.
- If you want the published artifact's accuracy, **run the published artifact
  on this port**. Nothing about its quality changes and you keep the speed and
  the exactness.

**What is not measured: MTPLX's own runtime.** Every speed comparison here is
against `mlx_lm.generate` on identical ids in the same sitting. MTPLX was read
closely and ported from, never benchmarked. Treat "faster than MTPLX" as
unproven; "faster than mlx-lm" is measured.

---

## Accuracy, and the published artifact wins it

64 fixed probe contexts, KL(bf16 parent ‖ variant) in nats, both captures
taken by this port with `ORNITH_INVARIANT=all` so the prefill kernel is not a
free parameter of the instrument, 58 live probes:

| | median | p90 | max | lower on |
|---|---:|---:|---:|---:|
| `attention8-bf16recurrence-vision-mtplx` (published) | **0.0158** | **0.1305** | **0.5730** | **42 of 58** |
| `q6k-full` (ours) | 0.0332 | 0.1898 | 5.2520 | 16 of 58 |

Median ratio **1.82x in the published artifact's favour**, and the largest
single gap is **+5.12 nats on one probe** (`Семью восемь равно`) — a known
cliff in this model where the top-1 flips from continuing the arithmetic to a
paragraph break, and which every low-bit format in this rebuild has fallen
off.

Perplexity over a fixed 6621-token corpus, both pinned:

| | mean NLL | perplexity |
|---|---:|---:|
| published | 1.847123 | **6.3415** |
| `q6k-full` (ours) | 1.857753 | 6.4093 |

**+1.07%.** So: on the corpus the two are within a percent; on the probe tail
they are not, and the published artifact is the more faithful one. It should
be — it spends 4-bit affine at group 32 on the expert body with 8-bit
attention, routing, embeddings and head, against our 187 Q4_K + 128 Q6_K +
41 eight-bit affine modules.

---

## Performance, and ours wins that

Quiet machine (load 1.0-2.4, 0.5 GB of swap), 200 greedy tokens from a
51-token chat prompt, best of three interleaved draws, **autoregression run
twice in every cell as a control against itself** (+0.2% and +1.1%):

| | published | `q6k-full` (ours) |
|---|---:|---:|
| decode, wall | 106.8 tok/s | **133.7 tok/s** (+25%) |
| decode, GPU busy | 8.78 ms/token | **6.97 ms/token** (−21%) |
| decode with `--mtp-depth 1 --frspec` | 134.2 tok/s | **154.1 tok/s** |
| prefill, 6621 tokens | **3.59 s** | 4.22 s (ours 17% **slower**) |
| on disk | 27 GB | **23 GB** |

The prefill row is not a rounding error and is not hidden: ggml's k-quant
tiling is slower than mlx's affine gemm at prefill width, and that is the
price of the format. If your workload is long prompts and short answers, the
published artifact is the better artifact.

### Against `mlx_lm.generate`, same sitting

Two comparisons, and they answer different questions. The first isolates the
*runtime* from the *artifact*; the second is the one a user actually chooses
between:

| on the published artifact | mlx-lm | ornith-rs |
|---|---:|---:|
| decode | 100.9 - 101.2 tok/s | **106.4 - 106.8** (+5.5%) |
| decode, `--mtp-depth 1 --frspec` | — | **133.8 - 134.2** (+32%) |
| prefill, 6621 tokens | 1837 - 1843 tok/s | 1842 - 1844 (level) |

| whole stack, decode | tok/s | against mlx-lm |
|---|---:|---:|
| `mlx_lm.generate` + published artifact | 100.9 - 101.2 | — |
| ornith-rs + published artifact | 106.4 - 106.8 | +5.5% |
| ornith-rs + `q6k-full` | **131.6 - 133.7** | **+32%** |
| ornith-rs + `q6k-full`, `--mtp-depth 1 --frspec` | **149.5 - 154.1** | **+53%** |

mlx-lm has no row on `q6k-full` because it cannot load a k-quant at all.

**One arithmetic trap, since this project fell into the general case of it
twice.** `HANDOFF.md` records this port at 82-92 tok/s and mlx-lm at 86 in an
older sitting, and it is tempting to read "86 → 133" as the gain. It is not:
that sitting ran on a machine carrying 18.5 GB of swap, and **mlx-lm alone
goes 86 → 101 when the swap is gone**, on unchanged code. Only numbers from
the same sitting may be subtracted, which is why every row above comes from
one quiet window with a control against itself. `REBUILD.md` Phase 42.

---

## What you get that the artifact plus MTPLX does not give you

**1. Speculation that cannot change the output.** MTPLX accepts a draft when
it equals the target's argmax — and a verify window row and the autoregressive
step it stands in for were *not the same arithmetic*, because a quantized
matmul picks its kernel by row count. 2.9% of verify rows came back with a
different argmax, and drafts that matched the autoregressive answer were
rejected for it. `ORNITH_INVARIANT` makes every row of a window compute what
it would have computed alone: acceptance **42.7% → 50.0%** at depth 3 at no
GPU cost, and `--mtp --greedy` becomes byte-identical to `--greedy`. Verified
end to end: 12 cells of 12 bit-identical at every window width a depth-3
verify produces.

**2. Defaults that were measured here.** `mtplx_runtime.json` recommends
depth 3. On four texts on a quiet machine the knee is **depth 1** (it wins
three of four, +15% to +26%), and **depth 3 is the worst speculative arm in
all four** — on one of them slower than not speculating at all. The port keeps
the artifact's contract as its default and prints the measurement beside it in
`--help`.

**3. FR-Spec priced as what it is.** A pruned draft head is not a flat win: its
65536-row sub-vocabulary is ranked on a *code* corpus, and acceptance under it
is the full head's acceptance times the fraction of emitted tokens the list
covers. English prose 100% coverage, code 99%, **Russian prose 65%** — where
`--frspec` costs 11-19% instead of paying 4-7%. `tools/merge_frspec.py`
extends the list (a superset, so nothing can regress) and takes Russian
coverage to 100% by adding the 18075 Cyrillic ids the code corpus never had.

**4. Artifacts MLX cannot load at all.** A k-quant has no MLX quantization
mode, so `mlx_lm.load` stops exactly where the interesting formats start. The
port runs them, and carries its own instruments for them: probe captures and
perplexity, both bit-exact against MLX on an artifact where both can run.

**5. A model you can read.** The goal here was understanding, not shipping a
runtime. Every kernel, the tokenizer, the chat template, safetensors, JSON and
the BPE are hand-ported, with `mlx` and `MTPLX` vendored beside them as the
oracle rather than as a dependency.

---

## What is verified, and how

The rule is **bit-exactness, never a tolerance**: golden tensors are dumped
from MLX on the same weights and compared with `max_abs == 0.0`.

- **128 tests**, 0 failed, 0 ignored (`--include-ignored --test-threads=1`).
- **58 of 64 probes bit-identical to MLX** end to end; the six that are not
  are a single float32 ULP at a bfloat16 tie, and the floor is documented.
- **Byte-identity** on every release check, with the byte count printed beside
  every hash — because `md5` of nothing equals `md5` of nothing: 200 greedy
  tokens and `--mtp --mtp-depth 1 --frspec` on both artifacts, plus a
  6621-token prompt whose hashes have now reproduced across three phases and
  several days.
- **Perplexity** from the port agrees with mlx's per-token NLL on **6620 of
  6620 values at `max_abs` 0** where both can run.

`REBUILD.md` is the phase-by-phase record, including every measurement that
was taken and then overturned; `HANDOFF.md` is the current state.

---

## Choosing

| you want | use |
|---|---|
| the best quality this family has | the **published artifact**, on this port |
| the fastest decode, and 1% of perplexity is acceptable | **`q6k-full`**, on this port |
| long prompts, short answers | the **published artifact** (prefill is 17% faster) |
| the smallest footprint | **`q6k-full`** (23 GB) |
| an MLX-native, supported runtime | **MTPLX** — this is a study, not a product |

That last row is not modesty. This project has no server, no batching, no
scheduler and no support; MTPLX has all four. What it has instead is an
implementation you can verify line by line against MLX, and numbers that come
with the control that produced them.

---

## Running it

```bash
ORNITH_MODEL=~/.mtplx/models/<artifact> cargo run --release -- --greedy \
  --prompt "..." --max-tokens 200

# speculative decoding, at the depth this machine measures rather than the
# one the artifact recommends
ORNITH_MODEL=... cargo run --release -- --greedy --mtp-depth 1 --frspec -p "..."

# an image: there is no decoder here, macOS ships one
sips -s format bmp photo.jpg --out photo.bmp
ORNITH_MODEL=... cargo run --release -- --image photo.bmp -p "what is this?"
```

`ORNITH_PROF=1` prints where a decode step's time went, down to the barrier
that forced each gap. `ORNITH_INVARIANT=0` returns to what MLX dispatches,
which is what every mlx-equivalence test asks for.
