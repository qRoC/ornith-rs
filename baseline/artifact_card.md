---
library_name: mlx
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
  - mlx
  - mtplx
  - vision
  - mtp
---

# Ornith 1.5 35B-A3B — MLX Vision + Native MTP

Hybrid MLX quant of `ornith-ai/Ornith-1.5-35B-A3B`, pinned to revision `10fbf86fed7ecee4a061f8b499a618f46001cac1`.

## Artifact

- 4-bit affine expert body with 8-bit attention, routing, embeddings, LM head, and late-layer projections
- 60 recurrent tensors retained in BF16
- 333 native BF16 vision tensors
- 785 native BF16 MTP tensors in `mtp/weights.safetensors`
- Recommended runtime: MTPLX 2.9.1, MLX 0.32.1, MLX-LM 0.31.3
- Recommended speculative depth: D3

## Run

```bash
mtplx serve \
  --model Shiftedx/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx \
  --profile sustained \
  --depth 3 \
  --mtp \
  --reasoning off
```

## Local qualification

Tested on an M4 Max with 64 GB unified memory.

| Gate | Result |
|---|---:|
| Executable Python tasks | 8/8 |
| Text exact-output smoke | pass |
| Vision smoke | pass |
| MTP tensor gate | 785/785 exact |

The final three-prompt, 512-token MTPLX sweep selected D3 at 109.0 decode tok/s. D1, D2, and D3 all passed the quality gate. Fans remained on automatic, so treat the speed as a host-specific measurement rather than a universal benchmark.

## Notes

Vision requires a compatible MLX-VLM or MTPLX runtime. Review generated code and tool calls before execution. The parent model and this derivative use the upstream MIT license; see the parent repository for original training and architecture details.
