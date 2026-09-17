//! A from-scratch port of the MLX inference path for
//! `ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx`
//! (a Qwen3.5-MoE hybrid: gated-delta linear attention + full attention,
//! 256-expert MoE, native MTP head, Qwen3-VL vision tower).
//!
//! Layout mirrors the sources it was ported from:
//!   metal/       <- mlx/backend/metal/device.cpp
//!   kernels/     <- mlx/backend/metal/kernels/*   (verbatim, under kernels/mlx)
//!   ops/         <- mlx/backend/metal/*.cpp       (kernel dispatch decisions)
//!   model/       <- mlx_lm/models/qwen3_5*.py, MTPLX's mtp patches
//!   tokenizer/   <- tokenizers' BPE + the model's chat_template.jinja

pub mod array;
pub mod chat;
pub mod generate;
pub mod json;
pub mod metal;
pub mod model;
pub mod mtp_generate;
pub mod ops;
pub mod refs;
pub mod sample;
pub mod safetensors;
pub mod tokenizer;
pub mod vision;
pub mod weights;
