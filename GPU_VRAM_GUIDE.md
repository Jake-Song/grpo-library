# GPU VRAM → Model Selection Guide

Quick reference for choosing which model to train based on your available GPU VRAM.
All estimates assume **LoRA fine-tuning** (not full fine-tune) with gradient checkpointing enabled.

---

## 1. VRAM Requirements by Model Size

| Model Params | Data Type | Approx. Model Weights VRAM | Min. VRAM to Train (LoRA) | Comfortable VRAM* |
|:------------:|:---------:|:--------------------------:|:-------------------------:|:-----------------:|
| 0.5–0.6B | bf16 | ~1.2 GB | ~3 GB | 6 GB |
| 0.5–0.6B | 4-bit | ~0.35 GB | ~2 GB | 4 GB |
| 1–1.5B | bf16 | ~2.5 GB | ~5 GB | 8 GB |
| 1–1.5B | 4-bit | ~0.7 GB | ~3 GB | 6 GB |
| 3B | bf16 | ~6 GB | ~10 GB | 16 GB |
| 3B | 4-bit | ~1.7 GB | ~5 GB | 10 GB |
| 7–8B | bf16 | ~15 GB | ~22 GB | 40 GB |
| 7–8B | 4-bit | ~4.5 GB | ~10 GB | 24 GB |
| 7–8B | FP8 | ~8 GB | ~16 GB | 24 GB |
| 13–14B | bf16 | ~28 GB | ~40 GB | 80 GB |
| 13–14B | 4-bit | ~8.5 GB | ~16 GB | 24 GB |
| 13–14B | FP8 | ~15 GB | ~24 GB | 48 GB |
| 32B | bf16 | ~64 GB | — (needs multi-GPU) | — |
| 32B | 4-bit | ~20 GB | ~30 GB | 48 GB |
| 70–72B | 4-bit | ~40 GB | ~55 GB | 80 GB |

> \* *"Comfortable VRAM" = enough room for model weights + activations + optimizer state + a reasonable batch size (4–8) + vLLM KV cache (for GRPO).*

---

## 2. What Fits on Common GPUs

### NVIDIA T4 — 16 GB VRAM (Colab Free, AWS g4dn)

| Mode | Max Model | Settings |
|:----:|:---------:|:---------|
| LoRA (4-bit) | **≤ 8B** | r=16, max_seq=2048, batch=4, fp16 |
| LoRA (4-bit) | **≤ 3B** (comfortable) | r=16, max_seq=4096, batch=4, fp16 |
| LoRA (4-bit) | **≤ 1.5B** (best throughput) | r=32, max_seq=4096, batch=8, fp16 |

> ⚠️ T4 does **not** support bf16. Use fp16. GRPO on this GPU is tight — favor small models (≤ 3B) and keep `gpu_memory_utilization` ≤ 0.6.

### NVIDIA L4 — 24 GB VRAM (Colab Pro, GCP g2)

| Mode | Max Model | Settings |
|:----:|:---------:|:---------|
| LoRA (4-bit) | **≤ 14B** | r=16, max_seq=2048, batch=2, bf16 |
| LoRA (4-bit) | **≤ 8B** (comfortable) | r=32, max_seq=4096, batch=4, bf16 |
| LoRA (bf16) | **≤ 3B** | r=32, max_seq=4096, batch=4, bf16 |

### NVIDIA A100 (40 GB) — 40 GB VRAM

| Mode | Max Model | Settings |
|:----:|:---------:|:---------|
| LoRA (bf16) | **≤ 8B** | r=32, max_seq=4096, batch=8, bf16 |
| LoRA (4-bit) | **≤ 14B** | r=32, max_seq=4096, batch=4, bf16 |
| LoRA (bf16) | **≤ 3B** (high throughput) | r=64, max_seq=8192, batch=16, bf16 |

### NVIDIA A100 (80 GB) — 80 GB VRAM

| Mode | Max Model | Settings |
|:----:|:---------:|:---------|
| LoRA (bf16) | **≤ 32B** | r=16, max_seq=4096, batch=2, bf16 |
| LoRA (bf16) | **≤ 14B** (comfortable) | r=32, max_seq=8192, batch=8, bf16 |
| LoRA (4-bit) | **≤ 70B** | r=16, max_seq=2048, batch=2, bf16 |

### NVIDIA H100 — 80 GB VRAM

| Mode | Max Model | Settings |
|:----:|:---------:|:---------|
| LoRA (FP8) | **≤ 32B** | r=32, max_seq=8192, batch=8, bf16 |
| LoRA (FP8) | **≤ 14B** (high throughput) | r=64, max_seq=8192, batch=16, bf16 |
| LoRA (bf16) | **≤ 14B** | r=32, max_seq=8192, batch=8, bf16 |

> H100 has native FP8 hardware support. Prefer FP8 (`load_in_fp8=True`) over 4-bit for faster training at similar VRAM savings.

---

## 3. VRAM Breakdown — What Consumes Memory

During a GRPO training step, VRAM is split across:

| Component | Typical Share | Notes |
|:----------|:-------------:|:------|
| **Model weights** | 20–40% | Depends on data type: bf16 = 2 bytes/param, 4-bit = 0.25 bytes/param, FP8 = 1 byte/param |
| **Optimizer state** (AdamW) | ~4 bytes × LoRA params | Only LoRA params get optimizer state — that's why LoRA is so VRAM-efficient |
| **Activation cache** | 10–30% | Scales with batch size × seq length. Gradient checkpointing trades compute for less cache |
| **vLLM KV cache** | 15–35% | `gpu_memory_utilization` controls this. Lower = less cache but slower generation |
| **Gradients** | ~same as LoRA weights | Tiny compared to full-model gradients |
| **Framework overhead** | ~1–2 GB | CUDA context, cuBLAS workspace, etc. |

**Key takeaway:** The model weights are just one piece. You need roughly **2–3× the model weight VRAM** for a comfortable training loop with GRPO.

---

## 4. VRAM-Saving Levers (Roughly Ordered by Impact)

1. **Quantization** — Biggest single lever
   - `bf16` → `4-bit`: ~4× VRAM reduction for weights
   - `bf16` → `FP8`: ~2× VRAM reduction (with less accuracy loss than 4-bit)

2. **LoRA rank (r)** — Directly controls adapter size
   - r=64 → r=16: cuts adapter VRAM by 4× (usually saves 1–3 GB on an 8B model)
   - Diminishing returns below r=16

3. **Sequence length** — Linear impact on activation cache
   - max_seq=8192 → 4096: ~halves activation memory
   - max_seq=4096 → 2048: ~halves again

4. **Batch size** — Linear impact
   - `per_device_train_batch_size`: 8 → 4 → 2 → 1
   - Compensate with `gradient_accumulation_steps` to keep effective batch size

5. **Gradient checkpointing** — ~50% activation memory savings at ~10–20% speed cost
   - Unsloth: `use_gradient_checkpointing="unsloth"` (optimized version)

6. **vLLM `gpu_memory_utilization`** — Shrinks KV cache
   - 0.9 → 0.7 → 0.5 (lower = less generation memory, more free for training)

7. **Optimizer** — `adamw_8bit` uses ~half the optimizer memory of standard AdamW

8. **Mixed precision** — fp16/bf16 vs fp32
   - Always use bf16 on Ampere+ (A100/H100), fp16 on Turing (T4)

---

## 5. Model Recommendations for This Project

Models used in this repo and where they fit:

| Model | Params | Recommended GPU | Data Type | Notes |
|:------|:------:|:---------------:|:---------:|:------|
| `unsloth/Qwen2.5-0.5B-Instruct` | 0.5B | T4 (16 GB) | 4-bit or bf16 | Lightweight SFT. Comfortable on any GPU. |
| `unsloth/Qwen3.5-0.6B` | 0.6B | T4 (16 GB) | 4-bit or bf16 | Default GRPO model in this repo. Fits everywhere. |
| `unsloth/Qwen3-8B` | 8B | A100-40GB or L4 (4-bit) | 4-bit / FP8 / bf16 | Tool-calling SFT. Needs ≥ 24 GB VRAM for GRPO. |
| `unsloth/Qwen2.5-14B` | 14B | A100-80GB or L4 (4-bit) | 4-bit / FP8 | Larger capacity. Tight on L4, comfortable on A100-80. |

### Quick Decision Tree

```
How much VRAM do you have?
│
├─ ≤ 8 GB     → Qwen 0.5–0.6B, 4-bit, r=16, seq=2048
├─ 10–16 GB   → Qwen 0.6–3B, 4-bit, r=16–32, seq=2048–4096
├─ 24 GB      → Qwen 8B, 4-bit, r=32, seq=4096  (or 14B at r=16, seq=2048)
├─ 40 GB      → Qwen 8B, bf16, r=32, seq=8192  (or 14B at 4-bit)
├─ 48 GB      → Qwen 14B, bf16 or FP8, r=32, seq=8192
└─ 80 GB      → Qwen 32B, 4-bit or FP8, r=32, seq=8192  (or 70B at 4-bit)
```

---

## 6. OOM Debugging Checklist

If you hit CUDA out-of-memory during training:

1. Reduce `per_device_train_batch_size` (try 1 if needed)
2. Increase `gradient_accumulation_steps` to compensate
3. Reduce `max_seq_length` or `max_completion_length`
4. Lower `num_generations` in GRPOConfig (fewer rollouts = fewer tokens)
5. Reduce LoRA rank `r`
6. Switch to 4-bit quantization if not already using it
7. Lower `gpu_memory_utilization` for vLLM (e.g., 0.7 → 0.5)
8. Switch optimizer to `adamw_8bit`
9. Enable gradient checkpointing (`use_gradient_checkpointing="unsloth"`)
