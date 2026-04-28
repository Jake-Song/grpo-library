r"""
SFT training on Colab via Unsloth + TRL.

Standalone single-file script for supervised fine-tuning. Uses
`trl.SFTTrainer` with Unsloth's `FastLanguageModel` for fast loading
and reduced VRAM.

Colab install (paste into a cell BEFORE running this script):

    %%capture
    import os, subprocess
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
    !pip install --upgrade -qqq uv
    if "COLAB_" not in "".join(os.environ.keys()):
        !pip install unsloth vllm
    else:
        try:
            import numpy, PIL
            _numpy = f'numpy=={numpy.__version__}'
            _pil = f'pillow=={PIL.__version__}'
        except Exception:
            _numpy, _pil = "numpy", "pillow"
        try:
            _gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode()
        except Exception:
            _gpu = ""
        if "T4" in _gpu:
            _vllm, _triton = "vllm==0.9.2", "triton==3.2.0"
        elif "L4" in _gpu or "A100" in _gpu or "H100" in _gpu:
            _vllm, _triton = "vllm==0.15.1", "triton"
        else:
            _vllm, _triton = "vllm==0.15.1", "triton"
        !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
        !uv pip install -qqq {_triton}
    !uv pip install -qqq transformers==4.56.2
    !uv pip install -qqq --no-deps trl==0.22.2

Then either paste the rest of this file into a cell, or:

    !python sft_train_unsloth.py
"""

import os
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import sys

# Unsloth MUST be imported before transformers/trl so patches are applied.
from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# =============================================================================
# Config
# =============================================================================

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train"
MAX_TRAIN_EXAMPLES = 1000

# Detect GPU and pick a profile
import subprocess as _sp

try:
    _GPU = _sp.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    ).decode().strip()
except Exception:
    _GPU = ""

if "H100" in _GPU:
    MAX_SEQ_LENGTH = 4096
    LORA_RANK = 32
    PER_DEVICE_BATCH_SIZE = 8
    LOAD_IN_4BIT = False
elif "A100" in _GPU:
    MAX_SEQ_LENGTH = 4096
    LORA_RANK = 32
    PER_DEVICE_BATCH_SIZE = 4
    LOAD_IN_4BIT = False
elif "L4" in _GPU:
    MAX_SEQ_LENGTH = 2048
    LORA_RANK = 32
    PER_DEVICE_BATCH_SIZE = 2
    LOAD_IN_4BIT = True
else:  # T4 or unknown
    MAX_SEQ_LENGTH = 2048
    LORA_RANK = 16
    PER_DEVICE_BATCH_SIZE = 2
    LOAD_IN_4BIT = True

GRAD_ACCUM_STEPS = 4
MAX_STEPS = 200
SAVE_STEPS = 200
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
SEED = 3407

# =============================================================================
# Helpers
# =============================================================================


def format_alpaca(example):
    """Convert Alpaca-style fields into a single text string."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return {"text": text}


def main() -> None:
    _quant = "4bit" if LOAD_IN_4BIT else "bf16"
    print(f"Loading {MODEL_NAME} on {_GPU or 'unknown GPU'} ({_quant} + LoRA r={LORA_RANK})...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=False,
        load_in_fp8=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    print(f"Loading dataset {DATASET_NAME}...")
    raw = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    if len(raw) > MAX_TRAIN_EXAMPLES:
        raw = raw.shuffle(seed=SEED).select(range(MAX_TRAIN_EXAMPLES))
    dataset = raw.map(format_alpaca, remove_columns=raw.column_names)
    print(f"Dataset rows: {len(dataset)}")

    training_args = SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        optim="adamw_8bit",
        logging_steps=10,
        report_to="none",
        seed=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    model.save_lora("sft_lora")
    print("Saved LoRA adapter to sft_lora/")

    # Inference smoke test
    print("\n===== Inference sample =====")
    FastLanguageModel.for_inference(model)
    sample_prompt = "### Instruction:\nList three benefits of regular exercise.\n\n### Response:\n"
    inputs = tokenizer(sample_prompt, return_tensors="pt", truncation=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated)


if __name__ == "__main__":
    main()
