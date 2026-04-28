#!/usr/bin/env python3
"""
Qwen3 Tool-Calling SFT Training Script (Unsloth)
Fine-tune Qwen3 on agentic tool-calling trajectories using Unsloth for 2x speedup
and lower memory usage.
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
import os

# ===================== CONFIGURATION =====================
MODEL_NAME = "unsloth/Qwen3-8B"  # Unsloth-optimized Qwen3 model
DATASET_NAME = "Mustafaege/qwen3.5-toolcalling-v2"
OUTPUT_DIR = "./qwen3-toolcalling-sft"

# Training hyperparameters
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 8192
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"

# LoRA Configuration
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.0  # Unsloth optimized for 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ===================== SETUP =====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading model: {MODEL_NAME}")
print(f"Loading dataset: {DATASET_NAME}")

# Load model + tokenizer with Unsloth (4-bit by default for memory efficiency)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (bf16 on Ampere+, fp16 otherwise)
    load_in_4bit=True,
)

# Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ===================== DATASET PROCESSING =====================
def formatting_func(examples):
    """Apply Qwen3 chat template to messages."""
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        for messages in examples["messages"]
    ]
    return {"text": texts}

# Load and preprocess dataset
dataset = load_dataset(DATASET_NAME, split="train")
eval_dataset = load_dataset(DATASET_NAME, split="test")

print(f"Training samples: {len(dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)
eval_dataset = eval_dataset.map(formatting_func, batched=True, remove_columns=eval_dataset.column_names)

# ===================== TRAINING ARGUMENTS =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER,

    # Precision (let Unsloth pick best)
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",

    # Training behavior
    max_grad_norm=1.0,
    weight_decay=0.01,

    # Evaluation / checkpoints
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,

    seed=3407,
    report_to="tensorboard",
)

# ===================== TRAINER =====================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    packing=False,
)

# Train only on assistant responses (mask user/system tokens from loss)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ===================== TRAIN =====================
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

trainer.train()

# ===================== SAVE =====================
print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Optional: save merged 16-bit model for inference
# model.save_pretrained_merged(f"{OUTPUT_DIR}-merged", tokenizer, save_method="merged_16bit")

print(f"\n✓ Training complete! Model saved to: {OUTPUT_DIR}")
print("=" * 80)

# ===================== INFERENCE TEST =====================
def test_inference(prompt):
    """Quick inference test with trained model"""
    from transformers import TextStreamer

    FastLanguageModel.for_inference(model)  # 2x faster inference

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

print("\n🧪 Testing inference...")
test_prompt = "I need to calculate the weather forecast for New York tomorrow"
print(f"\nPrompt: {test_prompt}")
print("\nResponse:")
test_inference(test_prompt)
