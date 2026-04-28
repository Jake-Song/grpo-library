#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen3 tool-calling model on the held-out test split.

Measures first-turn tool call accuracy:
  - tool_call_present_acc : predicted tool call presence matches ground truth
  - tool_name_acc         : exact match on function name
  - args_exact_acc        : exact match on arguments JSON string
  - args_f1               : field-level F1 over argument keys+values
  - full_tool_call_acc    : exact match on (name, arguments) pair
  - valid_json_rate       : predicted JSON is syntactically valid

Usage:
    uv run eval_toolcalling.py --model_path ./qwen3-toolcalling-sft --max_samples 500
"""

import argparse
import json
import re
import sys
from collections import Counter
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# ---------------------------------------------------------------------------
# Regex for Qwen3 <tool_call> blocks (accepts single-quoted or double-quoted)
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract all <tool_call> JSON blobs from *text*."""
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        raw = m.group(1).strip()
        # Some datasets wrap the JSON in markdown code fences
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            calls.append({"_raw": raw, "_valid": False})
    return calls


def build_prompt(messages: list[dict], tokenizer) -> str:
    """Return chat-templated text up to the first user turn, ready for generation."""
    # Find first user message index
    first_user_idx = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if first_user_idx is None:
        raise ValueError("No user message found")
    prefix = messages[: first_user_idx + 1]
    return tokenizer.apply_chat_template(
        prefix, tokenize=False, add_generation_prompt=True
    )


def get_first_assistant_content(messages: list[dict]) -> str | None:
    """Return content of the first assistant message after the first user message."""
    seen_user = False
    for m in messages:
        if m["role"] == "user":
            seen_user = True
        if seen_user and m["role"] == "assistant":
            return m["content"]
    return None


def arg_f1(pred_args: dict, gold_args: dict) -> float:
    """Compute token-ish F1 over flattened key=value strings."""
    pred_set = {f"{k}={json.dumps(v, ensure_ascii=False, sort_keys=True)}" for k, v in pred_args.items()}
    gold_set = {f"{k}={json.dumps(v, ensure_ascii=False, sort_keys=True)}" for k, v in gold_args.items()}
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_model_and_tokenizer(model_path: str):
    """Load base model + optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect whether *model_path* is a LoRA adapter (has adapter_config.json)
    import os as _os
    is_adapter = _os.path.exists(_os.path.join(model_path, "adapter_config.json"))

    common_kwargs = dict(
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if is_adapter:
        if PeftModel is None:
            raise ImportError("peft is required to load a LoRA adapter. Run: uv add peft")
        print(f"Loading LoRA adapter from {model_path}")
        with open(_os.path.join(model_path, "adapter_config.json")) as f:
            base_name = json.load(f).get("base_model_name_or_path", model_path)
        model = AutoModelForCausalLM.from_pretrained(base_name, **common_kwargs)
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # simplify inference
    else:
        print(f"Loading full model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)

    model.eval()
    return model, tokenizer


def evaluate(
    model_path: str,
    max_samples: int | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.95,
    device: str = "auto",
):
    dataset = load_dataset("Mustafaege/qwen3.5-toolcalling-v2", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # ---------- Baseline mode (no model loaded) ----------
    baseline = model_path is None
    if baseline:
        print("No --model_path provided — running **baseline** (always predict no tool)\n")
    else:
        model, tokenizer = load_model_and_tokenizer(model_path)
        device_str = next(model.parameters()).device
        print(f"Model loaded on {device_str}")

    print(f"Evaluating on {len(dataset)} examples...\n")

    stats = Counter()
    errors = []

    for idx, example in enumerate(dataset):
        messages = example["messages"]

        gold_content = get_first_assistant_content(messages)
        if gold_content is None:
            errors.append((idx, "No assistant turn"))
            continue

        gold_calls = parse_tool_calls(gold_content)
        gold_has_tool = len(gold_calls) > 0 and all("_valid" not in c for c in gold_calls)

        if baseline:
            # Naive baseline: never predicts a tool call
            pred_calls = []
        else:
            try:
                prompt = build_prompt(messages, tokenizer)
            except ValueError as e:
                errors.append((idx, str(e)))
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(device_str)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
            )
            pred_calls = parse_tool_calls(generated)

        pred_has_tool = len(pred_calls) > 0
        pred_valid = pred_has_tool and all("_valid" not in c for c in pred_calls)

        # 1. Presence accuracy
        if pred_has_tool == gold_has_tool:
            stats["tool_call_present_acc"] += 1

        if gold_has_tool and pred_has_tool:
            gc = gold_calls[0]
            pc = pred_calls[0]

            # valid JSON rate
            if pred_valid:
                stats["valid_json_rate"] += 1

            # name accuracy
            if pc.get("name") == gc.get("name"):
                stats["tool_name_acc"] += 1

            # args exact
            if json.dumps(pc.get("arguments", {}), sort_keys=True, ensure_ascii=False) == \
               json.dumps(gc.get("arguments", {}), sort_keys=True, ensure_ascii=False):
                stats["args_exact_acc"] += 1

            # args f1
            stats["args_f1_sum"] += arg_f1(
                pc.get("arguments", {}), gc.get("arguments", {})
            )
            stats["args_f1_count"] += 1

            # full tool call exact
            if pc.get("name") == gc.get("name") and \
               json.dumps(pc.get("arguments", {}), sort_keys=True, ensure_ascii=False) == \
               json.dumps(gc.get("arguments", {}), sort_keys=True, ensure_ascii=False):
                stats["full_tool_call_acc"] += 1

            stats["both_have_tool"] += 1
        elif not gold_has_tool and not pred_has_tool:
            # Both conversational: count as correct for all applicable metrics
            stats["tool_name_acc"] += 1
            stats["args_exact_acc"] += 1
            stats["full_tool_call_acc"] += 1
            stats["valid_json_rate"] += 1
            stats["args_f1_sum"] += 1.0
            stats["args_f1_count"] += 1
            stats["both_no_tool"] += 1

        if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
            print(f"  Processed {idx + 1}/{len(dataset)}")

    total = len(dataset) - len(errors)
    label = "BASELINE" if baseline else "RESULTS"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Total evaluated     : {total}")
    print(f"Skipped (errors)    : {len(errors)}")

    def pct(key: str, denom: int | None = None) -> str:
        d = denom if denom is not None else total
        return f"{stats.get(key, 0) / d * 100:.2f}%" if d else "N/A"

    print(f"Tool presence acc   : {stats['tool_call_present_acc']}/{total} = {pct('tool_call_present_acc')}")

    tool_denom = stats.get("both_have_tool", 0) + stats.get("both_no_tool", 0)
    print(f"Tool name acc       : {stats['tool_name_acc']}/{tool_denom} = {pct('tool_name_acc', tool_denom)}")
    print(f"Args exact acc      : {stats['args_exact_acc']}/{tool_denom} = {pct('args_exact_acc', tool_denom)}")
    print(f"Full tool call acc  : {stats['full_tool_call_acc']}/{tool_denom} = {pct('full_tool_call_acc', tool_denom)}")

    both_tool = stats.get("both_have_tool", 0)
    if both_tool:
        print(f"Valid JSON rate     : {stats.get('valid_json_rate', 0)}/{both_tool} = {pct('valid_json_rate', both_tool)}")
        avg_f1 = stats.get("args_f1_sum", 0.0) / stats.get("args_f1_count", 1)
        print(f"Avg args F1         : {avg_f1:.3f}")

    if errors:
        print(f"\nFirst 5 errors: {errors[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 tool-calling SFT model")
    parser.add_argument("--model_path", default=None, help="Path to model or LoRA adapter (omit for baseline)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit eval to N examples")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
