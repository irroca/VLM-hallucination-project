"""
Evaluation script for VLM Hallucination project.
Evaluates models on:
1. POPE (Polling-based Object Probing Evaluation) - hallucination benchmark
2. General VQA accuracy
3. Refusal rate, response length, inference latency
"""

import os
import sys
import json
import time
import torch
import logging
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(model_path: str, lora_path: str = None, quantize: bool = False):
    """Load model with optional LoRA adapter."""
    logger.info(f"Loading model from {model_path} (quantize={quantize})")

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
    }
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path)

    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, processor


def generate_response(model, processor, image_path: str, question: str, max_tokens: int = 128, add_yn_prompt: bool = False) -> Tuple[str, float]:
    """Generate response for an image+question pair. Returns (response, latency_ms)."""
    prompt_text = question
    if add_yn_prompt:
        prompt_text = question.rstrip("?. ") + "? Answer with yes or no."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy for eval
            temperature=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    latency_ms = (time.time() - start_time) * 1000

    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, prompt_len:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True)

    return response, latency_ms


def evaluate_pope(model, processor, pope_data: List[Dict], image_dir: str, max_samples: int = None) -> Dict:
    """
    Evaluate on POPE benchmark.

    Returns:
        - accuracy, precision, recall, F1
        - yes_rate (model's tendency to say "yes")
        - hallucination_rate
    """
    logger.info("Evaluating on POPE benchmark...")

    if max_samples:
        pope_data = pope_data[:max_samples]

    results = {
        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "total": 0,
        "latencies": [],
        "response_lengths": [],
        "refusal_count": 0,
        "errors": [],
        "category_results": defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}),
    }

    for item in tqdm(pope_data, desc="POPE Eval"):
        # Determine image path
        img_source = item.get("image", "")
        # Try direct match with .jpg extension
        img_path = os.path.join(image_dir, img_source + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, img_source)
        if not os.path.exists(img_path):
            # Try as COCO ID
            try:
                img_path = os.path.join(image_dir, f"COCO_val2014_{int(img_source):012d}.jpg")
            except (ValueError, TypeError):
                pass

        if not os.path.exists(img_path):
            results["errors"].append(f"Image not found: {img_source}")
            continue

        question = item["question"]
        gt_answer = item["answer"].strip().lower()

        try:
            response, latency = generate_response(model, processor, img_path, question, max_tokens=32, add_yn_prompt=True)
            response_lower = response.strip().lower()

            # Detect refusal
            refusal_keywords = ["i cannot", "i can't", "i'm unable", "sorry", "i don't"]
            is_refusal = any(kw in response_lower for kw in refusal_keywords)
            if is_refusal:
                results["refusal_count"] += 1

            # Determine predicted answer
            if "yes" in response_lower.split()[:3]:
                pred = "yes"
            elif "no" in response_lower.split()[:3]:
                pred = "no"
            else:
                # Check more broadly
                pred = "yes" if "yes" in response_lower else "no"

            # Compute TP/FP/TN/FN
            category = item.get("category", "unknown")
            if gt_answer == "yes" and pred == "yes":
                results["tp"] += 1
                results["category_results"][category]["tp"] += 1
            elif gt_answer == "no" and pred == "yes":
                results["fp"] += 1
                results["category_results"][category]["fp"] += 1
            elif gt_answer == "no" and pred == "no":
                results["tn"] += 1
                results["category_results"][category]["tn"] += 1
            elif gt_answer == "yes" and pred == "no":
                results["fn"] += 1
                results["category_results"][category]["fn"] += 1

            results["total"] += 1
            results["latencies"].append(latency)
            results["response_lengths"].append(len(response.split()))

        except Exception as e:
            results["errors"].append(str(e))

    # Compute metrics
    tp, fp, tn, fn = results["tp"], results["fp"], results["tn"], results["fn"]
    total = results["total"]

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    yes_rate = (tp + fp) / max(total, 1)
    hallucination_rate = fp / max(fp + tp, 1)  # false positive among all "yes" predictions

    avg_latency = sum(results["latencies"]) / max(len(results["latencies"]), 1)
    avg_length = sum(results["response_lengths"]) / max(len(results["response_lengths"]), 1)
    refusal_rate = results["refusal_count"] / max(total, 1)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "yes_rate": round(yes_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "refusal_rate": round(refusal_rate, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_response_length": round(avg_length, 1),
        "total_samples": total,
        "errors": len(results["errors"]),
    }

    # Per-category metrics
    category_metrics = {}
    for cat, cr in results["category_results"].items():
        cat_total = cr["tp"] + cr["fp"] + cr["tn"] + cr["fn"]
        cat_acc = (cr["tp"] + cr["tn"]) / max(cat_total, 1)
        cat_halluc = cr["fp"] / max(cr["fp"] + cr["tp"], 1)
        category_metrics[cat] = {
            "accuracy": round(cat_acc, 4),
            "hallucination_rate": round(cat_halluc, 4),
            "total": cat_total,
        }
    metrics["category_metrics"] = category_metrics

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--pope_data", type=str, default=os.path.join(PROJECT_DIR, "data", "eval", "pope_eval.json"))
    parser.add_argument("--image_dir", type=str, default=os.path.join(PROJECT_DIR, "data", "val2014"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_DIR, "logs", "eval_results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, processor = load_model(args.model_path, args.lora_path)

    # Load POPE data
    with open(args.pope_data, "r", encoding="utf-8") as f:
        pope_data = json.load(f)

    # Evaluate POPE
    pope_results = evaluate_pope(model, processor, pope_data, args.image_dir, args.max_samples)

    # Print results
    print("\n" + "=" * 60)
    print(f"POPE Evaluation Results — {args.model_name}")
    print("=" * 60)
    for k, v in pope_results.items():
        if k != "category_metrics":
            print(f"  {k}: {v}")
    print("\n  Per-category:")
    for cat, metrics in pope_results.get("category_metrics", {}).items():
        print(f"    {cat}: acc={metrics['accuracy']}, halluc={metrics['hallucination_rate']}")

    # Save results
    out_file = os.path.join(args.output_dir, f"pope_{args.model_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"model": args.model_name, "pope": pope_results}, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
