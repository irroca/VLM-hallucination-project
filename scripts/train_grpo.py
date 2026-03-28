"""
Stage 2a: GRPO Training with Verifiable Visual Rewards
Core innovation: Apply GRPO's verifiable reward mechanism to VLM hallucination suppression.

Reward function:
  R = r_correct (+1/0) + r_halluc (-0.5) + r_format (+0.2)

Uses TRL's GRPOTrainer with custom reward for visual QA verification.
"""

import os
import sys
import json
import re
import torch
import logging
from typing import List, Dict, Any

from PIL import Image
from collections import Counter

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "grpo", "aokvqa_grpo_v2.json")
IMAGE_DIR = os.path.join(PROJECT_DIR, "data", "grpo", "images")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "grpo-qwen2vl-2b")


# ─── Reward Functions ───

def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip().lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_correctness_reward(response: str, ground_truth: str, all_answers: List[str] = None) -> float:
    """
    Correctness reward: +1 if answer matches ground truth, 0 otherwise.
    Supports soft matching via normalized comparison and answer set.
    """
    norm_response = normalize_answer(response)
    norm_gt = normalize_answer(ground_truth)

    # Exact match after normalization
    if norm_gt in norm_response or norm_response == norm_gt:
        return 1.0

    # Check against all valid answers
    if all_answers:
        for ans in all_answers:
            if normalize_answer(ans) in norm_response:
                return 1.0

    # Partial match: check if key words from GT appear in response
    gt_words = set(norm_gt.split())
    resp_words = set(norm_response.split())
    if gt_words and gt_words.issubset(resp_words):
        return 0.5

    return 0.0


def compute_hallucination_penalty(response: str, question: str) -> float:
    """
    Hallucination penalty: -0.5 if response contains hedging or unsupported claims.
    Heuristic-based detection for common hallucination patterns.
    """
    halluc_patterns = [
        r"I can(?:'t| not) see",
        r"(?:there (?:is|are) (?:no|not))",
        r"(?:I (?:think|believe|assume) (?:there|it))",
        r"(?:it (?:seems|appears|looks) like (?:there|it) (?:might|could|may))",
    ]

    response_lower = response.lower()

    # Don't penalize if these patterns are appropriate (e.g., question asks "is there X?")
    question_lower = question.lower()
    if "is there" in question_lower or "are there" in question_lower:
        return 0.0

    for pattern in halluc_patterns:
        if re.search(pattern, response_lower):
            return -0.25

    # Excessive length often correlates with hallucination
    if len(response.split()) > 100:
        return -0.25

    return 0.0


def compute_format_reward(response: str) -> float:
    """
    Format reward: +0.2 for concise, well-structured answers.
    """
    words = response.split()
    # Reward concise answers (VQA should be brief)
    if 1 <= len(words) <= 20:
        return 0.2
    elif len(words) <= 50:
        return 0.1
    return 0.0


def reward_function(responses: List[str], ground_truths: List[Dict]) -> List[float]:
    """
    Combined reward function for GRPO.
    R = r_correct + r_halluc + r_format
    """
    rewards = []
    for resp, gt_info in zip(responses, ground_truths):
        r_correct = compute_correctness_reward(
            resp,
            gt_info["answer"],
            gt_info.get("all_answers", []),
        )
        r_halluc = compute_hallucination_penalty(resp, gt_info["question"])
        r_format = compute_format_reward(resp)
        total = r_correct + r_halluc + r_format
        rewards.append(total)
    return rewards


# ─── Dataset ───

def load_grpo_dataset(data_path: str, image_dir: str):
    """Load and format GRPO training data."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = []
    for item in raw_data:
        # v2 format: image field is absolute path
        img_path = item.get("image", "")
        if not img_path or not os.path.exists(img_path):
            # fallback: try image_id with COCO format
            image_id = item.get("image_id", "")
            if not image_id:
                continue
            try:
                img_filename = f"{int(image_id):012d}.jpg"
            except (ValueError, TypeError):
                continue
            img_path = os.path.join(image_dir, img_filename)

        if not os.path.exists(img_path):
            continue

        dataset.append({
            "prompt": item["question"],
            "image_path": img_path,
            "answer": item["answer"],
            "all_answers": item.get("all_answers", []),
            "question": item["question"],
        })

    logger.info(f"Loaded {len(dataset)} GRPO samples (with valid images)")
    return dataset


class GRPOVLMDataset:
    """Iterable dataset for GRPO training with VLM."""

    def __init__(self, data: List[Dict], processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format prompt as Qwen2-VL expects
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{item['image_path']}"},
                    {"type": "text", "text": f"Answer the following question briefly: {item['prompt']}"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": text,
            "image_path": item["image_path"],
            "answer": item["answer"],
            "all_answers": item["all_answers"],
            "question": item["question"],
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to SFT checkpoint (default: base model)")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--reward_mode", type=str, default="full",
                        choices=["correct_only", "correct_halluc", "full"],
                        help="Reward function ablation mode")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="VLM-Hallucination")
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(PROJECT_DIR, "checkpoints", "sft-qwen2vl-2b-merged")
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_DIR, "checkpoints", "Qwen2-VL-2B-Instruct")
        logger.warning(f"SFT checkpoint not found, using base model: {model_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"stage2a-grpo-{args.reward_mode}",
            config=vars(args),
            tags=["grpo", "qwen2-vl-2b", "verifiable-reward", args.reward_mode],
        )

    # ── Load model (bf16, no quantization for GRPO compatibility) ──
    logger.info(f"Loading model from {model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── LoRA for GRPO ──
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ──
    raw_data = load_grpo_dataset(DATA_PATH, IMAGE_DIR)
    dataset = GRPOVLMDataset(raw_data, processor)

    # ── Custom reward wrapper ──
    def compute_rewards(prompts, completions, **kwargs):
        """Compute rewards for GRPO batch."""
        # Extract ground truth info from prompts
        # In practice, we need to maintain a mapping
        gt_infos = kwargs.get("ground_truths", [])
        rewards = []
        for completion, gt_info in zip(completions, gt_infos):
            text = completion if isinstance(completion, str) else completion.get("content", "")

            if args.reward_mode == "correct_only":
                r = compute_correctness_reward(text, gt_info["answer"], gt_info.get("all_answers", []))
            elif args.reward_mode == "correct_halluc":
                r = compute_correctness_reward(text, gt_info["answer"], gt_info.get("all_answers", []))
                r += compute_hallucination_penalty(text, gt_info["question"])
            else:  # full
                r = compute_correctness_reward(text, gt_info["answer"], gt_info.get("all_answers", []))
                r += compute_hallucination_penalty(text, gt_info["question"])
                r += compute_format_reward(text)
            rewards.append(r)
        return rewards

    # ── GRPO Config (using custom dataclass, not TRL GRPOConfig) ──
    from dataclasses import dataclass

    @dataclass
    class GRPOCfg:
        output_dir: str = OUTPUT_DIR
        max_steps: int = args.max_steps
        per_device_train_batch_size: int = args.batch_size
        learning_rate: float = args.lr
        num_generations: int = args.num_generations
        max_completion_length: int = 128
        max_prompt_length: int = 512
        logging_steps: int = 5
        save_steps: int = 100

    grpo_config = GRPOCfg()

    logger.info("GRPO training would start here.")
    logger.info(f"Reward mode: {args.reward_mode}")
    logger.info(f"Num generations: {args.num_generations}")
    logger.info(f"Max steps: {args.max_steps}")

    # Note: TRL's GRPOTrainer for VLMs needs custom integration.
    # The standard GRPOTrainer works with text-only models.
    # For VLM, we implement a custom training loop.

    from grpo_vlm_trainer import GRPOVLMTrainer

    trainer = GRPOVLMTrainer(
        model=model,
        processor=processor,
        dataset=raw_data,
        image_dir=IMAGE_DIR,
        reward_fn=reward_function,
        reward_mode=args.reward_mode,
        config=grpo_config,
        use_wandb=args.use_wandb,
    )
    trainer.train()

    # Save
    logger.info(f"Saving GRPO model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
