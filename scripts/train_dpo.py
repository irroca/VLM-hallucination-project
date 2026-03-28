"""
Stage 2b: DPO Preference Alignment for VLM Hallucination Suppression
Uses RLAIF-V preference data (faithful vs hallucinated responses).
"""

import os
import json
import torch
import logging
from typing import List, Dict, Any
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "dpo", "rlaifv_dpo.json")
IMAGE_DIR = os.path.join(PROJECT_DIR, "data", "dpo", "images")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "dpo-qwen2vl-2b")


class DPOVLMDataset(Dataset):
    """Dataset for DPO training with image preference pairs."""

    def __init__(self, data_path: str, image_dir: str, processor, max_length: int = 1024):
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        # Filter for samples with existing images
        valid_data = []
        for i, item in enumerate(self.raw_data):
            img_path = os.path.join(image_dir, f"{i:06d}.jpg")
            if os.path.exists(img_path):
                item["_img_path"] = img_path
                item["_idx"] = i
                valid_data.append(item)
        self.data = valid_data
        logger.info(f"Loaded {len(self.data)} DPO preference pairs (with images)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["_img_path"]

        # Build prompt
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": prompt,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "image_path": img_path,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to checkpoint (SFT or GRPO)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="VLM-Hallucination")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    # Determine model path
    model_path = args.model_path
    if model_path is None:
        # Try SFT merged first
        sft_path = os.path.join(PROJECT_DIR, "checkpoints", "sft-qwen2vl-2b-merged")
        if os.path.exists(sft_path):
            model_path = sft_path
        else:
            model_path = os.path.join(PROJECT_DIR, "checkpoints", "Qwen2-VL-2B-Instruct")
    logger.info(f"Using model: {model_path}")

    run_name = args.run_name or f"stage2b-dpo-beta{args.beta}"
    output_dir = os.path.join(OUTPUT_DIR, f"beta-{args.beta}")
    os.makedirs(output_dir, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["dpo", "qwen2-vl-2b", f"beta-{args.beta}"],
        )

    # ── Load model ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── LoRA ──
    model = prepare_model_for_kbit_training(model)
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

    # ── Reference model ──
    ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Dataset ──
    dataset = DPOVLMDataset(DATA_PATH, IMAGE_DIR, processor, args.max_length)

    # Split: 90% train, 10% eval using index-based split
    total = len(dataset)
    train_size = int(0.9 * total)
    indices = list(range(total))
    import random as rng
    rng.seed(42)
    rng.shuffle(indices)
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    # Build HF datasets for DPOTrainer compatibility
    from datasets import Dataset as HFDataset
    train_records = [dataset[i] for i in train_indices]
    eval_records = [dataset[i] for i in eval_indices]

    train_data = HFDataset.from_dict({
        "prompt": [r["prompt"] for r in train_records],
        "chosen": [r["chosen"] for r in train_records],
        "rejected": [r["rejected"] for r in train_records],
    })
    eval_data = HFDataset.from_dict({
        "prompt": [r["prompt"] for r in eval_records],
        "chosen": [r["chosen"] for r in eval_records],
        "rejected": [r["rejected"] for r in eval_records],
    })

    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # ── DPO Training ──
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=512,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        report_to="wandb" if args.use_wandb else "none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=processor,
    )

    logger.info(f"Starting DPO training with beta={args.beta}...")
    trainer.train()

    # Save
    logger.info(f"Saving DPO model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    logger.info("DPO training complete!")

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
