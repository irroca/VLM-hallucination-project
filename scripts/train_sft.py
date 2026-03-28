"""
Stage 1: SFT Cold-start Training for Qwen2-VL-2B-Instruct
Uses QLoRA (r=32) on RLAIF-V chosen responses.
Processes one sample at a time (VLM images have variable token counts).
"""

import os
import json
import random
import torch
import logging
from typing import Dict

from torch.utils.data import Dataset

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "checkpoints", "Qwen2-VL-2B-Instruct")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "sft", "rlaifv_sft.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "sft-qwen2vl-2b")


class VLMSFTDataset(Dataset):
    def __init__(self, data_path: str, max_samples: int = None):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.data = [item for item in raw if os.path.exists(item["image"])]
        if max_samples:
            self.data = self.data[:max_samples]
        logger.info(f"Loaded {len(self.data)} SFT samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def process_sample(item: Dict, processor) -> Dict[str, torch.Tensor]:
    """Process one sample into model-ready tensors."""
    img_path = item["image"]
    question = item["messages"][0]["content"]
    answer = item["messages"][1]["content"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )

    labels = inputs["input_ids"].clone()
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="VLM-Hallucination")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name="stage1-sft-qwen2vl-2b",
                   config=vars(args), tags=["sft", "qwen2-vl-2b", "qlora"])

    # ── Model ──
    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── LoRA ──
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Data ──
    dataset = VLMSFTDataset(DATA_PATH, max_samples=args.max_samples)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(dataset) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
    )

    # ── Train ──
    logger.info(f"Training: {len(dataset)} samples, {args.epochs} epochs, grad_accum={args.grad_accum}")
    logger.info(f"Total optimizer steps: {total_steps}")

    model.train()
    device = next(model.parameters()).device
    global_step = 0
    accum_loss = 0.0
    skipped = 0

    for epoch in range(args.epochs):
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            item = dataset[idx]
            try:
                inputs = process_sample(item, processor)
                # Move to device, skip None values
                batch = {}
                for k, v in inputs.items():
                    if v is not None:
                        batch[k] = v.to(device)

                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                accum_loss += loss.item()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1
                continue
            except Exception as e:
                skipped += 1
                if skipped <= 5 or skipped % 100 == 0:
                    logger.warning(f"Skip #{skipped}: {type(e).__name__}: {e}")
                continue

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = accum_loss / 10 if global_step >= 10 else accum_loss
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        f"E{epoch+1} Step {global_step}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | LR {lr_now:.2e} | Skip {skipped}"
                    )
                    if args.use_wandb:
                        import wandb
                        wandb.log({"sft/loss": avg_loss, "sft/lr": lr_now,
                                   "sft/step": global_step, "sft/skipped": skipped})
                    accum_loss = 0.0

                if global_step % 200 == 0:
                    ckpt = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    logger.info(f"Checkpoint saved: {ckpt}")

    # ── Save ──
    logger.info(f"Saving final LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Merge
    merged_dir = os.path.join(PROJECT_DIR, "checkpoints", "sft-qwen2vl-2b-merged")
    logger.info(f"Merging LoRA → {merged_dir}")
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        processor.save_pretrained(merged_dir)
        logger.info("Merge complete!")
    except Exception as e:
        logger.warning(f"Merge failed (will use adapter instead): {e}")

    logger.info(f"SFT done! Steps={global_step}, Skipped={skipped}")
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
