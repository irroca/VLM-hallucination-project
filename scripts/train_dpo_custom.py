"""
Stage 2b: Custom DPO Training Loop for VLM
Bypasses TRL DPOTrainer's VLM compatibility issues.
Uses RLAIF-V preference data (faithful vs hallucinated).
"""

import os
import json
import random
import math
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List

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
DATA_PATH = os.path.join(PROJECT_DIR, "data", "dpo", "rlaifv_dpo.json")
IMAGE_DIR = os.path.join(PROJECT_DIR, "data", "dpo", "images")


def compute_log_probs(model, processor, img_path, question, response, device):
    """Compute log probability of a response given image+question."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{img_path}"},
            {"type": "text", "text": question},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": response},
        ]},
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
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    labels = inputs["input_ids"]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Sum over all tokens
    return token_log_probs.sum()


def dpo_loss(pi_logps_chosen, pi_logps_rejected, ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """
    DPO loss: -log sigmoid(beta * (log(pi(y_w)/ref(y_w)) - log(pi(y_l)/ref(y_l))))
    """
    pi_ratio = pi_logps_chosen - pi_logps_rejected
    ref_ratio = ref_logps_chosen - ref_logps_rejected
    logits = beta * (pi_ratio - ref_ratio)
    loss = -F.logsigmoid(logits)
    return loss, logits


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="VLM-Hallucination")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        PROJECT_DIR, "checkpoints", f"dpo-qwen2vl-2b-beta{args.beta}"
    )
    os.makedirs(output_dir, exist_ok=True)

    run_name = args.run_name or f"stage2b-dpo-beta{args.beta}"

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name,
                   config=vars(args), tags=["dpo", "custom", f"beta-{args.beta}"])

    # ── Load data ──
    with open(DATA_PATH) as f:
        raw_data = json.load(f)

    # Filter for existing images
    data = []
    for i, item in enumerate(raw_data):
        img_path = os.path.join(IMAGE_DIR, f"{i:06d}.jpg")
        if os.path.exists(img_path):
            item["_img_path"] = img_path
            item["_idx"] = i
            data.append(item)

    if args.max_samples and len(data) > args.max_samples:
        random.seed(42)
        data = random.sample(data, args.max_samples)

    logger.info(f"DPO data: {len(data)} preference pairs")

    # ── Load policy model ──
    logger.info(f"Loading policy model from {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28

    # LoRA
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Load reference model (frozen) ──
    logger.info("Loading reference model...")
    ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    device = next(model.parameters()).device

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(data) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # ── Training loop ──
    logger.info(f"DPO Training: {len(data)} pairs, beta={args.beta}, steps={total_steps}")
    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_margin = 0.0
    skipped = 0

    for epoch in range(args.epochs):
        random.shuffle(data)

        for i, item in enumerate(data):
            try:
                img_path = item["_img_path"]

                # Policy log probs
                pi_lp_chosen = compute_log_probs(
                    model, processor, img_path, item["question"], item["chosen"], device
                )
                pi_lp_rejected = compute_log_probs(
                    model, processor, img_path, item["question"], item["rejected"], device
                )

                # Reference log probs
                with torch.no_grad():
                    ref_lp_chosen = compute_log_probs(
                        ref_model, processor, img_path, item["question"], item["chosen"], device
                    )
                    ref_lp_rejected = compute_log_probs(
                        ref_model, processor, img_path, item["question"], item["rejected"], device
                    )

                loss, logits = dpo_loss(
                    pi_lp_chosen, pi_lp_rejected,
                    ref_lp_chosen, ref_lp_rejected,
                    beta=args.beta
                )
                (loss / args.grad_accum).backward()
                accum_loss += loss.item()
                accum_margin += logits.item()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1
                continue
            except Exception as e:
                skipped += 1
                if skipped <= 5 or skipped % 50 == 0:
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
                    n = min(10, global_step)
                    avg_loss = accum_loss / (n * args.grad_accum)
                    avg_margin = accum_margin / (n * args.grad_accum)
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        f"E{epoch+1} Step {global_step}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | Margin {avg_margin:.4f} | "
                        f"LR {lr_now:.2e} | Skip {skipped}"
                    )
                    if args.use_wandb:
                        import wandb
                        wandb.log({
                            "dpo/loss": avg_loss, "dpo/margin": avg_margin,
                            "dpo/lr": lr_now, "dpo/step": global_step,
                        })
                    accum_loss = 0.0
                    accum_margin = 0.0

                if global_step % 100 == 0:
                    ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    logger.info(f"Checkpoint: {ckpt}")

    # ── Save ──
    logger.info(f"Saving DPO model to {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    logger.info(f"DPO done! Steps={global_step}, Skipped={skipped}")
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
