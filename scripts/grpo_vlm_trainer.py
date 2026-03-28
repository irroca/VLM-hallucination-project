"""
Custom GRPO Trainer for Vision-Language Models.
Handles multimodal input (image + text) with GRPO's group relative policy optimization.

Key differences from text-only GRPO:
1. Each prompt includes an image that must be processed through the vision encoder
2. Reward computation uses visual grounding (VQA answer verification)
3. Generation requires image context
"""

import os
import json
import math
import random
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass

from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GRPOVLMConfig:
    num_generations: int = 4
    max_steps: int = 400
    batch_size: int = 2         # prompts per batch
    lr: float = 5e-6
    max_completion_length: int = 128
    max_prompt_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    kl_coeff: float = 0.01      # KL penalty coefficient
    clip_range: float = 0.2     # PPO-style clipping
    logging_steps: int = 5
    save_steps: int = 100
    warmup_steps: int = 40
    grad_accum_steps: int = 4


class GRPOVLMTrainer:
    """
    Group Relative Policy Optimization for VLMs.

    For each prompt (image + question):
    1. Generate G responses
    2. Compute reward for each response
    3. Compute group-relative advantages: A_i = (r_i - mean(r)) / std(r)
    4. Update policy to increase probability of high-advantage responses
    """

    def __init__(
        self,
        model,
        processor,
        dataset: List[Dict],
        image_dir: str,
        reward_fn: Callable,
        reward_mode: str = "full",
        config=None,
        use_wandb: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.image_dir = image_dir
        self.reward_fn = reward_fn
        self.reward_mode = reward_mode
        self.use_wandb = use_wandb

        # Parse config
        if config is not None:
            self.num_generations = getattr(config, "num_generations", 4)
            self.max_steps = getattr(config, "max_steps", 400)
            self.batch_size = getattr(config, "per_device_train_batch_size", 2)
            self.lr = getattr(config, "learning_rate", 5e-6)
            self.max_completion_length = getattr(config, "max_completion_length", 128)
            self.max_prompt_length = getattr(config, "max_prompt_length", 512)
            self.logging_steps = getattr(config, "logging_steps", 5)
            self.save_steps = getattr(config, "save_steps", 100)
            self.output_dir = getattr(config, "output_dir", "./grpo_output")
        else:
            cfg = GRPOVLMConfig()
            self.num_generations = cfg.num_generations
            self.max_steps = cfg.max_steps
            self.batch_size = cfg.batch_size
            self.lr = cfg.lr
            self.max_completion_length = cfg.max_completion_length
            self.max_prompt_length = cfg.max_prompt_length
            self.logging_steps = cfg.logging_steps
            self.save_steps = cfg.save_steps
            self.output_dir = "./grpo_output"

        self.temperature = 0.7
        self.kl_coeff = 0.01
        self.clip_range = 0.2
        self.grad_accum_steps = 4
        self.warmup_steps = 40

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=0.01)

        # Stats tracking
        self.stats = {
            "step": [],
            "mean_reward": [],
            "mean_advantage": [],
            "policy_loss": [],
            "kl_divergence": [],
            "correct_rate": [],
        }

    def _prepare_vlm_input(self, question: str, image_path: str):
        """Prepare VLM input with image and question."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": f"Answer the following question briefly: {question}"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        return inputs

    @torch.no_grad()
    def _generate_responses(self, inputs, num_generations: int) -> List[str]:
        """Generate multiple responses for a single prompt."""
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        responses = []
        for _ in range(num_generations):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
            # Decode only the generated part
            prompt_len = inputs["input_ids"].shape[1]
            generated = output_ids[0, prompt_len:]
            text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
            responses.append(text)

        return responses

    def _compute_log_probs(self, inputs, response_text: str) -> torch.Tensor:
        """Compute log probabilities of response given prompt."""
        device = next(self.model.parameters()).device

        # Encode the full sequence (prompt + response)
        full_text = self.processor.tokenizer.decode(inputs["input_ids"][0]) + response_text
        full_inputs = self.processor.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length + self.max_completion_length,
        ).to(device)

        # Forward pass
        outputs = self.model(
            input_ids=full_inputs["input_ids"],
            attention_mask=full_inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
        )

        # Compute log probs for response tokens only
        prompt_len = inputs["input_ids"].shape[1]
        logits = outputs.logits[:, prompt_len - 1:-1, :]  # shift by 1 for next-token prediction
        target_ids = full_inputs["input_ids"][:, prompt_len:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum()

    def _compute_grpo_loss(
        self,
        inputs,
        responses: List[str],
        rewards: List[float],
        old_log_probs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute GRPO loss with group-relative advantages.
        L = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        where r_t = π_new / π_old and A_t = (R_i - mean(R)) / std(R)
        """
        # Compute group-relative advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_r) / std_r

        total_loss = torch.tensor(0.0, device=next(self.model.parameters()).device, requires_grad=True)

        for i, (resp, adv, old_lp) in enumerate(zip(responses, advantages, old_log_probs)):
            # New log probs under current policy
            new_lp = self._compute_log_probs(inputs, resp)

            # Ratio
            ratio = torch.exp(new_lp - old_lp.detach())

            # Clipped objective
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            surr1 = ratio * adv.item()
            surr2 = clipped_ratio * adv.item()
            policy_loss = -torch.min(surr1, surr2)

            # KL penalty (approximate)
            kl = old_lp.detach() - new_lp
            kl_penalty = self.kl_coeff * kl

            total_loss = total_loss + policy_loss + kl_penalty

        return total_loss / len(responses)

    def train(self):
        """Main GRPO training loop."""
        logger.info(f"Starting GRPO training: {self.max_steps} steps, {self.num_generations} generations/prompt")

        self.model.train()
        random.shuffle(self.dataset)

        global_step = 0
        data_idx = 0
        accumulated_loss = 0.0
        accumulated_reward = 0.0
        accumulated_correct = 0.0
        accumulated_count = 0

        pbar = tqdm(total=self.max_steps, desc="GRPO Training")

        while global_step < self.max_steps:
            # Sample a batch of prompts
            batch_data = []
            for _ in range(self.batch_size):
                if data_idx >= len(self.dataset):
                    data_idx = 0
                    random.shuffle(self.dataset)
                batch_data.append(self.dataset[data_idx])
                data_idx += 1

            batch_loss = torch.tensor(0.0, device=next(self.model.parameters()).device, requires_grad=True)

            for item in batch_data:
                img_path = item.get("image_path", "")
                if not img_path:
                    img_id = item.get("image_id", "")
                    img_path = os.path.join(self.image_dir, f"{int(img_id):012d}.jpg")

                if not os.path.exists(img_path):
                    continue

                try:
                    # Prepare input
                    inputs = self._prepare_vlm_input(item["question"], img_path)

                    # Generate multiple responses
                    responses = self._generate_responses(inputs, self.num_generations)

                    # Compute rewards
                    gt_infos = [{"answer": item["answer"], "all_answers": item.get("all_answers", []), "question": item["question"]}] * len(responses)
                    rewards = self.reward_fn(responses, gt_infos)

                    # Compute old log probs (for ratio)
                    old_log_probs = []
                    for resp in responses:
                        with torch.no_grad():
                            lp = self._compute_log_probs(inputs, resp)
                            old_log_probs.append(lp)

                    # GRPO loss
                    loss = self._compute_grpo_loss(inputs, responses, rewards, old_log_probs)
                    batch_loss = batch_loss + loss

                    # Track stats
                    accumulated_reward += sum(rewards) / len(rewards)
                    accumulated_correct += sum(1 for r in rewards if r > 0) / len(rewards)
                    accumulated_count += 1

                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue

            # Backward
            if accumulated_count > 0:
                (batch_loss / max(1, len(batch_data))).backward()

            accumulated_loss += batch_loss.item()

            # Step optimizer every grad_accum_steps
            if (global_step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            global_step += 1
            pbar.update(1)

            # Logging
            if global_step % self.logging_steps == 0 and accumulated_count > 0:
                avg_loss = accumulated_loss / accumulated_count
                avg_reward = accumulated_reward / accumulated_count
                avg_correct = accumulated_correct / accumulated_count

                log_msg = (
                    f"Step {global_step}: loss={avg_loss:.4f}, "
                    f"reward={avg_reward:.4f}, correct_rate={avg_correct:.4f}"
                )
                logger.info(log_msg)

                self.stats["step"].append(global_step)
                self.stats["mean_reward"].append(avg_reward)
                self.stats["correct_rate"].append(avg_correct)
                self.stats["policy_loss"].append(avg_loss)

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "grpo/loss": avg_loss,
                        "grpo/mean_reward": avg_reward,
                        "grpo/correct_rate": avg_correct,
                        "grpo/step": global_step,
                    })

                accumulated_loss = 0.0
                accumulated_reward = 0.0
                accumulated_correct = 0.0
                accumulated_count = 0

            # Save checkpoint
            if global_step % self.save_steps == 0:
                save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

        pbar.close()

        # Save final stats
        stats_path = os.path.join(self.output_dir, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Training complete. Stats saved to {stats_path}")
