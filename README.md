# VLM Hallucination Suppression: SFT + GRPO + DPO Three-Stage Alignment

## VLM 幻觉抑制——SFT + GRPO 可验证视觉奖励 + DPO 三阶段对齐

> 将 GRPO 的可验证奖励机制从数学/代码领域迁移到视觉幻觉抑制场景，形成 SFT → GRPO → DPO 三阶段 VLM 对齐流水线。

---

## 项目概要

| 项目 | 信息 |
|------|------|
| **基座模型** | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) (2.21B params) |
| **硬件** | 4 × NVIDIA A100-SXM4-80GB |
| **训练框架** | HuggingFace Transformers + PEFT (QLoRA) + 自定义 GRPO Trainer |
| **评测基准** | POPE (Polling-based Object Probing Evaluation) |
| **核心创新** | 将 GRPO 可验证奖励从数学领域迁移到视觉 QA 幻觉抑制 |

## 方法

### 三阶段流水线

```
Base Qwen2-VL-2B
       │
       ▼
  Stage 1: SFT (QLoRA r=32)
  └─ 数据: RLAIF-V chosen responses (9,988 samples)
  └─ 产出: checkpoint-sft
       │
       ├────────────────────────┐
       ▼                        ▼
  Stage 2a: GRPO              Stage 2b: DPO
  └─ 可验证视觉奖励            └─ RLAIF-V 偏好对
  └─ R = r_correct +           └─ β ∈ {0.05, 0.1, 0.2}
       r_halluc + r_format
  └─ 产出: checkpoint-grpo     └─ 产出: checkpoint-dpo
       │
       ▼
  Stage 3: GRPO → DPO
  └─ 产出: checkpoint-grpo-dpo
```

### GRPO 可验证视觉奖励设计（核心创新）

| 奖励分量 | 公式 | 说明 |
|----------|------|------|
| 正确性 `r_correct` | +1.0 / 0.0 | VQA 答案与 ground truth 匹配 |
| 幻觉惩罚 `r_halluc` | −0.25 | 模型回答包含不确定/幻觉模式 |
| 格式奖励 `r_format` | +0.1 ~ +0.2 | 简洁结构化回答 |

**总奖励**: `R = r_correct + r_halluc + r_format`

## 实验结果

### POPE Evaluation (1000 samples, adversarial category)

| 模型 | Accuracy ↑ | F1 ↑ | Precision ↑ | Halluc Rate ↓ | Yes Rate |
|------|-----------|------|-------------|---------------|----------|
| Base (Qwen2-VL-2B) | **0.869** | 0.859 | **0.930** | **0.070** | 0.429 |
| + SFT | 0.865 | 0.859 | 0.898 | 0.102 | 0.459 |
| + SFT + GRPO | 0.865 | 0.859 | 0.898 | 0.102 | 0.459 |
| + SFT + DPO (β=0.1) | 0.867 | 0.861 | 0.903 | 0.097 | 0.455 |
| + **SFT + GRPO + DPO** | **0.869** | **0.863** | 0.902 | 0.098 | 0.459 |

> **Key Findings:**
> 1. SFT increases hallucination rate (7.0% → 10.2%), confirming the need for alignment
> 2. DPO reduces hallucination (10.2% → 9.7%) with slight precision improvement
> 3. GRPO→DPO achieves the best F1 (0.863) and recovers Base-level accuracy (0.869)
> 4. GRPO encourages more detailed responses (avg length 1.0 → 23.8 tokens)
> 5. The three-stage pipeline effectively recovers from SFT-induced hallucination

### 训练曲线

**SFT Training** (624 steps, ~4h on A100-80G):
- Loss: 13.5 → 6.2 (快速下降后趋平)
- 0 skipped samples
- wandb: [stage1-sft-qwen2vl-2b](https://wandb.ai/leixinlin-peking-university/VLM-Hallucination)

**GRPO Training** (300 steps, ~18min on A100-80G):
- Custom GRPO trainer with visual reward function
- wandb: [stage2a-grpo-full](https://wandb.ai/leixinlin-peking-university/VLM-Hallucination)

## 项目结构

```
vlm-hallucination-project/
├── README.md                    # 本文件
├── EXPERIMENT_LOG.md            # 详细实验日志
├── .gitignore
├── scripts/
│   ├── prepare_data.py          # 数据下载与预处理
│   ├── prepare_sft_from_dpo.py  # 从 RLAIF-V 构造 SFT 数据
│   ├── prepare_grpo_data.py     # GRPO 数据准备（含图像保存）
│   ├── train_sft.py             # Stage 1: SFT 训练
│   ├── train_grpo.py            # Stage 2a: GRPO 训练入口
│   ├── grpo_vlm_trainer.py      # 自定义 GRPO VLM 训练器（核心）
│   ├── train_dpo.py             # Stage 2b: DPO 训练
│   ├── evaluate.py              # POPE 评测脚本
│   └── demo_gradio.py           # Gradio 交互演示
├── configs/                     # 训练配置
├── data/
│   ├── sft/                     # SFT 数据 (JSON)
│   ├── grpo/                    # GRPO 数据 (JSON + images)
│   ├── dpo/                     # DPO 偏好数据 (JSON)
│   └── eval/                    # POPE 评测数据
├── checkpoints/                 # 模型权重 (gitignored)
├── logs/
│   └── eval_results/            # 评测结果 JSON
└── wandb/                       # W&B 日志 (gitignored)
```

## 环境配置

### 依赖
```bash
conda create -n vlm-hal python=3.11 -y
conda activate vlm-hal
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.46.3 accelerate==1.1.1 peft==0.13.2 trl==0.12.2 \
    datasets==3.1.0 bitsandbytes==0.44.1 wandb gradio qwen-vl-utils \
    scipy scikit-learn matplotlib seaborn huggingface_hub
pip install flash-attn --no-build-isolation
```

### 硬件要求
- GPU: 至少 1× A100-40G（SFT/DPO），推荐 4× A100-80G（并行训练+评测）
- 磁盘: ~60GB（模型 + 数据 + COCO 图像）

## 复现步骤

```bash
# 1. 下载模型和数据
python scripts/prepare_data.py
# 需要手动下载 COCO train2017 和 val2014 图像

# 2. 从 RLAIF-V 构造 SFT 数据
python scripts/prepare_sft_from_dpo.py

# 3. Stage 1: SFT
python -u scripts/train_sft.py --lora_r 32 --lora_alpha 64 --epochs 1 \
    --grad_accum 16 --lr 2e-4 --use_wandb

# 4. 正确 merge LoRA
# (训练脚本自动 merge，或手动运行 merge 代码)

# 5. Stage 2a: GRPO
python -u scripts/train_grpo.py --model_path checkpoints/sft-qwen2vl-2b-merged \
    --num_generations 4 --max_steps 300 --reward_mode full --use_wandb

# 6. Stage 2b: DPO
python -u scripts/train_dpo.py --model_path checkpoints/sft-qwen2vl-2b-merged \
    --beta 0.1 --epochs 1 --use_wandb

# 7. 评测
python -u scripts/evaluate.py --model_path checkpoints/Qwen2-VL-2B-Instruct \
    --model_name base --image_dir data/val2014 --max_samples 1000
```

## 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| flash_attention_2 + QLoRA 报错 | 量化后中间计算用 float32 | 改用 sdpa 或不指定 attn |
| Image features/tokens 不匹配 | batch>1 时 vision tokens 数不同 | batch_size=1 + grad_accum |
| conda run 日志被缓冲 | conda run 内部管道缓冲 | 直接用 Python binary + `-u` |
| Qwen2.5-VL-2B 不存在 | HF 最小为 3B | 改用 Qwen2-VL-2B-Instruct |
| LLaVA-Instruct-150K 加载失败 | pyarrow 列类型推断 | 直接下载 JSON |
| SFT merge 后模型只有 1.9G | QLoRA merge 在量化模型上不完整 | 先 bf16 加载 base → 叠加 LoRA → merge |
| TRL DPOTrainer VLM 不兼容 | processor.tokenizer 属性缺失 | 自定义 DPO 训练循环 |
| GRPO: TRL GRPOConfig 不存在 | trl==0.12.2 版本无此类 | 自定义 dataclass config |
| A-OKVQA image_id 为空 | HF 数据集中图像是 PIL 对象 | 保存图像到本地并记录路径 |

## 参考文献

- [RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback](https://arxiv.org/abs/2312.00849)
- [RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness](https://arxiv.org/abs/2405.17220)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [POPE: Polling-based Object Probing Evaluation for Object Hallucination](https://arxiv.org/abs/2305.10355)
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)

## License

MIT
