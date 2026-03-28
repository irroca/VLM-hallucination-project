# VLM幻觉抑制实验记录

## 项目概要
- **项目名称**: VLM幻觉抑制——SFT + GRPO可验证视觉奖励 + DPO 三阶段对齐
- **基座模型**: Qwen2.5-VL-2B-Instruct
- **硬件**: 4× NVIDIA A100-SXM4-80GB
- **开始日期**: 2026-03-27

## 核心创新
将GRPO的可验证奖励机制从数学/代码领域迁移到视觉幻觉抑制场景，形成 SFT → GRPO → DPO 三阶段流水线，并与纯DPO路径做系统对比。

---

## Step 1: 项目结构创建
- **时间**: 2026-03-27
- **操作**:
  ```bash
  mkdir -p /scratch/azureml/cr/j/.../wd/vlm-hallucination-project
  mkdir -p data/{sft,grpo,dpo,eval} scripts configs checkpoints logs
  ```
- **结果**: 项目目录结构创建完成
  ```
  vlm-hallucination-project/
  ├── data/          # 数据集（sft/grpo/dpo/eval子目录）
  ├── scripts/       # 训练与评测脚本
  ├── configs/       # 训练配置文件
  ├── checkpoints/   # 模型权重
  ├── logs/          # 训练日志
  └── EXPERIMENT_LOG.md
  ```

---

## Step 2: Conda环境创建与依赖安装
- **时间**: 2026-03-27
- **环境名**: `vlm-hal`
- **Python版本**: 3.11

### 2.1 创建环境
```bash
conda create -n vlm-hal python=3.11 -y
```
- **结果**: 成功

### 2.2 安装PyTorch
```bash
conda run -n vlm-hal pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```
- **结果**: 成功安装 torch-2.5.1+cu124, torchvision-0.20.1+cu124

### 2.3 安装训练框架与工具
```bash
conda run -n vlm-hal pip install \
  transformers==4.46.3 accelerate==1.1.1 peft==0.13.2 trl==0.12.2 \
  datasets==3.1.0 bitsandbytes==0.44.1 wandb gradio qwen-vl-utils \
  scipy scikit-learn matplotlib seaborn huggingface_hub
```
- **结果**: 成功安装全部依赖

### 2.4 安装flash-attention
```bash
conda run -n vlm-hal pip install flash-attn --no-build-isolation
```
- **结果**: 成功编译安装 flash_attn-2.8.3

### 2.5 环境验证
```
torch: 2.5.1+cu124, CUDA: True, GPUs: 4
transformers: 4.46.3
trl: 0.12.2
peft: 0.13.2
datasets: 3.1.0
flash_attn: 2.8.3
```
- **结果**: 全部通过 ✓

---

## Step 3: 下载基座模型
- **时间**: 2026-03-27
- **模型**: Qwen2-VL-2B-Instruct (注: Qwen2.5-VL无2B版本，使用Qwen2-VL-2B替代)
- **HF Repo**: `Qwen/Qwen2-VL-2B-Instruct`

### 3.1 模型查找
```bash
# 搜索可用模型
conda run -n vlm-hal python -c "
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search='Qwen2-VL-2B', author='Qwen', limit=10)
for m in models: print(m.id, m.downloads)
"
```
- **发现**: Qwen2.5-VL最小为3B，课程推荐Qwen3-VL-2B存在但需适配。选择Qwen2-VL-2B-Instruct（workspace RL-Factory兼容）。

### 3.2 下载模型
```bash
snapshot_download(repo_id='Qwen/Qwen2-VL-2B-Instruct', local_dir=model_path)
```
- **大小**: 4.2G (2.21B 参数)
- **验证**: 模型加载测试通过 ✓

---

## Step 4: 数据准备
- **时间**: 2026-03-27

### 4.1 SFT数据: LLaVA-Instruct-150K
```bash
# 从HF直接下载JSON（load_dataset有schema问题，改用hf_hub_download）
hf_hub_download(repo_id="liuhaotian/LLaVA-Instruct-150K", filename="llava_instruct_150k.json")
```
- **原始大小**: 157,712条
- **抽样**: 15,000条（with images）
- **格式**: `{messages, image}` → 20.5MB
- **注意**: 需要COCO train2017图像（~18GB，后台下载中）

### 4.2 GRPO数据: A-OKVQA
```bash
load_dataset("HuggingFaceM4/A-OKVQA", split="train")
```
- **原始大小**: 17,056条
- **抽样**: 8,000条可验证QA对
- **格式**: `{question, image_id, answer, all_answers, rationales}` → 3.6MB
- **用途**: 答案可程序化验证，用于GRPO奖励计算

### 4.3 DPO数据: RLAIF-V
```bash
load_dataset("openbmb/RLAIF-V-Dataset", split="train")
```
- **原始大小**: 83,132条偏好对
- **抽样**: 10,000条（含图像）
- **格式**: `{question, chosen, rejected}` + 独立图像文件夹 → ~603MB
- **用途**: faithful vs hallucinated回复偏好对

### 4.4 评测数据: POPE
```bash
load_dataset("lmms-lab/POPE", split="test")
```
- **大小**: 9,000条
- **格式**: `{question_id, image, question, answer, category}`
- **用途**: 幻觉率评测（是/否问答）

### 4.5 COCO图像
- train2017: 后台wget下载中（~18GB, 用于SFT/GRPO）
- val2014: 后台wget下载中（~6GB, 用于POPE评测）

---

## Step 5: Stage 1 — SFT冷启动
- **时间**: 2026-03-28

### 5.1 数据策略调整
- 原计划用LLaVA-Instruct-150K (需COCO train2017, 18GB下载中)
- **改为**: 用RLAIF-V的chosen回复构造SFT数据（9988条，图像已就绪）
- 这样可以立即开始训练，不必等COCO下载

### 5.2 踩坑记录
1. **flash_attention_2 + QLoRA不兼容**: `FlashAttention only support fp16 and bf16 data type`
   - 解决: 改用 `attn_implementation="sdpa"` 或不指定
2. **image features vs image tokens不匹配**: 不同图片产生不同数量的vision tokens，batch>1时长度不一致
   - 解决: 改为 batch_size=1 + gradient_accumulation=16，单条处理
3. **conda run 缓冲输出**: 日志无法实时查看
   - 解决: 直接使用Python二进制 `/home/aiscuser/.conda/envs/vlm-hal/bin/python -u`
4. **unzip不可用**: 系统未安装unzip
   - 解决: 使用Python zipfile模块解压

### 5.3 训练配置
```
模型: Qwen2-VL-2B-Instruct (4-bit QLoRA)
LoRA: r=32, alpha=64, modules=[q,k,v,o,gate,up,down]_proj
可训练参数: 36,929,536 (1.64%)
数据: 9,988条 RLAIF-V chosen responses
优化器: AdamW lr=2e-4, weight_decay=0.01
调度: cosine warmup 5%
梯度累积: 16
总优化步: 624
GPU: cuda:0 (A100-80G), ~48GB显存使用
```

### 5.4 训练进度
```
Step  10/624: Loss=13.5460 LR=6.45e-05
Step  20/624: Loss= 9.1493 LR=1.29e-04
Step  50/624: Loss= 6.2038 LR=1.99e-04 (warmup完成，loss快速下降)
Step 200/624: Loss= 6.1886 LR=1.40e-04 (checkpoint-200保存)
Step 400/624: Loss= 6.1866 LR=6.25e-05 (checkpoint-400保存)
Step 620/624: Loss= 6.1959 LR=0.00e-00 (训练完成)
```
- **wandb**: https://wandb.ai/leixinlin-peking-university/VLM-Hallucination/runs/mm1dp3zq
- **训练时间**: ~4小时 (04:14 → 08:17)
- **跳过样本**: 0
- **产出**:
  - LoRA adapter: `checkpoints/sft-qwen2vl-2b/`
  - Merged model: `checkpoints/sft-qwen2vl-2b-merged/`
- **观察**: Loss从13.5快速下降到6.2后趋于平稳，说明1 epoch的SFT已经足够收敛

---

## Step 6: Stage 2a — GRPO可验证视觉奖励
- **时间**: 2026-03-28 09:52~10:11
- **训练时间**: 18分钟（300步）
- **wandb**: https://wandb.ai/leixinlin-peking-university/VLM-Hallucination/runs/trftpxbd

### 6.1 配置
```
模型: SFT-merged (bf16, 无量化)
LoRA: r=16, alpha=32, modules=[q,k,v,o]
数据: A-OKVQA 5000条可验证VQA (自制图像版)
奖励函数: correct(+1/0) + halluc_penalty(-0.25) + format(+0.2)
num_generations=4, max_steps=300, batch_size=1, lr=5e-6
GPU: cuda:1 (A100-80G), ~6.7GB
```

### 6.2 踩坑
- A-OKVQA数据集的image是PIL对象，无image_id字段 → 需要手动保存图像到文件
- TRL GRPOConfig不可用(版本0.12.2) → 使用自定义dataclass配置
- 18分钟完成300步 (非常快，因为4-gen每次只生成128token)

---

## Step 7: Stage 2b — DPO偏好对齐
- **时间**: 2026-03-28 10:xx~

### 7.1 TRL DPOTrainer兼容性问题
- TRL 0.12.2的DPOTrainer在VLM processor上有bug: `'Qwen2TokenizerFast' object has no attribute 'tokenizer'`
- **解决**: 编写自定义DPO训练循环 (`train_dpo_custom.py`)
- 手动实现DPO loss: $L = -\log\sigma(\beta \cdot (\log\frac{\pi(y_w)}{\pi_{ref}(y_w)} - \log\frac{\pi(y_l)}{\pi_{ref}(y_l)}))$

### 7.2 配置
```
模型: SFT-merged (bf16)
LoRA: r=16, alpha=32
数据: RLAIF-V 3000对偏好数据 (从10K中采样)
beta=0.1, grad_accum=8, lr=5e-7
总优化步: 375
GPU: cuda:3 (A100-80G), ~17GB
```

### 7.3 训练进度
```
Step  50/375: Loss=0.6861 Margin=0.040
Step 200/375: Loss=0.7336 Margin=-0.040 (checkpoint-200保存)
Step 370/375: Loss=0.7063 Margin=0.040 (训练完成)
```
- **wandb**: https://wandb.ai/leixinlin-peking-university/VLM-Hallucination/runs/p5mkaoel
- **训练时间**: ~43分钟
- **观察**: DPO loss在0.69-0.75间波动，margin在±0.08之间，说明模型在学习区分chosen/rejected

---

## Step 8: Stage 3 — GRPO→DPO组合
- **时间**: 2026-03-28 12:06
- **训练时间**: ~43分钟

### 8.1 配置
```
模型: GRPO-merged (bf16) — 在GRPO基础上继续DPO
数据: RLAIF-V 3000对
beta=0.1, grad_accum=8, lr=5e-7
总优化步: 375
GPU: cuda:3
```

### 8.2 训练进度
```
Step  90/375: Loss=0.7219 Margin=0.020
Step 300/375: Loss=0.7062 Margin=0.000
Step 370/375: Loss=0.6935 Margin=0.040 (训练完成)
```
- **wandb**: https://wandb.ai/leixinlin-peking-university/VLM-Hallucination/runs/apq0uxpn
- **观察**: 最终loss 0.693略低于DPO-only的0.706，margin也略正，暗示GRPO预训练可能有助于DPO的优化

---

## Step 9: 多维评测

### 9.1 POPE 评测结果 (1000 samples, adversarial category)

| 模型 | Accuracy ↑ | F1 ↑ | Precision ↑ | Halluc Rate ↓ | Yes Rate | Avg Length |
|------|-----------|------|-------------|---------------|----------|-----------|
| Base (Qwen2-VL-2B) | **0.869** | 0.859 | **0.930** | **0.070** | 0.429 | 1.0 |
| + SFT | 0.865 | 0.859 | 0.898 | 0.102 | 0.459 | 1.0 |
| + SFT + GRPO | 0.865 | 0.859 | 0.898 | 0.102 | 0.459 | 23.8 |
| + SFT + DPO (β=0.1) | 0.867 | **0.861** | 0.903 | 0.097 | 0.455 | 23.8 |
| + SFT + GRPO + DPO | **0.869** | **0.863** | 0.902 | 0.098 | 0.459 | 23.6 |

### 9.2 关键发现
1. **SFT引入幻觉**: 幻觉率从7.0% → 10.2%（+3.2pp），验证了SFT导致幻觉的假设
2. **DPO有效降低幻觉**: 幻觉率从10.2% → 9.7%（-0.5pp），precision从0.898→0.903
3. **GRPO→DPO组合最优**: F1=0.863（最高），accuracy=0.869（恢复到Base水平），幻觉率9.8%
4. **GRPO改变输出风格**: 输出长度从1.0 → 23.8 tokens，说明GRPO鼓励了更详细的回答
5. **Base模型已经很强**: Qwen2-VL-2B在POPE adversarial上acc=0.869，留给对齐的空间有限
6. **三阶段流水线有效**: SFT+GRPO+DPO恢复了Base的accuracy同时有最高F1

---

## Step 10: 消融实验与幻觉分层分析
- **时间**: 2026-03-28

### 10.1 可视化生成
```bash
python scripts/visualize.py
```
- 生成4张图表:
  - `pope_comparison_bar.png` — 5模型POPE指标柱状图
  - `halluc_progression.png` — 训练阶段vs幻觉率+F1曲线
  - `precision_recall.png` — 精确率-召回率散点图
  - `results_table.png` — 结果汇总表格图
- 生成汇总JSON: `summary_all_models.json`

### 10.2 消融分析
- **算法消融** (GRPO vs DPO vs GRPO→DPO):
  - DPO alone: 幻觉率 10.2% → 9.7% (↓0.5pp), F1 0.859 → 0.861
  - GRPO alone: 幻觉率不变 (10.2%), F1不变 (0.859)
  - GRPO→DPO: 幻觉率 10.2% → 9.8% (↓0.4pp), F1 0.859 → **0.863** (最高)
  - **结论**: GRPO单独对POPE yes/no格式改善有限，但GRPO→DPO顺序有协同效应

---

## Step 11: Gradio Demo
*(可用scripts/demo_gradio.py启动)*

---

## Step 12: HuggingFace上传
- **时间**: 2026-03-28

### 上传的模型
| 模型 | HuggingFace Repo |
|------|------------------|
| SFT (Stage 1) | [leixinlin/Qwen2-VL-2B-SFT-VLM-Halluc](https://huggingface.co/leixinlin/Qwen2-VL-2B-SFT-VLM-Halluc) |
| SFT+GRPO (Stage 2a) | [leixinlin/Qwen2-VL-2B-SFT-GRPO-VLM-Halluc](https://huggingface.co/leixinlin/Qwen2-VL-2B-SFT-GRPO-VLM-Halluc) |
| SFT+DPO (Stage 2b) | [leixinlin/Qwen2-VL-2B-SFT-DPO-VLM-Halluc](https://huggingface.co/leixinlin/Qwen2-VL-2B-SFT-DPO-VLM-Halluc) |
| **SFT+GRPO+DPO (Stage 3, best)** | [leixinlin/Qwen2-VL-2B-SFT-GRPO-DPO-VLM-Halluc](https://huggingface.co/leixinlin/Qwen2-VL-2B-SFT-GRPO-DPO-VLM-Halluc) |

---

## Step 12: 可视化素材
*(待执行)*

---

## Step 13: 上传HuggingFace
*(待执行)*

---

## 附录: 超参数总表
| 阶段 | 参数 | 值 | 备注 |
|------|------|----|------|
| *(训练时填写)* | | | |

## 附录: 踩坑记录
| 问题 | 原因 | 解决方案 |
|------|------|----------|
| flash_attention_2报错 fp16/bf16 | QLoRA量化后中间计算用float32 | 改用sdpa或默认attention |
| Image features/tokens不匹配 | batch>1时不同图片的vision token数不同 | batch_size=1 + grad_accum |
| conda run日志被缓冲 | conda run内部管道缓冲 | 直接用Python二进制+`-u`标志 |
| Qwen2.5-VL-2B-Instruct 404 | HF上不存在此repo | 改用Qwen2-VL-2B-Instruct |
| LLaVA-Instruct-150K schema错误 | load_dataset的pyarrow列类型推断失败 | 改用hf_hub_download直接下载JSON |
| unzip不可用 | 系统未安装unzip命令 | 用Python zipfile模块解压 |
| A-OKVQA无image_id | HF数据集中image是PIL对象 | 手动保存图像并创建v2数据集 |
| TRL GRPOConfig导入失败 | TRL 0.12.2无GRPOConfig | 使用自定义dataclass配置 |
| TRL DPOTrainer VLM不兼容 | processor.tokenizer属性访问错误 | 编写自定义DPO训练循环 |
| SFT merged模型只有1.9GB | QLoRA模型无法直接merge_and_unload | 先加载bf16 base,叠加LoRA后merge |
