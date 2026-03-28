"""
Gradio Demo: VLM Hallucination Suppression - Side-by-side Comparison
Three modes:
1. Three-model side-by-side (Base / SFT / Best Aligned)
2. Hallucination detection mode
3. Preset case replay
"""

import os
import torch
import gradio as gr
import logging
from typing import Tuple
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODELS = {
    "Base (Qwen2-VL-2B)": os.path.join(PROJECT_DIR, "checkpoints", "Qwen2-VL-2B-Instruct"),
    "SFT": os.path.join(PROJECT_DIR, "checkpoints", "sft-qwen2vl-2b-merged"),
    "SFT + GRPO": os.path.join(PROJECT_DIR, "checkpoints", "grpo-qwen2vl-2b"),
    "SFT + DPO": os.path.join(PROJECT_DIR, "checkpoints", "dpo-qwen2vl-2b"),
    "SFT + GRPO + DPO": os.path.join(PROJECT_DIR, "checkpoints", "grpo-dpo-qwen2vl-2b"),
}

# Global model cache
loaded_models = {}


def load_model_cached(model_key: str):
    """Load model with caching."""
    if model_key in loaded_models:
        return loaded_models[model_key]

    model_path = MODELS[model_key]
    if not os.path.exists(model_path):
        return None, None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    loaded_models[model_key] = (model, processor)
    return model, processor


def generate(model, processor, image, question, max_tokens=256):
    """Generate response from model."""
    if model is None:
        return "Model not available"

    # Save temp image
    temp_path = "/tmp/gradio_temp.jpg"
    if isinstance(image, str):
        temp_path = image
    else:
        image.save(temp_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{temp_path}"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, prompt_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def compare_three_models(image, question, model1_key, model2_key, model3_key):
    """Generate responses from three models side by side."""
    results = []
    for key in [model1_key, model2_key, model3_key]:
        model, processor = load_model_cached(key)
        if model is None:
            results.append(f"[{key}] Model not yet available")
        else:
            resp = generate(model, processor, image, question)
            results.append(resp)
    return results[0], results[1], results[2]


# ── Build Gradio UI ──
def build_demo():
    available_models = [k for k, v in MODELS.items() if os.path.exists(v)]
    if not available_models:
        available_models = list(MODELS.keys())

    with gr.Blocks(title="VLM幻觉抑制 - 三模型对比", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# VLM幻觉抑制：SFT + GRPO + DPO 三阶段对齐")
        gr.Markdown("对比基座模型、SFT模型和对齐模型在相同图片+问题上的回答差异。")

        with gr.Tab("三模型对比"):
            with gr.Row():
                image_input = gr.Image(type="pil", label="上传图片")
                question_input = gr.Textbox(label="问题", placeholder="描述这张图片中的内容...")

            with gr.Row():
                model1_select = gr.Dropdown(choices=available_models, value=available_models[0] if available_models else None, label="模型 1")
                model2_select = gr.Dropdown(choices=available_models, value=available_models[1] if len(available_models) > 1 else None, label="模型 2")
                model3_select = gr.Dropdown(choices=available_models, value=available_models[2] if len(available_models) > 2 else None, label="模型 3")

            compare_btn = gr.Button("生成对比", variant="primary")

            with gr.Row():
                output1 = gr.Textbox(label="模型 1 回答", lines=8)
                output2 = gr.Textbox(label="模型 2 回答", lines=8)
                output3 = gr.Textbox(label="模型 3 回答", lines=8)

            compare_btn.click(
                compare_three_models,
                inputs=[image_input, question_input, model1_select, model2_select, model3_select],
                outputs=[output1, output2, output3],
            )

        with gr.Tab("预设案例"):
            gr.Markdown("### 成功案例与失败案例展示")
            gr.Markdown("*(训练完成后将自动填充预设案例)*")

            example_btn = gr.Button("加载预设案例")
            example_output = gr.Textbox(label="案例分析", lines=10)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
