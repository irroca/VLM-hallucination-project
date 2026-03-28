"""
Re-prepare GRPO data: save images from A-OKVQA with proper references.
"""
import os
import json
import random
from datasets import load_dataset
from collections import Counter

random.seed(42)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, "data", "grpo")
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

print("Loading A-OKVQA...")
ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
print(f"Full dataset: {len(ds)}")

grpo_data = []
for i, item in enumerate(ds):
    direct_answers = item.get("direct_answers", [])
    if not direct_answers:
        continue

    answer_counts = Counter(direct_answers)
    best_answer = answer_counts.most_common(1)[0][0]

    # Save image
    img_filename = f"aokvqa_{i:06d}.jpg"
    img_path = os.path.join(IMG_DIR, img_filename)
    if not os.path.exists(img_path):
        try:
            item["image"].save(img_path)
        except Exception:
            continue

    grpo_data.append({
        "question": item["question"],
        "image": img_path,  # absolute path
        "answer": best_answer,
        "all_answers": direct_answers,
        "rationales": item.get("rationales", []),
    })

# Sample 5K
if len(grpo_data) > 5000:
    grpo_data = random.sample(grpo_data, 5000)

out_file = os.path.join(OUT_DIR, "aokvqa_grpo_v2.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(grpo_data, f, ensure_ascii=False, indent=1)

print(f"Saved {len(grpo_data)} samples to {out_file}")
print(f"Images in {IMG_DIR}: {len(os.listdir(IMG_DIR))}")
