"""
Prepare SFT data from RLAIF-V chosen responses.
Uses the DPO dataset's 'chosen' responses as gold standard for SFT.
This avoids the need to wait for COCO train2017 download.
"""

import os
import json

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPO_DATA = os.path.join(PROJECT_DIR, "data", "dpo", "rlaifv_dpo.json")
DPO_IMAGES = os.path.join(PROJECT_DIR, "data", "dpo", "images")
OUT_FILE = os.path.join(PROJECT_DIR, "data", "sft", "rlaifv_sft.json")

with open(DPO_DATA, "r") as f:
    dpo_data = json.load(f)

sft_data = []
for i, item in enumerate(dpo_data):
    img_path = os.path.join(DPO_IMAGES, f"{i:06d}.jpg")
    if not os.path.exists(img_path):
        continue
    sft_data.append({
        "messages": [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["chosen"]},
        ],
        "image": img_path,  # absolute path to image
    })

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=1)

print(f"Created {len(sft_data)} SFT samples from RLAIF-V chosen responses")
print(f"Saved to {OUT_FILE}")
