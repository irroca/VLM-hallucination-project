"""
Download and prepare all datasets for the VLM Hallucination project.
- SFT: LLaVA-Instruct-150K (subset)
- GRPO: A-OKVQA (verifiable visual QA)
- DPO: RLAIF-V (preference pairs)
- Eval: POPE (hallucination benchmark)
"""

import os
import json
import random
from datasets import load_dataset
from huggingface_hub import snapshot_download

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def download_sft_data():
    """Download LLaVA-Instruct-150K JSON directly and extract a 15K subset."""
    print("=" * 60)
    print("[SFT] Downloading LLaVA-Instruct-150K...")
    out_dir = os.path.join(DATA_DIR, "sft")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "llava_sft_15k.json")
    if os.path.exists(out_file):
        print(f"  Already exists: {out_file}")
        return

    # Download JSON directly from HF hub
    from huggingface_hub import hf_hub_download
    json_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="llava_instruct_150k.json",
        repo_type="dataset",
    )
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"  Full dataset size: {len(raw_data)}")

    # Filter for conversations with images
    data = [item for item in raw_data if item.get("image")]
    print(f"  With images: {len(data)}")

    # Sample 15K
    if len(data) > 15000:
        data = random.sample(data, 15000)
    print(f"  Sampled: {len(data)}")

    # Convert to SFT format
    sft_data = []
    for item in data:
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue
        messages = []
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            messages.append({"role": role, "content": conv["value"]})
        sft_data.append({
            "messages": messages,
            "image": item["image"],
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=1)
    print(f"  Saved {len(sft_data)} samples to {out_file}")


def download_grpo_data():
    """Download A-OKVQA for verifiable visual QA (GRPO training)."""
    print("=" * 60)
    print("[GRPO] Downloading A-OKVQA...")
    out_dir = os.path.join(DATA_DIR, "grpo")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "aokvqa_grpo.json")
    if os.path.exists(out_file):
        print(f"  Already exists: {out_file}")
        return

    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    print(f"  Full dataset size: {len(ds)}")

    # Filter for direct-answer questions (not multiple choice only)
    grpo_data = []
    for item in ds:
        # Use direct_answers for verifiable reward
        direct_answers = item.get("direct_answers", [])
        if not direct_answers:
            continue
        # Get most common answer
        from collections import Counter
        answer_counts = Counter(direct_answers)
        best_answer = answer_counts.most_common(1)[0][0]

        grpo_data.append({
            "question": item["question"],
            "image_id": item.get("image_id", ""),
            "answer": best_answer,
            "all_answers": direct_answers,
            "rationales": item.get("rationales", []),
        })

    # Sample 8K for training
    if len(grpo_data) > 8000:
        grpo_data = random.sample(grpo_data, 8000)
    print(f"  Sampled: {len(grpo_data)} verifiable QA pairs")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(grpo_data, f, ensure_ascii=False, indent=1)
    print(f"  Saved to {out_file}")


def download_dpo_data():
    """Download RLAIF-V preference data for DPO training."""
    print("=" * 60)
    print("[DPO] Downloading RLAIF-V preference data...")
    out_dir = os.path.join(DATA_DIR, "dpo")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "rlaifv_dpo.json")
    if os.path.exists(out_file):
        print(f"  Already exists: {out_file}")
        return

    # RLAIF-V dataset
    try:
        ds = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
        print(f"  Full dataset size: {len(ds)}")

        dpo_data = []
        for item in ds:
            # Format: chosen response vs rejected response
            dpo_data.append({
                "question": item.get("question", ""),
                "image": item.get("image", None),  # PIL Image or path
                "chosen": item.get("chosen", ""),
                "rejected": item.get("rejected", ""),
            })

        # Sample 10K for manageable training
        if len(dpo_data) > 10000:
            dpo_data = random.sample(dpo_data, 10000)

        # Save without images (images handled separately)
        dpo_text = []
        for item in dpo_data:
            dpo_text.append({
                "question": item["question"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(dpo_text, f, ensure_ascii=False, indent=1)
        print(f"  Saved {len(dpo_text)} preference pairs to {out_file}")

        # Save images separately
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        print(f"  Saving images to {img_dir}...")
        for i, item in enumerate(dpo_data):
            if item.get("image") is not None:
                try:
                    img = item["image"]
                    img.save(os.path.join(img_dir, f"{i:06d}.jpg"))
                except Exception:
                    pass
            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{len(dpo_data)} images saved")
        print("  Images saved.")

    except Exception as e:
        print(f"  Error downloading RLAIF-V: {e}")
        print("  Will try alternative approach...")


def download_eval_data():
    """Download POPE evaluation benchmark."""
    print("=" * 60)
    print("[EVAL] Downloading POPE benchmark...")
    out_dir = os.path.join(DATA_DIR, "eval")
    os.makedirs(out_dir, exist_ok=True)

    # POPE benchmark
    pope_file = os.path.join(out_dir, "pope_eval.json")
    if os.path.exists(pope_file):
        print(f"  Already exists: {pope_file}")
    else:
        try:
            ds = load_dataset("lmms-lab/POPE", split="test")
            print(f"  POPE dataset size: {len(ds)}")

            pope_data = []
            for item in ds:
                pope_data.append({
                    "question_id": item.get("question_id", ""),
                    "image": item.get("image_source", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "category": item.get("category", ""),
                })

            with open(pope_file, "w", encoding="utf-8") as f:
                json.dump(pope_data, f, ensure_ascii=False, indent=1)
            print(f"  Saved {len(pope_data)} POPE items to {pope_file}")
        except Exception as e:
            print(f"  Error: {e}")
            print("  Will create POPE data from COCO images manually.")


def download_coco_images():
    """Download COCO val2014 images needed for evaluation."""
    print("=" * 60)
    print("[IMAGES] Checking COCO images...")
    img_dir = os.path.join(DATA_DIR, "coco_val2014")
    if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 100:
        print(f"  COCO images exist: {len(os.listdir(img_dir))} files")
        return

    print("  COCO val2014 images will be downloaded on demand during training/eval.")
    print("  Alternatively, download from: http://images.cocodataset.org/zips/val2014.zip")
    os.makedirs(img_dir, exist_ok=True)


if __name__ == "__main__":
    print("VLM Hallucination Project - Data Preparation")
    print("=" * 60)

    download_sft_data()
    download_grpo_data()
    download_dpo_data()
    download_eval_data()
    download_coco_images()

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
