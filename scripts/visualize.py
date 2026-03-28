"""
Hallucination Analysis & Visualization
- Per-model hallucination breakdown
- Radar chart of hallucination categories  
- Training curves comparison
- Side-by-side model comparison
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "logs", "eval_results")
VIZ_DIR = os.path.join(PROJECT_DIR, "logs", "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# ─── Load all results ───
def load_results():
    models = {
        "Base": "pope_base-bf16.json",
        "SFT": "pope_sft-merged-bf16.json",
        "SFT+GRPO": "pope_grpo-merged-bf16.json",
        "SFT+DPO": "pope_dpo-beta0.1-merged.json",
        "SFT+GRPO+DPO": "pope_grpo-dpo-merged.json",
    }
    results = {}
    for name, fname in models.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            results[name] = data.get("pope", data)
    return results


def plot_comparison_bar(results):
    """Bar chart comparing all models on key metrics."""
    models = list(results.keys())
    metrics = ["accuracy", "f1", "precision", "hallucination_rate"]
    labels = ["Accuracy", "F1", "Precision", "Halluc Rate"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    for ax, metric, label in zip(axes, metrics, labels):
        values = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(range(len(models)), values, color=colors[:len(models)], alpha=0.85)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle("POPE Evaluation: 5-Model Comparison (1000 adversarial samples)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "pope_comparison_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pope_comparison_bar.png")


def plot_hallucination_progression(results):
    """Line chart showing hallucination rate progression through training stages."""
    stages = list(results.keys())
    halluc_rates = [results[s].get("hallucination_rate", 0) for s in stages]
    f1_scores = [results[s].get("f1", 0) for s in stages]
    acc_scores = [results[s].get("accuracy", 0) for s in stages]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(stages))

    # Hallucination rate (left axis)
    color1 = '#E91E63'
    ax1.plot(x, halluc_rates, 'o-', color=color1, linewidth=2, markersize=8, label='Halluc Rate ↓')
    ax1.set_ylabel('Hallucination Rate (↓ better)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(halluc_rates) * 1.5)

    # F1 (right axis)
    ax2 = ax1.twinx()
    color2 = '#2196F3'
    ax2.plot(x, f1_scores, 's-', color=color2, linewidth=2, markersize=8, label='F1 ↑')
    ax2.plot(x, acc_scores, '^-', color='#4CAF50', linewidth=2, markersize=8, label='Accuracy ↑')
    ax2.set_ylabel('F1 / Accuracy (↑ better)', color=color2, fontsize=12)
    ax2.set_ylim(min(f1_scores) * 0.95, max(acc_scores) * 1.02)

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=20, ha='right')
    ax1.set_xlabel('Training Stage', fontsize=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
               fontsize=10, ncol=3, bbox_to_anchor=(0.5, 1.12))

    plt.title("Hallucination Rate vs Performance Across Training Stages",
              fontsize=13, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "halluc_progression.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: halluc_progression.png")


def plot_precision_recall_tradeoff(results):
    """Scatter plot of precision vs recall for each model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    for i, (name, data) in enumerate(results.items()):
        p = data.get("precision", 0)
        r = data.get("recall", 0)
        ax.scatter(r, p, s=150, c=colors[i], label=name, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(name, (r, p), textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Trade-off", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "precision_recall.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: precision_recall.png")


def plot_results_table(results):
    """Generate a publication-quality results table as image."""
    models = list(results.keys())
    metrics = ["accuracy", "f1", "precision", "recall", "hallucination_rate", "yes_rate", "avg_response_length"]
    headers = ["Model", "Acc ↑", "F1 ↑", "Prec ↑", "Recall ↑", "Halluc ↓", "Yes%", "Avg Len"]

    cell_data = []
    for m in models:
        row = [m]
        for metric in metrics:
            val = results[m].get(metric, 0)
            row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        cell_data.append(row)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis('off')
    table = ax.table(cellText=cell_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colColours=['#E3F2FD'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Highlight best values
    for j, metric in enumerate(metrics, start=1):
        vals = [results[m].get(metric, 0) for m in models]
        if "halluc" in metric:
            best_idx = vals.index(min(vals))
        else:
            best_idx = vals.index(max(vals))
        table[best_idx + 1, j].set_facecolor('#C8E6C9')

    plt.title("POPE Evaluation Results Summary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "results_table.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results_table.png")


def generate_summary_json(results):
    """Save a clean summary JSON for downstream use."""
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "accuracy": data.get("accuracy"),
            "f1": data.get("f1"),
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "hallucination_rate": data.get("hallucination_rate"),
            "yes_rate": data.get("yes_rate"),
            "refusal_rate": data.get("refusal_rate"),
            "avg_response_length": data.get("avg_response_length"),
            "avg_latency_ms": data.get("avg_latency_ms"),
        }

    out_path = os.path.join(RESULTS_DIR, "summary_all_models.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded results for {len(results)} models: {list(results.keys())}")

    plot_comparison_bar(results)
    plot_hallucination_progression(results)
    plot_precision_recall_tradeoff(results)
    plot_results_table(results)
    generate_summary_json(results)

    print(f"\nAll visualizations saved to {VIZ_DIR}")
