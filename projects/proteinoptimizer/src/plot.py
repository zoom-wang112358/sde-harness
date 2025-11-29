#!/usr/bin/env python3
"""
plot_proteinoptimizer_results.py

Reads ProteinOptimizer JSON result files with names like:
  results_single_<task>_0_<model>.json

Expected JSON keys:
  - best_score
  - best_scores_history

Outputs two figures using seaborn Set2 style:
  1) Bar plot of final Top 1 per model, averaged across tasks
  2) Convergence plot of Top 1 vs iteration per model, averaged across tasks

Usage:
  python plot_proteinoptimizer_results.py --input_dir /path/to/results --out_dir /path/to/figs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FNAME_RE = re.compile(r"results_single_(?P<task>[^_]+)_0_(?P<model>.+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--title_prefix", type=str, default="")
    return p.parse_args()


def model_display_name(model_key: str) -> str:
    mapping = {
        "baseline": "Baseline",
        "gpt-5-mini": "GPT5-mini",
        "deepseek-reasoner": "DeepSeek",
        "claude-sonnet-4-5": "Claude-Sonnet-4-5",
        "gpt-5": "GPT-5",
        "gpt-5-chat-latest": "GPT-5-chat-latest",
    }
    return mapping.get(model_key, model_key)


def load_records(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "results_single_*_0_*.json")))
    files = [f for f in files if "__MACOSX" not in f and not os.path.basename(f).startswith("._")]

    rows = []
    for fp in files:
        m = FNAME_RE.search(os.path.basename(fp))
        if not m:
            continue

        task = m.group("task")
        model_key = m.group("model")

        with open(fp, "r") as f:
            d = json.load(f)

        hist = d.get("best_scores_history")
        if hist is None:
            best = d.get("best_score")
            if best is None:
                continue
            hist = [float(best)]
        else:
            hist = [float(x) for x in hist]

        for it, v in enumerate(hist, start=1):
            rows.append(
                {
                    "task": task,
                    "model_key": model_key,
                    "model": model_display_name(model_key),
                    "iteration": it,
                    "top1": float(v),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid JSON files found in {input_dir}")
    return df


def set_style() -> None:
    sns.set_theme(style="white", context="talk")
    sns.set_palette("Set2")


def make_color_map(models_in_order: List[str]) -> Dict[str, tuple]:
    # Pull colors from seaborn Set2 in order and map to model names
    palette = sns.color_palette("Set2", n_colors=len(models_in_order))
    return {m: palette[i] for i, m in enumerate(models_in_order)}


def plot_bar(df: pd.DataFrame, out_dir: str, title_prefix: str = "") -> None:
    # Final Top 1 is the last iteration per task-model; then average across tasks
    last_it = df.groupby(["task", "model"])["iteration"].max().reset_index()
    df_last = df.merge(last_it, on=["task", "model", "iteration"], how="inner")

    # Stable order, fall back to sorted if partial
    preferred = [
        "Baseline",
        "GPT5-mini",
        "DeepSeek",
        "Claude-Sonnet-4-5",
        "GPT-5",
        "GPT-5-chat-latest",
    ]
    present = [m for m in preferred if m in df_last["model"].unique()]
    if not present:
        present = sorted(df_last["model"].unique().tolist())

    bar = df_last.groupby("model")["top1"].mean().reindex(present)
    cmap = make_color_map(present)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.grid(False)  # Remove background grid
    x = np.arange(len(present))
    vals = bar.to_numpy()

    bars = ax.bar(x, vals, color=[cmap[m] for m in present], alpha=0.88, width=0.78, align="center")

    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=10)
    ax.tick_params(axis="x", pad=14, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["bottom"].set_position(("outward", 10))

    ax.set_xlabel("Model", fontsize=11, labelpad=18)
    ax.set_ylabel("Top 1 Score", fontsize=11)

    title = "Top 1 Performance"
    if title_prefix.strip():
        title = f"{title_prefix.strip()} {title}"
    ax.set_title(title, fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylim(0, min(1.1, float(np.max(vals) * 1.10)))

    for rect, v in zip(bars, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.01,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top1_barplot.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "top1_barplot.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_convergence(df: pd.DataFrame, out_dir: str, title_prefix: str = "") -> None:
    preferred = [
        "Baseline",
        "GPT5-mini",
        "DeepSeek",
        "Claude-Sonnet-4-5",
        "GPT-5",
        "GPT-5-chat-latest",
    ]
    present = [m for m in preferred if m in df["model"].unique()]
    if not present:
        present = sorted(df["model"].unique().tolist())

    cmap = make_color_map(present)

    agg = df.groupby(["model", "iteration"])["top1"].mean().reset_index()
    max_it = int(df["iteration"].max())

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.grid(False)  # Remove background grid

    for m in present:
        sub = agg[agg["model"] == m].sort_values("iteration")
        ax.plot(
            sub["iteration"].to_numpy(),
            sub["top1"].to_numpy(),
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            alpha=0.95,
            color=cmap[m],
            label=m,
        )

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Top 1 Score", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    title = "Convergence of Top 1 Score"
    if title_prefix.strip():
        title = f"{title_prefix.strip()} {title}"
    ax.set_title(title, fontsize=14)

    ax.set_xticks(np.arange(1, max_it + 1))
    ax.set_xlim(0.7, max_it + 0.3)

    y_min = float(agg["top1"].min())
    y_max = float(agg["top1"].max())
    ax.set_ylim(max(0.0, y_min - 0.08), min(1.1, y_max + 0.08))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower right")

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top1_convergence.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "top1_convergence.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()
    df = load_records(args.input_dir)
    plot_bar(df, args.out_dir, title_prefix=args.title_prefix)
    plot_convergence(df, args.out_dir, title_prefix=args.title_prefix)
    print("Wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()

