"""Rebuild the paired online figure from its summary and per-step evaluations."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "summary.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    styles = {
        "recompute": ("#43566a", "Per-completion rebuild", "--"),
        "share_prompt": ("#007468", "Shared-prompt rebuild", "-"),
    }
    for mode, (color, label, style) in styles.items():
        results = sorted((row for row in report["runs"] if row["mode"] == mode),
                         key=lambda row: row["seed"])
        for index, result in enumerate(results):
            trace = args.run / f"{result['seed']}-{mode}" / "eval.jsonl"
            points = [json.loads(line) for line in trace.read_text().splitlines()]
            reward = [point["reward"] for point in points]
            for axis, key in zip(axes[:2], ("time_s", "loss_tokens")):
                axis.plot([point[key] for point in points], reward, style, color=color,
                          alpha=0.72, label=label if index == 0 else None)
        axes[2].plot([row["seed"] for row in results], [row["loss_tokens_s"] for row in results],
                     marker="o", linestyle=style, color=color)
    for axis in axes[:2]:
        axis.set_ylabel("Held-out reward")
        axis.set_ylim(-0.04, 1.04)
    axes[0].set_xlabel("Wall time (s)")
    axes[1].set_xlabel("Training tokens with advantage")
    axes[2].set_ylabel("Training tokens with advantage / s")
    axes[2].set_xlabel("Paired seed")
    axes[2].set_xticks(report["protocol"]["seeds"])
    for axis in axes:
        axis.grid(alpha=0.23)
    axes[0].legend(frameon=False, loc="best")
    figure.savefig(args.out, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
