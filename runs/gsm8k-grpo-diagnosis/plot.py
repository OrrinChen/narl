"""Render the fixed-trajectory development localization; no training is run."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


run = Path(__file__).resolve().parent
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
for case, color, marker, label in (
    ("replay-high", "#C64538", "o", "LR 6e-4 (original)"),
    ("replay-low", "#2166AC", "s", "LR 1e-4 (off-policy replay)"),
):
    rows = json.loads((run / case / "summary.json").read_text())["checkpoints"]
    steps = sorted(map(int, rows))
    axes[0].plot(steps, [rows[str(s)]["metrics"]["correct"] for s in steps],
                 marker=marker, color=color, label=label)
    axes[1].plot(steps, [rows[str(s)]["metrics"]["median_tokens"] for s in steps],
                 marker=marker, color=color, label=label)
for ax in axes:
    ax.set_xlabel("Original effective-update index")
    ax.set_xticks([0, 6, 10, 12, 14, 16])
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Development correct / 24")
axes[0].set_ylim(0, 24)
axes[0].legend(fontsize=8, loc="upper right")
axes[1].set_ylabel("Median generated completion tokens")
axes[1].set_ylim(bottom=0)
fig.suptitle("Same stored rollout data; only learning rate changes", fontsize=12)
fig.savefig(run / "replay-localization.png", dpi=180)
fig.savefig(run / "replay-localization.svg")
svg = run / "replay-localization.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
