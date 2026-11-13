"""Plot matched-null advantage and gradient comparisons from Narl regrouping records."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="directory containing summary.json and pairs.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="figure path, e.g. .png or .svg")
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    pairs = summary["pairs"]
    null, = [name for name, pair in pairs.items() if pair["same_weights"]]
    treatments = sorted((name for name in pairs if name != null),
                        key=lambda name: pairs[name]["lag_optimizer_steps"])
    observations = defaultdict(dict)
    with (args.run / "pairs.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["sample_seed"], row["grouping_seed"])
            observations[row["pair"]][key] = row
    for name in treatments:
        if observations[name].keys() != observations[null].keys():
            raise ValueError(f"Pair {name} lacks matched same-weight controls")

    figure, (advantage, gradient) = plt.subplots(1, 2, figsize=(11.6, 4.5), layout="constrained")
    figure.suptitle("GRPO regrouping with fixed completion records", fontsize=14, weight="semibold")
    seeds = summary["sampling_seeds"]
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^")

    for index, seed in enumerate(seeds):
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        offset = 0.16 * (index - (len(seeds) - 1) / 2)
        for position, name in enumerate(treatments):
            deltas = [row["mean_abs_advantage_change"] - observations[null][key]["mean_abs_advantage_change"]
                      for key, row in observations[name].items() if key[0] == seed]
            middle = median(deltas)
            advantage.errorbar(position + offset, middle, yerr=[[middle - min(deltas)], [max(deltas) - middle]],
                               color=color, marker=marker, capsize=2, markersize=4.5, linewidth=1.2,
                               label=f"Seed {seed}" if position == 0 else None)
        for position, name in enumerate([null, *treatments]):
            values = [row["gradient_cosine"] for key, row in observations[name].items()
                      if key[0] == seed and row["gradient_cosine"] is not None]
            if not values:
                continue
            x = position + offset
            gradient.scatter([x] * len(values), values, color=color, marker=marker, alpha=0.28, s=11)
            gradient.scatter(x, median(values), color=color, marker=marker, s=29, zorder=4,
                             edgecolors="white", linewidths=0.5)

    labels = [f"{pairs[name]['older_step']} → {pairs[name]['current_step']}\n"
              f"KL {mean(row['mean_kl_old_to_current_nats'] for row in pairs[name]['drift_by_seed']):.3f}"
              for name in treatments]
    advantage.axhline(0, color="#555555", linewidth=1, linestyle="--")
    advantage.set_xticks(range(len(treatments)), labels)
    advantage.set(title="Advantage change relative to same-weight control",
                  xlabel="Optimizer steps of source snapshots · mean KL in nats",
                  ylabel="Excess mean absolute advantage change")
    low, high = advantage.get_ylim()
    advantage.set_ylim(low, high + 0.12 * (high - low))
    advantage.legend(loc="upper left", frameon=False, fontsize=8, ncols=len(seeds))

    gradient.axvspan(-0.48, 0.48, color="#e8ebee", alpha=0.6, zorder=0)
    gradient.axhline(1, color="#555555", linewidth=1, linestyle="--")
    gradient.set_xticks(range(len(treatments) + 1), ["Same\nweights", *labels])
    gradient.set(title="Gradient alignment after regrouping",
                 xlabel="Optimizer steps of source snapshots · mean KL in nats",
                 ylabel="Cosine between homogeneous and mixed gradients")

    for axis in (advantage, gradient):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)

    repeats = summary["regroupings_per_sampling_seed"]
    figure.supxlabel(f"{len(seeds)} sampling seeds · {repeats} regroupings per seed · "
                    "large markers show medians, bars show ranges, faint points show regroupings",
                    fontsize=9)
    figure.savefig(args.out, dpi=180, metadata={"Date": None} if args.out.suffix == ".svg" else None)


if __name__ == "__main__":
    main()
