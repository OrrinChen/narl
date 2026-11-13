"""Plot the actor and learner speedup curves from a prefix length sweep."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads((args.run / "summary.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    prompts = sorted({row["prompt_tokens"] for row in rows})
    lengths = sorted({row["output_tokens"] for row in rows})
    table = {(row["prompt_tokens"], row["output_tokens"]): row for row in rows}

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for axis, (phase, title) in zip(axes, (("actor", "Actor prefill + fixed-token decode"),
                                        ("learner", "Learner scoring + backward"))):
        for length in lengths:
            axis.plot(prompts, [table[prompt, length][f"{phase}_speedup"] for prompt in prompts],
                      marker="o", markersize=4, label=f"{length} output tokens")
        axis.axhline(1, color="0.4", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(prompts, [str(prompt) for prompt in prompts])
        axis.set_title(title)
        axis.set_xlabel("Prompt length (tokens)")
        axis.set_ylabel("Dense / shared latency (×)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper left", frameon=False)
    figure.suptitle(f"Shared prompt scaling on {protocol['device'].upper()} · G={protocol['args']['group_size']}")
    figure.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
