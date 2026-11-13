"""Plot recovery latency and retained-cache KL from a saved Narl replay summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

MODES = {
    "retain": ("Retain old KV", "#a64c24", "o"),
    "recompute": ("Recompute each completion", "#244b75", "s"),
    "share_prompt": ("Recompute with shared prompt", "#21795b", "^"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="figure path, e.g. .png or .svg")
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    cases = data["cases"]
    lengths = [case["prompt_length"] for case in cases]
    groups = {case["group_size"] for case in cases}
    suffixes = {case["suffix_length"] for case in cases}
    figure, (latency, divergence) = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")

    for mode, (label, color, marker) in MODES.items():
        times = [case["modes"][mode]["timing_ms"]["first_token_ms"] for case in cases]
        medians = [time["median"] for time in times]
        low = [time["median"] - time["min"] for time in times]
        high = [time["max"] - time["median"] for time in times]
        latency.errorbar(lengths, medians, yerr=(low, high), color=color, marker=marker,
                         capsize=3, linewidth=1.5, label=label)
    latency.set(xscale="log", yscale="log", xlabel="Prompt tokens",
                ylabel="Recovery to first logits (ms)", title="Weight-switch latency")
    latency.set_xticks(lengths, [str(length) for length in lengths])
    latency.legend(loc="upper left", frameon=False, fontsize=8)

    for case, marker in zip(cases, ("o", "s", "^")):
        series = case["modes"]["retain"]["vs_recompute"]["mean_kl_by_decode_step_nats"]
        divergence.plot(range(1, len(series) + 1), series, marker=marker, linewidth=1.5,
                        markersize=4, label=f"{case['prompt_length']} prompt tokens")
    divergence.set(xlabel="Decode step after weight switch", ylabel="Mean KL (nats)",
                   title="Retained KV vs fresh recomputation", ylim=(0, None))
    divergence.legend(loc="lower right", frameon=False, fontsize=8)
    for axis in (latency, divergence):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    if len(groups) == len(suffixes) == 1:
        detail = f" · G={next(iter(groups))} · cached suffix {next(iter(suffixes))} tokens"
    else:
        detail = ""
    figure.suptitle(
        f"{data['cpu']} · {data['threads']} thread{detail}\n"
        f"Latency median and range over {data['repetitions_per_mode']} repetitions per mode",
        fontsize=11,
    )
    figure.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
