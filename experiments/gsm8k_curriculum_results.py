"""Analyze the paired GSM8K curriculum comparison and its conditional dose extension."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.audit_gsm8k_accuracy import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from experiments.audit_gsm8k_stable import SEEDS, comparison, digest
from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_matched import save_summary


DEFAULT_RUN = Path("runs/curriculum-gsm8k-v1")
SHORT = 32768
LONG = 131072


def load_evaluation(path, ids, protocol_hash, checkpoint, arm, budget, seed):
    result = read_json(path)
    expected = {"protocol_sha256": protocol_hash, "checkpoint_sha256": checkpoint,
                "panel": path.parent.name, "arm": arm, "budget": budget, "seed": seed}
    require(all(result[k] == v for k, v in expected.items()), f"{path}: evaluation binding changed")
    rows = result["rows"]
    require([r["prompt_id"] for r in rows] == ids and all(r["reward"] in (0, 1) for r in rows),
            f"{path}: expected the frozen panel in order and binary exact-answer scores")
    require(sum(r["reward"] for r in rows) == result["correct"], f"{path}: incorrect evaluation count")
    return result


def compare_panels(baseline, candidate, ids, hard_ids):
    results = {}
    for name, selected in (("overall", ids), ("hard", hard_ids)):
        wanted = set(selected)

        def select(evaluations):
            return {key: {"rows": [r for r in entry["rows"] if r["prompt_id"] in wanted]}
                    for key, entry in evaluations.items()}

        old, new = select(baseline), select(candidate)
        stats = comparison(old, new, selected)
        # Use exactly the same paired question resamples as the repository's joint interval.
        n = len(selected)
        stats["question_count"] = n
        draw = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
        for seed in stats["per_seed"]:
            row = stats["per_seed"][seed]
            delta = np.array([b["reward"] - a["reward"] for a, b in
                              zip(old[seed]["rows"], new[seed]["rows"], strict=True)], dtype=np.int8)
            row["paired_question_95"] = np.quantile(delta[draw].mean(axis=1), [0.025, 0.975]).tolist()
        results[name] = stats
    return results


def with_density(counts):
    tokens = counts["generated_tokens"]
    return {**counts, "effective_groups_per_10000_generated_tokens": 10000 * counts["effective_groups"] / tokens}


def accounting(folder, arm, budget, seed, protocol_hash, protocol):
    report = read_json(folder / "training.json")
    binding = read_json(folder / "binding.json")
    start = protocol["start_sha256"]
    expected = {"arm": arm, "budget": budget, "seed": seed, "protocol_sha256": protocol_hash, "start_sha256": start}
    require(all(binding[k] == v for k, v in expected.items()) and report["start_sha256"] == start
            and report["seed"] == seed and report["lr"] == protocol["lr"]
            and report["accumulation_groups"] == protocol["accumulation_groups"],
            f"{folder}: training belongs to another run")
    trace = folder / "trace.jsonl"
    require(digest(trace) == report["trace_sha256"], f"{folder}: recorded training trace changed")
    counts = Counter()
    by_source = defaultdict(Counter)
    schedule = protocol["train_prompt_ids"][arm][str(seed)]
    labels = protocol["sampling_labels"][str(seed)]
    for index, row in enumerate(read_rows(trace)):
        samples = row["samples"]
        require(row["group_index"] == index and row["prompt_id"] == schedule[index] and 1 <= len(samples) <= 2,
                f"{folder}: malformed or unexpected group {index}")
        require(all(s["reward"] in (0, 1) for s in samples), f"{folder}: expected a binary training reward")
        tokens = sum(len(s["token_ids"]) for s in samples)
        require(tokens == row["generated_tokens"] and tokens > 0, f"{folder}: group token count changed")
        full = len(samples) == 2
        active = full and samples[0]["reward"] != samples[1]["reward"]
        values = {"generated_tokens": tokens, "groups": 1, "complete_groups": int(full), "effective_groups": int(active)}
        counts.update(values)
        source = "uniform" if arm == "uniform" else labels[index]
        require(source in ("uniform", "two_step"), f"{folder}: unknown sampling source {source}")
        by_source[source].update(values)
    require(counts["generated_tokens"] == report["accounting"]["generated_tokens"] == budget
            and counts["groups"] == report["accounting"]["groups"] and report["accounting"]["supervised_tokens"] == 0,
            f"{folder}: training budget mismatch")
    return counts, by_source, report["final_sha256"]


def load_endpoint(run, phase, arm, budget, protocol, ids, protocol_hash):
    label = f"{arm}-{budget}"
    pooled, pool_sources = Counter(), defaultdict(Counter)
    seeds, evaluations = {}, {}
    for seed in SEEDS:
        key = str(seed)
        folder = run / "models" / arm / str(budget) / f"seed-{seed}"
        counts, sources, endpoint = accounting(folder, arm, budget, seed, protocol_hash, protocol)
        pooled.update(counts)
        for source, totals in sources.items():
            pool_sources[source].update(totals)
        seeds[key] = {**with_density(counts), "by_sampling_source": {k: with_density(c) for k, c in sources.items()}}
        path = run / "eval" / phase / f"{label}-seed-{seed}.json"
        evaluations[key] = load_evaluation(path, ids, protocol_hash, endpoint, arm, budget, seed)
    return label, evaluations, {"per_seed": seeds, "pooled": {**with_density(pooled), "by_sampling_source": {
        k: with_density(c) for k, c in pool_sources.items()}}}


def overall_gate(stats):
    return (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
            and stats["bootstrap"]["question_block_median_95"][0] > 0)


def trigger_for(primary, path):
    return {"dose_authorized": primary["criteria"]["density_gain"] and not primary["passed"],
            "primary_results_sha256": digest(path), "protocol_sha256": primary["protocol_sha256"]}


def analyze(run, phase):
    protocol_path = run / "protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = digest(protocol_path)
    require(protocol["seeds"] == SEEDS and protocol["budgets"] == [SHORT, LONG] and protocol["group_size"] == 2,
            "expected the five matched G2 seeds at the two frozen budgets")
    if phase == "dose":
        path = run / "results-primary.json"
        primary = read_json(path)
        trigger = trigger_for(primary, path)
        require(primary["phase"] == "primary" and primary["protocol_sha256"] == protocol_hash
                and trigger["dose_authorized"] and trigger == read_json(run / "trigger.json"),
                "dose analysis requires the matching primary trigger")
    ids = protocol["eval_sets"][phase]
    hard = protocol["hard_eval_ids"][phase]
    hard_set = set(hard)
    require(ids and len(ids) == len(set(ids)) and hard and [i for i in ids if i in hard_set] == hard,
            "invalid frozen panel or hard subset")
    require(set(protocol["eval_sets"]["primary"]).isdisjoint(protocol["eval_sets"]["dose"]),
            "primary and dose evaluation questions overlap")
    initial = load_evaluation(run / "eval" / phase / "initial.json", ids, protocol_hash,
                              protocol["start_sha256"], None, None, None)
    starting = dict.fromkeys(map(str, SEEDS), initial)
    ref_arm = "uniform" if phase == "primary" else "mixed"
    cand_budget = SHORT if phase == "primary" else LONG
    ref, before, ref_counts = load_endpoint(run, phase, ref_arm, SHORT, protocol, ids, protocol_hash)
    cand, after, cand_counts = load_endpoint(run, phase, "mixed", cand_budget, protocol, ids, protocol_hash)
    contrasts = {"candidate_vs_reference": compare_panels(before, after, ids, hard),
                 "candidate_vs_initial": compare_panels(starting, after, ids, hard),
                 "reference_vs_initial": compare_panels(starting, before, ids, hard)}
    field = "effective_groups_per_10000_generated_tokens"
    deltas = {k: v[field] - ref_counts["per_seed"][k][field] for k, v in cand_counts["per_seed"].items()}
    density = {"per_seed_delta": deltas, "positive_seeds": sum(d > 0 for d in deltas.values()),
               "pooled_delta": cand_counts["pooled"][field] - ref_counts["pooled"][field]}
    criteria = {"candidate_vs_reference_gate": overall_gate(contrasts["candidate_vs_reference"]["overall"]),
                "candidate_vs_initial_gate": overall_gate(contrasts["candidate_vs_initial"]["overall"]),
                "hard_median_nonnegative": contrasts["candidate_vs_reference"]["hard"]["median_accuracy_gain"] >= 0}
    if phase == "primary":
        criteria["density_gain"] = density["positive_seeds"] >= 4 and density["pooled_delta"] > 0
    result = {"phase": phase, "protocol_sha256": protocol_hash, "reference": ref, "candidate": cand,
              "endpoint_accounting": {ref: ref_counts, cand: cand_counts}, "comparisons": contrasts,
              "criteria": criteria, "density": density, "passed": all(criteria.values())}
    if phase == "dose":
        result["additional_generated_tokens"] = {k: v["generated_tokens"] - ref_counts["per_seed"][k]["generated_tokens"]
                                                 for k, v in cand_counts["per_seed"].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--phase", choices=("primary", "dose"), required=True)
    args = parser.parse_args()
    result = analyze(args.out, args.phase)
    target = args.out / f"results-{args.phase}.json"
    save_summary(target, result)
    if args.phase == "primary":
        save_summary(args.out / "trigger.json", trigger_for(result, target))
    print(f"{args.phase} passed={result['passed']}, criteria={result['criteria']}")


if __name__ == "__main__":
    main()
