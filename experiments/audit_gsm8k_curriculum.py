"""Verify paired GSM8K prompt-selection runs, binary GRPO rewards, weights and strict evaluations."""

import argparse
from collections import Counter, defaultdict
import json
from math import isclose
from pathlib import Path
from random import Random

from experiments.audit_gsm8k_online_lag import MODEL, REVISION, TRAIN_SHA256, adapter_digest, validate_sample
from experiments.audit_gsm8k_stable import SEEDS, check_case, check_eval, comparison, digest, replay_cases
from experiments.audit_kv_online import read_json, require
from experiments.gsm8k_training_reward_v2 import score_training_reward


FROZEN = "e4c1029d3102aee7db7a336701d19775ab46ced6503d1bc22bfa7efcee354fc6"
DRAW = 2026112523


def read_train(path):
    require(digest(path) == TRAIN_SHA256, "GSM8K training split changed")
    train = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            rationale, number = row["answer"].rsplit("####", 1)
            train.append({"question": row["question"], "rationale": rationale.strip(),
                          "value": number.strip().replace(",", "")})
    return train


def check_binary(sample, example, cap, label, tokenizer):
    strict = {**sample, "reward": sample["strict_reward"]}
    del strict["reward_reason"]
    length = validate_sample(strict, float(example["value"]), cap, label, tokenizer)
    reward, reason = score_training_reward(strict, example["value"])
    require(sample["reward"] in (0, 1) and (sample["reward"], sample["reward_reason"]) == (reward, reason),
            f"{label}: binary v2 reward or reason does not match the decoded completion")
    return length


def check_protocol(run, runs, train):
    path = run / "protocol.json"
    require(digest(path) == FROZEN, "curriculum protocol differs from the preregistered freeze")
    p = read_json(path)
    base = runs / "gsm8k-stable-v1/plan.json"
    prior = runs / "gsm8k-reward-format-replication/protocol.json"
    exposure = read_json(prior)
    source = p["source_protocol"]
    require(p["source_sha256"] == digest(base) and p["exposure_protocol_sha256"] == digest(prior)
            and source == read_json(base)["source_protocol"] == read_json(runs / "gsm8k-budget-5seed/protocol.json")
            and source["model"] == MODEL and source["revision"] == REVISION
            and p["start_sha256"] == source["start_sha256"] == adapter_digest(runs / "gsm8k-budget-5seed/start-adapter.safetensors")
            and p["train_sha256"] == source["train_sha256"] == TRAIN_SHA256, "source model, weights, dataset or plan changed")
    require(p["lr"] == 1e-4 and p["accumulation_groups"] == 16 and p["group_size"] == 2 and p["seeds"] == SEEDS
            and p["seed_for_draw"] == DRAW and p["budgets"] == [32768, 131072] and p["max_new_tokens"] == 256
            and p["eval_batch"] == source["eval_batch"] == 12 and p["eval_max_new_tokens"] == source["eval_max_new_tokens"] == 256,
            "frozen training or evaluation configuration changed")
    for name, sha in p["reward_source_sha256"].items():
        require(sha == digest(Path(__file__).with_name(name)), f"{name}: frozen binary reward changed")
    excluded = set(exposure["excluded_ids"] + exposure["panel_ids"]["A"] + exposure["panel_ids"]["B"])
    require(p["excluded_ids"] == sorted(excluded), "previous exposures or prior held-out panels were dropped")
    for trace in (runs / "gsm8k-reward-composite-v1/verified_composite/models").glob("*/trace.jsonl"):
        with trace.open() as f:
            require(all(json.loads(line)["prompt_id"] in excluded for line in f), "unaccounted prior training exposure")
    keys = [" ".join(e["question"].casefold().split()) for e in train]
    old_keys = {keys[i] for i in excluded}
    first = {}
    for i, text in enumerate(keys):
        if text not in old_keys:
            first.setdefault(text, i)
    eligible = list(first.values())
    draw = Random(DRAW).sample(eligible, 512)
    require(p["eval_sets"] == {"primary": draw[:256], "dose": draw[256:]}, "evaluation draw or panel order changed")
    steps = [e["rationale"].count("<<") for e in train]
    require(p["hard_eval_ids"] == {k: [i for i in ids if steps[i] >= 4] for k, ids in p["eval_sets"].items()},
            "hard subsets differ from four or more reference calculator annotations")
    pool = sorted(set(eligible) - set(draw))
    require(pool == p["train_pool_ids"], "training pool includes reserved questions or repeated text")
    for s in SEEDS:
        key = str(s)
        control = Random(DRAW + 100 + s).sample(pool, 512)
        rng = Random(DRAW + 200 + s)
        shared = [j + rng.randrange(2) for j in range(0, 512, 2)]
        shared_ids = {control[j] for j in shared}
        enriched = Random(DRAW + 300 + s).sample([i for i in pool if steps[i] == 2 and i not in shared_ids], 256)
        pairs = iter(enriched)
        labels = ["uniform" if j in shared else "two_step" for j in range(512)]
        mixed = [i if label == "uniform" else next(pairs) for i, label in zip(control, labels, strict=True)]
        require(p["train_prompt_ids"]["uniform"][key] == control and p["train_prompt_ids"]["mixed"][key] == mixed
                and p["sampling_labels"][key] == labels and len(set(mixed)) == 512, f"seed {s}: sampling stream changed")
    return p


def training_metrics(case):
    report, rows, updates = case
    counts = Counter()
    for row in rows:
        samples = row["samples"]
        if len(samples) == 2:
            counts[sum(sample["reward"] for sample in samples)] += 1
    tokens = report["accounting"]["generated_tokens"]
    active = counts[1]
    require(active == sum(update["nonzero_groups"] for update in updates), "effective groups differ from optimizer inputs")
    return {"generated_tokens": tokens, "groups": len(rows), "complete_groups": sum(counts.values()), "effective_groups": active,
            "both_wrong_groups": counts[0], "both_correct_groups": counts[2],
            "effective_groups_per_10000_generated_tokens": 10000 * active / tokens,
            "changed_weights": report["accounting"]["changed_weights"],
            "start_sha256": report["start_sha256"], "final_sha256": report["final_sha256"]}


def source_metrics(rows, labels):
    sources = defaultdict(Counter)
    for i, row in enumerate(rows):
        counts = sources[labels[i]]
        counts["generated_tokens"] += row["generated_tokens"]
        counts["groups"] += 1
        counts["complete_groups"] += int(len(row["samples"]) == 2)
        counts["effective_groups"] += int(row["nonzero_advantage_tokens"] > 0)
    return {key: pool_counts([counts]) for key, counts in sources.items()}


def pool_counts(records):
    counts = {k: sum(x[k] for x in records) for k in ("generated_tokens", "complete_groups", "effective_groups", "groups")}
    counts["effective_groups_per_10000_generated_tokens"] = 10000 * counts["effective_groups"] / counts["generated_tokens"]
    return counts


def compare_endpoints(left, right, ids):
    import numpy as np

    result = comparison(left, right, ids)
    rng = np.random.default_rng(80161)
    draws = rng.integers(len(ids), size=(10000, len(ids)))
    result["question_count"] = len(ids)
    for s in SEEDS:
        key = str(s)
        delta = np.array([b["reward"] - a["reward"] for a, b in zip(left[key]["rows"], right[key]["rows"], strict=True)])
        result["per_seed"][key]["paired_question_95"] = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()
    return result


def paired_tables(initial, control, candidate, ids, hard_ids):
    """The question masks and their order must be identical for all paired endpoints."""
    result = {"candidate_vs_reference": {}, "candidate_vs_initial": {}, "reference_vs_initial": {}}
    for name, subset in (("overall", ids), ("hard", hard_ids)):
        allowed = set(subset)
        def select(evaluation):
            rows = [r for r in evaluation["rows"] if r["prompt_id"] in allowed]
            require([r["prompt_id"] for r in rows] == subset, f"{name}: unpaired or reordered question")
            return {"rows": rows}
        baseline = {str(s): select(initial) for s in SEEDS}
        a = {str(s): select(control[str(s)]) for s in SEEDS}
        b = {str(s): select(candidate[str(s)]) for s in SEEDS}
        result["candidate_vs_reference"][name] = compare_endpoints(a, b, subset)
        result["candidate_vs_initial"][name] = compare_endpoints(baseline, b, subset)
        result["reference_vs_initial"][name] = compare_endpoints(baseline, a, subset)
    return result


def check_prefix(short_case, long_case):
    short, rows, updates = short_case
    _, continued, more_updates = long_case
    require(len(continued) > len(rows) and len(more_updates) > len(updates)
            and continued[:len(rows)] == rows and more_updates[:len(updates)] == updates,
            "high-dose trace does not preserve the entire short-dose training prefix")
    require(continued[len(rows)]["sha256"] == more_updates[len(updates)]["old_sha256"] == short["final_sha256"],
            "continued training did not start at the short-budget endpoint")


def check_same_policy_pairs(uniform, mixed):
    count = 0
    # At most 16*2*256 tokens in the first batch, so both arms use the same batched sampling path.
    for a, b in zip(uniform[1][:16], mixed[1][:16], strict=True):
        if a["sha256"] != b["sha256"] or a["prompt_id"] != b["prompt_id"]:
            continue
        if [s["token_cap"] for s in a["samples"]] != [s["token_cap"] for s in b["samples"]]:
            continue
        require(a["group_index"] == b["group_index"] and a["prompt_ids"] == b["prompt_ids"] and
                all(s["token_ids"] == t["token_ids"] and s["behavior_logprobs"] == t["behavior_logprobs"]
                    for s, t in zip(a["samples"], b["samples"], strict=True)),
                "same seed, group index, question, cap and weights produced different samples")
        count += 1
    require(count > 0, "paired arms lack a matching shared-prompt group in the initial frozen batch")
    return count


def compare_density(reference, candidate):
    field = "effective_groups_per_10000_generated_tokens"
    def pooled(per_seed):
        sampling = defaultdict(list)
        for item in per_seed.values():
            for source, counts in item["by_sampling_source"].items():
                sampling[source].append(counts)
        return {**pool_counts(per_seed.values()), "by_sampling_source": {s: pool_counts(v) for s, v in sampling.items()}}
    a, b = pooled(reference), pooled(candidate)
    deltas = {s: candidate[s][field] - reference[s][field] for s in reference}
    return a, b, {"per_seed_delta": deltas, "positive_seeds": sum(d > 0 for d in deltas.values()),
                  "pooled_delta": b[field] - a[field]}


def check_trigger(run, protocol_hash, primary):
    expected = {"protocol_sha256": protocol_hash, "primary_results_sha256": digest(run / "results-primary.json"),
                "dose_authorized": primary["criteria"]["density_gain"] and not primary["passed"]}
    require(read_json(run / "trigger.json") == expected, "dose branch not bound to the registered primary outcome")
    return expected


def check_fields(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), f"{label}: missing derived field")
        for key, value in expected.items():
            check_fields(actual[key], value, f"{label}/{key}")
    elif isinstance(expected, float):
        require(isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"{label}: derived metric differs")
    else:
        require(actual == expected, f"{label}: derived value differs")


def accuracy_gate(stats):
    return (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
            and stats["bootstrap"]["question_block_median_95"][0] > 0)


def evaluate_phase(run, p, phase, train, tokenizer, cases):
    primary = phase == "primary"
    ids = p["eval_sets"][phase]
    reference = ("uniform", 32768) if primary else ("mixed", 32768)
    candidate = ("mixed", 32768) if primary else ("mixed", 131072)
    recipe_names = [f"{arm}-{budget}" for arm, budget in (reference, candidate)]
    gold = [(e["question"], float(e["value"])) for e in train]
    folder = run / "eval" / phase
    expected_files = {"initial.json"} | {f"{name}-seed-{s}.json" for name in recipe_names for s in SEEDS}
    require({x.name for x in folder.glob("*.json")} == expected_files, f"{phase}: missing or extra evaluation endpoint")

    def evaluation(name, arm, budget, seed, checkpoint):
        raw = read_json(folder / f"{name}.json")
        check_fields(raw, {"panel": phase, "arm": arm, "budget": budget, "seed": seed,
                           "protocol_sha256": FROZEN, "checkpoint_sha256": checkpoint}, name)
        check_eval(raw, ids, gold, tokenizer, f"{phase}/{name}")
        return raw

    initial = evaluation("initial", None, None, None, p["start_sha256"])
    endpoints, accounting = [], {}
    for name, (arm, budget) in zip(recipe_names, (reference, candidate), strict=True):
        by_seed, evaluations = {}, {}
        for seed in SEEDS:
            key = f"{name}-seed-{seed}"
            case_folder = run / "models" / arm / str(budget) / f"seed-{seed}"
            binding = {"protocol_sha256": FROZEN, "arm": arm, "seed": seed,
                       "budget": budget, "start_sha256": p["start_sha256"]}
            require(read_json(case_folder / "binding.json") == binding, f"{key}: training bound to another study or arm")
            if key not in cases:
                plan = {**p, "train_prompt_ids": p["train_prompt_ids"][arm], "token_budget": budget}
                cases[key] = check_case(case_folder, plan, train, tokenizer, p["lr"], p["accumulation_groups"], seed,
                                        training_sample_validator=check_binary, checkpoints=(32768,) if budget == 131072 else ())
            if budget == 131072:
                check_prefix(cases[f"mixed-32768-seed-{seed}"], cases[key])
            by_seed[str(seed)] = training_metrics(cases[key])
            labels = p["sampling_labels"][str(seed)] if arm == "mixed" else ["uniform"] * len(cases[key][1])
            by_seed[str(seed)]["by_sampling_source"] = source_metrics(cases[key][1], labels)
            evaluations[str(seed)] = evaluation(key, arm, budget, seed, by_seed[str(seed)]["final_sha256"])
        endpoints.append(evaluations)
        accounting[name] = {"per_seed": by_seed}
    a, b = recipe_names
    a_pool, b_pool, density = compare_density(accounting[a]["per_seed"], accounting[b]["per_seed"])
    accounting[a]["pooled"], accounting[b]["pooled"] = a_pool, b_pool
    comparisons = paired_tables(initial, *endpoints, ids, p["hard_eval_ids"][phase])
    criteria = {"candidate_vs_reference_gate": accuracy_gate(comparisons["candidate_vs_reference"]["overall"]),
                "candidate_vs_initial_gate": accuracy_gate(comparisons["candidate_vs_initial"]["overall"]),
                "hard_median_nonnegative": comparisons["candidate_vs_reference"]["hard"]["median_accuracy_gain"] >= 0}
    if primary:
        criteria["density_gain"] = density["positive_seeds"] >= 4 and density["pooled_delta"] > 0
    expected = {"phase": phase, "protocol_sha256": FROZEN, "reference": a, "candidate": b,
                "comparisons": comparisons, "criteria": criteria, "passed": all(criteria.values())}
    reported = read_json(run / f"results-{phase}.json")
    check_fields(reported, expected, phase)
    for name, item in accounting.items():
        actual = reported["endpoint_accounting"][name]
        check_fields(actual["pooled"], item["pooled"], f"{name}/pooled")
        for seed, metrics in item["per_seed"].items():
            check_fields(actual["per_seed"][seed], {k: metrics[k] for k in item["pooled"]}, f"{name}/seed-{seed}")
    check_fields(reported["density"], density, f"{phase}/density")
    if not primary:
        check_fields(reported, {"additional_generated_tokens": {str(s): 131072 - 32768 for s in SEEDS}}, phase)
    result = {**expected, "endpoint_accounting": accounting, "density": density}
    if primary:
        result["identical_policy_and_question_groups_checked"] = {
            str(s): check_same_policy_pairs(cases[f"{a}-seed-{s}"], cases[f"{b}-seed-{s}"]) for s in SEEDS}
    return result


def audit(run, runs, phase, tokenizer_path=None, protocol_only=False, verify_training=False):
    train = read_train(runs / "gsm8k-data/train.jsonl")
    p = check_protocol(run, runs, train)
    result = {"audit_passed": True, "protocol_sha256": FROZEN, "phase": phase, "protocol_only": protocol_only,
              "excluded_count": len(p["excluded_ids"]), "train_pool_size": len(p["train_pool_ids"]),
              "panel_sizes": {k: len(v) for k, v in p["eval_sets"].items()},
              "four_plus_counts": {k: len(v) for k, v in p["hard_eval_ids"].items()}}
    if protocol_only:
        require(not verify_training, "protocol-only audit cannot replay updates")
        return result
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or MODEL, revision=REVISION, local_files_only=True)
    cases = {}
    primary = evaluate_phase(run, p, "primary", train, tokenizer, cases)
    trigger = check_trigger(run, FROZEN, primary)
    result["trigger"] = trigger
    if not trigger["dose_authorized"]:
        require(not (run / "eval/dose").exists() and not any((run / "models").glob("*/131072/seed-*"))
                and not (run / "results-dose.json").exists(), "unauthorized dose artifacts found")
    if phase == "dose":
        require(trigger["dose_authorized"] is True, "dose phase was not authorized by the audited primary result")
        result["outcome"] = evaluate_phase(run, p, "dose", train, tokenizer, cases)
    else:
        result["outcome"] = primary
    if verify_training:
        selected = result["outcome"]
        replay = {f"{name}-seed-{s}": cases[f"{name}-seed-{s}"]
                  for name in (selected["reference"], selected["candidate"]) for s in SEEDS}
        result["training_replay"] = replay_cases(replay, runs / "gsm8k-budget-5seed", p, tokenizer_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", "--run", dest="run", type=Path, help="study output directory")
    parser.add_argument("--phase", choices=("primary", "dose"), default="primary")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--report", type=Path, help="write the audit report to this file")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--verify-training", action="store_true")
    args = parser.parse_args()
    run = args.run or args.runs / "curriculum-gsm8k-v1"
    result = audit(run, args.runs, args.phase, args.tokenizer, args.protocol_only, args.verify_training)
    destination = args.report or run / ("protocol-audit.json" if args.protocol_only else f"audit-{args.phase}.json")
    pending = destination.with_suffix(".tmp")
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pending.replace(destination)
    print(f"Audited {args.phase} at {destination}: audit_passed={result['audit_passed']}")
    if "outcome" in result:
        print(f"Learning criteria met: {result['outcome']['passed']}")


if __name__ == "__main__":
    main()
