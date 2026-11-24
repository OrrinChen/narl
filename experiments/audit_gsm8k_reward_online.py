"""Audit reward-only online training and strict GSM8K development evaluations."""

import argparse
from collections import Counter
from importlib import import_module
import json
from math import isclose, isfinite
from pathlib import Path

from experiments.audit_gsm8k_online_lag import TRAIN_SHA256, adapter_digest, validate_sample
from experiments.audit_gsm8k_stable import SEEDS, check_case, check_eval, comparison, digest, replay_cases
from experiments.audit_kv_online import read_json, read_rows, require


ORIGINAL = "lr-0.0001-groups-16"
SCORERS = {
    "parsed_outcome": ("experiments.gsm8k_training_reward", "score_training_reward"),
    "verified_steps": ("experiments.gsm8k_step_reward", "score_training_reward"),
    "verified_format": ("experiments.gsm8k_step_reward", "score_format_reward"),
}


def behavior(sample):
    return {k: sample[k] for k in ("id", "token_ids", "behavior_logprobs", "behavior_version", "behavior_sha256",
                                  "completion", "truncated", "has_answer_tag", "token_cap")}


def check_source(base, source, plan):
    prior = read_json(base / "audit.json")
    require(prior["audit_passed"] and not prior["partial"] and prior["plan_sha256"] == digest(base / "plan.json")
            and prior["stage"] == "development_gate_failed" and not prior["official_test_scored"],
            "original strict-reward audit/plan changed")
    require(read_json(base / "round-2/selection.json")["selected"] == ORIGINAL
            and plan["source_protocol_sha256"] == digest(source / "protocol.json")
            and plan["seeds"] == SEEDS and plan["token_budget"] == 32768 and plan["group_size"] == 2,
            "original initialization, recipe or token budget changed")
    require(adapter_digest(source / "start-adapter.safetensors") == plan["source_protocol"]["start_sha256"],
            "common starting adapter bytes changed")
    q1, q2 = map(set, plan["qualification_prompt_ids"])
    require(len(q1) == len(q2) == 128 and not q1.intersection(q2), "development splits changed")
    for indices in plan["train_prompt_ids"].values():
        require(not (q1 | q2).intersection(indices), "development questions in training schedule")
    for seed in SEEDS:
        item = prior["training_cases"][f"{ORIGINAL}/seed-{seed}"]
        report = read_json(base / "models" / ORIGINAL / f"seed-{seed}/training.json")
        require(item["final_sha256"] == report["final_sha256"] and item["accounting"] == report["accounting"]
                and prior["training_replay"][f"{ORIGINAL}/seed-{seed}"]["every_batch_hash_matches"],
                f"seed {seed}: original strict-reward endpoint not fully replayed")


def validate_recipe(protocol):
    recipe = protocol["reward_recipe"]
    module_name, func = SCORERS[recipe]
    outcome_name = SCORERS["parsed_outcome"][0]
    if "parser_base_sha256" in protocol:
        base_parser = import_module(outcome_name)
        require(protocol["parser_base_sha256"] == digest(Path(base_parser.__file__)), "base outcome parser changed")
        outcome_name = "experiments.gsm8k_training_reward_v2"
        if recipe == "parsed_outcome":
            module_name = outcome_name
    require(protocol["score_module"] == module_name, "unregistered reward module")
    module = import_module(module_name)
    require(protocol["reward_module_sha256"] == digest(Path(module.__file__)), "reward module differs from freeze")
    if recipe != "parsed_outcome" or "parser_base_sha256" in protocol:
        parsed = import_module(outcome_name)
        require(protocol["outcome_module_sha256"] == digest(Path(parsed.__file__)), "underlying outcome scorer changed")
        if recipe != "parsed_outcome":
            require(module.outcome_reward is parsed.score_training_reward, "step scorer is using a different outcome parser")
    return getattr(module, func)


def check_training(folder, base, plan, train, tokenizer, seed, scorer, recipe):
    counts, reasons = Counter(), Counter()
    def sample_check(sample, example, cap, label, tokenizer):
        strict = {**sample, "reward": sample["strict_reward"]}
        strict.pop("reward_reason")
        n = validate_sample(strict, float(example["value"]), cap, label, tokenizer)
        # Reconstruct the input the runner saw before overwriting "reward".
        score, reason = scorer(strict, example["value"] if recipe == "parsed_outcome" else example)
        require(isfinite(score) and isclose(sample["reward"], score, abs_tol=1e-12, rel_tol=0)
                and sample["reward_reason"] == reason, f"{label}: training reward or reason mismatch")
        counts["samples"] += 1
        counts["changed_reward"] += int(sample["reward"] != sample["strict_reward"])
        reasons[reason] += 1
        return n

    case = check_case(folder, plan, train, tokenizer, 0.0001, 16, seed, training_sample_validator=sample_check)
    report, rows, updates = case
    original_folder = base / "models" / ORIGINAL / f"seed-{seed}"
    original_train = read_json(original_folder / "training.json")
    require(original_train["trace_sha256"] == digest(original_folder / "trace.jsonl"), "source trace digest changed")
    original = read_rows(original_folder / "trace.jsonl")
    # Paired random streams must produce the same tokens while the policy, prompt,
    # RNG key (seed/index), and generation caps are identical.
    changed_group = None
    for index, row in enumerate(rows):
        old = original[index] if index < len(original) else None
        if old is None or row["sha256"] != old["sha256"]:
            counts["groups_after_divergence"] += 1
            if changed_group is None and len(row["samples"]) == 2:
                changed_group = index
        elif [s["token_cap"] for s in old["samples"]] == [s["token_cap"] for s in row["samples"]]:
            require(old["prompt_ids"] == row["prompt_ids"]
                    and [behavior(s) for s in old["samples"]] == [behavior(s) for s in row["samples"]],
                    f"seed {seed}/group {index}: paired same-policy samples differ")
            counts["same_policy_groups_checked"] += 1
    require(counts["same_policy_groups_checked"] > 0, "first frozen batch missing its strict-policy control")
    return case, {"accounting": report["accounting"], "final_sha256": report["final_sha256"],
                  "reward_reasons": dict(reasons), **counts}, changed_group


def training_replay(cases, probes, source, plan, train, tokenizer_path):
    from experiments.gsm8k_mlx_development import adapter_snapshot
    from experiments.gsm8k_online_lag import rollouts
    from experiments.gsm8k_regroup import adapter_digest as tensor_digest

    regenerated = {}
    def regenerate(model, tokenizer, key, rows, update):
        index = probes[key] if probes[key] is not None else 0
        if index not in update["group_indices"]:
            return
        row = rows[index]
        snap = adapter_snapshot(model)
        previous = (snap, tensor_digest(snap))
        require(previous[1] == row["sha256"], f"{key}: regeneration used wrong policy")
        remaining = plan["token_budget"] - sum(r["generated_tokens"] for r in rows[:index])
        ids, samples = rollouts(model, tokenizer, train[row["prompt_id"]], plan["source_protocol"],
                                int(key), "fresh", index, remaining, previous, previous, row["version"])
        require(ids == row["prompt_ids"] and [behavior(s) for s in samples] == [behavior(s) for s in row["samples"]]
                and [s["reward"] for s in samples] == [s["strict_reward"] for s in row["samples"]],
                f"seed {key}: independently regenerated samples disagree")
        regenerated[key] = {"group": index, "sample_count": len(samples), "diverged_from_original": probes[key] is not None,
                            "behavior_sha256": previous[1], "tokens_and_logprobs_match": True}

    batches = replay_cases(cases, source, plan, tokenizer_path, before_update=regenerate)
    require(set(regenerated) == set(cases), "missing independently regenerated batch probes")
    return {"updates": batches, "regenerated_groups": regenerated}


def gate_passes(stats):
    return (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
            and stats["bootstrap"]["question_block_median_95"][0] > 0)


def audit_evaluations(run, base, plan, binding, cases, gold, tokenizer, split, partial):
    folder = run / split
    index = int(split == "q2")
    ids = plan["qualification_prompt_ids"][index]
    hashes = {s: case[0]["final_sha256"] for s, case in cases.items()}
    strict_base = base / "round-2/qualification"
    allowed = {"gate.json", "initial.json", *(f"seed-{s}.json" for s in SEEDS),
               *(f"strict-seed-{s}.json" for s in SEEDS)}
    require(all(p.name in allowed for p in folder.glob("*.json")), "unexpected development evaluation")
    def evaluation(name, expected_hash, archived=None):
        path = folder / f"{name}.json"
        if not path.exists():
            require(partial, f"missing {split}/{name}")
            return None
        raw = read_json(path)
        require(expected_hash is not None and raw["checkpoint_sha256"] == expected_hash
                and raw["protocol_sha256"] == binding["protocol_sha256"], f"{split}/{name}: wrong weights or protocol")
        check_eval(raw, ids, gold, tokenizer, f"{split}/{name}")
        if archived is not None:
            record = {k: v for k, v in raw.items() if k not in ("checkpoint_sha256", "protocol_sha256")}
            require(record == read_json(archived), f"{split}/{name}: strict control differs from archived Q1")
        return raw

    initial = evaluation("initial", binding["start_sha256"], None if index else strict_base / "initial.json")
    strict, candidate = {}, {}
    for seed in SEEDS:
        key = str(seed)
        result = evaluation(f"seed-{seed}", hashes.get(key))
        if result is not None:
            candidate[key] = result
        orig = base / "models" / ORIGINAL / f"seed-{seed}"
        old_digest = read_json(orig / "training.json")["final_sha256"]
        result = evaluation(f"strict-seed-{seed}", old_digest, None if index else strict_base / f"seed-{seed}.json")
        if result is not None:
            strict[key] = result
    summary = {"initial_correct": None if initial is None else initial["correct"],
               "candidate_correct": {k: v["correct"] for k, v in candidate.items()},
               "strict_correct": {k: v["correct"] for k, v in strict.items()}}
    if not initial or len(strict) < 5 or len(candidate) < 5:
        require(partial and not (folder / "gate.json").exists(), "gate calculated from incomplete evaluations")
        return summary
    stats = comparison({str(s): initial for s in SEEDS}, candidate, ids)
    expected = {"passed": gate_passes(stats), "selected": binding["reward_recipe"], "qualification_index": index,
                "protocol_sha256": binding["protocol_sha256"], "endpoint_sha256": hashes,
                "statistics": stats, "vs_strict": comparison(strict, candidate, ids)}
    if not (folder / "gate.json").exists():
        require(partial, f"{split}: missing frozen gate")
        return summary
    require(expected == read_json(folder / "gate.json"), f"{split}: reported gate differs from strict rescoring")
    return {**summary, **expected}


def check_stop(folder, recipe, scores, case_keys):
    stop = read_json(folder / "stop.json")
    seed = str(stop["seed"])
    require(recipe != "parsed_outcome" and scores.get("passed") is not True
            and scores["initial_correct"] is not None and seed in scores["candidate_correct"]
            and scores["candidate_correct"][seed] < scores["initial_correct"]
            and set(scores["candidate_correct"]) == set(scores["strict_correct"]) == set(case_keys)
            and not (folder / "q2").exists(), "reward candidate lacks an evaluable negative seed for early rejection")
    expected = {"reason": "negative Q1 seed", "seed": int(seed), "correct": scores["candidate_correct"][seed],
                "initial_correct": scores["initial_correct"], "evaluated_seeds": sorted(map(int, case_keys))}
    require(all(stop.get(k) == v for k, v in expected.items()), "registered early-stop counts differ from scored Q1")
    return expected


def check_selection(run, protocol, study, results, base, plan, gold, tokenizer):
    root = run.parent
    q1 = results["q1"]
    expected = {"reward_recipe": protocol["reward_recipe"], "protocol_sha256": q1["protocol_sha256"],
                "q1_gate_sha256": digest(run / "q1/gate.json"), "endpoint_sha256": q1["endpoint_sha256"],
                "study_sha256": digest(root / "study.json")}
    require(read_json(root / "selection.json") == expected, "Q2 is not bound to the frozen Q1 winner")
    for recipe in study["order"][:study["order"].index(protocol["reward_recipe"])]:
        previous = root / recipe
        binding = read_json(previous / "binding.json")
        require(binding["reward_recipe"] == recipe and binding["protocol_sha256"] == digest(previous / "protocol.json"),
                "earlier candidate identity changed")
        # Check earlier eligibility directly against strictly rescored Q1 outputs.
        cases = {str(s): (read_json(previous / f"models/seed-{s}/training.json"),) for s in SEEDS
                 if (previous / f"models/seed-{s}/training.json").exists()}
        scores = audit_evaluations(previous, base, plan, binding, cases, gold, tokenizer, "q1",
                                   (previous / "stop.json").exists())
        if (previous / "stop.json").exists():
            check_stop(previous, recipe, scores, cases)
        else:
            require(scores.get("passed") is False, f"earlier recipe {recipe} was not eliminated")


def audit(args):
    from transformers import AutoTokenizer
    from experiments.gsm8k_signal import read_split

    train = read_split(args.train, TRAIN_SHA256)
    gold = [(e["question"], float(e["value"])) for e in train]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    plan = read_json(args.base / "plan.json")
    check_source(args.base, args.source, plan)
    protocol = read_json(args.protocol)
    study_path = args.run.parent / "study.json"
    study = read_json(study_path)
    require(protocol["study_sha256"] == digest(study_path) and study["order"] == list(SCORERS)
            and study["base_plan_sha256"] == digest(args.base / "plan.json"), "changed study progression or order")
    scorer = validate_recipe(protocol)
    require(protocol["seeds"] == SEEDS and protocol["budget"] == 32768 and protocol["lr"] == 1e-4
            and protocol["accumulation_groups"] == 16 and protocol["base_plan_sha256"] == digest(args.base / "plan.json"),
            "unexpected training setup or source plan")
    binding = read_json(args.run / "binding.json")
    expected = {"reward_recipe": protocol["reward_recipe"], "base_plan_sha256": protocol["base_plan_sha256"],
                "protocol_sha256": digest(args.protocol), "reward_module_sha256": protocol["reward_module_sha256"],
                "start_sha256": plan["source_protocol"]["start_sha256"]}
    require(binding == expected, "changed study protocol binding")
    require(not (args.run / "formal").exists(), "requested development study contains official scoring artifacts")
    folders = args.run / "models"
    require(all(f.name in {f"seed-{i}" for i in SEEDS} for f in folders.glob("*")),
            "unregistered training run")
    cases, details, probes = {}, {}, {}
    for seed in SEEDS:
        case_folder = folders / f"seed-{seed}"
        if not (case_folder / "training.json").exists():
            require(args.partial or args.early_stop, f"seed {seed}: missing completed training")
            require(not args.early_stop or not case_folder.exists(), "unfinished training not covered by early rejection")
            continue
        k = str(seed)
        cases[k], details[k], probes[k] = check_training(case_folder, args.base, plan, train, tokenizer, seed,
                                                         scorer, protocol["reward_recipe"])
    evaluations = {"q1": audit_evaluations(args.run, args.base, plan, binding, cases, gold,
                                           tokenizer, "q1", args.partial or args.early_stop)}
    result = {"audit_passed": True, "partial": args.partial, "binding": binding, "training": details,
              "strict_evaluations": evaluations,
              "new_online_generation_tokens": sum(d["accounting"]["generated_tokens"] for d in details.values())}
    q2_dirs = list(args.run.parent.glob("*/q2"))
    require(len(q2_dirs) <= 1, "more than one reward recipe opened Q2")
    if (args.run / "q2").exists():
        require(evaluations["q1"].get("passed") is True, "independent Q2 accessed without passing Q1")
        check_selection(args.run, protocol, study, evaluations, args.base, plan, gold, tokenizer)
        evaluations["q2"] = audit_evaluations(args.run, args.base, plan, binding, cases, gold, tokenizer,
                                               "q2", args.partial)
    result["stage"] = "in_progress"
    for split, e in evaluations.items():
        if "passed" in e:
            result["stage"] = f"{split}_{'passed' if e['passed'] else 'failed'}"
    if args.early_stop:
        result["stop"] = check_stop(args.run, binding["reward_recipe"], evaluations["q1"], cases.keys())
        result["stage"] = "q1_early_rejected"
    if args.verify_training and cases:
        result["model_replay"] = training_replay(cases, probes, args.source, plan, train, args.tokenizer)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("runs/gsm8k-stable-v1"))
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--partial", action="store_true")
    mode.add_argument("--early-stop", action="store_true", help="audit conclusive rejection using a completed negative Q1 seed")
    parser.add_argument("--verify-training", action="store_true")
    args = parser.parse_args()
    result = audit(args)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Audited", result["stage"], "completed seeds", list(result["training"]), flush=True)


if __name__ == "__main__":
    main()
