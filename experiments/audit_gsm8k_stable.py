"""Independently audit development-gated frozen-policy GRPO accumulation."""

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random
import re
from statistics import mean, median

from experiments.audit_gsm8k_accuracy import paired_accuracy
from experiments.audit_gsm8k_budget import prompt_tokens
from experiments.audit_gsm8k_online_lag import (
    MODEL, REVISION, TEST_SHA256, TRAIN_SHA256, adapter_digest,
    check_diagnostics, check_evaluation, normalized_advantages, validate_sample,
)
from experiments.audit_gsm8k_regroup import check_prompt_ids
from experiments.audit_kv_online import check_reported, read_json, read_rows, require


SEEDS = list(range(5))
SCREEN_SEEDS = [0, 4]
ROUNDS = [
    [(0.0001, 4), (0.0001, 8), (0.0003, 4), (0.0003, 8)],
    [(0.00003, 8), (0.0001, 16), (0.0003, 16)],
]
HASH = re.compile(r"[0-9a-f]{64}")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def config_name(lr, groups):
    return f"lr-{lr:g}-groups-{groups}"


def metrics(evaluation):
    rows = evaluation["rows"]
    return {"correct": sum(row["reward"] for row in rows), "count": len(rows),
            "truncated": sum(row["truncated"] for row in rows),
            "tagged": sum(row["has_answer_tag"] for row in rows),
            "median_tokens": median(len(row["token_ids"]) for row in rows),
            "pure_answer": sum(re.fullmatch(r"\s*<answer>[^<>]+</answer>\s*", row["completion"]) is not None
                               for row in rows)}


def check_eval(evaluation, ids, gold, tokenizer, label, reported_metrics=None):
    check_evaluation(evaluation, ids, gold, label, 256, tokenizer)
    derived = metrics(evaluation)
    if reported_metrics is not None:
        check_reported(reported_metrics, derived, label)
    return derived


def comparison(initial, candidates, ids):
    pairs = {}
    for seed in SEEDS:
        base, new = initial[str(seed)]["rows"], candidates[str(seed)]["rows"]
        require([r["prompt_id"] for r in base] == ids == [r["prompt_id"] for r in new],
                "paired evaluation questions or order changed")
        pairs[str(seed)] = ([r["reward"] for r in base], [r["reward"] for r in new],
                            [r["token_ids"] for r in base], [r["token_ids"] for r in new])
    result = paired_accuracy(pairs, len(ids))
    result.pop("qualified")
    return result


def check_plan(source_path, run, train, test):
    source = read_json(source_path / "protocol.json")
    plan = read_json(run / "plan.json")
    source_audit = read_json(source_path / "audit.json")
    require(source_audit["audit_passed"] and source_audit["model_replay"]["replayed_seeds"] == 5,
            "reused SFT and common initialization lack the original full replay audit")
    require((source["model"], source["revision"], source["train_sha256"], source["test_sha256"]) ==
            (MODEL, REVISION, TRAIN_SHA256, TEST_SHA256), "source model or dataset revision changed")
    require(plan["source_run"] == source_path.name and plan["source_protocol"] == source
            and plan["source_protocol_sha256"] == digest(source_path / "protocol.json")
            and adapter_digest(source_path / "start-adapter.safetensors") == source["start_sha256"],
            "source protocol or common starting weights changed")
    require(source["total_tokens"] == plan["token_budget"] == 32768 and plan["seeds"] == SEEDS
            and plan["screen_seeds"] == SCREEN_SEEDS
            and plan["rounds"] == [[[lr, n] for lr, n in rows] for rows in ROUNDS]
            and (plan["group_size"], plan["max_new_tokens"], plan["eval_max_new_tokens"], plan["eval_batch"]) ==
            (2, 256, 256, 12), "frozen budgets, seeds, candidate grid, or decode settings changed")
    require(plan["qualification_seed"] == 92317 and plan["test_seed"] == 92333
            and plan["screen_prompt_ids"] == source["dev_prompt_ids"]
            and len(plan["screen_prompt_ids"]) == 48, "screen split or random seeds changed")
    rules = {
        "sampling_policy": "one frozen policy per complete accumulation batch; prompt-local advantages",
        "loss_normalization": "mean over all complete sequences, including zero-advantage groups",
        "screen_rule": "both seeds nonnegative and at least one positive; mean, worst seed, lower LR, larger batch",
        "qualification_rule": "at least 4/5 positive, none negative, median positive, question-block median 95% lower > 0",
        "stopping_rule": "at most two predefined rounds and one untouched official-test evaluation after qualification",
        "cost_scope": "32768 response tokens per formal seed; development search and common initialization separate",
        "sft_reuse": "audited 32768-label SFT-only endpoints; no new SFT tuning on confirmation questions",
    }
    require(all(plan[key] == value for key, value in rules.items()) and plan["budget_unit"] == source["budget_unit"],
            "predeclared selection, qualification, stopping, or accounting rules changed")
    used = set(source["excluded_train_ids"] + source["dev_prompt_ids"])
    baseline = {}
    for seed in SEEDS:
        used.update(row["prompt_id"] for row in read_rows(source_path / f"seed-{seed}/grpo/trace.jsonl"))
        folder = source_path / f"seed-{seed}/sft"
        report = read_json(folder / "summary.json")
        used.update(row["prompt_id"] for row in report["training"])
        require(report["accounting"]["generated_tokens"] == 0
                and report["accounting"]["supervised_tokens"] == 32768
                and report["sft_start_sha256"] == source["start_sha256"]
                and report["final_sha256"] == adapter_digest(folder / "final-adapter.safetensors"),
                f"seed {seed}: reused SFT starting weights, labels, or final hash changed")
        baseline[str(seed)] = {"final_sha256": report["final_sha256"],
                               "summary_sha256": digest(folder / "summary.json"), "accounting": report["accounting"]}
    diagnosis = source_path.parent / "gsm8k-grpo-diagnosis/online-low/trace.jsonl"
    if diagnosis.exists():
        used.update(row["prompt_id"] for row in read_rows(diagnosis))
    require(plan["sft_reference"] == baseline and plan["qualification_excluded_prior_train_ids"] == sorted(used),
            "SFT provenance or prior actually-used training IDs changed")
    fresh = sorted(set(range(len(train))) - used)
    reserved = Random(92317).sample(fresh, 256)
    require(plan["qualification_prompt_ids"] == [reserved[:128], reserved[128:]],
            "qualification sets differ from the pre-training fresh draws")
    reserved_set = set(reserved)
    require(not {train[i]["question"] for i in reserved} & {train[i]["question"] for i in used},
            "qualification text repeats previously trained or inspected questions")
    require(set(plan["train_prompt_ids"]) == {str(seed) for seed in SEEDS}, "missing a training seed schedule")
    for seed in SEEDS:
        expected = [i for i in source["grpo_prompt_ids"][str(seed)] if i not in reserved_set]
        require(plan["train_prompt_ids"][str(seed)] == expected, f"seed {seed}: changed eligible pool or question order")
        require(not reserved_set.intersection(expected) and not set(plan["screen_prompt_ids"]).intersection(expected),
                f"seed {seed}: development leaked into GRPO training")
    excluded = sorted(set(source["excluded_test_ids"] + source["heldout_prompt_ids"]))
    require(len(excluded) == 329 and plan["excluded_test_ids"] == excluded,
            "official test exclusions differ from the 329 previously inspected questions")
    heldout = plan["heldout_prompt_ids"]
    check_prompt_ids(heldout, test, "new official test")
    require(len(heldout) == 192 and heldout == Random(92333).sample(sorted(set(range(len(test))) - set(excluded)), 192),
            "official test differs from the predeclared fresh 192-question draw")
    require(not {test[i]["question"] for i in heldout} & {row["question"] for row in train},
            "official test text overlaps the training split")
    return plan


def check_case(folder, plan, train, tokenizer, lr, accumulation, seed, *, training_sample_validator=None,
               training_group_reward=None):
    report = read_json(folder / "training.json")
    rows, updates = read_rows(folder / "trace.jsonl"), read_rows(folder / "updates.jsonl")
    require(rows and report["lr"] == lr and report["accumulation_groups"] == accumulation and report["seed"] == seed,
            f"{folder}: training recipe or seed differs from its registered case")
    current, version, batch = plan["source_protocol"]["start_sha256"], 0, 0
    pending, totals = [], dict.fromkeys(("generated_tokens", "supervised_tokens", "consumed_tokens",
                                       "nonzero_advantage_tokens", "learner_input_tokens", "generation_prompt_tokens",
                                       "groups", "optimizer_batches", "changed_weights"), 0)
    schedule = plan["train_prompt_ids"][str(seed)]
    for index, row in enumerate(rows):
        label = f"{folder.name}/group {index}"
        remaining = plan["token_budget"] - totals["generated_tokens"]
        require(remaining > 0 and row["group_index"] == index and row["prompt_id"] == schedule[index]
                and row["batch_index"] == batch and row["version"] == version and row["sha256"] == current,
                f"{label}: broken prompt, frozen-policy batch, or version chain")
        prompt = prompt_tokens(tokenizer, train[row["prompt_id"]])
        require(row["prompt_ids"] == prompt, f"{label}: prompt tokens changed")
        samples = row["samples"]
        require(len(samples) == min(2, remaining), f"{label}: incomplete or excessive generation slots")
        lengths = []
        example = train[row["prompt_id"]]
        if training_group_reward is not None:
            # Recreate the strict inputs from before the online training reward was assigned.
            strict_samples = [{**sample, "reward": sample["strict_reward"]} for sample in samples]
            for sample in strict_samples:
                del sample["reward_reason"]
        for slot, sample in enumerate(samples):
            cap = min(256, remaining - sum(lengths) - (len(samples) - slot - 1))
            name = f"{seed}:fresh:{index}:{slot}"
            require(sample["id"] == name and sample["behavior_version"] == version
                    and sample["behavior_sha256"] == current and sample["token_cap"] == cap,
                    f"{label}/slot {slot}: wrong source policy or generation cap")
            if training_sample_validator is None:
                strict = sample if training_group_reward is None else strict_samples[slot]
                lengths.append(validate_sample(strict, float(example["value"]), cap, name, tokenizer))
            else:
                lengths.append(training_sample_validator(sample, example, cap, name, tokenizer))
        if training_group_reward is not None:
            expected = list(training_group_reward(strict_samples, example))
            require(len(expected) == len(samples), f"{label}: group scorer returned the wrong number of rewards")
            for slot, ((score, reason), sample) in enumerate(zip(expected, samples, strict=True)):
                require(isfinite(score) and score == sample["reward"] and isinstance(reason, str) and reason
                        and reason == sample["reward_reason"], f"{label}/slot {slot}: group reward or reason changed")
        complete = len(samples) == 2
        values = normalized_advantages(samples) if complete else [0.0]
        require(len(row["advantages"]) == len(values) and
                all(isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(row["advantages"], values, strict=True)),
                f"{label}: advantage was not normalized within its prompt group")
        emitted = sum(lengths)
        signal = sum(length for length, value in zip(lengths, values, strict=True) if value != 0)
        inputs = sum(len(prompt) + length - 1 for length, value in zip(lengths, values, strict=True) if value != 0)
        check_reported(row, {"generated_tokens": emitted, "consumed_tokens": emitted if complete else 0,
                             "nonzero_advantage_tokens": signal, "learner_input_tokens": inputs}, label)
        totals["generated_tokens"] += emitted
        totals["consumed_tokens"] += emitted if complete else 0
        totals["nonzero_advantage_tokens"] += signal
        totals["learner_input_tokens"] += inputs
        totals["generation_prompt_tokens"] += len(prompt) * len(samples)
        totals["groups"] += 1
        if complete:
            pending.append(index)
        if pending and (len(pending) == accumulation or totals["generated_tokens"] == plan["token_budget"]):
            require(batch < len(updates), f"{label}: complete batch was never passed to the learner")
            update = updates[batch]
            batch_values = [value for i in pending for value in normalized_advantages(rows[i]["samples"])]
            nonzero = sum(any(normalized_advantages(rows[i]["samples"])) for i in pending)
            check_reported(update, {"batch_index": batch, "group_indices": pending, "sequence_count": 2 * len(pending),
                                    "nonzero_groups": nonzero, "old_version": version, "old_sha256": current,
                                    "cumulative_generated_tokens": totals["generated_tokens"]}, f"{label}/batch")
            check_diagnostics(update["diagnostics"], batch_values, f"{label}/batch")
            require(HASH.fullmatch(update["new_sha256"]) is not None, f"{label}: invalid next adapter digest")
            changed = update["new_sha256"] != current
            require(update["new_version"] == version + int(changed) and (nonzero > 0 or not changed),
                    f"{label}: version increment or zero-signal update is incorrect")
            current, version = update["new_sha256"], update["new_version"]
            totals["optimizer_batches"] += 1
            totals["changed_weights"] += int(changed)
            batch += 1
            pending = []
    require(totals["generated_tokens"] == plan["token_budget"] and not pending and batch == len(updates),
            f"{folder}: unused budget, unflushed groups, or extra learner updates")
    check_reported(report["accounting"], totals, str(folder))
    require(report["start_sha256"] == plan["source_protocol"]["start_sha256"] and
            report["final_sha256"] == current == adapter_digest(folder / "final-adapter.safetensors") and
            report["trace_sha256"] == digest(folder / "trace.jsonl") and
            report["updates_sha256"] == digest(folder / "updates.jsonl"), f"{folder}: endpoint or trace bytes changed")
    require(isfinite(report["training_elapsed_s"]) and report["training_elapsed_s"] >= 0, f"{folder}: invalid timing")
    return report, rows, updates


def replay_advantages(samples):
    """Use the original binary replay path; fractional rewards need torch's float32 std reduction."""
    import numpy as np
    values = [sample["reward"] for sample in samples]
    if all(value in (0, 1) for value in values):
        rewards = np.asarray(values, dtype=np.float32)
        return ((rewards - rewards.mean()) / (rewards.std(ddof=1) + np.float32(1e-4))).tolist()
    import torch
    rewards = torch.tensor([values], dtype=torch.float32)
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return (centered / (rewards.std(dim=1, correction=1, keepdim=True) + 1e-4))[0].tolist()


def replay_cases(cases, source_path, plan, tokenizer_path, *, before_update=None):
    from importlib.metadata import version

    from safetensors.torch import load_file
    import torch

    from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
    from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
    from experiments.gsm8k_regroup import adapter_digest as tensor_digest

    source = plan["source_protocol"]
    require((version("mlx"), version("mlx-lm")) == (source["mlx"], source["mlx_lm"]), "training replay MLX versions changed")
    torch.set_num_threads(1)
    model, tokenizer = load_mlx_model(tokenizer_path, len(source["qualification_protocol"]["lora_layers"]), 92333)
    start = load_file(source_path / "start-adapter.safetensors")
    checked = {}
    for key, (report, rows, updates) in cases.items():
        set_adapter(model, start)
        learner = MlxGrpoLearner(model, report["lr"])
        fractional_groups, max_double_delta = 0, 0.0
        for update in updates:
            if before_update is not None:
                before_update(model, tokenizer, key, rows, update)
            groups = []
            for index in update["group_indices"]:
                row = rows[index]
                values = replay_advantages(row["samples"])
                require(values == row["advantages"], f"{key}/group {index}: reconstructed float32 advantages differ")
                if any(s["reward"] not in (0, 1) for s in row["samples"]):
                    fractional_groups += 1
                    reference = normalized_advantages(row["samples"])
                    max_double_delta = max(max_double_delta, *(abs(a - b) for a, b in zip(values, reference, strict=True)))
                    require(all(isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(values, reference, strict=True)),
                            f"{key}/group {index}: float32 advantages disagree with independent double-precision check")
                groups.append((row["prompt_ids"], row["samples"], values))
            actual = learner.step_groups(groups)
            expected = update["diagnostics"]
            require(actual.keys() == expected.keys() and all(isclose(value, expected[name], rel_tol=1e-5, abs_tol=1e-6)
                                                           for name, value in actual.items()),
                    f"{key}/batch {update['batch_index']}: replay optimizer diagnostics changed")
            require(tensor_digest(adapter_snapshot(model)) == update["new_sha256"],
                    f"{key}/batch {update['batch_index']}: replay parameter bytes changed")
        require(tensor_digest(adapter_snapshot(model)) == report["final_sha256"], f"{key}: replay endpoint changed")
        checked[key] = {"groups": len(rows), "learner_batches": len(updates),
                        "changed_weights": report["accounting"]["changed_weights"], "every_batch_hash_matches": True}
        if fractional_groups:
            checked[key]["fractional_advantage_check"] = {
                "groups": fractional_groups, "matches_archived_float32": True,
                "max_abs_delta_vs_float64": max_double_delta, "float64_rtol": 1e-6, "float64_atol": 1e-7,
            }
        print("replayed", key, checked[key], flush=True)
    return checked


def audit(source_path, run, train_path, test_path, tokenizer_path, partial=False, verify_training=False):
    from transformers import AutoTokenizer

    from experiments.gsm8k_signal import read_split

    train, test = read_split(train_path, TRAIN_SHA256), read_split(test_path, TEST_SHA256)
    train_gold = [(row["question"], float(row["value"])) for row in train]
    test_gold = [(row["question"], float(row["value"])) for row in test]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    plan = check_plan(source_path, run, train, test)
    registered = {config_name(lr, n): (lr, n) for configs in ROUNDS for lr, n in configs}
    cases, case_results, unfinished = {}, {}, []
    for folder in sorted((run / "models").glob("*/seed-*")):
        name = folder.parent.name
        seed = int(folder.name.removeprefix("seed-"))
        require(name in registered and seed in SEEDS, f"unregistered training case {folder}")
        key = f"{name}/seed-{seed}"
        if not (folder / "training.json").exists():
            require(partial, f"unfinished training case {key}")
            unfinished.append(key)
            continue
        lr, accumulation = registered[name]
        cases[key] = check_case(folder, plan, train, tokenizer, lr, accumulation, seed)
        report = cases[key][0]
        case_results[key] = {"accounting": report["accounting"], "final_sha256": report["final_sha256"],
                             "training_elapsed_s": report["training_elapsed_s"]}
        if (folder / "screen.json").exists():
            screen = read_json(folder / "screen.json")
            require(seed in SCREEN_SEEDS and screen["final_sha256"] == report["final_sha256"],
                    f"{key}: changed screen checkpoint or unexpected screen seed")
            case_results[key]["screen"] = check_eval(screen["evaluation"], plan["screen_prompt_ids"], train_gold,
                                                       tokenizer, key, screen["metrics"])
    result = {"audit_passed": True, "partial": partial, "scope": "complete saved artifacts only" if partial else "complete study",
              "plan_sha256": digest(run / "plan.json"), "training_cases": case_results,
              "unfinished_cases": unfinished, "rounds": {}, "official_test_scored": False}
    screen_path = run / "screen-initial.json"
    if not screen_path.exists():
        require(partial, "missing common screen evaluation")
        result["stage"] = "plan_only"
        return result
    initial_screen = read_json(screen_path)
    check_eval(initial_screen, plan["screen_prompt_ids"], train_gold, tokenizer, "screen initial")
    prior_initial = read_json(source_path / "development.json")["initial_evaluation"]
    require(initial_screen == prior_initial, "unchanged common weights produced a different screen evaluation")
    used_qualification, winner, completed_rounds = 0, None, 0
    selected_names = set()
    for round_index, configs in enumerate(ROUNDS, 1):
        folder = run / f"round-{round_index}"
        if not (folder / "screen.json").exists():
            break
        screen = read_json(folder / "screen.json")
        allowed_names = {config_name(lr, n) for lr, n in configs}
        require(set(screen) <= allowed_names, f"round {round_index}: unregistered screening recipe")
        full = set(screen) == allowed_names
        eligible = []
        for name, entry in screen.items():
            lr, accumulation = registered[name]
            require(entry["lr"] == lr and entry["accumulation_groups"] == accumulation
                    and set(entry["seeds"]) <= {str(s) for s in SCREEN_SEEDS}, "screen recipe or seed set changed")
            full = full and set(entry["seeds"]) == {str(s) for s in SCREEN_SEEDS}
            for seed, report in entry["seeds"].items():
                cached = run / "models" / name / f"seed-{seed}" / "screen.json"
                require(cached.exists() and report == read_json(cached), "screen rollup differs from archived endpoint")
            if set(entry["seeds"]) == {str(s) for s in SCREEN_SEEDS}:
                deltas = [entry["seeds"][str(s)]["metrics"]["correct"] - initial_screen["correct"] for s in SCREEN_SEEDS]
                if "deltas" in entry:
                    require(entry["deltas"] == deltas, "screen gain differs from independently scored outputs")
                if min(deltas) >= 0 and max(deltas) > 0:
                    eligible.append((mean(deltas), min(deltas), -lr, accumulation, name))
        selection_path = folder / "selection.json"
        if not selection_path.exists():
            require(partial, f"round {round_index}: incomplete selection")
            break
        require(full, "selected a recipe before both screen seeds of every candidate completed")
        selected = max(eligible)[-1] if eligible else None
        selection = {"selected": selected, "qualification_index": used_qualification if selected else None,
                     "screen_sha256": digest(folder / "screen.json")}
        require(read_json(selection_path) == selection, "frozen selection disagrees with predefined ranking")
        completed_rounds = round_index
        round_result = {"selection": selection}
        result["rounds"][str(round_index)] = round_result
        qualification = folder / "qualification"
        if selected is None:
            require(not qualification.exists(), "qualification was scored without a selected candidate")
            continue
        selected_names.add(selected)
        ids = plan["qualification_prompt_ids"][used_qualification]
        used_qualification += 1
        if not (qualification / "initial.json").exists():
            require(partial, "selected candidate has no qualification baseline")
            break
        initial = read_json(qualification / "initial.json")
        check_eval(initial, ids, train_gold, tokenizer, f"round {round_index}/qualification initial")
        evaluations = {}
        for seed in SEEDS:
            path = qualification / f"seed-{seed}.json"
            if path.exists():
                require(f"{selected}/seed-{seed}" in cases, "qualification lacks its trained endpoint")
                evaluations[str(seed)] = read_json(path)
                check_eval(evaluations[str(seed)], ids, train_gold, tokenizer, f"round {round_index}/qualification/{seed}")
        gate_path = qualification / "gate.json"
        if not gate_path.exists():
            require(partial, "qualification has not finished all five paired seeds")
            break
        require(len(evaluations) == 5, "qualification gate omitted a seed")
        stats = comparison({str(s): initial for s in SEEDS}, evaluations, ids)
        passed = (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
                  and stats["bootstrap"]["question_block_median_95"][0] > 0)
        gate = {"passed": passed, "statistics": stats, "selected": selected, "qualification_index": used_qualification - 1}
        require(read_json(gate_path) == gate, "reported development gate differs from rescoring or frozen thresholds")
        round_result["qualification"] = {"initial_correct": initial["correct"], **gate}
        if passed:
            lr, accumulation = registered[selected]
            winner = {"round": round_index, "selected": selected, "lr": lr, "accumulation_groups": accumulation,
                      "qualification_gate_sha256": digest(gate_path)}
            require(not (run / f"round-{round_index + 1}").exists(), "searched another round after qualification passed")
            break
    for key in cases:
        name, seed_name = key.split("/")
        require(int(seed_name.removeprefix("seed-")) in SCREEN_SEEDS or name in selected_names,
                "trained a qualification seed before its configuration was selected")
    formal_folder, formal_path = run / "formal", run / "formal-protocol.json"
    expected_formal_files = {f"seed-{seed}-{arm}.json" for seed in SEEDS for arm in ("grpo", "sft")}
    require({p.name for p in formal_folder.glob("*.json")} <= expected_formal_files,
            "unregistered official-test evaluations appeared in the formal directory")
    if winner is None:
        require(not formal_path.exists() and not list(formal_folder.glob("*.json")),
                "official-test confirmation began before the development gate passed")
        if not partial:
            require(completed_rounds == len(ROUNDS), "predefined development search is incomplete")
            require(read_json(run / "development-status.json") ==
                    {"qualified": False, "official_test_scored": False, "completed_rounds": len(ROUNDS)},
                    "failed-development status changed or claims an official test")
        result["stage"] = "development_in_progress" if partial else "development_gate_failed"
    else:
        result["winner"] = winner
        formal = {"winner": winner, "plan_sha256": digest(run / "plan.json"),
                  "heldout_prompt_ids": plan["heldout_prompt_ids"]}
        if formal_path.exists():
            require(read_json(formal_path) == formal, "formal recipe or held-out set changed after qualification")
        else:
            require(partial, "missing frozen formal protocol")
        evaluations = {"grpo": {}, "sft": {}}
        for seed in SEEDS:
            for arm in ("grpo", "sft"):
                path = formal_folder / f"seed-{seed}-{arm}.json"
                if not path.exists():
                    require(partial, f"missing formal pair {seed}/{arm}")
                    continue
                require(formal_path.exists(), "official scoring started before formal protocol was frozen")
                report = read_json(path)
                if arm == "grpo":
                    expected = cases[f"{winner['selected']}/seed-{seed}"][0]
                else:
                    expected = read_json(source_path / f"seed-{seed}/sft/summary.json")
                require(report["final_sha256"] == expected["final_sha256"] and
                        report["accounting"] == expected["accounting"], "formal pair changed its endpoint or budget")
                require(report["accounting"]["generated_tokens"] + report["accounting"]["supervised_tokens"] == 32768,
                        "formal arm does not match the response-token budget")
                check_eval(report["evaluation"], plan["heldout_prompt_ids"], test_gold, tokenizer,
                           f"formal/{seed}/{arm}", report["metrics"])
                evaluations[arm][str(seed)] = report["evaluation"]
        count = sum(map(len, evaluations.values()))
        result["official_test_scored"] = count > 0
        result["stage"] = "formal_complete" if count == 10 else "qualified_confirmation_in_progress"
        if count == 10:
            result["formal_grpo_vs_sft"] = comparison(evaluations["sft"], evaluations["grpo"], plan["heldout_prompt_ids"])
            result["formal_correct"] = {str(s): {arm: evaluations[arm][str(s)]["correct"] for arm in evaluations} for s in SEEDS}
    result["physical_new_grpo_training_tokens"] = sum(row[0]["accounting"]["generated_tokens"] for row in cases.values())
    result["logical_reused_sft_training_tokens"] = 5 * 32768
    result["formal_grpo_training_tokens"] = 5 * 32768 if winner is not None else 0
    result["search_tokens_outside_formal_grpo_endpoints"] = (
        result["physical_new_grpo_training_tokens"] - result["formal_grpo_training_tokens"])
    if verify_training:
        result["training_replay"] = replay_cases(cases, source_path, plan, tokenizer_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("runs/gsm8k-data/test.jsonl"))
    for name in ("run", "tokenizer", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="check only complete artifacts during development")
    parser.add_argument("--verify-training", action="store_true", help="replay every completed candidate's batch updates")
    args = parser.parse_args()
    tokenizer_path = args.tokenizer.parent if args.tokenizer.is_file() else args.tokenizer
    report = audit(args.source, args.run, args.train, args.test, tokenizer_path, args.partial, args.verify_training)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Audited stable GRPO study:", report["stage"], "; completed cases:", len(report["training_cases"]))


if __name__ == "__main__":
    main()
