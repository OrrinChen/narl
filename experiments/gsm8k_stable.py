"""Tune frozen-policy GRPO group accumulation, qualify it, then compare with SFT."""

import argparse
from hashlib import sha256
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean, median
from time import perf_counter

import mlx.core as mx
from safetensors.torch import load_file, save_file
import torch

from experiments.audit_gsm8k_accuracy import paired_accuracy
from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_budget import evaluate
from experiments.gsm8k_grpo_diagnosis import output_metrics
from experiments.gsm8k_matched import save_summary
from experiments.gsm8k_mlx_development import adapter_snapshot, load_mlx_model, set_adapter
from experiments.gsm8k_mlx_grpo import MlxGrpoLearner
from experiments.gsm8k_online_lag import advantages, rollouts
from experiments.gsm8k_regroup import adapter_digest
from experiments.gsm8k_signal import TRAIN_SHA256, TEST_SHA256, read_split
from experiments.regroup import write_record


SEEDS = list(range(5))
SCREEN_SEEDS = [0, 4]
QUALIFICATION_SEED, TEST_SEED = 92317, 92333
ROUNDS = [
    [(0.0001, 4), (0.0001, 8), (0.0003, 4), (0.0003, 8)],
    [(0.00003, 8), (0.0001, 16), (0.0003, 16)],
]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def config_name(lr, groups):
    return f"lr-{lr:g}-groups-{groups}"


def freeze(args, train, test):
    source = read_json(args.source / "protocol.json")
    audit = read_json(args.source / "audit.json")
    require(audit["audit_passed"] and audit["model_replay"]["replayed_seeds"] == 5,
            "SFT reference and original initialization lack full replay audit")
    require(source["total_tokens"] == 32768 and source["seeds"] == SEEDS, "unexpected source budget or seeds")
    used = set(source["excluded_train_ids"] + source["dev_prompt_ids"])
    baseline = {}
    for seed in SEEDS:
        for row in read_rows(args.source / f"seed-{seed}/grpo/trace.jsonl"):
            used.add(row["prompt_id"])
        folder = args.source / f"seed-{seed}/sft"
        report = read_json(folder / "summary.json")
        require(report["accounting"]["generated_tokens"] == 0
                and report["accounting"]["supervised_tokens"] == 32768
                and report["sft_start_sha256"] == source["start_sha256"]
                and adapter_digest(load_file(folder / "final-adapter.safetensors")) == report["final_sha256"],
                "SFT endpoint or response-token budget differs from the audited reference")
        used.update(row["prompt_id"] for row in report["training"])
        baseline[str(seed)] = {"final_sha256": report["final_sha256"], "summary_sha256": digest(folder / "summary.json"),
                               "accounting": report["accounting"]}
    diagnosis = args.source.parent / "gsm8k-grpo-diagnosis/online-low/trace.jsonl"
    if diagnosis.exists():
        used.update(row["prompt_id"] for row in read_rows(diagnosis))
    fresh = sorted(set(range(len(train))) - used)
    reserved = Random(QUALIFICATION_SEED).sample(fresh, 256)
    qualification = [reserved[:128], reserved[128:]]
    schedules = {str(seed): [i for i in source["grpo_prompt_ids"][str(seed)] if i not in set(reserved)] for seed in SEEDS}
    excluded_test = sorted(set(source["excluded_test_ids"] + source["heldout_prompt_ids"]))
    heldout = Random(TEST_SEED).sample(sorted(set(range(len(test))) - set(excluded_test)), args.heldout)
    require(not {train[i]["question"] for i in reserved} & {train[i]["question"] for i in used},
            "fresh qualification text overlaps previously trained or inspected training questions")
    require(not {train[i]["question"] for i in reserved} &
            {train[i]["question"] for ids in schedules.values() for i in ids},
            "qualification text overlaps the new training pool")
    require(not {test[i]["question"] for i in heldout} & {row["question"] for row in train}, "test text overlaps train")
    return {"source_run": args.source.name, "source_protocol": source,
            "source_protocol_sha256": digest(args.source / "protocol.json"), "sft_reference": baseline,
            "seeds": SEEDS, "screen_seeds": SCREEN_SEEDS, "rounds": [[[lr, n] for lr, n in r] for r in ROUNDS],
            "screen_prompt_ids": source["dev_prompt_ids"], "qualification_seed": QUALIFICATION_SEED,
            "qualification_prompt_ids": qualification, "qualification_excluded_prior_train_ids": sorted(used),
            "train_prompt_ids": schedules, "test_seed": TEST_SEED, "excluded_test_ids": excluded_test,
            "heldout_prompt_ids": heldout, "token_budget": 32768, "group_size": 2, "max_new_tokens": 256,
            "eval_max_new_tokens": 256, "eval_batch": 12,
            "sampling_policy": "one frozen policy per complete accumulation batch; prompt-local advantages",
            "loss_normalization": "mean over all complete sequences, including zero-advantage groups",
            "screen_rule": "both seeds nonnegative and at least one positive; mean, worst seed, lower LR, larger batch",
            "qualification_rule": "at least 4/5 positive, none negative, median positive, question-block median 95% lower > 0",
            "stopping_rule": "at most two predefined rounds and one untouched official-test evaluation after qualification",
            "budget_unit": source["budget_unit"],
            "cost_scope": "32768 response tokens per formal seed; development search and common initialization separate",
            "sft_reuse": "audited 32768-label SFT-only endpoints; no new SFT tuning on confirmation questions"}


def pair_stats(initial, evaluations, count):
    pairs = {}
    for seed in SEEDS:
        rows = evaluations[str(seed)]["rows"]
        pairs[str(seed)] = ([r["reward"] for r in initial["rows"]], [r["reward"] for r in rows],
                            [r["token_ids"] for r in initial["rows"]], [r["token_ids"] for r in rows])
    result = paired_accuracy(pairs, count)
    result.pop("qualified")
    return result


def train_case(model, tokenizer, train, plan, start, lr, accumulation, seed, folder, *, training_reward=None,
               training_group_reward=None, restore_training=None):
    require(training_reward is None or training_group_reward is None, "choose one training reward callback")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "training.json").exists():
        return read_json(folder / "training.json")
    set_adapter(model, start)
    learner = MlxGrpoLearner(model, lr)
    current, version_ = (start, plan["source_protocol"]["start_sha256"]), 0
    generated = consumed = signal_tokens = learner_tokens = 0
    rows, updates, pending, pending_ids = [], [], [], []
    started = perf_counter()
    mode = "w"
    if restore_training is not None:
        rows, updates = restore_training(model, learner)
        if updates:
            current = adapter_snapshot(model), updates[-1]["new_sha256"]
            version_ = updates[-1]["new_version"]
            require(adapter_digest(current[0]) == current[1], "restored adapter differs from the last update")
        generated = sum(r["generated_tokens"] for r in rows)
        consumed = sum(r["consumed_tokens"] for r in rows)
        signal_tokens = sum(r["nonzero_advantage_tokens"] for r in rows)
        learner_tokens = sum(r["learner_input_tokens"] for r in rows)
        require(generated <= plan["token_budget"], "restored progress exceeds the budget")
        mode = "a"
    schedule = plan["train_prompt_ids"][str(seed)]
    with (folder / "trace.jsonl").open(mode) as trace_handle, (folder / "updates.jsonl").open(mode) as update_handle:
        for index in range(len(rows), len(schedule)):
            if generated == plan["token_budget"]:
                break
            prompt_id = schedule[index]
            prompt_ids, samples = rollouts(model, tokenizer, train[prompt_id], plan["source_protocol"], seed, "fresh", index,
                                           plan["token_budget"] - generated, current, current, version_)
            if training_group_reward is not None:
                for sample in samples:
                    sample["strict_reward"] = sample["reward"]
                scores = list(training_group_reward(samples, train[prompt_id]))
                require(len(scores) == len(samples), "expected one score and reason per generated completion")
                for sample, (score, reason) in zip(samples, scores, strict=True):
                    require(isfinite(score) and isinstance(reason, str) and reason, "invalid group reward or reason")
                    sample["reward"], sample["reward_reason"] = score, reason
            elif training_reward is not None:
                for sample in samples:
                    sample["strict_reward"] = sample["reward"]
                    sample["reward"], sample["reward_reason"] = training_reward(sample, train[prompt_id])
            emitted = sum(len(s["token_ids"]) for s in samples)
            complete = len(samples) == plan["group_size"]
            values, signal = advantages(prompt_ids, samples) if complete else ([0.0] * len(samples), 0)
            inputs = sum(len(prompt_ids) + len(s["token_ids"]) - 1
                         for s, a in zip(samples, values, strict=True) if a != 0)
            row = {"group_index": index, "batch_index": len(updates), "prompt_id": prompt_id, "prompt_ids": prompt_ids,
                   "samples": samples, "version": version_, "sha256": current[1], "generated_tokens": emitted,
                   "consumed_tokens": emitted if complete else 0, "nonzero_advantage_tokens": signal,
                   "learner_input_tokens": inputs, "advantages": values}
            rows.append(row)
            write_record(trace_handle, row)
            trace_handle.flush()
            generated += emitted
            consumed += row["consumed_tokens"]
            signal_tokens += signal
            learner_tokens += inputs
            if complete:
                pending.append((prompt_ids, samples, values))
                pending_ids.append(index)
            if pending and (len(pending) == accumulation or generated == plan["token_budget"]):
                diagnostics = learner.step_groups(pending)
                snapshot = adapter_snapshot(model)
                next_hash = adapter_digest(snapshot)
                update = {"batch_index": len(updates), "group_indices": pending_ids,
                          "sequence_count": sum(len(s) for _, s, _ in pending),
                          "nonzero_groups": sum(any(v != 0 for v in a) for _, _, a in pending),
                          "old_version": version_, "old_sha256": current[1],
                          "new_version": version_ + int(next_hash != current[1]), "new_sha256": next_hash,
                          "cumulative_generated_tokens": generated, "diagnostics": diagnostics}
                write_record(update_handle, update)
                update_handle.flush()
                updates.append(update)
                current, version_ = (snapshot, next_hash), update["new_version"]
                pending, pending_ids = [], []
                print(config_name(lr, accumulation), seed, f"{generated}/{plan['token_budget']}", f"v{version_}",
                      "active groups", update["nonzero_groups"], flush=True)
    require(generated == plan["token_budget"] and not pending, "incomplete generation budget or unflushed batch")
    save_file(current[0], folder / "final-adapter.safetensors")
    report = {"lr": lr, "accumulation_groups": accumulation, "seed": seed,
              "start_sha256": plan["source_protocol"]["start_sha256"], "final_sha256": current[1],
              "trace_sha256": digest(folder / "trace.jsonl"), "updates_sha256": digest(folder / "updates.jsonl"),
              "accounting": {"generated_tokens": generated, "supervised_tokens": 0, "consumed_tokens": consumed,
                             "nonzero_advantage_tokens": signal_tokens, "learner_input_tokens": learner_tokens,
                             "generation_prompt_tokens": sum(len(r["prompt_ids"]) * len(r["samples"]) for r in rows),
                             "groups": len(rows), "optimizer_batches": len(updates), "changed_weights": version_},
              "training_elapsed_s": perf_counter() - started}
    save_summary(folder / "training.json", report)
    return report


def load_endpoint(model, folder, expected=None):
    snapshot = load_file(folder / "final-adapter.safetensors")
    report = read_json(folder / "training.json") if expected is None else expected
    require(adapter_digest(snapshot) == report["final_sha256"], "trained endpoint digest mismatch")
    set_adapter(model, snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("runs/gsm8k-data/test.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heldout", type=int, default=192)
    parser.add_argument("--stage", choices=("plan", "screen", "qualify", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    train, test = read_split(args.train, TRAIN_SHA256), read_split(args.test, TEST_SHA256)
    plan = freeze(args, train, test)
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        require(read_json(plan_path) == plan, "settings changed from the frozen plan")
    else:
        save_summary(plan_path, plan)
    if args.stage == "plan":
        return
    start = load_file(args.source / "start-adapter.safetensors")
    require(adapter_digest(start) == plan["source_protocol"]["start_sha256"], "common initialization changed")
    model, tokenizer = load_mlx_model(None, len(plan["source_protocol"]["qualification_protocol"]["lora_layers"]), TEST_SEED)
    common_path = args.out / "screen-initial.json"
    if not common_path.exists():
        set_adapter(model, start)
        save_summary(common_path, evaluate(model, tokenizer, train, plan["screen_prompt_ids"], plan["source_protocol"]))
    screen_initial = read_json(common_path)
    winner, used_qualification = None, 0
    for round_index, configs in enumerate(plan["rounds"], 1):
        round_folder = args.out / f"round-{round_index}"
        round_folder.mkdir(exist_ok=True)
        results = {}
        for lr, accumulation in configs:
            name = config_name(lr, accumulation)
            results[name] = {"lr": lr, "accumulation_groups": accumulation, "seeds": {}}
            for seed in SCREEN_SEEDS:
                folder = args.out / "models" / name / f"seed-{seed}"
                training = train_case(model, tokenizer, train, plan, start, lr, accumulation, seed, folder)
                target = folder / "screen.json"
                if not target.exists():
                    load_endpoint(model, folder)
                    evaluation = evaluate(model, tokenizer, train, plan["screen_prompt_ids"], plan["source_protocol"])
                    save_summary(target, {"evaluation": evaluation, "metrics": output_metrics(evaluation),
                                          "final_sha256": training["final_sha256"]})
                report = read_json(target)
                results[name]["seeds"][str(seed)] = report
                save_summary(round_folder / "screen.json", results)
                print("screen", name, seed, report["metrics"], flush=True)
        eligible = []
        for name, entry in results.items():
            deltas = [entry["seeds"][str(s)]["metrics"]["correct"] - screen_initial["correct"] for s in SCREEN_SEEDS]
            entry["deltas"] = deltas
            if min(deltas) >= 0 and max(deltas) > 0:
                eligible.append((mean(deltas), min(deltas), -entry["lr"], entry["accumulation_groups"], name))
        save_summary(round_folder / "screen.json", results)
        selected = max(eligible)[-1] if eligible else None
        selection = {"selected": selected, "qualification_index": used_qualification if selected else None,
                     "screen_sha256": digest(round_folder / "screen.json")}
        selection_path = round_folder / "selection.json"
        if selection_path.exists():
            require(read_json(selection_path) == selection, "frozen selection changed")
        else:
            save_summary(selection_path, selection)
        print("selected", round_index, selection, flush=True)
        if args.stage == "screen":
            return
        if selected is None:
            continue
        entry = results[selected]
        ids = plan["qualification_prompt_ids"][used_qualification]
        used_qualification += 1
        qualification_folder = round_folder / "qualification"
        qualification_folder.mkdir(exist_ok=True)
        initial_path = qualification_folder / "initial.json"
        if not initial_path.exists():
            set_adapter(model, start)
            save_summary(initial_path, evaluate(model, tokenizer, train, ids, plan["source_protocol"]))
        initial = read_json(initial_path)
        evaluations = {}
        for seed in SEEDS:
            folder = args.out / "models" / selected / f"seed-{seed}"
            train_case(model, tokenizer, train, plan, start, entry["lr"], entry["accumulation_groups"], seed, folder)
            target = qualification_folder / f"seed-{seed}.json"
            if not target.exists():
                load_endpoint(model, folder)
                save_summary(target, evaluate(model, tokenizer, train, ids, plan["source_protocol"]))
            evaluations[str(seed)] = read_json(target)
            print("qualification", round_index, seed, evaluations[str(seed)]["correct"], initial["correct"], flush=True)
        stats = pair_stats(initial, evaluations, len(ids))
        passed = (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
                  and stats["bootstrap"]["question_block_median_95"][0] > 0)
        gate = {"passed": passed, "statistics": stats, "selected": selected, "qualification_index": used_qualification - 1}
        save_summary(qualification_folder / "gate.json", gate)
        print("development gate", round_index, gate, flush=True)
        if passed:
            winner = {"round": round_index, "selected": selected, "lr": entry["lr"],
                      "accumulation_groups": entry["accumulation_groups"],
                      "qualification_gate_sha256": digest(qualification_folder / "gate.json")}
            break
    if winner is None:
        save_summary(args.out / "development-status.json", {"qualified": False, "official_test_scored": False,
                                                            "completed_rounds": len(plan["rounds"])})
        return
    formal_path = args.out / "formal-protocol.json"
    formal = {"winner": winner, "plan_sha256": digest(plan_path), "heldout_prompt_ids": plan["heldout_prompt_ids"]}
    if formal_path.exists():
        require(read_json(formal_path) == formal, "formal recipe changed")
    else:
        save_summary(formal_path, formal)
    if args.stage == "qualify":
        return
    for seed in SEEDS:
        for arm in ("grpo", "sft"):
            target = args.out / "formal" / f"seed-{seed}-{arm}.json"
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                continue
            if arm == "grpo":
                folder = args.out / "models" / winner["selected"] / f"seed-{seed}"
                training = read_json(folder / "training.json")
                load_endpoint(model, folder)
            else:
                folder = args.source / f"seed-{seed}/sft"
                training = read_json(folder / "summary.json")
                load_endpoint(model, folder, training)
            evaluation = evaluate(model, tokenizer, test, plan["heldout_prompt_ids"], plan["source_protocol"])
            save_summary(target, {"evaluation": evaluation, "metrics": output_metrics(evaluation),
                                  "final_sha256": training["final_sha256"], "accounting": training["accounting"]})
            print("formal", seed, arm, output_metrics(evaluation), flush=True)


if __name__ == "__main__":
    main()
