"""Audit the disclosed single-candidate supplement after strict GRPO screen failure."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from experiments.audit_gsm8k_stable import (
    ROUNDS, SEEDS, audit as audit_base, check_case, check_eval, check_plan,
    comparison, config_name, replay_cases,
)
from experiments.audit_gsm8k_online_lag import TRAIN_SHA256, TEST_SHA256, adapter_digest
from experiments.audit_kv_online import read_json, require


DISCLOSURE = "Conditional exploratory supplement after partial old-screen observations; unchanged qualification gate."
FILES = ("training.json", "final-adapter.safetensors", "trace.jsonl", "updates.jsonl", "screen.json")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_policy(base, run, plan, policy_path):
    expected = {
        "base_run": base.name,
        "base_plan_sha256": digest(base / "plan.json"),
        "disclosure": "Prospective supplement after partial old-screen results; before any qualification evaluation.",
        "trigger": "Both predefined rounds finish with no selected candidate, no qualification initial/candidate scores, and no formal test scoring.",
        "selection": "One best positive paired-mean screen recipe across all seven existing configurations; tie-break worst seed, lower LR, larger accumulation.",
        "configuration_changes": "None; no extra hyperparameter configurations, no seed or checkpoint selection.",
        "training_reuse_seeds": [0, 4],
        "additional_training_seeds": [1, 2, 3],
        "qualification_prompt_ids": sum(plan["qualification_prompt_ids"], []),
        "qualification_gate": plan["qualification_rule"],
        "token_budget": 32768,
        "heldout_prompt_ids": plan["heldout_prompt_ids"],
        "test_policy": "Original 192 new official questions remain unscored until the unchanged gate passes.",
        "stopping_rule": "At most one supplementary candidate and one 256-question five-seed qualification; no retry after failure.",
        "provenance": "Preserve base v1 plan, traces and strict-failure record; bind final selection to completed screen/endpoint hashes before qualification.",
    }
    require(read_json(policy_path) == expected, "prospectively disclosed supplement policy changed")
    require(len(expected["qualification_prompt_ids"]) == len(set(expected["qualification_prompt_ids"])) == 256,
            "qualification does not combine exactly the two reserved fresh sets")
    return expected


def check_protocol(source, base, run, policy_path, plan, base_report):
    initial = read_json(base / "screen-initial.json")
    hashes = {"screen-initial.json": digest(base / "screen-initial.json")}
    candidates, endpoints = [], {}
    for index, configs in enumerate(ROUNDS, 1):
        folder = base / f"round-{index}"
        require(read_json(folder / "selection.json")["selected"] is None,
                "supplement triggered despite an eligible original candidate")
        require(not (folder / "qualification").exists(), "a reserved qualification set was previously evaluated")
        for filename in ("screen.json", "selection.json"):
            relative = f"round-{index}/{filename}"
            hashes[relative] = digest(base / relative)
        for lr, accumulation in configs:
            name = config_name(lr, accumulation)
            deltas = [base_report["training_cases"][f"{name}/seed-{seed}"]["screen"]["correct"] - initial["correct"]
                      for seed in (0, 4)]
            candidates.append({"selected": name, "lr": lr, "accumulation_groups": accumulation,
                               "mean_gain": mean(deltas), "worst_gain": min(deltas), "deltas": deltas})
            for seed in (0, 4):
                folder = base / "models" / name / f"seed-{seed}"
                training = read_json(folder / "training.json")
                endpoints[f"{name}/seed-{seed}"] = {"final_sha256": training["final_sha256"],
                                                    "files": {f: digest(folder / f) for f in FILES}}
    candidates.sort(key=lambda c: (c["mean_gain"], c["worst_gain"], -c["lr"], c["accumulation_groups"], c["selected"]),
                    reverse=True)
    chosen = candidates[0] if candidates[0]["mean_gain"] > 0 else None
    references = {}
    for seed in SEEDS:
        folder = source / f"seed-{seed}/sft"
        references[str(seed)] = {**plan["sft_reference"][str(seed)],
                                 "endpoint_file_sha256": digest(folder / "final-adapter.safetensors")}
    policy = read_json(policy_path)
    expected = {
        "disclosure": DISCLOSURE,
        "base_plan_sha256": digest(base / "plan.json"),
        "base_audit_sha256": digest(base / "audit.json"),
        "policy_sha256": digest(policy_path),
        "source_protocol_sha256": plan["source_protocol_sha256"],
        "source_audit_sha256": digest(source / "audit.json"),
        "start_file_sha256": digest(source / "start-adapter.safetensors"),
        "base_screen_sha256": hashes, "base_endpoints": endpoints, "sft_reference": references,
        "candidate": chosen, "screen_ranking": candidates, "seeds": SEEDS,
        "qualification_prompt_ids": sum(plan["qualification_prompt_ids"], []),
        "qualification_rule": plan["qualification_rule"], "heldout_prompt_ids": plan["heldout_prompt_ids"],
        "token_budget": 32768, "stopping_rule": policy["stopping_rule"],
        "training_reuse_seeds": [0, 4], "additional_training_seeds": [1, 2, 3],
        "planned_new_grpo_training_tokens": 3 * 32768 if chosen else 0,
        "planned_formal_grpo_training_tokens_if_qualified": 5 * 32768 if chosen else 0,
        "cost_scope": "Copied seeds cost zero new training tokens; common initialization, prior screening and reused SFT are separate; response tokens do not match FLOPs.",
    }
    require(read_json(run / "protocol.json") == expected,
            "supplement protocol, ranking, source bytes, endpoint binding, or accounting allocation changed")
    selection = {"selected": chosen["selected"] if chosen else None,
                 "protocol_sha256": digest(run / "protocol.json")}
    require(read_json(run / "selection.json") == selection, "candidate differs from its frozen independent selection")
    return expected


def audit(source, base, run, policy_path, train_path, test_path, tokenizer_path, partial=False, verify_training=False):
    from transformers import AutoTokenizer

    from experiments.gsm8k_signal import read_split

    train, test = read_split(train_path, TRAIN_SHA256), read_split(test_path, TEST_SHA256)
    train_gold = [(row["question"], float(row["value"])) for row in train]
    test_gold = [(row["question"], float(row["value"])) for row in test]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    plan = check_plan(source, base, train, test)
    check_policy(base, run, plan, policy_path)
    saved_base_audit = read_json(base / "audit.json")
    require(saved_base_audit.get("audit_passed") is True and saved_base_audit.get("partial") is False
            and saved_base_audit.get("stage") == "development_gate_failed"
            and saved_base_audit.get("official_test_scored") is False
            and saved_base_audit.get("plan_sha256") == digest(base / "plan.json"),
            "base audit did not certify the required original strict failure")
    # Revalidate the original logical failure, not just its saved success flag.
    base_report = audit_base(source, base, train_path, test_path, tokenizer_path, partial=False, verify_training=False)
    require(base_report["stage"] == "development_gate_failed" and not base_report["official_test_scored"]
            and len(base_report["training_cases"]) == 14,
            "original study is not the complete seven-configuration strict failure")
    require(not (base / "formal-protocol.json").exists() and not list((base / "formal").glob("*.json")),
            "original official test was already scored")
    protocol = check_protocol(source, base, run, policy_path, plan, base_report)
    candidate = protocol["candidate"]
    result = {
        "audit_passed": True, "partial": partial,
        "scope": "complete saved artifacts only" if partial else "complete supplementary study",
        "disclosure": protocol["disclosure"], "policy_sha256": digest(policy_path),
        "protocol_sha256": digest(run / "protocol.json"), "base_audit_sha256": digest(base / "audit.json"),
        "base_original_rule_stage": base_report["stage"], "base_original_rule_preserved": True,
        "candidate": candidate, "official_test_scored": False,
        "base_screen_training_tokens": base_report["physical_new_grpo_training_tokens"],
        "training_cases": {},
    }
    qualification, formal = run / "qualification", run / "formal"
    formal_path, status_path = run / "formal-protocol.json", run / "development-status.json"
    allowed_formal = {f"seed-{seed}-{arm}.json" for seed in SEEDS for arm in ("grpo", "sft")}
    require({p.name for p in formal.glob("*.json")} <= allowed_formal, "unregistered formal scoring appeared")
    if candidate is None:
        require(not list((run / "models").glob("*/seed-*/training.json"))
                and not list(qualification.glob("*.json")) and not formal_path.exists() and not list(formal.glob("*.json")),
                "supplement computed qualifications or tests despite having no positive-mean candidate")
        if status_path.exists():
            require(read_json(status_path) == {"qualified": False, "official_test_scored": False,
                                               "reason": "no positive screen mean"}, "no-candidate final status changed")
        else:
            require(partial, "missing no-positive-candidate stop status")
        result.update(stage="no_positive_screen_mean", physical_new_grpo_training_tokens=0,
                      copied_grpo_training_tokens=0, distinct_grpo_training_tokens_including_base=result["base_screen_training_tokens"])
        return result
    name, lr, accumulation = candidate["selected"], candidate["lr"], candidate["accumulation_groups"]
    require({p.name for p in (run / "models").glob("*") if p.is_dir()} <= {name},
            "trained another supplementary candidate after the single choice was frozen")
    cases, unfinished = {}, []
    required_case_files = ("training.json", "trace.jsonl", "updates.jsonl", "final-adapter.safetensors")
    for seed in SEEDS:
        folder = run / "models" / name / f"seed-{seed}"
        key = f"{name}/seed-{seed}"
        if not all((folder / f).exists() for f in required_case_files):
            require(partial, f"missing completed confirmation training case {seed}")
            unfinished.append(seed)
            continue
        if seed in (0, 4):
            reused = protocol["base_endpoints"][key]
            require(all((folder / f).exists() and digest(folder / f) == h for f, h in reused["files"].items()),
                    f"seed {seed}: copied original training or screen files changed")
        cases[key] = check_case(folder, plan, train, tokenizer, lr, accumulation, seed)
        training = cases[key][0]
        result["training_cases"][key] = {"accounting": training["accounting"], "final_sha256": training["final_sha256"],
                                          "reused_from_base": seed in (0, 4)}
    require({p.name for p in (run / "models" / name).glob("*") if p.is_dir()} <= {f"seed-{s}" for s in SEEDS},
            "unregistered seed or checkpoint selection appeared in the supplement")
    result["unfinished_seeds"] = unfinished
    result["physical_new_grpo_training_tokens"] = sum(r[0]["accounting"]["generated_tokens"] for key, r in cases.items()
                                                       if not key.endswith(("seed-0", "seed-4")))
    result["copied_grpo_training_tokens"] = sum(r[0]["accounting"]["generated_tokens"] for key, r in cases.items()
                                               if key.endswith(("seed-0", "seed-4")))
    result["distinct_grpo_training_tokens_including_base"] = (
        result["base_screen_training_tokens"] + result["physical_new_grpo_training_tokens"])
    initial_path = qualification / "initial.json"
    gate_path = qualification / "gate.json"
    if not initial_path.exists():
        require(partial and not gate_path.exists() and not formal_path.exists() and not list(formal.glob("*.json")),
                "confirmation lacks its common qualification reference")
        result["stage"] = "confirmation_in_progress"
        if verify_training:
            result["training_replay"] = replay_cases(cases, source, plan, tokenizer_path)
        return result
    ids = protocol["qualification_prompt_ids"]
    initial = read_json(initial_path)
    initial_metrics = check_eval(initial, ids, train_gold, tokenizer, "supplement qualification initial")
    evaluations, evaluation_metrics = {}, {}
    expected_qualification_files = {"initial.json", "metrics.json", "gate.json"} | {f"seed-{s}.json" for s in SEEDS}
    require({p.name for p in qualification.glob("*.json")} <= expected_qualification_files,
            "unregistered extra qualification evaluation appeared")
    for seed in SEEDS:
        path = qualification / f"seed-{seed}.json"
        if path.exists():
            require(f"{name}/seed-{seed}" in cases, "qualification score lacks its matching trained endpoint")
            evaluations[str(seed)] = read_json(path)
            evaluation_metrics[str(seed)] = check_eval(evaluations[str(seed)], ids, train_gold, tokenizer,
                                                       f"supplement qualification seed {seed}")
    result["qualification_initial"] = initial_metrics
    result["qualification_seeds"] = evaluation_metrics
    if not gate_path.exists():
        require(partial and not formal_path.exists() and not list(formal.glob("*.json")),
                "formal scoring preceded the completed five-seed qualification")
        result["stage"] = "qualification_in_progress"
    else:
        require(len(evaluations) == len(cases) == 5, "single qualification gate omitted a seed")
        require(read_json(qualification / "metrics.json") == {"initial": initial_metrics, "seeds": evaluation_metrics},
                "supplementary metrics disagree with independent rescoring")
        stats = comparison({str(s): initial for s in SEEDS}, evaluations, ids)
        passed = (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
                  and stats["bootstrap"]["question_block_median_95"][0] > 0)
        expected_gate = {"passed": passed, "statistics": stats, "selected": name}
        require(read_json(gate_path) == expected_gate, "supplement changed the original qualification gate or its scores")
        result["qualification"] = expected_gate
        if not passed:
            require(not formal_path.exists() and not list(formal.glob("*.json")),
                    "official test scoring followed the failed supplementary qualification")
            if status_path.exists():
                require(read_json(status_path) == {"qualified": False, "official_test_scored": False,
                                                   "reason": "single supplementary qualification failed"},
                        "failed single-qualification status changed")
            else:
                require(partial, "missing final supplementary failure status")
            result["stage"] = "supplementary_qualification_failed"
        else:
            expected_formal = {
                "selected": name, "protocol_sha256": digest(run / "protocol.json"),
                "qualification_gate_sha256": digest(gate_path), "heldout_prompt_ids": plan["heldout_prompt_ids"],
                "grpo_endpoints": {str(s): {"training_sha256": digest(run / "models" / name / f"seed-{s}/training.json"),
                                              "final_sha256": cases[f"{name}/seed-{s}"][0]["final_sha256"]} for s in SEEDS},
                "sft_reference": protocol["sft_reference"], "response_tokens_per_arm_seed": 32768,
            }
            if formal_path.exists():
                require(read_json(formal_path) == expected_formal, "post-qualification formal recipe or endpoints changed")
            else:
                require(partial, "passed qualification has no frozen formal protocol")
            outcomes = {"grpo": {}, "sft": {}}
            for seed in SEEDS:
                for arm in ("grpo", "sft"):
                    path = formal / f"seed-{seed}-{arm}.json"
                    if not path.exists():
                        require(partial, f"formal confirmation missing seed {seed}/{arm}")
                        continue
                    require(formal_path.exists(), "scored a formal question before freezing its recipe")
                    report = read_json(path)
                    training = (cases[f"{name}/seed-{seed}"][0] if arm == "grpo" else
                                read_json(source / f"seed-{seed}/sft/summary.json"))
                    require(report["accounting"] == training["accounting"] and report["final_sha256"] == training["final_sha256"]
                            and training["accounting"]["generated_tokens"] + training["accounting"]["supervised_tokens"] == 32768,
                            "formal endpoint or matched response-token budget changed")
                    check_eval(report["evaluation"], plan["heldout_prompt_ids"], test_gold, tokenizer,
                               f"supplement formal {seed}/{arm}", report["metrics"])
                    outcomes[arm][str(seed)] = report["evaluation"]
            count = sum(map(len, outcomes.values()))
            result["official_test_scored"] = count > 0
            result["stage"] = "formal_complete" if count == 10 else "qualified_formal_in_progress"
            if count == 10:
                result["formal_grpo_vs_sft"] = comparison(outcomes["sft"], outcomes["grpo"], plan["heldout_prompt_ids"])
                result["formal_correct"] = {str(s): {arm: outcomes[arm][str(s)]["correct"] for arm in outcomes} for s in SEEDS}
    if verify_training:
        result["training_replay"] = replay_cases(cases, source, plan, tokenizer_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--base", type=Path, default=Path("runs/gsm8k-stable-v1"))
    parser.add_argument("--run", type=Path, default=Path("runs/gsm8k-stable-confirmation"))
    parser.add_argument("--policy", type=Path, default=Path("runs/gsm8k-stable-confirmation/conditional-policy.json"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("runs/gsm8k-data/test.jsonl"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="validate only completed supplementary artifacts")
    parser.add_argument("--verify-training", action="store_true", help="independently replay every completed supplementary case")
    args = parser.parse_args()
    tokenizer_path = args.tokenizer.parent if args.tokenizer.is_file() else args.tokenizer
    result = audit(args.source, args.base, args.run, args.policy, args.train, args.test,
                   tokenizer_path, args.partial, args.verify_training)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Audited conditional independent GRPO confirmation:", result["stage"])


if __name__ == "__main__":
    main()
