"""Conditional exploratory supplement after partial old-screen observations; unchanged qualification gate."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import copytree
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze(path, value):
    if path.exists():
        require(read(path) == value, f"frozen artifact changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)


def preparation(args):
    """Only read metadata; return before heavy imports while the base study is unfinished."""
    require(args.out.resolve() not in (args.base.resolve(), args.source.resolve()), "supplement must preserve both source studies")
    plan, policy = read(args.base / "plan.json"), read(args.policy)
    ids = sum(plan["qualification_prompt_ids"], [])
    require(policy["base_plan_sha256"] == digest(args.base / "plan.json")
            and policy["qualification_prompt_ids"] == ids and len(set(ids)) == 256
            and policy["heldout_prompt_ids"] == plan["heldout_prompt_ids"]
            and len(plan["heldout_prompt_ids"]) == 192
            and policy["qualification_gate"] == plan["qualification_rule"]
            and policy["token_budget"] == plan["token_budget"] == 32768
            and policy["training_reuse_seeds"] == [0, 4]
            and policy["additional_training_seeds"] == [1, 2, 3]
            and plan["seeds"] == list(range(5)), "conditional policy differs from the frozen base plan")
    for index in (1, 2):
        path = args.base / f"round-{index}/selection.json"
        if not path.exists():
            print("waiting: both original screening rounds must finish", flush=True)
            return None
        if read(path)["selected"] is not None:
            print("ineligible: original study selected a candidate; supplement will never run", flush=True)
            return None
    status, audit_path = args.base / "development-status.json", args.base / "audit.json"
    if not status.exists() or not audit_path.exists():
        print("waiting: original strict failure status and full audit are required", flush=True)
        return None
    require(read(status) == {"qualified": False, "official_test_scored": False, "completed_rounds": 2},
            "original development study did not finish with strict failure")
    audit = read(audit_path)
    if audit.get("partial") is not False:
        print("waiting: original full audit has not finished", flush=True)
        return None
    require(audit.get("audit_passed") is True and audit.get("stage") == "development_gate_failed"
            and audit.get("official_test_scored") is False
            and audit.get("plan_sha256") == digest(args.base / "plan.json"), "base audit did not certify strict failure")
    require(not any((args.base / f"round-{i}/qualification").exists() for i in (1, 2))
            and not (args.base / "formal-protocol.json").exists()
            and not list((args.base / "formal").glob("*.json")), "reserved qualification or official test was already scored")
    require(digest(args.source / "protocol.json") == plan["source_protocol_sha256"], "source protocol changed")
    source_audit = read(args.source / "audit.json")
    require(source_audit["audit_passed"] and source_audit["model_replay"]["replayed_seeds"] == 5,
            "SFT reference lacks its five-seed replay audit")
    initial = read(args.base / "screen-initial.json")
    hashes = {"screen-initial.json": digest(args.base / "screen-initial.json")}
    candidates, endpoints, references = [], {}, {}
    for index, configs in enumerate(plan["rounds"], 1):
        screen = read(args.base / f"round-{index}/screen.json")
        for filename in ("screen.json", "selection.json"):
            relative = f"round-{index}/{filename}"
            hashes[relative] = digest(args.base / relative)
        for lr, groups in configs:
            name = f"lr-{lr:g}-groups-{groups}"
            entry = screen[name]
            require(entry["lr"] == lr and entry["accumulation_groups"] == groups
                    and set(entry["seeds"]) == {"0", "4"}, "incomplete or altered screening recipe")
            deltas = [entry["seeds"][str(s)]["metrics"]["correct"] - initial["correct"] for s in (0, 4)]
            candidates.append({"selected": name, "lr": lr, "accumulation_groups": groups,
                               "mean_gain": mean(deltas), "worst_gain": min(deltas), "deltas": deltas})
            for seed in (0, 4):
                folder = args.base / "models" / name / f"seed-{seed}"
                report = read(folder / "training.json")
                require(read(folder / "screen.json") == entry["seeds"][str(seed)]
                        and report["final_sha256"] == entry["seeds"][str(seed)]["final_sha256"], "screen endpoint changed")
                endpoints[f"{name}/seed-{seed}"] = {"final_sha256": report["final_sha256"], "files": {
                    f: digest(folder / f) for f in ("training.json", "final-adapter.safetensors", "trace.jsonl", "updates.jsonl", "screen.json")}}
    require(len(candidates) == 7, "expected exactly the seven original configurations")
    candidates.sort(key=lambda c: (c["mean_gain"], c["worst_gain"], -c["lr"], c["accumulation_groups"], c["selected"]), reverse=True)
    for seed in range(5):
        folder = args.source / f"seed-{seed}/sft"
        report, reference = read(folder / "summary.json"), plan["sft_reference"][str(seed)]
        require(digest(folder / "summary.json") == reference["summary_sha256"]
                and report["final_sha256"] == reference["final_sha256"]
                and report["accounting"] == reference["accounting"], "audited SFT reference changed")
        references[str(seed)] = {**reference, "endpoint_file_sha256": digest(folder / "final-adapter.safetensors")}
    chosen = candidates[0] if candidates[0]["mean_gain"] > 0 else None
    protocol = {"disclosure": __doc__, "base_plan_sha256": digest(args.base / "plan.json"),
                "base_audit_sha256": digest(audit_path), "policy_sha256": digest(args.policy),
                "source_protocol_sha256": plan["source_protocol_sha256"], "source_audit_sha256": digest(args.source / "audit.json"),
                "start_file_sha256": digest(args.source / "start-adapter.safetensors"),
                "base_screen_sha256": hashes, "base_endpoints": endpoints, "sft_reference": references,
                "candidate": chosen, "screen_ranking": candidates, "seeds": list(range(5)),
                "qualification_prompt_ids": ids, "qualification_rule": plan["qualification_rule"],
                "heldout_prompt_ids": plan["heldout_prompt_ids"], "token_budget": 32768,
                "stopping_rule": policy["stopping_rule"], "training_reuse_seeds": [0, 4],
                "additional_training_seeds": [1, 2, 3], "planned_new_grpo_training_tokens": 3 * 32768 if chosen else 0,
                "planned_formal_grpo_training_tokens_if_qualified": 5 * 32768 if chosen else 0,
                "cost_scope": "Copied seeds cost zero new training tokens; common initialization, prior screening and reused SFT are separate; response tokens do not match FLOPs."}
    if not (args.out / "selection.json").exists():
        require(not list((args.out / "qualification").glob("*.json"))
                and not (args.out / "formal-protocol.json").exists()
                and not list((args.out / "formal").glob("*.json"))
                and not list((args.out / "models").glob("*/seed-*/training.json")),
                "supplementary computation preceded its frozen selection")
    freeze(args.out / "protocol.json", protocol)
    freeze(args.out / "selection.json", {"selected": chosen["selected"] if chosen else None,
                                        "protocol_sha256": digest(args.out / "protocol.json")})
    return plan, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/gsm8k-stable-v1"))
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--policy", type=Path, default=Path("runs/gsm8k-stable-confirmation/conditional-policy.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/gsm8k-stable-confirmation"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("runs/gsm8k-data/test.jsonl"))
    parser.add_argument("--stage", choices=("plan", "qualify", "all"), default="all")
    args = parser.parse_args()
    prepared = preparation(args)
    if prepared is None or args.stage == "plan":
        return
    plan, protocol = prepared
    candidate = protocol["candidate"]
    if candidate is None:
        freeze(args.out / "development-status.json", {"qualified": False, "official_test_scored": False, "reason": "no positive screen mean"})
        print("stopped: no original recipe has positive paired mean gain", flush=True)
        return
    gate_path = args.out / "qualification/gate.json"
    if gate_path.exists() and read(gate_path)["passed"] is False:
        require(not (args.out / "formal-protocol.json").exists() and not list((args.out / "formal").glob("*.json")), "formal scoring followed failed qualification")
        freeze(args.out / "development-status.json", {"qualified": False, "official_test_scored": False, "reason": "single supplementary qualification failed"})
        print("stopped: supplementary qualification already failed; no retry", flush=True)
        return
    from safetensors.torch import load_file
    import torch
    from experiments import gsm8k_stable as stable

    torch.set_num_threads(1)
    train = stable.read_split(args.train, stable.TRAIN_SHA256)
    start = load_file(args.source / "start-adapter.safetensors")
    require(stable.adapter_digest(start) == plan["source_protocol"]["start_sha256"], "common initialization changed")
    name, lr, groups = candidate["selected"], candidate["lr"], candidate["accumulation_groups"]
    for seed in (0, 4):
        source, target = args.base / "models" / name / f"seed-{seed}", args.out / "models" / name / f"seed-{seed}"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            copytree(source, target)
        require(all(digest(target / f) == h for f, h in protocol["base_endpoints"][f"{name}/seed-{seed}"]["files"].items()),
                "reused screening training files changed")
    model, tokenizer = stable.load_mlx_model(None, len(plan["source_protocol"]["qualification_protocol"]["lora_layers"]), stable.TEST_SEED)
    qualification = args.out / "qualification"
    qualification.mkdir(exist_ok=True)
    initial_path = qualification / "initial.json"
    if not initial_path.exists():
        stable.set_adapter(model, start)
        stable.save_summary(initial_path, stable.evaluate(model, tokenizer, train, protocol["qualification_prompt_ids"], plan["source_protocol"]))
    initial, evaluations = read(initial_path), {}
    for seed in range(5):
        folder = args.out / "models" / name / f"seed-{seed}"
        if seed not in (0, 4):
            stable.train_case(model, tokenizer, train, plan, start, lr, groups, seed, folder)
        target = qualification / f"seed-{seed}.json"
        if not target.exists():
            stable.load_endpoint(model, folder)
            stable.save_summary(target, stable.evaluate(model, tokenizer, train, protocol["qualification_prompt_ids"], plan["source_protocol"]))
        evaluations[str(seed)] = read(target)
        print("qualification", seed, stable.output_metrics(evaluations[str(seed)]), "initial", initial["correct"], flush=True)
    stats = stable.pair_stats(initial, evaluations, 256)
    passed = (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
              and stats["bootstrap"]["question_block_median_95"][0] > 0)
    freeze(qualification / "metrics.json", {"initial": stable.output_metrics(initial),
                                           "seeds": {s: stable.output_metrics(e) for s, e in evaluations.items()}})
    freeze(qualification / "gate.json", {"passed": passed, "statistics": stats, "selected": name})
    if not passed:
        require(not (args.out / "formal-protocol.json").exists() and not list((args.out / "formal").glob("*.json")), "formal scoring preceded qualification")
        freeze(args.out / "development-status.json", {"qualified": False, "official_test_scored": False, "reason": "single supplementary qualification failed"})
        print("stopped: unchanged qualification gate failed; no official test scored", flush=True)
        return
    formal = {"selected": name, "protocol_sha256": digest(args.out / "protocol.json"),
              "qualification_gate_sha256": digest(qualification / "gate.json"), "heldout_prompt_ids": plan["heldout_prompt_ids"],
              "grpo_endpoints": {str(s): {"training_sha256": digest(args.out / "models" / name / f"seed-{s}/training.json"),
                                           "final_sha256": read(args.out / "models" / name / f"seed-{s}/training.json")["final_sha256"]} for s in range(5)},
              "sft_reference": protocol["sft_reference"], "response_tokens_per_arm_seed": 32768}
    freeze(args.out / "formal-protocol.json", formal)
    if args.stage == "qualify":
        return
    test = stable.read_split(args.test, stable.TEST_SHA256)
    for seed in range(5):
        for arm in ("grpo", "sft"):
            target = args.out / "formal" / f"seed-{seed}-{arm}.json"
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                continue
            folder = args.out / "models" / name / f"seed-{seed}" if arm == "grpo" else args.source / f"seed-{seed}/sft"
            training = read(folder / ("training.json" if arm == "grpo" else "summary.json"))
            require(training["accounting"]["generated_tokens"] + training["accounting"]["supervised_tokens"] == 32768, "formal response budget changed")
            stable.load_endpoint(model, folder, training)
            evaluation = stable.evaluate(model, tokenizer, test, plan["heldout_prompt_ids"], plan["source_protocol"])
            stable.save_summary(target, {"evaluation": evaluation, "metrics": stable.output_metrics(evaluation),
                                         "final_sha256": training["final_sha256"], "accounting": training["accounting"]})
            print("formal", seed, arm, stable.output_metrics(evaluation), flush=True)


if __name__ == "__main__":
    main()
