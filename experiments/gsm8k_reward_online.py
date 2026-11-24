"""Train on fresh GRPO rollouts with a frozen reward recipe and strictly score development answers."""

import argparse
import fcntl
from importlib import import_module
from pathlib import Path

from safetensors.torch import load_file
import torch

from experiments.audit_gsm8k_stable import comparison
from experiments.audit_kv_online import read_json, require
from experiments.gsm8k_budget import evaluate
from experiments.gsm8k_matched import save_summary
from experiments.gsm8k_mlx_development import load_mlx_model, set_adapter
from experiments.gsm8k_regroup import adapter_digest
from experiments.gsm8k_signal import TRAIN_SHA256, read_split
from experiments.gsm8k_stable import SEEDS, digest, load_endpoint, train_case


MODULES = {"parsed_outcome": "experiments.gsm8k_training_reward",
           "verified_steps": "experiments.gsm8k_step_reward", "verified_format": "experiments.gsm8k_step_reward"}
STRICT_RECIPE = "lr-0.0001-groups-16"


def bind(args):
    config = read_json(args.protocol)
    base = read_json(args.base / "plan.json")
    audit = read_json(args.base / "audit.json")
    recipe = config["reward_recipe"]
    v2 = "parser_base_sha256" in config
    outcome_name = "experiments.gsm8k_training_reward_v2" if v2 else MODULES["parsed_outcome"]
    expected_module = outcome_name if recipe == "parsed_outcome" else MODULES[recipe]
    require(config["score_module"] == expected_module, "unregistered reward scorer")
    module = import_module(config["score_module"])
    module_hash = digest(Path(module.__file__))
    require(module_hash == config["reward_module_sha256"], "training reward implementation changed")
    if v2:
        legacy = import_module(MODULES["parsed_outcome"])
        require(digest(Path(legacy.__file__)) == config["parser_base_sha256"], "base reward parser changed")
    if v2 or recipe != "parsed_outcome":
        outcome = import_module(outcome_name)
        require(digest(Path(outcome.__file__)) == config["outcome_module_sha256"], "underlying outcome scorer changed")
    require(config["base_plan_sha256"] == digest(args.base / "plan.json") == audit["plan_sha256"]
            and audit["audit_passed"] and not audit["partial"], "frozen base study or its audit changed")
    require(config["seeds"] == base["seeds"] == SEEDS and config["budget"] == base["token_budget"] == 32768
            and config["lr"] == 0.0001 and config["accumulation_groups"] == 16 and base["group_size"] == 2,
            "expected the matched five-seed G2 recipe")
    require(read_json(args.base / "round-2/selection.json")["selected"] == STRICT_RECIPE,
            "strict-reward control recipe changed")
    require(digest(args.source / "protocol.json") == base["source_protocol_sha256"],
            "starting study differs from original protocol")
    binding = {"protocol_sha256": digest(args.protocol), "base_plan_sha256": config["base_plan_sha256"],
               "reward_recipe": recipe, "reward_module_sha256": module_hash,
               "start_sha256": base["source_protocol"]["start_sha256"]}
    path = args.out / "binding.json"
    if path.exists():
        require(read_json(path) == binding, "output directory belongs to another reward protocol")
    else:
        require(not any((args.out / name).exists() for name in ("models", "q1", "q2")),
                "refusing to reuse unbound training or evaluation artifacts")
        save_summary(path, binding)

    def reward(sample, example):
        if recipe == "verified_format":
            return module.score_format_reward(sample, example)
        return module.score_training_reward(sample, example["value"] if recipe == "parsed_outcome" else example)

    return config, base, audit, binding, reward


def gate(args, base, binding, split):
    folder = args.out / split
    initial_path = folder / "initial.json"
    if not initial_path.exists():
        return None
    if any(not (folder / name).exists() for s in SEEDS for name in (f"seed-{s}.json", f"strict-seed-{s}.json")):
        return None
    initial = read_json(initial_path)
    candidates = {str(s): read_json(folder / f"seed-{s}.json") for s in SEEDS}
    controls = {str(s): read_json(folder / f"strict-seed-{s}.json") for s in SEEDS}
    index = 0 if split == "q1" else 1
    ids = base["qualification_prompt_ids"][index]
    hashes = {}
    for s in SEEDS:
        hashes[str(s)] = read_json(args.out / f"models/seed-{s}/training.json")["final_sha256"]
        strict_hash = read_json(args.base / "models" / STRICT_RECIPE / f"seed-{s}/training.json")["final_sha256"]
        for record, checkpoint_hash in ((initial, binding["start_sha256"]),
                                        (candidates[str(s)], hashes[str(s)]), (controls[str(s)], strict_hash)):
            require(record["checkpoint_sha256"] == checkpoint_hash
                    and record["protocol_sha256"] == binding["protocol_sha256"]
                    and [r["prompt_id"] for r in record["rows"]] == ids, "evaluation belongs to another study or model")
    stats = comparison({str(s): initial for s in SEEDS}, candidates, ids)
    return {"passed": stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0
            and stats["median_accuracy_gain"] > 0 and stats["bootstrap"]["question_block_median_95"][0] > 0,
            "selected": binding["reward_recipe"], "qualification_index": index,
            "protocol_sha256": binding["protocol_sha256"], "endpoint_sha256": hashes,
            "statistics": stats, "vs_strict": comparison(controls, candidates, ids)}


def run(args):
    config, plan, audit, binding, reward = bind(args)
    split = "q2" if args.stage == "q2" else "q1"
    if split == "q2":
        previous = gate(args, plan, binding, "q1")
        require(previous and previous["passed"] and previous == read_json(args.out / "q1/gate.json"),
                "Q2 requires the complete passing Q1 gate for these endpoints")
    train = read_split(args.train, TRAIN_SHA256)
    protocol = plan["source_protocol"]
    ids = plan["qualification_prompt_ids"][int(split == "q2")]
    start = load_file(args.source / "start-adapter.safetensors")
    require(adapter_digest(start) == binding["start_sha256"], "common initialization changed")
    model, tokenizer = load_mlx_model(None, len(protocol["qualification_protocol"]["lora_layers"]), plan["test_seed"])
    strict_q1 = args.base / "round-2/qualification"
    evaluations = args.out / split

    def score(name, checkpoint_hash, load_weights, original=None):
        path = evaluations / f"{name}.json"
        if path.exists():
            data = read_json(path)
        else:
            if original is None:
                load_weights()
                data = evaluate(model, tokenizer, train, ids, protocol)
            else:
                data = read_json(original)
            data = {**data, "checkpoint_sha256": checkpoint_hash, "protocol_sha256": binding["protocol_sha256"]}
            save_summary(path, data)
        require(data["checkpoint_sha256"] == checkpoint_hash and data["protocol_sha256"] == binding["protocol_sha256"]
                and [r["prompt_id"] for r in data["rows"]] == ids, f"cached {split}/{name} belongs to another endpoint")
        print(split, name, data["correct"], "/", len(ids), flush=True)

    if args.stage != "train":
        evaluations.mkdir(exist_ok=True)
        score("initial", binding["start_sha256"], lambda: set_adapter(model, start),
              original=strict_q1 / "initial.json" if split == "q1" else None)
    for seed in args.seeds:
        folder = args.out / f"models/seed-{seed}"
        if args.stage in ("train", "both"):
            train_case(model, tokenizer, train, plan, start, config["lr"], config["accumulation_groups"], seed, folder,
                       training_reward=reward)
        training = read_json(folder / "training.json")
        require(training["seed"] == seed and training["lr"] == config["lr"]
                and training["accumulation_groups"] == config["accumulation_groups"]
                and training["start_sha256"] == binding["start_sha256"]
                and training["accounting"]["generated_tokens"] == config["budget"]
                and training["trace_sha256"] == digest(folder / "trace.jsonl")
                and training["updates_sha256"] == digest(folder / "updates.jsonl"),
                f"seed {seed}: cached training provenance mismatch")
        if args.stage == "train":
            continue
        score(f"seed-{seed}", training["final_sha256"], lambda: load_endpoint(model, folder))
        strict = args.base / "models" / STRICT_RECIPE / f"seed-{seed}"
        sha = audit["training_cases"][f"{STRICT_RECIPE}/seed-{seed}"]["final_sha256"]
        require(sha == read_json(strict / "training.json")["final_sha256"], "strict control endpoint changed")
        score(f"strict-seed-{seed}", sha, lambda: load_endpoint(model, strict),
              original=strict_q1 / f"seed-{seed}.json" if split == "q1" else None)
    if args.stage != "train":
        result = gate(args, plan, binding, split)
        if result is not None:
            path = evaluations / "gate.json"
            if path.exists():
                require(read_json(path) == result, f"archived {split} gate changed")
            else:
                save_summary(path, result)
            print(split, "passed", result["passed"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("runs/gsm8k-stable-v1"))
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--stage", choices=("train", "q1", "q2", "both"), default="both",
                        help="both trains and evaluates Q1; Q2 is a separate explicit invocation")
    parser.add_argument("--seeds", nargs="+", choices=SEEDS, type=int, default=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
