"""Freeze two new training-split panels and evaluate fixed format-reward checkpoints."""

import argparse
import fcntl
from pathlib import Path
from random import Random

from experiments.audit_gsm8k_online_lag import adapter_digest, TRAIN_SHA256
from experiments.audit_gsm8k_stable import comparison, check_eval
from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_reward_study import digest, record


SEEDS = range(5)
DRAW_SEED = 2026112419
STRICT = "lr-0.0001-groups-16"
CANDIDATE = "verified_format"
ARCHIVE_FIELDS = ("training_prompt_ids", "development_prompt_ids", "snapshot_training_prompt_ids",
                  "train_prompt_ids", "extra_prompt_ids", "excluded_train_ids", "replay_prompt_ids",
                  "dev_prompt_ids", "probe_prompt_ids", "localization_prompt_ids")


def row_ids(obj):
    if isinstance(obj, dict):
        if "prompt_id" in obj:
            yield obj["prompt_id"]
        for key, child in obj.items():
            if key not in ("token_ids", "prompt_ids", "behavior_logprobs"):
                yield from row_ids(child)
    if isinstance(obj, list):
        for child in obj:
            if isinstance(child, (dict, list)):
                yield from row_ids(child)


def flatten_ids(obj):
    if isinstance(obj, dict):
        for values in obj.values():
            yield from flatten_ids(values)
    elif isinstance(obj, list):
        for value in obj:
            yield from flatten_ids(value)
    else:
        yield obj


def exposures(runs, excluded):
    """Find extra prior train exposures in rollout traces, SFT traces and development archives."""
    extras = {}

    def add(path, ids):
        extra = sorted(set(ids) - excluded)
        if extra:
            extras[str(path.relative_to(runs))] = {"additional_ids": extra, "sha256": digest(path)}

    for path in sorted(runs.glob("gsm8k-*/**/*.jsonl")):
        if path.name in ("trace.jsonl", "sft-trace.jsonl", "samples.jsonl", "groups.jsonl") and (
                "gsm8k-reward-composite-v1" not in path.parts):
            add(path, (row["prompt_id"] for row in read_rows(path)))
    for path in sorted(runs.glob("gsm8k-*/protocol.json")):
        old = read_json(path)
        add(path, (pid for key in ARCHIVE_FIELDS for pid in flatten_ids(old.get(key, []))))
    archives = [*runs.glob("gsm8k-*/development.json"), *runs.glob("gsm8k-*/warmup-samples.json"),
                *runs.glob("gsm8k-composite-reward-review/*.json"),
                runs / "gsm8k-reward-online-v2/preference-review.json"]
    for path in sorted(archives):
        add(path, row_ids(read_json(path)))
    parser = runs / "gsm8k-reward-online-v2/parser-review.json"
    add(parser, (pid for group in read_json(parser)["sets"] for pid in group["included_prompt_ids"]))
    return extras


def checkpoint_paths(runs):
    paths = {"initial": runs / "gsm8k-budget-5seed/start-adapter.safetensors"}
    for seed in SEEDS:
        paths[f"seed-{seed}"] = runs / f"gsm8k-reward-online-v2/{CANDIDATE}/models/seed-{seed}/final-adapter.safetensors"
        paths[f"strict-seed-{seed}"] = runs / f"gsm8k-stable-v1/models/{STRICT}/seed-{seed}/final-adapter.safetensors"
    return paths


def freeze(runs, out):
    base = runs / "gsm8k-stable-v1"
    old = runs / "gsm8k-reward-online-v2"
    supplemental = runs / "gsm8k-reward-composite-v1"
    plan = read_json(base / "plan.json")
    study = read_json(supplemental / "study.json")
    q1 = old / CANDIDATE / "q1/gate.json"
    gate = read_json(q1)
    old_scores = gate["statistics"]
    require(not gate["passed"] and old_scores["positive_seeds"] == 5
            and old_scores["bootstrap"]["question_block_median_95"][0] == 0, "format Q1 result changed")
    require(study["source_prior_study_sha256"] == digest(old / "study.json")
            and study["base_plan_sha256"] == digest(base / "plan.json"), "precursor protocols changed")
    data_path = runs / "gsm8k-data/train.jsonl"
    require(digest(data_path) == study["source_train_sha256"] == TRAIN_SHA256, "train dataset changed")
    data = read_rows(data_path)
    excluded = set(study["excluded_ids"]) | set(study["confirmation_prompt_ids"])
    for ids in [*plan["qualification_prompt_ids"], plan["screen_prompt_ids"],
                plan["qualification_excluded_prior_train_ids"],
                *(ids[:study["train_schedule_prefix"]] for ids in plan["train_prompt_ids"].values())]:
        require(set(ids) <= excluded, "historical exclusions or supplemental training pool missing")
    extras = exposures(runs, excluded)
    excluded.update(pid for source in extras.values() for pid in source["additional_ids"])
    traces = {}
    for name in read_json(old / "study.json")["order"]:
        for s in SEEDS:
            path = old / name / f"models/seed-{s}/trace.jsonl"
            traces[f"{name}/seed-{s}"] = digest(path)
            require(traces[f"{name}/seed-{s}"] == read_json(path.with_name("training.json"))["trace_sha256"]
                    and all(row["prompt_id"] in excluded for row in read_rows(path)), "prior online trace changed")
    require(all(type(i) is int and 0 <= i < len(data) for i in excluded), "invalid archived question ID")
    question = lambda i: " ".join(data[i]["question"].casefold().split())
    seen = {question(i) for i in excluded}
    pool = []
    for i in range(len(data)):
        if question(i) not in seen:
            pool.append(i)
            seen.add(question(i))
    ids = Random(DRAW_SEED).sample(pool, 512)
    weights = {name: adapter_digest(path) for name, path in checkpoint_paths(runs).items()}
    require(weights["initial"] == plan["source_protocol"]["start_sha256"], "start checkpoint changed")
    for s in SEEDS:
        require(weights[f"seed-{s}"] == gate["endpoint_sha256"][str(s)]
                and weights[f"strict-seed-{s}"] == read_json(base / "models" / STRICT / f"seed-{s}/training.json")["final_sha256"],
                "archived checkpoint mismatch")
    protocol = {"candidate": CANDIDATE, "seed": DRAW_SEED, "panel_ids": {"A": ids[:256], "B": ids[256:]},
                "excluded_ids": sorted(excluded), "pool_size": len(pool), "additional_archive_sources": extras,
                "v2_training_trace_sha256": traces, "checkpoint_sha256": weights,
                "base_plan_sha256": digest(base / "plan.json"), "reward_study_sha256": digest(old / "study.json"),
                "composite_study_sha256": digest(supplemental / "study.json"), "q1_gate_sha256": digest(q1),
                "train_sha256": TRAIN_SHA256, "seeds": list(SEEDS), "eval_batch": 12, "eval_max_new_tokens": 256,
                "selection_context": "The candidate was chosen after its five known Q1 improvements and zero lower bound.",
                "activation": "Score only after both preceding studies have completed with stable=false.",
                "gate": "Both panels independently require >=4 positive seeds versus the common start, none negative, positive paired median and positive lower question-block 95% bound. Score both panels once. Also report the paired comparison to strict reward.",
                "metric": "Strict original greedy exact match on each training-split panel."}
    target = out / "protocol.json"
    if not target.exists():
        require(not any((out / name).exists() for name in ("A", "B", "chunks", "summary.json")), "existing unbound evaluations")
    record(target, protocol)
    return protocol


def activation(runs):
    states = {}
    for name in ("gsm8k-reward-online-v2", "gsm8k-reward-composite-v1"):
        root = runs / name
        status = read_json(root / "status.json")
        require(status["stable"] is False, "the previous study confirmed a stable candidate")
        for recipe in read_json(root / "study.json")["order"]:
            folder = root / recipe
            if (folder / "stop.json").exists():
                stop = read_json(folder / "stop.json")
                seed = stop["seed"]
                require(read_json(folder / f"q1/seed-{seed}.json")["correct"] == stop["correct"] < stop["initial_correct"]
                        == read_json(folder / "q1/initial.json")["correct"], "invalid negative-seed stop")
                continue
            if read_json(folder / "q1/gate.json")["passed"]:
                require(status["selected"] == recipe and status["q2_scored"]
                        and not read_json(folder / "q2/gate.json")["passed"], "previous confirmation unresolved")
                break
        else:
            require(status == {"selected": None, "q2_scored": False, "stable": False}, "inconsistent study status")
        states[name] = digest(root / "status.json")
    return states


def evaluate_fixed(runs, out, protocol, panels, endpoints):
    from safetensors.torch import load_file
    import torch
    from experiments.gsm8k_budget import evaluate
    from experiments.gsm8k_mlx_development import load_mlx_model, set_adapter
    from experiments.gsm8k_signal import read_split

    statuses = activation(runs)
    protocol_hash = digest(out / "protocol.json")
    activation_path = out / "activation.json"
    record(activation_path, {"protocol_sha256": protocol_hash, "source_status_sha256": statuses})
    plan = read_json(runs / "gsm8k-stable-v1/plan.json")
    source = plan["source_protocol"]
    require(source["eval_batch"] == 12 and source["eval_max_new_tokens"] == 256, "evaluation settings changed")
    train = read_split(runs / "gsm8k-data/train.jsonl", TRAIN_SHA256)
    gold = [(row["question"], float(row["value"])) for row in train]
    # Any composite training after protocol freeze is constrained to excluded schedule prefixes.
    reserved = {i for ids in protocol["panel_ids"].values() for i in ids}
    for path in (runs / "gsm8k-reward-composite-v1").glob("*/models/seed-*/trace.jsonl"):
        require(not reserved.intersection(row["prompt_id"] for row in read_rows(path)), "composite training touched panels")
    torch.set_num_threads(1)
    model, tokenizer = load_mlx_model(None, len(source["qualification_protocol"]["lora_layers"]), plan["test_seed"])
    paths = checkpoint_paths(runs)
    for name in endpoints:
        set_adapter(model, load_file(paths[name]))
        binding = {"protocol_sha256": protocol_hash, "checkpoint_sha256": protocol["checkpoint_sha256"][name]}
        for panel in panels:
            folder = out / panel
            folder.mkdir(exist_ok=True)
            target = folder / f"{name}.json"
            ids = protocol["panel_ids"][panel]
            if target.exists():
                data = read_json(target)
                require(all(data[k] == v for k, v in binding.items()), "cached endpoint binding changed")
                check_eval(data, ids, gold, tokenizer, f"{panel}/{name}")
                continue
            scratch = out / "chunks" / panel / name
            scratch.mkdir(parents=True, exist_ok=True)
            rows = []
            for offset in range(0, len(ids), 12):
                chunk = scratch / f"{offset}.json"
                block = ids[offset:offset + 12]
                if not chunk.exists():
                    data = {**evaluate(model, tokenizer, train, block, source), **binding}
                    record(chunk, data)
                data = read_json(chunk)
                require(all(data[k] == v for k, v in binding.items()), "cached block binding changed")
                check_eval(data, block, gold, tokenizer, f"{panel}/{name}/{offset}")
                rows.extend(data["rows"])
            full = {"rows": rows, "correct": sum(x["reward"] for x in rows),
                    "tagged": sum(x["has_answer_tag"] for x in rows), "truncated": sum(x["truncated"] for x in rows), **binding}
            record(target, full)
            for offset in range(0, len(ids), 12):
                (scratch / f"{offset}.json").unlink(missing_ok=True)
            print(panel, name, full["correct"], "/", len(ids), flush=True)
    panels_passed = {}
    for panel, ids in protocol["panel_ids"].items():
        folder = out / panel
        if not all((folder / f"{name}.json").exists() for name in paths):
            continue
        data = {name: read_json(folder / f"{name}.json") for name in paths}
        for name, evaluation in data.items():
            require(evaluation["protocol_sha256"] == protocol_hash and
                    evaluation["checkpoint_sha256"] == protocol["checkpoint_sha256"][name], "gate checkpoint changed")
            check_eval(evaluation, ids, gold, tokenizer, f"{panel}/{name}")
        selected = {str(s): data[f"seed-{s}"] for s in SEEDS}
        stats = comparison({str(s): data["initial"] for s in SEEDS}, selected, ids)
        strict = comparison({str(s): data[f"strict-seed-{s}"] for s in SEEDS}, selected, ids)
        passed = (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
                  and stats["bootstrap"]["question_block_median_95"][0] > 0)
        record(folder / "gate.json", {"panel": panel, "protocol_sha256": protocol_hash, "passed": passed,
                                    "statistics": stats, "vs_strict": strict})
        panels_passed[panel] = passed
    if len(panels_passed) == 2:
        record(out / "summary.json", {"protocol_sha256": protocol_hash, "panels": panels_passed,
                                     "passed": all(panels_passed.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/gsm8k-reward-format-replication"))
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--panels", nargs="+", choices=("A", "B"), default=["A", "B"])
    names = list(checkpoint_paths(Path(".")))
    parser.add_argument("--endpoints", nargs="+", choices=names, default=names)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".runner.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = freeze(args.runs, args.out)
        if args.stage == "evaluate":
            evaluate_fixed(args.runs, args.out, protocol, args.panels, args.endpoints)
        else:
            print("Frozen 11 checkpoints and two 256-question panels.", digest(args.out / "protocol.json"))


if __name__ == "__main__":
    main()
