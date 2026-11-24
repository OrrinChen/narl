"""Independently rescore two fixed-checkpoint GSM8K replication panels, without model inference."""

import argparse
import json
from pathlib import Path
from random import Random

from experiments.audit_gsm8k_online_lag import MODEL, REVISION, TRAIN_SHA256, adapter_digest
from experiments.audit_gsm8k_regroup import check_prompt_ids, read_split
from experiments.audit_gsm8k_stable import SEEDS, check_eval, comparison, digest
from experiments.audit_kv_online import read_json, require


RUNS = Path("runs")
DRAW_SEED = 2026112419
PANEL_SIZE = 256
FORMAT = "verified_format"
STRICT = "lr-0.0001-groups-16"
FROZEN_PROTOCOL_SHA256 = "4db00758919349b22e5becc58ddb8bf59254cf8490fc2e7edf8a492509c59ef4"
PRECURSORS = ("gsm8k-reward-online-v2", "gsm8k-reward-composite-v1")
OLD_PROTOCOL_IDS = {
    "training_prompt_ids", "development_prompt_ids", "snapshot_training_prompt_ids", "train_prompt_ids",
    "extra_prompt_ids", "excluded_train_ids", "replay_prompt_ids", "dev_prompt_ids", "probe_prompt_ids",
    "localization_prompt_ids",
}


def check_endpoints(runs, protocol):
    base = runs / "gsm8k-stable-v1"
    source = runs / "gsm8k-budget-5seed"
    fmt = runs / "gsm8k-reward-online-v2" / FORMAT
    plan = read_json(base / "plan.json")
    base_protocol = read_json(source / "protocol.json")
    require(plan["source_protocol_sha256"] == digest(source / "protocol.json")
            and base_protocol["model"] == MODEL and base_protocol["revision"] == REVISION,
            "base model or source protocol changed")
    endpoints = {"initial": adapter_digest(source / "start-adapter.safetensors")}
    require(endpoints["initial"] == base_protocol["start_sha256"] == plan["source_protocol"]["start_sha256"],
            "common initialization differs from the base study")
    q1 = read_json(fmt / "q1/gate.json")
    require(q1["passed"] is False and q1["selected"] == FORMAT and q1["qualification_index"] == 0
            and q1["protocol_sha256"] == digest(fmt / "protocol.json"), "original format Q1 decision changed")
    for seed in SEEDS:
        for prefix, folder in (("", fmt / "models"), ("strict-", base / "models" / STRICT)):
            name = f"{prefix}seed-{seed}"
            checkpoint = folder / f"seed-{seed}"
            report = read_json(checkpoint / "training.json")
            value = adapter_digest(checkpoint / "final-adapter.safetensors")
            require(report["final_sha256"] == value and report["start_sha256"] == endpoints["initial"]
                    and report["seed"] == seed and report["lr"] == 0.0001 and report["accumulation_groups"] == 16
                    and report["accounting"]["generated_tokens"] == 32768, f"{name}: wrong trained endpoint")
            if not prefix:
                require(q1["endpoint_sha256"][str(seed)] == value, f"{name}: not the originally scored candidate")
            endpoints[name] = value
    require(protocol["checkpoint_sha256"] == endpoints, "replication does not bind exactly the eleven fixed adapters")
    return endpoints


def check_draw(registered, excluded, examples):
    require(registered["excluded_ids"] == sorted(excluded), "historical exclusions differ from the source records")
    check_prompt_ids(registered["excluded_ids"], examples, "excluded train prompts")
    questions = [" ".join(question.casefold().split()) for question, _ in examples]
    prior_texts = {questions[i] for i in excluded}
    first_ids = {}
    for i, question in enumerate(questions):
        if question not in prior_texts:
            first_ids.setdefault(question, i)
    available = list(first_ids.values())
    require(len(available) == registered["pool_size"], "eligible question pool changed")
    draw = Random(DRAW_SEED).sample(available, 2 * PANEL_SIZE)
    ids = {"A": draw[:PANEL_SIZE], "B": draw[PANEL_SIZE:]}
    require(registered["panel_ids"] == ids, "panels differ from the predeclared random, disjoint draw")
    texts = {questions[i] for i in draw}
    require(len(texts) == 2 * PANEL_SIZE and texts.isdisjoint(prior_texts),
            "selected question text repeats a historical question or another panel entry")
    return ids


def flatten_ids(values):
    if isinstance(values, dict):
        for item in values.values():
            yield from flatten_ids(item)
    elif isinstance(values, list):
        for item in values:
            yield from flatten_ids(item)
    else:
        yield values


def review_ids(node):
    if isinstance(node, dict):
        if "prompt_id" in node:
            yield node["prompt_id"]
        for key, child in node.items():
            if key not in {"token_ids", "prompt_ids", "behavior_logprobs"}:
                yield from review_ids(child)
    elif isinstance(node, list):
        for item in node:
            yield from review_ids(item)


def trace_ids(path):
    with path.open() as f:
        return {json.loads(line)["prompt_id"] for line in f}


def check_exclusions(runs, protocol):
    base = read_json(runs / "gsm8k-stable-v1/plan.json")
    old, composite = (runs / name for name in PRECURSORS)
    study = read_json(composite / "study.json")
    excluded = set(study["excluded_ids"] + study["confirmation_prompt_ids"])
    require(study["train_schedule_prefix"] == 512 and len(study["confirmation_prompt_ids"]) == 128,
            "composite training prefix or independent confirmation set changed")
    schedules = [base["qualification_excluded_prior_train_ids"], base["screen_prompt_ids"],
                 *base["qualification_prompt_ids"], *(ids[:512] for ids in base["train_prompt_ids"].values())]
    require(all(set(ids) <= excluded for ids in schedules), "old training schedule, Q1 or Q2 missing from exclusions")

    # Independently inventory physically consumed examples. Full unconsumed GRPO/SFT schedule tails stay eligible.
    extra = {}
    def check(path, ids):
        new = set(ids) - excluded
        if new:
            extra[str(path.relative_to(runs))] = {"additional_ids": sorted(new), "sha256": digest(path)}

    for path in runs.glob("gsm8k-*/**/*.jsonl"):
        if path.name in {"trace.jsonl", "sft-trace.jsonl", "groups.jsonl", "samples.jsonl"} and composite not in path.parents:
            check(path, trace_ids(path))
    for path in runs.glob("gsm8k-*/protocol.json"):
        previous = read_json(path)
        check(path, (i for key in OLD_PROTOCOL_IDS for i in flatten_ids(previous.get(key, []))))
    for path in [*runs.glob("gsm8k-*/development.json"), *runs.glob("gsm8k-*/warmup-samples.json"),
                 *runs.glob("gsm8k-composite-reward-review/*.json"), old / "preference-review.json"]:
        record = read_json(path)
        found = set(review_ids(record))
        if path.name == "development.json" and isinstance(record, dict):
            found.update(record.get("prompt_ids", []))
        check(path, found)
    review = old / "parser-review.json"
    check(review, (i for group in read_json(review)["sets"] for i in group["included_prompt_ids"]))
    require(extra == protocol["additional_archive_sources"], "historical exposures or additional archive hashes changed")
    excluded.update(i for record in extra.values() for i in record["additional_ids"])

    traces = {}
    for recipe in ("parsed_outcome", "verified_steps", FORMAT):
        for seed in SEEDS:
            case = old / recipe / "models" / f"seed-{seed}"
            path = case / "trace.jsonl"
            sha = digest(path)
            require(sha == read_json(case / "training.json")["trace_sha256"] and trace_ids(path) <= excluded,
                    f"{recipe}/{seed}: historical training trace or excluded IDs changed")
            traces[f"{recipe}/seed-{seed}"] = sha
    require(traces == protocol["v2_training_trace_sha256"], "fifteen prior online training traces not bound")
    return excluded


def passes(stats):
    return (stats["positive_seeds"] >= 4 and stats["negative_seeds"] == 0 and stats["median_accuracy_gain"] > 0
            and stats["bootstrap"]["question_block_median_95"][0] > 0)


def check_question_blocks(baselines, candidates, stats):
    """Second calculation of the primary CI directly from a question-by-five paired score matrix."""
    import numpy as np

    deltas = np.array([[c["reward"] - b["reward"] for b, c in zip(baselines[str(s)]["rows"],
                                                                  candidates[str(s)]["rows"], strict=True)]
                       for s in SEEDS], dtype=np.int8).T
    n_questions = len(deltas)
    net = deltas.sum(axis=0)
    require(stats["positive_seeds"] == sum(net > 0) and stats["negative_seeds"] == sum(net < 0)
            and stats["median_accuracy_gain"] == np.median(net) / n_questions, "paired seed improvements disagree")
    rng = np.random.default_rng(80161)
    medians = []
    for _ in range(100):
        indices = rng.integers(n_questions, size=(100, n_questions))
        # Each selected row contains all five results for one question.
        per_seed = deltas[indices].mean(axis=1)
        medians.extend(np.median(per_seed, axis=1))
    interval = np.quantile(medians, [0.025, 0.975]).tolist()
    require(stats["bootstrap"]["seed"] == 80161 and stats["bootstrap"]["replicates"] == 10000
            and stats["bootstrap"]["question_block_median_95"] == interval, "question-block interval disagrees")


def audit_panel(folder, ids, endpoints, protocol_hash, examples, tokenizer):
    expected_files = {"gate.json", *(f"{name}.json" for name in endpoints)}
    require({p.name for p in folder.glob("*.json")} == expected_files, f"{folder.name}: incomplete or extra evaluation")
    raw = {}
    for name, checkpoint in endpoints.items():
        record = read_json(folder / f"{name}.json")
        require(record["checkpoint_sha256"] == checkpoint and record["protocol_sha256"] == protocol_hash,
                f"{folder.name}/{name}: checkpoint or protocol changed")
        # Re-read every token, decode with the pinned tokenizer, and score the strict final tag against original gold.
        check_eval(record, ids, examples, tokenizer, f"{folder.name}/{name}")
        raw[name] = record
    initial = {str(s): raw["initial"] for s in SEEDS}
    candidate = {str(s): raw[f"seed-{s}"] for s in SEEDS}
    strict = {str(s): raw[f"strict-seed-{s}"] for s in SEEDS}
    # The stable-study comparison samples one shared vector of question indices per bootstrap replicate.
    statistics = comparison(initial, candidate, ids)
    vs_strict = comparison(strict, candidate, ids)
    check_question_blocks(initial, candidate, statistics)
    check_question_blocks(strict, candidate, vs_strict)
    reported = read_json(folder / "gate.json")
    expected = {"panel": folder.name, "protocol_sha256": protocol_hash, "passed": passes(statistics),
                "statistics": statistics, "vs_strict": vs_strict}
    require(reported == expected, f"{folder.name}: published gate differs from strict row-level rescoring")
    return expected


def check_activation(runs, run, excluded):
    manifest = {}
    for name in PRECURSORS:
        prior = runs / name
        status = read_json(prior / "status.json")
        require(status["stable"] is False, f"{name}: preceding study did not end unconfirmed")
        q1_winner = None
        for recipe in read_json(prior / "study.json")["order"]:
            folder = prior / recipe
            if (folder / "stop.json").is_file():
                stop = read_json(folder / "stop.json")
                require(stop["reason"] == "negative Q1 seed"
                        and read_json(folder / f"q1/seed-{stop['seed']}.json")["correct"] == stop["correct"]
                        < stop["initial_correct"] == read_json(folder / "q1/initial.json")["correct"],
                        f"{name}/{recipe}: early rejection does not match saved candidate and start")
            elif read_json(folder / "q1/gate.json")["passed"]:
                require(read_json(folder / "q2/gate.json")["passed"] is False, f"{name}/{recipe}: confirmation passed")
                q1_winner = recipe
                break
        require(status == {"stable": False, "q2_scored": q1_winner is not None, "selected": q1_winner},
                f"{name}: original study status disagrees with the gates")
        manifest[name] = digest(prior / "status.json")
    activation = {"protocol_sha256": FROZEN_PROTOCOL_SHA256, "source_status_sha256": manifest}
    require(read_json(run / "activation.json") == activation, "scoring is not bound to the prior study decisions")
    for path in (runs / PRECURSORS[1]).glob("*/models/seed-*/trace.jsonl"):
        require(trace_ids(path) <= excluded, "composite training consumed an unexcluded question")
    return activation


def audit(run, runs, train_path, tokenizer_path, protocol_only=False):
    protocol_path = run / "protocol.json"
    require(digest(protocol_path) == FROZEN_PROTOCOL_SHA256, "replication protocol differs from the pre-scoring freeze")
    protocol = read_json(protocol_path)
    base = runs / "gsm8k-stable-v1"
    old, composite = (runs / name for name in PRECURSORS)
    source_files = {"base_plan_sha256": base / "plan.json", "reward_study_sha256": old / "study.json",
                    "composite_study_sha256": composite / "study.json", "q1_gate_sha256": old / FORMAT / "q1/gate.json"}
    require(all(protocol[key] == digest(path) for key, path in source_files.items()), "source study hash changed")
    source = read_json(base / "plan.json")["source_protocol"]
    require(protocol["candidate"] == FORMAT and protocol["seed"] == DRAW_SEED and protocol["seeds"] == SEEDS
            and protocol["train_sha256"] == source["train_sha256"] == TRAIN_SHA256
            and protocol["eval_batch"] == source["eval_batch"] == 12
            and protocol["eval_max_new_tokens"] == source["eval_max_new_tokens"] == 256
            and source["qualification_protocol"]["prompt_mode"] == "reasoning", "evaluation configuration changed")
    examples = read_split(train_path, TRAIN_SHA256)
    excluded = check_exclusions(runs, protocol)
    panels = check_draw(protocol, excluded, examples)
    endpoints = check_endpoints(runs, protocol)
    report = {"audit_passed": True, "protocol_sha256": FROZEN_PROTOCOL_SHA256, "excluded_count": len(excluded),
              "eligible_pool_size": protocol["pool_size"], "panel_sizes": {p: len(ids) for p, ids in panels.items()},
              "checkpoint_sha256": endpoints, "new_training_tokens": 0, "official_test_scored": False}
    if protocol_only:
        return {**report, "stage": "protocol_only"}
    activation = check_activation(runs, run, excluded)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or MODEL, revision=REVISION, local_files_only=True)
    results = {p: audit_panel(run / p, ids, endpoints, FROZEN_PROTOCOL_SHA256, examples, tokenizer)
               for p, ids in panels.items()}
    decisions = {p: result["passed"] for p, result in results.items()}
    summary = {"protocol_sha256": FROZEN_PROTOCOL_SHA256, "panels": decisions, "passed": all(decisions.values())}
    require(read_json(run / "summary.json") == summary, "summary must retain both panel decisions and their conjunction")
    return {**report, "stage": "complete", "activation": activation, "strictly_rescored_answers": len(endpoints) * 512,
            "panels": results, "replication_passed": summary["passed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol-only", action="store_true", help="verify the freeze before the activation or scoring")
    args = parser.parse_args()
    run = args.run or args.runs / "gsm8k-reward-format-replication"
    report = audit(run, args.runs, args.train or args.runs / "gsm8k-data/train.jsonl", args.tokenizer, args.protocol_only)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["stage"], "protocol", report["protocol_sha256"], "replication_passed", report.get("replication_passed"))


if __name__ == "__main__":
    main()
