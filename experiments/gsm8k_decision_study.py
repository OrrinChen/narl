"""Run and audit the registered GSM8K decision-head inference study on Apple Silicon."""

import argparse
import fcntl
import json
import subprocess
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from time import perf_counter

from experiments.gsm8k_decision_plan import OUT, PRIOR, freeze
from experiments.gsm8k_reward_study import digest, read, record


def historical_rows(protocol, data):
    from tasks import extract_answer, gsm8k_reward

    rows = []
    for relative, expected in sorted(protocol["trace_sha256"].items()):
        path = PRIOR / relative
        if digest(path) != expected:
            raise ValueError(f"Historical training trace changed: {path}")
        for line in path.read_bytes().splitlines():
            group = json.loads(line)
            i = group["prompt_id"]
            for sample in group["samples"]:
                y = int(gsm8k_reward(None, sample["completion"], data[i]))
                if y != sample["strict_reward"] or sample["has_answer_tag"] != (
                    extract_answer(sample["completion"]) is not None
                ):
                    raise ValueError("Invalid historical label")
                rows.append({"prompt_id": i, "prompt_ids": group["prompt_ids"], **sample})
    return rows


def question_brier(ids, probabilities, labels):
    sq = defaultdict(list)
    for i, p, y in zip(ids, probabilities, labels, strict=True):
        sq[int(i)].append(float((p - y) ** 2))
    return fmean(fmean(values) for values in sq.values())


def train_head(out, protocol, data):
    import numpy as np
    import torch

    from experiments.gsm8k_decision_head import FrozenEncoder, encode_features, fit_head, predict_proba

    torch.set_num_threads(1)
    started = perf_counter()
    rows = historical_rows(protocol, data)
    ids = np.array([r["prompt_id"] for r in rows])
    train = np.isin(ids, protocol["hist_train_ids"])
    dev = np.isin(ids, protocol["hist_dev_ids"])
    if not np.all(train ^ dev) or set(ids) & set(protocol["eval_sets"]["holdout"]):
        raise ValueError("Unexpected question overlap or missing training rows")
    X, timing = encode_features(FrozenEncoder.load(), rows, batch_size=8, cache_path=out / "cache/history.npz")
    y = np.array([r["strict_reward"] for r in rows])
    heads, scores = {}, {}
    for l2 in protocol["l2_grid"]:
        head = fit_head(X[train], y[train], l2)
        heads[l2] = head
        scores[l2] = question_brier(ids[dev], predict_proba(head, X[dev]), y[dev])
    selected = min(scores, key=lambda l2: (scores[l2], -l2))
    record(out / "head.json", heads[selected])
    lock = {
        "protocol_sha256": digest(out / "protocol.json"),
        "head_sha256": digest(out / "head.json"),
        "train_questions": len(set(ids[train])),
        "dev_questions": len(set(ids[dev])),
        "train_candidates": int(sum(train)),
        "dev_candidates": int(sum(dev)),
        "selected_l2": selected,
        "dev_brier_by_l2": {str(k): v for k, v in scores.items()},
        "feature_binding": timing["feature_binding"],
    }
    record(out / "head-lock.json", lock)
    if not (out / "fit-timing.json").exists():
        record(
            out / "fit-timing.json",
            {k: v for k, v in timing.items() if k != "row_encoder_seconds"}
            | {"fit_stage_wall_s": perf_counter() - started},
        )
    print("Locked historical head", lock, flush=True)


def candidate_path(out, seed, q, slot):
    return out / "candidates" / "holdout" / f"seed-{seed}" / f"{q}-{slot}.json"


def locked_head(out):
    lock = read(out / "head-lock.json")
    if lock["protocol_sha256"] != digest(out / "protocol.json") or lock["head_sha256"] != digest(
        out / "head.json"
    ):
        raise ValueError("Unbound decision head")
    return read(out / "head.json")


def score_pool(out, protocol, verify=False):
    import numpy as np

    from experiments.gsm8k_decision_head import FrozenEncoder, encode_features, predict_proba

    head = locked_head(out)
    encoder = FrozenEncoder.load()
    dest = out / "scoring"
    dest.mkdir(exist_ok=True)
    for seed in protocol["seeds"]:
        target = dest / f"seed-{seed}.json"
        paths = [
            candidate_path(out, seed, q, slot) for q in protocol["eval_sets"]["holdout"] for slot in (0, 1)
        ]
        rows = [read(path) for path in paths]
        hashes = [digest(path) for path in paths]
        if target.is_file():
            prev = read(target)
            if (
                prev["protocol_sha256"] != digest(out / "protocol.json")
                or prev["head_sha256"] != digest(out / "head.json")
                or [r["candidate_sha256"] for r in prev["rows"]] != hashes
            ):
                raise ValueError("Cached scores are bound to different study inputs")
            if not verify:
                continue
        start = perf_counter()
        X, m = encode_features(encoder, rows, batch_size=1)
        probs = predict_proba(head, X)
        wall = perf_counter() - start
        if verify:
            ref = read(target)
            if not np.array_equal(probs, [r["decision_score"] for r in ref["rows"]]):
                raise ValueError("Frozen decision predictions could not be reproduced")
            print("Verified all scores for seed", seed, flush=True)
            continue
        scores = [
            {
                "prompt_id": row["prompt_id"],
                "slot": row["slot"],
                "decision_score": float(p),
                "candidate_sha256": h,
                "decision_s": s,
            }
            for row, p, h, s in zip(rows, probs, hashes, m["row_encoder_seconds"], strict=True)
        ]
        record(
            target,
            {
                "protocol_sha256": digest(out / "protocol.json"),
                "head_sha256": digest(out / "head.json"),
                "rows": scores,
                "wall_s": wall,
                "feature_binding": m["feature_binding"],
            },
        )
        print("Scored", seed, len(scores), "answers in", round(wall, 1), "seconds", flush=True)
    if verify:
        record(
            out / "score-check.json",
            {
                "scores_verified": True,
                "scores": sum(len(read(f)["rows"]) for f in dest.glob("seed-*.json")),
                "head_sha256": digest(out / "head.json"),
                "files_sha256": [digest(dest / f"seed-{s}.json") for s in protocol["seeds"]],
            },
        )


def pool_records(out, protocol, seeds, ids):
    rows = []
    for seed in seeds:
        scores = read(out / "scoring" / f"seed-{seed}.json")
        if scores["head_sha256"] != digest(out / "head.json") or scores["protocol_sha256"] != digest(
            out / "protocol.json"
        ):
            raise ValueError("Wrong score bindings")
        by_key = {(r["prompt_id"], r["slot"]): r for r in scores["rows"]}
        for q in ids:
            pair = []
            for slot in (0, 1):
                path = candidate_path(out, seed, q, slot)
                row, prediction = read(path), by_key[q, slot]
                if digest(path) != prediction["candidate_sha256"]:
                    raise ValueError("Replayed candidate has changed")
                pair.append(
                    row
                    | {"decision_score": prediction["decision_score"], "decision_s": prediction["decision_s"]}
                )
            rows.append({"prompt_id": q, "seed": seed, "samples": pair})
    return rows


def analyze(out, protocol, data):
    from transformers import AutoTokenizer

    from experiments.gsm8k_decision_candidates import truncate_candidate
    from experiments.gsm8k_decision_eval import evaluate

    locked_head(out)
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model"], revision=protocol["revision"], local_files_only=True
    )
    ids = protocol["eval_sets"]["holdout"]
    rows = pool_records(out, protocol, protocol["seeds"], ids)
    result = evaluate(
        rows,
        len(ids) * protocol["retry_tokens_per_question"],
        lambda r, s, n: truncate_candidate(tokenizer, data[r["prompt_id"]], s, n),
        protocol["uniform_seed"],
        protocol["bootstrap_seed"],
    )
    p, y = [], []
    for row in rows:
        for s in row["samples"]:
            p.append(s["decision_score"])
            y.append(s["strict_reward"])
    result["calibration"] = {
        "brier": fmean((a - b) ** 2 for a, b in zip(p, y, strict=True)),
        "n": len(y),
        "average_probability": fmean(p),
        "fraction_correct": fmean(y),
    }
    result["gate"] = {}
    for name, branch in (
        ("judge_vs_rule", "fixed_pair_contrasts"),
        ("judge_vs_confidence", "allocation_contrasts"),
    ):
        s = result[branch][name]
        result["gate"][name] = (
            s["question_block_median_95"][0] > 0 and s["positive_seeds"] >= 4 and s["negative_seeds"] == 0
        )
    result["protocol_sha256"] = digest(out / "protocol.json")
    result["head_sha256"] = digest(out / "head.json")
    record(out / "results.json", result)
    print("Results", result["gate"], flush=True)


def live_timing(out, protocol, data):
    from experiments.gsm8k_decision_candidates import load_frozen, sample_candidate, truncate_candidate
    from experiments.gsm8k_decision_eval import allocate, retry_order, rule_select
    from experiments.gsm8k_decision_head import FrozenEncoder, encode_features, predict_proba

    head = locked_head(out)
    seed, ids = protocol["clock_seed"], protocol["clock_ids"]
    model, tokenizer = load_frozen(protocol)
    encoder = FrozenEncoder(
        model, tokenizer, {k: protocol[k] for k in ("model", "revision", "start_sha256", "mlx", "mlx_lm")}
    )
    budget = len(ids) * protocol["retry_tokens_per_question"]
    replay = {r["prompt_id"]: r for r in pool_records(out, protocol, [seed], ids)}
    trunc = lambda r, s, n: truncate_candidate(tokenizer, data[r["prompt_id"]], s, n)

    def score(row):
        X, _ = encode_features(encoder, [row], batch_size=1)
        return float(predict_proba(head, X)[0])

    def same_outcome(row, expected):
        return all(
            row[k] == expected[k] for k in ("token_ids", "behavior_logprobs", "strict_reward", "has_answer_tag")
        )

    for policy in protocol["clock_order"]:
        target = out / f"live-{policy}.json"
        if target.exists():
            if read(target)["protocol_sha256"] != digest(out / "protocol.json"):
                raise ValueError("Incompatible live timing")
            continue
        warm = protocol["hist_dev_ids"][0]
        warmup = sample_candidate(model, tokenizer, data[warm], warm, seed, 0, protocol, cap=8)
        score(warmup)
        first, second, spent, scores = {}, {}, 0, {}
        start = perf_counter()
        for q in ids:
            first[q] = sample_candidate(model, tokenizer, data[q], q, seed, 0, protocol)
        judge_s = 0.0
        if policy == "judge":
            t = perf_counter()
            scores = {q: score(first[q]) for q in ids}
            judge_s = perf_counter() - t
        visible = {
            q: {"mean_logprob": first[q]["mean_logprob"], "decision_score": scores.get(q, 0)} for q in ids
        }
        order = retry_order(visible, policy, protocol["uniform_seed"], seed)
        slots = dict.fromkeys(ids, 0)
        for q in order:
            if spent == budget:
                break
            second[q] = sample_candidate(
                model,
                tokenizer,
                data[q],
                q,
                seed,
                1,
                protocol,
                cap=min(protocol["max_new_tokens"], budget - spent),
            )
            spent += len(second[q]["token_ids"])
            slots[q] = rule_select(first[q], second[q])
        elapsed = perf_counter() - start
        expected, _ = allocate(replay, seed, policy, budget, protocol["uniform_seed"], trunc)
        details = []
        for q in ids:
            got = {
                "prompt_id": q,
                "retry_tokens": len(second[q]["token_ids"]) if q in second else 0,
                "selected_slot": slots[q],
                "correct": (second[q] if slots[q] else first[q])["strict_reward"],
                "first_tokens_sha256": sha256(
                    json.dumps(first[q]["token_ids"], separators=(",", ":")).encode()
                ).hexdigest(),
                "retry_tokens_sha256": sha256(
                    json.dumps(second[q]["token_ids"], separators=(",", ":")).encode()
                ).hexdigest()
                if q in second
                else None,
            }
            ref = expected[q]
            if [got["retry_tokens"], got["selected_slot"], got["correct"]] != [
                ref["tokens"],
                ref["selected_slot"],
                ref["correct"],
            ] or not same_outcome(first[q], replay[q]["samples"][0]):
                raise ValueError(f"Live/replay first answer or decision differs for {policy}, {q}")
            if q in second and not same_outcome(
                second[q], trunc(replay[q], replay[q]["samples"][1], got["retry_tokens"])
            ):
                raise ValueError(f"Live/replay retry differs for {policy}, {q}")
            details.append(got)
        first_tokens = sum(len(s["token_ids"]) for s in first.values())
        if spent != budget:
            raise ValueError("Incomplete live retry budget")
        report = {
            "protocol_sha256": digest(out / "protocol.json"),
            "head_sha256": digest(out / "head.json"),
            "seed": seed,
            "policy": policy,
            "wall_seconds": elapsed,
            "judge_seconds": judge_s,
            "first_tokens": first_tokens,
            "retry_tokens": spent,
            "total_tokens": first_tokens + spent,
            "correct": sum(row["correct"] for row in details),
            "replay_verified": True,
            "questions": details,
        }
        record(target, report)
        print("Live", policy, report["correct"], "/", len(ids), "seconds", round(elapsed, 1), flush=True)


def run_all(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record(out / "protocol.json", freeze())

        def execute(module, *args):
            print("Running", module, *args, flush=True)
            subprocess.run(
                [sys.executable, "-u", "-m", f"experiments.{module}", "--out", str(out), *args], check=True
            )

        if not (out / "head-lock.json").exists():
            execute("gsm8k_decision_study", "--stage", "fit")
        locked_head(out)
        execute("gsm8k_decision_candidates")
        for stage in ("score", "analyze", "live", "verify-scores"):
            execute("gsm8k_decision_study", "--stage", stage)
        execute("audit_gsm8k_decision")
        print("Decision study finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--stage", default="all", choices=("all", "fit", "score", "verify-scores", "analyze", "live")
    )
    args = parser.parse_args()
    if args.stage == "all":
        return run_all(args.out)
    from experiments.gsm8k_signal import read_split

    protocol = read(args.out / "protocol.json")
    if protocol != freeze():
        raise ValueError("Frozen protocol changed")
    data = read_split(Path("runs/gsm8k-data/train.jsonl"), protocol["train_sha256"])
    if args.stage in ("score", "verify-scores"):
        score_pool(args.out, protocol, args.stage == "verify-scores")
    else:
        {"fit": train_head, "analyze": analyze, "live": live_timing}[args.stage](args.out, protocol, data)


if __name__ == "__main__":
    main()
