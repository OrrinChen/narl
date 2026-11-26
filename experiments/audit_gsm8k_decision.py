"""Independently reconstruct frozen GSM8K decision evaluation from token-level records."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random

import numpy as np

from experiments.audit_gsm8k_curriculum import check_protocol as audit_prior_protocol, read_train
from experiments.audit_gsm8k_online_lag import adapter_digest, validate_sample
from experiments.audit_gsm8k_regroup import ANSWER, NUMBER, check_score
from experiments.audit_kv_online import read_json, require
from experiments.gsm8k_reward_study import digest


FROZEN = "11ba13b9a2e5b3b3cc756200bd33a3d5dc5f63fa5d0ef8a4951e465f93cd7152"
ROOT = Path("runs")
ARMS = ("uniform", "confidence", "judge")


def matches(reported, expected, location):
    """Require each independently derived field. Files may also contain descriptive metadata."""
    if isinstance(expected, dict):
        require(isinstance(reported, dict) and set(expected) <= set(reported), f"{location}: missing fields")
        for k, v in expected.items():
            matches(reported[k], v, f"{location}/{k}")
    elif isinstance(expected, list):
        require(isinstance(reported, list) and len(reported) == len(expected), f"{location}: mismatched list")
        for i, (a, b) in enumerate(zip(reported, expected)):
            matches(a, b, f"{location}/{i}")
    elif isinstance(expected, (float, np.floating)):
        require(isinstance(reported, (int, float)) and not isinstance(reported, bool)
                and isclose(reported, expected, abs_tol=1e-10, rel_tol=1e-10), f"{location}: incorrect value")
    else:
        if expected is None or isinstance(expected, bool):
            require(reported is expected, f"{location}: incorrect flag")
        else:
            require(reported == expected, f"{location}: incorrect value")


def check_split(run, root, train):
    require(digest(run / "protocol.json") == FROZEN, "protocol differs from preregistration")
    p = read_json(run / "protocol.json")
    past = root / "curriculum-gsm8k-v1"
    prior = audit_prior_protocol(past, root, train)
    require(digest(past / "protocol.json") == p["prior_sha256"] and
            read_json(past / "audit-dose.json")["audit_passed"] and
            p["start_sha256"] == adapter_digest(root / "gsm8k-budget-5seed/start-adapter.safetensors"),
            "prior training or base weights changed")
    seen, history = set(), Counter()
    tag_counts = Counter()
    for arm, budget in [("uniform", 32768), ("mixed", 32768), ("mixed", 131072)]:
        for seed in p["seeds"]:
            rel = Path("models") / arm / str(budget) / f"seed-{seed}/trace.jsonl"
            path = past / rel
            expected_hash = read_json(path.with_name("training.json"))["trace_sha256"]
            require(digest(path) == expected_hash, "historical trace changed")
            long = budget == 131072
            if long:
                require(expected_hash == p["trace_sha256"][str(rel)], "head training trace changed")
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    q = row["prompt_id"]
                    seen.add(q)
                    if long:
                        for sample in row["samples"]:
                            check_score({**sample, "reward": sample["strict_reward"]},
                                        float(train[q]["value"]), "historical strict score")
                            history[q] += 1
                            tag_counts[q] += sample["has_answer_tag"]
    require(len(history) == 891 and sum(history.values()) == 3620, "wrong historical sample")
    counts = Random(2026112601).sample(sorted(history), len(history))
    unseen = sorted(set(prior["train_pool_ids"]) - seen)
    hold = Random(2026112602).sample(unseen, 256)
    matches(p, {"hist_train_ids": counts[179:], "hist_dev_ids": counts[:179], "eval_sets": {"holdout": hold},
                "clock_ids": hold[:32], "eligible_new_count": len(unseen)}, "frozen split")
    normalized = [" ".join(row["question"].casefold().split()) for row in train]
    excluded = set(prior["excluded_ids"] + prior["eval_sets"]["primary"] + prior["eval_sets"]["dose"]) | seen
    require(len({normalized[i] for i in hold}) == len(hold)
            and {normalized[i] for i in hold}.isdisjoint(normalized[i] for i in excluded),
            "held-out question text overlaps history")
    return p, {"train_questions": len(counts[179:]), "dev_questions": 179,
               "train_candidates": sum(history[i] for i in counts[179:]),
               "dev_candidates": sum(history[i] for i in counts[:179])}, sum(tag_counts[i] for i in counts[179:])


def check_head(run, p, counts, tagged):
    head_hash = digest(run / "head.json")
    head = read_json(run / "head.json")
    lock = read_json(run / "head-lock.json")
    matches(lock, {**counts, "protocol_sha256": FROZEN, "head_sha256": head_hash}, "head lock")
    scores = lock["dev_brier_by_l2"]
    require(set(scores) == set(map(str, p["l2_grid"])) and all(0 <= a <= 1 for a in scores.values()),
            "missing development candidates")
    best = min(p["l2_grid"], key=lambda x: (scores[str(x)], -x))
    require(best == head["l2"] == lock["selected_l2"], "development selection mismatch")
    matches(head, {"n_train_candidates": counts["train_candidates"], "n_train_tagged": tagged}, "head")
    return head_hash


def candidate_rows(run, p, train, tok, head_hash):
    from experiments.gsm8k_signal import prompt_text

    ids = p["eval_sets"]["holdout"]
    prompt_ids = {q: tok.encode(prompt_text(tok, {"question": train[q]["question"]}, "reasoning"),
                                add_special_tokens=False) for q in ids}
    pools, files = {}, []
    for seed in p["seeds"]:
        folder = run / "candidates/holdout" / f"seed-{seed}"
        score_path = run / "scoring" / f"seed-{seed}.json"
        predictions = read_json(score_path)
        files.append(digest(score_path))
        matches(predictions, {"head_sha256": head_hash, "protocol_sha256": FROZEN}, f"seed {seed} predictions")
        pairs = [(q, s) for q in ids for s in (0, 1)]
        require(set(f.name for f in folder.glob("*.json")) == {f"{q}-{s}.json" for q, s in pairs}
                and [(r["prompt_id"], r["slot"]) for r in predictions["rows"]] == pairs,
                "wrong pool coverage or scoring order")
        pool = {q: [] for q in ids}
        for (q, s), estimate in zip(pairs, predictions["rows"], strict=True):
            path = folder / f"{q}-{s}.json"
            sample = read_json(path)
            require(digest(path) == estimate["candidate_sha256"], "score bound to different candidacy")
            matches(sample, {"protocol_sha256": FROZEN, "checkpoint_sha256": p["start_sha256"],
                             "seed": seed, "prompt_id": q, "slot": s, "prompt_ids": prompt_ids[q],
                             "rng_seed": p["rng_base"] + (seed * 10000 + q) * 2 + s,
                             "token_cap": p["max_new_tokens"]}, str(path))
            n = validate_sample({**sample, "reward": sample["strict_reward"]}, float(train[q]["value"]),
                                p["max_new_tokens"], str(path), tok)
            matches(sample["mean_logprob"], sum(sample["behavior_logprobs"]) / n, str(path))
            times = sample["token_elapsed_s"]
            require(len(times) == n and all(isfinite(v) for v in times)
                    and times == sorted(times) and 0 < times[0] <= times[-1] <= sample["generation_s"]
                    and isfinite(sample["generation_s"]), "candidate time profile mismatch")
            prob = estimate["decision_score"]
            require(0 <= prob <= 1 and (sample["has_answer_tag"] or prob == 0) and
                    isfinite(estimate["decision_s"]) and estimate["decision_s"] >= 0, "invalid judgment")
            pool[q].append({**sample, "decision_score": prob, "decision_s": estimate["decision_s"]})
        pools[seed] = pool
    matches(read_json(run / "score-check.json"), {"head_sha256": head_hash, "scores_verified": True,
            "files_sha256": files, "scores": len(ids) * 2 * len(p["seeds"])}, "repeated forward verification")
    return pools


def selected(a, b):
    if a["has_answer_tag"] != b["has_answer_tag"]:
        return int(b["has_answer_tag"])
    return int(b["mean_logprob"] > a["mean_logprob"])


def capped(sample, n, gold, tokenizer):
    if n == len(sample["token_ids"]):
        return sample
    tokens = sample["token_ids"][:n]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    tags = ANSWER.findall(text)
    numbers = NUMBER.findall(tags[-1].replace(",", "")) if tags else []
    correct = int(bool(numbers) and abs(float(numbers[-1]) - gold) < 1e-4)
    return {"token_ids": tokens, "has_answer_tag": bool(tags), "strict_reward": correct,
            "mean_logprob": sum(sample["behavior_logprobs"][:n]) / n, "generation_s": sample["token_elapsed_s"][n-1]}


def expected_allocation(p, seed, pool, arm, gold, tokenizer):
    ids = sorted(pool)
    Random(p["uniform_seed"] + seed).shuffle(ids)
    tie = {q: i for i, q in enumerate(ids)}
    if arm != "uniform":
        key = "mean_logprob" if arm == "confidence" else "decision_score"
        ids.sort(key=lambda q: (pool[q][0][key], tie[q]))
    budget = len(ids) * p["retry_tokens_per_question"]
    result = {q: {"tokens": 0, "selected_slot": 0, "correct": x[0]["strict_reward"]} for q, x in pool.items()}
    spent, elapsed, partial = 0, 0.0, None
    for rank, q in enumerate(ids, 1):
        left = budget - spent
        if not left:
            break
        a, b = pool[q]
        n = min(len(b["token_ids"]), left)
        if n < len(b["token_ids"]):
            partial = q
        b = capped(b, n, gold[q], tokenizer)
        pos = selected(a, b)
        result[q] = {"priority_rank": rank, "retry_correct": b["strict_reward"], "tokens": n,
                     "selected_slot": pos, "correct": [a, b][pos]["strict_reward"]}
        spent += n
        elapsed += b["generation_s"]
    require(spent == budget, "independent schedule exhausted retry pool")
    retried = [q for q, d in result.items() if d["tokens"]]
    first_tokens = sum(len(a["token_ids"]) for a, _ in pool.values())
    changes = [v["correct"] - pool[q][0]["strict_reward"] for q, v in result.items()]
    first_s = sum(a["generation_s"] for a, _ in pool.values())
    judge_s = sum(a["decision_s"] for a, _ in pool.values()) if arm == "judge" else 0.0
    counts = [v["correct"] for v in result.values()]
    return result, {**score_total(counts), "first_tokens": first_tokens, "retry_tokens": budget,
        "total_tokens": first_tokens + budget, "retried_questions": len(retried), "partial_retry_question": partial,
        "rescued_vs_first": changes.count(1), "harmed_vs_first": changes.count(-1), "net_correct_vs_first": sum(changes),
        "wrong_first_retried": sum(pool[q][0]["strict_reward"] == 0 for q in retried),
        "wrong_first_correct_retry": sum(pool[q][0]["strict_reward"] == 0 and result[q]["retry_correct"] for q in retried),
        "retry_generation_s": elapsed, "first_generation_s": first_s, "judge_scoring_s": judge_s,
        "profile_estimated_s": first_s + judge_s + elapsed}


def score_total(values):
    return {"accuracy": sum(values) / len(values), "correct": sum(values), "questions": len(values)}


def compare_matrices(report, name, x, y, seeds, multiplicity):
    x, y = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
    delta = y - x
    # Bootstrap multiplicities encode shared question blocks rather than independently indexing each seed.
    estimates = (delta @ multiplicity.T) / x.shape[1]
    median95 = np.quantile(np.median(estimates, axis=0), [0.025, 0.975]).tolist()
    mean95 = np.quantile(estimates.mean(axis=0), [0.025, 0.975]).tolist()
    means = delta.mean(axis=1)
    per_seed = {str(s): {"baseline_correct": int(x[i].sum()), "candidate_correct": int(y[i].sum()),
                         "wins": int(sum(delta[i] == 1)), "losses": int(sum(delta[i] == -1)),
                         "ties": int(sum(delta[i] == 0)), "accuracy_gain": means[i]}
                for i, s in enumerate(seeds)}
    matches(report, {"per_seed": per_seed, "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
                     "ties": int((delta == 0).sum()), "mean_accuracy_gain": means.mean(),
                     "median_accuracy_gain": np.median(means), "positive_seeds": int((means > 0).sum()),
                     "negative_seeds": int((means < 0).sum()), "tied_seeds": int((means == 0).sum()),
                     "question_block_mean_95": mean95, "question_block_median_95": median95}, name)
    return bool(median95[0] > 0 and (means > 0).sum() >= 4 and min(means) >= 0)


def token_hash(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def check_live(run, p, seed, arm, pool, gold, tok, head_hash):
    ids = p["clock_ids"]
    sub = {q: pool[q] for q in ids}
    outcomes, counts = expected_allocation(p, seed, sub, arm, gold, tok)
    report = read_json(run / f"live-{arm}.json")
    expected = []
    for q in ids:
        n = outcomes[q]["tokens"]
        a, b = pool[q]
        expected.append({"prompt_id": q, "retry_tokens": n, "selected_slot": outcomes[q]["selected_slot"],
                         "correct": outcomes[q]["correct"], "first_tokens_sha256": token_hash(a["token_ids"]),
                         "retry_tokens_sha256": token_hash(b["token_ids"][:n]) if n else None})
    matches(report, {**{k: counts[k] for k in ("correct", "first_tokens", "retry_tokens", "total_tokens")},
                     "protocol_sha256": FROZEN, "head_sha256": head_hash, "seed": seed,
                     "policy": arm, "replay_verified": True, "questions": expected}, "live/" + arm)
    require(isfinite(report["wall_seconds"]) and isfinite(report["judge_seconds"])
            and report["wall_seconds"] > report["judge_seconds"] >= 0
            and ((report["judge_seconds"] > 0) == (arm == "judge")), "live timing mismatch")
    return {"tokens": counts["total_tokens"], "correct": counts["correct"], "wall_seconds": report["wall_seconds"],
            "prefix_hashes_verified": len(expected) + counts["retried_questions"]}


def check_results(run, p, pools, train, tok, head_hash):
    r = read_json(run / "results.json")
    seeds = p["seeds"]
    ids = sorted(p["eval_sets"]["holdout"])
    gold = {q: float(train[q]["value"]) for q in ids}
    samples = [(s["decision_score"], s["strict_reward"]) for pool in pools.values() for pair in pool.values() for s in pair]
    ps, ys = np.array(samples).T
    matches(r, {"calibration": {"n": len(ys), "average_probability": ps.mean(), "fraction_correct": ys.mean(),
                                "brier": np.mean((ps - ys)**2)},
                "seeds": seeds, "questions_per_seed": len(ids), "retry_budget_per_seed": len(ids)*64,
                "head_sha256": head_hash, "protocol_sha256": FROZEN, "bootstrap": {"seed": 80161, "replicates": 10000}},
            "reported study")
    ranking = {key: [] for key in ("first", "random_expected", "rule", "judge", "oracle")}
    allocation = {key: [] for key in ("first", *ARMS)}
    for seed in seeds:
        pool, per = pools[seed], r["per_seed"][str(seed)]
        rows = per["questions"]
        require([d["prompt_id"] for d in rows] == ids, "wrong reported question order")
        scores = {k: [] for k in ranking}
        for row, q in zip(rows, ids, strict=True):
            a, b = pool[q]
            aa, bb = a["strict_reward"], b["strict_reward"]
            base, judge = selected(a, b), int(a["decision_score"] < b["decision_score"])
            matches(row, {"first_correct": aa, "retry_correct": bb, "rule_slot": base,
                          "judge_slot": judge}, f"ranking {seed}/{q}")
            for k, v in zip(scores, [aa, (aa + bb) / 2, (aa, bb)[base], (aa, bb)[judge], max(aa, bb)], strict=True):
                scores[k].append(v)
        for k in scores:
            ranking[k].append(scores[k])
        matches(per["fixed_pair_scores"], {k: score_total(v) for k, v in scores.items()}, "fixed scores")
        matches(per, {"both_candidates_generation_s": sum(s["generation_s"] for pair in pool.values() for s in pair),
                      "both_candidates_judge_scoring_s": sum(s["decision_s"] for pair in pool.values() for s in pair)},
                "measured pool time")
        allocation["first"].append(scores["first"])
        for arm in ARMS:
            outcomes, summary = expected_allocation(p, seed, pool, arm, gold, tok)
            matches(per["allocation"][arm], summary, f"allocation {seed}/{arm}")
            for row in rows:
                expected = outcomes[row["prompt_id"]]
                require(set(row["allocation"][arm]) == set(expected), "unexpected retry or missing retry detail")
                matches(row["allocation"][arm], expected, f"allocation {seed}/{arm}/{row['prompt_id']}")
            allocation[arm].append([outcomes[q]["correct"] for q in ids])
    draw = np.random.default_rng(80161).integers(0, len(ids), size=(10000, len(ids)))
    multiplicity = np.array([np.bincount(sample, minlength=len(ids)) for sample in draw], dtype=np.int16)
    gate = {}
    for prefix, matrix, contrasts in (
        ("fixed_pair", ranking, [("judge", "rule"), ("judge", "first"), ("rule", "first")]),
        ("allocation", allocation, [("judge", "confidence"), ("judge", "uniform"), ("confidence", "uniform"),
                                    *((k, "first") for k in ARMS)])):
        matches(r[prefix + "_scores"], {k: score_total(np.asarray(seq).flatten().tolist()) for k, seq in matrix.items()},
                prefix + " pooled scores")
        for new, old in contrasts:
            name = new + "_vs_" + old
            passed = compare_matrices(r[prefix + "_contrasts"][name], name, matrix[old], matrix[new], seeds, multiplicity)
            if name in ("judge_vs_rule", "judge_vs_confidence"):
                gate[name] = passed
    matches(r["gate"], gate, "prespecified criteria")
    live = {arm: check_live(run, p, p["clock_seed"], arm, pools[p["clock_seed"]], gold, tok, head_hash)
            for arm in p["clock_order"]}
    return {"gate": gate, "live": live, "candidate_answers": len(samples), "paired_question_seed_rows": len(ids)*len(seeds),
            "retry_tokens_per_seed_per_arm": len(ids)*p["retry_tokens_per_question"],
            "results_sha256": digest(run / "results.json"), "score_forward_check_sha256": digest(run / "score-check.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"gsm8k-decision-v1")
    parser.add_argument("--runs", type=Path, default=ROOT)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    train = read_train(args.runs / "gsm8k-data/train.jsonl")
    p, counts, tagged = check_split(args.out, args.runs, train)
    result = {"audit_passed": True, "protocol_sha256": FROZEN, "protocol_only": args.protocol_only,
              "historical": counts, "holdout_questions": len(p["eval_sets"]["holdout"])}
    if not args.protocol_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(p["model"], revision=p["revision"], local_files_only=True)
        head_hash = check_head(args.out, p, counts, tagged)
        pools = candidate_rows(args.out, p, train, tok, head_hash)
        result.update(check_results(args.out, p, pools, train, tok, head_hash))
        result["head_sha256"] = head_hash
    path = args.out / ("protocol-audit.json" if args.protocol_only else "audit.json")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
    print("Verified", path, result.get("gate", {}))


if __name__ == "__main__":
    main()
