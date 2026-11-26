"""Supplementary format-priority retry control on the frozen primary candidate pool."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.audit_gsm8k_accuracy import BOOTSTRAP_REPLICATES
from experiments.audit_kv_online import read_json, require
from experiments.gsm8k_decision_eval import paired, retry_order, rule_select, validate
from experiments.gsm8k_reward_study import digest, record


FROZEN_SUPPLEMENT = "75419018393372ee20f3d6de2b7036e093f75808c9b72f1d460f566c1091a684"


def tag_priority(firsts, uniform_seed, seed):
    tie_order = retry_order(firsts, "uniform", uniform_seed, seed)
    return sorted(tie_order, key=lambda q: (firsts[q]["has_answer_tag"], firsts[q]["mean_logprob"]))


def replay(rows, budget, uniform_seed, seed, truncate):
    firsts = {q: r["samples"][0] for q, r in rows.items()}
    remaining = budget
    details = {q: {"tokens": 0, "selected_slot": 0, "correct": a["strict_reward"]} for q, a in firsts.items()}
    for rank, q in enumerate(tag_priority(firsts, uniform_seed, seed), 1):
        if not remaining:
            break
        first, retry = rows[q]["samples"]
        n = min(remaining, len(retry["token_ids"]))
        if n < len(retry["token_ids"]):
            partial = truncate(rows[q], retry, n)
            require(partial["token_ids"] == retry["token_ids"][:n], "truncation changed the candidate prefix")
            retry = partial
        chosen = rule_select(first, retry)
        details[q] = {"tokens": n, "selected_slot": chosen, "correct": (first, retry)[chosen]["strict_reward"],
                      "priority_rank": rank, "retry_correct": retry["strict_reward"]}
        remaining -= n
    require(remaining == 0, "supplementary policy exhausted the retry pool")
    return details


def compare_control(records, primary, uniform_seed, truncate):
    pools, ids = validate(records)
    budget = primary["retry_budget_per_seed"]
    seeds = primary["seeds"]
    require(sorted(pools) == seeds, "unexpected generation seeds")
    by_policy = {name: [] for name in ("tag_priority", "judge", "confidence")}
    subset, per_seed = Counter(), {}
    for seed in seeds:
        pool = pools[seed]
        reported = primary["per_seed"][str(seed)]
        require([x["prompt_id"] for x in reported["questions"]] == ids, "primary question order mismatch")
        choices = replay(pool, budget, uniform_seed, seed, truncate)
        by_policy["tag_priority"].append([choices[q]["correct"] for q in ids])
        for name in ("judge", "confidence"):
            by_policy[name].append([x["allocation"][name]["correct"] for x in reported["questions"]])
            require(reported["allocation"][name]["retry_tokens"] == budget, "primary retry budgets differ")
        tagged = Counter(dict.fromkeys(("pairs", "rule_correct", "judge_correct", "wins", "losses", "ties"), 0))
        rescues, regressions = 0, 0
        for q in ids:
            first, retry = pool[q]["samples"]
            gain = choices[q]["correct"] - first["strict_reward"]
            rescues += gain == 1
            regressions += gain == -1
            if first["has_answer_tag"] and retry["has_answer_tag"]:
                rule_slot = rule_select(first, retry)
                judge_slot = int(first["decision_score"] < retry["decision_score"])
                old = pool[q]["samples"][rule_slot]["strict_reward"]
                new = pool[q]["samples"][judge_slot]["strict_reward"]
                tagged.update(pairs=1, rule_correct=old, judge_correct=new, wins=int(new > old),
                              losses=int(new < old), ties=int(new == old))
        subset.update(tagged)
        first_n = sum(len(pool[q]["samples"][0]["token_ids"]) for q in ids)
        require(first_n + budget == reported["allocation"]["judge"]["total_tokens"] and
                sum(row["tokens"] for row in choices.values()) == budget, "unequal total token budgets")
        per_seed[str(seed)] = {
            "first_tokens": first_n, "retry_tokens": budget, "total_tokens": first_n + budget,
            "correct": sum(by_policy["tag_priority"][-1]), "retried_questions": sum(x["tokens"] > 0 for x in choices.values()),
            "rescued_vs_first": rescues, "harmed_vs_first": regressions,
            "both_tagged_ranking": dict(tagged), "questions": [{"prompt_id": q, **choices[q]} for q in ids],
        }
    draw = np.random.default_rng(primary["bootstrap"]["seed"]).integers(0, len(ids), size=(BOOTSTRAP_REPLICATES, len(ids)))
    return {
        "per_seed": per_seed, "both_tagged_ranking_pooled": dict(subset),
        "pooled_accuracy": {name: float(np.mean(answers)) for name, answers in by_policy.items()},
        "judge_vs_tag_priority": paired(by_policy["tag_priority"], by_policy["judge"], seeds, draw),
        "tag_priority_vs_confidence": paired(by_policy["confidence"], by_policy["tag_priority"], seeds, draw),
    }


def main():
    from transformers import AutoTokenizer
    from experiments.gsm8k_decision_candidates import truncate_candidate
    from experiments.gsm8k_decision_study import pool_records
    from experiments.gsm8k_signal import read_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/gsm8k-decision-v1"))
    args = parser.parse_args()
    out = args.out
    require(digest(out / "tag-control-plan.json") == FROZEN_SUPPLEMENT, "supplementary registration changed")
    lock = read_json(out / "tag-control-plan.json")
    for name, key in (("head.json", "head_sha256"), ("head-lock.json", "head_lock_sha256"),
                      ("protocol.json", "protocol_sha256")):
        require(digest(out / name) == lock[key], "supplementary analysis binding changed")
    p = read_json(out / "protocol.json")
    primary = read_json(out / "results.json")
    ids = p["eval_sets"]["holdout"]
    require(primary["protocol_sha256"] == lock["protocol_sha256"] and primary["head_sha256"] == lock["head_sha256"]
            and primary["seeds"] == p["seeds"] and primary["retry_budget_per_seed"] == len(ids) * 64
            and primary["bootstrap"]["seed"] == p["bootstrap_seed"], "wrong primary results")
    samples = pool_records(out, p, p["seeds"], ids)
    tokenizer = AutoTokenizer.from_pretrained(p["model"], revision=p["revision"], local_files_only=True)
    data = read_split(Path("runs/gsm8k-data/train.jsonl"), p["train_sha256"])
    report = compare_control(samples, primary, p["uniform_seed"],
                             lambda row, s, n: truncate_candidate(tokenizer, data[row["prompt_id"]], s, n))
    record(out / "tag-control-results.json", {**report, "plan_sha256": FROZEN_SUPPLEMENT,
           "primary_results_sha256": digest(out / "results.json"), "head_sha256": lock["head_sha256"]})
    print("Supplementary tag control complete", report["pooled_accuracy"])


if __name__ == "__main__":
    main()
