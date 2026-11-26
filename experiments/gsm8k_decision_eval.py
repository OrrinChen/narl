"""Paired answer selection and fixed-token retry allocation on frozen GSM8K candidates."""

from math import isfinite
from random import Random

import numpy as np

from experiments.audit_gsm8k_accuracy import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from experiments.audit_kv_online import require


POLICIES = ("uniform", "confidence", "judge")
SELECTORS = ("first", "random_expected", "rule", "judge", "oracle")


def retry_order(first_by_id, policy, uniform_seed, seed):
    """Return a priority order from first-answer signals alone; all policies share tie breaks."""
    require(policy in POLICIES, f"unknown retry policy: {policy}")
    ids = sorted(first_by_id)
    Random(uniform_seed + seed).shuffle(ids)
    if policy == "uniform":
        return ids
    field = "mean_logprob" if policy == "confidence" else "decision_score"
    return sorted(ids, key=lambda q: first_by_id[q][field])


def rule_select(first, retry):
    """Return the selected slot, favoring a complete answer tag, then mean logprob, then first."""
    return int((retry["has_answer_tag"], retry["mean_logprob"]) >
               (first["has_answer_tag"], first["mean_logprob"]))


def validate(records):
    require(len(records) > 0, "expected a nonempty evaluation")
    by_seed = {}
    for row in records:
        seed, q = row["seed"], row["prompt_id"]
        require(isinstance(q, int) and isinstance(seed, int), "invalid seed or question ID")
        rows = by_seed.setdefault(seed, {})
        require(q not in rows and len(row["samples"]) == 2, "duplicate question or incomplete pair")
        for sample in row["samples"]:
            require(sample["strict_reward"] in (0, 1) and sample["has_answer_tag"] in (False, True)
                    and sample["token_ids"] and isfinite(sample["mean_logprob"])
                    and 0 <= sample["decision_score"] <= 1, "invalid evaluation candidate")
            for field in ("generation_s", "decision_s"):
                if field == "generation_s" or sample.get(field) is not None:
                    require(isfinite(sample[field]) and sample[field] >= 0, f"invalid timing: {field}")
        rows[q] = row
    ids = sorted(next(iter(by_seed.values())))
    require(all(sorted(rows) == ids for rows in by_seed.values()), "seeds must share the same panel")
    return by_seed, ids


def select_pair(first, retry):
    rule = rule_select(first, retry)
    judge = int(retry["decision_score"] > first["decision_score"])
    a, b = first["strict_reward"], retry["strict_reward"]
    return {"first": a, "random_expected": (a + b) / 2, "rule": (a, b)[rule],
            "judge": (a, b)[judge], "oracle": max(a, b)}, rule, judge


def scoring_time(samples):
    values = [s.get("decision_s") for s in samples]
    return sum(values) if all(v is not None for v in values) else None


def allocate(rows, seed, policy, budget, uniform_seed, truncate):
    # Keep both the second candidates and all labels out of the priority interface.
    firsts = {q: row["samples"][0] for q, row in rows.items()}
    visible = {q: {k: sample[k] for k in ("decision_score", "mean_logprob")} for q, sample in firsts.items()}
    order = retry_order(visible, policy, uniform_seed, seed)
    results = {q: {"tokens": 0, "selected_slot": 0, "correct": int(a["strict_reward"])}
               for q, a in firsts.items()}
    remaining, retry_generation_s, partial = budget, 0.0, None
    for rank, q in enumerate(order):
        if remaining == 0:
            break
        row = rows[q]
        first, full_retry = row["samples"]
        tokens = min(remaining, len(full_retry["token_ids"]))
        retry = full_retry
        if tokens < len(full_retry["token_ids"]):
            retry = truncate(row, full_retry, tokens)
            require(retry["token_ids"] == full_retry["token_ids"][:tokens]
                    and retry["strict_reward"] in (0, 1)
                    and retry["has_answer_tag"] in (False, True)
                    and isfinite(retry["mean_logprob"]) and isfinite(retry["generation_s"])
                    and retry["generation_s"] >= 0, "invalid truncated retry")
            partial = q
        slot = rule_select(first, retry)
        results[q] = {"tokens": tokens, "priority_rank": rank + 1, "selected_slot": slot,
                      "retry_correct": int(retry["strict_reward"]),
                      "correct": int((first, retry)[slot]["strict_reward"])}
        retry_generation_s += retry["generation_s"]
        remaining -= tokens
    require(remaining == 0, "retry pool exhausted before the frozen budget was spent")
    spent = sum(r["tokens"] for r in results.values())
    require(spent == budget, "retry token mismatch")
    initial_tokens = sum(len(s["token_ids"]) for s in firsts.values())
    first_generation_s = sum(s["generation_s"] for s in firsts.values())
    judge_s = scoring_time(firsts.values()) if policy == "judge" and budget else 0.0
    attempted = [q for q, row in results.items() if row["tokens"] > 0]
    rescued = sum(firsts[q]["strict_reward"] == 0 and results[q]["correct"] == 1 for q in attempted)
    harmed = sum(firsts[q]["strict_reward"] == 1 and results[q]["correct"] == 0 for q in attempted)
    return results, {
        "rescued_vs_first": rescued, "harmed_vs_first": harmed, "net_correct_vs_first": rescued - harmed,
        "first_tokens": initial_tokens, "retry_tokens": spent, "total_tokens": initial_tokens + spent,
        "retried_questions": len(attempted), "partial_retry_question": partial,
        "wrong_first_retried": sum(firsts[q]["strict_reward"] == 0 for q in attempted),
        "wrong_first_correct_retry": sum(firsts[q]["strict_reward"] == 0 and results[q]["retry_correct"] == 1
                                         for q in attempted),
        "first_generation_s": first_generation_s, "retry_generation_s": retry_generation_s,
        "judge_scoring_s": judge_s,
        "profile_estimated_s": first_generation_s + retry_generation_s + judge_s if judge_s is not None else None,
    }


def accuracy(scores):
    return {"correct": sum(scores), "questions": len(scores), "accuracy": sum(scores) / len(scores)}


def paired(before, after, seeds, draw):
    old = np.asarray(before, dtype=np.int8)
    new = np.asarray(after, dtype=np.int8)
    delta = new - old
    seed_delta = delta.mean(axis=1)
    boot = np.stack([row[draw].mean(axis=1) for row in delta])
    return {
        "per_seed": {str(s): {"baseline_correct": int(old[i].sum()), "candidate_correct": int(new[i].sum()),
                              "wins": int((delta[i] == 1).sum()), "losses": int((delta[i] == -1).sum()),
                              "ties": int((delta[i] == 0).sum()), "accuracy_gain": float(seed_delta[i])}
                     for i, s in enumerate(seeds)},
        "wins": int((delta == 1).sum()), "losses": int((delta == -1).sum()), "ties": int((delta == 0).sum()),
        "positive_seeds": int((seed_delta > 0).sum()), "tied_seeds": int((seed_delta == 0).sum()),
        "negative_seeds": int((seed_delta < 0).sum()), "median_accuracy_gain": float(np.median(seed_delta)),
        "mean_accuracy_gain": float(seed_delta.mean()),
        "question_block_median_95": np.quantile(np.median(boot, axis=0), [0.025, 0.975]).tolist(),
        "question_block_mean_95": np.quantile(boot.mean(axis=0), [0.025, 0.975]).tolist(),
    }


def method_scores(panels, methods):
    return {name: accuracy([row[name] for panel in panels for row in panel]) for name in methods}


def compare(panels, seeds, draw, baseline, candidate):
    before = [[row[baseline] for row in panel] for panel in panels]
    after = [[row[candidate] for row in panel] for panel in panels]
    return paired(before, after, seeds, draw)


def evaluate(records, retry_budget, truncate, uniform_seed, bootstrap_seed=BOOTSTRAP_SEED):
    """Replay fixed policies once per seed. The callback re-decodes and scores an actual token prefix."""
    require(isinstance(retry_budget, int) and retry_budget >= 0, "expected an integer retry budget per seed")
    by_seed, ids = validate(records)
    seeds = sorted(by_seed)
    ranking_panels, allocation_panels, per_seed = [], [], {}
    for seed in seeds:
        rows = by_seed[seed]
        ranking = []
        details = []
        for q in ids:
            first, retry = rows[q]["samples"]
            scores, rule_slot, judge_slot = select_pair(first, retry)
            ranking.append(scores)
            details.append({"prompt_id": q, "first_correct": first["strict_reward"],
                            "retry_correct": retry["strict_reward"], "rule_slot": rule_slot,
                            "judge_slot": judge_slot, "allocation": {}})
        allocation = [{"first": row["first"]} for row in ranking]
        arms = {}
        for policy in POLICIES:
            result, arms[policy] = allocate(rows, seed, policy, retry_budget, uniform_seed, truncate)
            for i, q in enumerate(ids):
                allocation[i][policy] = result[q]["correct"]
                details[i]["allocation"][policy] = result[q]
            arms[policy].update(accuracy([x[policy] for x in allocation]))
        samples = [s for q in ids for s in rows[q]["samples"]]
        per_seed[str(seed)] = {
            "fixed_pair_scores": method_scores([ranking], SELECTORS), "allocation": arms,
            "both_candidates_generation_s": sum(s["generation_s"] for s in samples),
            "both_candidates_judge_scoring_s": scoring_time(samples),
            "questions": details,
        }
        ranking_panels.append(ranking)
        allocation_panels.append(allocation)

    draw = np.random.default_rng(bootstrap_seed).integers(0, len(ids), size=(BOOTSTRAP_REPLICATES, len(ids)))
    fixed_contrasts = [("rule", "judge"), ("first", "rule"), ("first", "judge")]
    retry_contrasts = [("uniform", "judge"), ("confidence", "judge"), ("uniform", "confidence")]
    retry_contrasts.extend(("first", policy) for policy in POLICIES)
    return {
        "seeds": seeds, "questions_per_seed": len(ids), "retry_budget_per_seed": retry_budget,
        "fixed_pair_scores": method_scores(ranking_panels, SELECTORS),
        "fixed_pair_contrasts": {b + "_vs_" + a: compare(ranking_panels, seeds, draw, a, b) for a, b in fixed_contrasts},
        "allocation_scores": method_scores(allocation_panels, ("first", *POLICIES)),
        "allocation_contrasts": {b + "_vs_" + a: compare(allocation_panels, seeds, draw, a, b)
                                 for a, b in retry_contrasts},
        "bootstrap": {"seed": bootstrap_seed, "replicates": BOOTSTRAP_REPLICATES,
                      "unit": "question ID, using the same draws across candidates, policies and generation seeds",
                      "scope": "Conditional on the realized candidate pool and once-per-panel retry allocations."},
        "timing_scope": "Profile estimates sum measured candidate generation/prefix and applicable scoring times. "
                        "They exclude live orchestration and do not time independently executed policies.",
        "per_seed": per_seed,
    }
