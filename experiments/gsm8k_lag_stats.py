"""Report paired held-out outcomes for the five-seed GSM8K lag-one experiment."""

from itertools import product
from statistics import median

from experiments.audit_gsm8k_matched import score_change
from experiments.audit_kv_online import require


ARMS = ("fresh", "stale", "mixed")
COMPARISONS = (
    ("fresh", "stale"), ("fresh", "mixed"), ("stale", "mixed"),
    ("initial", "fresh"), ("initial", "stale"), ("initial", "mixed"),
)
SEED_COUNT = 5
DIRECTION_THRESHOLD = 4


def paired_sign_flip_p(deltas):
    """Two-sided exact seed-level sign permutation of the sum of paired deltas."""
    observed = abs(sum(deltas))
    extremes = sum(abs(sum(sign * delta for sign, delta in zip(signs, deltas, strict=True))) >= observed
                   for signs in product((-1, 1), repeat=len(deltas)))
    return extremes / (2 ** len(deltas))


def evaluation_rows(evaluation, expected_ids):
    rows = evaluation["rows"]
    require([row["prompt_id"] for row in rows] == expected_ids, "held-out prompts or order changed between evaluations")
    scores = [row["reward"] for row in rows]
    require(all(score in (0, 1) for score in scores), "held-out scores must be binary")
    require("correct" not in evaluation or evaluation["correct"] == sum(scores), "saved held-out total disagrees with rows")
    return scores, [row["token_ids"] for row in rows]


def summarize_results(initial_eval, seed_arms):
    """Pair every seed on one ordered held-out set, retaining the question and seed units."""
    require(len(seed_arms) == SEED_COUNT, f"expected {SEED_COUNT} complete paired seeds")
    ids = [row["prompt_id"] for row in initial_eval["rows"]]
    require(ids and len(set(ids)) == len(ids), "expected nonempty, unique held-out prompt IDs")
    initial_scores, initial_tokens = evaluation_rows(initial_eval, ids)
    contrast_names = [f"{candidate}_vs_{baseline}" for baseline, candidate in COMPARISONS]
    per_seed = {}
    for seed, arms in seed_arms.items():
        require(set(arms) == set(ARMS), f"seed {seed} must contain all three arms")
        outcomes = {"initial": (initial_scores, initial_tokens)}
        for arm in ARMS:
            evaluation = arms[arm].get("evaluation", arms[arm])
            outcomes[arm] = evaluation_rows(evaluation, ids)
        contrasts = {}
        for name, (baseline, candidate) in zip(contrast_names, COMPARISONS, strict=True):
            base_scores, base_tokens = outcomes[baseline]
            candidate_scores, candidate_tokens = outcomes[candidate]
            contrasts[name] = score_change(base_scores, candidate_scores, base_tokens, candidate_tokens)
        per_seed[str(seed)] = {"correct": {name: sum(scores) for name, (scores, _) in outcomes.items()},
                              "contrasts": contrasts}
    require(len(per_seed) == SEED_COUNT, "paired seed IDs must remain unique when serialized")

    aggregate = {}
    for name in contrast_names:
        comparisons = [seed["contrasts"][name] for seed in per_seed.values()]
        deltas = [row["wins"] - row["losses"] for row in comparisons]
        better = sum(delta > 0 for delta in deltas)
        worse = sum(delta < 0 for delta in deltas)
        direction = "higher" if better >= DIRECTION_THRESHOLD else "lower" if worse >= DIRECTION_THRESHOLD else None
        aggregate[name] = {"seeds_higher": better, "seeds_lower": worse, "seeds_tied": deltas.count(0),
                           "median_accuracy_delta": median(deltas) / len(ids),
                           "min_accuracy_delta": min(deltas) / len(ids), "max_accuracy_delta": max(deltas) / len(ids),
                           "question_wins": sum(row["wins"] for row in comparisons),
                           "question_losses": sum(row["losses"] for row in comparisons),
                           "consistent_direction": direction, "exact_seed_sign_flip_p": paired_sign_flip_p(deltas)}
    return {"heldout_count": len(ids), "initial_correct": sum(initial_scores), "seed_count": SEED_COUNT,
            "direction_threshold": DIRECTION_THRESHOLD, "per_seed": per_seed, "contrasts": aggregate}
