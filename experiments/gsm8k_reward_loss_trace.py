"""Rescore archived G2 training samples and Q1 flips without loading a model.

Run from the repo root with .venv/bin/python -m experiments.gsm8k_reward_loss_trace.
All alternative scorers are descriptive sensitivity checks on fixed outputs.
"""

import argparse
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean, median


TAG = re.compile(r"<answer>(.*?)</answer>", re.S)
NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
FRACTION = re.compile(r"(-?\d+)\s*/\s*(\d+)")
SPACED_CENTS = re.compile(r"(\$\s*\d+)\.\s+(\d{2})(?!\d)")
CANDIDATE = "lr-0.0001-groups-16"


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def numbers(text):
    return [Decimal(token) for token in NUMBER.findall(text.replace(",", ""))]


def matches(values, gold):
    return bool(values and abs(values[-1] - gold) < Decimal("0.0001"))


def scores(row, gold):
    text = row["completion"]
    tags = list(TAG.finditer(text))
    tagged = tags[-1][1].strip() if tags else ""
    values = numbers(tagged)
    official = int(matches(values, gold))
    if official != row["reward"] or bool(tags) != row["has_answer_tag"]:
        raise ValueError(f"Archived score or tag status differs from independent reconstruction: {row.get('id', row.get('prompt_id'))}")
    # Only completed, untagged answers qualify for the last-number fallback.
    fallback = int(official or (not tags and not row["truncated"] and matches(numbers(text), gold)))
    return official, fallback, tagged, values, tags


def group_type(rewards):
    if len(rewards) != 2:
        raise ValueError("This trace study is restricted to complete two-response groups")
    return ("both_zero", "mixed", "both_one")[sum(rewards)]


def audit_training(run, gold):
    totals, transitions, positive_flags = Counter(), Counter(), Counter()
    examples, seed_reports, odds, active_batches = {}, [], [], []
    positive_lengths, negative_lengths = [], []
    multivalue, units, fractions = [], [], []
    for seed in range(5):
        folder = run / "models" / CANDIDATE / f"seed-{seed}"
        groups, updates = lines(folder / "trace.jsonl"), lines(folder / "updates.jsonl")
        counts = Counter()
        for group in groups:
            prompt_id, samples = group["prompt_id"], group["samples"]
            base, alternative = [], []
            for sample, adv in zip(samples, group["advantages"], strict=True):
                reward, fallback, tagged, values, tags = scores(sample, gold[prompt_id])
                text, truncated = sample["completion"], sample["truncated"]
                base.append(reward)
                alternative.append(fallback)
                counts["samples"] += 1
                counts["generated_tokens"] += len(sample["token_ids"])
                counts["nonzero_advantage_tokens"] += len(sample["token_ids"]) if adv else 0
                counts["reward_one"] += reward
                counts["truncated"] += truncated
                if not reward:
                    state = "tagged_wrong" if tags else "no_tag"
                    counts[f"reward_zero_{state}_{'truncated' if truncated else 'ended'}"] += 1
                info = {"sample_id": sample["id"], "prompt_id": prompt_id, "gold": str(gold[prompt_id]),
                        "reward": reward, "advantage": adv}
                if fallback != reward:
                    counts["fallback_recovered"] += 1
                    examples.setdefault("untagged_ended_correct_tail", []).append({**info, "ending": text[-160:]})
                if len(values) > 1:
                    multivalue.append({**info, "last_tag_payload": tagged})
                if reward and re.search(r"[A-Za-z%]", tagged):
                    units.append({**info, "last_tag_payload": tagged})
                fraction = FRACTION.fullmatch(tagged)
                if fraction and int(fraction[2]) != 0:
                    val = Fraction(int(fraction[1]), int(fraction[2]))
                    fractions.append({**info, "payload": tagged, "exact_fraction_equals_gold": val == Fraction(gold[prompt_id])})
                normalized = SPACED_CENTS.sub(r"\1.\2", tagged)
                if normalized != tagged and matches(numbers(normalized), gold[prompt_id]) != bool(reward):
                    examples.setdefault("spaced_currency_changes_score", []).append({**info, "payload": tagged})
                if reward:
                    tail = text[tags[-1].end():].strip()
                    positive_flags["nonempty_suffix"] += bool(tail)
                    positive_flags["suffix_at_least_20_chars"] += len(tail) >= 20
                    positive_flags["multiple_tags"] += len(tags) > 1
                    if sample["id"] in ("0:fresh:60:0", "2:fresh:72:0"):
                        examples.setdefault("positive_nonmath_suffix", []).append({**info, "tail": tail[:220], "token_count": len(sample["token_ids"])})
            group_class = group_type(base)
            counts[group_class] += 1
            transitions[f"{group_class}_to_{group_type(alternative)}"] += 1
            if group_class == "mixed":
                positive = len(samples[base.index(1)]["token_ids"])
                negative = len(samples[base.index(0)]["token_ids"])
                positive_lengths.append(positive)
                negative_lengths.append(negative)
                odds.append(negative / positive)
                counts["positive_shorter"] += positive < negative
        for update in updates:
            tokens = sum(groups[i]["nonzero_advantage_tokens"] for i in update["group_indices"])
            norm = update["diagnostics"]["gnorm"]
            counts["optimizer_batches"] += 1
            counts["changed_weight_batches"] += update["new_sha256"] != update["old_sha256"]
            if tokens:
                counts["active_batches"] += 1
                active_batches.append((tokens, update["diagnostics"]))
            elif norm != 0 or update["new_sha256"] != update["old_sha256"]:
                raise ValueError("Zero-signal batch moved weights")
        accounting = read(folder / "training.json")["accounting"]
        for key in ("generated_tokens", "nonzero_advantage_tokens", "optimizer_batches"):
            if counts[key] != accounting[key]:
                raise ValueError(f"Budget or update count mismatch: {seed} / {key}")
        totals.update(counts)
        seed_reports.append({"seed": seed, **counts})
    token_count = sum(n for n, _ in active_batches)
    norms = [d["gnorm"] for _, d in active_batches]
    if token_count != totals["nonzero_advantage_tokens"]:
        raise ValueError("Update token weighting disagrees with group records")
    return {
        "per_seed": seed_reports, "totals": dict(totals), "positive_flags": dict(positive_flags),
        "fallback_group_transitions": dict(transitions), "examples": examples,
        "tag_payload_review": {"multiple_numeric_literals": multivalue, "reward_one_with_letters_or_percent": units,
                               "simple_fractions": fractions,
                               "scope": "Lexical flags for reviewing number selection, fractions and units. The scorer checks no arithmetic derivation or unit equivalence."},
        "loss": {
            "active_groups": len(odds), "positive_median_length": median(positive_lengths),
            "negative_median_length": median(negative_lengths), "positive_shorter": sum(r > 1 for r in odds),
            "same_length": sum(r == 1 for r in odds),
            "median_pairwise_positive_to_negative_per_token_weight_at_ratio_one": median(odds),
            "ratio_weight_explanation": "In a mixed G2, absolute advantage matches. For sequence-normalized loss, positive/negative per-token coefficient at ratio=1 equals negative length/positive length. This proxy excludes token gradients, Adam, and probability ratios.",
            "active_tokens": token_count, "active_batches": len(norms),
            "active_preupdate_token_weighted_ratio": sum(n * d["ratio"] for n, d in active_batches) / token_count,
            "active_preupdate_token_weighted_clipfrac": sum(n * d["clipfrac"] for n, d in active_batches) / token_count,
            "active_preclip_gnorm_min_median_max": [min(norms), median(norms), max(norms)],
            "above_norm_clip_one": sum(n > 1 for n in norms),
            "active_batch_abs_loss_mean": mean(abs(d["loss"]) for _, d in active_batches),
            "scope": "Ratio and clipfrac come from stored, rounded pre-update diagnostics on nonzero-advantage tokens; clipfrac counts values outside the PPO interval. gnorm is the batch-level pre-clip norm. None measures out-of-sample accuracy or actual post-update probabilities."}}


def audit_dev(run, gold):
    q = run / "round-2/qualification"
    initial = read(q / "initial.json")["rows"]
    stats = []
    for seed in range(5):
        rows = read(q / f"seed-{seed}.json")["rows"]
        counts, flips = Counter(), Counter()
        for before, after in zip(initial, rows, strict=True):
            if before["prompt_id"] != after["prompt_id"]:
                raise ValueError("Question order changed")
            answer = gold[before["prompt_id"]]
            initial_score, initial_fallback, *_ = scores(before, answer)
            trained_score, trained_fallback, *_ = scores(after, answer)
            counts.update({"questions": 1, "initial": initial_score, "trained": trained_score,
                           "initial_fallback": initial_fallback, "trained_fallback": trained_fallback})
            if initial_score != trained_score:
                failed = before if trained_score else after
                category = "wrong_tag_value" if failed["has_answer_tag"] else "no_complete_tag"
                ended = "truncated" if failed["truncated"] else "ended"
                flips[f"{'gained' if trained_score else 'lost'}_{category}_{ended}"] += 1
        stats.append({"seed": seed, **counts, "official_flips_by_failed_output": dict(flips)})
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/gsm8k-stable-v1"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/gsm8k-reward-loss-probe/trace.json"))
    args = parser.parse_args()
    expected = read(args.run / "plan.json")["source_protocol"]["train_sha256"]
    if sha256(args.train.read_bytes()).hexdigest() != expected:
        raise ValueError("Unexpected GSM8K training data")
    gold = [Decimal(row["answer"].rsplit("####", 1)[1].strip().replace(",", "")) for row in lines(args.train)]
    result = {"run": args.run.name, "candidate": CANDIDATE, "train_sha256": expected,
              "study": "Archived-output diagnostics for five previously trained seeds. The fallback accepts a naturally ended untagged answer if its final numeric literal matches. It changes only the retrospective scores.",
              "training": audit_training(args.run, gold), "qualification": audit_dev(args.run, gold),
              "official_score_mismatches": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
