"""Rebuild matched GSM8K budgets, grouping, advantages and paired scores from saved rows."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import comb, isfinite, sqrt
from pathlib import Path
from random import Random
import re
from statistics import median

from experiments.audit_gsm8k_regroup import check_evaluated, check_prompt_ids, check_score, paired_result, read_split
from experiments.audit_kv_online import check_reported, read_json, read_rows, require


ARMS = ("version_homogeneous", "version_mixed", "same_version_shuffle")
INSPECTED_TEST_IDS = (106, 419, 426, 533, 629, 765, 771, 799, 927)


def check_development(path, protocol, qualification, train):
    dev_ids = protocol["development_prompt_ids"]
    train_ids = protocol["training_prompt_ids"]
    check_prompt_ids(dev_ids, train, "development")
    check_prompt_ids(train_ids, train, "SFT")
    require(not {train[i][0] for i in dev_ids} & {train[i][0] for i in train_ids}, "development overlaps SFT")
    require(dev_ids + train_ids == Random(protocol["seed"]).sample(range(len(train)), len(dev_ids) + len(train_ids)),
            "development split differs from its seeded schedule")
    selected = qualification["selected_step"]
    training = read_json(path / "development.json")
    require([row["step"] for row in training["training"]] == list(range(1, selected + 1))
            and [row["prompt_id"] for row in training["training"]] == train_ids[:selected],
            "SFT schedule differs from the saved development split")
    require(all(isfinite(row["loss"]) and row["loss"] >= 0
                and isfinite(row["gradient_norm"]) and row["gradient_norm"] >= 0 for row in training["training"]),
            "invalid SFT metrics")
    cap = protocol["max_new_tokens"]
    baseline = training["conditions"][f"{protocol['prompt_mode']}-{cap}"]
    before = sum(score_evaluation(baseline, dev_ids, train, "development baseline", cap))
    checkpoints = training["checkpoints"]
    expected = [step for step in protocol["checkpoints"] if step <= selected]
    require(set(checkpoints) == {str(step) for step in expected}, "missing or unexpected development checkpoints")
    scores = [sum(score_evaluation(checkpoints[str(step)], dev_ids, train, f"development step {step}", cap))
              for step in expected]
    require(qualification["qualified"] is True and qualification["development_count"] == len(dev_ids)
            and before == qualification["before_correct"] and scores[-1] == qualification["after_correct"]
            and scores[-1] - before >= protocol["minimum_correct_gain"]
            and all(score - before < protocol["minimum_correct_gain"] for score in scores[:-1]),
            "development outcomes do not support the first qualifying checkpoint")
    return {"before_correct": before, "after_correct": scores[-1], "count": len(dev_ids), "selected_step": selected,
            "checkpoint_scores": dict(zip(expected, scores, strict=True))}


def check_splits(protocol, train, test):
    dev = protocol["qualification_protocol"]
    replay = protocol["replay_prompt_ids"]
    heldout = protocol["heldout_prompt_ids"]
    splits = [dev["training_prompt_ids"], dev["development_prompt_ids"], replay, heldout]
    for ids, rows, name in zip(splits, (train, train, train, test), ("SFT", "development", "replay", "test"), strict=True):
        check_prompt_ids(ids, rows, name)
    questions = set()
    for ids, rows in zip(splits, (train, train, train, test), strict=True):
        current = {rows[i][0] for i in ids}
        require(not current & questions, "a question leaks across splits")
        questions |= current
    require(protocol["excluded_test_ids"] == list(INSPECTED_TEST_IDS), "inspected held-out set changed")
    excluded_train = set(splits[0] + splits[1])
    rng = Random(protocol["seed"])
    require(replay == rng.sample([i for i in range(len(train)) if i not in excluded_train], len(replay)),
            "replay prompt schedule differs from the frozen seed")
    require(heldout == rng.sample([i for i in range(len(test)) if i not in INSPECTED_TEST_IDS], len(heldout)),
            "held-out prompt schedule differs from the frozen seed")
    qualification = protocol["qualification"]
    before, after = qualification["before_correct"], qualification["after_correct"]
    require(qualification["qualified"] is True and 0 <= before <= after <= len(splits[1])
            and qualification["development_count"] == len(splits[1])
            and after - before >= dev["minimum_correct_gain"] and qualification["selected_step"] in dev["checkpoints"],
            "independent development gate did not pass")
    require((dev["dtype"], dev["device"], dev["max_new_tokens"], dev["eval_batch"]) ==
            (protocol["dtype"], protocol["device"], protocol["eval_max_new_tokens"], protocol["eval_batch"]),
            "held-out evaluation protocol differs from the independent development evaluation")
    return replay, heldout


def check_rollouts(records, replay_ids, train, size, cap):
    require([row["prompt_id"] for row in records] == replay_ids, "rollout IDs differ from frozen protocol")
    counts = {version: Counter(samples=0, tokens=0, correct=0, tagged=0, truncated=0) for version in ("old", "fresh")}
    pools = {}
    for record in records:
        index = record["prompt_id"]
        require(record["prompt_ids"] and all(type(token) is int and token >= 0 for token in record["prompt_ids"]),
                f"prompt {index} has invalid token IDs")
        require(set(record["cohorts"]) == set(counts), f"prompt {index} has missing cohorts")
        pools[index] = []
        for version, stats in counts.items():
            cohort = record["cohorts"][version]
            require(len(cohort) == 2 * size, f"prompt {index}/{version} has the wrong group size")
            for sample_index, sample in enumerate(cohort):
                expected_id = f"{index}:{version}:{sample_index}"
                require(sample["id"] == expected_id, f"unexpected sample ID {sample['id']}")
                correct = check_score(sample, train[index][1], expected_id)
                length = len(sample["token_ids"])
                require(1 <= length <= cap and (not sample["truncated"] or length == cap)
                        and all(type(token) is int and token >= 0 for token in sample["token_ids"]),
                        f"{expected_id} has invalid completion token IDs")
                require(len(sample["behavior_logprobs"]) == length
                        and all(isfinite(value) and value <= 0 for value in sample["behavior_logprobs"]),
                        f"{expected_id} has invalid behavior logprobs")
                pools[index].append((expected_id, correct, length))
                stats.update(samples=1, tokens=length, correct=correct, tagged=sample["has_answer_tag"],
                             truncated=sample["truncated"])
    return pools, {name: dict(stats) for name, stats in counts.items()}


def expected_groups(pool, size, seed, arm):
    groups = [pool[start:start + size] for start in range(0, len(pool), size)]
    if arm == "version_homogeneous":
        return groups
    rng = Random(seed)
    for group in groups:
        rng.shuffle(group)
    a, b, c, d = groups
    pairs = [(a, c), (b, d)] if arm == "version_mixed" else [(a, b), (c, d)]
    front = [left[:size // 2] + right[:size // 2] for left, right in pairs]
    back = [left[size // 2:] + right[size // 2:] for left, right in pairs]
    return front + back


def check_arm(arm, report, pools, protocol):
    require([row["prompt_id"] for row in report["updates"]] == list(pools), f"{arm} has the wrong update schedule")
    require(report["initial_adapter_sha256"] == protocol["qualification"]["fresh_adapter_sha256"],
            f"{arm} starts from the wrong adapter")
    require(re.fullmatch(r"[0-9a-f]{64}", report["final_adapter_sha256"]), f"{arm} is missing the final adapter hash")
    totals = Counter(completion_tokens=0, nonzero_advantage_tokens=0, mixed_reward_groups=0, nonzero_gradient_updates=0)
    advantages = {}
    size = protocol["group_size"]
    for row in report["updates"]:
        index = row["prompt_id"]
        groups = expected_groups(pools[index], size, protocol["seed"] + index, arm)
        require(row["groups"] == [[sample[0] for sample in group] for group in groups],
                f"{arm}/{index} changes the completion pool or group assignments")
        tokens = signal_tokens = 0
        for group in groups:
            rate = sum(sample[1] for sample in group) / size
            std = sqrt(rate * (1 - rate) * size / (size - 1))
            totals["mixed_reward_groups"] += int(0 < rate < 1)
            for key, reward, length in group:
                advantage = (reward - rate) / (std + 1e-4)
                advantages[key] = advantage
                tokens += length
                signal_tokens += length if advantage else 0
        require(row["completion_tokens"] == tokens and row["nonzero_advantage_tokens"] == signal_tokens,
                f"{arm}/{index} has inconsistent token counts")
        diagnostics = row["diagnostics"]
        require(all(isfinite(value) for value in diagnostics.values()) and diagnostics["gnorm"] >= 0,
                f"{arm}/{index} has invalid gradient diagnostics")
        totals.update(completion_tokens=tokens, nonzero_advantage_tokens=signal_tokens,
                      nonzero_gradient_updates=int(diagnostics["gnorm"] > 0))
    require(report["consumed_tokens"] == totals["completion_tokens"], f"{arm} consumed token total is inconsistent")
    if totals["nonzero_gradient_updates"]:
        require(report["initial_adapter_sha256"] != report["final_adapter_sha256"], f"{arm} did not change weights")
    return dict(totals), advantages


def score_evaluation(report, heldout, test, label, cap):
    rows = report["rows"]
    scores = check_evaluated(rows, heldout, test, label)
    for row in rows:
        length = len(row["token_ids"])
        require(1 <= length <= cap and (not row["truncated"] or length == cap), f"{label} has invalid evaluation length")
    stats = {"correct": sum(scores), "tagged": sum(row["has_answer_tag"] for row in rows),
             "truncated": sum(row["truncated"] for row in rows)}
    if "median_completion_tokens" in report:
        stats["median_completion_tokens"] = median(len(row["token_ids"]) for row in rows)
    check_reported(report, stats, label)
    return scores


def compare(baseline, candidate, scores, advantages, outputs):
    result = paired_result(scores[baseline], scores[candidate], advantages[baseline], advantages[candidate])
    result.update(score_change(scores[baseline], scores[candidate], outputs[baseline], outputs[candidate]))
    return result


def score_change(baseline, candidate, baseline_outputs, candidate_outputs):
    outcomes = Counter(zip(baseline, candidate, strict=True))
    wins, losses = outcomes[0, 1], outcomes[1, 0]
    result = {"wins": wins, "losses": losses, "accuracy_delta": (wins - losses) / len(baseline),
              "identical_greedy_token_outputs": sum(a == b for a, b in zip(baseline_outputs, candidate_outputs, strict=True))}
    discordant = result["wins"] + result["losses"]
    tail = sum(comb(discordant, i) for i in range(min(result["wins"], result["losses"]) + 1))
    result["exact_mcnemar_p"] = min(1.0, 2 * tail / 2 ** discordant)
    return result


def audit(run, train_path, test_path, qualification_path):
    protocol = read_json(run / "protocol.json")
    report = read_json(run / "summary.json")
    train = read_split(train_path, protocol["train_sha256"])
    test = read_split(test_path, protocol["test_sha256"])
    require(read_json(qualification_path / "protocol.json") == protocol["qualification_protocol"]
            and read_json(qualification_path / "dev-qualified.json") == protocol["qualification"],
            "development files differ from the frozen qualification")
    dev_report = check_development(qualification_path, protocol["qualification_protocol"], protocol["qualification"], train)
    replay, heldout = check_splits(protocol, train, test)
    size, cap = protocol["group_size"], protocol["max_new_tokens"]
    require(size >= 2 and size % 2 == 0 and cap > 0 and protocol["groups_per_prompt"] == 4
            and protocol["rollout_batch"] == 2 * size, "invalid group budgets")
    samples_path = run / "samples.jsonl"
    require(sha256(samples_path.read_bytes()).hexdigest() == report["sample_sha256"], "rollout pool digest does not match")
    pools, cohorts = check_rollouts(read_rows(samples_path), replay, train, size, cap)
    require(set(report["arms"]) == set(ARMS), "completed arms differ from protocol")
    verified = {}
    scores = {}
    outputs = {}
    advantages = {}
    eval_cap = protocol["eval_max_new_tokens"]
    initial = score_evaluation(report["initial_evaluation"], heldout, test, "initial", eval_cap)
    initial_outputs = [row["token_ids"] for row in report["initial_evaluation"]["rows"]]
    for arm, item in report["arms"].items():
        verified[arm], advantages[arm] = check_arm(arm, item, pools, protocol)
        scores[arm] = score_evaluation(item["evaluation"], heldout, test, arm, eval_cap)
        outputs[arm] = [row["token_ids"] for row in item["evaluation"]["rows"]]
        verified[arm].update(correct=sum(scores[arm]), accuracy=sum(scores[arm]) / len(heldout),
                             vs_initial=score_change(initial, scores[arm], initial_outputs, outputs[arm]))
    budgets = {item["completion_tokens"] for item in verified.values()}
    generated = sum(item["tokens"] for item in cohorts.values())
    require(budgets == {generated}, "arms did not consume the same generated token budget")
    return {"development_gate": dev_report, "heldout_count": len(heldout), "initial_correct": sum(initial),
            "generated_tokens": generated,
            "cohorts": cohorts, "arms": verified,
            "mixed_vs_homogeneous": compare(ARMS[0], ARMS[1], scores, advantages, outputs),
            "same_version_shuffle_vs_homogeneous": compare(ARMS[0], ARMS[2], scores, advantages, outputs),
            "mixed_vs_same_version_shuffle": compare(ARMS[2], ARMS[1], scores, advantages, outputs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "train", "test", "qualification", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.train, args.test, args.qualification)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Audited {result['generated_tokens']} matched tokens and {result['heldout_count']} held-out questions")


if __name__ == "__main__":
    main()
