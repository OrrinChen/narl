"""Check GSM8K replay cohorts, token budgets, and paired held-out scores."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
from random import Random
import re

from experiments.audit_kv_online import read_json, read_rows, require


ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
COHORTS = ("old", "fresh_a", "fresh_b")
ARMS = {
    "homogeneous": (("old", "fresh_a"), False),
    "mixed": (("old", "fresh_a"), True),
    "same_version_homogeneous": (("fresh_a", "fresh_b"), False),
    "same_version_mixed": (("fresh_a", "fresh_b"), True),
}


def read_split(path, digest):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, f"{path} does not match the protocol checksum")
    rows = [json.loads(line) for line in data.splitlines()]
    return [(row["question"], float(row["answer"].rsplit("####", 1)[1].strip().replace(",", "")))
            for row in rows]


def check_prompt_ids(ids, rows, name):
    require(ids and len(ids) == len(set(ids))
            and all(type(index) is int and 0 <= index < len(rows) for index in ids),
            f"{name} contains duplicate or invalid prompt IDs")


def check_schedule(protocol, train, test):
    replay_ids = protocol["replay_prompt_ids"]
    snapshot_ids = protocol["snapshot_training_prompt_ids"]
    heldout_ids = protocol["heldout_prompt_ids"]
    for ids, rows, name in ((replay_ids, train, "replay"), (snapshot_ids, train, "snapshot training"),
                            (heldout_ids, test, "held-out")):
        check_prompt_ids(ids, rows, name)
    seed = protocol["seed"]
    rng = Random(seed)
    selection = protocol.get("replay_selection", "random")
    require(selection in ("random", "explicit") and not set(replay_ids) & set(snapshot_ids),
            "invalid replay selection or reuse of snapshot training IDs")
    if selection == "explicit":
        replay_set = set(replay_ids)
        candidates = [index for index in range(len(train)) if index not in replay_set]
        require(rng.sample(candidates, len(snapshot_ids)) == snapshot_ids, "wrong snapshot training schedule")
    else:
        require(rng.sample(range(len(train)), len(replay_ids) + len(snapshot_ids)) == replay_ids + snapshot_ids,
                "wrong replay or snapshot training schedule")
    require(rng.sample(range(len(test)), len(heldout_ids)) == heldout_ids, "wrong held-out schedule")
    questions = [{rows[index][0] for index in ids}
                 for rows, ids in ((train, replay_ids), (train, snapshot_ids), (test, heldout_ids))]
    require(not questions[0] & questions[2] and not questions[1] & questions[2],
            "a held-out question was reused in replay or snapshot training")
    digests = [protocol[name] for name in ("old_adapter_sha256", "fresh_adapter_sha256")]
    require(all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests) and digests[0] != digests[1],
            "protocol must record different adapter digests")
    losses = protocol["snapshot_sft_losses"]
    require(len(losses) == len(snapshot_ids) and all(isfinite(loss) and loss >= 0 for loss in losses),
            "missing or invalid snapshot training losses")
    require(protocol["sampling_seeds"] == {name: seed + offset for offset, name in enumerate(COHORTS)},
            "incorrect sampling cohort seeds")
    return replay_ids, heldout_ids


def check_score(row, gold, label):
    matches = ANSWER.findall(row["completion"])
    numbers = NUMBER.findall(matches[-1].strip().replace(",", "")) if matches else []
    try:
        correct = int(bool(numbers) and abs(float(numbers[-1]) - gold) < 1e-4)
    except ValueError:
        correct = 0
    require(row["has_answer_tag"] is bool(matches), f"{label}: wrong answer-tag flag")
    require(type(row["truncated"]) is bool and row["reward"] == correct,
            f"{label}: wrong numeric reward or truncation flag")
    return correct


def check_samples(rows, replay_ids, train, size, cap):
    require([row["prompt_id"] for row in rows] == replay_ids, "sample prompts differ from the replay schedule")
    samples = {}
    prompt_cohorts = {}
    counts = {name: Counter() for name in COHORTS}
    for row in rows:
        prompt_id = row["prompt_id"]
        require(row["prompt_ids"] and all(type(token) is int and token >= 0 for token in row["prompt_ids"]),
                f"prompt {prompt_id}: missing or invalid prompt token IDs")
        require(set(row["cohorts"]) == set(COHORTS), f"prompt {prompt_id}: missing cohort")
        prompt_cohorts[prompt_id] = {}
        for name, cohort in row["cohorts"].items():
            require(len(cohort) == size, f"prompt {prompt_id}: wrong {name} group size")
            ids = []
            correct = 0
            for sample in cohort:
                key = sample["id"]
                require(isinstance(key, str) and key and key not in samples,
                        f"prompt {prompt_id}: invalid or duplicate sample ID")
                correct += check_score(sample, train[prompt_id][1], key)
                length = len(sample["token_ids"])
                require(1 <= length <= cap and (not sample["truncated"] or length == cap)
                        and all(type(token) is int and token >= 0 for token in sample["token_ids"]),
                        f"{key}: invalid completion length or token IDs")
                require(len(sample["behavior_logprobs"]) == length
                        and all(isfinite(value) and value <= 0 for value in sample["behavior_logprobs"]),
                        f"{key}: invalid behavior logprobs")
                samples[key] = (name, length, sample["reward"])
                ids.append(key)
                counts[name]["tokens"] += length
                counts[name]["truncated"] += sample["truncated"]
                counts[name]["tagged"] += sample["has_answer_tag"]
            counts[name]["samples"] += size
            counts[name]["correct"] += correct
            counts[name]["mixed_reward_groups"] += 0 < correct < size
            prompt_cohorts[prompt_id][name] = ids
    return samples, prompt_cohorts, counts


def check_assignments(arm, report, replay_ids, prompt_cohorts, samples, size, seed):
    cohorts, mixed = ARMS[arm]
    assignments = report["assignments"]
    updates = report["updates"]
    require(all([row["prompt_id"] for row in rows] == replay_ids for rows in (assignments, updates)),
            f"{arm}: wrong prompt schedule")
    consumed_tokens = 0
    mixed_reward_groups = 0
    nonzero_advantage_tokens = 0
    nonzero_gradient_updates = 0
    advantages = {}
    for row, update in zip(assignments, updates, strict=True):
        prompt = row["prompt_id"]
        groups = row["groups"]
        require(len(groups) == 2 and all(len(group) == size for group in groups),
                f"{arm}/{prompt}: wrong group dimensions")
        ids = groups[0] + groups[1]
        expected = {key for name in cohorts for key in prompt_cohorts[prompt][name]}
        require(len(ids) == len(set(ids)) and set(ids) == expected,
                f"{arm}/{prompt}: samples differ from the cohort pool")
        first, second = [prompt_cohorts[prompt][name][:] for name in cohorts]
        if mixed:
            rng = Random(seed + prompt)
            rng.shuffle(first)
            rng.shuffle(second)
            half = size // 2
            expected_order = [first[:half] + second[:half], first[half:] + second[half:]]
        else:
            expected_order = [first, second]
        require(groups == expected_order, f"{arm}/{prompt}: group assignment differs from the protocol")
        useful_groups = []
        for group in groups:
            rate = sum(samples[key][2] for key in group) / size
            # Sample standard deviation of binary rewards, with GRPO's epsilon.
            scale = sqrt(rate * (1 - rate) * size / (size - 1)) + 1e-4
            advantages.update((key, (samples[key][2] - rate) / scale) for key in group)
            if 0 < rate < 1:
                useful_groups.append(group)
        useful_tokens = sum(samples[key][1] for group in useful_groups for key in group)
        require(update["nonzero_advantage_tokens"] == useful_tokens,
                f"{arm}/{prompt}: nonzero advantage count disagrees with rewards")
        require(all(isfinite(value) for value in update.values()) and update["gnorm"] >= 0,
                f"{arm}/{prompt}: nonfinite metrics or invalid gradient norm")
        nonzero_gradient_updates += update["gnorm"] > 0
        nonzero_advantage_tokens += useful_tokens
        mixed_reward_groups += len(useful_groups)
        consumed_tokens += sum(samples[key][1] for key in ids)
    require(report["consumed_tokens"] == consumed_tokens, f"{arm}: consumed token count differs from assignments")
    return {"consumed_tokens": consumed_tokens, "mixed_reward_groups": mixed_reward_groups,
            "groups": 2 * len(assignments), "nonzero_advantage_tokens": nonzero_advantage_tokens,
            "nonzero_gradient_updates": nonzero_gradient_updates}, advantages


def check_evaluated(rows, heldout_ids, test, label):
    require([row["prompt_id"] for row in rows] == heldout_ids, f"{label}: held-out prompt schedule differs")
    return [check_score(row, test[row["prompt_id"]][1], f"{label}/{row['prompt_id']}") for row in rows]


def paired_result(baseline, candidate, base_adv, candidate_adv):
    outcomes = Counter((a, b) for a, b in zip(baseline, candidate, strict=True))
    require(base_adv.keys() == candidate_adv.keys(), "comparison arms contain different sample IDs")
    differences = [abs(base_adv[key] - candidate_adv[key]) for key in base_adv]
    return {"baseline_correct": sum(baseline), "candidate_correct": sum(candidate),
            "wins": outcomes[0, 1], "losses": outcomes[1, 0],
            "accuracy_delta": (sum(candidate) - sum(baseline)) / len(baseline),
            "changed_advantages": sum(delta > 1e-9 for delta in differences),
            "mean_absolute_advantage_change": sum(differences) / len(differences),
            "completion_count": len(differences)}


def audit(run, train_path, test_path):
    protocol = read_json(run / "protocol.json")
    report = read_json(run / "summary.json")
    train = read_split(train_path, protocol["train_sha256"])
    test = read_split(test_path, protocol["test_sha256"])
    replay_ids, heldout_ids = check_schedule(protocol, train, test)
    size = protocol["group_size"]
    cap = protocol["max_new_tokens"]
    require(type(size) is int and size >= 2 and size % 2 == 0 and type(cap) is int and cap > 0,
            "invalid replay group size or output cap")
    samples, prompt_cohorts, counts = check_samples(read_rows(run / "samples.jsonl"), replay_ids, train, size, cap)
    require(set(report["arms"]) == set(ARMS), "comparison arms do not match the protocol")
    initial = check_evaluated(report["initial_eval"], heldout_ids, test, "initial")
    scores = {}
    verified = {}
    advantages = {}
    for arm, arm_report in report["arms"].items():
        verified[arm], advantages[arm] = check_assignments(
            arm, arm_report, replay_ids, prompt_cohorts, samples, size, protocol["seed"],
        )
        scores[arm] = check_evaluated(arm_report["evaluated"], heldout_ids, test, arm)
        correct = sum(scores[arm])
        require(arm_report["heldout_correct"] == correct, f"{arm}: wrong held-out correct count")
        verified[arm].update(correct=correct, accuracy=correct / len(heldout_ids))
    return {"initial_correct": sum(initial), "heldout_count": len(heldout_ids),
            "cohorts": {name: dict(values) for name, values in counts.items()}, "arms": verified,
            "mixed_vs_homogeneous": paired_result(scores["homogeneous"], scores["mixed"],
                                                 advantages["homogeneous"], advantages["mixed"]),
            "same_version_shuffle_control": paired_result(scores["same_version_homogeneous"],
                                                          scores["same_version_mixed"],
                                                          advantages["same_version_homogeneous"],
                                                          advantages["same_version_mixed"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.train, args.test)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Audited {result['heldout_count']} held-out questions. Saved {args.out}")


if __name__ == "__main__":
    main()
