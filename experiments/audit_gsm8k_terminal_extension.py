"""Rescore the two stored training trajectories and the held-back archival answers."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from experiments.gsm8k_training_reward_v2 import score_training_reward as v2
from experiments.gsm8k_training_reward_v3 import score_training_reward as v3


ARCHIVES = ("gsm8k-dev-mlx-256", "gsm8k-dev-mlx-final", "gsm8k-budget-5seed", "gsm8k-accuracy-v1")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def archive_rows(obj, locator=""):
    if isinstance(obj, dict) and {"prompt_id", "completion", "truncated"} <= obj.keys():
        yield locator, obj
    elif isinstance(obj, (dict, list)):
        items = obj.items() if isinstance(obj, dict) else enumerate(obj)
        for key, child in items:
            yield from archive_rows(child, f"{locator}/{key}")


def cohort():
    return {"source_sha256": {}, "v2_v3_counts": Counter(), "changes": []}


def rescore(row, pid, gold, path, locator, report):
    old, _ = v2(row, gold)
    new, why = v3(row, gold)
    report["v2_v3_counts"][f"{old}/{new}"] += 1
    if new != old:
        report["changes"].append({"source": str(path), "locator": locator, "prompt_id": pid,
                                  "sample_id": row.get("id"), "gold": gold, "v2": old, "v3": new, "reason": why,
                                  "completion_sha256": sha256(row["completion"].encode()).hexdigest(),
                                  "ending": row["completion"][-230:]})
    return old, new


def training(folder, gold, excluded, verify_archived_reward):
    report = cohort()
    report["preference_transitions"] = Counter()
    report["seeds"] = {}
    for seed in range(5):
        path = folder / f"seed-{seed}/trace.jsonl"
        report["source_sha256"][str(path)] = digest(path)
        counts = Counter()
        transitions = Counter()
        for row in map(json.loads, path.read_text().splitlines()):
            pid = row["prompt_id"]
            excluded.add(pid)
            scores = []
            for slot, sample in enumerate(row["samples"]):
                locator = f'group-{row["group_index"]}/slot-{slot}'
                old, new = rescore(sample, pid, gold[pid], path, locator, report)
                if verify_archived_reward and old != sample["reward"]:
                    raise ValueError(f"v2 differs from the archived online training reward at {path} {locator}")
                scores.append((old, new))
                counts[f"{old}/{new}"] += 1
            if len(scores) != 2:
                continue
            before, after = zip(*scores)
            old = before[0] - before[1]
            new = after[0] - after[1]
            if old == new:
                event = "same_active" if old else "same_tie"
            else:
                event = "activated" if not old else "deactivated" if not new else "reversed"
            transitions[event] += 1
        report["preference_transitions"].update(transitions)
        report["seeds"][str(seed)] = {"v2_v3_counts": counts, "preference_transitions": transitions,
                                      "matched_all_rewards": counts["0/1"] == counts["1/0"] == 0}
    return report


def archives(root, gold, excluded):
    report = cohort()
    seen = set()
    raw = filtered = 0
    for name in ARCHIVES:
        path = root / name / "development.json"
        report["source_sha256"][str(path)] = digest(path)
        for locator, sample in archive_rows(read(path)):
            pid = sample["prompt_id"]
            raw += 1
            if pid in excluded:
                filtered += 1
                continue
            key = (pid, sample["completion"])
            if key not in seen:
                seen.add(key)
                rescore(sample, pid, gold[pid], path, locator, report)
    report.update(raw_records=raw, removed_prior_prompt=filtered, unique_samples=len(seen),
                  unique_prompts=len({pid for pid, _ in seen}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    root = args.runs
    target = root / "gsm8k-reward-online-v2"
    review = target / "parser-review.json"
    excluded = {pid for subset in read(review)["sets"] for pid in subset["included_prompt_ids"]}
    data = root / "gsm8k-data/train.jsonl"
    gold = [json.loads(line)["answer"].rsplit("####", 1)[1].strip().replace(",", "")
            for line in data.read_text().splitlines()]
    old = training(root / "gsm8k-stable-v1/models/lr-0.0001-groups-16", gold, excluded, False)
    online = training(target / "parsed_outcome/models", gold, excluded, True)
    archive = archives(root, gold, excluded)
    s1 = online["seeds"]["1"]
    s1_train = read(target / "parsed_outcome/models/seed-1/training.json")
    expected_hash = online["source_sha256"][str(target / "parsed_outcome/models/seed-1/trace.jsonl")]
    if not s1["matched_all_rewards"] or s1_train["trace_sha256"] != expected_hash:
        raise ValueError("Cannot infer the unchanged seed-1 training trajectory")
    if s1_train["final_sha256"] != "ff92fed879c5b44e2b7e9b7ab25e9a1f1ce2116f69d47faefb5272dc13445680":
        raise ValueError("Previously reported seed-1 score belongs to a different endpoint")
    source = Path(__file__).parent
    result = {
        "scope": "Offline rescoring. No candidate training or held-out inference was run.",
        "cohort_rule": "Include all samples from both five-seed training archives. Exclude all prompt IDs used by either training cohort or previously reviewed parser sets from the old development archives. Deduplicate exact (prompt_id, completion) within the archival comparison.",
        "train_data_sha256": digest(data), "parser_review_sha256": digest(review),
        "code_sha256": {f.name: digest(f) for f in (source / "gsm8k_training_reward.py",
                                                     source / "gsm8k_training_reward_v2.py",
                                                     source / "gsm8k_training_reward_v3.py", Path(__file__))},
        "cohorts": {"old": old, "parsed_online": online, "archival_check": archive},
        "seed_1_conditional_inference": {
            "v2_v3_counts": s1["v2_v3_counts"], "trace_sha256": expected_hash,
            "complete_groups": sum(s1["preference_transitions"].values()),
            "v2_endpoint_sha256": s1_train["final_sha256"],
            "v2_previously_reported_q1_correct": 44, "v2_previously_reported_q1_count": 128,
            "v2_score_source": "Published v2 seed-1 result supplied for this fixed run; this trace audit does not open development answers.",
            "v3_q1_measured": False,
            "basis": "Every reward on this v2 trajectory exactly equals the binary v3 reward. With the same deterministic runtime, initial state, sampling stream and optimizer, the same batch therefore produces the same update inductively. Replacing the binary reward alone would reproduce the v2 endpoint and its 44/128 score. No such inference is made for a changed reward stream, steps, or format bonuses."},
        "manual_review": "All 15 newly positive outputs in the fixed terminal-form probe were read against the training gold. All explicitly declared the matching final numeric value; this does not certify the derivation. The final implementation reproduced the same changed output strings. Recall over all unrecognized outputs was not exhaustively audited."}
    (target / "terminal-extension.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
