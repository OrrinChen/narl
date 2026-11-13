"""Reconstruct sequence scheduling results from CPU rollout and evaluation traces."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random
import re
from statistics import median

from experiments.audit_kv_online import check_reported, read_json, read_rows, require


QUERIES = ("red", "blue", "green", "gold")
LENGTH_SCHEDULE = (3, 12, 3, 16)
BUCKETS = ("short", "long")
ARITHMETIC_BUCKETS = ("all", "noncarry", "carry")


def workload(count, filler_start, distractors):
    fillers = Random(97 + filler_start).sample(range(filler_start, filler_start + count), count)
    rows = []
    for index, filler_id in enumerate(fillers):
        block, query = divmod(index, len(QUERIES))
        length = LENGTH_SCHEDULE[block % len(LENGTH_SCHEDULE)]
        distractor = distractors[(block // len(LENGTH_SCHEDULE)) % len(distractors)]
        filler = " ".join(format(filler_id, "012b"))
        rows.append({"prompt": f"{filler} {distractor} {QUERIES[query]}",
                     "answer_tokens": [str((query + step) % 2) for step in range(length)],
                     "completion_length": length, "bucket": "short" if length == 3 else "long",
                     "query": QUERIES[query], "distractor": distractor, "filler": filler})
    return rows


def arithmetic_workloads(train_count, eval_count, development):
    """Independently reconstruct disjoint unordered operand pairs and answers."""
    train, heldout = [], []
    offset = 0 if development else eval_count // 2
    pairs = list(combinations_with_replacement(range(10, 100), 2))
    for index, bucket in enumerate(ARITHMETIC_BUCKETS[1:]):
        selected = [(a, b) for a, b in pairs if (a % 10 + b % 10 >= 10) == (bucket == "carry")]
        Random(2041 + index).shuffle(selected)
        rows = [{"prompt": f"add {left % 10} {right % 10} + {left // 10} {right // 10} =",
                 "answer_tokens": list(f"{left + right:03d}"[::-1]), "completion_length": 3,
                 "bucket": bucket, "left": left, "right": right} for left, right in selected]
        heldout.extend(rows[offset:offset + eval_count // 2])
        train.extend(rows[eval_count:eval_count + train_count // 2])
    require(len(train) == train_count and len(heldout) == eval_count, "unbalanced arithmetic data budget")
    return train, heldout


def check_group(row, problem, protocol, vocab, version, arm, window):
    size, prefix_length = protocol["group_size"], protocol["prompt_tokens"]
    length = problem["completion_length"]
    require(all(row[key] == problem[key] for key in ("prompt", "bucket", "completion_length")),
            "unexpected dispatch workload")
    require(0 <= row["slot"] < window and 0 <= row["actor_id"] < protocol["actor_count"],
            "invalid actor or reservation slot")
    require(0 <= row["generated_s"] <= row["received_s"] and row["dispatch_wait_s"] >= 0,
            "invalid rollout timing")
    prompt = [vocab[token] for token in problem["prompt"].split()]
    digits = "0123456789" if protocol.get("task") == "arithmetic" else "01"
    action_ids = [vocab[digit] for digit in digits]
    prefix = (action_ids[:2] * ((prefix_length - len(prompt) + 1) // 2) + prompt)[-prefix_length:]
    answer = [vocab[token] for token in problem["answer_tokens"]]
    require(len(row["tokens"]) == len(row["behavior_logprobs"]) == len(row["rewards"]) == size,
            "incomplete rollout group")
    for tokens, logprobs, reward in zip(row["tokens"], row["behavior_logprobs"], row["rewards"]):
        completion = tokens[prefix_length:]
        require(tokens[:prefix_length] == prefix and len(completion) == length
                and all(type(token) is int and token in action_ids for token in completion),
                "wrong prefix, completion length or action support")
        require(len(logprobs) == length and all(isfinite(logp) and logp <= 0 for logp in logprobs),
                "invalid behavior logprobs or missing credited action positions")
        correct = sum(pred == gold for pred, gold in zip(completion, answer))
        expected_reward = float(correct == length) if protocol.get("reward_kind") == "exact_match" else correct / length
        require(isclose(reward, expected_reward, abs_tol=1e-7), "reward disagrees with the answer verifier")

    initial, reserved = row["initial_version"], row["reserved_version"]
    require(row["learner_version"] == version and 0 <= reserved <= initial <= version,
            "invalid learner or behavior version order")
    lag = version - initial
    require(row["lag"] == lag, "incorrect behavior lag")
    prefills = size if arm == "recompute" else 1
    require(row["initial_prompt_tokens_computed"] == prefills * prefix_length, "wrong number of initial prefills")
    versions = [initial] * length
    require(len(row["switches"]) <= 1, "more than one weight switch")
    for switch in row["switches"]:
        boundary, new = switch["after_tokens"], switch["to"]
        require(0 < boundary < length and boundary % 4 == 0 and switch["from"] == initial
                and initial < new <= version, "invalid weight switch boundary or version")
        versions[boundary:] = [new] * (length - boundary)
        require(switch["prompt_tokens_recomputed"] == prefills * prefix_length
                and switch["suffix_tokens_recomputed"] == size * (boundary - 1), "wrong KV rebuild token counts")
        compute, copy, recovery = (switch[key] for key in ("recompute_ms", "cache_copy_ms", "recovery_ms"))
        require(0 <= compute and 0 <= copy and compute + copy <= recovery + 1e-6,
                "invalid KV rebuild timing")
    require(row["version_by_token"] == versions, "behavior versions disagree with switch span")

    status = row["status"]
    require(status in ("consumed", "stale", "unbatched") and (status == "stale") == (lag > protocol["max_lag"]),
            "group status disagrees with expiry")
    require(arm != "version_aware" or (status == "consumed" and version - reserved <= protocol["max_lag"]),
            "version-aware reservation expired or went unbatched")
    rewards = Counter(row["rewards"])
    if status == "consumed" and len(rewards) > 1:
        # FP32 reduction order can leave a tiny nonzero advantage at a mathematical
        # tie. One trajectory's advantage must still cover all its action positions.
        mean = sum(row["rewards"]) / size
        possible_rows = {size, *(size - count for reward, count in rewards.items() if abs(reward - mean) <= 1e-6)}
    else:
        possible_rows = {0}
    require(row["loss_tokens"] in {length * count for count in possible_rows},
            "wrong GRPO credit count for full completion mask")


def check_evaluations(evals, groups, updates, published_versions, heldout, protocol, bucket_names, ready):
    args = protocol["args"]
    expected_counts = [*range(0, args["groups"], args["eval_every"]), args["groups"]]
    require([row["received_groups"] for row in evals] == expected_counts, "evaluation schedule disagrees with protocol")
    size = protocol["group_size"]
    arrival = sorted(groups, key=lambda row: row["receive_order"])
    require([row["receive_order"] for row in arrival] == list(range(1, len(groups) + 1)),
            "missing or duplicate receive order")
    buckets = Counter(problem["bucket"] for problem in heldout)
    if "all" in bucket_names:
        buckets["all"] = len(heldout)
    previous_time = -1.0
    targets = dict.fromkeys(bucket_names)
    snapshot_targets = dict.fromkeys(bucket_names)
    previous = None
    changed_only = protocol.get("publication") == "changed_weights_only"
    for row in evals:
        now, count = row["time_s"], row["received_groups"]
        start = row.get("eval_started_s", now)
        require(isfinite(now) and previous_time <= start <= now, "evaluation clock goes backwards")
        if ready is not None:
            if count:
                require(ready == row["training_started_s"] <= start
                        and isclose(row["training_time_s"], now - ready, abs_tol=1e-9),
                        "inconsistent training-only evaluation time")
            else:
                require(row.get("training_started_s", 0.0) == row.get("training_time_s", 0.0) == 0.0,
                        "initial evaluation counted timed training")
        previous_time = now
        require(row["received_tokens"] == size * sum(group["completion_length"] for group in arrival[:count]),
                "evaluation received-token count disagrees with arrival order")
        require(count == sum(group["received_s"] <= start for group in groups),
                "evaluation arrived before its received-group count")
        ordinal = sum(update["finished_s"] <= start for update in updates)
        require(row["update"] == ordinal, "evaluation does not match a completed training batch")
        if changed_only:
            require(row["policy_version"] == published_versions[ordinal],
                    "evaluation does not match the published policy version")
            if previous is not None and row["policy_version"] == previous["policy_version"]:
                require(row["accuracy"] == previous["accuracy"], "accuracy changed on the same policy snapshot")
        require(set(row["accuracy"]) == set(bucket_names), "evaluation is missing a bucket")
        for bucket, score in row["accuracy"].items():
            hits = score * buckets[bucket]
            require(0 <= score <= 1 and isclose(hits, round(hits), abs_tol=1e-8),
                    "bucket accuracy incompatible with held-out size")
        if "all" in bucket_names:
            matches = sum(row["accuracy"][bucket] * buckets[bucket] for bucket in bucket_names if bucket != "all")
            require(isclose(matches, row["accuracy"]["all"] * buckets["all"], abs_tol=1e-8),
                    "overall exact match does not agree with the carry strata")
        row["actor_completed_tokens"] = size * sum(group["completion_length"] for group in groups
                                                   if group["generated_s"] <= now)
        require(row["actor_completed_tokens"] >= row["received_tokens"], "rollout budget completed after receipt")
        if previous is not None and 0 < previous["update"] < row["update"]:
            for bucket in bucket_names:
                if min(previous["accuracy"][bucket], row["accuracy"][bucket]) >= args["threshold"]:
                    if targets[bucket] is None:
                        targets[bucket] = row
                    if (changed_only and snapshot_targets[bucket] is None
                            and previous["policy_version"] < row["policy_version"]):
                        snapshot_targets[bucket] = row
        previous = row
    require(evals[0]["time_s"] == 0 and evals[0]["update"] == 0, "initial evaluation was not on untrained weights")
    return targets, snapshot_targets


def retention(groups, size):
    statuses = Counter(row["status"] for row in groups)
    by_status = Counter()
    for row in groups:
        by_status[row["status"]] += size * row["completion_length"]
    consumed = [row for row in groups if row["status"] == "consumed"]
    post_swap = [row for row in consumed if row["switches"] and row["loss_tokens"]]
    total = sum(by_status.values())
    return {"statuses": dict(statuses), "groups": len(groups), "generated_tokens": total,
            "tokens_by_status": dict(by_status),
            "retained_group_fraction": len(consumed) / len(groups) if groups else None,
            "retained_token_fraction": by_status["consumed"] / total if total else None,
            "post_swap_nonzero_advantage_groups": len(post_swap),
            # The runner's legacy metric counts every row in a group with any
            # signal. The exact count excludes trajectories with zero advantage.
            "post_swap_nonzero_advantage_tokens": size * sum(
                row["completion_length"] - row["switches"][0]["after_tokens"] for row in post_swap),
            "exact_post_swap_nonzero_advantage_tokens": sum(
                row["loss_tokens"] // row["completion_length"]
                * (row["completion_length"] - row["switches"][0]["after_tokens"]) for row in post_swap)}


def check_publication(row, group_rows, protocol, version, weight_hash):
    changed = row["weights_changed"]
    before, after = row["weight_sha256_before"], row["weight_sha256_after"]
    require(type(changed) is bool and re.fullmatch(r"[0-9a-f]{64}", after) is not None
            and before == weight_hash and changed == (before != after), "broken update weight hash chain")
    require(row["version_before"] == version and row["version_after"] == version + int(changed),
            "policy version must advance exactly when weights change")
    require(0 <= row.get("publication_s", 0.0) <= row["update_s"], "invalid publication timing")
    size = protocol["group_size"]
    skipped = row["diagnostics"]["skipped"]
    expected_skips = size * sum(group["loss_tokens"] == 0 for group in group_rows)
    if protocol["train_config"]["skip_zero_adv"]:
        require(skipped == expected_skips, "optimizer skip count disagrees with zero-advantage groups")
    require(not changed or skipped != size * len(group_rows), "an all-skipped update changed the weights")
    return row["version_after"], after


def audit_arm(folder, reported, train, heldout, protocol, vocab):
    args = protocol["args"]
    changed_only = protocol.get("publication") == "changed_weights_only"
    arm = reported["arm"]
    prefix_benchmark = protocol.get("benchmark") == "prefix"
    bucket_names = ARITHMETIC_BUCKETS if prefix_benchmark else BUCKETS
    if prefix_benchmark:
        window, = args["fifo_windows"]
    elif arm == "version_aware":
        window = args["version_window"]
    else:
        window = int(arm.split("-")[1])
    groups, updates, waiting_ids = {}, [], []
    published_versions = [0]
    weight_hash = protocol.get("initial_weight_sha256")
    for row in read_rows(folder / "trace.jsonl"):
        ordinal = len(updates) + 1
        version = published_versions[-1]
        if "group_id" in row:
            gid = row["group_id"]
            require(type(gid) is int and 0 <= gid < args["groups"] and gid not in groups,
                    f"{folder.name}: invalid or duplicate group {gid}")
            try:
                check_group(row, train[gid % len(train)], protocol, vocab, version, arm, window)
            except ValueError as error:
                raise ValueError(f"{folder.name}: group {gid}: {error}") from None
            groups[gid] = row
            if row["status"] == "consumed":
                require(row["update"] == ordinal, "consumed group references the wrong update")
                waiting_ids.append(gid)
        else:
            require(row["update"] == ordinal and row["groups"] == waiting_ids
                    and len(waiting_ids) == protocol["optimizer_groups_per_update"], "wrong optimizer publication batch")
            require(max(groups[gid]["received_s"] for gid in waiting_ids) <= row["finished_s"]
                    and (not updates or updates[-1]["finished_s"] <= row["finished_s"])
                    and 0 <= row["update_s"] <= row["finished_s"]
                    and all(isfinite(value) for value in row["diagnostics"].values()), "invalid optimizer record")
            if changed_only:
                group_rows = [groups[gid] for gid in waiting_ids]
                version, weight_hash = check_publication(row, group_rows, protocol, version, weight_hash)
            else:
                version += 1
            published_versions.append(version)
            updates.append(row)
            waiting_ids.clear()
    require(not waiting_ids and len(groups) == args["groups"], f"{folder.name}: missing rollout groups or update")
    require(sum(row["status"] == "unbatched" for row in groups.values()) < protocol["optimizer_groups_per_update"],
            "more than one batch was left untrained at drain")
    groups = [groups[gid] for gid in range(len(groups))]
    ready = None
    if prefix_benchmark:
        startup = read_json(folder / "actor_ready.json")
        readiness, ready = startup["actor_ready_s"], startup["training_started_s"]
        require(set(readiness) == {str(actor_id) for actor_id in range(protocol["actor_count"])}
                and all(0 < stamp <= ready for stamp in readiness.values())
                and ready <= min(group["generated_s"] for group in groups), "invalid actor readiness timestamps")
    evals = read_rows(folder / "eval.jsonl")
    targets, snapshot_targets = check_evaluations(evals, groups, updates, published_versions, heldout, protocol,
                                                  bucket_names, ready)
    size = protocol["group_size"]
    total = size * sum(row["completion_length"] for row in groups)
    wall = evals[-1]["time_s"]
    require(evals[-1]["actor_completed_tokens"] == total and evals[-1]["update"] == len(updates) and wall > 0,
            "run ended before the full budget and all updates")
    buckets = {bucket: retention([row for row in groups if bucket == "all" or row["bucket"] == bucket], size)
               for bucket in bucket_names}
    by_length = {str(length): retention([row for row in groups if row["completion_length"] == length], size)
                 for length in sorted({row["completion_length"] for row in groups})}
    derived = {"window": window, "initial_accuracy": evals[0]["accuracy"], "final_accuracy": evals[-1]["accuracy"],
               "best_accuracy": {bucket: max(row["accuracy"][bucket] for row in evals) for bucket in bucket_names},
               "target": targets, "wall_s": wall, "generated_tokens": total, "updates": len(updates),
               "loss_tokens_s": sum(row["loss_tokens"] for row in groups) / wall}
    publications = published_versions[-1]
    if changed_only:
        require(sha256((folder / "published_weights.bin").read_bytes()).hexdigest() == weight_hash,
                f"{folder.name}: saved final policy snapshot disagrees with the weight hash chain")
        derived.update(publications=publications, no_op_updates=len(updates) - publications)
    check_reported(reported, derived, folder.name)
    for bucket, expected in reported["by_bucket"].items():
        check_reported(buckets[bucket], expected, f"{folder.name}/{bucket}")
    require(set(bucket_names) - {"all"} <= set(reported["by_bucket"]) <= set(bucket_names),
            "summary missing a retention bucket")
    signature = [(row["prompt"], row["completion_length"], row["bucket"]) for row in groups]
    result = {"seed": reported["seed"], "arm": reported["arm"], **derived, "by_bucket": buckets,
              "by_length": by_length, "publications": publications}
    if changed_only:
        full_batch_size = size * protocol["optimizer_groups_per_update"]
        result.update(distinct_snapshot_target=snapshot_targets, final_weight_sha256=weight_hash,
                      all_skipped_updates=sum(row["diagnostics"]["skipped"] == full_batch_size for row in updates))
    if prefix_benchmark:
        result.update(training_started_s=ready, training_wall_s=wall - ready)
    return result, signature


def comparison(pairs, baseline_key, treatment_key, prefix_benchmark):
    hits = [pair for pair in pairs if pair["time_saved_s"] is not None]
    metrics = ["time_saved_s", "time_saved_fraction", "actor_completed_tokens_saved", "received_tokens_saved"]
    if prefix_benchmark:
        metrics.extend(["training_time_saved_s", "training_time_saved_fraction"])
    medians = {f"median_{key}": median(row[key] for row in hits) if hits else None for key in metrics}
    result = {"pairs": pairs, "paired_hits": len(hits),
              f"{baseline_key}_hits": sum(pair[baseline_key] is not None for pair in pairs),
              f"{treatment_key}_hits": sum(pair[treatment_key] is not None for pair in pairs),
              f"{treatment_key}_faster": sum(pair["time_saved_s"] > 0 for pair in hits), **medians}
    if prefix_benchmark:
        result["shared_training_faster"] = sum(pair["training_time_saved_s"] > 0 for pair in hits)
    return result


def audit(run):
    protocol = read_json(run / "protocol.json")
    summary = read_json(run / "summary.json")
    require(protocol == summary["protocol"], "run and summary protocols differ")
    args, config = protocol["args"], protocol["train_config"]
    prefix_benchmark = protocol.get("benchmark") == "prefix"
    require(not prefix_benchmark or (protocol.get("task") == "arithmetic" and len(args["fifo_windows"]) == 1),
            "expected a matched two-arm arithmetic prefix benchmark")
    treatment_arm = "share_prompt" if prefix_benchmark else "version_aware"
    baseline_key, treatment_key = ("baseline", "shared") if prefix_benchmark else ("fifo", "version_aware")
    target_bucket = "all" if prefix_benchmark else "long"
    bucket_names = ARITHMETIC_BUCKETS if prefix_benchmark else BUCKETS
    publication = protocol.get("publication")
    require(publication in (None, "changed_weights_only"), "unknown policy publication mode")
    if publication == "changed_weights_only":
        require(re.fullmatch(r"[0-9a-f]{64}", protocol["initial_weight_sha256"]) is not None,
                "invalid initial flat-weight SHA256")
    require(protocol["optimizer_groups_per_update"] == config["n_prompts"] == 2
            and protocol["group_size"] == config["group_size"] == config["micro_batch"]
            and protocol["max_lag"] == config["max_staleness"], "learner protocol disagrees with training config")
    require(sha256((run / "model/model.safetensors").read_bytes()).hexdigest() == protocol["initial_model_sha256"],
            "initial model artifact differs from protocol")
    vocab = read_json(run / "model/tokenizer.json")["model"]["vocab"]
    heldout = read_json(run / "heldout.json")
    development = protocol["stage"] == "development"
    if prefix_benchmark:
        require(protocol["action_labels"] == list("0123456789")
                and [vocab[digit] for digit in protocol["action_labels"]] == list(range(3, 13)),
                "arithmetic tokenizer does not match the fixed decimal action support")
        train_pool, expected_heldout = arithmetic_workloads(config["n_examples"], config["eval_n"], development)
    else:
        train_pool = workload(config["n_examples"], 0, ("one", "two"))
        expected_heldout = workload(config["eval_n"], 1024 if development else 2048,
                                    ("three",) if development else ("four",))
    require(heldout == expected_heldout, "held-out prompts differ from the frozen split")
    arms = ["recompute"] if prefix_benchmark else [f"fifo-{window}" for window in args["fifo_windows"]]
    expected_arms = {(seed, arm) for seed in args["seeds"] for arm in [*arms, treatment_arm]}
    reports = {(row["seed"], row["arm"]): row for row in summary["runs"]}
    require(len(reports) == len(summary["runs"]) and set(reports) == expected_arms, "missing or duplicate run arms")
    verified, paired = [], {arm: [] for arm in arms}
    for seed in args["seeds"]:
        train = train_pool.copy()
        Random(seed).shuffle(train)
        treatment, expected = audit_arm(run / f"{seed}-{treatment_arm}", reports[seed, treatment_arm],
                                        train, heldout, protocol, vocab)
        verified.append(treatment)
        for arm in arms:
            control, signature = audit_arm(run / f"{seed}-{arm}", reports[seed, arm], train, heldout, protocol, vocab)
            require(signature == expected and control["generated_tokens"] == treatment["generated_tokens"]
                    and control["initial_accuracy"] == treatment["initial_accuracy"], f"seed {seed}: unmatched workload")
            verified.append(control)
            a, b = control["target"][target_bucket], treatment["target"][target_bucket]
            pair = {"seed": seed, baseline_key: a, treatment_key: b,
                    "time_saved_s": a["time_s"] - b["time_s"] if a and b else None,
                    "time_saved_fraction": 1 - b["time_s"] / a["time_s"] if a and b else None,
                    "actor_completed_tokens_saved": a["actor_completed_tokens"] - b["actor_completed_tokens"]
                    if a and b else None,
                    "received_tokens_saved": a["received_tokens"] - b["received_tokens"] if a and b else None,
                    "final_accuracy": {baseline_key: control["final_accuracy"],
                                       treatment_key: treatment["final_accuracy"]},
                    "retained_group_fraction_delta": {bucket: treatment["by_bucket"][bucket]["retained_group_fraction"]
                                                      - control["by_bucket"][bucket]["retained_group_fraction"]
                                                      for bucket in bucket_names}}
            if prefix_benchmark:
                pair.update(training_time_saved_s=a["training_time_s"] - b["training_time_s"] if a and b else None,
                            training_time_saved_fraction=1 - b["training_time_s"] / a["training_time_s"]
                            if a and b else None)
            paired[arm].append(pair)
    comparisons = {arm: comparison(rows, baseline_key, treatment_key, prefix_benchmark) for arm, rows in paired.items()}
    require(set(summary["comparisons"]) == set(arms), "summary comparisons do not match protocol")
    for arm, original in summary["comparisons"].items():
        check_reported(comparisons[arm], {key: value for key, value in original.items() if key != "pairs"}, arm)
        require(len(original["pairs"]) == len(comparisons[arm]["pairs"]), f"{arm}: wrong pair count")
        for pair, expected in zip(comparisons[arm]["pairs"], original["pairs"]):
            check_reported(pair, expected, f"{arm}/seed {pair['seed']}")
    return {"source_run": str(run), "protocol": protocol, "runs": verified, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Audited {len(result['runs'])} arms. Saved {args.out}")


if __name__ == "__main__":
    main()
