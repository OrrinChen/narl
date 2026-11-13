"""Validate online GRPO traces and summarize each matched CPU comparison."""

import argparse
from collections import Counter
import json
from math import isclose, isfinite
from pathlib import Path
import random
from statistics import median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def describe(values):
    return {"count": len(values), "median": median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def nonnegative(value):
    return isfinite(value) and value >= 0


def check_reported(reported, derived, label):
    for key, value in derived.items():
        actual = reported[key]
        matches = isclose(actual, value, rel_tol=1e-10, abs_tol=1e-12) if isinstance(value, float) else actual == value
        require(matches, f"{label}: summary {key}={actual!r}, trace derives {value!r}")


def expected_prompts(protocol, seed):
    """Reconstruct TokenChoiceTask's fixed split and seeded ring walk."""
    task = protocol["train_config"]
    queries = ("red", "blue", "green", "gold")
    rows = [(f"{' '.join(format(index // 4, 'b'))} {queries[index % 4]}", index % 4)
            for index in range(task["eval_n"], task["n_examples"])]
    random.Random(seed).shuffle(rows)
    return [rows[index % len(rows)] for index in range(protocol["groups"])]


def check_group(group, index, expected_problem, protocol, mode, vocab, learner_version):
    prompt_text, answer_index = expected_problem
    size = protocol["train_config"]["group_size"]
    prompt_length = protocol["prompt_tokens"]
    lengths = protocol["completion_lengths"]
    length = lengths[index % len(lengths)]
    first_content = protocol["first_content_token"]
    label = f"group {index}"
    require(group["group_id"] == index and group["prompt"] == prompt_text and group["completion_length"] == length,
            f"{label}: unexpected ID, prompt, or completion budget")

    prompt_ids = [vocab[token] for token in prompt_text.split()]
    expected_prefix = ([vocab["0"], vocab["1"]] * ((prompt_length - len(prompt_ids) + 1) // 2)
                       + prompt_ids)[-prompt_length:]
    tokens, logprobs, rewards = (group[key] for key in ("tokens", "behavior_logprobs", "rewards"))
    require(len(tokens) == len(logprobs) == len(rewards) == size, f"{label}: group is incomplete")
    answer_id = vocab[("one", "two", "three", "four")[answer_index]]
    reward_position = {"first": prompt_length, "last": prompt_length + length - 1}[
        protocol.get("reward_position", "first")]
    for ids, logp, reward in zip(tokens, logprobs, rewards):
        require(len(ids) == prompt_length + length and ids[:prompt_length] == expected_prefix,
                f"{label}: missing prompt or completion tokens")
        require(all(type(token) is int and first_content <= token < len(vocab) for token in ids[prompt_length:]),
                f"{label}: unexpected EOS, padding or token outside the sampling support")
        require(len(logp) == length and all(isfinite(value) and value <= 0 for value in logp),
                f"{label}: invalid behavior logprobs")
        require(reward == float(ids[reward_position] == answer_id), f"{label}: reward does not match scoring token")

    require(group["learner_version"] == learner_version, f"{label}: learner version does not match consumed updates")
    initial = group["initial_version"]
    require(type(initial) is int and 0 <= initial <= learner_version, f"{label}: invalid initial policy version")
    expected_versions = [initial] * length
    switches = group["switches"]
    require(len(switches) <= protocol["max_swaps_per_group"], f"{label}: excessive weight switches")
    for switch in switches:
        boundary = switch["after_tokens"]
        require(type(boundary) is int and 0 < boundary < length and boundary % protocol["weight_check_every_tokens"] == 0,
                f"{label}: invalid switch boundary")
        new = switch["to"]
        require(switch["from"] == initial and type(new) is int and initial < new <= learner_version,
                f"{label}: switch versions out of order")
        # The pending boundary token is sampled before the switch and excluded from rebuild.
        expected_versions[boundary:] = [new] * (length - boundary)
        expected_prompt_tokens = prompt_length * (size if mode == "recompute" else 1)
        require(switch["prompt_tokens_recomputed"] == expected_prompt_tokens
                and switch["suffix_tokens_recomputed"] == size * (boundary - 1),
                f"{label}: KV rebuild token counts do not match the switch boundary")
        compute, copy, recovery = (switch[key] for key in ("recompute_ms", "cache_copy_ms", "recovery_ms"))
        require(nonnegative(compute) and nonnegative(copy) and nonnegative(recovery)
                and compute + copy <= recovery + 1e-6, f"{label}: invalid switch timings")
        require(mode != "recompute" or copy == 0, f"{label}: independent rebuild unexpectedly copied cache")
    require(group["version_by_token"] == expected_versions, f"{label}: behavior versions do not match token spans")

    lag = learner_version - initial
    accepted = lag <= protocol["max_lag"]
    require(group["lag"] == lag and group["status"] == ("consumed" if accepted else "stale"),
            f"{label}: incorrect staleness decision")
    signal = accepted and len(set(rewards)) > 1
    expected_loss_tokens = size * length if signal else 0
    require(group["loss_tokens"] == expected_loss_tokens, f"{label}: gradient-token count disagrees with reward signal")
    for key in ("weight_sync_ms", "generation_s", "queue_s_including_backpressure", "publish_ms", "update_ms"):
        require(nonnegative(group[key]), f"{label}: invalid {key}")
    require(group["generation_s"] > 0, f"{label}: missing generation time")
    if accepted:
        require(group["update"] == learner_version + 1 and group["update_ms"] > 0 and group["publish_ms"] > 0,
                f"{label}: missing consumed update or publication")
        require(group["diagnostics"] is not None and all(isfinite(value) for value in group["diagnostics"].values()),
                f"{label}: invalid optimizer diagnostics")
    else:
        require(group["update"] is None and group["diagnostics"] is None
                and group["update_ms"] == group["publish_ms"] == 0,
                f"{label}: stale group changed the learner")


def distribution(groups):
    per_completion = [reward for group in groups for reward in group["rewards"]]
    per_group = [sum(group["rewards"]) / len(group["rewards"]) for group in groups]
    return {
        "groups": len(groups),
        "completion_length_histogram_groups": dict(sorted(Counter(group["completion_length"] for group in groups).items())),
        "completion_reward_histogram": dict(sorted(Counter(per_completion).items())),
        "mean_reward": sum(per_completion) / len(per_completion) if per_completion else None,
        "group_reward": describe(per_group),
        "lag_histogram_groups": dict(sorted(Counter(group["lag"] for group in groups).items())),
    }


def check_evaluation(rows, consumed, protocol, label):
    updates = len(consumed)
    default_interval = 12 if protocol.get("reward_position") == "last" else 8
    eval_interval = protocol.get("eval_every_updates", default_interval)
    expected_schedule = [0, *range(eval_interval, updates + 1, eval_interval), updates]
    require([row["update"] for row in rows] == expected_schedule, f"{label}: incorrect evaluation schedule")
    tokens_by_update = [0]
    for group in consumed:
        tokens_by_update.append(tokens_by_update[-1] + group["loss_tokens"])
    prev_time = 0
    size = protocol["train_config"]["eval_n"]
    for row in rows:
        reward = row["reward"]
        require(prev_time <= row["time_s"] and isfinite(row["time_s"]), f"{label}: invalid evaluation time")
        require(row["loss_tokens"] == tokens_by_update[row["update"]], f"{label}: evaluation training budget mismatch")
        require(0 <= reward <= 1 and isclose(reward * size, round(reward * size), abs_tol=1e-7),
                f"{label}: evaluation reward incompatible with the held-out size")
        prev_time = row["time_s"]
    require(rows[0]["time_s"] == 0 and rows[-1]["time_s"] > 0, f"{label}: missing evaluation endpoints")
    if len(rows) > 2 and rows[-2]["update"] == updates:
        require(rows[-2]["reward"] == rows[-1]["reward"], f"{label}: repeated final evaluation changed")


def audit_arm(run, protocol, vocab, reported):
    seed, mode = reported["seed"], reported["mode"]
    label = f"{seed}-{mode}"
    path = run / label
    groups = read_rows(path / "groups.jsonl")
    evaluations = read_rows(path / "eval.jsonl")
    producer_put_ms = read_json(path / "actor_queue.json")
    require(len(groups) == len(producer_put_ms) == protocol["groups"], f"{label}: missing generation or queue records")
    require(all(nonnegative(value) for value in producer_put_ms), f"{label}: invalid producer put timings")
    consumed, stale = [], []
    latest_initial = -1
    for index, (group, problem) in enumerate(zip(groups, expected_prompts(protocol, seed))):
        try:
            check_group(group, index, problem, protocol, mode, vocab, len(consumed))
            require(group["initial_version"] >= latest_initial, f"group {index}: actor moved backwards in weight versions")
            latest_initial = group["switches"][-1]["to"] if group["switches"] else group["initial_version"]
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from None
        (consumed if group["status"] == "consumed" else stale).append(group)

    check_evaluation(evaluations, consumed, protocol, label)
    wall_s = evaluations[-1]["time_s"]
    size = protocol["train_config"]["group_size"]
    generated = sum(group["completion_length"] for group in groups) * size
    accepted = sum(group["completion_length"] for group in consumed) * size
    loss_tokens = sum(group["loss_tokens"] for group in consumed)
    switches = [switch for group in groups for switch in group["switches"]]
    consumed_switches = sum(len(group["switches"]) for group in consumed)
    require(switches and loss_tokens > 0 and any(group["diagnostics"]["gnorm"] > 0 for group in consumed),
            f"{label}: expected live weight switch and nonzero training gradient")

    derived = {
        "wall_s": wall_s, "generated_tokens": generated, "generated_tokens_s": generated / wall_s,
        "accepted_tokens": accepted, "loss_tokens": loss_tokens, "loss_tokens_s": loss_tokens / wall_s,
        "consumed_groups": len(consumed), "consumed_groups_s": len(consumed) / wall_s,
        "dropped_stale_groups": len(stale), "dropped_overflow_groups": 0,
        "zero_signal_groups": sum(group["loss_tokens"] == 0 for group in consumed),
        "lag_histogram_consumed": dict(Counter(str(group["lag"]) for group in consumed)),
        "weight_switches_in_flight": len(switches), "initial_reward": evaluations[0]["reward"],
        "final_reward": evaluations[-1]["reward"], "queue_put_median_ms": median(producer_put_ms),
        "queue_put_max_ms": max(producer_put_ms),
    }
    check_reported(reported, derived, label)
    phases = {
        "actor_generate": [1_000 * group["generation_s"] for group in groups],
        "actor_pull_weights": [group["weight_sync_ms"] for group in groups],
        "producer_queue_put": producer_put_ms,
        "queue_to_learner_including_backpressure": [1_000 * group["queue_s_including_backpressure"] for group in groups],
        "learner_update_consumed": [group["update_ms"] for group in consumed],
        "learner_publish_consumed": [group["publish_ms"] for group in consumed],
        "switch_recompute": [switch["recompute_ms"] for switch in switches],
        "switch_cache_copy": [switch["cache_copy_ms"] for switch in switches],
        "switch_recovery": [switch["recovery_ms"] for switch in switches],
    }
    arm = {
        **derived, "accepted_tokens_s": accepted / wall_s,
        "consumed_in_flight_switches": consumed_switches, "stale_group_fraction": len(stale) / len(groups),
        "queue_put_over_1ms_count": sum(duration > 1 for duration in producer_put_ms),
        "reward_change": evaluations[-1]["reward"] - evaluations[0]["reward"],
        "phase_ms": {key: describe(values) for key, values in phases.items()},
        "consumed": distribution(consumed), "dropped_stale": distribution(stale),
        "evaluation": evaluations,
    }
    workload = [(group["prompt"], group["tokens"][0][:protocol["prompt_tokens"]], group["completion_length"])
                for group in groups]
    return arm, workload


def audit(run):
    protocol = read_json(run / "protocol.json")
    summary = read_json(run / "summary.json")
    require(summary["protocol"] == protocol, "Run protocol differs from summary protocol")
    seeds, modes = protocol["seeds"], protocol["comparison"]
    require(seeds and len(set(seeds)) == len(seeds), "Expected distinct sampling seeds")
    require(set(modes) == {"recompute", "share_prompt"} and len(modes) == 2, "Unexpected mode pair")
    require(protocol["fixed_horizon"] and protocol["max_swaps_per_group"] == 1
            and protocol["publish_every_accepted_groups"] == 1 and protocol["queue_capacity"] > 0,
            "Unexpected workload, publication or queue contract")
    require(protocol.get("reward_position", "first") in {"first", "last"}, "Unexpected reward token position")
    reported = {(row["seed"], row["mode"]): row for row in summary["runs"]}
    require(len(summary["runs"]) == len(reported) and set(reported) == {(seed, mode) for seed in seeds for mode in modes},
            "Missing or duplicated seed/mode runs")
    vocab = read_json(run / "model/tokenizer.json")["model"]["vocab"]
    comparisons = []
    for seed in seeds:
        baseline, workload = audit_arm(run, protocol, vocab, reported[seed, "recompute"])
        shared, shared_workload = audit_arm(run, protocol, vocab, reported[seed, "share_prompt"])
        require(workload == shared_workload and baseline["generated_tokens"] == shared["generated_tokens"],
                f"seed {seed}: treatment did not match prompt stream and generated budget")
        require(baseline["initial_reward"] == shared["initial_reward"], f"seed {seed}: evaluation baselines differ")
        baseline_recovery = baseline["phase_ms"]["switch_recovery"]["median"]
        shared_recovery = shared["phase_ms"]["switch_recovery"]["median"]
        differences = {
            "switch_recovery_median_ms": shared_recovery - baseline_recovery,
            "wall_s": shared["wall_s"] - baseline["wall_s"],
            "generated_tokens_s": shared["generated_tokens_s"] - baseline["generated_tokens_s"],
            "accepted_tokens_s": shared["accepted_tokens_s"] - baseline["accepted_tokens_s"],
            "loss_tokens_s": shared["loss_tokens_s"] - baseline["loss_tokens_s"],
            "consumed_groups_s": shared["consumed_groups_s"] - baseline["consumed_groups_s"],
            "dropped_stale_groups": shared["dropped_stale_groups"] - baseline["dropped_stale_groups"],
            "final_reward": shared["final_reward"] - baseline["final_reward"],
        }
        comparisons.append({"seed": seed, "recompute": baseline, "share_prompt": shared,
                            "difference_shared_minus_recompute": differences})
    return {
        "source_run": str(run), "protocol": protocol, "paired_seeds": len(comparisons),
        "comparison_timing": "Per-arm phase medians use the groups observed in that arm. Wall time includes actor startup and evaluation.",
        "queue_measurement": "Queue capacity comes from the protocol. The queue trace records producer put latency, not actual queue depth.",
        "pairs": comparisons,
        "paired_differences_across_seeds": {
            key: describe([pair["difference_shared_minus_recompute"][key] for pair in comparisons])
            for key in comparisons[0]["difference_shared_minus_recompute"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output JSON path")
    args = parser.parse_args()
    result = audit(args.run.resolve())
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Audited {result['paired_seeds']} seed pairs. Saved {args.out}")


if __name__ == "__main__":
    main()
