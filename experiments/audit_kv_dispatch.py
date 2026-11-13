"""Reconstruct dispatch metrics from traces and compare matched seeds."""

import argparse
from collections import Counter
import json
from math import ceil, isclose, isfinite
from pathlib import Path

from experiments.audit_kv_online import (
    check_reported, describe, expected_prompts, read_json, read_rows, require,
)


def check_work(group, problem, protocol, vocab, learner_version):
    group_id = group["group_id"]
    length = protocol["completion_lengths"][group_id % len(protocol["completion_lengths"])]
    size, prompt_length = protocol["group_size"], protocol["prompt_tokens"]
    prompt, answer_index = problem
    ids = [vocab[token] for token in prompt.split()]
    prefix = ([vocab["0"], vocab["1"]] * ((prompt_length - len(ids) + 1) // 2) + ids)[-prompt_length:]
    answer = vocab[("one", "two", "three", "four")[answer_index]]
    require(group["prompt"] == prompt and group["completion_length"] == length,
            "unexpected prompt or completion budget")
    require(len(group["tokens"]) == len(group["behavior_logprobs"]) == len(group["rewards"]) == size,
            "incomplete prompt group")
    for sequence, logprobs, reward in zip(group["tokens"], group["behavior_logprobs"], group["rewards"]):
        require(len(sequence) == prompt_length + length and sequence[:prompt_length] == prefix,
                "unexpected token history or incomplete completion")
        require(all(type(token) is int and vocab["0"] <= token < len(vocab) for token in sequence[prompt_length:]),
                "generated token outside the fixed-horizon sampling support")
        require(len(logprobs) == length and all(isfinite(value) and value <= 0 for value in logprobs),
                "missing or invalid behavior logprobs")
        require(reward == float(sequence[-1] == answer), "reward disagrees with final generated token")

    reserved, initial = group["reserved_version"], group["initial_version"]
    require(type(reserved) is type(initial) is int and 0 <= reserved <= initial <= learner_version,
            "invalid reserved, initial or learner version order")
    require(group["learner_version"] == learner_version, "learner did not publish exactly once per consumed group")
    versions = [initial] * length
    switches = group["switches"]
    require(len(switches) <= 1, "multiple weight switches in one group")
    for switch in switches:
        boundary, new = switch["after_tokens"], switch["to"]
        require(type(boundary) is int and 0 < boundary < length and boundary % 4 == 0,
                "invalid switch token boundary")
        require(switch["from"] == initial and type(new) is int and initial < new <= learner_version,
                "invalid weight switch versions")
        versions[boundary:] = [new] * (length - boundary)
        require(switch["prompt_tokens_recomputed"] == prompt_length
                and switch["suffix_tokens_recomputed"] == size * (boundary - 1),
                "KV rebuild token counts do not match one shared prompt and all cached suffixes")
        compute, copy, recovery = (switch[key] for key in ("recompute_ms", "cache_copy_ms", "recovery_ms"))
        require(all(isfinite(value) and value >= 0 for value in (compute, copy, recovery))
                and compute + copy <= recovery + 1e-6, "invalid KV rebuild timing")
    require(group["version_by_token"] == versions, "behavior versions disagree with expected spans")
    require(group["initial_prompt_tokens_computed"] == prompt_length, "initial prompt was not shared")

    lag = learner_version - initial
    accepted = lag <= protocol["max_lag"]
    require(group["lag"] == lag and group["status"] == ("consumed" if accepted else "stale"),
            "incorrect lag or admission decision")
    loss_tokens = size * length if accepted and len(set(group["rewards"])) > 1 else 0
    require(group["loss_tokens"] == loss_tokens, "loss token count disagrees with group signal")
    diagnostics = group["diagnostics"]
    if accepted:
        require(diagnostics is not None and all(isfinite(value) for value in diagnostics.values()),
                "consumed group lacks finite optimizer diagnostics")
    else:
        require(diagnostics is None, "rejected group changed the learner")


def check_concurrency(by_id, protocol, arm, window, label):
    # Dispatch is stamped after reserve. Admission is stamped before release.
    # Their interval therefore bounds observed outstanding work from below.
    previous_slot, previous_actor = {}, {}
    events = []
    for row in by_id:
        slot, actor = row["slot"], row["actor_id"]
        dispatched, generated = row["dispatched_at"], row["generated_at"]
        admitted = generated + row["queue_s"]
        if slot in previous_slot:
            prior = previous_slot[slot]
            prior_admitted = prior["generated_at"] + prior["queue_s"]
            published = prior["learner_version"] + int(prior["status"] == "consumed")
            require(dispatched >= prior_admitted and row["reserved_version"] >= published,
                    f"{label}: slot {slot} was reused before prior admission/publication")
        if actor in previous_actor:
            require(dispatched >= previous_actor[actor], f"{label}: actor {actor} generation intervals overlap")
        previous_slot[slot] = row
        previous_actor[actor] = generated
        if admitted > dispatched:
            events.extend(((dispatched, 1), (admitted, -1)))
    outstanding = peak = 0
    for _, delta in sorted(events):
        outstanding += delta
        peak = max(peak, outstanding)
    cap = min(window, protocol["max_lag"] + 1) if arm == "version_aware" else window
    require(outstanding == 0 and peak <= cap, f"{label}: observed outstanding work exceeds reservation bound")
    return peak


def audit_arm(run, protocol, vocab, reported):
    seed, arm, window = reported["seed"], reported["arm"], reported["window"]
    label = f"{seed}-{arm}"
    rows = read_rows(run / label / "groups.jsonl")
    size = protocol["group_size"]
    group_ids = [row["group_id"] for row in rows]
    require(all(type(group_id) is int for group_id in group_ids)
            and sorted(group_ids) == list(range(protocol["groups"])), f"{label}: missing or duplicate group IDs")
    problems = expected_prompts(protocol, seed)
    expected_window = protocol["version_aware_window"] if arm == "version_aware" else int(arm[5:])
    require(window == expected_window, f"{label}: reservation window mismatch")
    wall = reported["wall_s"]
    require(isfinite(wall) and wall > 0, f"{label}: invalid wall clock")
    require(protocol["train_config"]["group_size"] == size, f"{label}: group size metadata mismatch")
    consumed, stale = [], []
    for row in rows:
        group_id = row["group_id"]
        try:
            check_work(row, problems[group_id], protocol, vocab, len(consumed))
            require(0 <= row["actor_id"] < protocol["actors"] and 0 <= row["slot"] < window,
                    "invalid actor or reservation slot")
            require(all(isfinite(row[key]) and 0 <= row[key] <= wall for key in (
                "generation_s", "queue_s", "dispatch_to_finish_s", "idle_s")), "invalid duration")
            require(isclose(row["generation_s"], row["generated_at"] - row["dispatched_at"], abs_tol=1e-8)
                    and row["generation_s"] + row["queue_s"] <= row["dispatch_to_finish_s"] + 1e-8,
                    "inconsistent dispatch, generation or queue timestamps")
        except ValueError as error:
            raise ValueError(f"{label}: group {group_id}: {error}") from None
        (consumed if row["status"] == "consumed" else stale).append(row)
    require(consumed, f"{label}: no learner updates")
    if arm == "version_aware":
        require(not stale and all(row["learner_version"] - row["reserved_version"] <= protocol["max_lag"]
                                  for row in rows), f"{label}: reservation expired despite version-aware dispatch")
    by_id = sorted(rows, key=lambda row: row["group_id"])
    reservations = [row["reserved_version"] for row in by_id]
    require(reservations == sorted(reservations), f"{label}: reservation versions are not monotonic by dispatch ID")
    peak_concurrency = check_concurrency(by_id, protocol, arm, window, label)

    generated = sum(row["completion_length"] for row in rows) * size
    stale_tokens = sum(row["completion_length"] for row in stale) * size
    accepted = generated - stale_tokens
    loss_tokens = sum(row["loss_tokens"] for row in consumed)
    latencies = sorted(row["dispatch_to_finish_s"] for row in consumed)
    buckets = {}
    for length in protocol["completion_lengths"]:
        length_rows = [row for row in rows if row["completion_length"] == length]
        stale_groups = sum(row["status"] == "stale" for row in length_rows)
        buckets[str(length)] = {"groups": len(length_rows), "stale_groups": stale_groups,
                                "stale_tokens": stale_groups * length * size}
    derived = {
        "generated_tokens": generated, "generated_tokens_s": generated / wall,
        "consumed_groups": len(consumed), "consumed_groups_s": len(consumed) / wall,
        "accepted_tokens_s": accepted / wall, "loss_tokens": loss_tokens,
        "loss_tokens_s": loss_tokens / wall, "stale_groups": len(stale), "stale_tokens": stale_tokens,
        "p95_consumed_group_s": latencies[ceil(0.95 * len(latencies)) - 1],
        "lag_histogram_consumed": dict(Counter(str(row["lag"]) for row in consumed)), "buckets": buckets,
    }
    check_reported(reported, derived, label)

    idle = sum(row["idle_s"] for row in rows)
    total_idle = reported["actor_idle_s"]
    terminal = reported.get("terminal_idle_s_by_actor")
    if terminal is not None:
        require(set(terminal) == {str(i) for i in range(protocol["actors"])}
                and all(isfinite(value) and 0 <= value <= wall for value in terminal.values())
                and isclose(idle + sum(terminal.values()), total_idle, abs_tol=1e-8),
                f"{label}: actor idle aggregate does not match raw and terminal records")
    else:
        require(isfinite(total_idle) and idle <= total_idle <= protocol["actors"] * wall,
                f"{label}: actor idle summary disagrees with raw lower bound")

    switches = [switch for row in rows for switch in row["switches"]]
    prompt_recomputed = sum(switch["prompt_tokens_recomputed"] for switch in switches)
    suffix_recomputed = sum(switch["suffix_tokens_recomputed"] for switch in switches)
    for key in ("initial_reward", "final_reward"):
        reward = reported[key]
        eval_size = protocol["train_config"]["eval_n"]
        require(0 <= reward <= 1 and isclose(reward * eval_size, round(reward * eval_size), abs_tol=1e-7),
                f"{label}: invalid {key}")
    result = {
        **derived, "wall_s": wall, "window": window, "accepted_tokens": accepted,
        "actor_idle_s": total_idle, "actor_nonterminal_idle_s": idle, "actor_terminal_idle_audited": terminal is not None,
        "initial_reward": reported["initial_reward"], "final_reward": reported["final_reward"],
        "weight_switches": len(switches), "switch_recovery_ms": describe([switch["recovery_ms"] for switch in switches]),
        "initial_prompt_tokens_computed": sum(row["initial_prompt_tokens_computed"] for row in rows),
        "prompt_tokens_recomputed": prompt_recomputed, "suffix_tokens_recomputed": suffix_recomputed,
        "recomputed_tokens_s": (prompt_recomputed + suffix_recomputed) / wall,
        "peak_observed_outstanding_lower_bound": peak_concurrency,
    }
    workload = [(row["prompt"], row["tokens"][0][:protocol["prompt_tokens"]], row["completion_length"])
                for row in by_id]
    return result, workload


def audit(run, baselines):
    protocol = read_json(run / "protocol.json")
    summary = read_json(run / "summary.json")
    require(protocol == summary["protocol"], "Run and summary protocols differ")
    require(protocol["max_lag"] == protocol["train_config"]["max_staleness"] == 1,
            "Expected an explicit one-version lag limit")
    arms = {f"fifo-{window}" for window in protocol["fifo_windows"]} | {"version_aware"}
    require(len(set(baselines)) == len(baselines) and set(baselines) <= arms - {"version_aware"},
            "Baseline must identify a FIFO arm in this run")
    require(protocol["seeds"] and len(set(protocol["seeds"])) == len(protocol["seeds"]), "Missing or duplicate seeds")
    reported = {(row["seed"], row["arm"]): row for row in summary["runs"]}
    require(len(reported) == len(summary["runs"]) and set(reported) == {
        (seed, arm) for seed in protocol["seeds"] for arm in arms}, "Missing or duplicate seed/arm results")
    vocab = read_json(run / "model/tokenizer.json")["model"]["vocab"]
    pairs = {baseline: [] for baseline in baselines}
    metrics = ("stale_groups", "stale_tokens", "generated_tokens_s", "accepted_tokens_s", "loss_tokens_s",
               "consumed_groups_s", "p95_consumed_group_s", "wall_s", "actor_idle_s", "final_reward",
               "prompt_tokens_recomputed", "suffix_tokens_recomputed", "recomputed_tokens_s")
    ratio_metrics = ("loss_tokens_s", "accepted_tokens_s", "p95_consumed_group_s")
    for seed in protocol["seeds"]:
        treatment, workload = audit_arm(run, protocol, vocab, reported[seed, "version_aware"])
        for arm in sorted(arms - {"version_aware"}):
            control, control_workload = audit_arm(run, protocol, vocab, reported[seed, arm])
            require(workload == control_workload and treatment["generated_tokens"] == control["generated_tokens"],
                    f"seed {seed}: {arm} did not match version-aware workload by dispatch ID")
            require(treatment["initial_reward"] == control["initial_reward"],
                    f"seed {seed}: {arm} initial evaluation differs")
            if arm in pairs:
                difference = {key: treatment[key] - control[key] for key in metrics}
                relative_change = {key: treatment[key] / control[key] - 1 if control[key] else None
                                   for key in ratio_metrics}
                pairs[arm].append({"seed": seed, "baseline": control, "version_aware": treatment,
                                   "difference_version_aware_minus_baseline": difference,
                                   "relative_change_version_aware_over_baseline_minus_one": relative_change})
    comparisons = {}
    for arm, rows in pairs.items():
        differences = [row["difference_version_aware_minus_baseline"] for row in rows]
        relative_changes = [row["relative_change_version_aware_over_baseline_minus_one"] for row in rows]
        comparisons[arm] = {
            "pairs": rows,
            "paired_differences_across_seeds": {key: describe([row[key] for row in differences]) for key in metrics},
            "paired_relative_changes_across_seeds": {
                key: describe([row[key] for row in relative_changes if row[key] is not None]) for key in ratio_metrics
            },
        }
    return {
        "source_run": str(run), "protocol": protocol, "paired_seeds": len(protocol["seeds"]),
        "actor_idle_definition": "Sum of actor time waiting for reservations, including the final acquire call. It excludes startup and queue puts.",
        "latency_definition": "Nearest-rank p95 of consumed groups from dispatch to completed learner handling. The consumed populations can differ by arm.",
        "concurrency_definition": "Dispatch timestamps follow reserve and admission timestamps precede release. Their overlap is a lower bound on outstanding reservations.",
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", action="append", required=True, help="FIFO arm, repeat to compare another arm")
    args = parser.parse_args()
    result = audit(args.run.resolve(), args.baseline)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Audited {result['paired_seeds']} seed pairs across {len(args.baseline)} comparisons. Saved {args.out}")


if __name__ == "__main__":
    main()
