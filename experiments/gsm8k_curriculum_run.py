"""Train and strictly evaluate frozen GSM8K curriculum comparisons on the training split."""

import argparse
import fcntl
import json
from math import isclose
from importlib.metadata import version
from pathlib import Path

from experiments.audit_gsm8k_stable import check_eval
from experiments.audit_kv_online import read_json, require
from experiments.gsm8k_reward_study import digest, record


SHORT, LONG = 32_768, 131_072
DEFAULT_OUT = Path("runs/curriculum-gsm8k-v1")


def check_request(args, protocol, protocol_hash):
    source = protocol["source_protocol"]
    require(protocol["seeds"] == list(range(5)) and protocol["budgets"] == [SHORT, LONG],
            "unexpected seeds or token budgets")
    require((protocol["group_size"], protocol["max_new_tokens"], source["eval_batch"],
             source["eval_max_new_tokens"]) == (source["group_size"], source["max_new_tokens"], 12, 256)
            and (source["group_size"], source["max_new_tokens"]) == (2, 256), "generation settings changed")
    require(source["start_sha256"] == protocol.get("start_sha256", source["start_sha256"]),
            "conflicting common start")
    require(source["train_sha256"] == protocol["train_sha256"], "conflicting training data")
    for name in ("gsm8k_training_reward.py", "gsm8k_training_reward_v2.py"):
        require(digest(Path(__file__).with_name(name)) == protocol["reward_source_sha256"][name],
                f"frozen reward source changed: {name}")
    primary, dose = (protocol["eval_sets"][panel] for panel in ("primary", "dose"))
    require(len(set(primary) | set(dose)) == 512 and len(primary) == len(dose) == 256,
            "expected two disjoint 256-question panels")
    if args.stage == "initial":
        require(args.arm is None and args.seed is None and args.budget is None, "common initial takes only a phase")
    else:
        require(args.arm in ("uniform", "mixed") and args.seed in protocol["seeds"] and args.budget in protocol["budgets"],
                "train/eval require an arm, seed and budget")
    require(not (args.stage == "train" and args.phase == "dose" and args.budget == SHORT),
            "short training already belongs to the primary phase")
    require(args.budget != LONG or args.arm == "mixed", "only the mixed curriculum continues to 131072")
    require(not (args.stage == "eval" and args.budget == LONG and args.phase != "dose"), "long endpoint uses dose panel")
    require(args.phase != "dose" or args.arm in (None, "mixed"), "dose panel is reserved for the mixed curriculum")
    if args.phase == "dose" or args.budget == LONG:
        trigger_path = args.out / "trigger.json"
        require(trigger_path.is_file(), "dose stage requires the frozen primary decision")
        from experiments.gsm8k_curriculum_results import trigger_for

        path = args.out / "results-primary.json"
        primary = read_json(path)
        trigger = trigger_for(primary, path)
        require(primary["phase"] == "primary" and trigger["dose_authorized"] is True
                and trigger["protocol_sha256"] == protocol_hash and trigger == read_json(trigger_path),
                "dose stage is closed or belongs to another primary decision")


def training_folder(args, budget):
    return args.out / "models" / args.arm / str(budget) / f"seed-{args.seed}"


def binding_for(args, protocol, protocol_hash, budget):
    return {"protocol_sha256": protocol_hash, "arm": args.arm, "seed": args.seed, "budget": budget,
            "start_sha256": protocol["source_protocol"]["start_sha256"]}


def check_training(folder, binding, protocol):
    from experiments.audit_gsm8k_online_lag import adapter_digest

    require(read_json(folder / "binding.json") == binding, "training binding changed")
    report = read_json(folder / "training.json")
    require(report["start_sha256"] == binding["start_sha256"] and report["seed"] == binding["seed"]
            and report["lr"] == protocol["lr"] and report["accumulation_groups"] == protocol["accumulation_groups"]
            and report["accounting"]["generated_tokens"] == binding["budget"]
            and report["accounting"]["supervised_tokens"] == 0, "training endpoint configuration changed")
    require(report["trace_sha256"] == digest(folder / "trace.jsonl")
            and report["updates_sha256"] == digest(folder / "updates.jsonl")
            and report["final_sha256"] == adapter_digest(folder / "final-adapter.safetensors"),
            "training trace, optimizer updates or adapter bytes changed")
    return report


def restore_batches(model, learner, folder, prefix, schedule, start_hash):
    """Restore Adam by replaying each committed batch. Retry only its unfinished successor."""
    from experiments.gsm8k_mlx_development import adapter_snapshot
    from experiments.gsm8k_regroup import adapter_digest

    streams, ends = {}, {}
    inherited_count = {}
    for name in ("trace", "updates"):
        target = folder / f"{name}.jsonl"
        inherited = (prefix / target.name).read_bytes() if prefix is not None else b""
        if not target.exists():
            temp = target.with_suffix(".pending")
            temp.write_bytes(inherited)
            temp.replace(target)
        data = target.read_bytes()
        require(data.startswith(inherited) and (not inherited or inherited.endswith(b"\n")),
                f"{target}: inherited 32k prefix changed")
        inherited_count[name] = inherited.count(b"\n")
        complete = data[:data.rfind(b"\n") + 1].splitlines(keepends=True)
        streams[name] = [json.loads(line) for line in complete]
        ends[name] = [0]
        for line in complete:
            ends[name].append(ends[name][-1] + len(line))

    rows, updates = streams["trace"], streams["updates"]
    require(len(rows) <= len(schedule) and all(r["group_index"] == i and r["prompt_id"] == schedule[i]
            for i, r in enumerate(rows)), "committed trace belongs to another prompt schedule")
    current, ver = start_hash, 0
    done, cumulative = 0, 0
    for index, update in enumerate(updates):
        require(update["batch_index"] == index and (update["old_sha256"], update["old_version"]) == (current, ver),
                "replay update does not follow the previous weights")
        first, target = done, update["cumulative_generated_tokens"]
        while done < len(rows) and cumulative < target:
            r = rows[done]
            require(r["batch_index"] == index and r["sha256"] == current and r["version"] == ver,
                    "replay rollout batch or generating weights changed")
            cumulative += r["generated_tokens"]
            done += 1
        ids = [j for j in range(first, done) if rows[j]["consumed_tokens"]]
        require(cumulative == target and ids == update["group_indices"] and ids, "optimizer batch trace is incomplete")
        groups = [(rows[j]["prompt_ids"], rows[j]["samples"], rows[j]["advantages"]) for j in ids]
        actual = learner.step_groups(groups)
        expected = update["diagnostics"]
        new_hash = adapter_digest(adapter_snapshot(model))
        require(actual.keys() == expected.keys() and all(isclose(v, expected[k], rel_tol=1e-5, abs_tol=1e-6)
                for k, v in actual.items()) and new_hash == update["new_sha256"],
                f"optimizer replay diverged at batch {index}")
        ver += int(new_hash != current)
        require(update["new_version"] == ver, "restored batch version changed")
        current = new_hash
    # The source may end with one budget-limited singleton after its last update.
    retain = max(done, inherited_count["trace"])
    if prefix is not None:
        source = read_json(prefix / "training.json")
        require(sum(r["generated_tokens"] for r in rows[:inherited_count["trace"]]) == SHORT
                and source["final_sha256"] == (updates[inherited_count["updates"] - 1]["new_sha256"]
                                              if inherited_count["updates"] else start_hash),
                "wrong inherited endpoint or budget")
    require(all(not r["consumed_tokens"] for r in rows[done:retain]), "inherited tail has an uncommitted complete group")
    for name, count in (("trace", retain), ("updates", len(updates))):
        with (folder / f"{name}.jsonl").open("rb+") as handle:
            handle.truncate(ends[name][count])
    return rows[:retain], updates


def evaluate_cached(args, model, tokenizer, train, source, ids, binding):
    from experiments.gsm8k_budget import evaluate

    name = "initial" if args.stage == "initial" else f"{args.arm}-{args.budget}-seed-{args.seed}"
    path = args.out / "eval" / args.phase / f"{name}.json"
    gold = [(row["question"], float(row["value"])) for row in train]

    def read_checked(file, expected):
        data = read_json(file)
        require(all(data[k] == v for k, v in binding.items()), f"{file}: cached endpoint binding changed")
        check_eval(data, expected, gold, tokenizer, str(file))
        return data

    if path.exists():
        return read_checked(path, ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "chunks" / args.phase / name
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(0, len(ids), source["eval_batch"]):
        block = ids[offset:offset + source["eval_batch"]]
        chunk = scratch / f"{offset}.json"
        if not chunk.exists():
            record(chunk, {**evaluate(model, tokenizer, train, block, source), **binding})
        rows.extend(read_checked(chunk, block)["rows"])
    result = {**binding, "rows": rows, "correct": sum(r["reward"] for r in rows),
              "tagged": sum(r["has_answer_tag"] for r in rows), "truncated": sum(r["truncated"] for r in rows)}
    check_eval(result, ids, gold, tokenizer, str(path))
    record(path, result)
    for offset in range(0, len(ids), source["eval_batch"]):
        (scratch / f"{offset}.json").unlink()
    return result


def run(args):
    from safetensors.torch import load_file
    import torch
    from experiments.gsm8k_mlx_development import load_mlx_model, set_adapter
    from experiments.gsm8k_regroup import adapter_digest
    from experiments.gsm8k_signal import TRAIN_SHA256, read_split
    from experiments.gsm8k_stable import load_endpoint, train_case
    from experiments.gsm8k_training_reward_v2 import score_training_reward

    path = args.out / "protocol.json"
    protocol, protocol_hash = read_json(path), digest(path)
    check_request(args, protocol, protocol_hash)
    source = protocol["source_protocol"]
    require(source == read_json(args.source / "protocol.json"), "inherited source protocol changed")
    require((source["mlx"], source["mlx_lm"]) == (version("mlx"), version("mlx-lm")), "MLX versions changed")
    require(protocol["train_sha256"] == TRAIN_SHA256, "wrong GSM8K split")
    train = read_split(args.train, TRAIN_SHA256)
    start = load_file(args.source / "start-adapter.safetensors")
    require(adapter_digest(start) == source["start_sha256"], "common starting adapter changed")
    folder, binding, prefix = None, None, None
    if args.stage != "initial":
        folder = training_folder(args, args.budget)
        binding = binding_for(args, protocol, protocol_hash, args.budget)
        if args.stage == "train":
            folder.mkdir(parents=True, exist_ok=True)
            require((folder / "binding.json").is_file() or not any(folder.iterdir()), "unbound training directory")
            record(folder / "binding.json", binding)
            if args.budget == LONG:
                prefix = training_folder(args, SHORT)
                check_training(prefix, binding_for(args, protocol, protocol_hash, SHORT), protocol)
            if (folder / "training.json").exists():
                check_training(folder, binding, protocol)
                return
        else:
            report = check_training(folder, binding, protocol)

    torch.set_num_threads(1)
    model, tokenizer = load_mlx_model(None, len(source["qualification_protocol"]["lora_layers"]), 92333)
    if args.stage == "train":
        plan = {"source_protocol": source, "train_prompt_ids": protocol["train_prompt_ids"][args.arm],
                "token_budget": args.budget, "group_size": protocol["group_size"]}
        kwargs = {"training_reward": lambda sample, example: score_training_reward(sample, example["value"])}
        kwargs["restore_training"] = lambda model, learner: restore_batches(
            model, learner, folder, prefix, plan["train_prompt_ids"][str(args.seed)], source["start_sha256"])
        train_case(model, tokenizer, train, plan, start, protocol["lr"], protocol["accumulation_groups"],
                   args.seed, folder, **kwargs)
        check_training(folder, binding, protocol)
        return
    if args.stage == "initial":
        set_adapter(model, start)
        endpoint = source["start_sha256"]
    else:
        load_endpoint(model, folder, report)
        endpoint = report["final_sha256"]
    eval_binding = {"protocol_sha256": protocol_hash, "checkpoint_sha256": endpoint, "panel": args.phase,
                    "arm": args.arm, "budget": args.budget, "seed": args.seed}
    ids = protocol["eval_sets"][args.phase]
    result = evaluate_cached(args, model, tokenizer, train, source, ids, eval_binding)
    print(args.phase, args.stage, args.arm, args.budget, args.seed, result["correct"], "/", len(ids), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-budget-5seed"))
    parser.add_argument("--train", type=Path, default=Path("runs/gsm8k-data/train.jsonl"))
    parser.add_argument("--stage", choices=("train", "eval", "initial"), required=True)
    parser.add_argument("--phase", "--panel", choices=("primary", "dose"), default="primary")
    parser.add_argument("--arm", choices=("uniform", "mixed"))
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--budget", type=int, choices=(SHORT, LONG))
    args = parser.parse_args()
    with (args.out / ".runner.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
