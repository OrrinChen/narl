"""Compare three GRPO group assignments on one budget-matched GSM8K rollout pool."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import __version__ as transformers_version

from algos import grpo
from core import Batch
from experiments.audit_gsm8k_matched import check_development
from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_regroup import Learner, adapter_digest, adapter_snapshot, move_model, trajectory
from experiments.gsm8k_signal import (
    MODEL, REVISION, TEST_SHA256, TRAIN_SHA256, generate_batch, load_model, read_split,
)
from experiments.regroup import write_json, write_record
from train import Config, optimize


ARMS = ("version_homogeneous", "version_mixed", "same_version_shuffle")
INSPECTED_TEST_IDS = (106, 419, 426, 533, 629, 765, 771, 799, 927)


def group_indices(size, seed, arm):
    groups = [list(range(start, start + size)) for start in range(0, 4 * size, size)]
    if arm == "version_homogeneous":
        return groups
    rng = Random(seed)
    for group in groups:
        rng.shuffle(group)
    pairs = ((0, 2), (1, 3)) if arm == "version_mixed" else ((0, 1), (2, 3))
    half = size // 2
    return [groups[i][start:start + half] + groups[j][start:start + half]
            for start in (0, half) for i, j in pairs]


def qualify(path, train):
    dev = read_json(path / "protocol.json")
    result = read_json(path / "dev-qualified.json")
    require((dev["model"], dev["revision"], dev["train_sha256"]) == (MODEL, REVISION, TRAIN_SHA256),
            "qualification used a different model or dataset revision")
    require(result["qualified"] is True and result["development_count"] == len(dev["development_prompt_ids"])
            and result["after_correct"] - result["before_correct"] >= dev["minimum_correct_gain"]
            and result["selected_step"] in dev["checkpoints"], "the independent development gate did not pass")
    check_development(path, dev, result, [(row["question"], float(row["value"])) for row in train])
    snapshots = {name: load_file(path / f"{name}-adapter.safetensors") for name in ("old", "fresh")}
    digests = {f"{name}_adapter_sha256": adapter_digest(snapshot) for name, snapshot in snapshots.items()}
    require(digests["old_adapter_sha256"] == dev["old_adapter_sha256"]
            and len(set(digests.values())) == 2
            and all(result[key] == value for key, value in digests.items()),
            "qualified snapshot hashes do not match the recorded weights")
    return dev, result, snapshots


def make_protocol(args, train, test, dev, qualification):
    excluded = set(dev["training_prompt_ids"] + dev["development_prompt_ids"])
    rng = Random(args.seed)
    replay = rng.sample([i for i in range(len(train)) if i not in excluded], args.replay_prompts)
    heldout = rng.sample([i for i in range(len(test)) if i not in INSPECTED_TEST_IDS], args.heldout_prompts)
    seen = {train[i]["question"] for i in excluded}
    replay_questions = {train[i]["question"] for i in replay}
    require(not seen & replay_questions,
            "replay questions overlap snapshot training or development")
    require(not (seen | replay_questions) & {test[i]["question"] for i in heldout},
            "final held-out questions overlap the train split")
    protocol = {
        "train_sha256": TRAIN_SHA256, "test_sha256": TEST_SHA256, "model": MODEL, "revision": REVISION,
        "seed": args.seed, "replay_prompt_ids": replay, "heldout_prompt_ids": heldout,
        "excluded_test_ids": list(INSPECTED_TEST_IDS), "qualification": qualification,
        "qualification_protocol": dev, "max_new_tokens": args.max_new_tokens,
        "eval_max_new_tokens": args.eval_max_new_tokens, "eval_batch": args.eval_batch,
        "group_size": args.group_size, "groups_per_prompt": 4, "rollout_batch": 2 * args.group_size, "lr": args.lr,
        "device": args.device, "dtype": args.dtype, "learner_device": args.learner_device,
        "learner_dtype": args.learner_dtype, "threads": args.threads,
        "generation_attention": "mlx" if args.device == "mlx" else "sdpa",
        "learner_attention": "mlx" if args.device == "mlx" else "eager",
        "torch": torch.__version__, "transformers": transformers_version,
    }
    if args.device == "mlx":
        protocol.update(mlx=version("mlx"), mlx_lm=version("mlx-lm"))
        require((protocol["mlx"], protocol["mlx_lm"]) == (dev["mlx"], dev["mlx_lm"]),
                "MLX versions differ from the independent development gate")
    return protocol


def save_summary(path, report):
    pending = path.with_suffix(".tmp")
    write_json(pending, report)
    pending.replace(path)


def capture(model, tokenizer, train, protocol, snapshots, path, *, generate, set_adapter, seed_rollout):
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            # Only complete newline-terminated records survived the last capture.
            with path.open("r+b") as handle:
                handle.truncate(data.rfind(b"\n") + 1)
    records = read_rows(path) if path.exists() else []
    ids = protocol["replay_prompt_ids"]
    require([row["prompt_id"] for row in records] == ids[:len(records)], "saved rollout order does not match protocol")
    with path.open("a") as handle:
        for prompt_id in ids[len(records):]:
            cohorts = {}
            prompt = None
            for version, snapshot in snapshots.items():
                set_adapter(model, snapshot)
                seed_rollout((protocol["seed"] * 2 + (version == "fresh")) * 100_000 + prompt_id)
                outputs = generate(
                    model, tokenizer, [train[prompt_id]] * protocol["rollout_batch"], protocol["max_new_tokens"],
                    sample=True, capture=True, prompt_mode=protocol["qualification_protocol"]["prompt_mode"],
                )
                require(len(outputs) == protocol["rollout_batch"], "rollout batch is incomplete")
                prompt_ids = outputs[0][0]
                samples = [row for _, row in outputs]
                require(all(ids == prompt_ids for ids, _ in outputs), "one prompt produced different token inputs")
                require(prompt is None or prompt == prompt_ids, "checkpoint switch changed prompt tokens")
                prompt = prompt_ids
                for index, sample in enumerate(samples):
                    sample["id"] = f"{prompt_id}:{version}:{index}"
                cohorts[version] = samples
            write_record(handle, {"prompt_id": prompt_id, "prompt_ids": prompt, "cohorts": cohorts})
            handle.flush()
            stats = {version: {"rewards": [s["reward"] for s in group],
                               "tagged": sum(s["has_answer_tag"] for s in group),
                               "truncated": sum(s["truncated"] for s in group),
                               "tokens": sum(len(s["token_ids"]) for s in group)}
                     for version, group in cohorts.items()}
            print(f"capture {prompt_id}", stats, flush=True)


def evaluate_test(model, tokenizer, test, protocol, generate):
    rows = []
    ids = protocol["heldout_prompt_ids"]
    size = protocol["eval_batch"]
    for start in range(0, len(ids), size):
        block = ids[start:start + size]
        outputs = generate(model, tokenizer, [test[i] for i in block], protocol["eval_max_new_tokens"],
                           prompt_mode=protocol["qualification_protocol"]["prompt_mode"])
        rows.extend({"prompt_id": i, **row} for i, (_, row) in zip(block, outputs, strict=True))
    return {"rows": rows, "correct": sum(row["reward"] for row in rows),
            "tagged": sum(row["has_answer_tag"] for row in rows), "truncated": sum(row["truncated"] for row in rows)}


def fit_arm(model, tokenizer, snapshot, records, test, protocol, arm, *, generate, set_adapter, get_adapter):
    set_adapter(model, snapshot)
    start_hash = adapter_digest(get_adapter(model))
    require(start_hash == protocol["qualification"]["fresh_adapter_sha256"], "learner did not load qualified weights")
    size = protocol["group_size"]
    mlx = protocol["device"] == "mlx"
    if mlx:
        from experiments.gsm8k_mlx_grpo import MlxGrpoLearner

        learner = MlxGrpoLearner(model, protocol["lr"], micro_batch=1)
    else:
        move_model(model, protocol["learner_device"], protocol["learner_dtype"])
        model.get_base_model().set_attn_implementation(protocol["learner_attention"])
        learner = Learner(model)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=protocol["lr"])
        cfg = Config(model=MODEL, task="gsm8k", algo="grpo", group_size=size, micro_batch=1,
                     lr=protocol["lr"], max_new_tokens=protocol["max_new_tokens"], device=protocol["learner_device"])
    updates = []
    for record in records:
        prompt_id = record["prompt_id"]
        pool = record["cohorts"]["old"] + record["cohorts"]["fresh"]
        rows = [trajectory(record["prompt_ids"], sample) for sample in pool]
        batch = Batch.from_groups([rows[start:start + size] for start in range(0, len(rows), size)])
        ordering = group_indices(size, protocol["seed"] + prompt_id, arm)
        grouped = Batch.from_groups([[rows[index] for index in group] for group in ordering])
        inverse = torch.argsort(torch.tensor(ordering).flatten())
        advantage = grpo(grouped, learner)[inverse]
        tokens = int(batch.mask.sum())
        if mlx:
            scalars = [float(adv[len(record["prompt_ids"])]) for adv in advantage]
            diagnostics = learner.step(record["prompt_ids"], pool, scalars)
        else:
            diagnostics = optimize(learner, optimizer, batch.to(cfg.device), advantage.to(cfg.device), cfg)
        require(all(isfinite(value) for value in diagnostics.values()), f"nonfinite {arm} update for prompt {prompt_id}")
        updates.append({"prompt_id": prompt_id, "groups": [[pool[i]["id"] for i in group] for group in ordering],
                        "completion_tokens": tokens, "nonzero_advantage_tokens": int(advantage.ne(0).sum()),
                        "diagnostics": diagnostics})
        print(arm, prompt_id, tokens, diagnostics["gnorm"], flush=True)
    if not mlx:
        optimizer.zero_grad(set_to_none=True)
        move_model(model, protocol["device"], protocol["dtype"])
        model.get_base_model().set_attn_implementation(protocol["generation_attention"])
    end_hash = adapter_digest(get_adapter(model))
    return {"initial_adapter_sha256": start_hash, "final_adapter_sha256": end_hash,
            "consumed_tokens": sum(row["completion_tokens"] for row in updates), "updates": updates,
            "evaluation": evaluate_test(model, tokenizer, test, protocol, generate)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "test", "qualification", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("capture", "fit", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "mps", "mlx"))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--learner-device", choices=("cpu", "mps", "mlx"))
    parser.add_argument("--learner-dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--model-path", type=Path, help="local MLX model snapshot matching the qualified revision")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-batch", type=int)
    parser.add_argument("--replay-prompts", type=int, default=12)
    parser.add_argument("--heldout-prompts", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=8461)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    train = read_split(args.train, TRAIN_SHA256)
    test = read_split(args.test, TEST_SHA256)
    dev, qualification, snapshots = qualify(args.qualification, train)
    args.device = args.device or dev["device"]
    args.dtype = args.dtype or dev["dtype"]
    args.learner_device = args.learner_device or ("mlx" if args.device == "mlx" else "cpu")
    args.learner_dtype = args.learner_dtype or (args.dtype if args.learner_device == "mlx" else "float32")
    if "mps" in (args.device, args.learner_device) and not torch.backends.mps.is_available():
        parser.error("PyTorch cannot see an MPS device")
    if (args.device == "mlx") != (args.learner_device == "mlx"):
        parser.error("native MLX rollout and learning use one shared backend")
    if args.device == "mlx" and args.learner_dtype != args.dtype:
        parser.error("native MLX uses the loaded base dtype throughout the experiment")
    args.eval_batch = dev["eval_batch"] if args.eval_batch is None else args.eval_batch
    args.eval_max_new_tokens = dev["max_new_tokens"] if args.eval_max_new_tokens is None else args.eval_max_new_tokens
    if min(args.threads, args.eval_batch, args.replay_prompts, args.heldout_prompts, args.max_new_tokens, args.lr) <= 0:
        parser.error("budgets and learning rate must be positive")
    if args.group_size < 2 or args.group_size % 2:
        parser.error("group size must be positive, even and at least two")
    if ((args.eval_max_new_tokens, args.dtype, args.device, args.eval_batch) !=
            (dev["max_new_tokens"], dev["dtype"], dev["device"], dev["eval_batch"])):
        parser.error("evaluation token cap, dtype, device and batch size must match the independent development gate")
    protocol = make_protocol(args, train, test, dev, qualification)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    samples_path = args.out / "samples.jsonl"
    summary_path = args.out / "summary.json"
    if protocol_path.exists():
        require(read_json(protocol_path) == protocol, "requested settings differ from the frozen run protocol")
    else:
        require(args.stage != "fit", "fit requires a frozen rollout pool")
        save_summary(protocol_path, protocol)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    if args.device == "mlx":
        from mlx import core as mx
        from experiments.gsm8k_mlx_development import adapter_snapshot as get_adapter, load_mlx_model, set_adapter
        from experiments.gsm8k_mlx_generate import generate_batch as generate

        model, tokenizer = load_mlx_model(args.model_path, len(dev["lora_layers"]), args.seed)
        seed_rollout = mx.random.seed
    else:
        model, tokenizer = load_model(getattr(torch, args.dtype))
        config = LoraConfig(r=dev["lora_rank"], lora_alpha=dev["lora_alpha"], lora_dropout=0,
                            target_modules=dev["lora_targets"], layers_to_transform=dev["lora_layers"], task_type="CAUSAL_LM")
        model = get_peft_model(model, config).to(args.device).eval()
        generate, set_adapter, get_adapter = generate_batch, set_peft_model_state_dict, adapter_snapshot
        seed_rollout = torch.manual_seed
    if args.stage in ("capture", "all"):
        capture(model, tokenizer, train, protocol, snapshots, samples_path, generate=generate,
                set_adapter=set_adapter, seed_rollout=seed_rollout)
    if args.stage == "capture":
        return
    started = perf_counter()
    records = read_rows(samples_path)
    require([row["prompt_id"] for row in records] == protocol["replay_prompt_ids"], "fit requires a complete rollout pool")
    sample_sha = sha256(samples_path.read_bytes()).hexdigest()
    report = read_json(summary_path) if summary_path.exists() else {"sample_sha256": sample_sha, "arms": {}}
    require(report["sample_sha256"] == sample_sha, "saved samples changed after fitting an arm")
    if "initial_evaluation" not in report:
        set_adapter(model, snapshots["fresh"])
        report["initial_evaluation"] = evaluate_test(model, tokenizer, test, protocol, generate)
        save_summary(summary_path, report)
        print("initial", report["initial_evaluation"]["correct"], flush=True)
    for arm in ARMS:
        if arm not in report["arms"]:
            result = fit_arm(model, tokenizer, snapshots["fresh"], records, test, protocol, arm,
                             generate=generate, set_adapter=set_adapter, get_adapter=get_adapter)
            report["arms"][arm] = result
            report["fit_elapsed_s"] = report.get("fit_elapsed_s", 0) + perf_counter() - started
            save_summary(summary_path, report)
            started = perf_counter()
            print(arm, result["evaluation"]["correct"], result["consumed_tokens"], flush=True)


if __name__ == "__main__":
    main()
