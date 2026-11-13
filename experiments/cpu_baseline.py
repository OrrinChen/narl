"""Run nanoRL's CPU GRPO entry points on a locally generated tiny decoder."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks import SMOKE_CHOICES, SMOKE_QUERIES

CONFIG = ROOT / "configs/narl_cpu_baseline.yaml"


def save_fixture(path: Path):
    special = ("[PAD]", "[EOS]", "[UNK]")
    words = (*special, "0", "1", *SMOKE_QUERIES, *SMOKE_CHOICES)
    vocabulary = {token: index for index, token in enumerate(words)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
    ).save_pretrained(path)

    config = LlamaConfig(
        vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=0, eos_token_id=1,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)
    return sum(parameter.numel() for parameter in model.parameters())


def read_steps(path: Path):
    steps = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            steps.setdefault(int(row["step"]), {})[row["key"]] = float(row["value"])
    return steps


def summarize(path: Path, elapsed: float, updates: int, max_lag: int):
    steps = read_steps(path)
    if set(steps) != set(range(updates)) or not any(row["gnorm"] > 0 for row in steps.values()):
        raise RuntimeError(f"{path} must contain {updates} updates and a nonzero training signal")
    for row in steps.values():
        lag = row["learner_version"] - row["behavior_version"]
        if lag != row["staleness"] or not 0 <= lag <= max_lag:
            raise RuntimeError(f"{path} records invalid lag {lag}")
    last = steps[max(steps)]
    return {
        "updates": len(steps),
        "nonzero_gradient_updates": sum(row["gnorm"] > 0 for row in steps.values()),
        "groups_consumed": int(sum(row["groups"] for row in steps.values())),
        "accepted_tokens": int(sum(row["accepted_tokens"] for row in steps.values())),
        "lag_values": sorted({int(row["staleness"]) for row in steps.values()}),
        "dropped_stale_batches": int(last.get("dropped_stale", 0)),
        "dropped_overflow_batches": int(last.get("dropped_overflow", 0)),
        "last_heldout_eval": next(row["eval_acc"] for row in reversed(steps.values())
                                  if "eval_acc" in row),
        "elapsed_s_including_startup": round(elapsed, 3),
    }


def run_roles(base, env, output: Path, config):
    start = time.perf_counter()
    with (output / "sync.log").open("w") as log:
        result = subprocess.run([*base, "--out", str(output / "sync")], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError(f"sync failed, see {output / 'sync.log'}")
    sync = summarize(output / "sync.csv", time.perf_counter() - start, config["steps"], 0)

    # The smoke uses a loopback listener and an ephemeral per-run token.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    start = time.perf_counter()
    with (output / "trainer.log").open("w") as trainer_log, (output / "rollout.log").open("w") as rollout_log:
        trainer = subprocess.Popen(
            [*base, "--role", "trainer", "--serve-host", "127.0.0.1", "--serve-port", str(port),
             "--out", str(output / "async")], cwd=ROOT, env=env, stdout=trainer_log,
            stderr=subprocess.STDOUT,
        )
        processes = [trainer]
        try:
            rollout = subprocess.Popen(
                [*base, "--role", "rollout", "--trainer-url", f"http://127.0.0.1:{port}"],
                cwd=ROOT, env=env, stdout=rollout_log, stderr=subprocess.STDOUT,
            )
            processes.append(rollout)
            if trainer.wait(timeout=120):
                raise RuntimeError(f"trainer failed, see {output / 'trainer.log'}")
            if rollout.poll() not in (None, 0):
                raise RuntimeError(f"rollout failed, see {output / 'rollout.log'}")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    async_result = summarize(output / "async.csv", time.perf_counter() - start,
                             config["steps"], config["max_staleness"])
    return sync, async_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True)
    fixture = output / "model"
    parameters = save_fixture(fixture)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "NANORL_TOKEN": secrets.token_urlsafe(24),
    }
    base = [
        sys.executable, str(ROOT / "train.py"), "--config", str(CONFIG),
        "--model", str(fixture), "--seed", str(args.seed),
    ]
    config = yaml.safe_load(CONFIG.read_text())
    (output / "config.yaml").write_text(CONFIG.read_text())
    sync, async_result = run_roles(base, env, output, config)
    report = {
        "upstream_commit": "793e9b5b3d176ebb40161d8ba8d2125fe5153473",
        "machine": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "tokenizers", "pyyaml")},
        "seed": args.seed,
        "threads_per_role": 1,
        "model_parameters": parameters,
        "sync": sync,
        "async": async_result,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
