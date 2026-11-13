"""Profile the CPU replay on its saved traces and validate seeded sampling."""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path
import platform
from unittest.mock import patch

from experiments import kv_replay as kv

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM

STAGE_PREFIX = "narl/"
ATTENTION_EVENT = "aten::scaled_dot_product_attention"


def scoped(name, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with record_function(STAGE_PREFIX + name):
            return function(*args, **kwargs)

    return wrapped


def stage_totals(events):
    totals = {}
    for event in events:
        if event.name.startswith(STAGE_PREFIX):
            name = event.name.removeprefix(STAGE_PREFIX)
            totals.setdefault(name, {"cpu_ms": 0.0, "sdpa_ms": 0.0, "calls": 0})
            totals[name]["cpu_ms"] += event.cpu_time_total / 1000
            totals[name]["calls"] += 1
        if event.name == ATTENTION_EVENT:
            parent = event.cpu_parent
            while parent is not None and not parent.name.startswith(STAGE_PREFIX):
                parent = parent.cpu_parent
            if parent is None:
                raise RuntimeError("attention operator had no replay stage")
            name = parent.name.removeprefix(STAGE_PREFIX)
            totals.setdefault(name, {"cpu_ms": 0.0, "sdpa_ms": 0.0, "calls": 0})
            totals[name]["sdpa_ms"] += event.cpu_time_total / 1000
    return totals


def profile_replay(actor, old, new, prompt, suffix, continuation, mode, repetitions):
    groups = suffix.shape[0]
    history = torch.cat((prompt.expand(groups, -1), suffix), dim=1)
    original_forward = kv._forward
    is_old_prefill = False
    forwards = {
        name: scoped(name, original_forward)
        for name in ("old_prefill", "prompt_prefill", "prompt_and_suffix_prefill",
                     "suffix_prefill", "continuation_decode")
    }

    def forward(model, ids, attention_mask, cache=None, **kwargs):
        if is_old_prefill:
            name = "old_prefill"
        elif cache is None:
            name = "prompt_prefill" if ids.shape == prompt.shape else "prompt_and_suffix_prefill"
        else:
            name = "suffix_prefill" if ids.shape[1] > 1 else "continuation_decode"
        return forwards[name](model, ids, attention_mask, cache, **kwargs)

    def prepare_old_cache():
        nonlocal is_old_prefill
        actor.load_state_dict(old)
        is_old_prefill = True
        try:
            return kv.prefill(actor, history).past_key_values
        finally:
            is_old_prefill = False

    def recover(cache):
        with record_function(STAGE_PREFIX + "replay_total"):
            kv.replay(actor, mode, prompt, suffix, continuation, cache)

    for _ in range(3):
        cache = prepare_old_cache()
        actor.load_state_dict(new)
        recover(cache if mode == "retain" else None)

    cache_type = type(cache)
    prepare = scoped("old_cache_prepare", prepare_old_cache)
    load = scoped("weight_load", actor.load_state_dict)
    with patch.object(kv, "_forward", forward), patch.object(
        cache_type, "batch_repeat_interleave",
        scoped("branch_copy", cache_type.batch_repeat_interleave),
    ), profile(activities=[ProfilerActivity.CPU]) as profiler:
        for _ in range(repetitions):
            cache = prepare()
            load(new)
            recover(cache if mode == "retain" else None)
            del cache

    stages = stage_totals(profiler.events())
    for item in stages.values():
        item["mean_cpu_ms_per_replay"] = item["cpu_ms"] / repetitions
        item["mean_sdpa_ms_per_replay"] = item["sdpa_ms"] / repetitions
    recovery_sdpa_ms = sum(value["sdpa_ms"] for name, value in stages.items()
                           if name != "old_prefill")
    total_ms = stages["replay_total"]["cpu_ms"]
    if not 0 < recovery_sdpa_ms <= total_ms:
        raise RuntimeError("unexpected recovery attention time")
    return {
        "stages": stages,
        "mean_recovery_sdpa_ms": recovery_sdpa_ms / repetitions,
        "recovery_sdpa_fraction": recovery_sdpa_ms / total_ms,
        "top_operators_by_self_cpu": [
            {"name": event.key, "self_cpu_ms": event.self_cpu_time_total / 1000,
             "calls": event.count}
            for event in sorted(
                (event for event in profiler.key_averages() if event.key.startswith("aten::")),
                key=lambda event: event.self_cpu_time_total, reverse=True,
            )[:10]
        ],
    }


def fixed_draw_equivalence(actor, prompt, suffix, boundary, steps, seed):
    """Sample each arm independently with the same per-branch inverse-CDF draws."""
    group_size = suffix.shape[0]
    uniforms = torch.rand((group_size, steps), dtype=torch.float64,
                          generator=torch.Generator().manual_seed(seed))
    samples = {}
    for mode in ("recompute", "share_prompt"):
        forced = boundary.clone()
        for step in range(steps):
            logits, _ = kv.replay(actor, mode, prompt, suffix, forced)
            cdf = logits[:, -1].double().softmax(-1).cumsum(-1)
            token = torch.searchsorted(cdf, uniforms[:, step:step + 1].contiguous())
            forced = torch.cat((forced, token), dim=1)
        samples[mode] = forced[:, 1:]
    # Fixed-length draws exercise the token sampler. EOS masking is checked separately
    # in the replay tests and is outside this parity check.
    if not torch.equal(samples["recompute"], samples["share_prompt"]):
        raise AssertionError("shared prefill changed tokens under common sampling draws")
    return {
        "draw_seed": seed,
        "compared_sampled_tokens": samples["recompute"].numel(),
        "sampled_tokens": samples["recompute"].tolist(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="saved KV replay directory")
    parser.add_argument("--out", type=Path, required=True, help="new output JSON path")
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summary = json.loads((args.run / "summary.json").read_text())
    for version in ("old", "new"):
        snapshot = summary["snapshots"][version]
        actual_digest = hashlib.sha256((args.run / snapshot["path"]).read_bytes()).hexdigest()
        if actual_digest != snapshot["sha256"]:
            raise ValueError(f"snapshot hash mismatch: {version}")
    actor = AutoModelForCausalLM.from_pretrained(
        args.run / "model", local_files_only=True, attn_implementation="sdpa",
    ).eval()
    old = {key: value.clone() for key, value in actor.state_dict().items()}
    new = torch.load(args.run / summary["snapshots"]["new"]["path"], weights_only=True)
    cases = []
    for trace in json.loads((args.run / "inputs.json").read_text()):
        prompt = torch.tensor(trace["prompt"])
        suffix = torch.tensor(trace["cached_suffix"])
        continuation = torch.tensor(trace["boundary_and_forced_tokens"])[:, :-1]
        profiles = {
            mode: profile_replay(actor, old, new, prompt, suffix, continuation, mode, args.repetitions)
            for mode in kv.MODES
        }
        actor.load_state_dict(new)
        cases.append({
            "prompt_length": prompt.shape[1],
            "cached_suffix_length": suffix.shape[1],
            "group_size": suffix.shape[0],
            "profile": profiles,
            "fixed_draw_equivalence": fixed_draw_equivalence(
                actor, prompt, suffix, continuation[:, :1], continuation.shape[1], seed=547,
            ),
        })
    cuda_available = torch.cuda.is_available()
    report = {
        "machine": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "dtype": str(next(actor.parameters()).dtype),
        "replay_source": str(args.run),
        "snapshots": summary["snapshots"],
        "repetitions_per_mode": args.repetitions,
        "profiler_boundary": (
            "Stage times include profiler overhead. Old cache preparation and state_dict copy "
            "are outside replay_total. SDPA includes its child operators and measures only the "
            "attention computation, excluding Q/K/V and output projections. Operators in "
            "top_operators_by_self_cpu cover all stages."
        ),
        "cuda_gate": {
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count(),
            "mps_available": torch.backends.mps.is_available(),
            "decision": "profile a CUDA candidate" if cuda_available else "no CUDA benchmark",
        },
        "cases": cases,
    }
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
