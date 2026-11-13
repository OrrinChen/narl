"""Replay the same token history across a weight switch on the Narl CPU fixture."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns

from experiments.cpu_baseline import CONFIG, ROOT, save_fixture

import torch
import torch.nn.functional as F
import transformers
import yaml

from train import Config, make_policy, train

MODES = ("retain", "recompute", "share_prompt")
ATOL = 2e-6
RTOL = 1e-5
PROMPT_LENGTHS = (32, 128, 384)
GROUP_SIZE = 8
SUFFIX_LENGTH = 16
DECODE_STEPS = 8
WARMUPS = 3


def _forward(model, ids, attention_mask, cache=None, *, use_cache=True):
    positions = attention_mask.long().cumsum(dim=1) - 1
    positions.masked_fill_(attention_mask == 0, 0)
    return model(
        input_ids=ids, attention_mask=attention_mask, position_ids=positions[:, -ids.size(1):],
        past_key_values=cache, use_cache=use_cache, logits_to_keep=1,
    )


def prefill(model, token_ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(token_ids)
    return _forward(model, token_ids, attention_mask)


def valid_prediction_mask(continuation, eos_id):
    """Predict after a forced input only while that completion has not reached EOS."""
    return continuation.eq(eos_id).cumsum(dim=1).eq(0)


def _ms(start):
    return (perf_counter_ns() - start) / 1e6


def rebuild_cache(model, mode, prompt, suffix, prompt_mask, history_mask):
    """Rebuild the same group history under one snapshot and time the prompt copy."""
    start = perf_counter_ns()
    if mode == "recompute":
        history = torch.cat((prompt.expand(suffix.size(0), -1), suffix), dim=1)
        return prefill(model, history, history_mask).past_key_values, _ms(start), 0.0
    if mode != "share_prompt":
        raise ValueError(f"unknown recompute mode: {mode}")
    cache = prefill(model, prompt, prompt_mask).past_key_values
    prompt_ms = _ms(start)
    copy_start = perf_counter_ns()
    cache.batch_repeat_interleave(suffix.size(0))
    copy_ms = _ms(copy_start)
    suffix_start = perf_counter_ns()
    cache = _forward(model, suffix, history_mask, cache).past_key_values
    return cache, prompt_ms + _ms(suffix_start), copy_ms


@torch.inference_mode()
def replay(model, mode, prompt, suffix, continuation, old_cache=None, prompt_mask=None):
    """Return next-token logits after each forced continuation input, and timings.

    The first continuation input is the sampled boundary token pending at the swap.
    Cache positions follow the physical prefix length, while RoPE positions skip left padding.
    """
    if mode not in MODES:
        raise ValueError(f"unknown replay mode: {mode}")
    if mode == "retain" and old_cache is None:
        raise ValueError("retain requires a cache built before the weight switch")
    groups = suffix.size(0)
    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt)
    history_mask = torch.cat((prompt_mask.expand(groups, -1), torch.ones_like(suffix)), dim=1)
    decode_mask = torch.cat((history_mask, torch.ones_like(continuation)), dim=1)
    active_prompt_tokens = int(prompt_mask.sum())
    recompute_ms = 0.0
    cache_copy_ms = 0.0
    prompt_tokens_recomputed = 0
    suffix_tokens_recomputed = 0

    start = perf_counter_ns()
    if mode == "retain":
        cache = old_cache
    else:
        cache, recompute_ms, cache_copy_ms = rebuild_cache(
            model, mode, prompt, suffix, prompt_mask, history_mask,
        )
        prompt_tokens_recomputed = active_prompt_tokens * (groups if mode == "recompute" else 1)
        suffix_tokens_recomputed = suffix.numel()

    logits = []
    past_length = history_mask.size(1)
    decode_start = perf_counter_ns()
    for step, token in enumerate(continuation.split(1, dim=1)):
        out = _forward(model, token, decode_mask[:, :past_length + step + 1], cache)
        if step == 0:
            first_decode_ms = _ms(decode_start)
            first_token_ms = _ms(start)
        logits.append(out.logits)
        cache = out.past_key_values
    return torch.cat(logits, dim=1), {
        "prompt_tokens_recomputed": prompt_tokens_recomputed,
        "suffix_tokens_recomputed": suffix_tokens_recomputed,
        "recompute_ms": recompute_ms,
        "cache_copy_ms": cache_copy_ms,
        "first_decode_ms": first_decode_ms,
        "first_token_ms": first_token_ms,
        "replay_ms": _ms(start),
    }


def _full_logits(model, prompt, suffix, continuation):
    history = torch.cat((prompt.expand(suffix.size(0), -1), suffix, continuation), dim=1)
    out = model(input_ids=history, use_cache=False, logits_to_keep=continuation.size(1))
    return out.logits


def _compare(reference, candidate, targets, valid):
    if not valid.any():
        raise ValueError("the trace must contain at least one prediction before EOS")
    ref_logp = F.log_softmax(reference.double(), dim=-1)
    candidate_logp = F.log_softmax(candidate.double(), dim=-1)
    kl = (ref_logp.exp() * (ref_logp - candidate_logp)).sum(dim=-1).clamp_min(0)
    logp_delta = (ref_logp - candidate_logp).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return {
        "mean_kl_nats": float(kl[valid].mean()),
        "max_kl_nats": float(kl[valid].max()),
        "mean_kl_by_decode_step_nats": [
            float(values[active].mean()) if active.any() else None
            for values, active in zip(kl.T, valid.T)
        ],
        "max_abs_logit_error": float((reference - candidate).abs()[valid].max()),
        "max_abs_forced_token_logprob_error": float(logp_delta.abs()[valid].max()),
        "greedy_matches": int((reference.argmax(-1) == candidate.argmax(-1))[valid].sum()),
        "predictions": int(valid.sum()),
    }


def _check_alignment(reference, candidate, targets, valid):
    torch.testing.assert_close(candidate[valid], reference[valid], atol=ATOL, rtol=RTOL)
    metrics = _compare(reference, candidate, targets, valid)
    if metrics["greedy_matches"] != metrics["predictions"]:
        raise AssertionError("numerically close replay changed the greedy token")
    if metrics["max_abs_forced_token_logprob_error"] > ATOL:
        raise AssertionError("forced token logprob outside the absolute tolerance")
    return metrics


def _timing_summary(rows):
    keys = ("weight_load_ms", "recompute_ms", "cache_copy_ms",
            "first_decode_ms", "first_token_ms", "replay_ms")
    summary = {}
    for key in keys:
        values = [row[key] for row in rows]
        summary[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return summary


@torch.inference_mode()
def run_case(actor, old_state, new_state, prompt, suffix, stream, repetitions):
    continuation = stream[:, :-1]
    targets = stream[:, 1:]
    valid = valid_prediction_mask(continuation, actor.config.eos_token_id)
    groups, suffix_length = suffix.shape
    prompt_length = prompt.size(1)
    history = torch.cat((prompt.expand(groups, -1), suffix), dim=1)

    actor.load_state_dict(old_state)
    full_old = _full_logits(actor, prompt, suffix, continuation)
    same_weight, _ = replay(
        actor, "retain", prompt, suffix, continuation,
        old_cache=prefill(actor, history).past_key_values,
    )
    same_weight_metrics = _check_alignment(full_old, same_weight, targets, valid)

    traces = []
    logits_by_mode = {}
    for repetition in range(-WARMUPS, repetitions):
        # Rotate order to avoid always measuring a path in the same thermal/cache position.
        rotation = repetition % len(MODES)
        for mode in MODES[rotation:] + MODES[:rotation]:
            actor.load_state_dict(old_state)
            old_cache = prefill(actor, history).past_key_values
            swap_start = perf_counter_ns()
            actor.load_state_dict(new_state)
            weight_load_ms = _ms(swap_start)
            if mode != "retain":
                del old_cache
                old_cache = None
            logits, timings = replay(actor, mode, prompt, suffix, continuation, old_cache)
            if not torch.isfinite(logits).all():
                raise AssertionError(f"non-finite replay output in {mode}")
            if repetition >= 0:
                traces.append({
                    "prompt_length": prompt_length, "suffix_length": suffix_length,
                    "group_size": groups, "repetition": repetition, "mode": mode,
                    "weight_load_ms": weight_load_ms, **timings,
                })
                if mode in logits_by_mode:
                    torch.testing.assert_close(logits, logits_by_mode[mode], atol=ATOL, rtol=RTOL)
                else:
                    logits_by_mode[mode] = logits

    reference = logits_by_mode["recompute"]
    fresh_full_metrics = _check_alignment(
        _full_logits(actor, prompt, suffix, continuation), reference, targets, valid,
    )
    shared_metrics = _check_alignment(reference, logits_by_mode["share_prompt"], targets, valid)
    comparisons = {
        "retain": _compare(reference, logits_by_mode["retain"], targets, valid),
        "recompute": _compare(reference, reference, targets, valid),
        "share_prompt": shared_metrics,
    }
    expected_counts = {
        "retain": (0, 0),
        "recompute": (groups * prompt_length, groups * suffix_length),
        "share_prompt": (prompt_length, groups * suffix_length),
    }
    modes = {}
    for mode in MODES:
        rows = [row for row in traces if row["mode"] == mode]
        counts = expected_counts[mode]
        for row in rows:
            if (row["prompt_tokens_recomputed"], row["suffix_tokens_recomputed"]) != counts:
                raise AssertionError(f"unexpected recompute token counts in {mode}")
        modes[mode] = {
            "timing_ms": _timing_summary(rows),
            "prompt_tokens_recomputed": counts[0],
            "suffix_tokens_recomputed": counts[1],
            "vs_recompute": comparisons[mode],
        }
    return {
        "prompt_length": prompt_length, "suffix_length": suffix_length, "group_size": groups,
        "weight_switch_after_sampled_token": suffix_length + 1,
        "old_cache_tokens_per_branch": prompt_length + suffix_length,
        "same_weight_retain_vs_full": same_weight_metrics,
        "new_weight_recompute_vs_full": fresh_full_metrics,
        "modes": modes,
    }, traces


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.repetitions < 5:
        parser.error("--repetitions must be at least 5")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.out.resolve()
    output.mkdir(parents=True)
    fixture = output / "model"
    parameter_count = save_fixture(fixture)

    cfg = Config(**yaml.safe_load(CONFIG.read_text()), model=str(fixture), seed=args.seed,
                 out=str(output / "snapshot_training"))
    (output / "snapshot_config.yaml").write_text(CONFIG.read_text())
    actor = make_policy(cfg, None).model.eval()
    old_state = {name: tensor.clone() for name, tensor in actor.state_dict().items()}
    trained_policy = train(cfg)
    new_state = trained_policy.model.state_dict()
    max_weight_delta = max(float((tensor - old_state[name]).abs().max())
                           for name, tensor in new_state.items())
    if max_weight_delta == 0:
        raise RuntimeError("GRPO did not produce a distinct snapshot")
    new_snapshot = output / "new_state.pt"
    torch.save(new_state, new_snapshot)

    generator = torch.Generator().manual_seed(args.seed)
    cases = []
    traces = []
    input_traces = []
    for length in PROMPT_LENGTHS:
        # Only content tokens. The fixed history is shared by all modes, including the
        # pending boundary input. The final stream token is a target, not another decode.
        prompt = torch.randint(3, actor.config.vocab_size, (1, length), generator=generator)
        suffix = torch.randint(3, actor.config.vocab_size, (GROUP_SIZE, SUFFIX_LENGTH),
                               generator=generator)
        stream = torch.randint(3, actor.config.vocab_size, (GROUP_SIZE, DECODE_STEPS + 1),
                               generator=generator)
        case, case_traces = run_case(actor, old_state, new_state, prompt, suffix, stream,
                                     args.repetitions)
        cases.append(case)
        traces.extend(case_traces)
        input_traces.append({"prompt_length": length, "prompt": prompt.tolist(),
                             "cached_suffix": suffix.tolist(), "boundary_and_forced_tokens": stream.tolist()})

    with (output / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=traces[0].keys())
        writer.writeheader()
        writer.writerows(traces)
    (output / "inputs.json").write_text(json.dumps(input_traces, indent=2) + "\n")
    report = {
        "machine": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin" else platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "attention_backend": actor.config._attn_implementation,
        "dtype": str(next(actor.parameters()).dtype),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "seed": args.seed,
        "parameters": parameter_count,
        "upstream_commit": "793e9b5b3d176ebb40161d8ba8d2125fe5153473",
        "source_commit": os.environ.get("NARL_SOURCE_COMMIT") or subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "snapshots": {
            "old": {"optimizer_steps": 0, "path": "model/model.safetensors",
                    "sha256": _digest(fixture / "model.safetensors")},
            "new": {"optimizer_steps": cfg.steps, "path": new_snapshot.name,
                    "sha256": _digest(new_snapshot)},
            "max_abs_weight_delta": max_weight_delta,
        },
        "warmups_per_mode": WARMUPS,
        "repetitions_per_mode": args.repetitions,
        "decode_steps": DECODE_STEPS,
        "tolerance": {"atol": ATOL, "rtol": RTOL, "forced_logprob_atol": ATOL},
        "timing_definition": "First token measures recovery and first decode to logits. "
                             "Old prefill, weight load, cache discard, mask setup and sampling "
                             "are outside this timer. Weight load is an in-memory state_dict copy.",
        "kl_direction": "KL(P_recompute || P_candidate), same teacher-forced history",
        "cases": cases,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(traces)} measured runs and alignment results to {output}")


if __name__ == "__main__":
    main()
