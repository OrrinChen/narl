"""Measure shared prefix scaling on fixed token histories, on CPU or CUDA."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, __version__ as transformers_version

from experiments.kv_online import _scoring, _threads
from experiments.kv_replay import _forward, group_prefill


@torch.no_grad()
def actor(model, prompt, suffix, attention, share):
    group, length = suffix.shape
    prefix = prompt.size(1)
    out = group_prefill(model, prompt, attention[:1, :prefix], group, share)
    scores = [out.logits.expand(group, -1, -1)]
    for step in range(length - 1):
        out = _forward(model, suffix[:, step:step + 1], attention[:, :prefix + step + 1],
                       out.past_key_values)
        scores.append(out.logits)
    return torch.cat(scores, dim=1)


def measure_case(model, device, dtype, prefix, length, group, warmups, repetitions, seed):
    generator = torch.Generator().manual_seed(seed)
    vocab = model.config.vocab_size
    prompt = torch.randint(0, vocab, (1, prefix), generator=generator).to(device)
    suffix = torch.randint(0, vocab, (group, length), generator=generator).to(device)
    states = torch.cat((prompt.expand(group, -1), suffix), dim=1)
    attention = torch.ones_like(states)
    mask = torch.cat((torch.zeros_like(attention[:, :prefix]), attention[:, prefix:]), dim=1)
    batch = SimpleNamespace(states=states, mask=mask)
    weights = torch.randn((group, length), generator=generator).to(device)
    parameter = model.get_input_embeddings().weight
    scorers = []
    for share in (False, True):
        policy = SimpleNamespace(model=model)
        _scoring(policy, share_learner=share, suffix_logits_only=True, prompt_tokens=prefix, first_content_id=0)
        scorers.append(policy.logprobs)

    def learner(index):
        model.zero_grad(set_to_none=True)
        scores = scorers[index](batch)[:, prefix:]
        (scores.float() * weights).mean().backward()
        return scores.detach()

    def sampling(index):
        return actor(model, prompt, suffix, attention, bool(index))

    # CUDA operations are asynchronous, so bracket each sample with synchronization.
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    fp32 = dtype == torch.float32
    errors = {}
    for name, run in (("actor", sampling), ("learner", learner)):
        reference = run(0).float()
        if name == "learner":
            # Untouched rows are also covered on CPU. Sample used input rows for
            # large vocabularies to avoid copying the entire CUDA embedding grad.
            grad_rows = states.unique() if device.type == "cuda" else None
            grad = (parameter.grad.detach().index_select(0, grad_rows).float() if grad_rows is not None
                    else parameter.grad.detach().clone().float())
        candidate = run(1).float()
        atol, rtol = (2e-5, 2e-5) if fp32 else (0.25, 0.05)
        torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
        errors[f"{name}_max_abs_error"] = float((reference - candidate).abs().max())
        if name == "learner":
            grad_candidate = (parameter.grad.detach().index_select(0, grad_rows).float() if grad_rows is not None
                              else parameter.grad.detach().float())
            torch.testing.assert_close(grad_candidate, grad, atol=2e-5 if fp32 else 0.03,
                                       rtol=1e-4 if fp32 else 0.10)
            errors["embedding_grad_max_abs_error"] = float((grad - grad_candidate).abs().max())
        del reference, candidate
        if name == "learner":
            del grad, grad_candidate

    rows = []
    for phase, run in (("actor", sampling), ("learner", learner)):
        for index in range(warmups + repetitions):
            for mode in ((0, 1) if index % 2 == 0 else (1, 0)):
                synchronize()
                start = perf_counter()
                run(mode)
                synchronize()
                elapsed = (perf_counter() - start) * 1000
                if index >= warmups:
                    rows.append({"prompt_tokens": prefix, "output_tokens": length, "phase": phase,
                                 "repeat": index - warmups, "mode": "shared" if mode else "dense",
                                 "ms": elapsed})
    model.zero_grad(set_to_none=True)
    summary = {"prompt_tokens": prefix, "output_tokens": length, **errors}
    for phase in ("actor", "learner"):
        dense, shared = (median(row["ms"] for row in rows if row["phase"] == phase and row["mode"] == mode)
                         for mode in ("dense", "shared"))
        summary.update({f"{phase}_dense_ms": dense, f"{phase}_shared_ms": shared,
                        f"{phase}_speedup": dense / shared})
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--model", help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--revision", help="model commit for a remote checkpoint")
    parser.add_argument("--prompts", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 384])
    parser.add_argument("--outputs", type=int, nargs="+", default=[3, 16, 64, 128])
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=9)
    args = parser.parse_args()
    if min(*args.prompts, *args.outputs, args.group_size, args.repetitions) < 1 or args.warmups < 0:
        parser.error("token lengths, group size and repetitions must be positive")
    if len(set(args.prompts)) != len(args.prompts) or len(set(args.outputs)) != len(args.outputs):
        parser.error("prompt and output lengths must each be unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot see a CUDA device")
    if args.revision and not args.model:
        parser.error("--revision requires --model")
    _threads()
    torch.manual_seed(0)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype,
                                                   attn_implementation="sdpa")
    else:
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=16, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
            attn_implementation="sdpa",
        ))
    model.to(device=device, dtype=dtype).eval()
    if max(args.prompts) + max(args.outputs) > model.config.max_position_embeddings:
        parser.error("requested history exceeds the model context")
    out = args.out
    out.mkdir(parents=True)
    source_files = ("experiments/kv_length_sweep.py", "experiments/kv_replay.py",
                    "experiments/kv_online.py", "experiments/shared_prefix.py")
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True)
    cpu = (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
           if sys.platform == "darwin" else platform.processor())
    protocol = {"args": {**vars(args), "out": str(out)}, "device": str(device), "platform": platform.platform(),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else cpu,
                "parameters": sum(p.numel() for p in model.parameters()), "torch": torch.__version__,
                "transformers": transformers_version, "threads": torch.get_num_threads(),
                "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "source_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                                  for name in source_files},
                "attention_backend": model.config._attn_implementation,
                "actor": "prefill and teacher-forced cached decode, full logits, fixed weights",
                "learner": "suffix logprobs and backward, no optimizer step",
                "embedding_grad_check": "all input-token rows on CUDA; all rows on CPU"}
    if device.type == "cuda":
        protocol["cuda"] = torch.version.cuda
        protocol["cuda_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary = []
    with (out / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ("prompt_tokens", "output_tokens", "phase", "repeat", "mode", "ms"),
                                lineterminator="\n")
        writer.writeheader()
        for prefix in args.prompts:
            for length in args.outputs:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                rows, result = measure_case(model, device, dtype, prefix, length, args.group_size,
                                            args.warmups, args.repetitions, seed=100_000 * prefix + length)
                if device.type == "cuda":
                    result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                writer.writerows(rows)
                handle.flush()
                summary.append(result)
                (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
