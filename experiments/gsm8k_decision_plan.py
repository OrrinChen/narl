"""Freeze prompt-disjoint training and inference-policy comparisons on GSM8K."""

import argparse
import json
from pathlib import Path
from random import Random

from experiments.gsm8k_reward_study import digest, read, record


OUT = Path("runs/gsm8k-decision-v1")
PRIOR = Path("runs/curriculum-gsm8k-v1")
SPLIT_SEED = 2026112601


def freeze(prior=PRIOR):
    previous = read(prior / "protocol.json")
    if digest(prior / "protocol.json") != "e4c1029d3102aee7db7a336701d19775ab46ced6503d1bc22bfa7efcee354fc6":
        raise ValueError("unexpected exposure inventory")
    audited = read(prior / "audit-dose.json")
    if not audited["audit_passed"]:
        raise ValueError("previous trajectories were not audited")
    source = previous["source_protocol"]
    seen, history, bindings = set(), set(), {}
    for path in sorted((prior / "models").glob("*/*/seed-*/trace.jsonl")):
        ids = {json.loads(line)["prompt_id"] for line in path.read_bytes().splitlines()}
        seen.update(ids)
        if path.parents[1].name == "131072" and path.parents[2].name == "mixed":
            history.update(ids)
            bindings[str(path.relative_to(prior))] = digest(path)
    pool = set(previous["train_pool_ids"])
    if len(bindings) != 5 or not seen <= pool:
        raise ValueError("incomplete or unexpected prior trajectories")
    unseen = sorted(pool - seen)
    shuffled = Random(SPLIT_SEED).sample(sorted(history), len(history))
    holdout = Random(SPLIT_SEED + 1).sample(unseen, 256)
    return {
        "study": "Frozen answer correctness head and fixed-budget retry allocation",
        "model": source["model"], "revision": source["revision"], "start_sha256": source["start_sha256"],
        "train_sha256": source["train_sha256"], "mlx": source["mlx"], "mlx_lm": source["mlx_lm"],
        "prior_sha256": digest(prior / "protocol.json"), "trace_sha256": bindings,
        "hist_train_ids": shuffled[179:], "hist_dev_ids": shuffled[:179], "eligible_new_count": len(unseen),
        "eval_sets": {"holdout": holdout}, "split_seed": SPLIT_SEED,
        "seeds": list(range(5)), "rng_base": 260112, "max_new_tokens": 256,
        "uniform_seed": 261126, "bootstrap_seed": 80161, "retry_tokens_per_question": 64,
        "l2_grid": [0.03, 0.3, 3.0], "clock_ids": holdout[:32], "clock_seed": 0,
        "clock_order": Random(SPLIT_SEED + 2).sample(["uniform", "confidence", "judge"], 3),
        "generator": "Use common 80-example SFT adapter, no further generator training. Temperature 1 full categorical. Each question, seed, slot has a separate reproducible RNG key; each candidate runs alone. Slot 0 is the primary answer and slot 1 the retry.",
        "label": "Frozen strict numeric GSM8K scorer. No labels, reference solutions or reward fields may enter features or policy decisions.",
        "head": "Frozen final-layer Qwen representation of question and candidate, complete answer tag, mean sampled-token log probability. Logistic ridge fitted on tagged historical training candidates only. Untagged candidates have probability zero. Choose l2 from the frozen grid by minimum equal-question-weighted historical development Brier score, ties to larger l2. Freeze this head before generating any held-out answers.",
        "selection": "Same two candidates per seed and question. Compare learned probability with the cheap tag-then-mean-logprob rule, first answer, uniform random expectation and any-correct oracle. Both selectors keep the first answer on ties. Primary contrast learned vs cheap. Report exact match, first-to-final rescues and regressions, paired question-block uncertainty across seeds, calibration and cost.",
        "allocation": "First answer for all prompts. All arms then spend exactly 64*n additional response tokens. Repeated generation is limited to one additional answer per prompt. Spend final remaining tokens on a capped answer and strictly re-score its visible prefix. Random order vs lowest first-answer mean logprob vs lowest learned probability of first-answer correctness. Ties use the same seeded permutation. After retry the same cheap selection rule applies to all arms. Primary contrast judge vs confidence, also compare judge vs uniform. Low correctness probability is not a trained marginal-gain forecast.",
        "timing": "Replay uses measured per-candidate generation costs and synchronized time-to-token for partial tails. Separately run all 3 allocation policies live, sequentially at batch size one, on the 32 preregistered prompts, seed 0. Shared warm-up, loaded weights, exclusive wall clock including judging, ordering and generating. Record live parity against the same 32-question trace replay. Measure isolated decision-scoring cost in the ranking comparison. Report costs for pooled candidate collection separately.",
        "interpretation": "Use question-cluster paired 95% bootstrap intervals with common resamples across the five fixed seeds; independent units are questions. Report positive, tied and negative per-seed deltas separately. A stable gain for each primary contrast requires a positive lower bound, >=4 positive seed deltas and no negative seed. The two primary contrasts are distinct and both will be reported. Do not score the official test split.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record(args.out / "protocol.json", freeze())
    print("Frozen", digest(args.out / "protocol.json"))


if __name__ == "__main__":
    main()
