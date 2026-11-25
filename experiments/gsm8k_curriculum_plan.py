"""Freeze the prompt streams and independent panels for the GSM8K curriculum comparison."""

import argparse
from pathlib import Path
from random import Random

from experiments.audit_kv_online import read_json, read_rows, require
from experiments.gsm8k_reward_study import digest, record


RUNS = Path("runs")
OUT = RUNS / "curriculum-gsm8k-v1"
DRAW_SEED = 2026112523
SCHEDULE_LENGTH = 512
SEEDS = list(range(5))
REPLICATION_SHA = "4db00758919349b22e5becc58ddb8bf59254cf8490fc2e7edf8a492509c59ef4"


def annotations(row):
    return row["answer"].rsplit("####", 1)[0].count("<<")


def freeze(runs):
    prior = runs / "gsm8k-reward-format-replication"
    old = read_json(prior / "protocol.json")
    require(digest(prior / "protocol.json") == REPLICATION_SHA and read_json(prior / "audit.json")["audit_passed"],
            "previous exposure inventory or audit changed")
    base_path = runs / "gsm8k-stable-v1/plan.json"
    base = read_json(base_path)
    src = base["source_protocol"]
    data_path = runs / "gsm8k-data/train.jsonl"
    require(digest(data_path) == old["train_sha256"] == src["train_sha256"], "different training split")
    data = read_rows(data_path)
    excluded = set(old["excluded_ids"]).union(*old["panel_ids"].values())
    for path in (runs / "gsm8k-reward-composite-v1/verified_composite/models").glob("*/trace.jsonl"):
        require({r["prompt_id"] for r in read_rows(path)} <= excluded, "unaccounted composite training exposure")
    def question(i):
        return " ".join(data[i]["question"].casefold().split())
    seen = {question(i) for i in excluded}
    eligible = []
    for i in range(len(data)):
        key = question(i)
        if key not in seen:
            eligible.append(i)
            seen.add(key)
    reserved = Random(DRAW_SEED).sample(eligible, 512)
    eval_sets = {"primary": reserved[:256], "dose": reserved[256:]}
    hard = {k: [i for i in ids if annotations(data[i]) >= 4] for k, ids in eval_sets.items()}
    train = sorted(set(eligible) - set(reserved))
    schedules = {"uniform": {}, "mixed": {}}
    labels = {}
    for s in SEEDS:
        key = str(s)
        control = Random(DRAW_SEED + 100 + s).sample(train, SCHEDULE_LENGTH)
        flips = Random(DRAW_SEED + 200 + s)
        shared = {j + flips.randrange(2) for j in range(0, len(control), 2)}
        shared_ids = {control[j] for j in shared}
        target_pool = [i for i in train if i not in shared_ids and annotations(data[i]) == 2]
        target = iter(Random(DRAW_SEED + 300 + s).sample(target_pool, len(control) - len(shared)))
        mixed = [control[j] if j in shared else next(target) for j in range(len(control))]
        require(len(set(mixed)) == len(control), "repeated question in mixed stream")
        schedules["uniform"][key], schedules["mixed"][key] = control, mixed
        labels[key] = ["uniform" if j in shared else "two_step" for j in range(len(control))]

    scorer = read_json(runs / "gsm8k-reward-online-v2/parsed_outcome/protocol.json")
    hashes = {"gsm8k_training_reward.py": scorer["parser_base_sha256"],
              "gsm8k_training_reward_v2.py": scorer["reward_module_sha256"]}
    for name, expected in hashes.items():
        require(digest(Path(__file__).with_name(name)) == expected, "binary training reward changed")
    return {
        "study": "Matched sampling of uniformly drawn versus two-calculator-annotation-enriched prompts.",
        "source_protocol": src, "source_sha256": digest(base_path), "train_sha256": old["train_sha256"],
        "exposure_protocol_sha256": REPLICATION_SHA, "start_sha256": src["start_sha256"],
        "reward_source_sha256": hashes, "seeds": SEEDS, "budgets": [32768, 131072],
        "lr": 0.0001, "accumulation_groups": 16, "group_size": 2, "max_new_tokens": 256,
        "eval_batch": 12, "eval_max_new_tokens": 256, "seed_for_draw": DRAW_SEED,
        "excluded_ids": sorted(excluded), "train_pool_ids": train, "eval_sets": eval_sets, "hard_eval_ids": hard,
        "train_prompt_ids": schedules, "sampling_labels": labels, "eval_split": "GSM8K train",
        "design": "Shuffle eligible unique normalized question texts without replacement. Draw 256 primary and 256 dose questions in a single unstratified draw. Per seed uniformly draw 512 unique training IDs. For each adjacent pair, randomly choose one slot to be shared between arms. Fill the other mixed slots without replacement from all eligible two-annotation IDs excluding the shared slots. Uniform slots can themselves have two annotations. Use identical per-group sampling RNG keys. No evaluation labels, difficulty predictors, or extra pilot rollouts for selection.",
        "density": "Complete G2 groups with opposite v2 binary rewards per 10000 generated response tokens including all tails. Improvement requires >0 for at least 4/5 seed deltas and >0 in the pooled rate delta.",
        "accuracy": "Strict original greedy numeric match. Primary contrast mixed32 - uniform32. Also compare each to initialization. Stable accuracy gain requires at least four positive seeds, none negative, positive paired median and strictly positive lower 95% question-block bootstrap median. Primary success requires density improvement, stable gain versus uniform and initialization, and nonnegative median versus uniform on the >=4-annotation subgroup. The subgroup guard checks a point estimate only.",
        "dose": "Only if density improves but primary success fails, extend all five mixed32 policies to 131072 total generated response tokens, counting the existing prefix once, preserving its 32768 boundary flush, optimizer state, sampling stream and group indices. Evaluate the initial weights and both mixed checkpoints on the untouched dose panel. Stable dose benefit uses the same overall gates versus mixed32 and initialization and nonnegative hard-subgroup median versus mixed32. This phase has no uniform131 arm; attribute its result only to training longer on the fixed mixed recipe.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = freeze(args.runs)
    path = args.out / "protocol.json"
    record(path, protocol)
    print("Frozen", digest(path), "eligible training prompts", len(protocol["train_pool_ids"]),
          "hard panel sizes", {k: len(v) for k, v in protocol["hard_eval_ids"].items()})


if __name__ == "__main__":
    main()
