import copy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments import audit_gsm8k_decision as audit
from experiments.gsm8k_decision_eval import evaluate


class ToyTokenizer:
    def decode(self, tokens, *, skip_special_tokens):
        return "working" + ("<answer>1</answer>" if tokens[-1] == 91 else "<answer>0</answer>" if tokens[-1] == 90 else "")


def artificial_sample(length, y, logp, p):
    return {"token_ids": [30]*(length-1) + [90+y], "strict_reward": y, "has_answer_tag": True,
            "mean_logprob": logp, "behavior_logprobs": [logp]*length, "token_elapsed_s": list(range(1, length+1)),
            "generation_s": length+0.1, "decision_score": p, "decision_s": 0.1}


def write(path, body):
    path.write_text(json.dumps(body))


class IndependentDecisionAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.p = {"eval_sets": {"holdout": [2, 3, 1, 0]}, "seeds": list(range(5)), "uniform_seed": 261126,
                  "clock_seed": 0, "clock_ids": [3, 1], "clock_order": ["judge", "confidence", "uniform"],
                  "retry_tokens_per_question": 64}
        self.train = [{"value": "1"} for _ in range(4)]
        self.tok = ToyTokenizer()
        self.pools = {}
        records, labels, probabilities = [], [], []
        for seed in self.p["seeds"]:
            pool = {}
            for q in self.p["eval_sets"]["holdout"]:
                first = artificial_sample(60, (seed + q) % 2, -0.25 * (q+1), (q+1)/10)
                retry = artificial_sample(101, 1-first["strict_reward"], -0.4, (4-q)/10)
                pool[q] = [first, retry]
                records.append({"prompt_id": q, "seed": seed, "samples": [first, retry]})
                labels += [s["strict_reward"] for s in pool[q]]
                probabilities += [s["decision_score"] for s in pool[q]]
            self.pools[seed] = pool

        def cut(row, sample, n):
            return {**sample, "token_ids": sample["token_ids"][:n], "strict_reward": 0,
                    "has_answer_tag": False, "mean_logprob": sample["mean_logprob"], "generation_s": n}

        def report(rows):
            return evaluate(rows, 64 * len(set(x["prompt_id"] for x in rows)), cut, self.p["uniform_seed"])

        details = report(records)
        p, y = np.array(probabilities), np.array(labels)
        details.update({"head_sha256": "head", "protocol_sha256": audit.FROZEN,
                        "calibration": {"n": len(y), "average_probability": p.mean(), "fraction_correct": y.mean(),
                                        "brier": np.mean((p-y)**2)}})
        for k, branch in (("judge_vs_rule", "fixed_pair"), ("judge_vs_confidence", "allocation")):
            info = details[branch + "_contrasts"][k]
            details.setdefault("gate", {})[k] = bool(info["question_block_median_95"][0] > 0 and
                                                     info["negative_seeds"] == 0 and info["positive_seeds"] >= 4)
        self.pristine = details
        write(self.run / "results.json", details)
        write(self.run / "score-check.json", {})
        subset = [r for r in records if r["seed"] == 0 and r["prompt_id"] in self.p["clock_ids"]]
        live = report(subset)["per_seed"]["0"]
        self.live_files = {}
        for arm in audit.ARMS:
            clocks = {q["prompt_id"]: q["allocation"][arm] for q in live["questions"]}
            qs = []
            for q in self.p["clock_ids"]:
                row = clocks[q]
                a, b = self.pools[0][q]
                n = row["tokens"]
                hash_tokens = lambda ids: sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                qs.append({"prompt_id": q, "retry_tokens": n, "selected_slot": row["selected_slot"],
                           "correct": row["correct"], "first_tokens_sha256": hash_tokens(a["token_ids"]),
                           "retry_tokens_sha256": hash_tokens(b["token_ids"][:n]) if n else None})
            summary = live["allocation"][arm]
            self.live_files[arm] = {**{k: summary[k] for k in ("first_tokens", "retry_tokens", "total_tokens", "correct")},
                                    "policy": arm, "seed": 0, "protocol_sha256": audit.FROZEN, "head_sha256": "head",
                                    "wall_seconds": 10, "judge_seconds": 1 if arm == "judge" else 0,
                                    "replay_verified": True, "questions": qs}
            write(self.run / ("live-" + arm + ".json"), self.live_files[arm])

    def check(self):
        return audit.check_results(self.run, self.p, self.pools, self.train, self.tok, "head")

    def test_full_report_then_changed_interval(self):
        self.assertEqual(self.check()["candidate_answers"], 40)
        report = copy.deepcopy(self.pristine)
        report["fixed_pair_contrasts"]["judge_vs_rule"]["question_block_median_95"] = [1.5, 1.5]
        write(self.run / "results.json", report)
        with self.assertRaisesRegex(ValueError, "question_block_median_95"):
            self.check()

    def test_capped_candidate_cannot_inherit_full_reward(self):
        report = copy.deepcopy(self.pristine)
        dd = report["per_seed"]["0"]
        q = dd["allocation"]["judge"]["partial_retry_question"]
        row = next(r for r in dd["questions"] if r["prompt_id"] == q)["allocation"]["judge"]
        row["retry_correct"] = 1
        write(self.run / "results.json", report)
        with self.assertRaisesRegex(ValueError, "retry_correct"):
            self.check()

    def test_live_prefix_hash_must_match_capped_replay(self):
        report = self.live_files["judge"]
        report["questions"][0]["first_tokens_sha256"] = "lost"
        write(self.run / "live-judge.json", report)
        with self.assertRaisesRegex(ValueError, "first_tokens_sha256"):
            self.check()
