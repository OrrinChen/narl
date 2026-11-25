import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import gsm8k_curriculum_results as results


def store(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CurriculumResultsTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name)
        small_budgets = patch.multiple(results, SHORT=8, LONG=32)
        small_budgets.start()
        self.addCleanup(small_budgets.stop)
        self.ids = {"primary": list(range(10, 18)), "dose": list(range(20, 28))}
        labels = {str(s): ["uniform", "two_step"] * 8 for s in results.SEEDS}
        sched = {str(s): list(range(100, 116)) for s in results.SEEDS}
        self.plan = {"start_sha256": "start", "seeds": results.SEEDS, "budgets": [8, 32], "group_size": 2,
                     "lr": 1e-4, "accumulation_groups": 16, "eval_sets": self.ids,
                     "hard_eval_ids": {p: ids[::2] for p, ids in self.ids.items()}, "sampling_labels": labels,
                     "train_prompt_ids": {"uniform": sched, "mixed": sched}}
        path = self.run_dir / "protocol.json"
        store(path, self.plan)
        self.protocol_hash = results.digest(path)

    def model(self, arm, budget, seed):
        folder = self.run_dir / "models" / arm / str(budget) / f"seed-{seed}"
        binding = {"seed": seed, "protocol_sha256": self.protocol_hash, "arm": arm, "budget": budget,
                   "start_sha256": "start"}
        store(folder / "binding.json", binding)
        rows = []
        def group(lengths):
            idx = len(rows)
            samples = [{"reward": int(arm == "mixed" and slot == 1), "token_ids": [1] * n}
                       for slot, n in enumerate(lengths)]
            rows.append({"group_index": idx, "prompt_id": 100 + idx, "samples": samples,
                         "generated_tokens": sum(lengths)})
        group([2, 2])
        group([2, 1])
        group([1])
        for _ in range((budget - 8) // 4):
            group([2, 2])
        path = folder / "trace.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        store(folder / "training.json", {"start_sha256": "start", "final_sha256": f"{arm}{budget}{seed}",
              "seed": seed, "trace_sha256": results.digest(path), "lr": 1e-4, "accumulation_groups": 16,
              "accounting": {"generated_tokens": budget, "groups": len(rows), "supervised_tokens": 0}})

    def evaluation(self, phase, arm=None, budget=None, seed=None, correct=0):
        filename = "initial" if arm is None else f"{arm}-{budget}-seed-{seed}"
        checkpoint = "start" if arm is None else f"{arm}{budget}{seed}"
        rows = [{"prompt_id": i, "reward": int(pos < correct), "token_ids": [int(pos < correct) + 1]}
                for pos, i in enumerate(self.ids[phase])]
        store(self.run_dir / "eval" / phase / f"{filename}.json", {
              "protocol_sha256": self.protocol_hash, "checkpoint_sha256": checkpoint, "panel": phase,
              "arm": arm, "budget": budget, "seed": seed, "correct": correct, "rows": rows})

    def test_density_failure_of_accuracy_triggers_separate_dose_panel(self):
        self.evaluation("primary")
        for seed in results.SEEDS:
            for arm in ("uniform", "mixed"):
                self.model(arm, 8, seed)
                self.evaluation("primary", arm, 8, seed, correct=4)
        first = results.analyze(self.run_dir, "primary")
        self.assertFalse(first["passed"])
        self.assertTrue(first["criteria"]["density_gain"])
        self.assertEqual(first["endpoint_accounting"]["mixed-8"]["pooled"]["effective_groups"], 10)
        self.assertEqual(first["endpoint_accounting"]["mixed-8"]["pooled"]["generated_tokens"], 40)
        self.assertEqual(first["density"]["pooled_delta"], 2500)
        self.assertEqual(first["comparisons"]["candidate_vs_reference"]["hard"]["per_seed"]["1"]["paired_question_95"],
                         [0, 0])
        primary = self.run_dir / "results-primary.json"
        store(primary, first)
        decision = results.trigger_for(first, primary)
        self.assertTrue(decision["dose_authorized"])
        store(self.run_dir / "trigger.json", {**decision, "dose_authorized": False})
        with self.assertRaisesRegex(ValueError, "matching primary trigger"):
            results.analyze(self.run_dir, "dose")
        store(self.run_dir / "trigger.json", decision)
        self.evaluation("dose")
        for seed in results.SEEDS:
            self.model("mixed", 32, seed)
            self.evaluation("dose", "mixed", 8, seed, correct=0)
            self.evaluation("dose", "mixed", 32, seed, correct=8)
        second = results.analyze(self.run_dir, "dose")
        self.assertTrue(second["passed"])
        self.assertEqual(second["additional_generated_tokens"], dict.fromkeys(map(str, results.SEEDS), 24))
        for subset in ("overall", "hard"):
            contrast = second["comparisons"]["candidate_vs_reference"][subset]
            self.assertEqual(contrast["per_seed"]["0"]["paired_question_95"], [1, 1])
            self.assertEqual(contrast["median_accuracy_gain"], 1)
        rewritten = {**first, "passed": True}
        store(primary, rewritten)
        self.assertFalse(results.trigger_for(rewritten, primary)["dose_authorized"])
        with self.assertRaisesRegex(ValueError, "matching primary trigger"):
            results.analyze(self.run_dir, "dose")


if __name__ == "__main__":
    unittest.main()
