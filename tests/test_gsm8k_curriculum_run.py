"""Regression tests for continuing a curriculum endpoint across the budget boundary."""

from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch


class Interrupted(Exception):
    pass


class StatefulLearner:
    """A dropped optimizer step counter changes all subsequent model bytes."""

    def __init__(self, model, lr):
        self.model = model
        self.step = 0

    def step_groups(self, groups):
        self.step += 1
        self.model.weight += self.step * sum(prompt[0] for prompt, _, _ in groups)
        return {"step": self.step, "weight": self.model.weight}


@skipUnless(all(find_spec(name) for name in ("mlx", "mlx_lm", "peft")), "optional MLX dependencies")
class CurriculumRecoveryTest(TestCase):
    def test_resume_replays_optimizer_and_preserves_short_singleton(self):
        from experiments import gsm8k_curriculum_run as curriculum, gsm8k_mlx_development, gsm8k_stable
        from experiments.audit_kv_online import read_rows

        model = SimpleNamespace(weight=0.0)
        schedule = list(range(10))
        examples = [{"value": "1"} for _ in schedule]

        def snapshot(model):
            return {"weight": torch.tensor([model.weight])}

        def set_weight(model, checkpoint):
            model.weight = checkpoint["weight"].item()

        start = snapshot(model)
        start_hash = gsm8k_stable.adapter_digest(start)

        def train(folder, budget, *, prefix=None, stop_at=None):
            folder.mkdir(exist_ok=True)
            seen = []

            def generate(model, tokenizer, example, protocol, seed, arm, i, remaining, old, current, version):
                if i == stop_at:
                    raise Interrupted
                self.assertEqual(current[0]["weight"].item(), model.weight)
                seen.append(i)
                samples = [{"token_ids": [1], "behavior_logprobs": [-0.5], "truncated": False, "reward": slot}
                           for slot in range(min(remaining, 2))]
                return [i + 1], samples

            def restore(model, learner):
                return curriculum.restore_batches(model, learner, folder, prefix, schedule, start_hash)

            plan = {"source_protocol": {"start_sha256": start_hash}, "train_prompt_ids": {"0": schedule},
                    "group_size": 2, "token_budget": budget}
            with patch.object(gsm8k_stable, "rollouts", generate):
                report = gsm8k_stable.train_case(model, None, examples, plan, start, 0.001, 2, 0, folder,
                                                 restore_training=restore)
            self.assertEqual(report["accounting"]["generated_tokens"], budget)
            return seen

        with TemporaryDirectory() as temp, ExitStack() as patches:
            for module, name, value in ((gsm8k_stable, "MlxGrpoLearner", StatefulLearner),
                                        (gsm8k_stable, "set_adapter", set_weight),
                                        (gsm8k_stable, "adapter_snapshot", snapshot),
                                        (gsm8k_mlx_development, "adapter_snapshot", snapshot),
                                        (curriculum, "SHORT", 5)):
                patches.enter_context(patch.object(module, name, value))
            short, reference, resumed = (Path(temp) / name for name in ("short", "reference", "resumed"))
            self.assertEqual(train(short, 5), [0, 1, 2])
            self.assertEqual(len(read_rows(short / "trace.jsonl")[-1]["samples"]), 1)
            self.assertEqual(train(reference, 14, prefix=short), [3, 4, 5, 6, 7])
            for name in ("trace.jsonl", "updates.jsonl"):
                self.assertTrue((reference / name).read_bytes().startswith((short / name).read_bytes()))
            updates = read_rows(reference / "updates.jsonl")
            self.assertEqual([u["diagnostics"]["step"] for u in updates], [1, 2, 3])
            self.assertEqual([u["diagnostics"]["weight"] for u in updates], [3.0, 21.0, 60.0])

            with self.assertRaises(Interrupted):
                train(resumed, 14, prefix=short, stop_at=6)
            for name in ("trace.jsonl", "updates.jsonl"):
                with (resumed / name).open("ab") as handle:
                    handle.write(b'{"partial":')
            self.assertEqual(train(resumed, 14, prefix=short), [5, 6, 7])
            for name in ("trace.jsonl", "updates.jsonl", "final-adapter.safetensors"):
                self.assertEqual((resumed / name).read_bytes(), (reference / name).read_bytes())

    def test_strict_evaluation_resumes_at_batch_boundary(self):
        from experiments import gsm8k_budget, gsm8k_curriculum_run as curriculum

        completion = "<answer>1</answer>"
        ids = list(range(27))
        examples = [{"question": str(i), "value": "1"} for i in ids]
        tokenizer = SimpleNamespace(decode=lambda token_ids, skip_special_tokens: completion)
        offsets = []

        def generate(model, tokenizer, train, ids, source):
            offsets.append(ids[0])
            if len(offsets) == 2:
                raise Interrupted
            rows = [{"prompt_id": i, "token_ids": [0, 151645], "completion": completion, "has_answer_tag": True,
                     "reward": 1, "truncated": False} for i in ids]
            return {"rows": rows, "correct": len(rows), "tagged": len(rows), "truncated": 0}

        with TemporaryDirectory() as temp, patch.object(gsm8k_budget, "evaluate", generate):
            args = SimpleNamespace(out=Path(temp), phase="primary", stage="initial")
            binding = {"protocol_sha256": "protocol", "checkpoint_sha256": "checkpoint", "panel": "primary",
                       "arm": None, "seed": None, "budget": None}

            def evaluate():
                return curriculum.evaluate_cached(args, None, tokenizer, examples, {"eval_batch": 12}, ids, binding)

            with self.assertRaises(Interrupted):
                evaluate()
            self.assertEqual(evaluate()["correct"], 27)
            self.assertEqual(evaluate()["correct"], 27)
            self.assertEqual(offsets, [0, 12, 12, 24])
            binding["checkpoint_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "binding"):
                evaluate()
