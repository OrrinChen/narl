"""Checks for independent candidate keys, exact budget truncation and interrupted collection."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from experiments import gsm8k_decision_candidates as candidates
from experiments.gsm8k_reward_study import read


class Tokenizer:
    gsm8k_stop_ids = {0}
    chat_template = "test"

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def encode(self, text, **kwargs):
        return [len(text)]

    def decode(self, tokens, **kwargs):
        return "".join({1: "<answer>", 2: "5", 3: "</answer>", 0: ""}[token] for token in tokens)


class DecisionCandidateTest(TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.examples = [{"question": "2+3?", "value": "5"}, {"question": "3+3?", "value": "6"}]
        self.protocol = {"rng_base": 260112, "eval_sets": {"holdout": [0, 1]},
                         "seeds": [0, 1], "max_new_tokens": 8, "start_sha256": "frozen"}

    def fake(self, model, tokenizer, example, prompt_id, seed, slot, protocol):
        from experiments.gsm8k_signal import prompt_text

        tokens = [1, 2, 3, 0]
        text = tokenizer.decode(tokens)
        return {"seed": seed, "prompt_id": prompt_id, "slot": slot,
                "rng_seed": candidates.sampling_key(protocol, seed, prompt_id, slot),
                "token_cap": protocol["max_new_tokens"], "checkpoint_sha256": protocol["start_sha256"],
                "prompt_ids": tokenizer.encode(prompt_text(tokenizer, example, "reasoning")),
                "token_ids": tokens, "truncated": False, "behavior_logprobs": [-1.0, -2.0, -3.0, -4.0],
                "mean_logprob": -2.5, "token_elapsed_s": [.1, .2, .3, .4], "generation_s": .5,
                "completion": text, **candidates.answer_fields(text, example)}

    def test_budget_prefix_is_rescored_and_uses_only_arrived_tokens_and_time(self):
        example = self.examples[0]
        sample = self.fake(None, self.tok, example, 0, 0, 1, self.protocol)
        prefix = candidates.truncate_candidate(self.tok, example, sample, 2)
        self.assertEqual((prefix["completion"], prefix["strict_reward"], prefix["has_answer_tag"]),
                         ("<answer>5", 0, False))
        self.assertEqual((prefix["mean_logprob"], prefix["token_cap"], prefix["generation_s"]), (-1.5, 2, .2))
        self.assertTrue(prefix["truncated"])
        tagged = candidates.truncate_candidate(self.tok, example, sample, 3)
        self.assertEqual((tagged["strict_reward"], tagged["truncated"]), (1, True))
        exact = candidates.truncate_candidate(self.tok, example, sample, 4)
        self.assertEqual((exact["generation_s"], exact["token_cap"], exact["truncated"]), (.5, 4, False))
        self.assertEqual(sample["token_cap"], 8)
        with self.assertRaises(ValueError):
            candidates.truncate_candidate(self.tok, example, sample, 0)

    def test_rng_per_prompt_seed_and_slot_is_disjoint_and_replayable(self):
        keys = [candidates.sampling_key(self.protocol, seed, i, slot)
                for seed in self.protocol["seeds"] for i in (0, 1, 7472) for slot in (0, 1)]
        self.assertEqual(len(set(keys)), 12)
        self.assertEqual(candidates.sampling_key(self.protocol, 1, 17, 1), 280147)
        with self.assertRaises(ValueError):
            candidates.sampling_key(self.protocol, 0, 10_000, 0)

    def test_streamed_prefix_matches_actual_short_cap_call(self):
        import importlib.util
        if not importlib.util.find_spec("mlx_lm"):
            self.skipTest("optional Apple MLX dependency")

        class Stream:
            def __init__(self, *args, **kwargs):
                pass

            def insert(self, prompt_ids, cap):
                sequence = [1, 2, 3, 0][:cap]
                self.responses = iter([
                    [SimpleNamespace(token=token, logprobs=Mock(item=Mock(return_value=-float(i + 1))),
                                     finish_reason=("stop" if token == 0 else
                                                    "length" if i + 1 == cap else None))]
                    for i, token in enumerate(sequence)])

            def next(self):
                return next(self.responses, [])

            def close(self):
                pass

        args = SimpleNamespace(layers=[]), self.tok, self.examples[0], 0, 0, 1, self.protocol
        with patch("experiments.gsm8k_mlx_generate.FP32SamplingBatch", Stream):
            full = candidates.sample_candidate(*args)
            short = candidates.sample_candidate(*args, cap=2)
        prefix = candidates.truncate_candidate(self.tok, self.examples[0], full, 2)
        for field in short.keys() - {"generation_s", "token_elapsed_s"}:
            self.assertEqual(short[field], prefix[field], field)
        self.assertLessEqual(full["token_elapsed_s"][-1], full["generation_s"])

    def test_resume_trusts_only_completed_candidates_with_matching_bindings(self):
        with TemporaryDirectory() as name:
            folder = Path(name)
            calls = []

            def interrupt(*args):
                i, _, slot = args[3:6]
                calls.append((i, slot))
                if (i, slot) == (0, 1):
                    raise RuntimeError("interrupted mid-pair")
                return self.fake(*args)

            args = folder, self.protocol, "protocol-SHA", "holdout", 0, self.examples, None, self.tok
            with self.assertRaisesRegex(RuntimeError, "interrupted mid-pair"):
                candidates.collect_seed(*args, sample=interrupt)
            self.assertTrue((folder / "0-0.json").is_file())
            self.assertFalse((folder / "0-1.json").exists())
            retained = (folder / "0-0.json").read_bytes()
            with patch.object(self, "fake", wraps=self.fake) as sample:
                candidates.collect_seed(*args, sample=sample)
            self.assertEqual([(c.args[3], c.args[5]) for c in sample.call_args_list], [(0, 1), (1, 0), (1, 1)])
            self.assertEqual((folder / "0-0.json").read_bytes(), retained)
            row = read(folder / "0-1.json")
            self.assertEqual((row["strict_reward"], row["rng_seed"], row["protocol_sha256"]),
                             (1, 260113, "protocol-SHA"))
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                candidates.collect_seed(folder, self.protocol, "other", *args[3:], sample=self.fake)
