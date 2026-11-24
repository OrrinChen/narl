import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.audit_gsm8k_reward_online import OUTCOME_V3, validate_recipe, validate_selection, validate_study
from experiments.audit_gsm8k_stable import digest
from experiments.audit_kv_online import read_json
from experiments import gsm8k_training_reward_v3


RUNS = Path(__file__).resolve().parents[1] / "runs"
V1 = RUNS / "gsm8k-reward-online-v1"
V2 = RUNS / "gsm8k-reward-online-v2"


def binary_v3():
    protocol = read_json(V2 / "parsed_outcome/protocol.json")
    return {**protocol, "score_module": OUTCOME_V3,
            "reward_module_sha256": digest(Path(gsm8k_training_reward_v3.__file__))}


class RewardProtocolTest(unittest.TestCase):
    def test_existing_pilot_and_all_three_v2_recipes(self):
        sample = {"completion": "<answer>#100</answer>", "truncated": False}
        for folder in (V1 / "parsed_outcome", V2 / "parsed_outcome", V2 / "verified_steps", V2 / "verified_format"):
            # Each archived protocol pins both its study and its scoring implementation.
            path = folder / "protocol.json"
            protocol = read_json(path)
            validate_study(protocol, path.parent.parent / "study.json")
            scorer = validate_recipe(protocol)
            if protocol["reward_recipe"] == "parsed_outcome":
                self.assertEqual(scorer(sample, "100")[0], 0)
            else:
                expected = 0.8 if protocol["reward_recipe"] == "verified_format" else 1.0
                self.assertEqual(scorer({**sample, "completion": "The answer is 100."},
                                        {"value": "100", "rationale": "#### 100"})[0], expected)
            binding = path.with_name("binding.json")
            if binding.is_file():
                self.assertEqual(read_json(binding)["protocol_sha256"], digest(path))

    def test_v3_requires_the_three_hashes_and_the_binary_recipe(self):
        protocol = binary_v3()
        scorer = validate_recipe(protocol)
        self.assertEqual(scorer({"completion": "<answer>#100</answer>", "truncated": False}, "100")[0], 1)
        for field in ("parser_base_sha256", "outcome_module_sha256", "reward_module_sha256"):
            for corrupted in ({**protocol, field: "0" * 64}, {k: v for k, v in protocol.items() if k != field}):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_recipe(corrupted)
        for recipe in ("verified_steps", "verified_format"):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                validate_recipe({**protocol, "reward_recipe": recipe})

    def test_v3_requires_a_separate_study(self):
        protocol = binary_v3()
        with self.assertRaises(ValueError):
            validate_study(protocol, V2 / "study.json")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "study.json"
            study = {"base_plan_sha256": protocol["base_plan_sha256"], "order": ["parsed_outcome"],
                     "score_module": OUTCOME_V3}
            path.write_text(json.dumps(study))
            protocol["study_sha256"] = digest(path)
            self.assertEqual(validate_study(protocol, path), study)
            with self.assertRaises(ValueError):
                validate_study({**read_json(V2 / "parsed_outcome/protocol.json"),
                                "study_sha256": digest(path)}, path)

    def test_selection_pins_the_winner_before_q2(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "parsed_outcome"
            (run / "q1").mkdir(parents=True)
            (root / "study.json").write_text("{}")
            gate = {"protocol_sha256": "protocol", "endpoint_sha256": {"0": "checkpoint"}}
            (run / "q1/gate.json").write_text(json.dumps(gate))
            selection = {**gate, "reward_recipe": run.name, "q1_gate_sha256": digest(run / "q1/gate.json"),
                         "study_sha256": digest(root / "study.json")}
            path = root / "selection.json"
            path.write_text(json.dumps(selection))
            validate_selection(run, run.name, gate)
            for field in selection:
                path.write_text(json.dumps({**selection, field: "stale"}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_selection(run, run.name, gate)
