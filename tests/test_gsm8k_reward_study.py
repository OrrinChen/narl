from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from experiments.gsm8k_reward_study import digest, read, record, study


class RewardStudyTest(TestCase):
    def test_stops_negative_candidate_and_freezes_first_passing_checkpoints_before_confirmation(self):
        with TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "source", Path(tmp) / "out"
            source.mkdir()
            out.mkdir()
            names = ["outcome", "steps", "format"]
            record(source / "study.json", {"order": names})
            for name in names:
                (source / name).mkdir()
                record(source / name / "protocol.json", {"reward_recipe": name,
                                                         "study_sha256": digest(source / "study.json")})
            calls = []
            def run(folder, stage, seeds):
                calls.append((folder.name, stage, seeds))
                if stage == "q2":
                    selected = read(out / "selection.json")
                    self.assertEqual(selected["q1_gate_sha256"], digest(folder / "q1/gate.json"))
                    self.assertEqual(selected["endpoint_sha256"], {"seed": "frozen"})
                    (folder / "q2").mkdir()
                    record(folder / "q2/gate.json", {"passed": False})
                    return
                q1 = folder / "q1"
                q1.mkdir(exist_ok=True)
                record(q1 / "initial.json", {"correct": 46})
                for s in seeds:
                    score = 45 if folder.name == "steps" else 48
                    record(q1 / f"seed-{s}.json", {"correct": score})
                    record(q1 / f"strict-seed-{s}.json", {"correct": 46})
                if len(list(q1.glob("seed-*.json"))) == 5:
                    record(q1 / "gate.json", {"passed": folder.name == "format", "endpoint_sha256": {"seed": "frozen"}})

            with patch("experiments.gsm8k_reward_study.run_stage", side_effect=run):
                study(source, out)
                self.assertEqual(read(out / "steps/stop.json")["evaluated_seeds"], [4])
                self.assertEqual(read(out / "status.json"), {"selected": "format", "q2_scored": True, "stable": False})
                self.assertEqual(sum(stage == "q2" for _, stage, _ in calls), 1)
                n = len(calls)
                study(source, out)
                self.assertEqual(len(calls), n)
