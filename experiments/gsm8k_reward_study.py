"""Run the frozen GSM8K reward candidates in order and confirm the first eligible one."""

import argparse
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def freeze(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Conflicting frozen record: {path}")
        return
    temp = path.with_suffix(".new")
    temp.write_bytes(data)
    temp.replace(path)


def record(path, values):
    if path.exists() and read(path) == values:
        return
    freeze(path, (json.dumps(values, indent=2) + "\n").encode())


def run_stage(folder, stage, seeds):
    command = [sys.executable, "-u", "-m", "experiments.gsm8k_reward_online",
               "--out", str(folder), "--protocol", str(folder / "protocol.json"),
               "--stage", stage, "--seeds", *map(str, seeds)]
    with (folder / "runner.log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def study(source, out, prior=None):
    study_path = out / "study.json"
    freeze(study_path, (source / "study.json").read_bytes())
    plan = read(study_path)
    order = plan["order"]
    if "source_prior_study_sha256" in plan:
        if prior is None or digest(prior / "study.json") != plan["source_prior_study_sha256"]:
            raise ValueError("Supplemental reward study requires its registered precursor")
        status = read(prior / "status.json")
        if status["stable"]:
            raise ValueError("Supplemental study did not trigger: the previous candidate was confirmed")
        for name in read(prior / "study.json")["order"]:
            previous = prior / name
            if (previous / "stop.json").is_file():
                continue
            decision = read(previous / "q1/gate.json")
            if decision["passed"]:
                if status["selected"] != name or not status["q2_scored"] or read(previous / "q2/gate.json")["passed"]:
                    raise ValueError("Previous confirmation is missing or successful")
                break
    for recipe in order:
        folder = out / recipe
        folder.mkdir(exist_ok=True)
        protocol = folder / "protocol.json"
        freeze(protocol, (source / recipe / "protocol.json").read_bytes())
        config = read(protocol)
        if config["study_sha256"] != digest(study_path) or config["reward_recipe"] != recipe:
            raise ValueError(f"Unbound candidate protocol: {protocol}")

    for index, recipe in enumerate(order):
        folder = out / recipe
        dev = folder / "q1"
        rejected = False
        for seed in plan.get("seed_order", (4, 0, 3, 1, 2)):
            if not all((dev / f"{prefix}seed-{seed}.json").exists() for prefix in ("", "strict-")):
                run_stage(folder, "both", [seed])
            score, start = (read(dev / file)["correct"] for file in (f"seed-{seed}.json", "initial.json"))
            print(recipe, f"Q1 seed {seed}: {score} (initial {start})", flush=True)
            early = index > 0 or recipe in plan.get("stop_on_negative", [])
            if early and score < start and not (dev / "gate.json").exists():
                seeds = sorted(int(p.stem.split("-")[-1]) for p in dev.glob("seed-*.json"))
                record(folder / "stop.json", {"reason": "negative Q1 seed", "seed": seed, "correct": score,
                                              "initial_correct": start, "evaluated_seeds": seeds})
                rejected = True
                break
        if rejected:
            continue
        gate = read(dev / "gate.json")
        if gate["passed"]:
            record(out / "selection.json", {"reward_recipe": recipe, "protocol_sha256": digest(folder / "protocol.json"),
                                           "q1_gate_sha256": digest(dev / "gate.json"), "endpoint_sha256": gate["endpoint_sha256"],
                                           "study_sha256": digest(study_path)})
            if not (folder / "q2/gate.json").exists():
                run_stage(folder, "q2", [0, 1, 2, 3, 4])
            record(out / "status.json", {"selected": recipe, "q2_scored": True,
                                         "stable": read(folder / "q2/gate.json")["passed"]})
            return
    record(out / "status.json", {"selected": None, "q2_scored": False, "stable": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/gsm8k-reward-online-v2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=Path("runs/gsm8k-reward-online-v2"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".study.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study(args.source, args.out, args.prior)


if __name__ == "__main__":
    main()
