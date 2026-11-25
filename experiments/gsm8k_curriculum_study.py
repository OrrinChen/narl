"""Run the registered paired sampling study and its conditional budget comparison."""

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

from experiments.audit_kv_online import read_json
from experiments.gsm8k_curriculum_plan import OUT


def execute(out, module, *args):
    command = [sys.executable, "-u", "-m", f"experiments.{module}", "--out", str(out), *map(str, args)]
    print(module, *args, flush=True)
    subprocess.run(command, check=True)


def measure(out, phase, stage, arm=None, budget=None, seed=None):
    if stage == "train":
        cached = out / "models" / arm / str(budget) / f"seed-{seed}/training.json"
    else:
        name = "initial" if stage == "initial" else f"{arm}-{budget}-seed-{seed}"
        cached = out / "eval" / phase / (name + ".json")
    if cached.is_file():
        return
    args = ["--stage", stage, "--phase", phase]
    if arm is not None:
        args.extend(("--arm", arm, "--budget", budget, "--seed", seed))
    execute(out, "gsm8k_curriculum_run", *args)


def analyze(out, phase):
    execute(out, "gsm8k_curriculum_results", "--phase", phase)
    execute(out, "audit_gsm8k_curriculum", "--phase", phase, "--verify-training")


def study(out):
    execute(out, "gsm8k_curriculum_plan")
    protocol = read_json(out / "protocol.json")
    seeds = protocol["seeds"]
    short, long = protocol["budgets"]
    for seed in seeds:
        for arm in (("uniform", "mixed") if seed % 2 == 0 else ("mixed", "uniform")):
            measure(out, "primary", "train", arm, short, seed)
    measure(out, "primary", "initial")
    for seed in seeds:
        for arm in ("uniform", "mixed"):
            measure(out, "primary", "eval", arm, short, seed)
    analyze(out, "primary")
    if read_json(out / "trigger.json")["dose_authorized"]:
        for seed in seeds:
            measure(out, "dose", "train", "mixed", long, seed)
        measure(out, "dose", "initial")
        for seed in seeds:
            for budget in (short, long):
                measure(out, "dose", "eval", "mixed", budget, seed)
        analyze(out, "dose")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study(args.out)


if __name__ == "__main__":
    main()
