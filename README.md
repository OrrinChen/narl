# Narl

Narl studies weight updates, KV-cache recomputation, and group composition in asynchronous GRPO. It extends [nanoRL at `793e9b5`](https://github.com/alex000kim/nanoRL/tree/793e9b5b3d176ebb40161d8ba8d2125fe5153473) under its original MIT license. The upstream instructions follow the Narl quickstart.

## CPU baseline

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-cpu.txt
.venv/bin/python tests/test_narl_baseline.py
.venv/bin/python experiments/cpu_baseline.py --out runs/cpu-baseline-repro
```

Use a new output directory for each run. The smoke run creates a 19,424-parameter Llama decoder and a tokenizer locally, then runs 12 synchronous and 12 asynchronous GRPO steps on a synthetic token-choice task. Both modes start from the same saved weights. The asynchronous actor and learner run as separate processes and communicate over loopback. The run directory contains its configuration, model fixture, role logs, per-step CSVs, and summary JSON.

The contract checks use a fixed batch to compare the zero-lag optimizer update with the synchronous update and verify rejection of an entire group that expires in the queue. The smoke run checks that each role trains and reports consumed versions and separate stale/overflow counts. Its elapsed time includes process startup and evaluation. `accepted_tokens` counts completion tokens admitted into training, including groups with zero advantage.

## CPU KV replay

```bash
OMP_NUM_THREADS=1 .venv/bin/python tests/test_narl_kv_replay.py
.venv/bin/python -m experiments.kv_replay --out runs/kv-replay-repro
```

The replay uses the initial tiny decoder and its snapshot after 12 synchronous GRPO steps. It feeds the same synthetic token history through three paths across a single weight switch. The old cache contains the prompt and 16 branch tokens. The boundary token was sampled before the switch and is fed at the first new-weight decode step. Both recompute paths rebuild the same history under the new weights. The shared path pre-fills the prompt once and physically copies its cache into eight branches before rebuilding their distinct suffixes.

`summary.json` records alignment checks, KL against per-completion recomputation, and timing medians and ranges. `timings.csv` contains each measured repetition. `inputs.json`, the model snapshots, and the snapshot-training configuration and CSV record the workload and weights. Recovery-to-first-logits timing includes recompute and branch copying, with in-memory weight loading recorded separately. The replay measures fixed-history conditional distributions and CPU latency on the tiny model.

## CPU regrouping and online validation

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.regroup --out runs/regroup-repro
.venv/bin/python tests/test_narl_kv_online.py
.venv/bin/python -m experiments.kv_online --out runs/online-repro
.venv/bin/python -m experiments.audit_kv_online --run runs/online-repro --out runs/online-repro/audit.json
```

Regrouping compares the same completions under two group assignments at a fixed learner snapshot. It saves checkpoint hashes, sampled tokens and log probabilities, group assignments, and gradient diagnostics.

The online arm uses a 19,424-parameter Llama with FP32 weights, one actor and one learner, one PyTorch intra-op and inter-op thread per process, and the same fixed initial model in each run. The synthetic token-choice split uses the first 16 of 80 prompts for evaluation. Training prompts are selected from the remaining 64 in the same seeded order in both arms. All training prompts are 128 tokens after adding deterministic filler. Groups have eight completions and alternate 16, 24, and 32 generated tokens. Reward uses the last token. Evaluation greedily generates 16 tokens for each held-out prompt. The queue holds two complete groups with producer backpressure, the actor can read one published snapshot, and publication follows each admitted batch. The learner checks lag against the oldest behavior version in the group. Each arm processes 96 generated groups per seed with the same token budget, including dropped groups.

The actor samples freely and checks for a newer snapshot every four tokens. At most one in-flight swap per group is allowed. Both arms rebuild the history under the new snapshot and differ in whether they share the prompt prefill. Their chronological schedules and realized swaps can differ. The clipped loss uses the probabilities recorded at sampling time, including those for old-weight tokens. The saved per-token versions and group traces feed the audit and throughput report.

Separate stress and sensitivity runs reuse the online entry point:

```bash
.venv/bin/python -m experiments.kv_online --out runs/stress-repro --groups 18 --seeds 0 --stress-delay-s 4
.venv/bin/python -m experiments.kv_online --out runs/sensitivity-repro --groups 48 --seeds 0 --max-lag 2
```

## Rebuild the figures and CPU profile

Run the plotting modules with a Python environment that includes Matplotlib. Pass an unused output file to the profiling command.

```bash
.venv/bin/python -m experiments.kv_profile --run runs/kv-replay-repro --out runs/kv-replay-repro/profile.json
python3 -m experiments.plot_kv_replay --summary runs/kv-replay-repro/summary.json --out runs/kv-replay-repro/replay.png
python3 -m experiments.plot_regroup --run runs/regroup-repro --out runs/regroup-repro/regroup.png
python3 -m experiments.plot_kv_online --run runs/online-repro --out runs/online-repro/online.png
```

## Local Kubernetes Job

With a local Docker runtime, kind, and kubectl available, run `bash k8s/run-kv-replay.sh`. The runner creates a temporary loopback kind cluster and runs the same KV replay entry point in one Pod with five repetitions. It saves the Job and Pod status, logs, image ID, dependencies, and the Pod's replay artifacts under a new `runs/k8s-kv-replay.*` directory.

---

# nanoRL

One RL training loop that scales from a laptop CPU to a GPU cluster. **No GPU required**: the
same disaggregated trainer/worker setup that runs on 16 GPUs runs as two pods on your laptop.
~2,500 lines across 7 files, no Ray, TRL or DeepSpeed. Like
[nanoGPT](https://github.com/karpathy/nanoGPT), a codebase to fork, not a library to import.

```python
loss = -(advantage * logprob).mean()      # + PPO ratio clip, + optional KL
```

REINFORCE, PPO, GRPO and RLOO differ only in how `advantage(...)` is computed
([algos.py](algos.py)); sync vs async only in where batches come from.

![The training loop](assets/loop.svg)

## Quickstart, on CPU

```bash
uv pip install torch gymnasium numpy transformers "datasets<4" peft pyyaml

python train.py --task cartpole --algo reinforce        # ~10 s
python train.py --config configs/cartpole_ppo.yaml      # ~20 s
python train.py --config configs/ifeval_local_cpu.yaml  # Qwen3-0.6B + LoRA, ~86 min
```

The last one runs GRPO on real prompts from
[RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval). Each prompt carries three
formatting constraints, such as "around 40 words" or "never use the words the or and", and a
couple of lines of Python check them. There is no reward model and no LLM judge.

![Local CPU run](assets/local_run.png)

Held-out score goes from 0.448 to 0.604. On a matched before/after over the same 64 held-out
prompts, strict accuracy (all three constraints, which is what IFEval itself reports) goes
from 0.109 to 0.250. The hard constraints move too. "Never use the words the or and" improves
by 0.33, and satisfying it means rewriting the sentence. Four of the nine constraints do not
move at all.

Per-step train reward is left off the plot. At `n_prompts: 2` it mostly reflects which two
prompts were drawn, and it drifts down over the run while the held-out score rises.

Three settings drive that result, and getting any of them wrong flattens the curve:

- Three constraints per prompt, scored as a fraction. With a single binary constraint, 58% of
  groups score identically, because a 0.6B model answers a given prompt much the same way
  every time. A group with no reward variance yields no advantage and no gradient.
- Constraints spread across difficulty. An earlier pool held five the model already satisfied
  (all lowercase scored 1.00), which is free reward. Training started at 0.693 with little
  left to learn. `IFEVAL_CONSTRAINTS` in [tasks.py](tasks.py) carries the measured
  per-constraint rates.
- `inner_epochs: 4`. Generation dominates wall clock, so spending each batch on one update
  wastes it, and at `1` the ratio is exactly 1 so the PPO clip never engages (`clipfrac`
  measured 0.0000). Raising it took the gain from 0.06 to 0.16.

The same run, split across a trainer pod and a rollout pod on a local
[kind](https://kind.sigs.k8s.io/) cluster (Docker Desktop >= 12 GB):

```bash
sky local up && sky jobs launch sky/local_jobgroup.yaml   # sky jobs logs <id> trainer
```

[`configs/countdown_local_cpu.yaml`](configs/countdown_local_cpu.yaml) is the arithmetic
alternative (0.039 to 0.164). Its ceiling is the model, since five configurations all stopped
at 0.172-0.195.

## Scaling up

| Command | Hardware |
|---|---|
| `--config configs/countdown_grpo.yaml` | 1x 24 GB GPU |
| `torchrun --nproc_per_node=8 train.py --config configs/countdown_grpo_gemma4.yaml` | 8x H100 |
| `sky jobs launch sky/jobgroup.yaml` | 2+ GPUs, k8s only |

The last one as it ships: Qwen3-8B + LoRA, one 8x H100 node training while eight L40S generate
with vLLM, wired by hostname via a
[SkyPilot Job Group](https://docs.skypilot.ai/en/latest/examples/job-groups.html). Held-out
0.391 to 0.609 at step 60; 512 sequences/update, 102k in 90 min; staleness mean 1.8, 0 of
1,603 batches dropped. CSV in [results/](results/).

![Training curves](assets/async_run.png)

The trainer recomputes logprobs under the weights that actually sampled each batch. An earlier
run used vLLM's as `old_logp`, the ratio drifted to 1.165 and accuracy fell from 0.500 to
0.188.

![Async health](assets/async_health.png)

## Knobs

All off by default, and a few lines of code each:

| Flag | |
|---|---|
| `--tis-clip 2` | Async only. The trainer recomputes `old_logp`, but vLLM sampled the tokens, so this is the importance weight for the gap that remains |
| `--eps-high 0.28` | Clip-higher: raise the ceiling alone, so unlikely but useful tokens can still grow |
| `--dual-clip 3` | Floor the surrogate where the advantage is negative and PPO's clip is one-sided |
| `--overlong-coef 0.5` | Penalize completions that crowd the token budget |
| `--eval-k 8` | pass@k at temperature 1 instead of greedy pass@1 |
| `--max-turns 3` | Multi-turn episodes: the task's `env_response()` answers each turn until it returns None. Countdown's calculator replies `4*5+2 = 22, not 23. Try again.`; eval scores the final turn as `acc` and the first as `acc_turn1` |

`--skip-zero-adv` is on by default. A group scoring uniformly has zero advantage, so its
forward/backward is skipped. That is a speedup only, since the loss denominators still count
its tokens. Log columns: `entropy` (which falls before the reward does), `clipfrac`, `akl`,
`resp_len`, `trunc`, `dead`, and `turns` under `--max-turns`.

## Layout

```
algos.py    ~90   advantage estimators: reinforce / baseline / ppo (GAE) / grpo / rloo
core.py    ~120   Trajectory + Batch
utils.py   ~170   masked reductions, returns/GAE, dist plumbing, CSV logger
serve.py   ~250   async transport: weight publishing, rollout queue, staleness rejection
model.py   ~420   MLPPolicy, HFPolicy (train + score), VLLMGenerator (rollout only)
tasks.py   ~660   CartPole, IFEval, Countdown, GSM8K; task = sample + rollout + reward fns
train.py   ~780   the loop: rollout, advantage, clipped update, publish
```

`--role trainer` on one box and `--role rollout` on the others talk over stdlib HTTP;
`max_staleness` is rejection bound, queue depth and snapshot count at once. 75 CPU-only tests:
`python tests/test_nanorl.py`.

A dataset is an `LLMTask` subclass: load rows, write `build_prompt`, pick reward functions
`(prompt, completion, answer) -> float`, register in `make_task`. Override
`env_response(problem, turns)` to make it multi-turn: each reply is appended to the sampled
token ids, never re-rendered, and only the model's turns are trained. Any
`AutoModelForCausalLM` works with `--model` (~12B on 48 GB with LoRA). Not included:
Megatron-scale parallelism, MoE, multi-tenant scheduling, dashboards.

MIT
