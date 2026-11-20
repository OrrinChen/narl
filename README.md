# Narl

Narl studies weight updates, KV-cache recomputation, and group composition in asynchronous GRPO. It extends [nanoRL at `793e9b5`](https://github.com/alex000kim/nanoRL/tree/793e9b5b3d176ebb40161d8ba8d2125fe5153473) under its original MIT license. The upstream instructions follow the Narl quickstart.

## Key results on Apple M2

- **KV recovery after a weight swap.** Replayed the same token history on a 19K-parameter Llama with retained KV, per-completion rebuild, and shared-prompt rebuild. With 384-token prompts, 16 cached suffix tokens, and eight completions per group, sharing cut median recovery-to-first-logit latency from **49.2 to 10.0 ms**. The maximum logit difference from per-completion rebuild was **1.19e-7**. At 32 prompt tokens, shared rebuild took **2.67 ms** versus **1.93 ms** for per-completion rebuild. [Three-way replay](RESULTS.md#a-权重切换后重算-kv)
- **Policy versioning.** Traced most long rollout expiries in Narl's FIFO runner to optimizer steps that advanced the version without changing the weights. Publishing a new version only when parameter bytes changed reduced expired long groups from **234 to 19 out of 1,272**, across five paired seeds with a one-version lag limit. [Snapshot-version rerun](RESULTS.md#h-权重实际变化才升版)

In the online token-choice experiment with a 19K-parameter Llama, 128-token prompts, and 16-, 24-, or 32-token outputs, sharing raised useful training-token throughput by a paired median **45.6%**. In the separate two-digit addition experiment with a 113K-parameter Llama, padded 128-token prompts, and three-token answers, the paired median reduction in time to 70% held-out accuracy was **60.9%** from actor readiness and **42.3%** from actor launch. The fixed-weight length sweep covers 28 shapes. At a 128-token prompt, actor speedup fell from **2.99×** to **1.06×** as outputs grew from 3 to 128 tokens. [Online throughput](runs/online-prefix-5seed/summary.json) · [Time to accuracy](RESULTS.md#i-两位数加法上的共享前缀) · [Length sweep](runs/prefix-length-cpu-final/speedup.png)

On GSM8K, LoRA SFT raised tagged exact-match accuracy from 3/24 to 9/24 on a separate development split. Three GRPO arms then trained on the same 27,877 sampled tokens. Version-homogeneous, mixed-version, and within-version shuffled groups scored 10/32, 11/32, and 12/32 on the same fresh test questions, versus 11/32 before GRPO. [Matched-budget results](RESULTS.md#l-gsm8k-开发集验收与同预算分组对照)

The rolling GSM8K experiment started from two adjacent GRPO checkpoints and generated 6,144 tokens per arm in each of five paired seeds. Every pairwise comparison of fresh, one-version-stale, and mixed-version groups had two wins, two losses, and one tie on 32 held-out questions. [Online lag results](RESULTS.md#m-gsm8k-在线滞后一版)

On the same GRPO rollout budget, adding 30–40 supervised answer tokens per seed reduced held-out gold-digit NLL in all five paired seeds. The median reduction versus plain GRPO was 3.01% across 64 fixed test questions. Greedy exact-match results varied by seed. [Held-out loss results](RESULTS.md#n-gsm8k-留出答案损失)

On 96 fresh GSM8K test questions, adding 256 supervised worked solutions after GRPO raised free-generation exact match in all five paired seeds. The median gain was 10.42 percentage points at the same 6,144 rollout tokens per arm. [Free-generation results](RESULTS.md#o-gsm8k-自由生成准确率)

An equal-response-token ablation then compared SFT alone, GRPO alone, and GRPO followed by SFT from the same pre-GRPO initialization. Each arm received 32,768 additional completion or supervised-label tokens per seed. On another 96 fresh test questions, SFT scored **34, 34, 28, 32, 35**, GRPO **23, 25, 24, 25, 13**, and GRPO then SFT **32, 30, 33, 34, 36**. The mixed recipe beat SFT in three seeds and lost in two; its median paired gain was **1.04 pp**, with a question-block 95% interval of **−5.21 to +6.25 pp**. This fixed recipe did not establish an additional GRPO benefit over SFT. Response tokens match; FLOPs do not. [Attribution ablation](RESULTS.md#p-gsm8k-同新增响应-token-预算对照)

A development-only diagnosis then found learning-rate-sensitive drift in the worst GRPO seed. Replaying its identical sampled data at `6e-4` scored **8/48**, versus **26/48** at `1e-4`; a new on-policy `1e-4` run at the same 32,768-token budget scored **24/48**, against **27/48** at initialization. Lowering the rate reduced collapse without establishing a learning gain. The earlier official-test results remain unchanged and describe fixed recipes, rather than GRPO's best attainable performance. [GRPO drift diagnosis](RESULTS.md#q-grpo-only-退化诊断)

Kubernetes validation covered one KV-replay Job in one Pod on local kind. [Job details](#local-kubernetes-job)

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

For the expanded comparison, share prompt computation at actor startup, after weight
switches, and during learner scoring. The learner keeps the prefix KV attached to
autograd, so gradients from every completion flow through the same prefix forward.
The dense arm and fixed evaluation path use the same settings as before.

```bash
.venv/bin/python tests/test_shared_prefix.py
.venv/bin/python -m experiments.kv_online --out runs/prefix-repro --groups 384 --seeds 0 1 2 3 4 --share-initial --share-learner
.venv/bin/python -m experiments.audit_kv_online --run runs/prefix-repro --out runs/prefix-repro/audit.json
```

Separate stress and sensitivity runs reuse the online entry point:

```bash
.venv/bin/python -m experiments.kv_online --out runs/stress-repro --groups 18 --seeds 0 --stress-delay-s 4
.venv/bin/python -m experiments.kv_online --out runs/sensitivity-repro --groups 48 --seeds 0 --max-lag 2
```

## CPU time to held-out accuracy

The learner trains a tiny decoder to classify four query tokens into two labels.
The fixed held-out set uses distractor and filler contexts excluded from training.
Only the first generated token receives credit. The remaining tokens supply
asynchronous rollout load, including weight swaps. Both arms generate the same
number of groups and tokens. The reported time is the second evaluation in the
first pair of consecutive evaluations that reach at least 75% accuracy.

```bash
.venv/bin/python -m unittest tests.test_narl_kv_learning
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m experiments.kv_learning --out runs/learning-repro
```

The run records the frozen protocol, held-out set, per-group rollout trace, and
evaluation times with both learner-received and actor-completed token counts.

## CPU rollout dispatch

Three actor processes run fixed 8-, 24-, and 64-token completions against one learner.
Every arm uses shared prefix computation and the same one-version lag limit.
FIFO reserves a fixed number of groups and consumes in arrival order. Version-aware
dispatch admits a new group only if every existing reservation can tolerate one
more learner publication. Each reservation lasts through the learner's disposition
of the group, including the optimizer step and weight publication when accepted.

```bash
.venv/bin/python tests/test_kv_dispatch.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m experiments.kv_dispatch --out runs/dispatch-repro --groups 384 --seeds 0 1 2 3 4 --windows 1 2
.venv/bin/python -m experiments.audit_kv_dispatch --run runs/dispatch-repro --out runs/dispatch-repro/audit.json \
  --baseline fifo-1 --baseline fifo-2
```

## Two-group CPU updates on short and long sequences

The model generates 3-, 12-, or 16-token alternating bit sequences conditioned
on a query token. The training reward is the fraction of positions that match.
Greedy evaluation requires an exact sequence match in each length bucket and
uses held-out distractor tokens and filler strings. Both arms use shared prefix
computation, three actors, two full groups per optimizer step, and a one-version
lag limit. Version-aware admission reserves a learner slot before a group's
oldest behavior version can expire.

The sequence benchmark advances the policy version only when the published
parameter bytes change. Optimizer batches have a separate ordinal, which the
two-evaluation accuracy target still uses. Each update records the candidate
and preceding snapshot hashes, and each run saves the final published snapshot.

```bash
.venv/bin/python -m experiments.kv_sequence_pilot --out runs/sequence-pilot-repro
.venv/bin/python -m experiments.kv_sequence --out runs/sequence-repro --seeds 0 1 2 3 4 \
  --groups 512 --threshold .75 --lr .002 --eval-every 32 --fifo-windows 2 --version-window 2
.venv/bin/python -m experiments.audit_kv_sequence --run runs/sequence-repro --out runs/sequence-repro/audit.json
```

Use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TOKENIZERS_PARALLELISM=false` for the timed comparison. The formal FIFO window
was selected on development seeds 120 and 121 by comparing windows 1 through 4.
See the run protocol and results for the target and timing definitions.

## CPU prefix comparison on two-digit addition

Both arms use a common arithmetic checkpoint, FIFO with two in-flight groups,
two complete groups per update, and a one-version lag limit. The actor and
learner share the prompt within each rollout group in one arm. The dense arm
processes each prompt separately and projects only the logits used by the loss.
The learner grades all three answer digits by exact match.

The comparison reports time from actor launch and from the point when all
actors are ready. Both clocks include the subsequent held-out evaluations.
The run records the selected checkpoint hash, exact-match curves, fixed
generation budget, per-group traces, and the final published snapshots.
Three-digit answers finish before the first in-flight weight-swap check.

```bash
.venv/bin/python -m experiments.kv_arithmetic_task --out runs/arithmetic-sft-repro \
  --sft-steps 1380 --sft-eval-every 60
.venv/bin/python -m experiments.kv_arithmetic_task --out runs/arithmetic-pilot-repro \
  --model runs/arithmetic-sft-repro/sft-1380 --grpo-groups 768 --eval-every 96 --grpo-lr .0001
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m experiments.kv_sequence --out runs/arithmetic-prefix-repro \
  --checkpoint runs/arithmetic-sft-repro/sft-1380 --seeds 0 1 2 3 4 \
  --groups 768 --eval-every 96 --threshold .70 --lr .0001
.venv/bin/python -m experiments.audit_kv_sequence --run runs/arithmetic-prefix-repro \
  --out runs/arithmetic-prefix-repro/audit.json
```

## Rebuild the figures and CPU profile

Run the plotting modules with a Python environment that includes Matplotlib. Pass an unused output file to the profiling command.

```bash
.venv/bin/python -m experiments.kv_profile --run runs/kv-replay-repro --out runs/kv-replay-repro/profile.json
python3 -m experiments.plot_kv_replay --summary runs/kv-replay-repro/summary.json --out runs/kv-replay-repro/replay.png
python3 -m experiments.plot_regroup --run runs/regroup-repro --out runs/regroup-repro/regroup.png
python3 -m experiments.plot_kv_online --run runs/online-repro --out runs/online-repro/online.png
```

## Prefix and output length sweep

The fixed-token benchmark measures actor prefill plus cached decoding, and
learner scoring plus backward, separately. It uses identical token histories
and the same attention backend within each pair. The dense learner only
projects the answer logits. See [the CPU results](RESULTS.md#j-前缀长度与输出长度的影响)
and [the GPU and GSM8K protocol](docs/gpu-validation.md).

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m experiments.kv_length_sweep --out runs/prefix-length-repro \
  --warmups 3 --repetitions 15
python3 -m experiments.plot_kv_lengths --run runs/prefix-length-repro \
  --out runs/prefix-length-repro/speedup.png
```

## Local Kubernetes Job

With a local Docker runtime, kind, and kubectl available, run `bash k8s/run-kv-replay.sh`. The runner creates a temporary loopback kind cluster and runs the same KV replay entry point in one Pod with five repetitions. It saves the Job and Pod status, logs, image ID, dependencies, and the Pod's replay artifacts under a new `runs/k8s-kv-replay.*` directory.

## Related work

Prefix Grouper shares the differentiable GRPO learner prefix. PipelineRL compares
retained and recomputed KV by KL after a weight change. Narl's fixed-history
comparison includes a third, group-shared rebuild and records latency and
numerical agreement across prompt lengths.
[Prefix Grouper](https://arxiv.org/abs/2506.05433) · [PipelineRL](https://arxiv.org/html/2509.19128)

PSRL documents group-level staleness control. Axolotl documents keeping the
policy version when an empty training step leaves weights unchanged. Narl
diagnoses and measures the corresponding expiry bug in its CPU FIFO runner.
[PSRL](https://psrl.readthedocs.io/en/latest/design/staleness_control.html#group-sampling) ·
[Axolotl](https://docs.axolotl.ai/docs/grpo.html#async-training-features)

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
