# GPU and GSM8K validation

## Same-model CUDA comparison

Use a single NVIDIA GPU and the same model revision, dtype, attention backend,
group size, and token histories for dense and shared runs. Start with 128 and
384 prompt tokens and 3 and 16 answer tokens. Save the environment, GPU memory
peaks, numerical differences, and raw timings before extending the grid.

In a CUDA-enabled PyTorch environment with Transformers 4.57.6 and Narl's
Python dependencies, run the fixed-token benchmark below. The model is
[Qwen2.5-1.5B-Instruct at the pinned revision](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).

```bash
python -m experiments.kv_length_sweep --out runs/prefix-length-cuda-smoke \
  --device cuda --dtype bfloat16 --model Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --prompts 128 384 --outputs 3 16 --warmups 1 --repetitions 3
```

This compares PyTorch actor and learner phases at fixed weights. For in-flight
recovery, use two distinct trained checkpoints and the same recorded suffix in
all three modes. Measure recovery-to-first-logits with CUDA synchronization
and record weight transfer separately. Compare KL against fresh dense rebuild
on the exact same history. [PipelineRL's GPU KV/KL comparison](https://arxiv.org/html/2509.19128)
provides a related reference point.

## vLLM rollout baseline

Run vLLM in its own compatible CUDA environment. Benchmark grouped generation
with APC off and on at identical settings. Keep cold and reused-prefix workloads
separate. For the second workload the following example uses a 384-token common
prefix, 32 suffix tokens per prompt, and eight answers per prompt. Use the
same model revision, backend, seed, and concurrency in both runs.
[vLLM 0.28 benchmark flags](https://docs.vllm.ai/en/v0.28.0/cli/bench/throughput/)

```bash
mkdir -p runs/vllm-apc
for cache in off on; do
  if [ "$cache" = off ]; then
    apc=--no-enable-prefix-caching
  else
    apc=--enable-prefix-caching
  fi
  vllm bench throughput --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --dtype bfloat16 \
    --dataset-name prefix_repetition --num-prompts 64 --n 8 --seed 0 \
    --prefix-repetition-num-prefixes 8 --prefix-repetition-prefix-len 384 \
    --prefix-repetition-suffix-len 32 --prefix-repetition-output-len 16 \
    "$apc" --output-json "runs/vllm-apc/${cache}.json"
done
```

Record actual prefix cache hits before attributing a difference to APC.
APC concerns prefill reuse in the serving engine. FlashInfer cascade concerns
attention over shared KV, and a vLLM FlashInfer backend selection alone does
not establish that cascade ran. Verify the selected kernel for any cascade
claim. Report the vLLM actor result and the PyTorch learner result separately.
[vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) ·
[FlashInfer cascade](https://docs.flashinfer.ai/api/cascade.html) ·
[vLLM 0.28 FlashInfer backend](https://docs.vllm.ai/en/v0.28.0/api/vllm/v1/attention/backends/flashinfer/)

## Learning under policy-version mixing

### Reproduce the local pilot

The local experiment uses Qwen2.5-0.5B-Instruct, PyTorch 2.8.0, and a
LoRA adapter. It derives two weight snapshots using supervised updates on
disjoint official training questions. Both replay arms start from the newer
snapshot and receive the same sampled completions and behavior logprobs in
the same update order. Only the within-prompt group assignment changes.

```bash
uv pip install --python .venv/bin/python peft==0.17.1 accelerate==1.10.1
data=runs/gsm8k-data
source=https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data
mkdir -p "$data"
curl -fsSL "$source/train.jsonl" -o "$data/train.jsonl"
curl -fsSL "$source/test.jsonl" -o "$data/test.jsonl"

.venv/bin/python -m experiments.gsm8k_signal --train "$data/train.jsonl" \
  --out runs/gsm8k-signal-repro --prompts 13

out=runs/gsm8k-regroup-local-repro
PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m experiments.gsm8k_regroup \
  --train "$data/train.jsonl" --test "$data/test.jsonl" --out "$out" \
  --replay-prompt-ids 1907 1514 2138 6869 --sft-prompts 8 --heldout-prompts 8 \
  --group-size 4 --max-new-tokens 128 --lora-last-layers 4 \
  --device mps --learner-device cpu
.venv/bin/python -m experiments.audit_gsm8k_regroup \
  --train "$data/train.jsonl" --test "$data/test.jsonl" --run "$out" --out "$out/audit.json"
```

The local run used MPS BF16 for snapshot preparation, rollout and evaluation,
and CPU FP32 for GRPO. To replay the exact same sampled answers and adapters,
add `--replay-from runs/gsm8k-regroup-local-fp32` with a new
output directory. Use `--device cpu` for a fresh run without an MPS device.
The four local replay prompts were selected for reward variation during a
training-split screening run. The eight test prompts are drawn independently
of rewards. [Model revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) · [GSM8K dataset](https://github.com/openai/grade-school-math)

Diagnose the original 0/8 directly from the saved greedy generations.

```bash
.venv/bin/python -m experiments.gsm8k_zero_diagnosis \
  --run runs/gsm8k-regroup-local-fp32 --test runs/gsm8k-data/test.jsonl
```

Six answers had complete tags and all six contained the wrong number. Seven
generations ended within 4–13 tokens. The remaining generation hit the
128-token cap after giving the wrong final number. Scoring the final number
without requiring tags still gives 0/8. All four training arms generated the
same tokens as the starting checkpoint on each question. [Per-question diagnosis](../runs/gsm8k-zero-diagnosis/summary.json)

For a learning comparison, reserve development questions from the official
training split before any updates. Keep those questions disjoint from the
supervised examples and GRPO rollout prompts. Use the same prompt and scorer
before and after training to verify a development accuracy gain. Freeze the
checkpoint, prompt, scorer, and training budget before evaluating on a fresh
sample of official test questions. Exclude the eight inspected pilot questions
from that sample. Use the same test questions for every arm and report paired
correct counts, wins, losses, and consumed tokens.

### Reproduce the matched-budget comparison on Apple Silicon

Use the pinned Qwen checkpoint above in the local Hugging Face cache. The
development run uses MLX BF16, four-example evaluation batches, and a 256-token
generation cap. It selects the first SFT checkpoint to gain at least three
correct answers on 24 train-derived development questions. All three replay
arms use the same sampled completions and start from that checkpoint.

```bash
uv pip install --python .venv/bin/python mlx==0.29.2 mlx-lm==0.29.1
export TOKENIZERS_PARALLELISM=false
data=runs/gsm8k-data
dev=runs/gsm8k-dev-mlx-repro
out=runs/gsm8k-matched-mlx-repro

.venv/bin/python -m experiments.gsm8k_mlx_development \
  --train "$data/train.jsonl" --out "$dev" --max-new-tokens 256 \
  --eval-batch 4 --checkpoints 16 40 80 160
.venv/bin/python -m experiments.gsm8k_matched \
  --train "$data/train.jsonl" --test "$data/test.jsonl" --qualification "$dev" \
  --out "$out" --replay-prompts 16 --heldout-prompts 32 --max-new-tokens 256
.venv/bin/python -m experiments.audit_gsm8k_matched \
  --train "$data/train.jsonl" --test "$data/test.jsonl" --qualification "$dev" \
  --run "$out" --out "$out/audit.json"
```

For an exact replay of the qualified adapters, set `dev=runs/gsm8k-dev-mlx-final`
and run the last two commands. See [the matched-budget results](../RESULTS.md#l-gsm8k-开发集验收与同预算分组对照).

### Reproduce rolling one-version lag on Apple Silicon

The runner starts all arms from the same pair of adjacent GRPO checkpoints.
It uses the qualified SFT adapter as version 0 and executes a GRPO update on
four saved completions generated by that adapter to produce version 1. It
generates a new group with version 1 and uses the same optimizer to produce
version 2. Both updates are outside the measured budget. Each arm resets
the optimizer, generates new samples, and updates its own learner. Version
publication follows a change in adapter bytes. A no-op preserves both the
current and previous version.

```bash
data=runs/gsm8k-data
dev=runs/gsm8k-dev-mlx-final
warmup=runs/gsm8k-matched-mlx-final
out=runs/gsm8k-online-lag-repro
tokenizer="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json"

TOKENIZERS_PARALLELISM=false .venv/bin/python -m experiments.gsm8k_online_lag \
  --train "$data/train.jsonl" --test "$data/test.jsonl" --qualification "$dev" \
  --warmup-run "$warmup" --out "$out"
.venv/bin/python -m experiments.audit_gsm8k_online_lag \
  --train "$data/train.jsonl" --test "$data/test.jsonl" --run "$out" \
  --warmup-source "$warmup/samples.jsonl" --qualification "$dev" \
  --tokenizer "$tokenizer" --verify-training --out "$out/audit.json"
```

The local runner interleaves generation and learning on one Apple M2. Fresh
and stale homogeneous groups use batched decoding when the remaining budget
allows it. Mixed groups decode each version separately and alternate which
version occupies the first slot. Training prompt order and random seeds are
paired. The different batch shapes can draw different samples from those
seeds. Each arm stops at exactly 6,144 emitted completion tokens including
EOS. The final group is capped by the remaining budget, and an incomplete
group consumes no learner tokens. Report realized updates and consumed tokens
alongside accuracy. All arms evaluate the same 32 previously uninspected
official test questions.

### Larger fixed-rollout and online runs

Use Qwen2.5-1.5B-Instruct with a LoRA pilot and the existing numeric answer
verifier. Freeze 256–512 training prompts and a development set from the
GSM8K train split. Freeze a separate set of 128–256 official test prompts.
Check nonzero reward variance by prompt, numeric exact match, answer-format
failure, and truncation before selecting the fixed training budget.
[GSM8K source and splits](https://github.com/openai/grade-school-math)

For the first controlled replay, sample eight answers from each of two
checkpoints per prompt. Use exactly the same 16 answers and behavior logprobs
in both arms. One arm trains on two version-homogeneous groups. The other
trains on two groups containing four answers from each version. Use the
same initial learner and optimizer, total generated and consumed tokens,
and fixed held-out numeric accuracy. Repeat the regrouping with two independent
sampling streams at the same checkpoint to estimate the effect of regrouping
alone. This extends Narl's fixed-sample experiment to held-out learning.

For an online follow-up, isolate actual version lag from mixed grouping with
separate fresh, delayed homogeneous, and mixed arms. Match the delayed and
mixed arms' age distributions and report completion length, consumed budget,
stale rejection, and importance-ratio clipping. The existing online server
carries one behavior version per batch, so this arm requires extending
provenance to each completion before measuring mixed-version learning.
