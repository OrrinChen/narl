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
