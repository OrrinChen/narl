"""Train the MLX LoRA learner on fixed, grouped GSM8K completions."""

from math import isfinite

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW, clip_grad_norm
from mlx.utils import tree_map


class MlxGrpoLearner:
    def __init__(self, model, lr, micro_batch=1):
        from experiments.gsm8k_mlx_development import suffix_logits

        if micro_batch < 1:
            raise ValueError("micro_batch must be positive")
        self.model = model
        self.micro_batch = micro_batch
        self.suffix_logits = suffix_logits
        self.optimizer = AdamW(learning_rate=lr, bias_correction=True)
        self.value_and_grad = nn.value_and_grad(model, self.loss)

    def loss(self, inputs, actions, behavior_logp, mask, advantages, sequence_count):
        logits = self.suffix_logits(self.model, inputs, actions.shape[1]).astype(mx.float32)
        logp = -nn.losses.cross_entropy(logits, actions, reduction="none")
        log_ratio = mx.where(mask, logp - behavior_logp, 0.0)
        ratio = mx.exp(log_ratio)
        clipped = mx.clip(ratio, 0.8, 1.2)
        surrogate = mx.minimum(ratio * advantages, clipped * advantages)
        sequence_mean = mx.where(mask, surrogate, 0.0).sum(axis=1) / mask.sum(axis=1)
        loss = -sequence_mean.sum() / sequence_count
        stats = {
            "ratio": mx.where(mask, ratio, 0.0).sum(),
            "clipfrac": (mask & ((ratio < 0.8) | (ratio > 1.2))).sum(),
            "akl": -log_ratio.sum(),
            "entropy": mx.where(mask, -logp, 0.0).sum(),
        }
        return loss, stats

    def step(self, prompt_ids, samples, advantages):
        """Apply one clipped-GRPO step to the canonical sample order."""
        advantages = [float(value) for value in advantages]
        if not prompt_ids or not samples or len(samples) != len(advantages):
            raise ValueError("provide one advantage per completion and a nonempty prompt")

        active = [(sample, value) for sample, value in zip(samples, advantages, strict=True) if value != 0.0]
        stats = dict.fromkeys(("ratio", "clipfrac", "akl", "entropy"), 0.0)
        total_loss, scored_tokens, gradients = 0.0, 0, None
        for start in range(0, len(active), self.micro_batch):
            chunk = active[start:start + self.micro_batch]
            lengths = [len(sample["token_ids"]) for sample, _ in chunk]
            width = max(lengths)
            actions, behavior_logp, mask, values = [], [], [], []
            for (sample, value), length in zip(chunk, lengths, strict=True):
                if not length or len(sample["behavior_logprobs"]) != length:
                    raise ValueError("expected one behavior logprob per completion token")
                padding = width - length
                actions.append(sample["token_ids"] + [0] * padding)
                behavior_logp.append(sample["behavior_logprobs"] + [0.0] * padding)
                mask.append([True] * length + [False] * padding)
                values.append([value])
            inputs = [prompt_ids + row[:-1] for row in actions]
            (loss, batch_stats), grad = self.value_and_grad(
                mx.array(inputs), mx.array(actions), mx.array(behavior_logp, mx.float32),
                mx.array(mask), mx.array(values, mx.float32), len(samples),
            )
            gradients = grad if gradients is None else tree_map(lambda left, right: left + right, gradients, grad)
            # Evaluate after each microbatch to release saved forward activations.
            mx.eval(loss, batch_stats, gradients)
            total_loss += float(loss.item())
            scored_tokens += sum(lengths)
            for name, value in batch_stats.items():
                stats[name] += float(value.item())

        means = {name: round(value / max(scored_tokens, 1), 4) for name, value in stats.items()}
        norm = 0.0
        if gradients is not None:
            clipped, grad_norm = clip_grad_norm(gradients, 1.0)
            norm = float(grad_norm.item())
            if not all(isfinite(value) for value in (total_loss, norm, *means.values())):
                raise RuntimeError("nonfinite MLX GRPO update")
            self.optimizer.update(self.model, clipped)
            mx.eval(self.model.trainable_parameters(), self.optimizer.state)
        return {"loss": total_loss, "pg": total_loss, "gnorm": norm,
                "skipped": len(samples) - len(active), **means}
