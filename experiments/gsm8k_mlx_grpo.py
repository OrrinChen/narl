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

    def step(self, prompt_ids, samples, advantages, *, auxiliary=None):
        """Apply one clipped-GRPO step to the canonical sample order."""
        return self.step_groups([(prompt_ids, samples, advantages)], auxiliary=auxiliary)

    def step_groups(self, groups, *, auxiliary=None):
        """Average all completions across frozen-policy groups, then update once.

        Advantages must already be computed within each prompt's group. Zero-signal
        groups stay in the denominator; microbatches never mix prompt lengths.
        """
        prepared = []
        for prompt_ids, samples, advantages in groups:
            values = [float(value) for value in advantages]
            if not prompt_ids or not samples or len(samples) != len(values):
                raise ValueError("provide one advantage per completion and a nonempty prompt")
            prepared.append((prompt_ids, samples, values))
        if not prepared:
            raise ValueError("provide at least one prompt group")
        sequence_count = sum(len(samples) for _, samples, _ in prepared)
        stats = dict.fromkeys(("ratio", "clipfrac", "akl", "entropy"), 0.0)
        total_loss, scored_tokens, active_count, gradients = 0.0, 0, 0, None
        for prompt_ids, samples, advantages in prepared:
            active = [(sample, value) for sample, value in zip(samples, advantages, strict=True) if value != 0.0]
            active_count += len(active)
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
                    mx.array(mask), mx.array(values, mx.float32), sequence_count,
                )
                gradients = grad if gradients is None else tree_map(lambda left, right: left + right, gradients, grad)
                # Evaluate after each microbatch to release saved forward activations.
                mx.eval(loss, batch_stats, gradients)
                total_loss += float(loss.item())
                scored_tokens += sum(lengths)
                for name, value in batch_stats.items():
                    stats[name] += float(value.item())

        auxiliary_nll = None
        if auxiliary is not None:
            aux_prompt, target, weight = auxiliary
            ids = mx.array([aux_prompt + target[:-1]])
            actions = mx.array([target])

            def numeric_loss(model):
                logits = self.suffix_logits(model, ids, len(target)).astype(mx.float32)
                return weight * nn.losses.cross_entropy(logits, actions, reduction="mean")

            aux_loss, aux_grad = nn.value_and_grad(self.model, numeric_loss)(self.model)
            gradients = aux_grad if gradients is None else tree_map(lambda a, b: a + b, gradients, aux_grad)
            mx.eval(aux_loss, gradients)
            auxiliary_nll = float(aux_loss.item()) / weight
            if not isfinite(auxiliary_nll):
                raise RuntimeError("nonfinite gold-answer loss")

        means = {name: round(value / max(scored_tokens, 1), 4) for name, value in stats.items()}
        norm = 0.0
        if gradients is not None:
            clipped, grad_norm = clip_grad_norm(gradients, 1.0)
            norm = float(grad_norm.item())
            if not all(isfinite(value) for value in (total_loss, norm, *means.values())):
                raise RuntimeError("nonfinite MLX GRPO update")
            self.optimizer.update(self.model, clipped)
            mx.eval(self.model.trainable_parameters(), self.optimizer.state)
        result = {"loss": total_loss, "pg": total_loss, "gnorm": norm,
                  "skipped": sequence_count - active_count, **means}
        if auxiliary_nll is not None:
            result["aux_nll"] = auxiliary_nll
        return result
