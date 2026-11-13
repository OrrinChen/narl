"""Balanced short and long sequence tasks for the CPU GRPO scheduler benchmark."""
from __future__ import annotations

from random import Random

import torch

from experiments.kv_online import FIRST_CONTENT_ID, _prompt
from tasks import LLMTask, SMOKE_CHOICES, SMOKE_QUERIES


LABELS = ("0", "1")
SHORT_LENGTH = 3
LONG_LENGTHS = (12, 16)
TRAIN_EXAMPLES = 320
HELDOUT_EXAMPLES = 64
PILOT_SEEDS = (120, 121)


def _rows(count, distractors, filler_start):
    # Both classes have the same length distribution. Within each class,
    # short is half the work and each long length is one quarter.
    schedule = (SHORT_LENGTH, LONG_LENGTHS[0], SHORT_LENGTH, LONG_LENGTHS[1])
    fillers = Random(97 + filler_start).sample(range(filler_start, filler_start + count), count)
    for index, value in enumerate(fillers):
        query = index % len(SMOKE_QUERIES)
        length = schedule[index // len(SMOKE_QUERIES) % len(schedule)]
        distractor = distractors[index // (len(schedule) * len(SMOKE_QUERIES)) % len(distractors)]
        filler = " ".join(format(value, "012b"))
        yield {
            "prompt": f"{filler} {distractor} {SMOKE_QUERIES[query]}",
            "answer_tokens": [LABELS[(query + step) % 2] for step in range(length)],
            "completion_length": length,
            "bucket": "short" if length == SHORT_LENGTH else "long",
            "query": SMOKE_QUERIES[query],
            "distractor": distractor,
            "filler": filler,
        }


class SequenceTask(LLMTask):
    """Generate the phase-conditioned alternating bit sequence from a color token."""

    training_distractors = SMOKE_CHOICES[:2]
    heldout_distractor = SMOKE_CHOICES[3]
    heldout_filler_start = 2048

    def __init__(self, n_examples=TRAIN_EXAMPLES, n_eval=HELDOUT_EXAMPLES, seed=0):
        if not (0 < n_examples <= 1024 and 0 < n_eval <= 1024):
            raise ValueError("training and held-out sizes must be between 1 and 1024")
        heldout = list(_rows(n_eval, (self.heldout_distractor,), self.heldout_filler_start))
        training = list(_rows(n_examples, self.training_distractors, 0))
        self._split(heldout + training, n_eval, seed)

    def build_prompt(self, problem):
        return problem["prompt"]


class DevelopmentSequenceTask(SequenceTask):
    heldout_distractor = SMOKE_CHOICES[2]
    heldout_filler_start = 1024


@torch.no_grad()
def evaluate(policy, task):
    """Greedy exact match per bucket, using the same binary action support as rollout."""
    correct = {"short": 0, "long": 0}
    total = {"short": 0, "long": 0}
    for length in (SHORT_LENGTH, *LONG_LENGTHS):
        rows = [row for row in task.eval_data if row["completion_length"] == length]
        if not rows:
            continue
        prompts = torch.cat([_prompt(policy, task, row) for row in rows])
        generated = policy.model.generate(
            input_ids=prompts, attention_mask=torch.ones_like(prompts),
            do_sample=False, max_new_tokens=length,
            suppress_tokens=[*range(FIRST_CONTENT_ID),
                             *range(FIRST_CONTENT_ID + len(LABELS), policy.model.config.vocab_size)],
        )[:, -length:]
        answers = torch.tensor([policy.tok.convert_tokens_to_ids(row["answer_tokens"]) for row in rows])
        bucket = rows[0]["bucket"]
        correct[bucket] += int(generated.eq(answers).all(dim=1).sum())
        total[bucket] += len(rows)
    if not all(total.values()):
        raise ValueError("evaluation requires both short and long prompts")
    return {bucket: correct[bucket] / total[bucket] for bucket in total}
