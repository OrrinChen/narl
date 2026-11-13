"""Differentiable shared-prompt scoring for one online GRPO group."""
import torch
import torch.nn.functional as F


def shared_prefix_logprobs(model, batch, prompt_tokens, first_content_id):
    """Score each completion, keeping all branches connected to the prefix graph.

    Each row starts with the same unpadded prompt. Prompt positions and any
    right padding are masked out by the caller. The model must allow cached
    forwards with gradients enabled.
    """
    states = batch.states
    prompt = model(states[:1, :prompt_tokens], use_cache=True, logits_to_keep=1)
    first_logits = prompt.logits.expand(states.shape[0], -1, -1)

    suffix_inputs = states[:, prompt_tokens:-1]
    if suffix_inputs.shape[1]:
        cache = prompt.past_key_values
        # The repeat is differentiable: prefix gradients sum across all branches.
        cache.batch_repeat_interleave(states.shape[0])
        suffix = model(suffix_inputs, past_key_values=cache, use_cache=True)
        logits = torch.cat((first_logits, suffix.logits), dim=1)
    else:
        logits = first_logits

    targets = (states[:, prompt_tokens:] - first_content_id).clamp_min(0)
    logprobs = -F.cross_entropy(
        logits[:, :, first_content_id:].transpose(1, 2), targets, reduction="none",
    )
    return F.pad(logprobs, (prompt_tokens, 0)) * batch.mask
