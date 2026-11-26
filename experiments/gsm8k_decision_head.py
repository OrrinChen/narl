"""A frozen Qwen correctness probe for completed GSM8K answers."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np


START_SHA256 = "2c6a2883a67adfb9e2f5bb2ff2f5c14dcab8203847b0f3522b1790f021f56dd2"
FEATURE_SCHEMA = "qwen-last-newline-tag-mean-logp-v1"


class FrozenEncoder:
    def __init__(self, model, tokenizer, identity):
        self.model = model
        self.pad_id = tokenizer.pad_token_id
        self.readout_ids = tokenizer.encode("\n", add_special_tokens=False)
        self.identity = identity
        if not self.readout_ids:
            raise ValueError("The readout marker must encode to at least one token")

    @classmethod
    def load(cls, source=Path("runs/gsm8k-budget-5seed")):
        from experiments.gsm8k_decision_candidates import load_frozen

        source = Path(source)
        protocol = json.loads((source / "protocol.json").read_bytes())
        identity = {key: protocol[key] for key in ("model", "revision", "start_sha256", "mlx", "mlx_lm")}
        if identity["start_sha256"] != START_SHA256:
            raise ValueError("Decision encoder differs from the pinned 80-question SFT checkpoint")
        model, tokenizer = load_frozen(protocol, source)
        model.freeze()
        return cls(model, tokenizer, identity)

    def last_hidden(self, sequences):
        import mlx.core as mx

        lengths = [len(row) for row in sequences]
        width = max(lengths)
        tokens = mx.array([row + [self.pad_id] * (width - len(row)) for row in sequences])
        # Causal attention ensures each readout precedes and cannot see its right padding.
        hidden = self.model.model(tokens)[mx.arange(len(sequences)), mx.array(lengths) - 1, :]
        return np.asarray(hidden.astype(mx.float32))


def encode_features(encoder, records, batch_size=8, cache_path=None):
    """Return features in input order, with cold measured batch time allocated equally per row.

    Only the prompt IDs, completion IDs, format flag and behavior logprobs are read.
    On a cache hit, recorded timings describe the original cold encoding and
    encoder_seconds describes work performed by the current call.
    """
    started = perf_counter()
    if batch_size <= 0 or not len(records):
        raise ValueError("Decision encoding requires records and a positive batch size")
    key = sha256(json.dumps([FEATURE_SCHEMA, encoder.identity, encoder.readout_ids, batch_size],
                            sort_keys=True).encode())
    sequences, numeric = [], []
    for record in records:
        prompt, completion = record["prompt_ids"], record["token_ids"]
        logp = np.asarray(record["behavior_logprobs"], dtype=np.float64)
        tag = record["has_answer_tag"]
        if (not len(prompt) or not len(completion) or logp.shape != (len(completion),)
                or not np.isfinite(logp).all() or not isinstance(tag, (bool, np.bool_))):
            raise ValueError("Malformed decision feature inputs")
        ids = list(prompt) + list(completion) + encoder.readout_ids
        scalar = np.array([tag, logp.mean()], dtype="<f4")
        sequences.append(ids)
        numeric.append(scalar)
        key.update(len(ids).to_bytes(8, "little"))
        key.update(np.asarray(ids, dtype="<i4").tobytes())
        key.update(scalar.tobytes())
    binding = key.hexdigest()
    cache_path = None if cache_path is None else Path(cache_path)
    hit = cache_path is not None and cache_path.exists()
    if hit:
        with np.load(cache_path, allow_pickle=False) as stored:
            if str(stored["binding"].item()) != binding:
                raise ValueError(f"Decision encoder cache belongs to different inputs or weights: {cache_path}")
            features, seconds = stored["features"], stored["row_encoder_seconds"]
    else:
        order = sorted(range(len(records)), key=lambda i: len(sequences[i]))
        features = None
        seconds = np.empty(len(records), dtype=np.float64)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_started = perf_counter()
            hidden = encoder.last_hidden([sequences[i] for i in indices])
            if hidden.ndim != 2 or len(hidden) != len(indices):
                raise ValueError("Unexpected shape of the decision encoder output")
            if features is None:
                features = np.empty((len(records), hidden.shape[1] + 2), dtype=np.float32)
            features[indices, :-2] = hidden
            features[indices, -2:] = [numeric[i] for i in indices]
            seconds[indices] = (perf_counter() - batch_started) / len(indices)
    if (features.ndim != 2 or features.shape[0] != len(records) or features.shape[1] < 3
            or not np.isfinite(features).all() or not np.array_equal(features[:, -2:], np.asarray(numeric))
            or seconds.shape != (len(records),) or not np.isfinite(seconds).all() or np.any(seconds < 0)):
        raise ValueError("Invalid stored features or timing in the decision encoder cache")
    if not hit and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache_path.parent, delete=False) as tmp:
            pending = Path(tmp.name)
        try:
            with pending.open("wb") as handle:
                np.savez(handle, binding=binding, features=features, row_encoder_seconds=seconds)
            pending.replace(cache_path)
        finally:
            pending.unlink(missing_ok=True)
    recorded_seconds = float(np.sum(seconds))
    return features, {
        "feature_binding": binding, "rows": len(records), "input_tokens": sum(map(len, sequences)),
        "batch_size": batch_size, "cache_hit": hit, "wall_seconds": perf_counter() - started,
        "encoder_seconds": 0.0 if hit else recorded_seconds, "recorded_encoder_seconds": recorded_seconds,
        "row_encoder_seconds": seconds.tolist(),
    }


def _inputs(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x).all() or not np.isin(x[:, -2], (0, 1)).all():
        raise ValueError("Expected hidden features, a binary answer-tag flag and a mean generation logprob")
    return np.delete(x, -2, axis=1), x[:, -2].astype(bool)


def fit_head(X, y, l2=0.3):
    """Fit normalized, L2-regularized logistic regression on tagged training rows."""
    import torch
    import torch.nn.functional as F

    x, tagged = _inputs(X)
    y = np.asarray(y)
    if y.shape != (len(x),) or not np.isin(y, (0, 1)).all() or np.any(y[~tagged]):
        raise ValueError("Expected strict binary labels with zero reward for every untagged answer")
    if not np.isfinite(l2) or l2 <= 0 or len(np.unique(y[tagged])) != 2:
        raise ValueError("Positive regularization and both outcome classes among tagged examples are required")
    training = x[tagged]
    mean, scale = training.mean(axis=0), training.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = torch.from_numpy((training - mean) / scale)
    target = torch.from_numpy(y[tagged].astype(np.float64))
    weights = torch.zeros(design.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], max_iter=200, tolerance_grad=1e-8,
                                  tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = design @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, target) + (l2 / 2) * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    if not torch.isfinite(weights).all() or not torch.isfinite(bias):
        raise RuntimeError("Nonfinite decision head after fitting")
    return {
        "schema": FEATURE_SCHEMA, "l2": l2, "n_train_candidates": len(x), "n_train_tagged": int(tagged.sum()),
        "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.detach().tolist(),
        "bias": float(bias.detach()),
    }


def predict_proba(head, X):
    """Return probabilities of a strictly correct answer, applying the required tag rule."""
    x, tagged = _inputs(X)
    if head["schema"] != FEATURE_SCHEMA:
        raise ValueError("Decision head uses an incompatible feature schema")
    z = ((x - head["mean"]) / head["scale"]) @ head["weights"] + head["bias"]
    probabilities = np.exp(-np.logaddexp(0.0, -z))
    return np.where(tagged, probabilities, 0.0)
