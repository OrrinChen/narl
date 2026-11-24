"""Training-only extension for naturally ended, explicit numerical answers."""

import re

from experiments.gsm8k_training_reward import TAG, UNCERTAIN, VALUE, scalar
from experiments.gsm8k_training_reward_v2 import score_training_reward as v2_reward


QUANTITY = rf"(?:[$£€]\s*)?{VALUE}"
UNIT = (r"(?:dollars?|cents?|minutes?|hours?|seconds?|days?|weeks?|years?|inches?|feet|foot|"
        r"miles?|meters?|metres?|liters?|litres?|gallons?|pounds?|ounces?|calories?|sections?)")
BOX = re.compile(r"\\boxed\{\s*([^{}]+(?:\\text\{[^{}]+\})?)\s*\}")
BOX_END = re.compile(r"\s*(?:\\[)\]]|\$\$?)?\s*[.!]?\s*")
BOX_CUE = re.compile(r"(?:^|[.!]\s+)(?:so|therefore|thus|hence)\b|\b(?:final (?:answer|result)|the answer)\b", re.I)
HEDGE = re.compile(r"\b(?:intermediate|partial|subtotal|first step|estimate|either|might|around|nearly|"
                   r"rather than|ignore)\b|[?≈~]", re.I)
EARLIER_CLAIM = re.compile(r"\b(?:final answer|the answer)\s*(?:is|:)\s*[$£€]?[-+]?\d", re.I)
ANSWER_CLAUSE = re.compile(r"(?:^|[.!]\s+|,\s+)(?:the (?:correct |final )?answer (?:here )?is|final answer\s*:)\s*(.+?)\s*$", re.I)
TAG_CLAIM = re.compile(rf"[A-Za-z][A-Za-z'’ -]{{0,65}}?\b(?:left with|ends? up with|received|earned|spent|"
                       rf"costs?|has|have|is|are)\s+({QUANTITY})(?:\s+{UNIT})?\s*[.!]?", re.I)
STANDALONE = re.compile(rf"{QUANTITY}\s+{UNIT}[.!]?", re.I)


def unwrap(text):
    text = text.strip().replace(r"\$", "$")
    for left, right in (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
    box = BOX.fullmatch(text)
    if box:
        text = box[1].strip()
    text = re.sub(r"\\text\{\s*([A-Za-z -]+?)\s*\}", r" \1", text)
    if re.fullmatch(rf"#{VALUE}", text):
        text = text[1:]
    return text.strip()


def replacement(text):
    """Locate one terminal claim to normalize; this function never reads gold."""
    tags = list(TAG.finditer(text))
    if tags:
        tag = tags[-1]
        body = tag[2].strip()
        if HEDGE.search(body) or UNCERTAIN.search(body):
            return None
        value = unwrap(body)
        claim = TAG_CLAIM.fullmatch(value)
        if claim:
            value = claim[1]
        if value != body:
            return tag.start(), tag.end(), value, "terminal_tag_claim"
        return None

    lines = text.splitlines()
    last = lines[-1].strip() if lines else ""
    if not last or HEDGE.search(last) or UNCERTAIN.search(last):
        return None
    start = text.rfind(last)
    box = BOX.search(last)
    if box and text.count(r"\boxed") == 1 and BOX_END.fullmatch(last[box.end():]):
        prefix = last[:box.start()].strip().rstrip("$").strip()
        prefix = prefix.removesuffix(r"\(").removesuffix(r"\[").strip()
        preceding = next((s.strip() for s in reversed(lines[:-1]) if s.strip()), "")
        if EARLIER_CLAIM.search(prefix):
            return None
        if not prefix and (HEDGE.search(preceding) or UNCERTAIN.search(preceding)):
            return None
        if not prefix or BOX_CUE.search(prefix):
            return start, len(text), unwrap(box[0]), "terminal_box"

    claim = ANSWER_CLAUSE.search(last)
    if claim and not EARLIER_CLAIM.search(last[:claim.start()]):
        return start + claim.start(), len(text), unwrap(claim[1]), "terminal_answer_clause"
    if STANDALONE.fullmatch(text):
        return 0, len(text), text, "standalone_quantity"
    return None


def score_training_reward(sample, gold):
    original = v2_reward(sample, gold)
    if original[0] or sample["truncated"] or original[1] not in ("ambiguous_payload", "no_final_claim"):
        return original
    text = sample["completion"].strip()
    match = replacement(text)
    if match is None:
        return original
    start, end, number, reason = match
    if scalar(number) is None:
        return original
    # Let v2 recheck contradictory claims, suffixes, sign, fraction and gold.
    normalized = text[:start] + "<answer>" + number + "</answer>" + text[end:]
    reward, _ = v2_reward({**sample, "completion": normalized}, gold)
    return (reward, reason) if reward else original
