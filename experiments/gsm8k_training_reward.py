"""Binary GSM8K training reward for explicit final numeric claims.

The official evaluator stays in tasks.py. Extraction below never sees the gold
answer. It judges the displayed number, without converting units or checking
whether the reasoning that produced it is valid.
"""

from fractions import Fraction
import re


TAG = re.compile(r"<(answer|solution|number)>\s*([^<>]*)\s*</\1>", re.I)
AMOUNT = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
VALUE = rf"[+-]?{AMOUNT}(?:\s*/\s*{AMOUNT})?"
SCALAR = re.compile(rf"(?:dollars\s+)?(?:[$£€]\s*|R(?=\d))?({VALUE})\s*(?:%|[A-Za-z]+(?:[- ][A-Za-z]+)?)?", re.I)
DIGIT = re.compile(rf"(?<![\w.]){VALUE}")
# A comparison, hedge, or magnitude word needs more than literal number extraction.
UNCERTAIN = re.compile(r"\b(?:or|maybe|probably|possibly|guess|not|if|about|perhaps|approximately|wrong|incorrect|minus|plus|times|million|billion|thousand|hundred)\b|[?≈~]", re.I)
CUE = re.compile(r"^(?:(?:therefore|thus|so|hence)[,:]?\s+)?(?:the\s+)?(?:final\s+)?answer\s*(?:is\b\s*:?|:)\s*(.+)$", re.I)
CONCLUSION = re.compile(r"^(?:therefore|thus|hence|so)[,:]\s+[^=:?]*?\b(?:is|are|has|have|costs|earns)\s+(.+)$", re.I)


def scalar(text):
    # A spaced cents typo is only repaired after an explicit currency symbol.
    text = re.sub(r"([$£€]\s*\d+)\.\s+(\d{2})(?!\d)", r"\1.\2", text.strip())
    text = re.sub(r"([+-])([$£€])", r"\2\1", text).rstrip(".!").strip()
    if UNCERTAIN.search(text):
        return None
    if "=" in text:
        lhs, text = text.rsplit("=", 1)
        unit_equation = re.fullmatch(rf"{VALUE}\s+[A-Za-z]+\s*[×÷+*/-]\s*{VALUE}", lhs.strip())
        numeric_equation = re.fullmatch(r"[\d\s.,()$£€+*/×÷x=−-]+", lhs) and re.search(r"\d", lhs)
        if not numeric_equation and not unit_equation:
            return None
    match = SCALAR.fullmatch(text.strip())
    if not match:
        return None
    try:
        operands = [Fraction(x.strip().replace(",", "")) for x in match[1].split("/")]
        return operands[0] if len(operands) == 1 else operands[0] / operands[1]
    except (ValueError, ZeroDivisionError):
        return None


def conclusion(line):
    if UNCERTAIN.search(line):
        return None
    cue = CUE.fullmatch(line.strip()) or CONCLUSION.fullmatch(line.strip())
    return scalar(cue[1]) if cue else None


def extract_training_answer(text, truncated):
    """Return (exact displayed value or None, extraction reason)."""
    text = text.strip()
    tags = list(TAG.finditer(text))
    if tags:
        last = tags[-1]
        source = last[1].lower()
        prefix = text[:last.start()].split("\n")[-1].strip()
        if re.search(r"\b(?:maybe|perhaps|guess|possibly|probably|not)\s*:?\s*$", prefix, re.I):
            return None, "uncertain_tag"
        if truncated and source != "answer":
            return None, "truncated_alternative"
        value = scalar(last[2])
        if value is None:
            return None, "ambiguous_payload"
        # Arithmetic explanations can legitimately tag several intermediate steps.
        # Adjacent competing final answers do not identify a single result.
        if len(tags) > 1:
            prev = tags[-2]
            previous_value = scalar(prev[2])
            between = text[prev.end():last.start()].strip()
            if previous_value is not None and previous_value != value and (
                len(between) <= 32 or (UNCERTAIN.search(between) and len(between) < 80)
            ):
                return None, "conflicting_tags"
        for line in text[:last.start()].splitlines():
            prior_claim = CUE.fullmatch(line.strip())
            previous_value = scalar(prior_claim[1]) if prior_claim else None
            if previous_value is not None and previous_value != value:
                return None, "conflicting_claims"
        raw_tail = text[last.end():]
        tail = raw_tail.strip()
        if re.search(r"<(?:answer|number|solution)\b", tail, re.I):
            return None, "incomplete_final_tag"
        # Permit a closing wrapper or a short, naturally ended restatement of
        # this number. A new question, conflicting answer or rambling suffix
        # receives no credit for the earlier tag.
        if tail and not re.fullmatch(r"(?:[\s.,!\"'”)]|</\w+>)+", tail):
            # Naturally ended answers sometimes append the unit on the same
            # line, or repeat the number in a short explanatory sentence.
            numbers = [scalar(m[0]) for m in DIGIT.finditer(tail)]
            same_line_unit = not raw_tail.startswith("\n") and re.fullmatch(r"[A-Za-z -]{1,48}[.!]?", tail)
            duplicate = re.fullmatch(r"<([A-Za-z]+)>\s*([^<>]+?)\s*</\1>", tail)
            repeated_wrapper = duplicate is not None and scalar(duplicate[2]) == value
            repeated = bool(numbers) and numbers[0] == value
            claims = re.finditer(r"\b(?:(?:final )?answer|final result)\s*(?:is\b|would be\b|:)\s*:?\s*([^\n]+)", tail, re.I)
            conflict = any(scalar(claim[1]) != value for claim in claims)
            if len(set(numbers)) > 1 and re.search(r"[.!]\s+[A-Z]|\b(?:intermediate|instead|should be|however|but)\b", tail, re.I):
                conflict = True
            if (truncated or conflict or len(tail) > 180 or "?" in tail or ("<" in tail and not repeated_wrapper) or
                    re.search(r"\b(?:actually|maybe|perhaps|wrong|incorrect|guess|not)\b|\d\s+or\s+[$£€]?\d", tail, re.I)
                    or not (same_line_unit or repeated or repeated_wrapper)):
                return None, "nonterminal_tag"
        return value, source + "_tag"

    if truncated:
        return None, "truncated_untagged"
    if re.search(r"</?(?:answer|solution|number)\b", text, re.I):
        return None, "malformed_tag"
    lines = text.splitlines()
    if not lines:
        return None, "no_final_claim"
    value = conclusion(lines[-1])
    if value is None:
        return None, "no_final_claim"
    for line in lines[:-1]:
        prior_claim = CUE.fullmatch(line.strip())
        prior = scalar(prior_claim[1]) if prior_claim else None
        if prior is not None and prior != value:
            return None, "conflicting_claims"
    return value, "explicit_conclusion"


def score_training_reward(sample, gold):
    """Return a binary reward and a reason usable in each rollout trace."""
    value, reason = extract_training_answer(sample["completion"], sample["truncated"])
    expected = Fraction(str(gold).replace(",", "").strip())
    return int(value is not None and value == expected), reason
