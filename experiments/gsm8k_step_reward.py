"""Training-only partial credit for explicit equations matching GSM8K calculator steps."""

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
import re
from operator import add, mul, sub, truediv

from experiments.gsm8k_training_reward_v2 import score_training_reward as outcome_reward


OPERATIONS = {"+": add, "-": sub, "*": mul, "/": truediv}
NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
EQUATION = re.compile(
    rf"(?<![\w.,])({NUMBER})\s*([+*/×÷xX−-])\s*({NUMBER})\s*=\s*\$?\s*({NUMBER})(?!\w|[.,]\d)"
)
ANNOTATION = re.compile(r"<<([^<>]+)>>")
ANSWER_START = re.compile(r"<(?:answer|solution|number)\b", re.I)
FINAL_FORMAT = re.compile(r"<answer>\s*[-+]?\$?[\d,.]+\s*</answer>\s*[.!]?\s*$", re.I)


def equations(text):
    # Read one explicit binary relation. A suffix of a compound operation
    # or one side of an equation chain is not a verified step.
    found = set()
    for line in text.splitlines():
        for match in EQUATION.finditer(line):
            if line[:match.start()].rstrip().endswith(("+", "*", "/", "-", "×", "÷", "−", "=", ".")):
                continue
            if line[match.end():].lstrip().startswith(("+", "*", "/", "-", "×", "÷", "−", "=")):
                continue
            left, operator, right, result = match.groups()
            if any(len(x) > 28 for x in (left, right, result)):
                continue
            try:
                a, b, c = (Fraction(Decimal(v.replace(",", ""))) for v in (left, right, result))
                operator = {"x": "*", "X": "*", "×": "*", "÷": "/", "−": "-"}.get(operator, operator)
                calculated = OPERATIONS[operator](a, b)
            except (InvalidOperation, ZeroDivisionError):
                continue
            if calculated == c:
                if operator in {"+", "*"} and b < a:
                    a, b = b, a
                found.add((a, operator, b, c))
    return frozenset(found)


@lru_cache(maxsize=8192)
def reference_steps(rationale):
    return frozenset().union(*(equations(annotation) for annotation in ANNOTATION.findall(rationale)))


def classify(sample, example, prefer_format):
    correct, reason = outcome_reward(sample, example["value"])
    text = sample["completion"]
    clean = not sample["truncated"] and bool(FINAL_FORMAT.search(text)) and text.lower().count("<answer>") == 1
    if correct:
        return (1.0 if not prefer_format or clean else 0.8), reason
    reference = reference_steps(example["rationale"])
    matched = len(reference & equations(ANSWER_START.split(text, maxsplit=1)[0]))
    score = 0.4 * matched / max(1, len(reference))
    if prefer_format and clean:
        score += 0.05
    return score, f"{reason}/steps={matched}/{len(reference)}" + ("/complete" if prefer_format and clean else "")


def score_training_reward(sample, example):
    return classify(sample, example, False)


def score_format_reward(sample, example):
    return classify(sample, example, True)
