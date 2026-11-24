"""Numeric outcome reward with guarded support for math typography and explanations."""

from fractions import Fraction
import operator
import re

from experiments.gsm8k_training_reward import (
    TAG, UNCERTAIN, VALUE, extract_training_answer, scalar,
)


# Avoid wholesale NFKC: for example 3² should never become the integer 32.
MATH_PUNCTUATION = str.maketrans("（）＄，．＝＋－％０１２３４５６７８９", "()$,.=+-%0123456789")
UNIT_SUFFIX = re.compile(r"(?<=\d)\s*\(([A-Za-z]+(?:[- ][A-Za-z]+)?)\)\s*$")
EQUATION = re.compile(rf"(?:^|=)\s*([$£€]?{VALUE})\s*([+*×÷/-])\s*([$£€]?{VALUE})\s*=\s*([$£€]?{VALUE})\s*[.!]?\s*$")
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "×": operator.mul,
       "/": operator.truediv, "÷": operator.truediv}
MATH_WORD = re.compile(r"\b(?:add(?:ing|ed|ition)?|subtract(?:ing|ed|ion)?|multipli(?:ed|cation)|multiply(?:ing)?|divid(?:ing|ed|e)|division|combin(?:e|ing|ed))\b", re.I)
IRRELEVANT = re.compile(r"\b(?:another|different|new|instead|ignore|however|actually|should|else|question|example|next|imagine|change|replace|revise)\b", re.I)
COMMON = frozenset("these those their remaining overall number difference problem concept using combining between then total find that them have with answer there from".split())


def normalize_tag(match):
    body = match[2].translate(MATH_PUNCTUATION)
    body = UNIT_SUFFIX.sub(r" \1", body)
    return f"<{match[1]}>{body}</{match[1]}>"


def supported_explanation(text, tag):
    """Allow one short qualitative summary after a locally verified arithmetic result."""
    prefix, tail = text[:tag.start()], text[tag.end():].strip()
    target = scalar(tag[2])
    if (target is None or len(tail) > 200 or not re.fullmatch(r"[A-Za-z ,'-]+[.!]?", tail)
            or not MATH_WORD.search(tail) or UNCERTAIN.search(tail) or IRRELEVANT.search(tail)):
        return False
    # A math word alone could also describe an unrelated new problem.
    before = set(re.findall(r"[a-z]{4,}", prefix.lower()))
    after = set(re.findall(r"[a-z]{4,}", tail.lower()))
    if len((before & after) - COMMON) < 2:
        return False
    # Only count self-contained numeric calculations, not suffixes of longer
    # expressions. The exact result must also equal the answer printed in the tag.
    for line in reversed(prefix.splitlines()[-5:]):
        match = EQUATION.search(line)
        if match:
            a, b, c = (scalar(match[i]) for i in (1, 3, 4))
            if None in (a, b, c) or (not b and match[2] in "/÷"):
                continue
            return c == target and OPS[match[2]](a, b) == c
    return False


def score_training_reward(sample, gold):
    text = TAG.sub(normalize_tag, sample["completion"].strip())
    value, reason = extract_training_answer(text, sample["truncated"])
    if reason == "nonterminal_tag" and not sample["truncated"]:
        tag = list(TAG.finditer(text))[-1]
        if supported_explanation(text, tag):
            value, reason = extract_training_answer(text[:tag.end()], False)
            if value is not None:
                reason = "supported_explanation"
    expected = Fraction(str(gold).replace(",", "").strip())
    return int(value is not None and value == expected), reason
