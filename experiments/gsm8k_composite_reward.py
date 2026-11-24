"""Training-only credit for exact arithmetic relations in annotated GSM8K solutions."""

import ast
from fractions import Fraction
from functools import lru_cache
import operator
import re

from experiments.gsm8k_step_reward import (
    ANNOTATION, ANSWER_START, equations, reference_steps,
    score_training_reward as simple_reward,
)
from experiments.gsm8k_training_reward_v2 import score_training_reward as outcome_reward


OPERATIONS = {
    ast.Add: ('+', operator.add), ast.Sub: ('-', operator.sub),
    ast.Mult: ('*', operator.mul), ast.Div: ('/', operator.truediv),
}
MATH_TEXT = re.compile(r'[0-9+*/=().\-\t ]+')
FRACTION = re.compile(r'\\(?:[td]?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
SYMBOLS = str.maketrans({'×': '*', '÷': '/', '−': '-', '$': '', '£': '', '€': ''})
LATEX = {r'\times': '*', r'\cdot': '*', r'\div': '/', r'\left': '', r'\right': '',
         r'\,': '', r'\(': ' ', r'\)': ' ', r'\[': ' ', r'\]': ' ', r'\%': '%'}


def normalize(text):
    text = text.translate(SYMBOLS)
    for source, replacement in LATEX.items():
        text = text.replace(source, replacement)
    for _ in range(3):
        text = FRACTION.sub(r'((\1)/(\2))', text)
    text = re.sub(r'(?<=\d)\s*[xX]\s*(?=[\d(])', '*', text)
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
    return re.sub(r'(?<!\w)(\d+(?:\.\d+)?)\s*%', r'(\1/100)', text)


def operation_key(symbol, left, right):
    if symbol in ('+', '*'):
        operands = []
        for term in (left, right):
            operands.extend(term[1:] if term[0] == symbol else (term,))
        return (symbol, *sorted(operands))
    return (symbol, left, right)


def parse_expression(source):
    source = source.strip().rstrip('.')
    if not source or len(source) > 130 or not re.fullmatch(r'[\d\s()+*/.\-]+', source):
        raise ValueError('invalid arithmetic expression')

    def visit(node, depth):
        if depth > 10:
            raise ValueError('too many nested operations')
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(source, node)
            if len(literal) > 30:
                raise ValueError('numeric literal too long')
            value = Fraction(literal)
            return value, ('n', str(value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value, key = visit(node.operand, depth + 1)
            if isinstance(node.op, ast.UAdd):
                return value, key
            return -value, ('n', str(-value)) if key[0] == 'n' else ('neg', key)
        if not isinstance(node, ast.BinOp) or type(node.op) not in OPERATIONS:
            raise ValueError('unsupported operation')
        lhs, left = visit(node.left, depth + 1)
        rhs, right = visit(node.right, depth + 1)
        symbol, calculate = OPERATIONS[type(node.op)]
        value = calculate(lhs, rhs)
        # Inside another operation, a literal fraction can also be written as a decimal.
        if depth and symbol == '/' and left[0] == right[0] == 'n':
            return value, ('n', str(value))
        return value, operation_key(symbol, left, right)

    return visit(ast.parse(source, mode='eval').body, 0)


def relations(text):
    """Extract exact expression–scalar equalities, retaining distinct operation trees."""
    found = set()
    for line in normalize(text).splitlines():
        for match in MATH_TEXT.finditer(line):
            start, end = match.span()
            previous, following = line[max(0, start - 1):start], line[end:end + 1]
            parts = match[0].split('=')
            for index, (left, right) in enumerate(zip(parts, parts[1:])):
                # A variable or unsupported operator can invalidate one side of a
                # chain while leaving a later self-contained numeric equality intact.
                if index == 0 and (previous in ('^', '%', '_')
                                   or (previous.isalpha() and left.lstrip().startswith(('+', '-')))):
                    continue
                if index == len(parts) - 2 and following in ('^', '%', '_'):
                    continue
                try:
                    lhs, a = parse_expression(left)
                    rhs, b = parse_expression(right)
                except (ValueError, ZeroDivisionError, SyntaxError, RecursionError, OverflowError):
                    continue
                if lhs != rhs:
                    continue
                if a[0] != 'n' and b[0] == 'n':
                    found.add((a, str(rhs)))
                elif b[0] != 'n' and a[0] == 'n':
                    found.add((b, str(lhs)))
    return frozenset(found)


def simple_key(equation):
    left, symbol, right, result = equation
    return operation_key(symbol, ('n', str(left)), ('n', str(right))), str(result)


@lru_cache(maxsize=8192)
def references(rationale):
    binary = reference_steps(rationale)
    composite = frozenset().union(*(relations(fragment) for fragment in ANNOTATION.findall(rationale)))
    distinct = composite | frozenset(simple_key(eq) for eq in binary)
    return binary, composite, distinct


def matched_steps(sample, example):
    binary, composite, distinct = references(example['rationale'])
    body = ANSWER_START.split(sample['completion'], maxsplit=1)[0]
    old = frozenset(simple_key(eq) for eq in equations(body) & binary)
    return old | (relations(body) & composite), distinct


def partial_credit(matched, reference, reason):
    fraction = len(matched) / max(1, len(reference))
    return 0.4 * fraction, f'{reason}/composite={len(matched)}/{len(reference)}'


def score_training_reward(sample, example):
    """Score one response independently. The online two-sample recipe uses the group API."""
    correct, reason = outcome_reward(sample, example['value'])
    if correct:
        return 1.0, reason
    matched, reference = matched_steps(sample, example)
    return partial_credit(matched, reference, reason)


def score_group_training_reward(samples, example):
    """Keep the binary-step ordering, breaking a two-wrong tie only on strict containment."""
    baseline = [simple_reward(sample, example) for sample in samples]
    if len(baseline) == 1:  # The token cap may leave only one response at the end.
        return baseline
    if len(baseline) != 2:
        raise ValueError('expected two responses or one budget-limited tail response')
    if baseline[0][0] != baseline[1][0] or baseline[0][0] == 1:
        return baseline
    first, reference = matched_steps(samples[0], example)
    second, _ = matched_steps(samples[1], example)
    if not (first > second or second > first):
        return baseline
    return [partial_credit(matched, reference, prior[1].partition('/steps=')[0])
            for matched, prior in zip((first, second), baseline)]
