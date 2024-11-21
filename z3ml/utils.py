"""Utilities for working with z3."""

import collections.abc
from typing import Any

import numpy as np
import z3

Constraint = Any
Constraints = list[Constraint]


def get_value(x):
    """Convert z3 variables to python data."""
    if z3.is_int_value(x):
        return x.as_long()
    if z3.is_rational_value(x):
        return x.numerator_as_long() / x.denominator_as_long()
    if z3.is_algebraic_value(x):
        return x.approx()
    if x == None:
        return 0
    raise ValueError(f"I don't know what {x} is.")


def argmax(logits, idx, max_val) -> Constraints:
    """Create constraints to that idx has to assigned with the index of the largest logit.

    Based on http://www.hakank.org/z3/argmax.py

    Note: This only works when *one* of the logits is the max, it UNSAT's when
          multiple logits have the *exact* same value.
    """
    constraints = []
    # Make sure that the max value is one of the logits.
    # This is duplicated from the max if you use both?
    constraints.append(z3.Or([max_val == l for l in logits]))
    for i, logit in enumerate(logits):
        constraints.append((idx == i) == (logit == max_val))
    return constraints


def maximum(logits, max_val) -> Constraints:
    """Create constraints so that max_val has to be assigned the largest value in logits.

    Based on http://www.hakank.org/z3/max.py

    Note: This only works when *one* of the logits is the max, it UNSAT's when
          multiple logits have the *exact* same value.
    """
    constraints = []
    # Make sure the the max value is one of the logits.
    constraints.append(z3.Or([max_val == l for l in logits]))
    for logit in logits:
        constraints.append(max_val >= logit)
    return constraints


def relu(x) -> Constraints:
    """max(0, x)

    Based on: https://gist.github.com/philzook58/5aab67b65b476bb55e6b9c403ccabed2
    """
    return np.vectorize(lambda y: z3.If(y >= 0, y, z3.RealVal(0)))(x)


def is_sequence(x: Any) -> bool:
    """Is `x` a sequence type?"""
    if isinstance(x, str):
        return False
    return isinstance(x, (collections.abc.Sequence, collections.abc.MappingView))


def listify(x: list[Any] | Any) -> list[Any]:
    """Make sure `x` is a sequence type."""
    if is_sequence(x) or isinstance(x, np.ndarray):
        return x
    return [x] if x is not None else []
