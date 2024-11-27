"""z3 versions of ML losses.

Losses return a list of lists of constraints (first list elements are the loss
constraints that apply to a particular example).
"""

from typing import Protocol

import numpy as np
import z3

from z3ml import utils


class LossFunction(Protocol):
    def __call__(self, logits, y) -> list[utils.Constraints]:
        """Loss Function type signature."""


def threshold_loss(logits, y, threshold: float = 0) -> list[utils.Constraints]:
    """Constraints that force each logit to be above the threshold for the positive class.

    This is the loss to use for a 2 class classifier.

    Note: This is a single list of a constraints, it isn't separated by example.
    """
    return [
        np.vectorize(lambda l, y: z3.If(bool(y == 0), l < threshold, l > threshold))(
            logits, y
        )
    ]


def one_vs_one_loss(logits, y, loss: LossFunction = threshold_loss):
    return {c: loss(l, y[c]) for c, l in logits.items()}


def multiclass_loss(logits, y) -> list[utils.Constraints]:
    """Constraint that says the argmax variable needs to be assigned to the correct class index.

    Losses for multiclass classification.

    Note:
      Expects sparse (mutually-exclusive) labels.
    """
    constraints = []
    for i, (l, y_) in enumerate(zip(logits, y)):
        loss_constraints = []
        idx = z3.Int(f"argmax_{i}")
        max_val = z3.Real(f"max_{i}")
        loss_constraints.extend(utils.maximum(l, max_val))
        loss_constraints.extend(utils.argmax(l, idx, max_val))
        loss_constraints.append(idx == y_)
        constraints.append(loss_constraints)
    return constraints
