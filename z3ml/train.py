"""Utilities for creating training constraints."""


import z3ml


def train(
    x, y, model: z3ml.models.z3ML, loss: z3ml.losses.LossFunction
) -> list[z3ml.utils.Constraints]:
    """The 'training' loop."""
    # Create a symbolic representation of the logits created by the model for
    # each data point.
    logits = model.forward(x)
    # Create constraints that the symbolic logits represent the correct answer.
    return loss(logits, y)
