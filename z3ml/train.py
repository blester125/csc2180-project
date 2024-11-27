"""Utilities for creating training constraints."""

import numpy as np

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


def one_vs_one(
    x,
    y,
    model: z3ml.models.z3ML,
    loss: z3ml.losses.LossFunction = z3ml.losses.one_vs_one_loss,
) -> list[z3ml.models.z3ML]:
    x, y = model.filter_data(x, y)
    return train(x, y, model, loss)
