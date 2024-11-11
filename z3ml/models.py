"""ML models implemented in z3."""

import abc

import numpy as np
import z3

from z3ml import utils


def _realize(param, model):
    """Convert an np.array of z3 variables to an np.array of values."""
    return np.array([utils.get_value(model[p]) for p in param.ravel()]).reshape(
        *param.shape
    )


class z3ML(metaclass=abc.ABCMeta):
    """z3 ML Model base class."""

    def __init__(self, *args, **kwargs):
        self.parameters = {}
        self.realized = False

    @abc.abstractmethod
    def forward(self, x):
        """Calculate the forward pass, returning the logits."""

    @abc.abstractmethod
    def predict(self, x):
        """Run the model outputting a final prediction."""

    def realize(self, model):
        """Convert the z3 parameters to real np ones."""
        self.parameters = {k: _realize(v, model) for k, v in self.parameters.items()}
        return self


class z3Linear(z3ML):
    """z3 ML Model with binary output."""

    def __init__(self, n_features: int, dtype=z3.Real):
        super().__init__()
        self.parameters["w"] = np.array([dtype(f"w_{i}") for i in range(n_features)])
        self.parameters["b"] = np.array([dtype("b")])

    def forward(self, x):
        return x @ self.parameters["w"] + self.parameters["b"]

    def predict(self, x):
        logits = self.forward(x)
        return logits > 0


class z3MultiClassLinear(z3Linear):
    """z3 ML Model with multiclass outputs."""

    def __init__(self, n_features: int, n_output: int, dtype=z3.Real):
        # Ugly, think about later.
        self.parameters = {}
        self.realized = False

        self.parameters["w"] = np.array(
            [[dtype(f"w_{i},{j}") for j in range(n_output)] for i in range(n_features)]
        )
        self.parameters["b"] = np.array([dtype(f"b_{i}") for i in range(n_output)])

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)


class z3MLP(z3MultiClassLinear):
    """z3 ML version of a MLP."""

    def __init__(
        self, n_features: int, n_hidden: list[int] | int, n_output: int, dtype=z3.Real
    ):
        # Ugly, think about later.
        self.parameters = {}
        self.realized = False

        n_hidden = utils.listify(n_hidden)

        for i, (in_, out) in enumerate(
            zip([n_features, *n_hidden], [*n_hidden, n_output])
        ):
            self.parameters[f"w_{i}"] = np.array(
                [[dtype(f"w_{i},{j},{k}") for k in range(out)] for j in range(in_)]
            )
            self.parameters[f"b_{i}"] = np.array(
                [dtype(f"b_{i},{j}") for j in range(out)]
            )
        self.num_layers = i + 1
        self.ws, self.bs = self._index_parameters()
        self.relu = utils.relu

    def _index_parameters(self):
        ws = [self.parameters[f"w_{i}"] for i in range(self.num_layers)]
        bs = [self.parameters[f"b_{i}"] for i in range(self.num_layers)]
        return ws, bs

    def forward(self, x):
        for w, b in zip(self.ws[:-1], self.bs[:-1]):
            x = x @ w + b
            x = self.relu(x)
        return x @ self.ws[-1] + self.bs[-1]

    def realize(self, model):
        super().realize(model)
        self.relu = lambda x: np.maximum(x, 0)
        self.ws, self.bs = self._index_parameters()
        return self
