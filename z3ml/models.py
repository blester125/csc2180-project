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
        return (logits > 0).astype(np.int32)


class z3OneVsOne(z3ML):
    def __init__(
        self, classes: list, model_factory: type[z3ML] = z3Linear, *args, **kwargs
    ):
        super().__init__()
        # {(c1, c2): c1-vs-c2-classifier}
        self.models = {}
        # The list classes we are classifying.
        self.classes = classes
        # A mapping from 0, 1 -> the original classes.
        self.idx = []
        for c1, c2 in self.enumerate_classes():
            # Create a model for each class combo.
            self.models[(c1, c2)] = model_factory(*args, **kwargs)
            # Track the 0, 1 translation.
            self.idx.append([c1, c2])
        # idx[m.predict(x)] will translate back to og labels
        self.idx = np.array(self.idx)
        # one_hot[pred] will give the one-hot version.
        self.one_hot = np.eye(len(classes))

    def forward(self, x):
        if isinstance(x, dict):
            return {c: self.models[c].forward(x_) for c, x_ in x.items()}
        return {c: m.forward(x) for c, m in self.models.items()}

    def realize(self, models):
        self.models = {k: m.realize(models[k]) for k, m in self.models.items()}
        return self

    def predict(self, x):
        votes = [model.predict(x) for model in self.models.values()]  # list[B]
        # Stack the 0,1 output of each classifier in the one-vs-one
        votes = np.stack(votes, axis=0)  # [V, B]
        # Convert to original labels and to one hot
        votes = np.take_along_axis(self.idx, votes, axis=-1)
        votes = self.one_hot[votes]
        # Sum over the classifier votes and then argmax to get most voted class.
        return np.argmax(np.sum(votes, axis=0), axis=-1)

    def enumerate_classes(self):
        for i, c1 in enumerate(self.classes):
            for c2 in self.classes[i + 1 :]:
                yield c1, c2

    def filter_data(self, x, y):
        xs = {}
        ys = {}
        for c1, c2 in self.enumerate_classes():
            idx = (y == c1) | (y == c2)
            y_ = y[idx]
            y_ = np.where(y_ == c1, 0, 1)  # Convert to 0, 1
            xs[(c1, c2)] = x[idx]
            ys[(c1, c2)] = y_
        return xs, ys


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
