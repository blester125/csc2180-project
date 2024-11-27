"""Tools for visualizing decision boundaries."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

COLORS = np.array(list(mcolors.TABLEAU_COLORS.keys()))


def plot(x, y, colors=COLORS, ax=None):
    """Scatter plot (2d) x points and color according to y."""
    if ax is None:
        ax = plt.gca()
    ax.scatter(*x.T, color=colors[y])
    return ax


def plot_boundary(
    x,
    y,
    model,
    h: float = 0.01,
    alpha: float = 0.5,
    ax=None,
    colors=COLORS,
    num_classes: int = -1,
):
    """Plot and color the areas of x-space based on predictions from model(x)."""
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = model(np.vstack([xx.ravel(), yy.ravel()]).T)
    z = z.reshape(xx.shape)
    if ax is None:
        ax = plt.gca()
    if num_classes == -1:
        num_classes = len(set(y))
    ax.contourf(
        xx, yy, z, levels=list(range(-1, num_classes)), colors=colors, alpha=alpha
    )
    return ax


# So we don't need to fix all my typos yet lol
plot_boundry = plot_boundary


def plot_unsat(x, y, unsat, colors=COLORS, edgecolor="k", ax=None):
    """Scatter plot points but outline unsat points."""
    ax = plot(x, y, colors, ax)
    ax.scatter(*x[unsat].T, color=colors[y[unsat]], edgecolor=edgecolor)
    return ax


def plot_predict(x, y, y_pred, colors=COLORS, ax=None):
    """Scatter plot points with true labels as colors and predictions as edges."""
    ax = plot(x, y, colors, ax)
    wrong = y != y_pred
    ax.scatter(*x[wrong].T, color=colors[y[wrong]], edgecolor=colors[y_pred[wrong]])
    return ax
