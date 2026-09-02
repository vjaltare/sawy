"""
Yin-Yang dataset generator -- a small, 2D, non-linearly-separable
classification dataset (Kriener et al., "The Yin-Yang dataset", 2021 style).
It's a good interpretability testbed: because the input is 2D you can
scatter-plot the raw data and overlay any trained network's decision
boundary directly, and because the swirl boundary is curved (not a single
hyperplane, not even XOR-style axis-aligned quadrants) a network with a
single hidden layer of modest width typically cannot draw it cleanly --
two hidden layers make it easy. That makes it a nice small stand-in for
CIFAR-10 when the goal is inspecting *what the network's activations and
boundary look like* rather than benchmarking.

Geometry
--------
Three classes live inside a disk of radius r_big centered at
(r_big, r_big), so the whole dataset fits in the square
[0, 2*r_big] x [0, 2*r_big] (defaults to the unit square with r_big=0.5):

  - two small "eye" dots of radius r_small, centered at
    (0.5*r_big, r_big) and (1.5*r_big, r_big)   -> class 2 ("dot")
  - the classic S-shaped swirl boundary, built from two semicircles of
    radius r_big/2 centered at those same two points, splits the rest of
    the disk into the "yin" and "yang" swirls -> classes 0 and 1

Classes are exactly balanced by rejection sampling: cycle through the
target class and keep resampling uniformly in the disk until a point of
that class turns up.
"""
__all__ = ["classify_point", "generate_yin_yang", "get_train_test", "plot_dataset", "plot_decision_boundary"]

import jax
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm



CLASS_NAMES = ["yin", "yang", "dot"]
CLASS_COLORS = ["#3b3b58", "#f2c14e", "#d1495b"]  # dark / light / accent



def _boundary_v(u, r_big):
    """
    v-height of the yin/yang swirl boundary at horizontal offset `u`
    (centered coordinates: u, v are relative to the disk center).
    Point-symmetric about the origin: boundary_v(-u) == -boundary_v(u),
    which is exactly what makes the swirl a proper yin-yang (180-degree
    rotation swaps yin and yang).
    """
    r_half = r_big / 2.0
    # u <= 0: upper half of the small circle centered at (-r_half, 0)
    left = np.sqrt(np.maximum(r_half ** 2 - (u + r_half) ** 2, 0.0))
    # u > 0: lower half of the small circle centered at (r_half, 0)
    right = -np.sqrt(np.maximum(r_half ** 2 - (u - r_half) ** 2, 0.0))
    return np.where(u <= 0, left, right)


def classify_point(x, y, r_small=0.1, r_big=0.5):
    """
    Classify (x, y) point(s) that already lie inside the big circle.
    Returns 0 (yin), 1 (yang), or 2 (dot). Vectorized over arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    u = x - r_big
    v = y - r_big
    d_left = np.sqrt((x - 0.5 * r_big) ** 2 + (y - r_big) ** 2)
    d_right = np.sqrt((x - 1.5 * r_big) ** 2 + (y - r_big) ** 2)
    is_dot = (d_left <= r_small) | (d_right <= r_small)
    is_yang = v > _boundary_v(u, r_big)
    label = np.where(is_yang, 1, 0)
    label = np.where(is_dot, 2, label)
    return label


def generate_yin_yang(n_samples=1000, r_small=0.1, r_big=0.5, seed=0):
    """
    Generate a class-balanced Yin-Yang dataset.

    Returns
    -------
    X : (n_samples, 2) float32 array of (x, y) in [0, 2*r_big]^2
    y : (n_samples,) int32 array of class labels in {0, 1, 2}
        0 = yin, 1 = yang, 2 = dot
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, 2), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int32)

    for i in range(n_samples):
        goal_class = i % 3
        while True:
            x, yy = rng.uniform(0.0, 2 * r_big, size=2)
            if (x - r_big) ** 2 + (yy - r_big) ** 2 > r_big ** 2:
                continue  # outside the big circle -- resample
            c = classify_point(x, yy, r_small=r_small, r_big=r_big)
            if int(c) == goal_class:
                break
        X[i] = (x, yy)
        y[i] = c

    # Shuffle so the class-cycling used for rejection sampling isn't visible
    # in the ordering of the returned arrays.
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def get_train_test(n_train=5000, n_test=1000, r_small=0.1, r_big=0.5, seed=0):
    """Convenience wrapper: independent, differently-seeded train/test sets."""
    X_train, y_train = generate_yin_yang(n_train, r_small, r_big, seed=seed)
    X_test, y_test = generate_yin_yang(n_test, r_small, r_big, seed=seed + 1)
    return X_train, y_train, X_test, y_test


# -------------------------------------------------
# Visualization helpers
# -------------------------------------------------
def plot_dataset(X, y, ax=None, title="Yin-Yang dataset", s=8):
    # import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    for c, name, color in zip(range(3), CLASS_NAMES, CLASS_COLORS):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], s=s, color=color, label=name, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return ax


def plot_decision_boundary(predict_fn, X=None, y=None, ax=None, resolution=300,
                            title="Decision boundary"):
    """
    predict_fn: callable mapping an (N, 2) float32 array of grid points to
    an (N,) array of predicted integer classes in {0, 1, 2}.

    Evaluates predict_fn on a dense grid over [0, 1]^2, shades the
    predicted regions, and (optionally) overlays the dataset points.
    """
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import ListedColormap, BoundaryNorm

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    xx, yy = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    preds = np.asarray(predict_fn(grid)).reshape(xx.shape)

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.pcolormesh(xx, yy, preds, cmap=cmap, norm=norm, shading="auto", alpha=0.35)

    if X is not None and y is not None:
        plot_dataset(X, y, ax=ax, title=title, s=6)
    else:
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return ax


if __name__ == "__main__":
    # import matplotlib
    # matplotlib.use("Agg")
    # import matplotlib.pyplot as plt

    X_train, y_train, X_test, y_test = get_train_test(n_train=5000, n_test=1000, seed=0)
    print(f"Train: {X_train.shape}, class counts: {np.bincount(y_train)}")
    print(f"Test:  {X_test.shape}, class counts: {np.bincount(y_test)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_dataset(X_train, y_train, ax=axes[0], title="Train")
    plot_dataset(X_test, y_test, ax=axes[1], title="Test")
    fig.tight_layout()
    fig.savefig("yin_yang_dataset.png", dpi=150)
    print("Saved preview to yin_yang_dataset.png")