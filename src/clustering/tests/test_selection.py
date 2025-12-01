# src/tests/test_selection.py

import numpy as np
import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import make_blobs

from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from src.selection import plot_elbow, plot_dendrogram


def test_plot_elbow_outputs_figure(tmp_path):
    X, _ = make_blobs(n_samples=50, centers=3, random_state=0)
    fig = plot_elbow(X, k_range=range(1, 6), random_state=0)

    # Should return a Figure or an Axes object
    assert hasattr(fig, "savefig") or hasattr(fig, "figure")

    # There should be at least one line in the elbow plot
    axes = fig.axes if hasattr(fig, "axes") else [fig]
    total_lines = sum(len(ax.get_lines()) for ax in axes)
    assert total_lines >= 1

    plt.close(fig)


def test_plot_dendrogram_outputs_figure(tmp_path):
    X, _ = make_blobs(n_samples=30, centers=2, random_state=1)
    fig = plot_dendrogram(X, method="ward")

    # Should return a Figure or an Axes object
    assert hasattr(fig, "savefig") or hasattr(fig, "figure")

    axes = fig.axes if hasattr(fig, "axes") else [fig]
    has_dendro_artists = False

    # SciPy's dendrogram typically uses Line2D or LineCollection
    for ax in axes:
        for child in ax.get_children():
            if isinstance(child, (Line2D, LineCollection)):
                has_dendro_artists = True
                break
        if has_dendro_artists:
            break

    assert has_dendro_artists, "Expected Line2D or LineCollection in dendrogram axes"
    plt.close(fig)