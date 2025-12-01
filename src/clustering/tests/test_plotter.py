# test_plotter.py

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from src.plotter import plot_all_and_unique_metrics


@pytest.fixture
def sample_metrics_df():
    # create a DataFrame matching compute_metrics_over_k output
    ks = np.arange(1, 6)
    df = pd.DataFrame({
        'n_clusters': ks,
        'inertia': np.linspace(10, 5, 5),
        'silhouette': np.linspace(0.1, 0.9, 5),
        'calinski_harabasz': np.linspace(50, 200, 5),
        'davies_bouldin': np.linspace(1.0, 0.5, 5),
        'avg_distance_mean': np.linspace(5, 1, 5),
        'avg_distance_std': np.linspace(0.5, 0.1, 5),
        'cluster_frac_mean': np.linspace(0.2, 0.8, 5),
        'cluster_frac_std': np.linspace(0.05, 0.01, 5),
    })
    return df


def test_plot_all_and_unique_metrics_basic(sample_metrics_df):
    df = sample_metrics_df
    ks = df['n_clusters'].to_numpy()

    # define dummy unique metrics
    unique_vals = [
        np.linspace(0.1, 0.5, len(ks)),
        np.linspace(0.2, 0.6, len(ks)),
        np.linspace(0.3, 0.7, len(ks)),
    ]
    unique_errs = [
        None,
        np.linspace(0.01, 0.03, len(ks)),
        None,
    ]
    unique_titles = ['U1', 'U2', 'U3']

    fig = plot_all_and_unique_metrics(df, unique_vals, unique_errs, unique_titles)

    # Should create 3x3 axes
    axes = fig.axes
    assert len(axes) == 9

    # Check that all subplots have a vertical span matching k-range
    for ax in axes:
        xlim = ax.get_xlim()
        assert pytest.approx(xlim[0]) == ks.min()
        assert pytest.approx(xlim[1]) == ks.max()

    # Check that bottom row has x-axis labels
    bottom_axes = axes[6:]
    for ax in bottom_axes:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        # at least one non-empty label
        assert any(label != '' for label in labels)

    # Check that explanatory text is present in figure text elements
    texts = [t for t in fig.texts if t.get_text()]
    assert any('Inertia' in t.get_text() for t in texts)
    assert any('U1' in t.get_text() for t in texts)

    plt.close(fig)