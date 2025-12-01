# test_metrics.py

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage
from sklearn.mixture import GaussianMixture

import src.metrics as metrics
from src.clusterer import KMeansClusterer


@pytest.fixture
def blob_data():
    X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=0)
    return X


def test_compute_cophenetic_correlation_perfect_line():
    # 1D evenly spaced points → perfect cophenetic correlation
    X = np.arange(10).reshape(-1, 1)
    corr = metrics.compute_cophenetic_correlation(X, method='single')
    assert pytest.approx(corr, rel=1e-6) == 1.0


def test_compute_inconsistency_and_dendrogram_stats(blob_data):
    X = blob_data
    inc = metrics.compute_inconsistency_stats(X, method='ward', metric='euclidean')
    assert set(inc.keys()) == {'inconsistency_mean', 'inconsistency_std', 'inconsistency_max'}
    assert inc['inconsistency_std'] >= 0
    dh = metrics.compute_dendrogram_cut_height_stats(X)
    assert set(dh.keys()) == {'merge_height_mean', 'merge_height_std', 'merge_height_max'}
    assert dh['merge_height_max'] >= dh['merge_height_mean']


def test_compute_gap_statistic_trend(blob_data):
    X = blob_data
    # gap for true k=3 should exceed gap for k=4 (worse fit)
    gap3 = metrics.compute_gap_statistic(X, KMeansClusterer, k=3, B=5, random_state=0)
    gap4 = metrics.compute_gap_statistic(X, KMeansClusterer, k=4, B=5, random_state=0)
    assert isinstance(gap3, float)
    assert isinstance(gap4, float)
    assert gap3 >= gap4


def test_wcss_per_cluster_and_unbalanced():
    X = np.array([[0,0],[1,1],[10,10],[11,11]])
    labels = np.array([0,0,1,1])
    cents  = np.array([[0.5,0.5],[10.5,10.5]])
    wcss = metrics.compute_wcss_per_cluster(X, labels, cents)
    # each cluster sum of squares = 0.5 + 0.5 = 1.0? Actually each pair contributes .5 per point, sum=1.0
    assert pytest.approx(wcss[0]) == 0.5 + 0.5
    assert pytest.approx(wcss[1]) == 0.5 + 0.5

    ub1 = metrics.compute_unbalanced_factor(np.array([0,0,1]))
    # largest cluster=2, smallest=1 → factor=2.0
    assert pytest.approx(ub1) == 2.0
    # single cluster or empty → nan
    ub2 = metrics.compute_unbalanced_factor(np.array([0,0,0]))
    assert np.isnan(ub2)


def test_gmm_bic_aic_and_avg_log_lik(blob_data):
    X = blob_data
    gm = GaussianMixture(n_components=3, random_state=0).fit(X)
    bic = metrics.compute_gmm_bic(gm, X)
    aic = metrics.compute_gmm_aic(gm, X)
    avgll = metrics.compute_avg_log_likelihood_per_component(gm, X)
    assert isinstance(bic, float)
    assert isinstance(aic, float)
    # one entry per component
    assert set(avgll.keys()) == {0, 1, 2}
    for val in avgll.values():
        assert isinstance(val, float)


def test_dbscan_noise_core_border():
    # labels: two noise, two core, one border
    labels = np.array([0,0,1,-1,-1])
    core_idx = [0,2]
    stats = metrics.compute_dbscan_noise_core_border(labels, core_idx)
    assert pytest.approx(stats['noise_proportion']) == 2/5
    assert pytest.approx(stats['core_proportion']) == 2/5
    assert pytest.approx(stats['border_proportion']) == 1/5