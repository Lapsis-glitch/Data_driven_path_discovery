import numpy as np
import pytest
from sklearn.datasets import make_blobs

# Adjust this import if your module path is different:
from src.clusterer import (
    KMeansClusterer,
    AgglomerativeClusterer,
    GMMClusterer,
    DBSCANClusterer,
)


@pytest.fixture
def blob_data():
    """
    Generate a simple 2D blob dataset with 3 centers.
    """
    X, _ = make_blobs(
        n_samples=60,
        centers=3,
        cluster_std=0.5,
        random_state=0
    )
    return X


def test_kmeans_clusterer_basic(blob_data):
    X = blob_data
    km = KMeansClusterer(n_clusters=3, random_state=0)
    km.fit(X)

    # labels_ array
    assert isinstance(km.labels_, np.ndarray)
    assert km.labels_.shape == (X.shape[0],)

    # virtual centroids
    virt = km.get_virtual_centroids()
    assert isinstance(virt, np.ndarray)
    assert virt.shape == (3, X.shape[1])

    # real centroids must lie in the original data
    real = km.get_real_centroids()
    assert real.shape == virt.shape
    for rc in real:
        assert any(np.all(rc == xi) for xi in X)

    # metrics dictionary
    metrics = km.get_metrics()
    for key in (
        'silhouette',
        'calinski_harabasz',
        'davies_bouldin',
        'population',
        'avg_distance'
    ):
        assert key in metrics

    # population counts sum to total samples
    total_pop = sum(metrics['population'].values())
    assert total_pop == X.shape[0]


def test_agglomerative_clusterer_basic(blob_data):
    X = blob_data
    ac = AgglomerativeClusterer(n_clusters=3)
    ac.fit(X)

    # virtual & real centroids
    virt = ac.get_virtual_centroids()
    real = ac.get_real_centroids()
    assert virt.shape == (3, X.shape[1])
    assert real.shape == virt.shape

    # metrics
    metrics = ac.get_metrics()
    for key in (
        'silhouette',
        'calinski_harabasz',
        'davies_bouldin',
        'population',
        'avg_distance'
    ):
        assert key in metrics

    # population counts
    assert sum(metrics['population'].values()) == X.shape[0]


def test_gmm_clusterer_basic(blob_data):
    X = blob_data
    gm = GMMClusterer(n_clusters=3, random_state=0)
    gm.fit(X)

    # underlying GMM model and centroids_
    assert hasattr(gm, 'model')
    assert hasattr(gm, 'centroids_')

    virt = gm.get_virtual_centroids()
    assert isinstance(virt, np.ndarray)
    assert virt.shape == (3, X.shape[1])

    real = gm.get_real_centroids()
    assert real.shape == virt.shape

    # metrics
    metrics = gm.get_metrics()
    for key in (
        'silhouette',
        'calinski_harabasz',
        'davies_bouldin',
        'population',
        'avg_distance'
    ):
        assert key in metrics

    # population counts
    assert sum(metrics['population'].values()) == X.shape[0]


def test_dbscan_clusterer_basic(blob_data):
    X = blob_data
    db = DBSCANClusterer(eps=0.7, min_samples=5)
    db.fit(X)

    # labels_ for each sample
    labels = db.labels_
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)

    # metrics
    metrics = db.get_metrics()
    for key in (
        'silhouette',
        'calinski_harabasz',
        'davies_bouldin',
        'population',
        'avg_distance'
    ):
        assert key in metrics

    # virtual & real centroids shape agreement
    virt = db.get_virtual_centroids()
    real = db.get_real_centroids()
    assert virt.shape == real.shape
    assert virt.ndim == 2


def test_dbscan_all_noise(blob_data):
    X = blob_data
    db = DBSCANClusterer(eps=0.01, min_samples=5)  # too small eps → all noise
    db.fit(X)

    # Expect zero centroids
    virt = db.get_virtual_centroids()
    real = db.get_real_centroids()
    assert virt.shape[0] == 0
    assert real.shape[0] == 0

    # Silhouette should be NaN when fewer than 2 clusters
    metrics = db.get_metrics()
    assert np.isnan(metrics['silhouette'])