# src/tests/test_clustering_utils.py

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

# Adjust imports to match your module paths
from src.clustering_utils import (
    compute_metrics_over_k,
    compute_consensus_score,
    assign_clusters,
    run_and_report, run_dbscan
)
from src.clusterer import (
    KMeansClusterer,
    AgglomerativeClusterer,
    GMMClusterer,
    DBSCANClusterer
)


@pytest.fixture
def X():
    """Small synthetic dataset with 3 centers."""
    X, _ = make_blobs(
        n_samples=60,
        centers=3,
        cluster_std=0.5,
        random_state=0
    )
    return X


def test_compute_metrics_over_k_kmeans(X):
    df = compute_metrics_over_k(
        KMeansClusterer,
        X,
        k_range=range(2, 5),
        random_state=0
    )

    # One row per k value
    assert list(df["n_clusters"]) == [2, 3, 4]

    # Common metric columns must exist and not be all NaN
    common = [
        "inertia",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "avg_distance_mean",
        "cluster_frac_mean"
    ]
    for col in common:
        assert col in df.columns
        assert not df[col].isna().all()


def test_compute_consensus_score_and_best_k_map():
    # Toy DataFrame: two ks and two metrics
    df = pd.DataFrame({
        "n_clusters": [2, 3],
        "inertia":    [3.0, 1.0],   # lower is better → pick index 1
        "silhouette": [0.1, 0.9],   # higher is better → pick index 1
    })

    # Get both consensus scores and per-metric best_k_map
    cons, best_k_map = compute_consensus_score(
        df,
        method="borda",
        elbow_metrics=[]  # disable inertia elbow
    )

    # Consensus is a Series indexed by df.index
    assert isinstance(cons, pd.Series)
    assert list(cons.index) == [0, 1]

    # best_k_map covers exactly the two metric columns
    expected = {"inertia", "silhouette"}
    assert set(best_k_map.keys()) == expected

    # Check picks
    assert best_k_map["inertia"] == 1
    assert best_k_map["silhouette"] == 1


def test_assign_clusters_writes_csv(tmp_path, X):
    # Test KMeans branch of assign_clusters
    centroids_csv = tmp_path / "centroids.csv"
    clusters_csv  = tmp_path / "clusters.csv"

    labels, cl = assign_clusters(
        KMeansClusterer,
        X,
        n_clusters=3,
        random_state=0,
        centroids_csv=str(centroids_csv),
        clusters_csv=str(clusters_csv)
    )

    # labels array correct shape
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)

    # virtual centroids shape
    virt = cl.get_virtual_centroids()
    assert virt.shape == (3, X.shape[1])

    # centroids CSV exists and has 3 rows
    df_cent = pd.read_csv(centroids_csv, index_col="cluster")
    assert df_cent.shape == (3, X.shape[1])

    # clusters CSV exists with one row per sample and 'cluster' column
    df_pts = pd.read_csv(clusters_csv)
    assert df_pts.shape[0] == X.shape[0]
    assert "cluster" in df_pts.columns


@pytest.mark.parametrize("clusterer_cls,name", [
    (KMeansClusterer,       "kmeans"),
    (AgglomerativeClusterer, "agglomerative"),
    (GMMClusterer,           "gmm"),
])
def test_run_and_report_generates_outputs(tmp_path, monkeypatch, X, clusterer_cls, name):
    """
    Verify that run_and_report writes:
      - metrics CSV
      - virtual & real centroid CSVs
      - 3×3 metrics plot PNG
    """
    monkeypatch.chdir(tmp_path)

    # Call without passing n_clusters in **clusterer_kwargs,
    # run_and_report uses k_range to instantiate clusterers.
    run_and_report(
        name=name,
        cls=clusterer_cls,
        X=X,
        k_range=range(2, 5),
        random_state=0
    )

    # metrics CSV
    metrics_csv = tmp_path / f"{name}_metrics.csv"
    assert metrics_csv.exists()
    df = pd.read_csv(metrics_csv)
    assert list(df["n_clusters"]) == [2, 3, 4]
    assert "consensus_score" in df.columns

    # centroid CSVs
    virt_csv = tmp_path / f"{name}_virtual_centroids.csv"
    real_csv = tmp_path / f"{name}_real_centroids.csv"
    assert virt_csv.exists()
    assert real_csv.exists()

    # plot file
    plot_png = tmp_path / f"{name}_3x3_metrics.png"
    assert plot_png.exists()


def test_run_and_report_dbscan(tmp_path, monkeypatch, X):
    monkeypatch.chdir(tmp_path)
    # DBSCAN uses eps/min_samples, no k_range
    run_dbscan(
        name="dbscan",
        X=X,
        eps=0.5,
        min_samples=5
    )

    # metrics CSV
    metrics_csv = tmp_path / "dbscan_metrics.csv"
    assert metrics_csv.exists()
    df = pd.read_csv(metrics_csv)
    # DBSCAN yields one row for the single clustering
    assert df.shape[0] >= 1

    # centroid CSVs
    assert (tmp_path / "dbscan_virtual_centroids.csv").exists()
    assert (tmp_path / "dbscan_real_centroids.csv").exists()