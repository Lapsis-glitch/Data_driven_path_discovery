# run.py

import matplotlib.pyplot as plt
from src.clustering_utils import run_method, save_cluster_labels, assign_clusters
from src.clusterer import (
    KMeansClusterer,
    AgglomerativeClusterer,
    GMMClusterer,
    DBSCANClusterer,
)
from sklearn.datasets import make_blobs


def main():
    # 1) Generate the dataset
    X, _ = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.6,
        random_state=42
    )

    # 2) Optional: visualize the dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
    plt.title("Synthetic Dataset")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("dataset.png")
    plt.close()

    # 3) Define k‐range
    k_range = range(1, 5)

    # 4) Run clustering pipelines
    run_method(
        name="kmeans_sieved",
        clusterer_cls=KMeansClusterer,
        X=X,
        k_range=range(2, 8),
        sieve=10,  # 10% of X used to fit
        random_state=42,
    )
    run_method(
        name="kmeans",
        clusterer_cls=KMeansClusterer,
        X=X,
        k_range=range(2, 8),
        random_state=42,
    )

    run_method(
        name="agglo",
        clusterer_cls=AgglomerativeClusterer,
        X=X,
        k_range=k_range
    )
    run_method(
        name="gmm",
        clusterer_cls=GMMClusterer,
        X=X,
        k_range=k_range,
        random_state=42
    )
    run_method(
        name="dbscan",
        clusterer_cls=DBSCANClusterer,
        X=X,
        eps=0.8,
        min_samples=10
    )

    # 5) Save labels for k=3

    labels = assign_clusters(
        KMeansClusterer,
        X,
        sieve_frac=0.1,
        random_state=0,
        n_clusters=5  # must pass k here
    )

    save_cluster_labels(X, labels, filepath="k4_labels.csv")



if __name__ == "__main__":
    main()
