import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_elbow(X, k_range=range(1, 11), **km_kwargs):
    """
    Elbow plot of inertia vs. k for KMeans.
    """
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, **km_kwargs)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(list(k_range), inertias, "bx-")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal k")
    return fig


def plot_dendrogram(X, method="ward", **linkage_kwargs):
    """
    Dendrogram for Agglomerative Clustering.
    """
    Z = linkage(X, method=method, **linkage_kwargs)
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(Z, ax=ax)
    ax.set_title(f"Dendrogram ({method.capitalize()})")
    return fig