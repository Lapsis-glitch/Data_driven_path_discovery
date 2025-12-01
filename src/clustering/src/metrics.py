import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage, inconsistent
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances


def compute_silhouette(X, labels):
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    return np.nan


def compute_calinski_harabasz(X, labels):
    if len(set(labels)) > 1:
        return calinski_harabasz_score(X, labels)
    return np.nan


def compute_davies_bouldin(X, labels):
    if len(set(labels)) > 1:
        return davies_bouldin_score(X, labels)
    return np.nan


def cluster_population_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def average_distance_to_centroids(X, labels, centroids):
    if centroids is None:
        return {}
    distances = {}
    for idx, center in enumerate(centroids):
        pts = X[labels == idx]
        if len(pts) > 0:
            distances[idx] = np.mean(np.linalg.norm(pts - center, axis=1))
        else:
            distances[idx] = np.nan
    return distances

# ── Hierarchical ───────────────────────────────────────────────────────────────

def compute_cophenetic_correlation(X, method='ward', metric='euclidean'):
    """
    Cophenetic correlation between pairwise distances in X and dendrogram.
    """
    Z = linkage(X, method=method, metric=metric)
    coph_corr, _ = cophenet(Z, pdist(X, metric))
    # if the denominator was zero (constant distances), treat as perfect correlation
    if np.isnan(coph_corr):
        return 1.0
    return float(coph_corr)


def compute_inconsistency_stats(X, method='ward', metric='euclidean'):
    """
    Inconsistency of each merge in the linkage:
      returns mean, std, and max of the inconsistency column.
    """
    Z = linkage(X, method=method, metric=metric)
    inc = inconsistent(Z)  # shape (n-1, 4)
    vals = inc[:, 3]       # the “inconsistency” values
    return {
        'inconsistency_mean': float(np.mean(vals)),
        'inconsistency_std':   float(np.std(vals)),
        'inconsistency_max':   float(np.max(vals)),
    }


def compute_dendrogram_cut_height_stats(X, method='ward', metric='euclidean'):
    """
    Statistics on the distribution of merge heights in the dendrogram,
    useful to pick a cut.
    """
    Z = linkage(X, method=method, metric=metric)
    heights = Z[:, 2]
    return {
        'merge_height_mean': float(np.mean(heights)),
        'merge_height_std':  float(np.std(heights)),
        'merge_height_max':  float(np.max(heights)),
    }

# ── KMeans ─────────────────────────────────────────────────────────────────────

def compute_gap_statistic(X, clusterer_cls, k, B=10, random_state=None):
    """
    Gap statistic for KMeans (or any BaseClusterer subclass that defines centroids_):
      G(k) = E*[log(W_k*)] – log(W_k),
    where W_k is within-cluster dispersion.

    Uses the clusterer_cls wrapper—relies on BaseClusterer.fit having
    populated `.centroids_` and `.labels_`.
    """
    # 1) Fit on the real data
    km = clusterer_cls(n_clusters=k, random_state=random_state)
    km.fit(X)

    # Access the cluster labels and centroids from the wrapper
    labels = km.labels_
    cents  = km.centroids_  # THIS IS NOW POPULATED BY BaseClusterer.fit()

    # Compute within‐cluster sum of squares Wk
    Wk = 0.0
    for i in range(k):
        pts = X[labels == i]
        if len(pts):
            Wk += np.sum((pts - cents[i])**2)

    # 2) Generate B reference datasets and compute Wk*
    rng    = np.random.RandomState(random_state)
    mins   = X.min(axis=0)
    maxs   = X.max(axis=0)
    W_refs = []
    for _ in range(B):
        Xb = rng.uniform(mins, maxs, size=X.shape)
        km_b = clusterer_cls(n_clusters=k, random_state=rng.randint(1e6))
        km_b.fit(Xb)
        lblb = km_b.labels_
        c_b  = km_b.centroids_
        Wb   = 0.0
        for j in range(k):
            ptsb = Xb[lblb == j]
            if len(ptsb):
                Wb += np.sum((ptsb - c_b[j])**2)
        W_refs.append(Wb)

    # 3) Gap = E[log(Wb)] – log(Wk)
    gap = np.log(np.mean(W_refs)) - np.log(Wk) if Wk > 0 else np.nan
    return float(gap)



def compute_wcss_per_cluster(X, labels, centroids):
    """
    Returns dict {cluster_id: within-cluster sum of squares}.
    """
    wcss = {}
    for idx, c in enumerate(centroids):
        pts = X[labels == idx]
        wcss[idx] = float(np.sum((pts - c)**2)) if len(pts) else 0.0
    return wcss


def compute_unbalanced_factor(labels):
    """
    Ratio of largest cluster size to smallest non-empty cluster size.
    """
    unique, counts = np.unique(labels[labels>=0], return_counts=True)
    if len(counts) < 2:
        return float('nan')
    return float(counts.max() / counts.min())


# ── Gaussian Mixture ──────────────────────────────────────────────────────────

def compute_gmm_bic(model, X):
    """Bayesian Information Criterion from the fitted GaussianMixture."""
    return float(model.bic(X))


def compute_gmm_aic(model, X):
    """Akaike Information Criterion from the fitted GaussianMixture."""
    return float(model.aic(X))


def compute_avg_log_likelihood_per_component(model, X):
    """
    Average log-likelihood per point for each component:
      returns dict {component_index: avg_log_likelihood}.
    """
    # responsibilities shape (n_samples, n_components)
    resps = model.predict_proba(X)
    # log-prob per sample under each component
    log_prob = model._estimate_log_prob(X)  # shape (n_samples, n_components)
    avg_ll = {
        j: float(np.average(log_prob[:, j], weights=resps[:, j]))
        for j in range(model.n_components)
    }
    return avg_ll


# ── DBSCAN ─────────────────────────────────────────────────────────────────────

def compute_dbscan_noise_core_border(labels, core_sample_indices):
    """
    Fraction of noise, core, and border points.
    """
    n = len(labels)
    noise  = np.sum(labels == -1)
    core   = len(core_sample_indices)
    border = n - core - noise
    return {
        'noise_proportion': float(noise / n),
        'core_proportion':  float(core / n),
        'border_proportion':float(border / n),
    }


def compute_all_metrics(X, labels, centroids=None):
    return {
        "silhouette": compute_silhouette(X, labels),
        "calinski_harabasz": compute_calinski_harabasz(X, labels),
        "davies_bouldin": compute_davies_bouldin(X, labels),
        "population": cluster_population_distribution(labels),
        "avg_distance": average_distance_to_centroids(X, labels, centroids),
    }