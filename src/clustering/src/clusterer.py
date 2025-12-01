import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from src.metrics import compute_all_metrics



class BaseClusterer:
    def __init__(self, n_clusters=2, **kwargs):
        self.n_clusters = n_clusters
        self.model = None

    def _validate_input(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, np.ndarray):
            return X
        raise ValueError("Input must be numpy array or pandas DataFrame")

    def fit(self, X):
        X_arr = self._validate_input(X)
        self.model.fit(X_arr)
        # get labels
        self.labels_ = (
            self.model.labels_
            if hasattr(self.model, "labels_")
            else self.model.predict(X_arr)
        )
        # try to get centroids (if available)
        self.centroids_ = getattr(self.model, "cluster_centers_", None)
        self.X_ = X_arr
        return self

    def get_virtual_centroids(self):
        """
        Return model‐provided centers if available, else compute
        the mean of each cluster’s points (ignores label -1).
        """
        if hasattr(self, "centroids_") and self.centroids_ is not None:
            return np.array(self.centroids_)

        labels = self.labels_
        unique = sorted(l for l in np.unique(labels) if l >= 0)
        centers = []
        for lbl in unique:
            pts = self.X_[labels == lbl]
            centers.append(pts.mean(axis=0))
        return np.vstack(centers) if centers else np.empty((0, self.X_.shape[1]))

    def get_real_centroids(self):
        """
        Map each virtual centroid to the nearest actual data point.
        Returns array of shape (n_clusters, n_features).
        """
        virtual = self.get_virtual_centroids()
        real = []
        for vc in virtual:
            dists = np.linalg.norm(self.X_ - vc, axis=1)
            idx = np.nanargmin(dists)
            real.append(self.X_[idx])
        return np.vstack(real) if real else np.empty_like(virtual)

    def get_labels(self):
        return self.labels_

    def get_metrics(self):
        return compute_all_metrics(self.X_, self.labels_, self.centroids_)


class KMeansClusterer(BaseClusterer):
    def __init__(self, n_clusters=2, **kwargs):
        super().__init__(n_clusters=n_clusters)
        self.model = KMeans(n_clusters=n_clusters, **kwargs)


class AgglomerativeClusterer(BaseClusterer):
    def __init__(self, n_clusters=2, **kwargs):
        super().__init__(n_clusters=n_clusters)
        self.model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)


class DBSCANClusterer(BaseClusterer):
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        super().__init__(n_clusters=None)
        self.model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)


class GMMClusterer(BaseClusterer):
    def __init__(self, n_clusters=2, **kwargs):
        super().__init__(n_clusters=n_clusters)
        self.model = GaussianMixture(n_components=n_clusters, **kwargs)

    def fit(self, X):
        X_arr = self._validate_input(X)
        self.model.fit(X_arr)
        self.labels_ = self.model.predict(X_arr)
        # GMM has means_ instead of cluster_centers_
        self.centroids_ = self.model.means_
        self.X_ = X_arr
        return self