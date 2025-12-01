# clustering_utils.py
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, inconsistent, fcluster,cophenet
from typing import Optional, Union, Tuple, Any
import inspect
from kneed import KneeLocator
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from src.clusterer import (
    KMeansClusterer,
    AgglomerativeClusterer,
    GMMClusterer,
    DBSCANClusterer,
)
from src.metrics import (
    compute_gap_statistic,
    compute_wcss_per_cluster,
    compute_unbalanced_factor,
    compute_gmm_bic,
    compute_gmm_aic,
    compute_avg_log_likelihood_per_component,
    compute_dbscan_noise_core_border,
)
from src.plotter import plot_all_and_unique_metrics


def _detect_knee_for_metric(
    df: pd.DataFrame,
    metric: str,
    x_col: str = "n_clusters",
    curve: str = "convex",
    direction: str = "decreasing"
) -> int:
    """
    Runs KneeLocator on df[x_col] vs df[metric] and returns the integer k.
    """
    ks = df[x_col].to_numpy(dtype=float)
    ys = df[metric].to_numpy(dtype=float)
    kl = KneeLocator(ks, ys, curve=curve, direction=direction)
    if kl.knee is None:
        raise RuntimeError(f"No knee found for metric {metric}")
    return int(kl.knee)


def _consensus_vote(
    df: pd.DataFrame,
    metrics: list,
    higher_is_better: dict,
    elbow_metrics: list = None,
    elbow_curve: str = "convex",
    elbow_direction: str = "decreasing"
) -> (int, pd.Series):
    """
    Hybrid vote:
      - For m in elbow_metrics: use knee to pick k_m
      - For all other m: argmax/argmin pick k_m
    Then count votes across all metrics.
    Returns (best_k, vote_counts Series).
    """
    if elbow_metrics is None:
        elbow_metrics = ["inertia"]

    votes = []
    for m in metrics:
        col = df[m].dropna()
        if col.empty:
            continue

        if m in elbow_metrics:
            if m in ["Unbalanced Factor"]:
                elbow_direction = "increasing"
            else:
                elbow_direction = "decreasing"
            try:
                k_m = _detect_knee_for_metric(
                    df, m,
                    x_col="n_clusters",
                    curve=elbow_curve,
                    direction=elbow_direction
                )
            except RuntimeError:
                # fallback to argmax/argmin if no knee found
                if higher_is_better.get(m, True):
                    k_m = int(col.idxmax())
                else:
                    k_m = int(col.idxmin())
        else:
            if higher_is_better.get(m, True):
                k_m = int(col.idxmax())
            else:
                k_m = int(col.idxmin())

        votes.append(k_m)

    if not votes:
        raise RuntimeError("No votes cast; check your metrics/elbow_metrics lists.")

    vote_counts = pd.Series(Counter(votes)).sort_index()
    top = vote_counts.max()
    winners = vote_counts[vote_counts == top].index
    best_k = int(min(winners))  # tie-break by smallest k

    return best_k, vote_counts



def _make_clusterer(clusterer_cls, *, n_clusters=None, random_state=None, **extra_kwargs):
    """
    Instantiate a clusterer without accidental or duplicate arguments.

    1) Peeks at clusterer_cls.__init__ to find which kwargs it accepts.
    2) Injects n_clusters if accepted.
    3) Injects random_state only if accepted and applicable.
    4) Injects any other extra_kwargs only if they appear in the signature.
    5) Silently drops any 'sieve' / 'eps' / 'min_samples' / 'soup' keys that don't belong.
    """
    sig = inspect.signature(clusterer_cls.__init__)
    # parameters of __init__, except 'self'
    valid = {p.name for p in sig.parameters.values() if p.name != 'self'}

    init_kwargs = {}

    # 1) n_clusters
    if n_clusters is not None and 'n_clusters' in valid:
        init_kwargs['n_clusters'] = n_clusters

    # 2) random_state (only where it makes sense)
    if random_state is not None and 'random_state' in valid:
        # AgglomerativeClusterer doesn't accept it, so this guard is often redundant
        init_kwargs['random_state'] = random_state

    # 3) everything else from extra_kwargs that is actually accepted
    for k, v in extra_kwargs.items():
        if k in valid:
            init_kwargs[k] = v

    return clusterer_cls(**init_kwargs)


def compute_metrics_over_k(
    clusterer_cls,
    X: np.ndarray,
    k_range,
    *,
    sieve: int = None,
    random_state: int = None,
    **clusterer_kwargs
) -> pd.DataFrame:
    """
    k-range metrics, fitting on X[::sieve] but computing all diagnostics
    on the full X via nearest-centroid assignment.

    Args:
      clusterer_cls   : KMeansClusterer, AgglomerativeClusterer, or GMMClusterer
      X               : full dataset (n_samples, n_features)
      k_range         : iterable of cluster counts
      sieve           : if int>1, fit only on X[::sieve]
      random_state    : passed into the clusterer init
      **clusterer_kwargs: any other init args (e.g. for GMM)
    """
    records    = []
    n_samples  = X.shape[0]

    # build the sieve subset once
    if sieve and sieve > 1:
        idx   = np.arange(0, n_samples, sieve)
        X_fit = X[idx]
    else:
        X_fit = X

    for k in k_range:
        cl = _make_clusterer(
            clusterer_cls,
            n_clusters=k,
            random_state=random_state,
            **clusterer_kwargs
        )

        # fit only on the subsampled X
        cl.fit(X_fit)

        # get virtual centroids and assign entire X
        cents = cl.get_virtual_centroids()            # (k, dim)
        diffs = X[:, None, :] - cents[None, :, :]     # (n_samples, k, dim)
        d2    = np.einsum('ijk,ijk->ij', diffs, diffs)
        labels_full = d2.argmin(axis=1)

        # override for get_metrics()
        cl.X_         = X
        cl.labels_    = labels_full
        cl.centroids_ = cents

        # call your existing get_metrics() for silhouette, CH, DB, pop, avg_distance
        m = cl.get_metrics()

        # inertia via virtual centroids (exclude noise)
        mask  = labels_full >= 0
        pts0  = X[mask]
        lbls0 = labels_full[mask]
        cents0= cents[lbls0]
        inertia = float(np.sum((pts0 - cents0)**2))

        # real-centroid distance stats
        real   = cl.get_real_centroids()
        rcmap  = {i: real[i] for i in range(len(real))}
        dists  = [np.linalg.norm(x - rcmap[l]) for x,l in zip(X, labels_full) if l>=0]
        dist_mean = float(np.mean(dists)) if dists else np.nan
        dist_std  = float(np.std(dists))  if dists else np.nan

        # cluster fraction
        pops      = list(m["population"].values()) if m["population"] else []
        fracs     = [p / n_samples for p in pops]
        frac_mean = float(np.mean(fracs)) if fracs else np.nan
        frac_std  = float(np.std(fracs))  if fracs else np.nan

        records.append({
            "n_clusters":        k,
            "inertia":           inertia,
            "silhouette":        m["silhouette"],
            "calinski_harabasz": m["calinski_harabasz"],
            "davies_bouldin":    m["davies_bouldin"],
            "avg_distance_mean": dist_mean,
            "avg_distance_std":  dist_std,
            "cluster_frac_mean": frac_mean,
            "cluster_frac_std":  frac_std,
        })

    df = pd.DataFrame(records).set_index("n_clusters", drop=False)
    return df


def _consensus_borda(
    df: pd.DataFrame,
    metrics: list,
    hib: dict
) -> pd.Series:
    # Rank each metric (1 = best), then sum ranks; invert so larger is better.
    rank_df = pd.DataFrame(index=df.index)
    for m in metrics:
        asc = not hib.get(m, True)
        rank_df[m] = df[m].rank(method='min', ascending=asc)
    total_rank = rank_df.sum(axis=1)
    total_rank = total_rank.replace(0, np.finfo(float).eps)
    return 1.0 / total_rank


def _consensus_topsis(
    df: pd.DataFrame,
    metrics: list,
    hib: dict,
    weights: list = None
) -> pd.Series:
    X = df[metrics].values.astype(float)
    # normalize
    norms = np.linalg.norm(X, axis=0)
    Xn = X / norms
    # weights
    if weights is None:
        w = np.ones(n_metrics) / n_metrics
    else:
        w = np.array(weights, float)
        w = w / w.sum()  # normalize

    W = Xn * w[np.newaxis, :]
    # ideal / anti‐ideal
    hib_mask = np.array([hib.get(m, True) for m in metrics])
    ideal    = np.where(hib_mask, W.max(axis=0), W.min(axis=0))
    anti     = np.where(hib_mask, W.min(axis=0), W.max(axis=0))
    # distances
    d_plus  = np.linalg.norm(W - ideal,  axis=1)
    d_minus = np.linalg.norm(W - anti,   axis=1)
    return pd.Series(d_minus / (d_plus + d_minus), index=df.index)


def compute_consensus_score(
    df: pd.DataFrame,
    *,
    method: str = "vote",
    metrics: list = None,
    higher_is_better: dict = None,
    weights: list = None,
    elbow_metrics: list = ['inertia', "Avg WCSS per Cluster", "avg_distance_mean", "cluster_frac_mean",
                           "GMM BIC","GMM AIC","Dendrogram Cut Height","Unbalanced Factor"],
    elbow_curve: str = "convex",
    elbow_direction: str = "decreasing"

) -> pd.Series:
    """
    Compute a consensus score over *all* metric columns in `df` by default,
    or a user‐specified subset.  Supports 'borda' and 'topsis'.

    Args:
      df                  : DataFrame with one row per k and metric columns
      method              : "borda", "topsis", "vote"
      metrics             : list of column‐names to include (default = all except n_clusters/consensus_score)
      higher_is_better    : dict mapping {metric_name: bool}; defaults only for the 6 common metrics
      weights             : list of weights (only for TOPSIS; length must match metrics)

    Returns:
      Series of consensus scores, higher=better
    """
    # 1) pick out which columns to use
    if metrics is None:
        metrics = [
            c for c in df.columns
            if c not in ("n_clusters", "consensus_score")
        ]

    # 2) default directions for the six common metrics
    default_hib = {
        "inertia": False,
        "silhouette": True,
        "calinski_harabasz": True,
        "davies_bouldin": False,
        "avg_distance_mean": False,
        "cluster_frac_mean": False,

        # unique KMeans
        "Gap Statistic": True,
        "Unbalanced Factor": False,
        "Avg WCSS per Cluster": False,

        # unique Hierarchical
        "Dendrogram Cut Height": True,
        "Cophenetic Correlation": True,
        "Inconsistency Mean": False,
        "Merge-Height Mean": False,

        # unique GMM
        "GMM BIC": False,
        "GMM AIC": False,
        "Avg Log-Lik per Component": True,

    }
    # merge user overrides
    hib = dict(default_hib)
    if higher_is_better:
        hib.update(higher_is_better)
    # any metric not in hib → assume higher_is_better=True

    if method.lower() == "borda":
        consensus = _consensus_borda(df, metrics, hib)
    elif method.lower() == "topsis":
        consensus = _consensus_topsis(df, metrics, hib, weights)
    elif method.lower() == "vote":
        # Use the vote+elbow hybrid for all “vote” requests
        best_k, vote_counts = _consensus_vote(
            df,
            metrics,
            hib,
            elbow_metrics=elbow_metrics,
            elbow_curve=elbow_curve,
            elbow_direction=elbow_direction
        )
        # One‐hot Series
        s = pd.Series(0.0, index=df.index)
        s.loc[best_k] = 1.0
        consensus = s
    else:
        raise ValueError(f"Unknown consensus method: {method!r}")

    ks = df["n_clusters"].to_numpy(dtype=float)
    best_k_map = {}
    for metric in metrics:
        arr = df[metric].to_numpy(dtype=float)
        # elbow only for inertia or any custom elbow_metrics
        if metric in (elbow_metrics or []) or (metric == "inertia" and elbow_metrics is None):
            try:
                kl = KneeLocator(ks, arr, curve=elbow_curve, direction=elbow_direction)
                best = kl.knee
            except Exception:
                best = None
            if best is None:
                # fall back to argmin for inertia‐type curves
                best = ks[np.nanargmin(arr)]
            best_k_map[metric] = int(best)
        else:
            if hib.get(metric, True):
                idx = int(df[metric].idxmax())
            else:
                idx = int(df[metric].idxmin())
            best_k_map[metric] = idx

    # 5) return both
    return consensus, best_k_map


def run_and_report(
    name: str,
    cls,
    X: np.ndarray,
    k_range,
    *,
    sieve: int = None,
    random_state: int = None,
    consensus_method: str = "borda",
    consensus_weights: list = None,
    **clusterer_kwargs,
):


    print(f"\n\n=== {name.upper()} ===")

    # 1) compute k-range metrics using sieve + random_state
    df = compute_metrics_over_k(
        cls,
        X,
        k_range,
        sieve=sieve,
        random_state=random_state,
        **clusterer_kwargs
    )




    # 3) assemble unique bottom-row arrays (unchanged) …
    if cls is AgglomerativeClusterer:
        Z = linkage(X, method='ward')

        inc = inconsistent(Z)
        n = len(X)
        D0 = pdist(X)  # original pairwise distances
        _, Dcoph = cophenet(Z, D0)


        coph_corr, cut_heights, cut_incons, size_std = [], [], [], []
        for k in k_range:
            idx = n - 1 - k
            cut_heights.append(float(Z[idx, 2]))
            cut_incons.append(float(inc[idx, 3]))
            labels_k = fcluster(Z, t=k, criterion='maxclust')

            # 3) build a mask of within‐cluster pairs
            #    best to convert D‐vectors to square matrices
            D0_mat = squareform(D0)
            Dcoph_mat = squareform(Dcoph)
            mask = (labels_k[:, None] == labels_k[None, :])
            # we only want the upper‐triangle entries (i<j)
            tri_idx = np.triu_indices(n, k=1)
            in_mask = mask[tri_idx]

            # 4) correlation of within‐cluster distances
            d0_in = D0_mat[tri_idx][in_mask]
            dc_in = Dcoph_mat[tri_idx][in_mask]
            # if a cluster has only one point, you may get empty arrays
            if len(d0_in) >= 2:
                cc, _ = pearsonr(d0_in, dc_in)
                coph_corr.append(float(cc))
            else:
                coph_corr.append(np.nan)

            cl_k = cls(n_clusters=k)
            cl_k.fit(X)

            pops = list(cl_k.get_metrics()['population'].values())
            size_std.append(float(np.std(pops)) if pops else np.nan)

        unique_vals = [coph_corr, cut_incons, size_std,cut_heights]
        unique_errs = [None, None, None]
        unique_titles = [
            "Cophenetic Correlation",
            "Inconsistency at Cut",
            "Cluster-Size Std",
            "Dendrogram Cut Height",
        ]

    elif cls is KMeansClusterer:
        gaps, ubs, wcss_means, wcss_stds = [], [], [], []
        for k in k_range:
            km = cls(n_clusters=k, random_state = random_state, **clusterer_kwargs)
            km.fit(X)

            gaps.append(compute_gap_statistic(X, cls, k, B=10, random_state=random_state))
            ubs.append(compute_unbalanced_factor(km.labels_))

            wcss = compute_wcss_per_cluster(
                X, km.labels_, km.get_virtual_centroids()
            )
            vals = list(wcss.values())
            wcss_means.append(float(np.mean(vals)))
            wcss_stds.append(float(np.std(vals)))

        unique_vals = [gaps, ubs, wcss_means]
        unique_errs = [None, None, wcss_stds]
        unique_titles = [
            "Gap Statistic",
            "Unbalanced Factor",
            "Avg WCSS per Cluster"
        ]

    elif cls is GMMClusterer:
        bics, aics, avgll_means, avgll_stds = [], [], [], []
        for k in k_range:
            gm = cls(n_clusters=k, random_state = random_state, **clusterer_kwargs)
            gm.fit(X)

            bics.append(compute_gmm_bic(gm.model, X))
            aics.append(compute_gmm_aic(gm.model, X))

            comp_ll = compute_avg_log_likelihood_per_component(gm.model, X)
            vals = list(comp_ll.values())
            avgll_means.append(float(np.mean(vals)))
            avgll_stds.append(float(np.std(vals)))

        unique_vals = [bics, aics, avgll_means]
        unique_errs = [None, None, avgll_stds]
        unique_titles = [
            "GMM BIC",
            "GMM AIC",
            "Avg Log-Lik per Component"
        ]

    else:
        # should not happen here for k-range methods
        unique_vals = [[np.nan] * len(k_range)] * 3
        unique_errs = [None] * 3
        unique_titles = ["N/A"] * 3

    for title, vals in zip(unique_titles, unique_vals):
        df[title] = vals

    df['consensus_score'],metric_best_k = compute_consensus_score(
        df,
        method=consensus_method,
        weights=consensus_weights
    )

    df.to_csv(f"{name}_metrics.csv", index=False)

    best_k = int(df.loc[df['consensus_score'].idxmax(), 'n_clusters'])
    print(f"Best k by consensus: {best_k}")

    # 4) plot 3×3 with highlight …
    fig = plot_all_and_unique_metrics(
        df,
        unique_vals,
        unique_errs,
        unique_titles,
        fontsize=12,
        use_tex=True,
        linewidth=2,
        capsize=4,
        best_k_map=metric_best_k
    )
    for ax in fig.axes:
        ax.axvline(best_k, color='red', linestyle='--', linewidth = 2, label = 'Consensus Best k')
        ax.legend()
    fig.savefig(f"{name}_3x3_metrics.png")
    plt.close(fig)

    # 5) final centroids: fit on the same sieve at best_k
    if sieve and sieve > 1:
        sieve_idx = np.arange(0, X.shape[0], sieve)
        X_fit     = X[sieve_idx]
    else:
        X_fit = X

    if cls is AgglomerativeClusterer:
        # Agglomerative ignores random_state
        final = cls(n_clusters=best_k, **clusterer_kwargs)
    else:
        final = cls(n_clusters=best_k, random_state=random_state, **clusterer_kwargs)
    final.fit(X_fit)
    virt = final.get_virtual_centroids()
    real = final.get_real_centroids()

    pd.DataFrame(virt).to_csv(f"{name}_virtual_centroids.csv", index=False)
    pd.DataFrame(real).to_csv(   f"{name}_real_centroids.csv",    index=False)





def run_dbscan(name, X, eps=0.7, min_samples=5):
    """
    Single-run DBSCAN: compute metrics, consensus, and save CSVs + centroids.
    """
    print(f"\n\n=== {name.upper()} ===")
    db = DBSCANClusterer(eps=eps, min_samples=min_samples)
    db.fit(X)
    mdb = db.get_metrics()
    print(mdb)

    virt = db.get_virtual_centroids()
    real = db.get_real_centroids()
    print("Virtual centroids:\n", virt)
    print("Real centroids:\n",    real)

    # safe statistics
    pop_vals  = np.array(list(mdb["population"].values()),    dtype=float)
    dist_vals = np.array(list(mdb["avg_distance"].values()), dtype=float)

    def safe_mean(a): return float(a.mean()) if a.size else np.nan
    def safe_std(a):  return float(a.std())  if a.size else np.nan

    db_df = pd.DataFrame([{
        "inertia":            np.nan,
        "silhouette":         mdb["silhouette"],
        "calinski_harabasz":  mdb["calinski_harabasz"],
        "davies_bouldin":     mdb["davies_bouldin"],
        "avg_distance_mean":  safe_mean(dist_vals),
        "avg_distance_std":   safe_std(dist_vals),
        "cluster_frac_mean":  safe_mean(pop_vals) / X.shape[0],
        "cluster_frac_std":   safe_std(pop_vals)  / X.shape[0],
    }])
    # db_df["consensus_score"], metric_best_k = compute_consensus_score(db_df)
    db_df.to_csv(f"{name}_metrics.csv", index=False)
    print(f"Saved DBSCAN metrics to {name}_metrics.csv")

    pd.DataFrame(virt).to_csv(f"{name}_virtual_centroids.csv", index=False)
    pd.DataFrame(real).to_csv( f"{name}_real_centroids.csv",    index=False)
    print(f"Saved DBSCAN centroids to {name}_virtual_centroids.csv and {name}_real_centroids.csv")

# def assign_clusters(
#     clusterer_cls,
#     X,
#     *,
#     n_clusters: int = None,
#     cutoff_height: float = None,
#     eps: float = None,
#     min_samples: int = None,
#     random_state: int = None,
#     method: str = 'ward',
#     metric: str = 'euclidean',
#     **kwargs
# ) -> np.ndarray:
#     """
#     Fit a clustering algorithm and return labels for each sample in X.
#
#     Args:
#       clusterer_cls: one of KMeansClusterer, AgglomerativeClusterer,
#                      GMMClusterer, or DBSCANClusterer.
#       X            : (n_samples, n_features) data array.
#       n_clusters   : number of clusters (for k-based methods).
#       cutoff_height: linkage cut height (for hierarchical only).
#       eps          : eps parameter (for DBSCAN only).
#       min_samples  : min_samples parameter (for DBSCAN only).
#       random_state : random seed for k-based methods.
#       method, metric: passed to scipy linkage for hierarchical if cutoff_height is used.
#
#     Returns:
#       labels: array of shape (n_samples,) with integer cluster labels.
#     """
#
#     # 1) DBSCAN
#     if clusterer_cls is DBSCANClusterer:
#         cl = DBSCANClusterer(eps=eps, min_samples=min_samples)
#         cl.fit(X)
#         return cl.labels_
#
#     # 2) Agglomerative (hierarchical)
#     if clusterer_cls is AgglomerativeClusterer:
#         # If user supplied a cutoff height, do SciPy linkage + fcluster
#         if cutoff_height is not None:
#             Z = linkage(X, method=method, metric=metric)
#             # criterion='distance' cuts so that all merges above cutoff are separate clusters
#             labels = fcluster(Z, t=cutoff_height, criterion='distance')
#             return labels - 1  # make zero-based
#         # otherwise fall back to n_clusters
#         if n_clusters is None:
#             raise ValueError("Must supply n_clusters or cutoff_height for hierarchical clustering")
#         cl = AgglomerativeClusterer(n_clusters=n_clusters)
#         cl.fit(X)
#         return cl.labels_
#
#     # 3) K-means
#     if clusterer_cls is KMeansClusterer:
#         if n_clusters is None:
#             raise ValueError("Must supply n_clusters for KMeansClusterer")
#         cl = KMeansClusterer(n_clusters=n_clusters, random_state=random_state, **kwargs)
#         cl.fit(X)
#         return cl.labels_
#
#     # 4) Gaussian Mixture
#     if clusterer_cls is GMMClusterer:
#         if n_clusters is None:
#             raise ValueError("Must supply n_clusters for GMMClusterer")
#         cl = GMMClusterer(n_clusters=n_clusters, random_state=random_state, **kwargs)
#         cl.fit(X)
#         return cl.labels_
#
#     raise ValueError(f"Unsupported clusterer class: {clusterer_cls}")


def run_method(
    name: str,
    clusterer_cls,
    X,
    *,
    k_range=None,
    sieve: int = None,
    random_state: int = None,
    eps: float = None,
    min_samples: int = None,
    consensus_method="vote",
    **clusterer_kwargs
):
    from clustering_utils import run_and_report, run_dbscan

    if clusterer_cls is DBSCANClusterer:
        # DBSCAN ignores sieve
        run_dbscan(name=name, X=X, eps=eps, min_samples=min_samples)
    elif clusterer_cls is AgglomerativeClusterer:
        # AgglomerativeClusterer ignores random_state
        if k_range is None:
            raise ValueError("k_range is required for k-based methods")
        run_and_report(
            name=name,
            cls=clusterer_cls,
            X=X,
            k_range=k_range,
            sieve=sieve,
            consensus_method=consensus_method,
            **clusterer_kwargs
        )
    else:
        if k_range is None:
            raise ValueError("k_range is required for k-based methods")
        run_and_report(
            name=name,
            cls=clusterer_cls,
            X=X,
            k_range=k_range,
            sieve=sieve,
            random_state=random_state,
            consensus_method=consensus_method,
            **clusterer_kwargs
        )


def save_cluster_labels(
    X: np.ndarray,
    labels: np.ndarray,
    filepath: str
) -> None:
    """
    Dump a CSV of X coordinates with their cluster label.

    Args:
      X        : array, shape (n_samples, n_features)
      labels   : int array, shape (n_samples,)
      filepath : path to output CSV; columns = ['x0','x1',…,'label']
    """
    n_features = X.shape[1]
    data = { f"x{i}": X[:, i] for i in range(n_features) }
    data['label'] = labels
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def assign_clusters_with_sieve(
    clusterer_cls,
    X: np.ndarray,
    *,
    sieve_frac: Optional[float] = None,
    sieve_size: Optional[int] = None,
    random_state: Optional[int] = None,
    **clusterer_kwargs
) -> np.ndarray:
    """
    1) Randomly sample a subset ("sieve") of X
    2) Fit clusterer_cls on that subset
    3) Extract its virtual centroids
    4) Assign every point in the full X to the nearest centroid

    Args:
      clusterer_cls : one of KMeansClusterer, AgglomerativeClusterer,
                       GMMClusterer, DBSCANClusterer
      X             : array, shape (n_samples, n_features)
      sieve_frac    : fraction of samples to keep (0<frac<=1). Mutually
                      exclusive with sieve_size.
      sieve_size    : absolute number of samples to keep. Mutually
                      exclusive with sieve_frac.
      random_state  : seed for reproducible sampling & clustering
      **clusterer_kwargs : parameters passed to clusterer_cls initializer
         - For k–based methods: pass n_clusters=...
         - For DBSCAN: pass eps=..., min_samples=...
         - For GMM: pass n_clusters=..., random_state=...
         - For Agglomerative: pass n_clusters=...

    Returns:
      labels_full : int array, shape (n_samples,)
         Cluster label (0..k−1) for each point in X.
    """
    n_samples = X.shape[0]
    rs = np.random.RandomState(random_state)

    # decide how many to sample
    if sieve_size is not None and sieve_frac is not None:
        raise ValueError("Specify only one of sieve_size or sieve_frac")
    if sieve_frac is not None:
        if not (0 < sieve_frac <= 1):
            raise ValueError("sieve_frac must be in (0,1]")
        m = int(np.ceil(n_samples * sieve_frac))
    elif sieve_size is not None:
        m = int(sieve_size)
    else:
        # no sieve → use full dataset
        m = n_samples

    m = n_samples
    m = min(m, n_samples)
    # sample indices
    perm = rs.permutation(n_samples)
    sieve_idx = perm[:m]
    X_sieve = X[sieve_idx]

    # 1) fit on the sieve
    cl = clusterer_cls(**clusterer_kwargs)
    cl.fit(X_sieve)

    # 2) get centroids (virtual—always defined for our clusterers)
    centroids = cl.get_virtual_centroids()  # shape (k, d)

    # 3) assign each full point to nearest centroid
    # compute squared distances [n_samples, k]
    # broadcasting X[:,None,:] - centroids[None,:,:]
    diffs = X[:, None, :] - centroids[None, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diffs, diffs)
    labels_full = dist2.argmin(axis=1)

    return labels_full

def assign_clusters(
    clusterer_cls,
    X: np.ndarray,
    *,
    random_state: Optional[int] = None,
    centroids_csv: str = None,   # path to write virtual centroids
    clusters_csv: str = None,    # path to write full X + labels
    **clusterer_kwargs
) -> Tuple[np.ndarray, Any]:
    """
    1) Fit clusterer_cls on X (uses random_state if supported)
    2) Extract its virtual centroids
    3) Assign every point in X to nearest virtual centroid
    4) Optionally write centroids and/or assignments to CSV

    Args:
      clusterer_cls    : your Clusterer class
      X                : array, shape (n_samples, n_features)
      random_state     : seed (if supported by clusterer_cls)
      centroids_csv    : if given, save cl.get_virtual_centroids() to this CSV path
      clusters_csv     : if given, save X coords + labels_full to this CSV path
      **clusterer_kwargs: passed through to clusterer_cls(...)

    Returns:
      labels_full      : int array, cluster index for each point in X
      cl               : fitted clusterer instance
    """

    # 1) instantiate & fit
    if clusterer_cls is not AgglomerativeClusterer:
        cl = clusterer_cls(random_state=random_state, **clusterer_kwargs)
    else:
        # Agglomerative ignores random_state
        cl = clusterer_cls(**clusterer_kwargs)
    cl.fit(X)

    # 2) get virtual centroids
    centroids = cl.get_virtual_centroids()  # shape (n_clusters, n_features)

    # 3) assign every point in X to nearest centroid
    diffs       = X[:, None, :] - centroids[None, :, :]
    dist2       = np.einsum('ijk,ijk->ij', diffs, diffs)
    labels_full = dist2.argmin(axis=1)

    # 4) write out centroids if requested
    if centroids_csv:
        df_c = pd.DataFrame(
            centroids,
            columns=[f"dim_{i}" for i in range(centroids.shape[1])]
        )
        df_c.index.name = 'cluster'
        df_c.to_csv(centroids_csv)

    # 5) write out full assignments if requested
    if clusters_csv:
        # build DataFrame of X coords + assigned label
        df_x = pd.DataFrame(
            X,
            columns=[f"dim_{i}" for i in range(X.shape[1])]
        )
        df_x['cluster'] = labels_full
        df_x.to_csv(clusters_csv, index=False)

    return labels_full, cl

