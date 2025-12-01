import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm


def _set_axes_limits(ax, ks, y, yerr, num_yticks):
    kmin, kmax = int(ks.min()), int(ks.max())
    ax.set_xlim(kmin, kmax)
    # shared ticks will be set once, but we keep this for limit logic
    ax.set_xticks(np.arange(kmin, kmax + 1))

    if yerr is not None:
        vals = np.concatenate([y, y - yerr, y + yerr])
    else:
        vals = y

    finite = np.isfinite(vals)
    if finite.any():
        ymin, ymax = float(np.nanmin(vals[finite])), float(np.nanmax(vals[finite]))
    else:
        ymin, ymax = 0.0, 1.0

    if ymin == ymax:
        delta = abs(ymin) * 0.1 if ymin != 0 else 1.0
        ymin -= delta
        ymax += delta

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, num=num_yticks))


def plot_all_and_unique_metrics(
    metrics_df,
    unique_vals,
    unique_errs,
    unique_titles,
    fontsize=12,
    use_tex=True,
    linewidth=2,
    capsize=4,
    palette=None,
    num_yticks=5,
    best_k_map=None
):

    """
    3×3 grid with a shared X‐axis:
      Rows 0–1: 6 common metrics
      Row 2:   3 unique metrics

    Only the bottom row shows X‐ticks and X‐labels. Error bars are black;
    all lines use palette[0].
    """
    mpl.rcParams['font.size'] = fontsize
    if use_tex:
        try:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.family']  = 'serif'
            mpl.rcParams['font.serif']   = ['Computer Modern']
        except:
            mpl.rcParams['text.usetex'] = False

    default_palette = [
        "#0072B2", "#E69F00", "#009E73",
        "#D55E00", "#CC79A7", "#56B4E9",
        "#F0E442", "#0072B2", "#D55E00"
    ]
    if palette is None:
        palette = default_palette
    line_color = palette[0]

    # sharex=True makes all subplots use the same x‐axis limits and ticks
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    axes_flat = axes.flatten()
    ks = metrics_df['n_clusters'].to_numpy()

    common = [
            ('inertia',             'Inertia',                      False, None),
            ('silhouette',          'Silhouette Score',             False, None),
            ('calinski_harabasz',   'Calinski–Harabasz Index',      False, None),
            ('davies_bouldin',      'Davies–Bouldin Index',         False, None),
            ('avg_distance_mean',   'Avg Distance to Real Centroid',True, 'avg_distance_std'),
            ('cluster_frac_mean',   'Avg Cluster Fraction',         True, 'cluster_frac_std'),
        ]

    # Top 2 rows: common metrics
    for i, (col, title, is_err, std_col) in enumerate(common):
        ax   = axes_flat[i]
        y    = metrics_df[col].to_numpy()
        err  = metrics_df[std_col].to_numpy() if is_err else None

        if is_err:
            ax.errorbar(
                ks, y, yerr=err,
                fmt='o-', lw=linewidth, capsize=capsize,
                color=line_color, markeredgecolor=line_color,
                ecolor='black'
            )
        else:
            ax.plot(ks, y, 'o-', lw=linewidth, color=line_color)

            # annotate if best_k_map provided
        if best_k_map and col in best_k_map:
            best_k = best_k_map[col]
            ax.axvline(best_k, color=default_palette[1], linestyle='--', linewidth=2, label='Best k for metric')

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        _set_axes_limits(ax, ks, y, err, num_yticks)

    # Bottom row: unique metrics
    for j in range(3):
        ax   = axes_flat[6 + j]
        y    = np.array(unique_vals[j])
        err  = np.array(unique_errs[j]) if unique_errs[j] is not None else None

        ax.plot(ks, y, 'o-', lw=linewidth, color=line_color)
        if err is not None:
            ax.errorbar(
                ks, y, yerr=err, fmt='none',
                capsize=capsize, ecolor='black'
            )

        if best_k_map and col in best_k_map:
            best_k = best_k_map[col]
            ax.axvline(best_k, color=default_palette[1], linestyle='--', linewidth=2, label='Best k for metric')


        title = unique_titles[j]
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        _set_axes_limits(ax, ks, y, err, num_yticks)

    # Only show X‐ticks & labels on the bottom row

    for ax in axes_flat[:6]:
        ax.tick_params(axis='x', labelbottom=False)


    for ax in axes_flat[6:]:
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('Number of clusters')



    # Explanatory text (left‐aligned, slightly larger)
    descriptions = {
        'Inertia':                       'Sum of squared distances to virtual centroids; lower indicates tighter clusters.',
        'Silhouette Score':              'Mean silhouette coefficient (range –1 to +1); higher indicates well-separated clusters.',
        'Calinski–Harabasz Index':       'Ratio of between-cluster to within-cluster dispersion; higher indicates well-defined clusters.',
        'Davies–Bouldin Index':          'Average similarity of each cluster with its most similar one; lower indicates better separation.',
        'Avg Distance to Real Centroid': 'Mean Euclidean distance of each point to its real centroid; lower indicates compact clusters.',
        'Avg Cluster Fraction':          'Mean fraction of points in each cluster; ideal ~1/k for balanced clusters.',
        # Unique‐metrics—these three keys must match whatever unique_titles you passed in
        unique_titles[0]:                'Gap statistic: difference between observed and reference within-cluster dispersion (E[log(W_ref)] – log(W_obs)); peaks at optimal k.',
        unique_titles[1]:                'Cophenetic correlation: how faithfully the dendrogram preserves original pairwise distances; higher is better.',
        unique_titles[2]:                'Merge-height statistics: distribution of dendrogram merge heights (mean, std, max); helps choose a cut threshold.',
        # Fallbacks for other algos (GMM)
        'GMM BIC':                       'Bayesian Information Criterion for GMM; lower favors simpler, better-fitting models.',
        'GMM AIC':                       'Akaike Information Criterion for GMM; lower favors simpler, better-fitting models.',
        'Avg Log-Lik per Component':     'Average log-likelihood per point for each Gaussian component, weighted by responsibilities.',
        # KMeans extras
        'Unbalanced Factor':             'Ratio of largest to smallest cluster size; lower indicates more balanced clusters.',
        'Avg WCSS per Cluster':          'Average within-cluster sum of squares per cluster; lower indicates tighter clusters.',
        # Hierarchical extras (if you ever plot them separately)
        'Inconsistency Mean ± Std':      'Stats (mean±std) of dendrogram‐link inconsistency; lower indicates uniform merges.',
        'Merge-Height Mean ± Std':       'Statistics of merge heights in the dendrogram: mean, std, and max; aids threshold choice.',
        'Cophenetic Correlation':        'Correlation between cophenetic distances and original pairwise distances; higher = more faithful dendrogram.',

    }

    all_titles = [t for _, t, *_ in common] + unique_titles
    lines = [
        f"{i+1}. {title}: {descriptions.get(title, '')}"
        for i, title in enumerate(all_titles)
    ]
    expl = "\n".join(lines)


    fig.subplots_adjust(bottom=0.28)
    fig.text(
        0.01, 0.01, expl,
        ha='left', va='bottom',
        fontsize=fontsize * 0.9,
        wrap=True,
        multialignment='left'
    )

    return fig

def plot_clusters(
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray = None,
        title: str = None,
        palette: list = None,
        figsize: tuple = (6, 6),
        savepath: str = None,
        point_size: int = 20,
        alpha: float = 0.7
) -> plt.Axes:
    """
    Scatter‐plot X colored by `labels`.  Optionally overplot `centroids`.

    Args:
      X          : array-like, shape (n_samples, 2)
      labels     : int array, shape (n_samples,)
      centroids  : array, shape (n_clusters, 2), optional
      title      : figure title
      palette    : list of colors (len >= n_clusters+1), defaults to tab10
      figsize    : figure size
      savepath   : if given, calls fig.savefig(savepath)
      point_size : marker size for data points
      alpha      : point transparency

    Returns:
      ax : the matplotlib Axes instance
    """
    # 1) Prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    # 2) Unique labels (noise = -1)
    uniq = np.unique(labels)
    n_clusters = len(uniq[uniq >= 0])

    # 3) Choose palette
    if palette is None:
        # Use tab10, reserve index 0 for noise if present
        base = cm.get_cmap('tab10').colors
        palette = list(base) + ['#444444']  # last color for noise

    # 4) Plot each cluster
    for lab in uniq:
        mask = labels == lab
        col = palette[int(lab)] if lab >= 0 and lab < len(palette) else palette[-1]
        label_text = f"Cluster {lab}" if lab >= 0 else "Noise"
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=[col],
            s=point_size,
            alpha=alpha,
            label=label_text,
            edgecolor='k' if lab >= 0 else None,
            linewidth=0.2
        )

    # 5) Optionally plot centroids
    if centroids is not None:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c='none',
            edgecolor='black',
            s=200,
            marker='X',
            linewidth=1.5,
            label='Centroids'
        )

    # 6) Final touches
    ax.set_aspect('equal', 'box')
    if title:
        ax.set_title(title)
    ax.legend(loc='best', fontsize='small', framealpha=0.8)
    ax.grid(True)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath)

    return ax