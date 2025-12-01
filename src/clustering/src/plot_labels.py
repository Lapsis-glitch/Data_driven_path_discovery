from src.clustering_utils import assign_clusters, plot_clusters
from src.clusterer import KMeansClusterer
from src.synthetic_data import generate_challenging_dataset

X = generate_challenging_dataset(random_state=123)

# 1) Compute labels and centroids for k=4
labels = assign_clusters(KMeansClusterer, X, n_clusters=4, random_state=123)
km = KMeansClusterer(n_clusters=4, random_state=123)
km.fit(X)
centroids = km.get_virtual_centroids()

# 2) Plot and save
ax = plot_clusters(
    X, labels, centroids=centroids,
    title="KMeans (k=4)",
    savepath="kmeans_labels.png"
)
