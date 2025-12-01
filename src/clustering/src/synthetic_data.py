# synthetic_data.py

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles

def generate_challenging_dataset(random_state=42):
    """
    Combine several cluster shapes and noise:
      - Dense small blob
      - Wide diffuse blob
      - Anisotropic (elliptical) blob
      - Two‐moon nonlinear clusters
      - Concentric circles
      - Uniform background noise
    Returns X (n_samples, 2).
    """
    rs = np.random.RandomState(random_state)

    # 1) Dense small blob
    X1, _ = make_blobs(
        n_samples=300,
        centers=[[0, 0]],
        cluster_std=0.2,
        random_state=rs
    )

    # 2) Wide diffuse blob
    X2, _ = make_blobs(
        n_samples=300,
        centers=[[5, 5]],
        cluster_std=1.5,
        random_state=rs.randint(0, 1e6)
    )

    # 3) Anisotropic blob (elliptical)
    X3, _ = make_blobs(
        n_samples=300,
        centers=[[10, 0]],
        cluster_std=0.3,
        random_state=rs.randint(0, 1e6)
    )
    # apply linear transformation
    transform = np.array([[0.6, -0.8],
                          [-0.8, -0.6]])
    X3 = X3.dot(transform)

    # 4) Two‐moon nonlinear clusters
    X4, _ = make_moons(
        n_samples=200,
        noise=0.05,
        random_state=rs.randint(0, 1e6)
    )
    # scale & shift to avoid overlap
    X4 = X4 * [2.5, 1.0] + [5, -3]

    # 5) Concentric circles
    X5, _ = make_circles(
        n_samples=200,
        factor=0.5,
        noise=0.05,
        random_state=rs.randint(0, 1e6)
    )
    # scale & shift
    X5 = X5 * [3.0, 3.0] + [10, -5]

    # 6) Uniform background noise
    Xn = rs.uniform(low=[-5, -8], high=[15, 8], size=(100, 2))

    # Stack all parts
    X = np.vstack([X1, X2, X3, X4, X5, Xn])
    return X