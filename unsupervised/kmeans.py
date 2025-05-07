import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # tolerance for centroid movement
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples, n_features = X.shape

        # Step 1: Randomly initialize centroids
        np.random.seed(42)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Step 2: Assign points to the nearest centroid
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)

            # Step 3: Compute new centroids
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) if np.any(self.labels == i) else self.centroids[i]
                for i in range(self.k)
            ])

            # Check for convergence
            diff = np.linalg.norm(self.centroids - new_centroids)
            if diff < self.tol:
                break

            self.centroids = new_centroids

    def _compute_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def plot_clusters(self, X):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis', alpha=0.6, edgecolors='k')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='red', marker='x', s=150, linewidths=2)
        plt.title("K-Means Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()

# Generate synthetic data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Run KMeans
kmeans = KMeans(k=3)
kmeans.fit(X)
kmeans.plot_clusters(X)
