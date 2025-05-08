import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    def __init__(self, n_components=2, max_iter=100, tol=1e-6):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_params(self, X):
        n, d = X.shape
        self.weights = np.ones(self.k) / self.k
        self.means = X[np.random.choice(n, self.k, replace=False)]
        self.covs = np.array([np.cov(X, rowvar=False) for _ in range(self.k)])

    def e_step(self, X):
        """Compute responsibilities γ(z_i)."""
        n = X.shape[0]
        self.resp = np.zeros((n, self.k))
        for i in range(self.k):
            self.resp[:, i] = self.weights[i] * multivariate_normal.pdf(X, self.means[i], self.covs[i])
        self.resp /= self.resp.sum(axis=1, keepdims=True)

    def m_step(self, X):
        """Update weights, means, and covariances."""
        n, d = X.shape
        Nk = self.resp.sum(axis=0)
        self.weights = Nk / n
        self.means = (self.resp.T @ X) / Nk[:, None]
        for i in range(self.k):
            diff = X - self.means[i]
            self.covs[i] = (self.resp[:, i][:, None] * diff).T @ diff / Nk[i]

    def compute_log_likelihood(self, X):
        ll = 0
        for i in range(self.k):
            ll += self.weights[i] * multivariate_normal.pdf(X, self.means[i], self.covs[i])
        return np.sum(np.log(ll))

    def fit(self, X):
        self.initialize_params(X)
        self.log_likelihoods = []

        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            ll = self.compute_log_likelihood(X)
            self.log_likelihoods.append(ll)

            if iteration > 0 and abs(ll - self.log_likelihoods[-2]) < self.tol:
                break

    def predict_proba(self, X):
        self.e_step(X)
        return self.resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def plot_clusters(self, X):
        labels = self.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.scatter(self.means[:, 0], self.means[:, 1], c='red', s=100, marker='x')
        plt.title("GMM Cluster Assignments")
        plt.grid(True)
        plt.show()


from sklearn.datasets import make_blobs

# Generate synthetic 2D data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

gmm = GaussianMixtureModel()
gmm.fit(X)
gmm.plot_clusters(X)

