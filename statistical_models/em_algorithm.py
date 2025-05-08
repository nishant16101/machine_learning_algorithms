import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class EM_GMM_1D:
    def __init__(self, n_components=2, max_iter=100, tol=1e-6):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_params(self, X):
        n = len(X)
        self.weights = np.ones(self.k) / self.k
        self.means = np.random.choice(X, self.k)
        self.variances = np.random.random(self.k) + 0.5

    def e_step(self, X):
        n = len(X)
        self.resp = np.zeros((n, self.k))
        for i in range(self.k):
            self.resp[:, i] = self.weights[i] * norm.pdf(X, self.means[i], np.sqrt(self.variances[i]))
        self.resp /= self.resp.sum(axis=1, keepdims=True)

    def m_step(self, X):
        Nk = self.resp.sum(axis=0)
        self.weights = Nk / len(X)
        self.means = (self.resp.T @ X) / Nk
        for i in range(self.k):
            diff = X - self.means[i]
            self.variances[i] = np.sum(self.resp[:, i] * diff**2) / Nk[i]

    def log_likelihood(self, X):
        likelihood = 0
        for i in range(self.k):
            likelihood += self.weights[i] * norm.pdf(X, self.means[i], np.sqrt(self.variances[i]))
        return np.sum(np.log(likelihood))

    def fit(self, X):
        X = np.asarray(X)
        self.initialize_params(X)
        self.log_likelihoods = []

        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            ll = self.log_likelihood(X)
            self.log_likelihoods.append(ll)

            if iteration > 0 and np.abs(ll - self.log_likelihoods[-2]) < self.tol:
                break

    def predict(self, X):
        self.e_step(X)
        return np.argmax(self.resp, axis=1)

    def plot(self, X):
        plt.hist(X, bins=30, density=True, alpha=0.6, color='gray')
        x_vals = np.linspace(min(X), max(X), 1000)
        for i in range(self.k):
            plt.plot(x_vals, self.weights[i] * norm.pdf(x_vals, self.means[i], np.sqrt(self.variances[i])),
                     label=f"Component {i+1}")
        plt.title("Fitted GMM using EM Algorithm")
        plt.legend()
        plt.grid(True)
        plt.show()

np.random.seed(0)
X1 = np.random.normal(0, 1, 300)
X2 = np.random.normal(5, 1, 300)
X = np.concatenate([X1, X2])

model = EM_GMM_1D(n_components=2)
model.fit(X)
model.plot(X)
