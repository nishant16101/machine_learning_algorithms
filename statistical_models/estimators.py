import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta,norm

class MLEEstimator:
    """for common distribution"""
    @staticmethod
    def bernoulli(data):
        return np.mean(data)
    @staticmethod
    def normal(data):
        mu_hat  = np.mean(data)
        sigma_hat = np.var(data,ddof=0)
        return mu_hat,sigma_hat
    
class BayesianEstimator:
    @staticmethod
    def beta_posterior(prior_a,prior_b,data):
        success = np.sum(data)
        failures = len(data) - success
        post_a = prior_a + success
        post_b = prior_b +success
        return post_a,post_b
    @staticmethod
    def plot_beta(prior_a, prior_b, post_a, post_b):
        """Plot Beta prior and posterior."""
        x = np.linspace(0, 1, 500)
        plt.plot(x, beta.pdf(x, prior_a, prior_b), label="Prior", color="blue")
        plt.plot(x, beta.pdf(x, post_a, post_b), label="Posterior", color="green")
        plt.title("Beta Prior vs Posterior (Bernoulli)")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def normal_mean_known_variance(mu0, sigma0, sigma_known, data):
        """Posterior for Normal likelihood (known variance) + Normal prior on mean."""
        n = len(data)
        x_bar = np.mean(data)

        post_var = 1 / (n / sigma_known**2 + 1 / sigma0**2)
        post_mu = post_var * (x_bar * n / sigma_known**2 + mu0 / sigma0**2)

        return post_mu, np.sqrt(post_var)
    
    @staticmethod
    def plot_normal(mu0, sigma0, mu_post, sigma_post):
        """Plot Normal prior and posterior."""
        x = np.linspace(mu0 - 4 * sigma0, mu0 + 4 * sigma0, 500)
        plt.plot(x, norm.pdf(x, mu0, sigma0), label="Prior", color="blue")
        plt.plot(x, norm.pdf(x, mu_post, sigma_post), label="Posterior", color="green")
        plt.title("Normal Prior vs Posterior (Normal Mean Inference)")
        plt.legend()
        plt.grid(True)
        plt.show()


data = [1, 0, 1, 1, 1, 0, 1]

p_mle = MLEEstimator.bernoulli(data)
mu_mle, sigma2_mle = MLEEstimator.normal(data)
print("MLE Bernoulli p̂:", p_mle)
print("MLE Normal μ̂, σ̂²:", mu_mle, sigma2_mle)

#beta bernoulii
prior_a, prior_b = 2, 2
post_a, post_b = BayesianEstimator.beta_posterior(prior_a, prior_b, data)
BayesianEstimator.plot_beta(prior_a, prior_b, post_a, post_b)

#with known variance
data = np.random.normal(loc=4.0, scale=2.0, size=20)
mu0, sigma0 = 0.0, 1.0
sigma_known = 2.0

mu_post, sigma_post = BayesianEstimator.normal_mean_known_variance(mu0, sigma0, sigma_known, data)
BayesianEstimator.plot_normal(mu0, sigma0, mu_post, sigma_post)

