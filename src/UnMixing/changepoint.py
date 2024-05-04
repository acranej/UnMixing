import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Generate synthetic data with a change point in a mixture of two Gaussians
np.random.seed(42)
n_data = 1000
change_point = 500

# First segment: Gaussian distribution
mu1 = 3.0
sigma1 = 1.0
data_before = np.random.normal(loc=mu1, scale=sigma1, size=change_point)

# Second segment: Gaussian distribution
mu2 = 6.0
sigma2 = 1.5
data_after = np.random.normal(loc=mu2, scale=sigma2, size=n_data - change_point)

# Concatenate data
data = np.concatenate([data_before, data_after])

# Plot the data
plt.hist(data, bins=50, alpha=0.7, label='Data')
plt.axvline(change_point, color='red', linestyle='--', label='True change point')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Synthetic Data')
plt.legend()
plt.show()

# Define the Bayesian model
model = pm.Model()

with model:
    # Define priors for the change point, means, and standard deviations
    change_point_prior = pm.DiscreteUniform('change_point', lower=0, upper=n_data - 1)
    mu1_prior = pm.Normal('mu1', mu=0, sigma=10)
    mu2_prior = pm.Normal('mu2', mu=0, sigma=10)
    sigma1_prior = pm.HalfNormal('sigma1', sigma=10)
    sigma2_prior = pm.HalfNormal('sigma2', sigma=10)

    # Define the likelihood of the data given the change point and distribution parameters
    obs = pm.Normal(
        'obs',
        mu=pm.math.switch(np.arange(n_data) <= change_point_prior, mu1_prior, mu2_prior),
        sigma=pm.math.switch(np.arange(n_data) <= change_point_prior, sigma1_prior, sigma2_prior),
        observed=data
    )

    # Perform Bayesian inference to estimate parameters and change point
    trace = pm.sample(1000, chains=4, tune=1000, return_inferencedata=True)

# Plot the trace of the posterior distributions
pm.plot_trace(trace)
plt.show()