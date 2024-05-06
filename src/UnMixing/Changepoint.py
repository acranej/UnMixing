import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
#from pymc_experimental.marginal_model import MarginalModel
#https://gist.github.com/ricardoV94/f986686ce86511b293c5dd6be374e51d


# Define the Bayesian model
@dataclass
class ChangePoint():
    data: list[float]
    trace: az.InferenceData = field(default=None, init=False)

    def sample(self, draws: int = 1000, tune: int = 1000, seed: int = None):
        #MCMC sampling to estimate the changepoint, needs some work 
        with pm.Model() as model:
            # changepoints to evaluate should be a uniform distribution across the range of data
            changepoint = pm.Uniform('changepoint', lower=0, upper=len(self.data))
            changepoint_int = pm.Deterministic('changepoint_int', changepoint.astype(int))
            
            # take the array and split it at the changepoint
            before_data = pm.Deterministic('before_cp', self.data[:int(changepoint_int)])
            after_data = pm.Deterministic('after_cp', self.data[int(changepoint_int):])
            
            # If you do it like this you should be able to try multiple distributions
            self.distribution_before("before_dist", before_data)
            self.distribution_after("after_dist", after_data)
            
            # Sample the model
            self.trace = pm.sample(draws=draws, tune=tune, seed=seed)

    
    def plot_changepoint(self):
        if self.trace is None:
            print('No sampling to plot')
        else:
            changepoint_mean = self.trace.posterior["changepoint"].mean(dim=("chain", "draw")).values
            plt.figure(figsize=(10, 6))
            plt.plot(self.data, label="Data")
            plt.axvline(x=changepoint_mean, color='red', linestyle='--', label=f'Changepoint at {changepoint_mean:.2f}')
            plt.show()

# Plot the trace of the posterior distributions
#pm.plot_trace(trace)
#plt.show()