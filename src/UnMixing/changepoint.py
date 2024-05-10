import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from dataclasses import dataclass, field
#from pymc_experimental.marginal_model import MarginalModel
#https://gist.github.com/ricardoV94/f986686ce86511b293c5dd6be374e51d


# Define the Bayesian model
@dataclass
class ChangePoint():
    data: list[float]
    trace: az.InferenceData = field(default=None, init=False)

    def changepoint_get(self, gaussian, exponential):
        # set the number of "states", will be set to 2 since we believe it is one gaussian and one poisson
        #https://www.youtube.com/watch?v=iwNju1o5yQo&t=387s --> abuzar mahmood
        n_states = 2
        
        ### bin into mean counts for the number of "states" or distributions we are expecting
        mean_counts = np.stack([x.mean(axis=-1) for x in np.array_split(self.data, n_states, axis=-1)]).T
        print(mean_counts)
        

        with pm.Model() as changepoint_model:
            # Latent mean counts
            # 1) can use poisson or a negative binomial
            # poisson gives the probability for the number of events taking place in the given period
            # exponential gives the probabilities for time between events
            # poisson needs to be positive and continuous which will be modeled by exponential
            
            lambda_prior = pm.Exponential('lambda', lam = 0.1, initval=mean_counts, shape = n_states)

            #Changepoint position
            # Beta distribution and use continuous, since counts are continuous (plus get NUTS sampler)
            # initializing to 1,1 gives a uniform distribution
            # the sort makes sure that the changepoints come in the right order
            tau_ = pm.Beta('tau_', alpha=1, beta=1, initval = np.linspace(0,1,n_states+1)[1:-1], shape = (n_states-1),).sort(axis=-1)
            
            # scale tau to the length of our data
            tau = pm.Deterministic('tau', self.data.min() + (self.data.max() - self.data.min()) * tau_,)

            # calculate change in distribution over time
            # i guess sigmoids do a good job of modeling this (link above)
            # we need a timecourse determined by our tau

            f_stack = pt.math.sigmoid(self.data[np.newaxis, :]-tau[:, np.newaxis])
            f_stack = pt.concatenate([np.ones((1, len(self.data))), f_stack], axis = 0)

            i_stack = 1-f_stack[1:]
            i_stack= pt.concatenate([i_stack, np.ones((1, len(self.data)))], axis=0)
            
            weight_stack = np.multiply(f_stack, i_stack)

            # the weight stack needs to be populated
            lambda_dp = pt.tensordot(lambda_prior, weight_stack, axes = (0,0))

            # likelihood calculation
            res = pm.Poisson("res", lambda_dp, observed = self.data)
            
        with changepoint_model:
            trace = pm.sample(draws=1000, tune = 1000)
            poisson_ppc = pm.sample_posterior_predictive(trace)
        ppc_values = poisson_ppc['posterior_predictive']['res']
        #mean_ppc, std_ppc = np.mean(ppc_values,axis=(0,1)),np.std(ppc_values,axis=(0,1))

        poisson_switch_inferred = trace["posterior"]["tau"].values
        #mean_switch = poisson_switch_inferred.mean(axis=(0,1)) 
        #std_switch = poisson_switch_inferred.std(axis=(0,1))
        fig, axs = plt.subplots(1,2, figsize=(12,6))
        for num, change in enumerate(poisson_switch_inferred.T):
            axs[0].hist(change.flatten(),
                     bins = 50, alpha = 0.7,
                     label = f'Changepoint {num}')
        axs[0].legend()
        axs[0].set_title('Inferred Changepoint Distribution')
        axs[0].set_ylabel('Density')
        axs[0].set_xlabel('Count')
        
        hist_gaussian, bin_edges_gaussian = np.histogram(gaussian, bins=50, density=True)
        # Calculate the bin centers for plotting lines
        bin_centers_gaussian = (bin_edges_gaussian[:-1] + bin_edges_gaussian[1:]) / 2
        # Plot Gaussian distribution as a line
        axs[1].plot(bin_centers_gaussian, hist_gaussian, alpha=0.7, color='blue', label='Gaussian')

        # Calculate histogram for Pareto distribution
        hist_expo, bin_edges_expo = np.histogram(exponential, bins=50, density=True)
        # Calculate the bin centers for plotting lines
        bin_centers_expo = (bin_edges_expo[:-1] + bin_edges_expo[1:]) / 2
        # Plot Pareto distribution as a line
        axs[1].plot(bin_centers_expo, hist_expo, alpha=0.7, color='red', label='Exponential')
        mean_change = np.mean(poisson_switch_inferred)

        # plot combined distribution 
        hist_data, bin_edges_data = np.histogram(self.data, bins=50, density=True)
        bin_centers_data = (bin_edges_data[:-1] + bin_edges_data[1:]) / 2
        axs[1].plot(bin_centers_data, hist_data, alpha=0.7, color='green', label='Combined')
        axs[1].axvline(x=mean_change, color='black', linestyle='--', label=f'Mean Changepoint: {mean_change:.2f}')

        # Set title, labels, and legend for the second subplot
        axs[1].set_title('Gaussian and Pareto Distributions')
        axs[1].set_ylabel('Density')
        axs[1].set_xlabel('Value')
        axs[1].legend()
        plt.show()


            



            
    
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