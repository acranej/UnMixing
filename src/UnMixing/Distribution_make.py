import pymc as pm
#import polars as pl
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from dataclasses import dataclass

rng = np.random.default_rng()

mu = 0
sigma = 1

@dataclass
class Distribution_make:
    type: str
    values: list[float]
    mu: float 
    sigma: float 
    mle: pd.DataFrame
    lower_cutoff: float = None
    upper_cutoff: float = None
    sampling_size: int = 1000

    def evaluate(self):
        print("---Type: ", self.type)
        print("        Mu: ", self.mu.eval())
        print("        Sigma: ", self.sigma.eval())
        print("        Lower_cut: ", self.lower_cutoff)
        

def TruncNormal(x: list[float]):
    print("Making turncated normal distribution...")
    lower = 1
    upper = 10
    temp = [i for i in x if lower <= i <= upper]
    mu_trunc = np.mean(temp)
    sigma_trunc = np.std(temp)
    print("Making MLE...")
    mle_res = pd.DataFrame({'MLE':trunc_norm_fit(np.array(temp), lower, upper), 'True':pd.Series(dict(mu=mu, sigma=sigma))}) 

    dist_temp = Distribution_make(type = 'Truncated_Normal',
                                  values = temp,
                                  mu= mu_trunc,
                                  sigma = sigma_trunc,
                                  mle = mle_res,
                                  lower_cutoff = lower,
                                  upper_cutoff = upper)    
    
    return dist_temp

#### MLE functions #####
@np.vectorize
def _phi(eta): ### Calculates the standard normal density function (PDF)
    return np.exp(-0.5*eta**2)/np.sqrt(2*np.pi)

def _Phi(x): ###Calculates the cumulative distribution function (CDF) of the standard normal distribution
    return 0.5*(1 + scipy.special.erf(x/np.sqrt(2)))

@np.vectorize
def f(x, mu, sigma, a, b): ### Calculates the truncated normal PDF for a given set of parameters
    return _phi((x - mu)/sigma)/(sigma*(_Phi((b-mu)/sigma) - _Phi((a-mu)/sigma)))

def neg_log_likelihood(params, X, size_cutoff_low, size_cutoff_high): ### Calculates the negative log-likelihood for given parameters
    """returns negative log-likelihood of ln_tumors given a truncated normal distribution.

    From https://en.wikipedia.org/wiki/Truncated_normal_distribution, the pdf f(x; mu, sigma, a, b) is
    f(x; mu, sigma, a, b) = \frac{1}{sigma} phi_x/[Phi_b - Phi_a]
    Where: 
        phi_x = \phi((x - mu)/sigma)
        PHI_y = \Phi((y - mu)/sigma)
    Note: 
        PHI_b -> 1, as b -> inf (our case)
    Thus:
        log_f = log(phi_x) - (log(sigma) + log(1 - PHI_a)),
            where
        ln_phi_x = log(phi_x) = -0.5*\eta**2 - log(sqrt(2*pi))
        """   
    mu, sigma = params
    return -(np.log(f(X, mu, sigma, size_cutoff_low, size_cutoff_high)).sum())

### function to find the MLE of mean (mu) and standard deviation (sigma) for the truncated normal distribution given the input values and bounds
def trunc_norm_fit(ln_tumors, size_cutoff_low, size_cutoff_high, min_mean=0):
    """trunc_norm_fit(ln(tumor_sizes), ln(size_cutoff), ln(min_mean)) 
    
    returns MLE of [mu, sigma] for best-fitting Gaussian Distribution w/ tumor size cutoff `size_cutoff`. 
    """ 
    from scipy.optimize import minimize
    
    max_mean = ln_tumors.mean()
    std_min = ln_tumors.std()
    std_max = max_mean - min_mean + std_min 

    bounds = np.array([[min_mean, max_mean], [std_min, std_max]])
    res = minimize(neg_log_likelihood, 
        [mu, sigma],
        args = (ln_tumors, size_cutoff_low, size_cutoff_high),
        bounds=bounds)
    return pd.Series(res.x, index=['mu', 'sigma'])

#### End MLE Functions ####

def Gaussian():
    mu = 1
    sigma = 0

def Exponential():
    mu = 1
    sigma = 0

