import pymc as pm
#import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Distribution_make:
    type: str
    mu: list[float]
    sigma: list[float]
    lower_cutoff: float
    upper_cutoff: float
    sampling_size: int = 1000

    def evaluate(self):
        print("Type: ", self.type)
        print("Mu: ", self.mu.eval())
        print("Sigma: ", self.sigma.eval())
        

def TruncNormal(x: list[float], plot = True) -> Distribution_make:
    lower = 6
    upper = 8
    temp = [i for i in x if lower <= i <= upper]
    mu_trunc = pm.Normal.dist(mu=np.mean(temp), sigma=np.std(temp))
    sigma_trunc = pm.HalfNormal.dist(sigma=np.std(temp)) 
    dist_temp = Distribution_make(type = "Truncated_Normal",
                             mu = mu_trunc,
                             sigma = sigma_trunc,
                             lower_cutoff = lower,
                             upper_cutoff = upper)
    dist_temp.evaluate()
    if plot:
        pm.draw([0,1], [1,2], random_seed=1)

    return dist_temp

def Gaussian():
    mu = 1
    sigma = 0

def Exponential():
    mu = 1
    sigma = 0

