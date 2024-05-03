from dataclasses import dataclass 
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
@dataclass
class MixResults:
    w: float
    mu: float
    sigma: float
    target_vals: list[float]
    target_mle: pd.DataFrame
    inert_vals: list[float]
    inert_mle: pd.DataFrame

    
    @property
    def meanW(self) -> float:
        df_w = pl.DataFrame({"w": self.w})
        val_w = df_w["w"].mean()
        return val_w
    
    @property
    def meanMu(self) -> float:
        df_mu= pl.DataFrame({"mu": self.mu})
        val_mu = df_mu["mu"].mean()
        return val_mu
    
    @property
    def meanSigma(self) -> float:
        df_sigma = pl.DataFrame({"sigma": self.sigma})
        val_sigma = df_sigma["sigma"].mean()
        return val_sigma
    
def plot_MixResults(self):
    print(self.mu.values)
    print(self.sigma.values)
    print(self.w.values)
    plt.subplot(1,2,1)
    plt.hist(self.inert_vals, density=True, bins=30, color="black")
    plt.axvline(x = self.target_mle['MLE'][0], color = 'b', label='pyMC')
    plt.plot(np.linspace(1,10,1000), norm.pdf(np.linspace(1,10,1000), self.target_mle['MLE'][0], self.target_mle['MLE'][1]), color='b')

    plt.subplot(1,2,2)
    plt.hist(self.target_vals, density=True, bins=30, color="black", alpha=0.50)
    plt.plot(np.linspace(1,10,1000), 
             (1-self.w.values[0])*norm.pdf(np.linspace(1,10,1000), 
                                        self.target_mle['MLE'][0], self.target_mle['MLE'][1]), color='b', label='Dist1')
    plt.plot(np.linspace(1,10,1000), self.w.values[0]*norm.pdf(np.linspace(1,10,1000), self.mu, self.sigma), color='red', label = 'Dist2')
    plt.plot(np.linspace(1,10,1000), 
             (1-self.w.values[0])*norm.pdf(np.linspace(1,10,1000), 
                                        self.target_mle['MLE'][0], 
                                        self.target_mle['MLE'][1]) + self.w.values[0]*norm.pdf(np.linspace(1,10,1000), self.mu, self.sigma),  color='black', label='combined')
    plt.legend()
    #plt.figure()
    plt.show()
    

    
    