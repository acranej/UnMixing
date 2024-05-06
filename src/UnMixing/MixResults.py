from dataclasses import dataclass, field 
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az
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
    model: pm.Model #= field(init=False, default=None)
    trace: pm.backends.base.MultiTrace #= field(init=False, default=None)

    
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
    #print(self.mu.values)
    #print(self.sigma.values)
    #print(self.w.values)
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(self.inert_vals, density=True, bins=30, color="black")
    plt.axvline(x = self.inert_mle['MLE'].iloc[0], color = 'b', label='pyMC')
    plt.plot(np.linspace(1,10,1000), norm.pdf(np.linspace(1,10,1000), self.inert_mle['MLE'].iloc[0], self.inert_mle['MLE'].iloc[1]), color='b')
    # Add information about dist 1 to the plot
    text1 = f"Dist 1 Mean: {self.inert_mle['MLE'].iloc[0]:.2f},Dist 1 Sigma: {self.inert_mle['MLE'].iloc[1]:.2f}\nWeight: {(1-self.w.values[0]):.2f}"
    plt.text(0.05, -0.135, text1, transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='left')

    plt.subplot(1,2,2)
    plt.hist(self.target_vals, density=True, bins=30, color="black", alpha=0.50)
    plt.plot(np.linspace(1,10,1000), 
             (1-self.w.values[0])*norm.pdf(np.linspace(1,10,1000), 
                                        self.inert_mle['MLE'].iloc[0], self.inert_mle['MLE'].iloc[1]), color='b', label='Dist1')
    plt.plot(np.linspace(1,10,1000), self.w.values[0]*norm.pdf(np.linspace(1,10,1000), self.mu, self.sigma), color='red', label = 'Dist2')
    plt.plot(np.linspace(1,10,1000), 
             (1-self.w.values[0])*norm.pdf(np.linspace(1,10,1000), 
                                        self.inert_mle['MLE'].iloc[0], 
                                        self.inert_mle['MLE'].iloc[1]) + self.w.values[0]*norm.pdf(np.linspace(1,10,1000), self.mu, self.sigma),  color='black', label='combined')
    text2 = f"Dist 2 Mean: {self.mu:.2f},Dist 2 Sigma: {self.sigma:.2f}\nWeight: {self.w.values[0]:.2f}"
    plt.text(0.05, -0.135, text2, transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='left')

    plt.legend()
    plt.show()

def plot_gof(self):
    pm.model_to_graphviz(self.model)
    pm.plot_trace(self.trace)
    az.plot_posterior(self.trace) # gives HDI which is a log scale distribution
    az.plot_forest(self.trace)
    az.plot_ppc(self.trace, kind = 'cumulative')

    
    