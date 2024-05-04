import numpy as np
import pymc as pm
import multiprocessing 
import sys
import warnings
# Suppress FutureWarning related to pandas Series, will be changed to polars in time {todo} change to polars...
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import MixResults, Distribution_make

# multiprocessing.set_start_method('fork')

# dist 1 needs to be the targets and dist 2 needs to be the MLE of the inerts
def UnMix(dist1, dist2):
    print("UnMixing...")
    if dist1.type == "Truncated_Normal":
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.array([10, 10]))
            mu1 = pm.Normal("mu1", mu=dist1.mu, sigma=dist1.sigma)
            sig1 = pm.HalfNormal("sig1", sigma=dist1.sigma)

            dist1_temp = pm.TruncatedNormal.dist(mu=mu1,
                                                 sigma=sig1,
                                                 lower=dist1.lower_cutoff,
                                                 upper=dist1.upper_cutoff)
            
            
            dist2_temp = pm.TruncatedNormal.dist(mu=dist2.mle['MLE'].iloc[0],
                                                 sigma=dist2.mle['MLE'].iloc[1],
                                                 lower=dist2.lower_cutoff,
                                                 upper=dist2.upper_cutoff)
            like = pm.Mixture(name="like",
                              w=w,
                              comp_dists=[dist1_temp,dist2_temp],
                              observed=dist1.values)
            #model.debug()
        with model:
            idata=pm.sample(draws=1000, tune=1000, target_accept=0.95)

        mu_temp = idata.posterior["mu1"].mean(("chain", "draw"))
        sig_temp = idata.posterior["sig1"].mean(("chain", "draw"))
        w_temp = idata.posterior["w"].mean(("chain", "draw"))
        print("Generating results...")
        res = MixResults.MixResults(w = w_temp, mu = mu_temp, sigma= sig_temp, target_vals = dist1.values, 
                                    target_mle = dist1.mle, inert_vals = dist2.values, inert_mle = dist2.mle)

    return res