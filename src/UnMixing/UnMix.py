import numpy as np
import pymc as pm
import sys
sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import MixResults, Distribution_make

def UnMix(dist1, dist2):
    print("UnMixing...")
    if dist1.type and dist2.type == "Truncated_Normal":
        with pm.Model() as model:
            w = pm.Dirichlet('w', a=np.array([1, 1]))
            mu1 = pm.Normal("mu1", mu=dist1.mu, sigma=dist1.sigma)
            sig1 = pm.HalfNormal('sig1', sigma=dist1.sigma)
            mu2 = pm.Normal("mu2", mu=dist2.mu, sigma=dist2.sigma)
            sig2 = pm.HalfNormal('sig2', sigma=dist2.sigma)

            dist1_temp = pm.TruncatedNormal.dist(mu=mu1,
                                                 sigma=sig1,
                                                 lower=dist1.lower_cutoff,
                                                 upper=dist1.upper_cutoff)
            
            
            dist2_temp = pm.TruncatedNormal.dist(mu=dist2.mle['MLE'][0],
                                                 sigma=dist2.mle['MLE'][1],
                                                 lower=dist2.lower_cutoff,
                                                 upper=dist2.upper_cutoff)
            like = pm.Mixture(name="like",
                              w=w,
                              comp_dists=[dist1_temp,dist2_temp],
                              observed=dist1.values)
            model.debug()
        with model:
            idata=pm.sample(draws=1000, tune=1000, target_accept=0.95, chains = 1)

        mu_temp = idata.posterior["mu1"].mean(("chain", "draw"))
        sig_temp = idata.posterior["sig1"].mean(("chain", "draw"))
        w_temp = idata.posterior["w"].mean(("chain", "draw"))
        print("Generating results...")
        res = MixResults.MixResults(mu_temp, sig_temp, w_temp, dist1.values, dist1.mle, dist2.values, dist2.mle)

    return res