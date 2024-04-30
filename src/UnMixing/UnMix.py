import os
import numpy as np
import pymc as pm
import MixResults
# 
def UnMix(dist1, dist2, observed):
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.array([1, 1]))

        like = pm.Mixture(name="like",
                          w=w,
                          comp_dists=[dist1,dist2],
                          observed=observed)
    with model:
        idata=pm.sample(draws=1000, tune=1000, target_accept=0.95)

    result = 1