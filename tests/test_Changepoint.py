import unittest
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, pareto
sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import Changepoint

rng = np.random.default_rng()
class TestChangepoint(unittest.TestCase):

    def test_changepoint(self):
        gaussian = rng.normal(loc=5, scale = 0.6, size = 900)
        exponential = rng.exponential(scale = 2, size = 100) + 5.6
        #exponential = [i for i in exponential < (gaussian.mean() + (gaussian.std() * 6)) ]
        #print(gaussian.mean() + (gaussian.std() * 6))
        data = np.concatenate((gaussian, exponential))
       # g_dist = pm.Normal.dist('gaussian', mu=np.mean(data), sigma=np.std(data), observed=data)
        #p_dist = pm.Pareto.dist('pareto', alpha=2, m=np.min(data), observed=data)

        res = Changepoint.ChangePoint(data)
        res.changepoint_get(gaussian=gaussian, exponential=exponential)
        #res.plot_changepoint()


        
        
        

if __name__ == '__main__':
    unittest.main()