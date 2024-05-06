import unittest
import numpy as np
import sys
import pymc as pm
sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from UnMixing import Changepoint

rng = np.random.default_rng()
class TestChangepoint(unittest.TestCase):

    def test_changepoint(self):
        gaussian = rng.normal(loc=5, scale = 0.6, size = 900)
        pareto = rng.pareto(a = 2, size = 100) + 6.5
        data = np.concatenate((gaussian, pareto))
       # g_dist = pm.Normal.dist('gaussian', mu=np.mean(data), sigma=np.std(data), observed=data)
        #p_dist = pm.Pareto.dist('pareto', alpha=2, m=np.min(data), observed=data)

        res = Changepoint.ChangePoint(data = data)
        res.sample()
        res.plot_changepoint()

        
        
        

if __name__ == '__main__':
    unittest.main()