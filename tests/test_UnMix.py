import unittest
import sys
import numpy as np
sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import UnMix, Distribution_make, MixResults

rng = np.random.default_rng()

class TestUnMix(unittest.TestCase):
    def test_1(self):
        ### dist 2 must be the one that makes MLE for now
        inert = rng.normal(loc= 5.8, scale = 0.8, size = 10000) 
        inert_comp = rng.normal(loc= 5.9, scale = 1, size = 1000)
        target_comp = rng.normal(loc= 6.3, scale = 0.9, size = 9000) 
        target = np.concatenate((inert_comp, target_comp))
        a_dist = Distribution_make.TruncNormal(target, target= True) 
        b_dist = Distribution_make.TruncNormal(inert, target = False) 
        res = UnMix.UnMix(a_dist, b_dist)
        
        MixResults.plot_MixResults(res)


if __name__ == '__main__':
    unittest.main()