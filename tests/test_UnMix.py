import unittest
import sys
import numpy as np
sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import UnMix, Distribution_make, MixResults

rng = np.random.default_rng()

class TestUnMix(unittest.TestCase):
    def test_1(self):
        ### dist 2 must be the one that makes MLE for now
        inert = rng.normal(loc= 5, scale = 0.5, size = 1000) ### becomes dist 2 in plotting
        inert_comp = rng.normal(loc=5.2, scale = 0.6, size = 100)
        target_comp = rng.normal(loc= 5.5, scale = 0.7, size = 900) ### becomes dist 1 in plotting
        target = np.concatenate((inert_comp, target_comp))

        a_dist = Distribution_make.TruncNormal(target, target= True) 
        b_dist = Distribution_make.TruncNormal(inert, target = False) 
        res = UnMix.UnMix(a_dist, b_dist)
        MixResults.plot_gof(res)
        MixResults.plot_MixResults(res)


if __name__ == '__main__':
    unittest.main()