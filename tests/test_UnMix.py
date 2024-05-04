import unittest
import sys
import numpy as np
sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import UnMix, Distribution_make, MixResults

rng = np.random.default_rng()

class TestUnMix(unittest.TestCase):
    def test_1(self):
        a = rng.normal(loc= 5, scale = 0.7, size = 100000)
        b = rng.normal(loc= 5, scale = 0.7, size = 100000)
        a_dist = Distribution_make.TruncNormal(a)
        b_dist = Distribution_make.TruncNormal(b)
        res = UnMix.UnMix(a_dist, b_dist)
        
        MixResults.plot_MixResults(res)


if __name__ == '__main__':
    unittest.main()