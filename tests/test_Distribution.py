import unittest
import pymc as pm
import sys
from matplotlib import pyplot as plt
sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import Distribution

class TestMixResults(unittest.TestCase):

    def test_truncGaus(self):
        dist_test = Distribution.TruncNormal(x = [1,2,3,4,5,6,7,8,9,10])
        plt.hist(dist_test.mu.shape[0])
        
        

if __name__ == '__main__':
    unittest.main()