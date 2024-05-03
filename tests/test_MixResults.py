import unittest
import sys 

sys.path.append('/Users/alexander_crane/Desktop/Research/UnMixing/')
from src.UnMixing import MixResults

obj = MixResults.MixResults(w=[0.05, 0.1, 0.15],
                 mu=[2.0,4.0,6.0],
                 sigma=[1.0, 1.5, 2.0])

class TestMixResults(unittest.TestCase):

    def test_w(self):
        self.assertAlmostEqual(obj.meanW, 0.1, places = 10)

    def test_mu(self):
         self.assertAlmostEqual(obj.meanMu, 4.0, places = 10)

    def test_sigma(self):
         self.assertAlmostEqual(obj.meanSigma, 1.5, places = 10)

    def test_plot(self):
        res_temp = MixResults.MixResults(w=7.13982473,
                                         mu=0.51505243,
                                         sigma=[0.49270536, 0.50729464])

if __name__ == '__main__':
    unittest.main()
