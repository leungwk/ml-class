import linreg
import util.matrix

import numpy as np
import scipy.io as spio
import unittest

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.data = spio.loadmat("data/ex5data1.mat")
        self.m,self.nf = self.data['X'].shape[0],self.data['X'].shape[1]
        self.X = util.matrix.col_concat_ones(self.data['X'])
        self.y = self.data['y']

    def test_cost(self):
        th = np.ones((1+self.nf,))
        self.assertAlmostEqual(303.993,linreg.cost(th,self.X,self.y,lda=1),places=3)

if __name__ == '__main__':
    unittest.main()
