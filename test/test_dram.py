# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:51:37 2019

@author: Rachel
"""

import unittest
from QUAD import DRAM

class PriorLogLike(unittest.TestCase):

    def test_io(self):
        a = DRAM.prior_loglike(par=0.0,m0=0,sd0=1)
        self.assertTrue(isinstance(a, float), msg="Expected output of float")