# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:51:37 2019

@author: Rachel
"""

import unittest
from QUAD import dram
import numpy as np

class PriorLogLike(unittest.TestCase):

    def test_io(self):
        a = dram.prior_loglike(par=0.0, m0=0, sd0=1)
        self.assertTrue(isinstance(a, float), msg="Expected output of float")
        a = dram.prior_loglike(par=np.ones(10), m0=0, sd0=1)
        self.assertTrue(isinstance(a, float), msg="Expected output of float")

    def test_expected_behavior(self):
        a = dram.prior_loglike(par=1.0, m0=1.0, sd0=1)
        self.assertEqual(a, 0, msg='Expect 0 if par = m0')
        a = dram.prior_loglike(par=np.ones(10), m0=1.0, sd0=1)
        self.assertEqual(a, 0, msg='Expect 0 if par = m0')


class Z2Par(unittest.TestCase):

    def test_io_w_n_by_none(self):
        lower = np.array([0, 0, 0])
        upper = np.array([1, 1, 1])
        z = np.random.rand(3)
        a = dram.z2par(z, lower, upper)
        self.assertTrue(isinstance(a, np.ndarray), msg='Expect numpy array')
        self.assertEqual(a.shape, z.shape, msg='Expect matching size array')

    def test_io_w_n_by_1(self):
        lower = np.array([0, 0, 0]).reshape(3, 1)
        upper = np.array([1, 1, 1]).reshape(3, 1)
        z = np.random.rand(3).reshape(3, 1)
        a = dram.z2par(z, lower, upper)
        self.assertTrue(isinstance(a, np.ndarray), msg='Expect numpy array')
        self.assertEqual(a.shape, z.shape, msg='Expect matching size array')

    def test_io_w_grad_true(self):
        lower = np.array([0, 0, 0]).reshape(3, 1)
        upper = np.array([1, 1, 1]).reshape(3, 1)
        z = np.random.rand(3).reshape(3, 1)
        a = dram.z2par(z, lower, upper, grad=True)
        self.assertTrue(isinstance(a, np.ndarray), msg='Expect numpy array')
        self.assertEqual(a.shape, z.shape, msg='Expect matching size array')
        lower = np.array([0, 0, 0])
        upper = np.array([1, 1, 1])
        z = np.random.rand(3)
        a = dram.z2par(z, lower, upper, grad=True)
        self.assertTrue(isinstance(a, np.ndarray), msg='Expect numpy array')
        self.assertEqual(a.shape, z.shape, msg='Expect matching size array')


class CalcBSplineBasis(unittest.TestCase):

    def test_io(self):
        x = np.arange(0, 10)
        L = 20
        B = dram.calculate_bsplinebasis(x, L)
        self.assertTrue(isinstance(B, np.ndarray), msg='Expect numpy array')
        self.assertEqual(B.shape, (x.size, L), msg='Expect nxL')
