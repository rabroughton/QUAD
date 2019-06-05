# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:51:37 2019

@author: Rachel
"""

import unittest
from QUAD import dram
from pseudo_gsas_tools import Calculator
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


class SmoothYData(unittest.TestCase):

    def test_io(self):
        x = np.random.rand(10,)
        y = np.random.rand(10,)
        y_sm = dram.smooth_ydata(x, y)
        self.assertTrue(isinstance(y, np.ndarray), msg='Expect numpy array')
        self.assertEqual(y_sm.shape, y.shape, msg='Expect matching size array')

    def test_wrong_size_array(self):
        x = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        y_sm = dram.smooth_ydata(x, y)
        self.assertTrue(isinstance(y, np.ndarray), msg='Expect numpy array')
        self.assertEqual(y_sm.shape, (10,), msg='Expect matching size array')


class LogPost(unittest.TestCase):

    def test_io(self):
        y = np.random.random_sample((100,))
        x = y.copy()
        BG = np.random.random_sample((100,))
        Calc = Calculator(shape=y.shape)
        paramList = ['a']
        z = np.random.random_sample((9,))
        lower = np.array([1000.0, 0.0, 0.0, 0.0, -50.0, 0, -10, -0.1, 500])
        upper = np.array([2000.0, 20.0, 1.0, 0.5, -19.0, 100, 0, 0.1, 1500])
        m0 = 0.
        sd0 = 1.
        tau_y = 1.
        scale = np.ones((100, ))
        a = dram.log_post(y, x, BG, Calc, paramList, z, lower,
                          upper, scale, tau_y, m0, sd0)
        self.assertTrue(isinstance(a, float), msg='Explect float return')


class UpdateBackground(unittest.TestCase):

    def test_io(self):
        y = np.random.random_sample((100,))
        Calc = Calculator(shape=y.shape)
        tau_y = 1.
        tau_b = 1.
        L = 20
        B = np.random.random_sample((y.size, L))
        var_scale = np.ones((100, ))
        a = dram.update_background(B, var_scale, tau_y, tau_b, L, Calc, y)
        self.assertTrue(isinstance(a, tuple), msg='Explect tuple return')
        self.assertEqual(a[0].shape, (L,),
                         msg='Expect array shape (L,)')
        self.assertEqual(a[1].shape, y.shape,
                         msg='Expect array shape matching y')


class InitializeCov(unittest.TestCase):

    def test_io(self):
        varS1 = dram.initialize_cov(None, q=3)
        self.assertEqual(varS1.shape, (3, 3), msg='Expect (3, 3)')
        self.assertEqual(list(np.diag(varS1)),
                         [0.05, 0.05, 0.05],
                         msg='Expect 0.05 along main diagonal')

    def test_user_defined(self):
        initCov = np.random.random_sample(size=(3, 3))
        varS1 = dram.initialize_cov(initCov=initCov, q=3)
        self.assertEqual(varS1.shape, (3, 3), msg='Expect (3, 3)')
        self.assertTrue(np.array_equal(varS1, initCov),
                                       msg='Expect arrays equal')

    def test_poor_user_defined(self):
        initCov = np.random.random_sample(size=(3, 3))
        with self.assertRaises(ValueError):
            dram.initialize_cov(initCov=initCov, q=4)
        initCov = np.random.random_sample(size=(4, 3))
        with self.assertRaises(ValueError):
            dram.initialize_cov(initCov=initCov, q=4)
        initCov = np.random.random_sample(size=(3, 4))
        with self.assertRaises(ValueError):
            dram.initialize_cov(initCov=initCov, q=4)


class InitializeOutput(unittest.TestCase):

    def items(self, iters, q, n_keep, L, update, res):
        self.assertEqual(res[0].shape, (iters, q),
                         msg='all_z.shape = (iters, q)')
        self.assertEqual(res[1].shape, (n_keep, q),
                         msg='keep_params.shape = (n_keep, q)')
        self.assertEqual(res[2].shape, (n_keep, L),
                         msg='keep_gamma.shape = (n_keep, L)')
        self.assertEqual(res[3].shape, (n_keep,),
                         msg='keep_b.shape = (n_keep,)')
        self.assertEqual(res[4].shape, (n_keep,),
                         msg='keep_tau_y.shape = (n_keep,)')
        self.assertEqual(res[5].shape, (n_keep,),
                         msg='keep_tau_b.shape = (n_keep,)')
        self.assertEqual(res[6].shape, (n_keep//update,),
                         msg='accept_rate_S1.shape = (n_keep//update,)')
        self.assertEqual(res[7].shape, (n_keep//update,),
                         msg='accept_rate_S2.shape = (n_keep//update,)')
        
    def test_init_output(self):
        iters = 100
        q = 3
        n_keep = 10
        L = 20
        update = 500
        res = dram._initialize_output(iters, q, n_keep, L, update)
        self.items(iters, q, n_keep, L, update, res)

    def test_init_output_2(self):
        iters = 1000
        q = 3
        n_keep = 10
        L = 20
        update = 500
        res = dram._initialize_output(iters, q, n_keep, L, update)
        self.items(iters, q, n_keep, L, update, res)

    def test_init_output_3(self):
        iters = 1000
        q = 3
        n_keep = 1000
        L = 20
        update = 500
        res = dram._initialize_output(iters, q, n_keep, L, update)
        self.items(iters, q, n_keep, L, update, res)


class State2AcceptProb(unittest.TestCase):

    def test_io(self):
        can1_post = 0.3
        can2_post = 0.5
        cur_post = 0.5
        can_z1 = np.random.random_sample((3,))
        can_z2 = np.random.random_sample((3,))
        z = np.random.random_sample((3,))
        varS1 = dram.initialize_cov(None, 3)
        R2 = dram.stage2_acceptprob(can1_post, can2_post, cur_post, can_z1,
                                    can_z2, z, varS1)
        self.assertTrue(isinstance(R2, float), msg='Expect float return')

    def test_io_2(self):
        can1_post = 0.8
        can2_post = 0.2
        cur_post = 0.1
        can_z1 = np.random.random_sample((3,))
        can_z2 = np.random.random_sample((3,))
        z = np.random.random_sample((3,))
        varS1 = dram.initialize_cov(None, 3)
        R2 = dram.stage2_acceptprob(can1_post, can2_post, cur_post, can_z1,
                                    can_z2, z, varS1)
        self.assertTrue(isinstance(R2, float), msg='Expect float return')
