# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:51:37 2019

@author: Rachel
"""

import unittest
from QUAD import dram
from pseudo_gsas_tools import Calculator
import numpy as np
import os
from mock import patch


Calc = Calculator(path='test' + os.sep + 'gsas_objects')


def setup_problem(q=9, L=20):
    n = Calc._n
    return dict(
            q=q,
            n=n,
            y=Calc._y,
            x=Calc._x,
            Calc=Calc,
            BG=np.random.random_sample((n,)),
            variables=Calc._variables,
            paramList=list(Calc._paramList),
            z=np.random.random_sample((q,)),
#            lower=np.array([1000.0, 0.0, 0.0, 0.0, -50.0, 0, -10, -0.1, 500]),
#            upper=np.array([2000.0, 20.0, 1.0, 0.5, -19.0, 100, 0, 0.1, 1500]),
            lower=np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]),
            upper=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            m0=0.,
            sd0=1.,
            tau_y=1.,
            tau_b=1.,
            var_scale=np.ones((n, )),
            scale=np.ones((n,)),
            varS1=np.random.random_sample((q, q)),
            L=L,
            B=np.random.random_sample((n, L)),
            delta=1e-3,
            start=np.random.random_sample((q,)),
#            init_z=np.random.random_sample((q,)),
            )


def setup_args(tmp, keys):
    items = {}
    for key in keys:
        items[key]=tmp[key]
    return items


class EstimateCovariance(unittest.TestCase):

    def test_io(self):
        tmp = setup_problem()
        q = tmp['q']
        # this list much match the order of input arguments
        keys = ['paramList', 'start', 'Calc', 'upper', 'lower',
                'x', 'y', 'L', 'delta']
        items = setup_args(tmp, keys)
        a = dram.estimatecovariance(**items)
        self.assertTrue(isinstance(a, dict),
                        msg='Expect dict return')
        self.assertEqual(a['cov'].shape, (q, q),
                         msg='Expect (q, q) array')
        self.assertTrue(isinstance(a['s2'], float),
                         msg='Expect float')
        self.assertTrue(isinstance(a['evals'], tuple))
        self.assertEqual(a['evals'][0].shape, (q,),
                         msg='Expect eigenval. as vector')
        self.assertEqual(a['evals'][1].shape, (q, q),
                         msg='Expect eigenvec. as square-mtx.')


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


class LogPost(unittest.TestCase):

    def test_io(self):
        tmp = setup_problem()
        # this list much match the order of input arguments
        keys = ['y', 'x', 'BG', 'Calc', 'paramList', 'z', 'lower',
                'upper', 'scale', 'tau_y', 'm0', 'sd0']
        items = setup_args(tmp, keys)
        a = dram.log_post(**items)
        self.assertTrue(isinstance(a, float), msg='Explect float return')


class CalcBSplineBasis(unittest.TestCase):

    def test_io(self):
        x = np.arange(0, 10)
        L = 20
        B = dram.calculate_bsplinebasis(x, L)
        self.assertTrue(isinstance(B, np.ndarray), msg='Expect numpy array')
        self.assertEqual(B.shape, (x.size, L), msg='Expect nxL')


class DiffractionFileData(unittest.TestCase):

    def test_io(self):
        x = Calc._x
        y = Calc._y
        n = Calc._n
        a = dram.diffraction_file_data(x, y, Calc)
        self.assertTrue(isinstance(a, tuple),
                        msg='Expect tuple return')
        self.assertEqual(len(a), 2,
                         msg='Expect tuple of length 2')
        self.assertEqual(a[0].shape, (n,),
                         msg='Expect (n,) array')
        self.assertEqual(a[1].shape, (n,),
                         msg='Expect (n,) array')
        self.assertTrue(np.array_equal(a[0], x),
                         msg='Expect array match')
        self.assertTrue(np.array_equal(a[1], y),
                         msg='Expect array match')

    def test_xy_none(self):
        x = None
        y = None
        n = Calc._n
        a = dram.diffraction_file_data(x, y, Calc)
        self.assertTrue(isinstance(a, tuple),
                        msg='Expect tuple return')
        self.assertEqual(len(a), 2,
                         msg='Expect tuple of length 2')
        self.assertEqual(a[0].shape, (n,),
                         msg='Expect (n,) array')
        self.assertEqual(a[1].shape, (n,),
                         msg='Expect (n,) array')


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


class UpdateBackground(unittest.TestCase):

    def test_io(self):
        tmp = setup_problem()
        n = tmp['n']
        # this list much match the order of input arguments
        keys = ['B', 'var_scale', 'tau_y', 'tau_b', 'L', 'Calc', 'y']
        items = setup_args(tmp, keys)
        a = dram.update_background(**items)
        self.assertTrue(isinstance(a, tuple), msg='Explect tuple return')
        self.assertEqual(len(a), 2, msg='Explect tuple of length 2')
        self.assertEqual(a[0].shape, (items['L'],),
                         msg='Expect array shape (L,)')
        self.assertEqual(a[1].shape, (n,),
                         msg='Expect array shape matching y')


class State1AcceptProb(unittest.TestCase):

    def test_io(self):
        tmp = setup_problem()
        q = tmp['q']
        # this list much match the order of input arguments
        keys = ['z', 'varS1', 'y', 'x', 'BG', 'Calc', 'paramList', 'lower',
                'upper', 'var_scale', 'tau_y', 'm0', 'sd0']
        items = setup_args(tmp, keys)
        a = dram.stage1_acceptprob(**items)
        self.assertTrue(isinstance(a, tuple), msg='Explect tuple return')
        self.assertEqual(len(a), 4, msg='Explect tuple of length 4')
        self.assertEqual(a[0].shape, (q,),
                         msg='Expect array shape (q,)')
        self.assertTrue(isinstance(a[1], float),
                         msg='Expect float return')
        self.assertTrue(isinstance(a[2], float),
                 msg='Expect float return')
        self.assertTrue(isinstance(a[3], float),
         msg='Expect float return')


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


class AdaptCovariance(unittest.TestCase):

    def test_no_adapt(self):
        tmp = setup_problem()
        q, varS1 = tmp['q'], tmp['varS1']
        adapt = 20
        s_p = 2.4**2/q
        iters = 1000
        all_Z = np.random.random_sample((iters, q))
        epsilon = 0.0001
        i = 0        
        a = dram.adapt_covariance(i, adapt, s_p, all_Z,
                                  epsilon, q, varS1)
        self.assertTrue(np.array_equal(a, varS1),
                        msg='Expect array equal')

    def test_adapt(self):
        tmp = setup_problem()
        q, varS1 = tmp['q'], tmp['varS1']
        adapt = 20
        s_p = 2.4**2/q
        iters = 1000
        all_Z = np.random.random_sample((iters, q))
        epsilon = 0.0001
        i = adapt      
        a = dram.adapt_covariance(i, adapt, s_p, all_Z,
                                  epsilon, q, varS1)
        self.assertFalse(np.array_equal(a, varS1),
                        msg='Expect array equal')


class UpdateTauB(unittest.TestCase):

    def test_io(self):
        d_g, c_g, L = 0.1, 0.1, 20
        gamma = np.random.random_sample((L,))
        a = dram.update_taub(d_g, gamma, c_g, L)
        self.assertTrue(isinstance(a, float),
                        msg='Expect float return')


class UpdateTauY(unittest.TestCase):

    def test_io(self):
        d_y, c_y = 0.1, 0.1
        tmp = setup_problem()
        y, BG, Calc, var_scale, n = (
                tmp['y'], tmp['BG'], tmp['Calc'],
                tmp['var_scale'], tmp['n'])
        a = dram.update_tauy(y, BG, Calc, var_scale, d_y, c_y, n)
        self.assertTrue(isinstance(a, float),
                        msg='Expect float return')


class Traceplots(unittest.TestCase):

    def test_no_plot(self):
        plot = False
        tmp = setup_problem()
        q, paramList = tmp['q'], tmp['paramList']
        path = './results'
        iters = 100
        keep_params = np.random.random_sample((iters, q))
        curr_keep = 20
        n_keep = 50
        update = 20
        a = dram.traceplots(plot, q, keep_params, curr_keep, paramList,
                     n_keep, update, path)
        self.assertEqual(a, None)

    def test_plot(self):
        plot = True
        tmp = setup_problem()
        q, paramList = tmp['q'], tmp['paramList']
        path_name = '.'
        iters = 100
        keep_params = np.random.random_sample((iters, q))
        curr_keep = 20
        n_keep = 50
        update = 100
        a = dram.traceplots(plot, q, keep_params, curr_keep, paramList,
                     n_keep, update, path_name)
        self.assertEqual(a, None)
        fn = 'DRAM_Trace.png'
        self.assertTrue(os.path.exists(fn))
        os.remove(fn)


class InitializeIntensityWeight(unittest.TestCase):

    def test_io(self):
        tmp = setup_problem()
        x, y, n = tmp['x'], tmp['y'], tmp['n']
        a = dram.initialize_intensity_weight(x, y)
        self.assertTrue(isinstance(a, np.ndarray),
                        msg='Expect array return')
        self.assertEqual(a.shape, (n,),
                         msg='Expect (n,) array')


class Sample(unittest.TestCase):

#    @patch('QUAD.gsas_tools.Calculator',
#           return_value=Calc)
   @patch('QUAD.dram.gsas_calculator',
           return_value=Calc)
    def test_io(self, mock_1):
        tmp = setup_problem()
        n = tmp['n']
        iters = 2000
        burn = 1000
        thin = 1
        adapt = 200
        n_keep = np.floor_divide(iters - burn - 1, thin) + 1
        q = tmp['q']
        L = tmp['L']
        varS1 = tmp['varS1']
        paramList, variables = tmp['paramList'], tmp['variables']
        start, lower, upper = tmp['start'], tmp['lower'], tmp['upper']
        path = './results'
        a = dram.sample(None, paramList, variables, start, lower, upper, path,
                        plot=False, iters=iters, burn=burn, thin=thin,
                        adapt=adapt)
        self.assertTrue(isinstance(a, dict),
                        msg='Expect dict return')
        self.assertEqual(a["param_samples"].shape, (n_keep, q),
                         msg='Expect (n_keep, q) array')
        self.assertEqual(a["number_samples"], n_keep,
                         msg='Expect n_keep')
        self.assertEqual(a["final_covariance"].shape, varS1.shape,
                         msg='Expect matching shape')
        self.assertEqual(a["model_variance"].shape, (iters-burn,),
                         msg='Expect (iters-burn,) array')
        self.assertEqual(a["gamma_samples"].shape, (iters-burn, L),
                         msg='Expect (iters-burn, L) array')
        self.assertTrue(isinstance(a["run_time"], float),
                         msg=str('Expect float - got {}'.format(type(a["run_time"]))))
        self.assertTrue(isinstance(a["stage1_accept"], np.ndarray),
                         msg=str('Expect array - got {}'.format(type(a["stage1_accept"]))))
        self.assertTrue(isinstance(a["stage2_accept"], np.ndarray),
                         msg=str('Expect array - got {}'.format(type(a["stage2_accept"]))))
