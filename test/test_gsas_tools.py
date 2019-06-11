#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:29:36 2019

@author: prmiles
"""

import unittest
from QUAD import gsas_tools as gsas
from pseudo_gsas_tools import Calculator
import os


class CheckLatticeParameters(unittest.TestCase):

    def test_io(self):
        parmVarDict = dict(a=0, b=1, c=2, alpha=3, beta=4, gamma=5,
                           hello='world', ii=27)
        tmp = gsas.check_lattice_parameters(parmVarDict)
        self.assertTrue(isinstance(tmp, dict))
        self.assertTrue(isinstance(tmp['lattice'], tuple),
                        msg='Expect tuple')
        self.assertEqual(len(tmp['lattice']), 6,
                         msg='Expect tuple of length 6')
        self.assertEqual(tmp['lattice'], (0, 1, 2, 3, 4, 5),
                         msg='Expected values')
        self.assertTrue(isinstance(tmp['parmVarDict'], dict),
                        msg='Expect dict')
        self.assertEqual(list(tmp['parmVarDict'].keys()), ['hello', 'ii'],
                         msg='Expect other keys removed')
        self.assertEqual(tmp['parmVarDict'], dict(hello='world', ii=27))


class CheckSymmetry(unittest.TestCase):

    def check_output(self, tmp):
        self.assertTrue(isinstance(tmp, dict),
                        msg='Expect dict return')
        self.assertTrue(isinstance(tmp['lattice'], tuple),
                        msg='Expect lattice value as tuple')
        self.assertEqual(len(tmp['lattice']), 6,
                        msg='Expect 6 elements in lattice')
        self.assertTrue(isinstance(tmp['parmDict'], dict),
                        msg='Expect parmDict value as dict')

    def test_cubic(self):
        a = 0.
        lattice = (a, 1., 2., 3., 4., 5.)
        Calc = Calculator(path='test' + os.sep + 'gsas_objects')
        parmDict = Calc._parmDict
        tmp = gsas.check_symmetry(lattice=lattice, parmDict=parmDict,
                                  symmetry='Cubic')
        self.check_output(tmp)
        self.assertEqual(tmp['parmDict']['a'], a)
        self.assertEqual(tmp['lattice'], (a, a, a, 90., 90., 90.),
                         msg='Expect (a, a, a, 90, 90, 90)')

    def test_tetragonal(self):
        symmetry = 'tetragonal'
        a = 0.
        c = 5.
        lattice = (a, 1., c, 3., 4., 5.)
        Calc = Calculator(path='test' + os.sep + 'gsas_objects')
        parmDict = Calc._parmDict.copy()
        tmp = gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                  symmetry=symmetry)
        self.check_output(tmp)
        self.assertEqual(tmp['parmDict']['a'], a)
        self.assertEqual(tmp['parmDict']['c'], c)
        self.assertEqual(tmp['lattice'], (a, a, c, 90., 90., 90.),
                         msg='Expect (a, a, a, 90, 90, 90)')
        lattice = (None, 1., c, 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing a'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (a, 1., None, 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing c'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)

    def test_hexagonal(self):
        symmetry = 'hexagonal'
        a = 0.
        c = 5.
        lattice = (a, 1., c, 3., 4., 5.)
        Calc = Calculator(path='test' + os.sep + 'gsas_objects')
        parmDict = Calc._parmDict.copy()
        tmp = gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                  symmetry=symmetry)
        self.check_output(tmp)
        self.assertEqual(tmp['parmDict']['a'], a)
        self.assertEqual(tmp['parmDict']['c'], c)
        self.assertEqual(tmp['lattice'], (a, a, c, 90., 120., 90.),
                         msg='Expect (a, a, a, 90, 120, 90)')
        lattice = (None, 1., c, 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing a'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (0., 1., None, 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing c'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)

    def test_rhombohedral(self):
        symmetry = 'rhombohedral'
        a = 0.
        alpha = 10.
        lattice = (a, 1., 2., alpha, 4., 5.)
        Calc = Calculator(path='test' + os.sep + 'gsas_objects')
        parmDict = Calc._parmDict.copy()
        tmp = gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                  symmetry=symmetry)
        self.check_output(tmp)
        self.assertEqual(tmp['parmDict']['a'], a)
        self.assertEqual(tmp['lattice'], (a, a, a, alpha, alpha, alpha),
                         msg='Expect (a, a, a, alpha, alpha, alpha)')
        lattice = (None, 1., 2., alpha, 4., 5.)
        with self.assertRaises(KeyError, msg='Missing a'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (a, 1., 2., None, 4., 5.)
        with self.assertRaises(KeyError, msg='Missing alpha'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)

    def test_orthorhombic(self):
        symmetry = 'orthorhombic'
        a = 0.
        b = 1.
        c = 2.
        lattice = (a, b, c, 3., 4., 5.)
        Calc = Calculator(path='test' + os.sep + 'gsas_objects')
        parmDict = Calc._parmDict.copy()
        tmp = gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                  symmetry=symmetry)
        self.check_output(tmp)
        self.assertEqual(tmp['parmDict']['a'], a)
        self.assertEqual(tmp['parmDict']['b'], b)
        self.assertEqual(tmp['parmDict']['c'], c)
        self.assertEqual(tmp['lattice'], (a, b, c, 90., 90., 90.),
                         msg='Expect (a, a, a, 90, 90, 90)')
        lattice = (None, b, c, 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing a'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (0., None, 2., 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing b'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (0., 1., None, 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing c'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)

    def test_monoclinic(self):
        symmetry = 'monoclinic'
        a = 0.
        b = 1.
        c = 2.
        beta = 4.
        lattice = (a, b, c, 3., beta, 5.)
        Calc = Calculator(path='test' + os.sep + 'gsas_objects')
        parmDict = Calc._parmDict.copy()
        tmp = gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                  symmetry=symmetry)
        self.check_output(tmp)
        self.assertEqual(tmp['parmDict']['a'], a)
        self.assertEqual(tmp['parmDict']['b'], b)
        self.assertEqual(tmp['parmDict']['c'], c)
        self.assertEqual(tmp['parmDict']['beta'], beta)
        self.assertEqual(tmp['lattice'], (a, b, c, 90., beta, 90.),
                         msg='Expect (a, a, a, 90, beta, 90)')
        lattice = (None, b, c, 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing a'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (0., None, 2., 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing b'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (0., 1., None, 3., 4., 5.)
        with self.assertRaises(KeyError, msg='Missing c'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (0., 1., 2., 3., None, 5.)
        with self.assertRaises(KeyError, msg='Missing beta'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)

def test_triclinic(self):
        symmetry = 'triclinic'
        a = 0.
        b = 1.
        c = 2.
        alpha = 3.
        beta = 4.
        gamma = 5.
        lattice = (a, b, c, alpha, beta, gamma)
        Calc = Calculator(path='test' + os.sep + 'gsas_objects')
        parmDict = Calc._parmDict.copy()
        tmp = gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                  symmetry=symmetry)
        self.check_output(tmp)
        self.assertEqual(tmp['parmDict']['a'], a)
        self.assertEqual(tmp['parmDict']['b'], b)
        self.assertEqual(tmp['parmDict']['c'], c)
        self.assertEqual(tmp['parmDict']['alpha'], alpha)
        self.assertEqual(tmp['parmDict']['beta'], beta)
        self.assertEqual(tmp['parmDict']['gamma'], gamma)
        self.assertEqual(tmp['lattice'], (a, b, c, alpha, beta, gamma),
                         msg='Expect (a, a, a, alpha, beta, gamma)')
        lattice = (None, b, c, alpha, beta, gamma)
        with self.assertRaises(KeyError, msg='Missing a'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (a, None, c, alpha, beta, gamma)
        with self.assertRaises(KeyError, msg='Missing b'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (a, b, None, alpha, beta, gamma)
        with self.assertRaises(KeyError, msg='Missing c'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (a, b, c, None, beta, gamma)
        with self.assertRaises(KeyError, msg='Missing alpha'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (a, b, c, alpha, None, gamma)
        with self.assertRaises(KeyError, msg='Missing beta'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        lattice = (a, b, c, alpha, beta, None)
        with self.assertRaises(KeyError, msg='Missing gamma'):
            gsas.check_symmetry(lattice=lattice, parmDict=parmDict.copy(),
                                symmetry=symmetry)
        