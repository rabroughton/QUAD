#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:29:36 2019

@author: prmiles
"""

#import unittest
#from QUAD import gsas_tools as gsas


#class CheckLatticeParameters(unittest.TestCase):
#
#    def test_io(self):
#        parmVarDict = dict(a=0, b=1, c=2, alpha=3, beta=4, gamma=5,
#                           hello='world', ii=27)
#        tmp = gsas.check_lattice_parameters(parmVarDict)
#        self.assertTrue(isinstance(tmp, dict))
#        self.assertTrue(isinstance(tmp['lattice'], tuple),
#                        msg='Expect tuple')
#        self.assertEqual(len(tmp['lattice']), 6,
#                         msg='Expect tuple of length 6')
#        self.assertEqual(tmp['lattice'], (0, 1, 2, 3, 4, 5),
#                         msg='Expected values')
#        self.assertTrue(isinstance(tmp['parmVarDict'], dict),
#                        msg='Expect dict')
#        self.assertEqual(list(tmp['parmVarDict'].keys()), ['hello', 'ii'],
#                         msg='Expect other keys removed')
#        self.assertEqual(tmp['parmVarDict'], dict(hello='world', ii=27))