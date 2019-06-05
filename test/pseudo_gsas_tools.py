# -*- coding: utf-8 -*-
import numpy as np


class Calculator:
    
    def __init__(self, shape=(100,)):
        self.shape = shape

    def Calculate(self):
        return np.random.random_sample(size=self.shape)

    def UpdateLattice(self, parmVarDict):
        for key in parmVarDict.keys():
            if key == 'a':
                parmVarDict.pop(key, None)
            elif key == 'b':
                parmVarDict.pop(key, None)
            elif key == 'c':
                parmVarDict.pop(key, None)
            elif key == 'alpha':
                parmVarDict.pop(key, None)
            elif key == 'beta':
                parmVarDict.pop(key, None)
            elif key == 'gamma':
                parmVarDict.pop(key, None)

        return parmVarDict

    def UpdateParameters(self, parmVarDict=None):
        self._updatedparameters = True