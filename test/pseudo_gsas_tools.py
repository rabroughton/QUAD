# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os

class Calculator:
    
    def __init__(self, path='gsas_objects'):
        # load histogram
        self._Histograms = loadfrompickle(path + os.sep + 'histograms')
        # load lower limit
        self._lowerLimit = np.load(path + os.sep + 'lowerlimit')
        # load upper limit
        self._upperLimit = np.load(path + os.sep + 'upperlimit')
        # load tth
        self._tth = np.load(path + os.sep + 'tth')
        # load tthsample
        self._tthsample = np.load(path + os.sep + 'tthsample')
        # load x and y
        self._x = np.load(path + os.sep + 'x')
        self._y = np.load(path + os.sep + 'y')
        self._shape = self._y.shape
        self._n = self._y.size
        # load variables
        self._variables = np.load(path + os.sep + 'variables')
        self._paramList = np.load(path + os.sep + 'paramlist')
        self._parmDict = loadfrompickle(path + os.sep + 'parmdict')
        self._Phases = loadfrompickle(path + os.sep + 'phases')
        self._calcControls = loadfrompickle(path + os.sep + 'calccontrols')
        self._pawleyLookup = loadfrompickle(path + os.sep + 'pawleylookup')
        self._restraintDict = loadfrompickle(path + os.sep + 'restraintdict')
        self._rigidbodyDict = loadfrompickle(path + os.sep + 'rigidbodydict')
        self._rbIds = loadfrompickle(path + os.sep + 'rbids')
        self._varyList = loadfrompickle(path + os.sep + 'varylist')

    def Calculate(self):
        return np.random.random_sample(size=self._shape)

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

    def mock_setup(self, base=True):
        Histograms = self._Histograms
        varyList = self._varyList
        parmDict = self._parmDict
        Phases = self._Phases
        calcControls = self._calcControls
        pawleyLookup = self._pawleyLookup
        restraintDict = self._restraintDict
        rigidbodyDict = self._rigidbodyDict
        rbIds = self._rbIds
        if base is True:
            return (Histograms, varyList, parmDict, Phases,
                    calcControls, pawleyLookup, restraintDict,
                    rigidbodyDict, rbIds)
        else:
            return ([], varyList, parmDict, Phases,
                    calcControls, pawleyLookup, restraintDict,
                    rigidbodyDict, rbIds)

def save2pickle(obj, fn):
    with open(fn + '.pkl', 'wb') as h:
        pickle.dump(obj, h)


def loadfrompickle(fn):
    with open(fn + '.pkl', 'rb') as h:
        tmp = pickle.load(h)
    return tmp


if __name__ == "__main__":
    Calc = Calculator()
    print(Calc._Histograms)
    print(Calc._lowerLimit)