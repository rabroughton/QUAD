# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os

class Calculator:
    
    def __init__(self, path='gsas_objects'):
        # load histogram
        with open(path + os.sep + "histograms.pkl", "rb") as input_file:
            self._Histograms = pickle.load(input_file)
        # load lower limit
        self._lowerLimit = np.load(path + os.sep + 'lowerlimit',
                                   allow_pickle=True)
        # load upper limit
        self._upperLimit = np.load(path + os.sep + 'upperlimit',
                                   allow_pickle=True)
        # load tth
        self._tth = np.load(path + os.sep + 'tth', allow_pickle=True)
        # load tthsample
        self._tthsample = np.load(path + os.sep + 'tthsample',
                                  allow_pickle=True)
        # load x and y
        self._x = np.load(path + os.sep + 'x', allow_pickle=True)
        self._y = np.load(path + os.sep + 'y', allow_pickle=True)
        self._shape = self._y.shape
        self._n = self._y.size
        # load variables
        self._variables = np.load(path + os.sep + 'variables',
                                  allow_pickle=True)
        self._paramList = np.load(path + os.sep + 'paramlist',
                                  allow_pickle=True)

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