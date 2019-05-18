# Here load up the package outside of the function to make it a one time thing
# minimal cost because this is only loaded once
import pylab as plt
import numpy as np
from .utilities import gsas_install_display

try:
    import GSASIIstrIO as G2stIO
except ModuleNotFoundError:
    gsas_install_display(module='GSASIIstrIO')

try:
    import GSASIIlattice as G2latt
except ModuleNotFoundError:
    gsas_install_display(module='GSASIIlattice')

try:
    import GSASIIstrMath as G2strMath
except ModuleNotFoundError:
    gsas_install_display(module='GSASIIstrMath')


class Calculator:
    '''
    How it works

    Variables are stored in various python dictionaries
    (parmDict, calcControls, Phases, and Histograms)
    for more information about dictionaries see
    http://www.tutorialspoint.com/python/python_dictionary.htm

    Variables of interest that are stored in parmDict include:
    Background parameters: :0:Back:x (x represent the order)
    X-ray absorption coef: :0:Absorption
    X-ray wavelength: :0:Lam
    Intensity scale factor: :0:Scale
    2-theta offset: :0:Zero
    Various peak parameters: :0:U, :0:V, :0:W, :0:L, :0:X, :0:SH/L

    Variables of interest that are stored in calcControls include:
    Model for background: :0:bakType (default is chebyschev)
    available models are: chebyschev (default),
    cosine, lin interpolate, inv interpolate, and log interpolate

    Phases contains information about the phases in the material
    (Space group, atom positions, Debye-Waller factors,
    preferred orientation, ....)
    
    Change between isotropic and anisotropic Debye-Waller factors:
        Phases[Phases.keys()[0]]['Atoms'][X][9] = 'I' (isotropic)
        or 'A' (anisotropic) [X represents atom number (0 to n-1)]

    Debye-Waller factors are controled by:
        Phases[Phases.keys()[0]]['Atoms'][X][10-16]
        (isotropic case only used element 10)

    Histograms contains information about the diffraction data
    file (this should not be changed)

    This script calls the neccessary functions for QUAD from
    the GSAS-II program which are regularly updated by the GSAS-II developers.
    '''
    def __init__(self, GPXfile=None):
        '''
        Initialize the setup using an existing GPXfile
        '''
        if not GPXfile:
            # TO DO: Raise name error
            raise NameError('Must input some GPX file')
        # Initallizing variables from input file GPXfile
        # Code from Chris
        # time0 = time.time()
        varyList = []
        parmDict = {}
        Controls = G2stIO.GetControls(GPXfile)
        calcControls = {}
        calcControls.update(Controls)
        constrDict, fixedList = G2stIO.GetConstraints(GPXfile)
        restraintDict = G2stIO.GetRestraints(GPXfile)
        Histograms, Phases = G2stIO.GetUsedHistogramsAndPhases(GPXfile)
        rigidbodyDict = G2stIO.GetRigidBodies(GPXfile)
        rbIds = rigidbodyDict.get('RBIds', {'Vector': [], 'Residue': []})
        rbVary, rbDict = G2stIO.GetRigidBodyModels(rigidbodyDict)
        (Natoms, atomIndx, phaseVary, phaseDict, pawleyLookup,
         FFtables, BLtables, MFtables, maxSSwave) = G2stIO.GetPhaseData(
                 Phases, restraintDict, rbIds, Print=False)
        calcControls['atomIndx'] = atomIndx
        calcControls['Natoms'] = Natoms
        calcControls['FFtables'] = FFtables
        calcControls['BLtables'] = BLtables
        calcControls['MFtables'] = MFtables
        calcControls['maxSSwave'] = maxSSwave
        hapVary, hapDict, controlDict = G2stIO.GetHistogramPhaseData(
                Phases, Histograms, Print=False)
        calcControls.update(controlDict)
        histVary, histDict, controlDict = G2stIO.GetHistogramData(
                Histograms, Print=False)
        calcControls.update(controlDict)
        varyList = rbVary+phaseVary+hapVary+histVary
        parmDict.update(rbDict)
        parmDict.update(phaseDict)
        parmDict.update(hapDict)
        parmDict.update(histDict)
        G2stIO.GetFprime(calcControls, Histograms)

        # Save the instance parameters
        self._Histograms = Histograms
        self._varyList = varyList
        self._parmDict = parmDict
        self._Phases = Phases
        self._calcControls = calcControls
        self._pawleyLookup = pawleyLookup
        self._restraintDict = restraintDict
        self._rigidbodyDict = rigidbodyDict
        self._rbIds = rbIds
        try:
            self._lowerLimit = Histograms[
                    list(Histograms.keys())[0]][u'Limits'][1][0]
            self._upperLimit = Histograms[
                    list(Histograms.keys())[0]][u'Limits'][1][1]
            self._tth = Histograms[list(Histograms.keys())[0]]['Data'][0][0:-1]
            self._tthsample = self._tth[np.where(
                    (self._tth > self._lowerLimit) &
                    (self._tth < self._upperLimit) == True)]
            self._SingleXtal = False
        except:
            self._tth = None
            self._MaxDiffSig = None
            self._Fosq = 0
            self._FoSig = 0
            self._Fcsq = 0
            self._SingleXtal = True

# print 'variable initallizing time: %.3fs...Successful'%(time.time()-time0)

    def Calculate(self):
        '''
        Calculate the f and g for the current parameter setup
        '''
        # time0 = time.time()
        '''
            #Load the parameters
            Histograms = self._Histograms
            varyList = self._varyList
            parmDict = self._parmDict
            Phases = self._Phases
            calcControls = self._calcControls
            pawleyLookup = self._pawleyLookup
            restraintDict = self._restraintDict
            rbIds = self._rbIds
        '''
        # getPowderProfile(parmDict, x, varylist, Histogram,
        #                  Phases, calcControls, pawleyLookup):
        yc1 = G2strMath.getPowderProfile(
                self._parmDict, self._tthsample, self._varyList,
                self._Histograms[list(self._Histograms.keys())[0]],
                self._Phases, self._calcControls, self._pawleyLookup)[0]
        return yc1

    def CalculateSXtal(self):
        G2strMath.GetFobsSq(self._Histograms, self._Phases,
                            self._parmDict, self._calcControls)
        Fosq, FoSig, Fcsq = [], [], []
        for key in self._Histograms.keys():
            for hkl in range(len(self._Histograms[key]['Data']['RefList'])):
                # Diff = np.abs(Calc._Histograms[key]['Data']['RefList'][hkl][5] - Calc._Histograms[key]['Data']['RefList'][hkl][5]) / Calc._Histograms[key]['Data']['RefList'][hkl][6]
                # if np.abs(self._Histograms[key]['Data']['RefList'][hkl][5] - self._Histograms[key]['Data']['RefList'][hkl][7]) / self._Histograms[key]['Data']['RefList'][hkl][6] < self._MaxDiffSig:
                #   Fosq.append(self._Histograms[key]['Data']['RefList'][hkl][5])
                #   Fcsq.append(self._Histograms[key]['Data']['RefList'][hkl][7])
                #   FoSig.append(self._Histograms[key]['Data']['RefList'][hkl][6])

                Fosq.append(self._Histograms[key]['Data']['RefList'][hkl][5])
                Fcsq.append(self._Histograms[key]['Data']['RefList'][hkl][7])
                FoSig.append(self._Histograms[key]['Data']['RefList'][hkl][6])

        if self._MaxDiffSig:
            print('Yes')
            Fosq = np.array(Fosq)
            Fcsq = np.array(Fcsq)
            FoSig = np.array(FoSig)
            Index = np.abs(Fosq-Fcsq)/FoSig < self._MaxDiffSig

            self._Fosq = Fosq[Index]
            self._Fcsq = Fcsq[Index]
            self._FoSig = FoSig[Index]
        else:
            self._Fosq = np.array(Fosq)
            self._Fcsq = np.array(Fcsq)
            self._FoSig = np.array(FoSig)

    def CalculateDeriv(self):
        return np.sum(G2strMath.getPowderProfileDervMP(
                self._parmDict, self._tthsample, self._varyList,
                self._Histograms[self._Histograms.keys()[0]],
                self._Phases, self._rigidbodyDict, self._calcControls,
                self._pawleyLookup), axis=1)

    def CalculateHessian(self):
        values = np.array(G2strMath.Dict2Values(
                self._parmDict, self._varyList))
        Z = G2strMath.HessRefine(
                values,
                [self._Histograms, self._Phases,
                 self._restraintDict, self._rigidbodyDict],
                self._parmDict, self._varyList,
                self._calcControls, self._pawleyLookup, None)
        return Z[1]

    def CalculateJacob(self):
        values = np.array(G2strMath.Dict2Values(
                self._parmDict, self._varyList))
        Y = G2strMath.dervRefine(
                values,
                [self._Histograms, self._Phases,
                 self._restraintDict, self._rigidbodyDict],
                self._parmDict, self._varyList,
                self._calcControls, self._pawleyLookup, None)
        return Y
#        return np.sum(Y, axis=1)

    def pinv(a, rcond=1e-15):
        '''
        Compute the (Moore-Penrose) pseudo-inverse of a matrix.

        Modified from numpy.linalg.pinv; \
        Assumes a is Hessian & returns no. zeros found
        Calculate the generalized inverse of a matrix using its
        singular-value decomposition (SVD) and including all
        *large* singular values.

        :param array a: (M, M) array_like - here assumed to be LS Hessian
          Matrix to be pseudo-inverted.
        :param float rcond: Cutoff for small singular values.
          Singular values smaller (in modulus) than
          `rcond` * largest_singular_value (again, in modulus)
          are set to zero.

        :returns: B : (M, M) ndarray
          The pseudo-inverse of `a`

        Raises: LinAlgError
          If the SVD computation does not converge.

        .. note::
          The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
          defined as: "the matrix that 'solves' [the least-squares problem]
          :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
          :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

        It can be shown that if :math:`Q_1 \\Sigma Q_2^T = A` is the singular
        value decomposition of A, then
        :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are
        orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting
        of A's so-called singular values, (followed, typically, by
        zeros), and then :math:`\\Sigma^+` is simply the diagonal matrix
        consisting of the reciprocals of A's singular values
        (again, followed by zeros). [1]

        References:
        .. [1] G. Strang, *Linear Algebra and Its Applications*, \
        2nd Ed., Orlando, FL, Academic Press, Inc., 1980, pp. 139-142.
        '''
        u, s, vt = np.linalg.svd(a, 0)
        cutoff = rcond*np.maximum.reduce(s)
        s = np.where(s > cutoff, 1./s, 0.)
        nzero = s.shape[0] - np.count_nonzero(s)
        # res = np.dot(np.transpose(vt),
        #              np.multiply(s[:, np.newaxis], np.transpose(u)))
        res = np.dot(vt.T, s[:, np.newaxis]*u.T)
        return res, nzero

    # Get CovMatrix
    # There are alternate methods to pull the covariance matrix out
    def getCovMatrix(self, ParmVarDict):
        # [self._Histograms,self._Phases,self._restraintDict,self._rigidbodyDict],
        # self._parmDict,ParmVarDict,self._calcControls,self._pawleyLookup
        '''
        Core optimization routines, shared between SeqRefine and Refine

        :returns: 5-tuple of ifOk (bool), Rvals (dict),\
        result, covMatrix, sig
        '''
        HistoPhases = (self._Histograms, self._Phases,
                       self._restraintDict, self._rigidbodyDict)
#        xtol = 1.e-6

        values = np.array(G2strMath.Dict2Values(self._parmDict, ParmVarDict))
        Chisq = G2strMath.errRefine(
                values, HistoPhases, self._parmDict, ParmVarDict,
                self._calcControls, self._pawleyLookup, None)
        Nobs = len(self.Calculate())
        Chisq = np.sum(Chisq**2)
        GOF = np.sqrt(Chisq/(Nobs-len(ParmVarDict)))
        Yvec, Amat = G2strMath.HessRefine(
                values, HistoPhases, self._parmDict,
                ParmVarDict, self._calcControls, self._pawleyLookup,
                None)
        Adiag = np.sqrt(np.diag(Amat))
        Anorm = np.outer(Adiag, Adiag)
        Amat = np.array(Amat/Anorm)
        # Moore-Penrose inversion (via SVD) & count of zeros
        Bmat, Nzero = self.pinv(Amat)
        Bmat = Bmat/Anorm

        covMatrix = Bmat*GOF**2
        return covMatrix

#    def SaveData(self,OutputFileName):
#        x, yc, yb  = self.Calculate()
#        outputFile = open( OutputFileName, 'w' )
#        for i in xrange( len( x ) ):
#            outputFile.write( str(x[i]) + '\t'
#                             + str(yc[i]) + '\t' + str(yb[i]) + '\n')

    def Draw(self, SaveFigure=None):
        '''
        Calculate the f and g for the current parameter setup
        '''
        x, yc, yb = self.Calculate()

        plt.plot(x, yc + yb)
        if SaveFigure:
            plt.savefig(SaveFigure)
        else:
            plt.show()

    def UpdateLattice(self, parmVarDict):
        a, b, c, alpha, beta, gamma = None, None, None, None, None, None
        for key in parmVarDict.keys():
            if key == 'a':
                a = parmVarDict[key]
                parmVarDict.pop(key, None)
            elif key == 'b':
                a = parmVarDict[key]
                parmVarDict.pop(key, None)
            elif key == 'c':
                c = parmVarDict[key]
                parmVarDict.pop(key, None)
            elif key == 'alpha':
                a = parmVarDict[key]
                parmVarDict.pop(key, None)
            elif key == 'beta':
                a = parmVarDict[key]
                parmVarDict.pop(key, None)
            elif key == 'gamma':
                a = parmVarDict[key]
                parmVarDict.pop(key, None)


# Symmetry Options:
# Cubic, Tetragonal, Hexagonal, Rhombohedral,
# Orthorhombic, Monoclinic, Triclinic

        if self.Symmetry == 'Cubic':
            b = a
            c = a
            alpha = 90.
            beta = 90.
            gamma = 90.
            self._parmDict['a'] = a
        elif self.Symmetry == 'Tetragonal':
            if a is None:
                a = self._parmDict['a']
            elif c is None:
                c = self._parmDict['c']
            b = a
            alpha = 90.
            beta = 90.
            gamma = 90.
            self._parmDict['a'] = a
            self._parmDict['c'] = c

        elif self.Symmetry == 'Hexagonal':
            if a is None:
                a = self._parmDict['a']
            elif c is None:
                c = self._parmDict['c']
            b = a
            alpha = 90.
            beta = 120.
            gamma = 90.
            self._parmDict['a'] = a
            self._parmDict['c'] = c
        elif self.Symmetry == 'Rhombohedral':
            if a is None:
                a = self._parmDict['a']
            elif alpha is None:
                alpha = self._parmDict['beta']
            b = a
            c = a
            beta = alpha
            gamma = alpha
            self._parmDict['a'] = a
            self._parmDict['alpha'] = alpha
        elif self.Symmetry == 'Orthorhombic':
            if a is None:
                a = self._parmDict['a']
            elif b is None:
                b = self._parmDict['b']
            elif c is None:
                c = self._parmDict['c']
            alpha = 90.
            beta = 90.
            gamma = 90.
            self._parmDict['a'] = a
            self._parmDict['b'] = b
            self._parmDict['c'] = c
        elif self.Symmetry == 'Monoclinic':
            if a is None:
                a = self._parmDict['a']
            elif b is None:
                b = self._parmDict['b']
            elif c is None:
                c = self._parmDict['c']
            elif beta is None:
                beta = self._parmDict['beta']
            alpha = 90.
            gamma = 90.
            self._parmDict['a'] = a
            self._parmDict['b'] = b
            self._parmDict['c'] = c
            self._parmDict['beta'] = beta
        elif self.Symmetry == 'Triclinic':
            if a is None:
                a = self._parmDict['a']
            elif b is None:
                b = self._parmDict['b']
            elif c is None:
                c = self._parmDict['c']
            elif alpha is None:
                alpha = self._parmDict['alpha']
            elif beta is None:
                beta = self._parmDict['beta']
            elif gamma is None:
                gamma = self._parmDict['gamma']
            self._parmDict['a'] = a
            self._parmDict['b'] = b
            self._parmDict['c'] = c
            self._parmDict['alpha'] = alpha
            self._parmDict['beta'] = beta
            self._parmDict['gamma'] = gamma

        [G, g] = G2latt.cell2Gmat([a, b, c, alpha, beta, gamma])[0]

        LatticeUpdate = {'0::A0': G[0, 0], '0::A1': G[1, 1], '0::A2': G[2, 2],
                         '0::A3': G[0, 1], '0::A4': G[0, 2], '0::A5': G[1, 2]}
        # print G
        self._parmDict.update(LatticeUpdate)
        return parmVarDict

    def UpdateParameters(self, parmVarDict=None):
        '''
            Update parameters in the current model
        '''
        for key in parmVarDict.keys():
            if key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
                parmVarDict = self.UpdateLattice(parmVarDict)
                break

        if 'EXT' in parmVarDict.keys():
            for key in self._parmDict.keys():
                if 'Eg' in key:
                    parmVarDict[key] = parmVarDict['EXT']

            parmVarDict.pop('EXT', None)

        self._parmDict.update(parmVarDict)
        '''
            #Load the parameters
            Histograms = self._Histograms
            varyList = self._varyList
            parmDict = self._parmDict
            Phases = self._Phases
            calcControls = self._calcControls
            pawleyLookup = self._pawleyLookup
            restraintDict = self._restraintDict
            rbIds = self._rbIds
        '''
# print 'Successful'
