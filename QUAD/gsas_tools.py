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

try:
    import GSASIIpath as G2path
except ModuleNotFoundError:
    gsas_install_display(module='GSASIIpath')


class Calculator:
    '''
    This script calls the neccessary functions for QUAD from
    the GSAS-II program which are regularly updated by the GSAS-II developers.
    (https://subversion.xray.aps.anl.gov/trac/pyGSAS)

    How it works:

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
    '''
    def __init__(self, GPXfile=None):
        '''
        Initialize the setup using an existing GPXfile
        '''
        if not GPXfile:
            # TO DO: Raise name error
            raise NameError('Must input some GPX file')
        # Initallizing variables from input file GPXfile
        G2path.SetBinaryPath()
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
                    (self._tth < self._upperLimit) == True)]  # noqa: E712
            self._SingleXtal = False
        except:  # noqa: E722
            self._tth = None
            self._MaxDiffSig = None
            self._Fosq = 0
            self._FoSig = 0
            self._Fcsq = 0
            self._SingleXtal = True

    def Calculate(self):
        '''
        Calculate the profile fit for the current parameter setup

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
        yc1 = G2strMath.getPowderProfile(
                self._parmDict, self._tthsample, self._varyList,
                self._Histograms[list(self._Histograms.keys())[0]],
                self._Phases, self._calcControls, self._pawleyLookup)[0]
        return yc1

    def UpdateLattice(self, parmVarDict):
        '''
        Update the lattice parameters in the current model
        '''
        tmp = check_lattice_parameters(parmVarDict)
        a, b, c, alpha, beta, gamma = tmp['lattice']
        parmVarDict = tmp['parmVarDict']

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


def check_lattice_parameters(parmVarDict):
    a, b, c, alpha, beta, gamma = None, None, None, None, None, None
    keys = list(parmVarDict.keys()).copy()
    for key in keys:
        if key == 'a':
            a = parmVarDict[key]
            parmVarDict.pop(key, None)
        elif key == 'b':
            b = parmVarDict[key]
            parmVarDict.pop(key, None)
        elif key == 'c':
            c = parmVarDict[key]
            parmVarDict.pop(key, None)
        elif key == 'alpha':
            alpha = parmVarDict[key]
            parmVarDict.pop(key, None)
        elif key == 'beta':
            beta = parmVarDict[key]
            parmVarDict.pop(key, None)
        elif key == 'gamma':
            gamma = parmVarDict[key]
            parmVarDict.pop(key, None)
    return dict(lattice=(a, b, c, alpha, beta, gamma),
                parmVarDict=parmVarDict)
