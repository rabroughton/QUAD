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
        (Histograms, varyList, parmDict, Phases,
         calcControls, pawleyLookup, restraintDict, rigidbodyDict,
         rbIds) = setup_from_gpxfile(GPXfile)
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
        out = check_symmetry(lattice=tmp['lattice'], parmDict=self._parmDict,
                             symmetry=self.Symmetry)
        a, b, c, alpha, beta, gamma = out['lattice']
        self._parmDict = out['parmDict']
        [G, g] = G2latt.cell2Gmat([a, b, c, alpha, beta, gamma])[0]
        LatticeUpdate = G2lattice(G)
        self._parmDict.update(LatticeUpdate)
        return tmp['parmVarDict']

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
    '''
    Check lattice parameters

    Check contents of parameter dictionary and update lattice
    constants accordingly.

    Args:
        * **parmVarDict** (:py:class:`dict`):

    Returns:
        * Dictionary with 2 elements.

        #. `lattice` (:py:class:`tuple`): a, b, c, alpha, beta, gamma
        #. `parmVarDict` (:py:class:`dict`): Updated dictionary
    '''
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


def check_symmetry(lattice, parmDict, symmetry):
    '''
    Check crystal symmetry

    Symmetry Options:
    - Cubic
    - Tetragonal
    - Hexagonal
    - Rhombohedral
    - Orthorhombic
    - Monoclinic
    - Triclinic

    Args:
        * **a** (:py:class:`float`):
        * **b** (:py:class:`float`):
        * **c** (:py:class:`float`):
        * **alpha** (:py:class:`float`):
        * **beta** (:py:class:`float`):
        * **gamma** (:py:class:`float`):
    '''
    if symmetry.lower() == 'cubic':
        out = _sym_cubic(lattice, parmDict)
    elif symmetry.lower() == 'tetragonal':
        out = _sym_tegragonal(lattice, parmDict)
    elif symmetry.lower() == 'hexagonal':
        out = _sym_hexagonal(lattice, parmDict)
    elif symmetry.lower() == 'rhombohedral':
        out = _sym_rhombohedral(lattice, parmDict)
    elif symmetry.lower() == 'orthorhombic':
        out = _sym_orthorhombic(lattice, parmDict)
    elif symmetry.lower() == 'monoclinic':
        out = _sym_monoclinic(lattice, parmDict)
    elif symmetry.lower() == 'triclinic':
        out = _sym_triclinic(lattice, parmDict)
    return out


def _sym_cubic(lattice, parmDict):
    a, b, c, alpha, beta, gamma = lattice
    b = a
    c = a
    alpha = 90.
    beta = 90.
    gamma = 90.
    parmDict['a'] = a
    return dict(lattice=(a, b, c, alpha, beta, gamma),
                parmDict=parmDict)


def _sym_tegragonal(lattice, parmDict):
    a, b, c, alpha, beta, gamma = lattice
    if a is None:
        a = parmDict['a']
    elif c is None:
        c = parmDict['c']
    b = a
    alpha = 90.
    beta = 90.
    gamma = 90.
    parmDict['a'] = a
    parmDict['c'] = c
    return dict(lattice=(a, b, c, alpha, beta, gamma),
                parmDict=parmDict)


def _sym_hexagonal(lattice, parmDict):
    a, b, c, alpha, beta, gamma = lattice
    if a is None:
        a = parmDict['a']
    elif c is None:
        c = parmDict['c']
    b = a
    alpha = 90.
    beta = 120.
    gamma = 90.
    parmDict['a'] = a
    parmDict['c'] = c
    return dict(lattice=(a, b, c, alpha, beta, gamma),
                parmDict=parmDict)


def _sym_rhombohedral(lattice, parmDict):
    a, b, c, alpha, beta, gamma = lattice
    if a is None:
        a = parmDict['a']
    elif alpha is None:
        alpha = parmDict['beta']
    b = a
    c = a
    beta = alpha
    gamma = alpha
    parmDict['a'] = a
    parmDict['alpha'] = alpha
    return dict(lattice=(a, b, c, alpha, beta, gamma),
                parmDict=parmDict)


def _sym_orthorhombic(lattice, parmDict):
    a, b, c, alpha, beta, gamma = lattice
    if a is None:
        a = parmDict['a']
    elif b is None:
        b = parmDict['b']
    elif c is None:
        c = parmDict['c']
    alpha = 90.
    beta = 90.
    gamma = 90.
    parmDict['a'] = a
    parmDict['b'] = b
    parmDict['c'] = c
    return dict(lattice=(a, b, c, alpha, beta, gamma),
                parmDict=parmDict)


def _sym_monoclinic(lattice, parmDict):
    a, b, c, alpha, beta, gamma = lattice
    if a is None:
        a = parmDict['a']
    elif b is None:
        b = parmDict['b']
    elif c is None:
        c = parmDict['c']
    elif beta is None:
        beta = parmDict['beta']
    alpha = 90.
    gamma = 90.
    parmDict['a'] = a
    parmDict['b'] = b
    parmDict['c'] = c
    parmDict['beta'] = beta
    return dict(lattice=(a, b, c, alpha, beta, gamma),
                parmDict=parmDict)


def _sym_triclinic(lattice, parmDict):
    a, b, c, alpha, beta, gamma = lattice
    if a is None:
        a = parmDict['a']
    elif b is None:
        b = parmDict['b']
    elif c is None:
        c = parmDict['c']
    elif alpha is None:
        alpha = parmDict['alpha']
    elif beta is None:
        beta = parmDict['beta']
    elif gamma is None:
        gamma = parmDict['gamma']
    parmDict['a'] = a
    parmDict['b'] = b
    parmDict['c'] = c
    parmDict['alpha'] = alpha
    parmDict['beta'] = beta
    parmDict['gamma'] = gamma
    return dict(lattice=(a, b, c, alpha, beta, gamma),
                parmDict=parmDict)


def G2lattice(G):
    '''
    Transform G to Lattice dictionary
    '''
    return {'0::A0': G[0, 0], '0::A1': G[1, 1], '0::A2': G[2, 2],
            '0::A3': G[0, 1], '0::A4': G[0, 2], '0::A5': G[1, 2]}


def setup_parmDict(rbDict, phaseDict, hapDict, histDict):
    parmDict = {}
    parmDict.update(rbDict)
    parmDict.update(phaseDict)
    parmDict.update(hapDict)
    parmDict.update(histDict)
    return parmDict


def setup_calcControls(Controls, atomIndx, Natoms, FFtables,
                       BLtables, MFtables, maxSSwave):
    calcControls = {}
    calcControls.update(Controls)
    calcControls['atomIndx'] = atomIndx
    calcControls['Natoms'] = Natoms
    calcControls['FFtables'] = FFtables
    calcControls['BLtables'] = BLtables
    calcControls['MFtables'] = MFtables
    calcControls['maxSSwave'] = maxSSwave
    return calcControls


def setup_from_gpxfile(GPXfile):
    '''
    Setup problem using information found in GPX file.

    Reads in information from GPX file using methods found in
    GSAS-II.  This method requires GSAS-II to be installed in
    order to work.

    Args:
        * **GPXfile** (:py:class:`str`): Path/Name of GPX file for analysis.

    Returns:
        * 9-:py:class:`tuple` with the following elements:

        #. `Histograms`:
        #. `varylist`:
        #. `parmDict`:
        #. `Phases`:
        #. `calcControls`:
        #. `pawleyLookup`:
        #. `restraintDict`:
        #. `rigidbodyDict`:
        #. `rbIds`:

    '''
    if not GPXfile:
        # TO DO: Raise name error
        raise NameError('Must input some GPX file')
    # Initallizing variables from input file GPXfile
    G2path.SetBinaryPath()
    # Extract information from GPX files
    Controls = G2stIO.GetControls(GPXfile)
    constrDict, fixedList = G2stIO.GetConstraints(GPXfile)
    restraintDict = G2stIO.GetRestraints(GPXfile)
    Histograms, Phases = G2stIO.GetUsedHistogramsAndPhases(GPXfile)
    rigidbodyDict = G2stIO.GetRigidBodies(GPXfile)
    # Extract information from dictionaries
    rbIds = rigidbodyDict.get('RBIds', {'Vector': [], 'Residue': []})
    rbVary, rbDict = G2stIO.GetRigidBodyModels(rigidbodyDict)
    (Natoms, atomIndx, phaseVary, phaseDict, pawleyLookup,
     FFtables, BLtables, MFtables, maxSSwave) = G2stIO.GetPhaseData(
             Phases, restraintDict, rbIds, Print=False)
    # Setup calcControls dictionary
    calcControls = setup_calcControls(
            Controls, atomIndx, Natoms, FFtables,
            BLtables, MFtables, maxSSwave)
    hapVary, hapDict, controlDict = G2stIO.GetHistogramPhaseData(
            Phases, Histograms, Print=False)
    calcControls.update(controlDict)
    histVary, histDict, controlDict = G2stIO.GetHistogramData(
            Histograms, Print=False)
    calcControls.update(controlDict)
    # Setup variable list
    varyList = []
    varyList = rbVary + phaseVary + hapVary + histVary
    # Setup parameter dictionary
    parmDict = setup_parmDict(rbDict, phaseDict, hapDict, histDict)

    G2stIO.GetFprime(calcControls, Histograms)
    return (Histograms, varyList, parmDict, Phases,
            calcControls, pawleyLookup, restraintDict, rigidbodyDict,
            rbIds)
