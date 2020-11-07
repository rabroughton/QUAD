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
        symmetry = {}
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
        # define phase symmetries, currently only works for one phase
        phase = G2stIO.GetPhaseNames(GPXfile)
        phaseData = {}
        for name in phase:
            phaseData[name] = G2stIO.GetAllPhaseData(GPXfile, name)
            symmetry[name] = phaseData[name]['General']['SGData']['SGSys']

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
        self._symmetry = symmetry
        self._phase = phase
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

    def UpdateLattice(self, parmVarDict, phase_index):
        '''
        Update the lattice parameters in the current model for all phases
        '''

# Symmetry Options:
# cubic, tetragonal, hexagonal, rhombohedral, trigonal,
# orthorhombic, monoclinic, triclinic

        if self._symmetry[self._phase[phase_index]] == 'cubic':
            a = parmVarDict[str(phase_index) + '::a']
            b = a
            c = a
            alpha = 90.
            beta = 90.
            gamma = 90.

        elif self._symmetry[self._phase[phase_index]] == 'tetragonal':
            a = parmVarDict[str(phase_index) + '::a']
            b = a
            c = parmVarDict[str(phase_index) + '::c']
            alpha = 90.
            beta = 90.
            gamma = 90.

        elif self._symmetry[self._phase[phase_index]] == 'hexagonal':
            a = parmVarDict[str(phase_index) + '::a']
            b = a
            c = parmVarDict[str(phase_index) + '::c']
            alpha = 90.
            beta = 90.
            gamma = 120.

        elif self._symmetry[self._phase[phase_index]] == 'trigonal':
            a = parmVarDict[str(phase_index) + '::a']
            b = a
            c = parmVarDict[str(phase_index) + '::c']
            alpha = 90.
            beta = 90.
            gamma = 120.

        elif self._symmetry[self._phase[phase_index]] == 'rhombohedral':
            a = parmVarDict[str(phase_index) + '::a']
            b = a
            c = a
            alpha = parmVarDict[str(phase_index) + '::alpha']
            beta = alpha
            gamma = alpha

        elif self._symmetry[self._phase[phase_index]] == 'orthorhombic':
            a = parmVarDict[str(phase_index) + '::a']
            b = parmVarDict[str(phase_index) + '::b']
            c = parmVarDict[str(phase_index) + '::c']
            alpha = 90.
            beta = 90.
            gamma = 90.

        elif self._symmetry[self._phase[phase_index]] == 'monoclinic':
            a = parmVarDict[str(phase_index) + '::a']
            b = parmVarDict[str(phase_index) + '::b']
            c = parmVarDict[str(phase_index) + '::c']
            beta = parmVarDict[str(phase_index) + '::beta']
            alpha = 90.
            gamma = 90.

        elif self._symmetry[self._phase[phase_index]] == 'triclinic':
            a = parmVarDict[str(phase_index) + '::a']
            b = parmVarDict[str(phase_index) + '::b']
            c = parmVarDict[str(phase_index) + '::c']
            alpha = parmVarDict[str(phase_index) + '::alpha']
            beta = parmVarDict[str(phase_index) + '::beta']
            gamma = parmVarDict[str(phase_index) + '::gamma']

        A = G2latt.cell2A([a, b, c, alpha, beta, gamma])

        LatticeUpdate = {str(phase_index) + '::A0': A[0],
                         str(phase_index) + '::A1': A[1],
                         str(phase_index) + '::A2': A[2],
                         str(phase_index) + '::A3': A[3],
                         str(phase_index) + '::A4': A[4],
                         str(phase_index) + '::A5': A[5]}

        self._parmDict.update(LatticeUpdate)
        return parmVarDict

    def UpdateParameters(self, parmVarDict=None):
        '''
            Update parameters in the current model
        '''
        for phase_index in range(len(self._phase)):
            if ((str(phase_index) + '::a') in parmVarDict.keys()) is True:
                parmVarDict = self.UpdateLattice(parmVarDict, phase_index)

        if 'EXT' in parmVarDict.keys():
            for key in self._parmDict.keys():
                if 'Eg' in key:
                    parmVarDict[key] = parmVarDict['EXT']

            parmVarDict.pop('EXT', None)

        self._parmDict.update(parmVarDict)

    def convert_lattice(self, paramList, params):
        '''
            Convert between A and unit cell parameters. A is defined in GSAS-II
            as A = [G11, G22, G33, 2*G12, 2*G13, 2*G23] with G as the
            reciprocal metric tensor elements.
        '''
        latparamList = ['0::A0', '1::A0', '2::A0', '3::A0', '4::A0', '5::A0',
                        '6::A0', '7::A0', '8::A0', '9::A0', '10::A0']
        check = any(item in paramList for item in latparamList)
        if check is True:
            return G2latt.A2cell(params)
        else:
            return G2latt.cell2A(params)
