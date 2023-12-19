# coding=UTF-8
import pyscf
import os
import sys
import numpy
import struct
from functools import reduce
from pyscf import tools
import time
import tempfile
import copy
import glob
import shutil
from subprocess import check_call, check_output, CalledProcessError
from pyscf.lib import logger
from pyscf import lib
from pyscf import tools
from pyscf import ao2mo
from pyscf import mcscf, fci
import numpy as np

import iCI_InputFile_Generator

# Constant/Configuration

iCI_ProgramName = "ICI_CPP"
FILE_RDM1_NAME = "rdm1.csv"
FILE_RDM2_NAME = "rdm2.csv"

# default configuration

CONFIG = {
    "inputfile": "iCI.inp",
    "rotatemo": 0,
    "state": [[0, 0, 1, [1]]],  # spintwo,irrep,nstates,weight
    "cmin": 1e-4,
    "perturbation": 0,
    "dumprdm": 0,
    "relative": 0,
    "inputocfg": 0,
    "etol": 1e-6,
}

# Other Util


def writeIntegralFile(iciobj, h1eff, eri_cas, ncas, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    if iciobj.groupname is not None and iciobj.orbsym is not []:
        # First removing the symmetry forbidden integrals. This has been done using
        # the pyscf internal irrep-IDs (stored in iciobj.orbsym)
        orbsym = numpy.asarray(iciobj.orbsym) % 10
        pair_irrep = (orbsym.reshape(-1, 1) ^ orbsym)[numpy.tril_indices(ncas)]
        sym_forbid = pair_irrep.reshape(-1, 1) != pair_irrep.ravel()
        eri_cas = ao2mo.restore(4, eri_cas, ncas)
        eri_cas[sym_forbid] = 0
        eri_cas = ao2mo.restore(8, eri_cas, ncas)
        # Convert the pyscf internal irrep-ID to molpro irrep-ID, why ?
        # orbsym = numpy.asarray(
        #     symmetry.convert_orbsym(iciobj.groupname, orbsym))
    else:
        orbsym = []
        eri_cas = ao2mo.restore(8, eri_cas, ncas)

    if not os.path.exists(iciobj.runtimedir):
        os.makedirs(iciobj.runtimedir)

    # The name of the FCIDUMP file, default is "FCIDUMP".
    integralFile = os.path.join(iciobj.runtimedir, iciobj.integralfile)
    tools.fcidump.from_integrals(integralFile, h1eff, eri_cas, ncas,
                                 neleca+nelecb, ecore, ms=abs(neleca-nelecb),
                                 orbsym=orbsym,
                                 )
    return integralFile


def execute_iCI(iciobj):
    iciobj.inputfile = iciobj.config["taskname"] + \
        "_" + str(iciobj.runtime) + ".inp"
    iciobj.outputfile = iciobj.config["taskname"] + \
        "_" + str(iciobj.runtime) + ".out"
    iciobj.energydat = iciobj.inputfile + ".enedat"
    iciobj.runtime += 1

    # Write input file

    # consider aimed selection

    iCI_InputFile_Generator._Generate_InputFile_iCI(inputfilename=iciobj.inputfile,
                                                    Segment=iciobj.config["segment"],
                                                    nelec_val=iciobj.config["nvalelec"],
                                                    rotatemo=iciobj.config["rotatemo"],
                                                    cmin=iciobj.config["cmin"],
                                                    perturbation=iciobj.config["perturbation"],
                                                    dumprdm=iciobj.config["dumprdm"],
                                                    relative=iciobj.config["relative"],
                                                    Task=iciobj.config["task"],
                                                    inputocfg=iciobj.config["inputocfg"],
                                                    etol=iciobj.config["etol"],
                                                    selection=iciobj.config["selection"],
                                                    direct=iciobj.config["direct"]
                                                    )

    # execute

    os.system("%s %s > %s" %
              (iciobj.executable, iciobj.inputfile, iciobj.outputfile))

    return iciobj


def read_energy(iciobj):
    calc_e = []
    # print("open file %s" % (iciobj.energydat))
    binfile = open(iciobj.energydat, "rb")
    # size = os.path.getsize(binfile)
    # if size != iciobj.nroots * 8:
    #     print("Fatal error in read_energy()")
    #     exit(1)
    # 读取能量
    for i in range(iciobj.nroots):
        data = binfile.read(8)
        calc_e.append(float(struct.unpack('d', data)[0]))
        # print(struct.unpack('d', data))
    return calc_e


class iCI(lib.StreamObject):
    r'''iCI program interface and object to hold iCI program input
    parameters.

    See also the reference:
    (1) https://pubs.acs.org/doi/10.1021/acs.jctc.9b01200
    (2) https://pubs.acs.org/doi/10.1021/acs.jctc.0c01187

    Attributes:

    (1) all the subroutine do in a state-averaged way, never do in  a state-specific way!
    (2)

    Examples:

    '''

    def __init__(self,
                 mo_coeff=None,
                 cmin=1e-4,
                 tol=1e-8,
                 inputocfg=0,
                 mol=None, state=None):
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout  # useless for iCI
            self.verbose = mol.verbose

        self.executable = os.getenv(iCI_ProgramName)
        self.runtimedir = '.'
        self.integralfile = "FCIDUMP"

        self.config = copy.deepcopy(CONFIG)
        # set config
        self.config["segment"] = None
        # self.config["nvalelec"] = nvalelec
        # if segment == None or nvalelec == None:
        #     print("Fatal Error in initializing objects of iCI")
        #     exit(1)
        self.config["cmin"] = str(cmin)
        self.config["etol"] = tol
        self.config["rotatemo"] = 0  # 永远都不旋转轨道
        # self.config["perturbation"] = perturbation  # 基本不会做 perturbation
        self.config["perturbation"] = 0  # 基本不会做 perturbation
        # self.config["dumprdm"] = dumprdm
        self.config["dumprdm"] = 2  # 无论如何都计算密度矩阵
        self.config["relative"] = 0  # 无论如何都不考虑相对论
        self.config["inputocfg"] = inputocfg
        self.config["hasfzc"] = False
        self.config["nfzc"] = 0
        self.config["readinocfg"] = None
        if state is not None:
            self.config["state"] = state
        self.config["task"], self.spin, self.weight = iCI_InputFile_Generator._generate_task_spinarray_weight(
            state)
        # print(self.weight)
        # self.weights = self.weight
        self.config["selection"] = 1
        self.config["aimedtarget"] = 0.0
        self.config["taskname"] = "iCI"
        self.config["direct"] = 0
        if mol is not None and mol.symmetry:
            self.groupname = mol.groupname
        else:
            self.groupname = None

        if mol is not None and mol.nao == mo_coeff.shape[0] and mol.symmetry:
            OrbSym = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                               mo_coeff)
            IrrepOrb = []
            for i in range(len(OrbSym)):
                IrrepOrb.append(pyscf.symm.irrep_name2id(
                    mol.groupname, OrbSym[i]))
            self.orbsymtot = IrrepOrb
        else:
            self.orbsymtot = []
        self.orbsym = []

        self.conv_tol = tol
        self.nroots = len(self.weight)
        self.restart = False
        # self.spin = None
        self.runtime = 0
        self.fixocfg_iter = -1

        self.inputfile = ""
        self.outputfile = ""
        self.energydat = ""
        self.converged = True  # 默认全部的根都收敛了

        ##################################################
        # DO NOT CHANGE these parameters, unless you know the code in details
        self.orbsym = []
        self._keys = set(self.__dict__.keys())  # For what ?

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** iCI flags ********')
        log.info('executable    = %s', self.executable)
        log.info('runtimedir    = %s', self.runtimedir)
        # log.debug1('config = %s', self.config)
        log.info('')
        return self

    def make_rdm1(self, state, norb, nelec, **kwargs):  # always return state averaged one
        dm_file = os.path.join(self.runtimedir, FILE_RDM1_NAME)
        if not os.path.isfile(dm_file):
            print("Fatal error in reading rdm1")
            exit(1)
        i, j, val = numpy.loadtxt(dm_file, dtype=numpy.dtype('i,i,d'),
                                  delimiter=',', skiprows=1, unpack=True)
        rdm1 = numpy.zeros((norb, norb))
        rdm1[i, j] = rdm1[j, i] = val
        nfzc = self.config["nfzc"]
        for j in range(nfzc):
            rdm1[j, j] = 2.0
        return rdm1

    def make_rdm12(self, state, norb, nelec, **kwargs):  # always return state averaged one
        rdm1 = self.make_rdm1(state, norb, nelec, kwargs=kwargs)
        dm_file = os.path.join(self.runtimedir, FILE_RDM2_NAME)
        nfzc = self.config["nfzc"]
        if not os.path.isfile(dm_file):
            print("Fatal error in reading rdm2")
            exit(1)
        i, j, k, l, val = numpy.loadtxt(dm_file, dtype=numpy.dtype('i,i,i,i,d'),
                                        delimiter=',', skiprows=1, unpack=True)
        rdm2 = numpy.zeros((norb, norb, norb, norb))
        rdm2[i, j, k, l] = rdm2[j, i, l, k] = val
        rdm2 = rdm2.transpose(0, 3, 1, 2) # p^+ q r^+ s

        # 补全 nfzc 
        for j in range(nfzc):
            rdm2[j,j,j,j]=2.0
        for j in range(nfzc):
            for i in range(nfzc):
                if i==j:
                    continue
                rdm2[i,i,j,j] = 4.0
                rdm2[i,j,j,i] = -2.0
        norb = rdm2.shape[0]
        for i in range(nfzc):
            for p in range(nfzc,norb):
                for q in range(nfzc,norb):
                    rdm2[i,i,p,q] = rdm1[p,q]*2.0
                    rdm2[p,q,i,i] = rdm1[p,q]*2.0
                    rdm2[p,q,i,i] = rdm1[p,q]*2.0
                    rdm2[i,p,q,i] = rdm1[p,q]*-1.0
                    rdm2[p,i,i,q] = rdm1[p,q]*-1.0

        return rdm1, rdm2

    def set_config(self, _key, _value):
        self.config[_key] = _value

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, restart=None,
               **kwargs):  # Driver

        # judge whether to restart
        if (self.runtime > 0):
            self.restart = True
        # print(self.restart, restart, self.runtime)
        if restart is None:
            restart = self.restart

        if restart == True and self.runtime == 0:
            print("Haven't run iCI fatal error since restart = true!")
            exit(1)

        if self.fixocfg_iter > 0:
            if self.runtime > self.fixocfg_iter:
                self.config["selection"] = 0

        print(self.runtime, self.fixocfg_iter)

        if restart == True:
            if 'approx' in kwargs and kwargs['approx'] is True:
                self.config["inputocfg"] = 3  # iCI is changed!
                # self.config["inputocfg"] = 2
            else:
                # self.config["inputocfg"] = 1
                if self.config["readinocfg"] != True:
                    self.config["inputocfg"] = 0  # iCI is changed
                else:
                    self.config["inputocfg"] = 2
        else:
            if self.config["readinocfg"] != True:
                self.config["inputocfg"] = 0  # iCI is changed
            else:
                self.config["inputocfg"] = 2

        if self.config["selection"] == 0:
            if self.config["readinocfg"] == True:
                self.config["inputocfg"] = 2
            else:
                self.config["inputocfg"] = 3

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']

        # wright integral files

        # generate nsegment and nvalelec

        nelectrons = 0

        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0] + nelec[1]

        if "nvalelec" not in self.config.keys():
            if (nelectrons <= 10):
                self.config["nvalelec"] = nelectrons
            else:
                if nelectrons % 2 == 0:
                    self.config["nvalelec"] = 10
                else:
                    self.config["nvalelec"] = 9

        nval = min(self.config["nvalelec"], norb)
        if norb <= 8:
            nval = norb            
        nval_hole = nval//2
        nval_part = nval - nval_hole
        ncore = (nelectrons - self.config["nvalelec"])//2
        nvir = norb - nval - ncore
        if nvir < 0:
            nval = norb - ncore
            nvir = 0
            nval_hole = nval//2
            nval_part = nval - nval_hole

        if self.config["segment"] == None:
            self.config["segment"] = "0 " + \
                str(ncore) + " "+str(nval_hole)+" " + \
                str(nval_part)+" "+str(nvir) + " 0"

        if self.orbsymtot is not []:
            nelectron_fzc = self.mol.nelectron - nelectrons
            nfzc = nelectron_fzc//2
            self.orbsym = self.orbsymtot[nfzc:nfzc+norb]
            # print(self.orbsym)

        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        if 'tol' in kwargs:
            self.config['tol'] = kwargs['tol']

        # execute iCI

        self = execute_iCI(self)

        #　read energy

        calc_e = read_energy(self)
        res_calc_e = 0.0
        for i in range(self.nroots):
            res_calc_e += calc_e[i]*self.weight[i]
            print("State %3d energy %20.12f" % (i, calc_e[i]))
        roots = list(range((self.nroots)))

        # return calc_e, roots
        return res_calc_e, 0

    def kernel_second_orbopt(self, h1e, eri, norb, nelec, ci0=None, ecore=0, restart=None,
                             **kwargs):  # Driver

        # judge whether to restart
        # if (self.runtime > 0):
        #     self.restart = True
        # print(self.restart, restart, self.runtime)
        # if restart is None:
        #     restart = self.restart

        if restart == True and self.runtime == 0:
            print("Haven't run iCI fatal error since restart = true!")
            exit(1)

        # if restart == True:
        #     if 'approx' in kwargs and kwargs['approx'] is True:
        #         self.config["inputocfg"] = 2
        #     else:
        #         self.config["inputocfg"] = 1
        # else:
        self.config["inputocfg"] = 0
        self.config["rotatemo"] = 2

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']

        # wright integral files

        # generate nsegment and nvalelec

        nelectrons = 0

        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0] + nelec[1]

        if "nvalelec" not in self.config.keys():
            if (nelectrons <= 10):
                self.config["nvalelec"] = nelectrons
            else:
                if nelectrons % 2 == 0:
                    self.config["nvalelec"] = 10
                else:
                    self.config["nvalelec"] = 9

        nval = min(self.config["nvalelec"], norb)
        nval_hole = nval//2
        nval_part = nval - nval_hole
        ncore = (nelectrons - self.config["nvalelec"])//2
        nvir = norb - nval - ncore
        nelectron_fzc = self.mol.nelectron - nelectrons
        nfzc = nelectron_fzc//2

        if self.config["segment"] == None:
            self.config["segment"] = str(nfzc) + " " + \
                str(ncore) + " "+str(nval_hole)+" " + \
                str(nval_part)+" "+str(nvir) + " 0"

        if self.orbsymtot is not []:
            #     nelectron_fzc = self.mol.nelectron - nelectrons
            #     nfzc = nelectron_fzc//2
            # self.orbsym = self.orbsymtot[nfzc:nfzc+norb]
            self.orbsym = self.orbsymtot[0:nfzc+norb]
            # print(self.orbsym)

        writeIntegralFile(self, h1e, eri, nfzc+norb, self.mol.nelectron, ecore)
        if 'tol' in kwargs:
            self.config['tol'] = kwargs['tol']

        # execute iCI

        self = execute_iCI(self)

        #　read energy

        calc_e = read_energy(self)
        res_calc_e = 0.0
        for i in range(self.nroots):
            res_calc_e += calc_e[i]*self.weight[i]
            print("State %3d energy %20.12f" % (i, calc_e[i]))
        roots = list(range((self.nroots)))

        # return calc_e, roots
        return res_calc_e, 0

    def approx_kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0,
                      restart=None, **kwargs):  # 近似求解 iCI
        # self.config["etol"] = self.conv_tol * 1e3
        # the same as kernel
        return self.kernel(h1e, eri, norb, nelec, ci0, ecore, restart, kwargs=kwargs, approx=True)

    def spin_square(self, civec, norb, nelec):
        state_id = civec
        spintwo = self.spin[state_id]
        s = float(spintwo)/2
        ss = s*s
        return ss, s*2+1

    def contract_2e(self, eri, civec, norb, nelec, client=None, **kwargs):  # Calculate Hc
        return None


def iCISCF(mf, norb, nelec, tol=1.e-8, cmin=1e-4, state=[[0, 0, 1]], *args, **kwargs):
    '''Shortcut function to setup CASSCF using the iCI solver.  The iCI
    solver is properly initialized in this function so that the 1-step
    algorithm can be applied with iCI-CASSCF.

    Examples:

    '''
    mc = mcscf.CASSCF(mf, norb, nelec, *args, **kwargs)
    mc.fcisolver = iCI(mol=mf.mol, tol=tol,
                       mo_coeff=mf.mo_coeff, state=state, cmin=cmin)
    # mc.fcisolver.config['get_1rdm_csv'] = True
    # mc.fcisolver.config['get_2rdm_csv'] = True
    # mc.fcisolver.config['var_only'] = True
    # mc.fcisolver.config['s2'] = True
    return mc


if __name__ == "__main__":

    from pyscf import gto, scf

    # Initialize C2 molecule
    b = 1.24253
    mol = gto.Mole()
    mol.build(
        verbose=4,
        output=None,
        atom=[
            ['C', (0.000000,  0.000000, -b/2)],
            ['C', (0.000000,  0.000000,  b/2)], ],
        basis={'C': 'ccpvdz', },
        symmetry='d2h',
    )

    # Create HF molecule
    mf = scf.RHF(mol).run()
    # print(mf.mo_coeff)

    # Number of orbital and electrons
    norb = 8
    nelec = 8
    dimer_atom = 'C'

    mciCI = mcscf.CASCI(mf, norb, nelec)
    mciCI.fcisolver = iCI(mol=mol, cmin=0.0, state=[
                          [0, 0, 1]], tol=1e-12, mo_coeff=mf.mo_coeff)
    mciCI.kernel()
    dm1, dm2 = mciCI.fcisolver.make_rdm12(0, norb, nelec)

    mc1 = mcscf.CASCI(mf, norb, nelec)
    mc1.kernel(mciCI.mo_coeff)
    dm1ref, dm2ref = mc1.fcisolver.make_rdm12(mc1.ci, norb, nelec)
    print(abs(dm1ref-dm1).max())
    print(abs(dm2ref-dm2).max())
    # print(dm1ref)
    # print(dm2ref)
    # exit()

    # mch = shci.SHCISCF(mf, norb, nelec)
    # mch.internal_rotation = True
    # mch.kernel()

    mc2step = mcscf.CASSCF(mf, norb, nelec)
    solver1 = fci.direct_spin1_symm.FCI(mol)
    solver1.wfnsym = 'ag'
    solver1.nroots = 3
    solver1.spin = 0
    # mc.fcisolver = solver1
    # mc2step.fcisolver =
    mc2step = mcscf.state_average_mix_(mc2step, [solver1], (0.5, 0.25, 0.25))
    # mc2step.mc2step()
    mc2step.mc1step()
    # mc2step.kernel()

    mymc2step = mcscf.CASSCF(mf, norb, nelec)
    mymc2step.fcisolver = iCI(mol=mol, cmin=0.0, state=[
        [0, 0, 3, [2, 1, 1]]], tol=1e-12, mo_coeff=mf.mo_coeff)
    # mymc2step.mc2step()
    mymc2step.mc1step()

    norb = 10
    nelec = 12

    mymc2step = mcscf.CASSCF(mf, norb, nelec)
    mymc2step.fcisolver = iCI(mol=mol, cmin=0.0, state=[
        [0, 0, 3, [2, 1, 1]]], tol=1e-12, mo_coeff=mf.mo_coeff)
    mymc2step.fcisolver.config["segment"] = "2 0 4 4 0 0"
    mymc2step.fcisolver.config["selection"] = 1
    mymc2step.fcisolver.config["nvalelec"] = 8
    mymc2step.fcisolver.config["nfzc"] = 2
    # mymc2step.mc2step()
    mymc2step.mc1step()
