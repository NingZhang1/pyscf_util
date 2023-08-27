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
import iCISCF
# import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats


def ReadIn_RDM1(_TaskName, _nao, _nstate, _skiprows=1):
    filename = _TaskName + ".csv"
    state, i, j, val = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,d'),
                                     delimiter=',', skiprows=1, unpack=True)
    rdm1 = numpy.zeros((_nstate, _nao, _nao))
    rdm1[state, i, j] = rdm1[state, j, i] = val
    return rdm1


def ReadIn_RDM2(_TaskName, _nao, _nstate, _skiprows=1):
    filename = _TaskName + ".csv"
    state, i, j, k, l, val = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,i,i,d'),
                                           delimiter=',', skiprows=1, unpack=True)
    rdm2 = numpy.zeros((_nstate, _nao, _nao, _nao, _nao))
    rdm2[state, i, j, k, l] = rdm2[state, j, i, l, k] = val
    rdm2 = rdm2.transpose(0, 1, 4, 2, 3)
    return rdm2


def Local_Spin(rdm1, rdm2, nao_group, _print = False):
    natm = len(nao_group)
    nstate = rdm1.shape[0]

    local_spin = numpy.zeros((nstate, natm))

    for istate in range(nstate):
    
        for iatm in range(natm):

            for orb_id in nao_group[iatm]:
                local_spin[istate][iatm] += 0.75 * (rdm1[istate][orb_id][orb_id] -  rdm2[istate][orb_id][orb_id][orb_id][orb_id])
                # print(istate, orb_id)
                print("%15.8f %15.8f" %(rdm1[istate][orb_id][orb_id], rdm2[istate][orb_id][orb_id][orb_id][orb_id]))

            for orb_id_i in nao_group[iatm]:
                for orb_id_j in nao_group[iatm]:
                    if orb_id_i == orb_id_j:
                        continue
                    local_spin[istate][iatm] -= 0.5 * (rdm2[istate][orb_id_i][orb_id_j][orb_id_j][orb_id_i] + 0.5 * rdm2[istate][orb_id_i][orb_id_i][orb_id_j][orb_id_j])
                    # print(istate, orb_id_i, orb_id_j)
                    print("%15.8f %15.8f" % (rdm2[istate][orb_id_i][orb_id_j][orb_id_j][orb_id_i], rdm2[istate][orb_id_i][orb_id_i][orb_id_j][orb_id_j]))

            print("----------------------------------------------------------")

    if _print:
        print("Local Spin^2")
        for istate in range(nstate):
            print("State %d" % istate)
            for iatm in range(natm):
                print("Atom %d: %20.12f" % (iatm, local_spin[istate][iatm]))

    return local_spin

def Spin_Spin_Correlation_Function(rdm1, rdm2, nao_group, _print = False):
    natm = len(nao_group)
    nstate = rdm1.shape[0]

    local_spin = Local_Spin(rdm1, rdm2, nao_group, False)

    Spin_Spin_Corr_Func = numpy.zeros((nstate, natm, natm))

    for istate in range(nstate):
        for iatm in range(natm):
            for jatm in range(natm):
                if iatm == jatm:
                    Spin_Spin_Corr_Func[istate][iatm][iatm] = local_spin[istate][iatm]
                else:
                    for orb_id_i in nao_group[iatm]:
                        for orb_id_j in nao_group[jatm]:
                            Spin_Spin_Corr_Func[istate][iatm][jatm] -= 0.5 * (rdm2[istate][orb_id_i][orb_id_j][orb_id_j][orb_id_i] + 0.5 * rdm2[istate][orb_id_i][orb_id_i][orb_id_j][orb_id_j])

    if _print:
        print("Spin Spin Correlation")
        for istate in range(nstate):
            print("State %d" % istate)
            for iatm in range(natm):
                for jatm in range(natm):
                    print("%20.12f " % (Spin_Spin_Corr_Func[istate][iatm][jatm]), end="")
                print("")

    return Spin_Spin_Corr_Func
