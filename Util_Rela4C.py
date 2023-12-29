from pyscf import gto, scf, lib
import numpy
from functools import reduce
from pyscf import lib
from pyscf import tools
import pyscf

import re

from pyscf import __config__

IOBLK_SIZE = getattr(__config__, 'ao2mo_outcore_ioblk_size', 256)  # 256 MB
IOBUF_WORDS = getattr(__config__, 'ao2mo_outcore_iobuf_words', 1e8)  # 1.6 GB
IOBUF_ROW_MIN = getattr(__config__, 'ao2mo_outcore_row_min', 160)
MAX_MEMORY = getattr(__config__, 'ao2mo_outcore_max_memory', 4000)  # 4GB

"""

Util_Rela4C.py is a module for generating integral files for Relativistic 4-component calculation.

Contents
--------

::

    FCIDUMP_Original_Ints
    FCIDUMP_Rela4C

Utils
-----

::

    _apply_time_reversal_op
    _time_reversal_symmetry_adapted
    _atom_spinor_spatial_parity
    _atom_Jz_adapted

"""


def FCIDUMP_Original_Ints(mol, my_RDHF):
    """ Print out the original integrals in FCIDUMP format
    Args:
        mol: a molecule object
        my_RDHF: a pyscf Restricted Dirac HF object

    Kwargs:

    Returns:

    """

    hcore = my_RDHF.get_hcore()

    n2c = mol.nao_2c()
    n4c = 2 * n2c

    int2e_res = numpy.zeros((n4c, n4c, n4c, n4c), dtype=numpy.complex128)
    c1 = .5 / lib.param.LIGHT_SPEED
    int2e_res[:n2c, :n2c, :n2c, :n2c] = mol.intor("int2e_spinor")  # LL LL
    tmp = mol.intor("int2e_spsp1_spinor") * c1**2
    int2e_res[n2c:, n2c:, :n2c, :n2c] = tmp  # SS LL
    int2e_res[:n2c, :n2c, n2c:, n2c:] = tmp.transpose(2, 3, 0, 1)  # LL SS
    int2e_res[n2c:, n2c:, n2c:, n2c:] = mol.intor(
        "int2e_spsp1spsp2_spinor") * c1**4  # SS SS

    int2e_breit = numpy.zeros(
        (n4c, n4c, n4c, n4c), dtype=numpy.complex128)

    ##### (LS|LS) and (SL|SL) #####

    tmp = mol.intor("int2e_breit_ssp1ssp2_spinor") * c1**2
    int2e_breit[:n2c, n2c:, :n2c, n2c:] = tmp
    tmp = mol.intor("int2e_breit_sps1sps2_spinor") * c1**2
    int2e_breit[n2c:, :n2c, n2c:, :n2c] = tmp

    ##### (LS|SL) and (SL|LS) #####

    tmp2 = mol.intor("int2e_breit_ssp1sps2_spinor") * c1**2
    int2e_breit[:n2c, n2c:, n2c:, :n2c] = tmp2  # (LS|SL)
    tmp2 = mol.intor("int2e_breit_sps1ssp2_spinor") * c1**2
    int2e_breit[n2c:, :n2c, :n2c, n2c:] = tmp2  # (SL|LS)

    ###############################

    print("Coulomb term")
    tol = 1e-10
    for i in range(n4c):
        for j in range(n4c):
            for k in range(n4c):
                for l in range(n4c):
                    if abs(int2e_res[i][j][k][l]) > tol:
                        print("%18.12E %18.12E %4d %4d %4d %4d" % (
                            int2e_res[i][j][k][l].real, int2e_res[i][j][k][l].imag, i+1, j+1, k+1, l+1))
    print("Breit term")
    for i in range(n4c):
        for j in range(n4c):
            for k in range(n4c):
                for l in range(n4c):
                    if abs(int2e_breit[i][j][k][l]) > tol:
                        print("%18.12E %18.12E %4d %4d %4d %4d" % (
                            int2e_breit[i][j][k][l].real, int2e_breit[i][j][k][l].imag, i+1, j+1, k+1, l+1))

    print("Hcore term")
    for i in range(n4c):
        for j in range(n4c):
            if abs(hcore[i][j]) > tol:
                print("%18.12E %18.12E %4d %4d %4d %4d" % (
                    hcore[i][j].real, hcore[i][j].imag, i+1, j+1, 0, 0))

### filename ###


coulomb_LLLL = "%s_coulomb_LLLL.h5"
coulomb_SSSS = "%s_coulomb_SSSS.h5"
coulomb_LLSS = "%s_coulomb_LLSS.h5"
coulomb_SSLL = "%s_coulomb_SSLL.h5"

breit_LSLS = "%s_breit_LSLS.h5"
breit_SLSL = "%s_breit_SLSL.h5"
breit_LSSL = "%s_breit_LSSL.h5"
breit_SLLS = "%s_breit_SLLS.h5"


def _r_outcore_Coulomb(mol, my_RDHF, prefix, max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE):

    from pyscf.ao2mo import r_outcore

    n2c = mol.nao_2c()
    mo_coeff = my_RDHF.mo_coeff
    mo_coeff_mat = numpy.matrix(mo_coeff)

    mo_coeff_pes = mo_coeff_mat[:, n2c:]
    mo_coeff_L = mo_coeff_pes[:n2c, :]
    mo_coeff_S = mo_coeff_pes[n2c:, :]

    r_outcore.general(mol, (mo_coeff_L, mo_coeff_L, mo_coeff_L, mo_coeff_L), coulomb_LLLL %
                      prefix, intor="int2e_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (mo_coeff_S, mo_coeff_S, mo_coeff_S, mo_coeff_S), coulomb_SSSS %
                      prefix, intor="int2e_spsp1spsp2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (mo_coeff_L, mo_coeff_L, mo_coeff_S, mo_coeff_S), coulomb_LLSS %
                      prefix, intor="int2e_spsp2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (mo_coeff_S, mo_coeff_S, mo_coeff_L, mo_coeff_L), coulomb_SSLL %
                      prefix, intor="int2e_spsp1_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')


def _r_outcore_Breit(mol, my_RDHF, prefix, max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE):

    from pyscf.ao2mo import r_outcore

    n2c = mol.nao_2c()
    mo_coeff = my_RDHF.mo_coeff
    mo_coeff_mat = numpy.matrix(mo_coeff)

    mo_coeff_pes = mo_coeff_mat[:, n2c:]
    mo_coeff_L = mo_coeff_pes[:n2c, :]
    mo_coeff_S = mo_coeff_pes[n2c:, :]

    r_outcore.general(mol, (mo_coeff_L, mo_coeff_S, mo_coeff_L, mo_coeff_S), breit_LSLS %
                      prefix, intor="int2e_breit_ssp1ssp2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (mo_coeff_S, mo_coeff_L, mo_coeff_S, mo_coeff_L), breit_SLSL %
                      prefix, intor="int2e_breit_sps1sps2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (mo_coeff_L, mo_coeff_S, mo_coeff_S, mo_coeff_L), breit_LSSL %
                      prefix, intor="int2e_breit_ssp1sps2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (mo_coeff_S, mo_coeff_L, mo_coeff_L, mo_coeff_S), breit_SLLS %
                      prefix, intor="int2e_breit_sps1ssp2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')


def _dump_2e_outcore(fout, n2c, prefix, with_breit, IsComplex, symmetry="s1", tol=1e-8):

    import h5py

    feri_coulomb_LLLL = h5py.File(coulomb_LLLL % prefix, 'r')
    feri_coulomb_LLSS = h5py.File(coulomb_LLSS % prefix, 'r')
    feri_coulomb_SSLL = h5py.File(coulomb_SSLL % prefix, 'r')
    feri_coulomb_SSSS = h5py.File(coulomb_SSSS % prefix, 'r')

    if with_breit:
        feri_breit_LSLS = h5py.File(breit_LSLS % prefix, 'r')
        feri_breit_SLSL = h5py.File(breit_SLSL % prefix, 'r')
        feri_breit_LSSL = h5py.File(breit_LSSL % prefix, 'r')
        feri_breit_SLLS = h5py.File(breit_SLLS % prefix, 'r')
    else:
        feri_breit_LSLS = None
        feri_breit_SLSL = None
        feri_breit_LSSL = None
        feri_breit_SLLS = None

    c1 = .5 / lib.param.LIGHT_SPEED

    if symmetry == "s1":
        if IsComplex:
            for i in range(n2c):
                for j in range(n2c):

                    ij = i * n2c + j

                    eri_coulomb = numpy.array(
                        feri_coulomb_LLLL['eri_mo'][ij]).reshape(n2c, n2c)
                    eri_coulomb += numpy.array(
                        feri_coulomb_LLSS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                    eri_coulomb += numpy.array(
                        feri_coulomb_SSLL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                    eri_coulomb += numpy.array(
                        feri_coulomb_SSSS['eri_mo'][ij]).reshape(n2c, n2c) * c1**4

                    if with_breit:
                        eri_breit = numpy.array(
                            feri_breit_LSLS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_SLSL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_LSSL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_SLLS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2

                    for k in range(n2c):
                        for l in range(n2c):
                            if abs(eri_coulomb[k][l]) > tol:
                                fout.write("%18.12E %18.12E %4d %4d %4d %4d\n" % (
                                    eri_coulomb[k][l].real, eri_coulomb[k][l].imag, i+1, j+1, k+1, l+1))
                            if with_breit:
                                if abs(eri_breit[k][l]) > tol:
                                    fout.write("%18.12E %18.12E %4d %4d %4d %4d\n" % (
                                        eri_breit[k][l].real, eri_breit[k][l].imag, n2c+i+1, n2c+j+1, n2c+k+1, n2c+l+1))
        else:
            for i in range(n2c):
                for j in range(n2c):

                    ij = i * n2c + j

                    eri_coulomb = numpy.array(
                        feri_coulomb_LLLL['eri_mo'][ij]).reshape(n2c, n2c)
                    eri_coulomb += numpy.array(
                        feri_coulomb_LLSS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                    eri_coulomb += numpy.array(
                        feri_coulomb_SSLL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                    eri_coulomb += numpy.array(
                        feri_coulomb_SSSS['eri_mo'][ij]).reshape(n2c, n2c) * c1**4

                    if with_breit:
                        eri_breit = numpy.array(
                            feri_breit_LSLS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_SLSL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_LSSL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_SLLS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2

                    for k in range(n2c):
                        for l in range(n2c):
                            if abs(eri_coulomb[k][l]) > tol:
                                fout.write("%18.12E %4d %4d %4d %4d\n" % (
                                    eri_coulomb[k][l].real, i+1, j+1, k+1, l+1))
                            if with_breit:
                                if abs(eri_breit[k][l]) > tol:
                                    fout.write("%18.12E %4d %4d %4d %4d\n" % (
                                        eri_breit[k][l].real, n2c+i+1, n2c+j+1, n2c+k+1, n2c+l+1))

    elif symmetry == "s4":

        if IsComplex:
            for i in range(n2c):
                for j in range(i+1):

                    ij = i * n2c + j

                    eri_coulomb = numpy.array(
                        feri_coulomb_LLLL['eri_mo'][ij]).reshape(n2c, n2c)
                    eri_coulomb += numpy.array(
                        feri_coulomb_LLSS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                    eri_coulomb += numpy.array(
                        feri_coulomb_SSLL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                    eri_coulomb += numpy.array(
                        feri_coulomb_SSSS['eri_mo'][ij]).reshape(n2c, n2c) * c1**4

                    if with_breit:
                        eri_breit = numpy.array(
                            feri_breit_LSLS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_SLSL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_LSSL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_SLLS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2

                    for k in range(i+1):
                        for l in range(n2c):
                            if abs(eri_coulomb[k][l]) > tol:
                                fout.write("%18.12E %18.12E %4d %4d %4d %4d\n" % (
                                    eri_coulomb[k][l].real, eri_coulomb[k][l].imag, i+1, j+1, k+1, l+1))
                            if with_breit:
                                if abs(eri_breit[k][l]) > tol:
                                    fout.write("%18.12E %18.12E %4d %4d %4d %4d\n" % (
                                        eri_breit[k][l].real, eri_breit[k][l].imag, n2c+i+1, n2c+j+1, n2c+k+1, n2c+l+1))
        else:
            for i in range(n2c):
                for j in range(i+1):

                    ij = i * n2c + j

                    eri_coulomb = numpy.array(
                        feri_coulomb_LLLL['eri_mo'][ij]).reshape(n2c, n2c)
                    eri_coulomb += numpy.array(
                        feri_coulomb_LLSS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                    eri_coulomb += numpy.array(
                        feri_coulomb_SSLL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                    eri_coulomb += numpy.array(
                        feri_coulomb_SSSS['eri_mo'][ij]).reshape(n2c, n2c) * c1**4

                    if with_breit:
                        eri_breit = numpy.array(
                            feri_breit_LSLS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_SLSL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_LSSL['eri_mo'][ij]).reshape(n2c, n2c) * c1**2
                        eri_breit += numpy.array(
                            feri_breit_SLLS['eri_mo'][ij]).reshape(n2c, n2c) * c1**2

                    for k in range(i+1):
                        for l in range(n2c):
                            if abs(eri_coulomb[k][l]) > tol:
                                fout.write("%18.12E %4d %4d %4d %4d\n" % (
                                    eri_coulomb[k][l].real, i+1, j+1, k+1, l+1))
                            if with_breit:
                                if abs(eri_breit[k][l]) > tol:
                                    fout.write("%18.12E %4d %4d %4d %4d\n" % (
                                        eri_breit[k][l].real, n2c+i+1, n2c+j+1, n2c+k+1, n2c+l+1))

    else:
        raise ValueError("Unknown symmetry %s" % symmetry)

    feri_coulomb_LLLL.close()
    feri_coulomb_LLSS.close()
    feri_coulomb_SSLL.close()
    feri_coulomb_SSSS.close()

    if with_breit:
        feri_breit_LSLS.close()
        feri_breit_SLSL.close()
        feri_breit_LSSL.close()
        feri_breit_SLLS.close()

def _dump_2e(fout, int2e_coulomb, int2e_breit, with_breit, IsComplex, symmetry="s1", tol=1e-8):
    """ Dump the 2-electron integrals in FCIDUMP format (**incore** mode)

    Args:
        fout: the file object to dump the integrals
        int2e_coulomb: the 2-electron Coulomb integrals
        int2e_breit: the 2-electron Breit integrals
        with_breit: whether to include Breit term
        IsComplex: whether the integrals are complex
        symmetry: the symmetry of the integrals

    Kwargs:

    Returns:

    """

    # tol = 1e-10
    n2c = int2e_coulomb.shape[0]

    if symmetry == "s1":
        if IsComplex:
            for i in range(n2c):
                for j in range(n2c):
                    for k in range(n2c):
                        for l in range(n2c):
                            if abs(int2e_coulomb[i][j][k][l]) > tol:
                                fout.write("%18.12E %18.12E %4d %4d %4d %4d\n" % (
                                    int2e_coulomb[i][j][k][l].real, int2e_coulomb[i][j][k][l].imag, i+1, j+1, k+1, l+1))
                            if with_breit:
                                if abs(int2e_breit[i][j][k][l]) > tol:
                                    fout.write("%18.12E %18.12E %4d %4d %4d %4d\n" % (
                                        int2e_breit[i][j][k][l].real, int2e_breit[i][j][k][l].imag, n2c+i+1, n2c+j+1, n2c+k+1, n2c+l+1))
        else:
            for i in range(n2c):
                for j in range(n2c):
                    for k in range(n2c):
                        for l in range(n2c):
                            if abs(int2e_coulomb[i][j][k][l]) > tol:
                                fout.write("%18.12E %4d %4d %4d %4d\n" % (
                                    int2e_coulomb[i][j][k][l].real, i+1, j+1, k+1, l+1))
                            if with_breit:
                                if abs(int2e_breit[i][j][k][l]) > tol:
                                    fout.write("%18.12E %4d %4d %4d %4d\n" % (
                                        int2e_breit[i][j][k][l].real, n2c+i+1, n2c+j+1, n2c+k+1, n2c+l+1))

    elif symmetry == "s4":

        if IsComplex:
            for i in range(n2c):
                for j in range(i+1):
                    for k in range(i+1):
                        for l in range(n2c):
                            if abs(int2e_coulomb[i][j][k][l]) > tol:
                                fout.write("%18.12E %18.12E %4d %4d %4d %4d\n" % (
                                    int2e_coulomb[i][j][k][l].real, int2e_coulomb[i][j][k][l].imag, i+1, j+1, k+1, l+1))
                            if with_breit:
                                if abs(int2e_breit[i][j][k][l]) > tol:
                                    fout.write("%18.12E %18.12E %4d %4d %4d %4d\n" % (
                                        int2e_breit[i][j][k][l].real, int2e_breit[i][j][k][l].imag, n2c+i+1, n2c+j+1, n2c+k+1, n2c+l+1))
        else:
            for i in range(n2c):
                for j in range(i+1):
                    for k in range(i+1):
                        for l in range(n2c):
                            if abs(int2e_coulomb[i][j][k][l]) > tol:
                                fout.write("%18.12E %4d %4d %4d %4d\n" % (
                                    int2e_coulomb[i][j][k][l].real, i+1, j+1, k+1, l+1))
                            if with_breit:
                                if abs(int2e_breit[i][j][k][l]) > tol:
                                    fout.write("%18.12E %4d %4d %4d %4d\n" % (
                                        int2e_breit[i][j][k][l].real, n2c+i+1, n2c+j+1, n2c+k+1, n2c+l+1))

    else:
        raise ValueError("Unknown symmetry %s" % symmetry)


def FCIDUMP_Rela4C(mol, my_RDHF, with_breit=False, filename="fcidump", mode="incore", orbsym_ID=None, IsComplex=True, tol=1e-8, debug=False):
    """ Dump the relativistic 4-component integrals in FCIDUMP format

    Args:
        mol: a molecule object
        my_RDHF: a pyscf Restricted Dirac HF object
        filename: the filename of the FCIDUMP file

    Kwargs:
        with_breit: whether to include Breit term
        mode: the mode to dump the integrals
        debug: whether to return the integrals

    Returns:

    """

    assert mode in ["original", "incore", "outcore"]

    PREFIX = "RELA_4C_%d" % (numpy.random.randint(1, 19951201+1))

    n2c = mol.nao_2c()
    mo_coeff = my_RDHF.mo_coeff
    mo_coeff_mat = numpy.matrix(mo_coeff)

    mo_coeff_pes = mo_coeff_mat[:, n2c:]

    hcore = my_RDHF.get_hcore()
    h1e = reduce(numpy.dot, (mo_coeff_pes.H, hcore, mo_coeff_pes))

    n4c = 2 * n2c

    if mode == "original":
        int2e_res = numpy.zeros((n4c, n4c, n4c, n4c), dtype=numpy.complex128)
        c1 = .5 / lib.param.LIGHT_SPEED
        int2e_res[:n2c, :n2c, :n2c, :n2c] = mol.intor("int2e_spinor")  # LL LL
        tmp = mol.intor("int2e_spsp1_spinor") * c1**2
        int2e_res[n2c:, n2c:, :n2c, :n2c] = tmp  # SS LL
        int2e_res[:n2c, :n2c, n2c:, n2c:] = tmp.transpose(2, 3, 0, 1)  # LL SS
        int2e_res[n2c:, n2c:, n2c:, n2c:] = mol.intor(
            "int2e_spsp1spsp2_spinor") * c1**4  # SS SS
        int2e_coulomb = numpy.einsum(
            "ijkl,ip->pjkl", int2e_res, mo_coeff_pes.conj())
        int2e_coulomb = numpy.einsum(
            "pjkl,jq->pqkl", int2e_coulomb, mo_coeff_pes)
        int2e_coulomb = numpy.einsum(
            "pqkl,kr->pqrl", int2e_coulomb, mo_coeff_pes.conj())
        int2e_coulomb = numpy.einsum(
            "pqrl,ls->pqrs", int2e_coulomb, mo_coeff_pes)
        if with_breit:
            int2e_breit = numpy.zeros(
                (n4c, n4c, n4c, n4c), dtype=numpy.complex128)
            ##### (LS|LS) and (SL|SL) #####
            tmp = mol.intor("int2e_breit_ssp1ssp2_spinor") * c1**2
            int2e_breit[:n2c, n2c:, :n2c, n2c:] = tmp
            tmp = mol.intor("int2e_breit_sps1sps2_spinor") * c1**2
            int2e_breit[n2c:, :n2c, n2c:, :n2c] = tmp
            ##### (LS|SL) and (SL|LS) #####
            tmp2 = mol.intor("int2e_breit_ssp1sps2_spinor") * c1**2
            int2e_breit[:n2c, n2c:, n2c:, :n2c] = tmp2  # (LS|SL)
            tmp2 = mol.intor("int2e_breit_sps1ssp2_spinor") * c1**2
            int2e_breit[n2c:, :n2c, :n2c, n2c:] = tmp2  # (SL|LS)
            ###############################
            int2e_breit = numpy.einsum(
                "ijkl,ip->pjkl", int2e_breit, mo_coeff_pes.conj())
            int2e_breit = numpy.einsum(
                "pjkl,jq->pqkl", int2e_breit, mo_coeff_pes)
            int2e_breit = numpy.einsum(
                "pqkl,kr->pqrl", int2e_breit, mo_coeff_pes.conj())
            int2e_breit = numpy.einsum(
                "pqrl,ls->pqrs", int2e_breit, mo_coeff_pes)
        else:
            int2e_breit = None
    elif mode == "incore":
        c1 = .5 / lib.param.LIGHT_SPEED
        ### LLLL part ###
        int2e_tmp = mol.intor("int2e_spinor")
        mo_coeff_L = mo_coeff_pes[:n2c, :]
        int2e_res = lib.einsum("pqrs,pi,qj,rk,sl->ijkl", int2e_tmp,
                               mo_coeff_L.conj(), mo_coeff_L, mo_coeff_L.conj(), mo_coeff_L)
        ### SSLL part ###
        int2e_tmp = mol.intor("int2e_spsp1_spinor") * c1**2
        mo_coeff_S = mo_coeff_pes[n2c:, :]
        int2e_tmp = lib.einsum("pqrs,pi,qj,rk,sl->ijkl", int2e_tmp,
                               mo_coeff_S.conj(), mo_coeff_S, mo_coeff_L.conj(), mo_coeff_L)
        int2e_res += int2e_tmp
        int2e_res += int2e_tmp.transpose(2, 3, 0, 1)
        ### SSSS part ###
        int2e_tmp = mol.intor("int2e_spsp1spsp2_spinor") * c1**4
        int2e_res += lib.einsum("pqrs,pi,qj,rk,sl->ijkl", int2e_tmp,
                                mo_coeff_S.conj(), mo_coeff_S, mo_coeff_S.conj(), mo_coeff_S)
        int2e_coulomb = int2e_res
        if with_breit:
            ### LSLS part ###
            int2e_tmp = mol.intor("int2e_breit_ssp1ssp2_spinor") * c1**2
            int2e_breit = lib.einsum("pqrs,pi,qj,rk,sl->ijkl", int2e_tmp,
                                     mo_coeff_L.conj(), mo_coeff_S, mo_coeff_L.conj(), mo_coeff_S)
            ### SLSL part ###
            int2e_tmp = int2e_tmp.conj().transpose(1, 0, 3, 2)
            int2e_breit += lib.einsum("pqrs,pi,qj,rk,sl->ijkl", int2e_tmp,
                                      mo_coeff_S.conj(), mo_coeff_L, mo_coeff_S.conj(), mo_coeff_L)
            ### LSSL part ###
            int2e_tmp = mol.intor("int2e_breit_ssp1sps2_spinor") * c1**2
            int2e_breit += lib.einsum("pqrs,pi,qj,rk,sl->ijkl", int2e_tmp,
                                      mo_coeff_L.conj(), mo_coeff_S, mo_coeff_S.conj(), mo_coeff_L)
            ### SLLS part ###
            int2e_tmp = int2e_tmp.transpose(2, 3, 0, 1)
            int2e_breit += lib.einsum("pqrs,pi,qj,rk,sl->ijkl", int2e_tmp,
                                      mo_coeff_S.conj(), mo_coeff_L, mo_coeff_L.conj(), mo_coeff_S)
        else:
            int2e_breit = None
    elif mode == "outcore":

        # pyscf.ao2mo.r_outcore.general
        # raise NotImplementedError("outcore mode is not implemented yet")

        _r_outcore_Coulomb(mol, my_RDHF, PREFIX)
        if with_breit:
            _r_outcore_Breit(mol, my_RDHF, PREFIX)

        int2e_coulomb = None
        int2e_breit = None

    else:
        raise ValueError("Unknown mode %s" % mode)

    energy_core = mol.get_enuc()

    nmo = n2c // 2
    nelec = mol.nelectron
    ms = 0
    tol = 1e-8
    nuc = energy_core
    float_format = " %18.12E"

    if orbsym_ID is None:
        orbsym_ID = []
        for _ in range(nmo):
            orbsym_ID.append(0)
    else:
        orbsym_ID = orbsym_ID[nmo:]

    with open(filename, 'w') as fout:  # 4-fold symmetry
        tools.fcidump.write_head(fout, nmo, nelec, ms, orbsym_ID)

        # output_format = float_format + float_format + ' %4d %4d %4d %4d\n'
        # if int2e_coulomb.ndim == 4:

        if mode != "outcore":
            if debug:
                _dump_2e(fout, int2e_coulomb, int2e_breit, with_breit,
                         IsComplex, symmetry="s1", tol=tol)
            else:
                _dump_2e(fout, int2e_coulomb, int2e_breit, with_breit,
                         IsComplex, symmetry="s4", tol=tol)
        else:
            if debug:
                _dump_2e_outcore(fout, n2c, PREFIX, with_breit,
                                 IsComplex, symmetry="s1", tol=tol)
            else:
                _dump_2e_outcore(fout, n2c, PREFIX, with_breit,
                                 IsComplex, symmetry="s4", tol=tol)

        # elif int2e_coulomb.ndim == 2:
        #     raise NotImplementedError("2-fold symmetry is not implemented yet")
        #     npair = n2c * (n2c + 1) // 2
        #     assert (int2e_coulomb.size == npair * npair)
        #     ij = 0
        #     for i in range(n2c):
        #         for j in range(0, i+1):
        #             kl = 0
        #             for k in range(0, n2c):
        #                 for l in range(0, k+1):
        #                     if abs(int2e_coulomb[ij, kl]) > tol:
        #                         fout.write(output_format %
        #                                    (int2e_coulomb[ij, kl].real, int2e_coulomb[ij, kl].imag, i+1, j+1, k+1, l+1))
        #                     kl += 1
        #             ij += 1
        # else:
        #     raise ValueError("Unknown int2e_coulomb.ndim %d" %
        #                      int2e_coulomb.ndim)

        ############################################ DUMP E1 #############################################

        if IsComplex:
            output_format = float_format + float_format + ' %4d %4d  0  0\n'
            for i in range(n2c):
                # for j in range(n2c):
                for j in range(i+1):
                    if abs(h1e[i, j]) > tol:
                        fout.write(output_format %
                                   (h1e[i, j].real, h1e[i, j].imag, i+1, j+1))
            output_format = float_format + ' 0.0  0  0  0  0\n'
            fout.write(output_format % nuc)
        else:
            output_format = float_format + ' %4d %4d  0  0\n'
            for i in range(n2c):
                # for j in range(n2c):
                for j in range(i+1):
                    if abs(h1e[i, j]) > tol:
                        fout.write(output_format % (h1e[i, j].real, i+1, j+1))
            output_format = float_format + ' 0  0  0  0\n'
            fout.write(output_format % nuc)

    ########### clean 

    if mode == "outcore":

        import os

        os.remove(coulomb_LLLL % PREFIX)
        os.remove(coulomb_LLSS % PREFIX)
        os.remove(coulomb_SSLL % PREFIX)
        os.remove(coulomb_SSSS % PREFIX)

        if with_breit:
            os.remove(breit_LSLS % PREFIX)
            os.remove(breit_SLSL % PREFIX)
            os.remove(breit_LSSL % PREFIX)
            os.remove(breit_SLLS % PREFIX)

    if debug:
        return int2e_coulomb, int2e_breit
    else:
        return None, None


def _apply_time_reversal_op(mol, mo_coeff, debug=False):
    """ Calculate the time reversal operator in the basis of the mo_coeff

    Args:
        mol: a molecule object
        mo_coeff: the molecular orbital coefficients

    Kwargs:

    Returns:
        tr_act_packed: a list of [index, coefficient] for the time reversal operator

    """

    trmaps = mol.time_reversal_map()
    idxA = numpy.where(trmaps > 0)[0]
    idxB = trmaps[idxA] - 1
    n = trmaps.size
    idx2 = numpy.hstack((idxA, idxA+n, idxB, idxB+n))

    if debug:
        print("trmaps = ", trmaps)
        print("idxA   = ", idxA)
        print("idxB   = ", idxB)

    time_reversal_m = numpy.zeros((2*n, 2*n), dtype=numpy.int64)

    for irow, data in enumerate(trmaps):
        icol = data
        elmt = 1
        if data < 0:
            icol = -data - 1
            elmt = -1
        else:
            icol = data - 1
            elmt = 1
        time_reversal_m[irow, icol] = elmt

    time_reversal_m[n:, n:] = time_reversal_m[:n, :n]

    if debug:
        print("time_reversal_m = ", time_reversal_m)

    ovlp_4C = pyscf.scf.dhf.get_ovlp(mol)

    tr_act = reduce(numpy.dot, (mo_coeff.T.conj(), ovlp_4C,
                    time_reversal_m, mo_coeff.conj()))
    tr_act_packed = []
    for i in range(tr_act.shape[0]):
        for j in range(tr_act.shape[1]):
            if abs(tr_act[i, j]) > 1e-6:
                if debug:
                    print("tr_act = ", i, j, tr_act[i, j])
                tr_act_packed.append([j, tr_act[i, j]])

    return tr_act_packed


def _time_reversal_symmetry_adapted(mol, mo_coeff,  debug=False):
    """ Adapt the molecular orbital coefficients to the time reversal symmetry

    Args:
        mol: a molecule object
        mo_coeff: the molecular orbital coefficients

    Kwargs:
        debug: whether to print the details 

    Returns:

    """

    trmaps = mol.time_reversal_map()
    idxA = numpy.where(trmaps > 0)[0]
    idxB = trmaps[idxA] - 1
    n = trmaps.size
    idx2 = numpy.hstack((idxA, idxA+n, idxB, idxB+n))

    if debug:
        print("trmaps = ", trmaps)
        print("idxA   = ", idxA)
        print("idxB   = ", idxB)

    time_reversal_m = numpy.zeros((2*n, 2*n), dtype=numpy.int64)

    for irow, data in enumerate(trmaps):
        icol = data
        elmt = 1
        if data < 0:
            icol = -data - 1
            elmt = -1
        else:
            icol = data - 1
            elmt = 1
        time_reversal_m[irow, icol] = elmt

    time_reversal_m[n:, n:] = time_reversal_m[:n, :n]

    if debug:
        print("time_reversal_m = ", time_reversal_m)

    ovlp_4C = pyscf.scf.dhf.get_ovlp(mol)

    ######### the first step is to rotate the orb so that the TR is really to be addapted! #########

    Res = mo_coeff.copy()

    idxA_all = numpy.hstack((idxA, idxA+n))
    ovlp_A = ovlp_4C[idxA_all, :][:, idxA_all]

    for i in range(0, 2*n, 2):
        mo_coeff_A = Res[idxA_all, i]
        norm_A = reduce(numpy.dot, (mo_coeff_A.T.conj(), ovlp_A, mo_coeff_A))
        if norm_A < 0.5:
            if debug:
                # print("norm_A = ", norm_A)
                # swap i and i+1
                print("swap %d and %d" % (i, i+1))
            # Res[:, i] = -Res[:, i]
            tmp = Res[:, i].copy()
            Res[:, i] = Res[:, i+1]
            Res[:, i+1] = tmp
            # if debug:
            #     print(Res[idx2, i:i+2])

        ### real orbital ###

        mo_coeff_A = Res[:, i]
        real_A = mo_coeff_A.real
        norm_real = numpy.linalg.norm(real_A)
        imag_A = mo_coeff_A.imag
        norm_imag = numpy.linalg.norm(imag_A)
        if norm_imag > norm_real:
            print("times i at ", i)
            print("norm_imag = ", norm_imag)
            print("norm_real = ", norm_real)
            Res[:, i] = -1.0j * Res[:, i]

        mo_coeff_B = Res[:, i+1]
        real_B = mo_coeff_B.real
        norm_real = numpy.linalg.norm(real_B)
        imag_B = mo_coeff_B.imag
        norm_imag = numpy.linalg.norm(imag_B)
        if norm_imag > norm_real:
            print("times i at ", i+1)
            print("norm_imag = ", norm_imag)
            print("norm_real = ", norm_real)
            Res[:, i+1] = -1.0j * Res[:, i+1]

    ######### the second step is to reorder the orb so that the TR seems to be addapted! #########

    tr_act = reduce(numpy.dot, (Res.T.conj(), ovlp_4C,
                    time_reversal_m, Res.conj()))
    tr_act_packed = []
    for i in range(tr_act.shape[0]):
        for j in range(tr_act.shape[1]):
            if abs(tr_act[i, j]) > 1e-6:
                if debug:
                    print("tr_act = ", i, j, tr_act[i, j])
                tr_act_packed.append([j, tr_act[i, j]])

    for i in range(0, 2*n, 2):
        if tr_act_packed[i][0] != i+1:
            print("Error in time reversal symmetry")
            exit(1)
        if tr_act_packed[i][1] < 0.0:
            if debug:
                print("plus -1 between %d and %d" % (i, i+1))
            Res[:, i+1] *= -1.0

    return Res


if __name__ == "__main__":
    # mol = gto.M(atom='H 0 0 0; H 0 0 1; O 0 1 0', basis='sto-3g', verbose=5)
    mol = gto.M(atom='F 0 0 0', basis='cc-pvdz', verbose=5,
                charge=-1, spin=0, symmetry="d2h")
    mol.build()
    mf = scf.dhf.RDHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    mf.with_breit = True
    mf.kernel()

    fock = mf.get_fock()

    print("mo_ene = ", mf.mo_energy)

    ovlp_4C = pyscf.scf.dhf.get_ovlp(mol)

    e, mo_coeff = mf._eigh(fock, ovlp_4C)

    print("mo_ene = ", e)

    print(numpy.allclose(mf.mo_energy, e))
    print(numpy.allclose(mf.mo_coeff, mo_coeff))

    _apply_time_reversal_op(mol, mo_coeff, True)

    mf.mo_coeff = mo_coeff

    print(mol.symm_orb)
    print(mol.irrep_id)

    sym_orb = numpy.concatenate([mol.symm_orb[0], mol.symm_orb[1]], axis=1)
    for i in range(2, len(mol.symm_orb)):
        sym_orb = numpy.concatenate([sym_orb, mol.symm_orb[i]], axis=1)

    from pyscf import symm
    orbsym_ID = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, sym_orb)

    print(orbsym_ID)

    # exit(1)

    int2e1, breit_1 = FCIDUMP_Rela4C(
        mol, mf, with_breit=True, filename="FCIDUMP_4C_Breit", mode="original", debug=True)

    int2e2, breit_2 = FCIDUMP_Rela4C(
        mol, mf, with_breit=True, filename="FCIDUMP_4C_incore", mode="incore", debug=True)

    for i in range(mol.nao_2c(), mol.nao_2c()*2):
        for j in range(mf.mo_coeff.shape[0]):
            if abs(mf.mo_coeff[j, i]) > 1e-6:
                print("%4d %4d %15.8f %15.8f" % (i-mol.nao_2c()+1, j,
                      mf.mo_coeff[j, i].real, mf.mo_coeff[j, i].imag))

    # FCIDUMP_Rela4C(mol, mf, filename="FCIDUMP_4C_incore2", mode="incore")

    print("diff = ", numpy.linalg.norm(int2e1 - int2e2))

    if breit_1 is not None:
        print("breit diff = ", numpy.linalg.norm(breit_1 - breit_2))

    #### check the time-reversal symmetry of the orbitals ####

    nao = mol.nao

    #### check breit term 4-fold symmetry ####

    for i in range(nao*2):
        for j in range(nao*2):
            for k in range(nao*2):
                for l in range(nao*2):
                    # print(breit_1[i,j,k,l], breit_1[j,i,l,k])
                    t1 = abs(breit_1[i, j, k, l] - breit_1[j, i, l, k].conj())
                    t2 = abs(breit_1[i, j, k, l] - breit_1[k, l, i, j].conj())
                    if t1 > 1e-8:
                        print("Breit 4-fold symmetry is not satisfied")
                        print(breit_1[i, j, k, l], breit_1[j, i, l, k])

    for i in range(nao):
        for j in range(nao):
            for k in range(nao):
                for l in range(nao):

                    t1 = abs(int2e1[2*i, 2*j, 2*k, 2*l] -
                             int2e1[2*i, 2*j, 2*l+1, 2*k+1])
                    t2 = abs(int2e1[2*j+1, 2*i+1, 2*k, 2*l] -
                             int2e1[2*j+1, 2*i+1, 2*l+1, 2*k+1])
                    t3 = abs(int2e1[2*i, 2*j, 2*k, 2*l] -
                             int2e1[2*j+1, 2*i+1, 2*k, 2*l])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Coulomb AAAA group is not time-reversal symmetric")
                        print(int2e1[2*i, 2*j, 2*k, 2*l], int2e1[2*i, 2*j, 2*l+1, 2*k+1],
                              int2e1[2*j+1, 2*i+1, 2*k, 2*l], int2e1[2*j+1, 2*i+1, 2*l+1, 2*k+1])

                    t1 = abs(breit_1[2*i, 2*j, 2*k, 2*l] +
                             breit_1[2*i, 2*j, 2*l+1, 2*k+1])
                    t2 = abs(breit_1[2*j+1, 2*i+1, 2*k, 2*l] +
                             breit_1[2*j+1, 2*i+1, 2*l+1, 2*k+1])
                    t3 = abs(breit_1[2*i, 2*j, 2*k, 2*l] +
                             breit_1[2*j+1, 2*i+1, 2*k, 2*l])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Breit AAAA group is not time-reversal symmetric")
                        print(breit_1[2*i, 2*j, 2*k, 2*l+1], -breit_1[2*i, 2*j, 2*l+1, 2*k+1], -
                              breit_1[2*j+1, 2*i+1, 2*k, 2*l], breit_1[2*j+1, 2*i+1, 2*l+1, 2*k+1])

                    t1 = abs(int2e1[2*i, 2*j, 2*k, 2*l+1] +
                             int2e1[2*i, 2*j, 2*l, 2*k+1])
                    t2 = abs(int2e1[2*j+1, 2*i+1, 2*k, 2*l+1] +
                             int2e1[2*j+1, 2*i+1, 2*l, 2*k+1])
                    t3 = abs(int2e1[2*i, 2*j, 2*k, 2*l+1] -
                             int2e1[2*j+1, 2*i+1, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Coulomb AAAB group is not time-reversal symmetric")
                        print(int2e1[2*i, 2*j, 2*k, 2*l+1], -int2e1[2*i, 2*j, 2*l, 2*k+1],
                              int2e1[2*j+1, 2*i+1, 2*k, 2*l+1], -int2e1[2*j+1, 2*i+1, 2*l, 2*k+1])

                    t1 = abs(breit_1[2*i, 2*j, 2*k, 2*l+1] -
                             breit_1[2*i, 2*j, 2*l, 2*k+1])
                    t2 = abs(breit_1[2*j+1, 2*i+1, 2*k, 2*l+1] -
                             breit_1[2*j+1, 2*i+1, 2*l, 2*k+1])
                    t3 = abs(breit_1[2*i, 2*j, 2*k, 2*l+1] +
                             breit_1[2*j+1, 2*i+1, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Breit AAAB group is not time-reversal symmetric")
                        print(breit_1[2*i, 2*j, 2*k, 2*l+1], breit_1[2*i, 2*j, 2*l, 2*k+1], -
                              breit_1[2*j+1, 2*i+1, 2*k, 2*l+1], -breit_1[2*j+1, 2*i+1, 2*l, 2*k+1])

                    t1 = abs(int2e1[2*i, 2*j+1, 2*k, 2*l+1] +
                             int2e1[2*i, 2*j+1, 2*l, 2*k+1])
                    t2 = abs(int2e1[2*j, 2*i+1, 2*k, 2*l+1] +
                             int2e1[2*j, 2*i+1, 2*l, 2*k+1])
                    t3 = abs(int2e1[2*i, 2*j+1, 2*k, 2*l+1] +
                             int2e1[2*j, 2*i+1, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Coulomb ABAB group is not time-reversal symmetric")
                        print(int2e1[2*i, 2*j+1, 2*k, 2*l+1], -int2e1[2*i, 2*j+1, 2*l, 2*k +
                              1], -int2e1[2*j, 2*i+1, 2*k, 2*l+1], int2e1[2*j, 2*i+1, 2*l, 2*k+1])

                    t1 = abs(breit_1[2*i, 2*j+1, 2*k, 2*l+1] -
                             breit_1[2*i, 2*j+1, 2*l, 2*k+1])
                    t2 = abs(breit_1[2*j, 2*i+1, 2*k, 2*l+1] -
                             breit_1[2*j, 2*i+1, 2*l, 2*k+1])
                    t3 = abs(breit_1[2*i, 2*j+1, 2*k, 2*l+1] -
                             breit_1[2*j, 2*i+1, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Breit ABAB group is not time-reversal symmetric")
                        print(breit_1[2*i, 2*j+1, 2*k, 2*l+1], breit_1[2*i, 2*j+1, 2*l, 2*k+1],
                              breit_1[2*j, 2*i+1, 2*k, 2*l+1], breit_1[2*j, 2*i+1, 2*l, 2*k+1])

                    t1 = abs(int2e1[2*i+1, 2*j, 2*k, 2*l+1] +
                             int2e1[2*i+1, 2*j, 2*l, 2*k+1])
                    t2 = abs(int2e1[2*j+1, 2*i, 2*k, 2*l+1] +
                             int2e1[2*j+1, 2*i, 2*l, 2*k+1])
                    t3 = abs(int2e1[2*i+1, 2*j, 2*k, 2*l+1] +
                             int2e1[2*j+1, 2*i, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Coulomb BAAB group is not time-reversal symmetric")
                        print(int2e1[2*i+1, 2*j, 2*k, 2*l+1], -int2e1[2*i+1, 2*j, 2*l, 2*k +
                              1], -int2e1[2*j+1, 2*i, 2*k, 2*l+1], int2e1[2*j+1, 2*i, 2*l, 2*k+1])

                    t1 = abs(breit_1[2*i+1, 2*j, 2*k, 2*l+1] -
                             breit_1[2*i+1, 2*j, 2*l, 2*k+1])
                    t2 = abs(breit_1[2*j+1, 2*i, 2*k, 2*l+1] -
                             breit_1[2*j+1, 2*i, 2*l, 2*k+1])
                    t3 = abs(breit_1[2*i+1, 2*j, 2*k, 2*l+1] -
                             breit_1[2*j+1, 2*i, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Breit BAAB group is not time-reversal symmetric")
                        print(breit_1[2*i+1, 2*j, 2*k, 2*l+1], breit_1[2*i+1, 2*j, 2*l, 2*k+1],
                              breit_1[2*j+1, 2*i, 2*k, 2*l+1], breit_1[2*j+1, 2*i, 2*l, 2*k+1])

    ##### check out_core mode #####

    def view(h5file, dataname='eri_mo'):
        import h5py
        f5 = h5py.File(h5file, 'r')
        print('dataset %s, shape %s' %
              (str(f5.keys()), str(f5[dataname].shape)))
        f5.close()

    _r_outcore_Coulomb(mol, mf, prefix="F", max_memory=5, ioblk_size=2)
    _r_outcore_Breit(mol, mf, prefix="F", max_memory=5, ioblk_size=2)

    view(coulomb_LLLL % "F")
    view(coulomb_LLSS % "F")
    view(coulomb_SSLL % "F")
    view(coulomb_SSSS % "F")

    view(breit_LSLS % "F")
    view(breit_SLSL % "F")
    view(breit_LSSL % "F")
    view(breit_SLLS % "F")

    ### check whether it is correct ###

    import h5py

    n = mol.nao_2c()
    c1 = .5 / lib.param.LIGHT_SPEED

    feri_coulomb_LLLL = h5py.File(coulomb_LLLL % "F", 'r')
    feri_coulomb_LLSS = h5py.File(coulomb_LLSS % "F", 'r')
    feri_coulomb_SSLL = h5py.File(coulomb_SSLL % "F", 'r')
    feri_coulomb_SSSS = h5py.File(coulomb_SSSS % "F", 'r')

    eri_coulomb = numpy.array(feri_coulomb_LLLL['eri_mo']).reshape(n, n, n, n)
    eri_coulomb += numpy.array(feri_coulomb_LLSS['eri_mo']
                               ).reshape(n, n, n, n) * c1**2
    eri_coulomb += numpy.array(feri_coulomb_SSLL['eri_mo']
                               ).reshape(n, n, n, n) * c1**2
    eri_coulomb += numpy.array(feri_coulomb_SSSS['eri_mo']
                               ).reshape(n, n, n, n) * c1**4

    feri_breit_LSLS = h5py.File(breit_LSLS % "F", 'r')
    feri_breit_SLSL = h5py.File(breit_SLSL % "F", 'r')
    feri_breit_LSSL = h5py.File(breit_LSSL % "F", 'r')
    feri_breit_SLLS = h5py.File(breit_SLLS % "F", 'r')

    eri_breit = numpy.array(feri_breit_LSLS['eri_mo']).reshape(n, n, n, n)
    eri_breit += numpy.array(feri_breit_SLSL['eri_mo']).reshape(n, n, n, n)
    eri_breit += numpy.array(feri_breit_LSSL['eri_mo']).reshape(n, n, n, n)
    eri_breit += numpy.array(feri_breit_SLLS['eri_mo']).reshape(n, n, n, n)
    eri_breit *= c1**2

    print("eri_coulomb diff = ", numpy.linalg.norm(eri_coulomb - int2e1))
    print("eri_breit diff = ", numpy.linalg.norm(eri_breit - breit_1))

    int2e2, breit_2 = FCIDUMP_Rela4C(
        mol, mf, with_breit=True, filename="FCIDUMP_4C_outcore_1", mode="outcore", debug=True)
    int2e2, breit_2 = FCIDUMP_Rela4C(
        mol, mf, with_breit=True, filename="FCIDUMP_4C_outcore_2", mode="outcore", debug=False)

    feri_coulomb_LLLL.close()
    feri_coulomb_LLSS.close()
    feri_coulomb_SSLL.close()
    feri_coulomb_SSSS.close()

    feri_breit_LSLS.close()
    feri_breit_SLSL.close()
    feri_breit_LSSL.close()
    feri_breit_SLLS.close()