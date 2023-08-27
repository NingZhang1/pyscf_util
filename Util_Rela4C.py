from pyscf import gto, scf
# import libzquatev
import sys
import pickle

import zquatev
import numpy
from functools import reduce
from pyscf import lib
from pyscf import tools


def FCIDUMP_Rela4C(mol, my_RDHF, filename):
    n2c = mol.nao_2c()
    mo_coeff = my_RDHF.mo_coeff
    mo_coeff_mat = numpy.matrix(mo_coeff)

    mo_coeff_pes = mo_coeff_mat[:, n2c:]

    hcore = my_RDHF.get_hcore()
    h1e = reduce(numpy.dot, (mo_coeff_pes.H, hcore, mo_coeff_pes))

    n4c = 2 * n2c

    int2e_res = numpy.zeros((n4c, n4c, n4c, n4c), dtype=numpy.complex128)
    c1 = .5 / lib.param.LIGHT_SPEED

    int2e_res[:n2c, : n2c, :n2c, :n2c] = mol.intor("int2e_spinor")  # LL LL

    tmp = mol.intor("int2e_spv1_spinor") * c1
    int2e_res[n2c:, : n2c, :n2c, :n2c] = tmp  # SL LL
    int2e_res[:n2c, :n2c, n2c:, : n2c] = tmp.transpose(2, 3, 0, 1)  # LL SL

    tmp = mol.intor("int2e_vsp1_spinor") * c1
    int2e_res[:n2c, n2c:, :n2c, :n2c] = tmp  # LS LL
    int2e_res[:n2c, :n2c, :n2c, n2c:] = tmp.transpose(2, 3, 0, 1)  # LL SL

    tmp = mol.intor("int2e_spsp1_spinor") * c1**2
    int2e_res[n2c:, n2c:, :n2c, :n2c] = tmp  # SS LL
    int2e_res[:n2c, :n2c, n2c:, n2c:] = tmp.transpose(2, 3, 0, 1)  # LL SS

    int2e_res[:n2c, n2c:, :n2c, n2c:] = mol.intor(
        "int2e_vsp1vsp2_spinor") * c1**2  # LS LS
    int2e_res[n2c:, :n2c, n2c:, :n2c] = mol.intor(
        "int2e_spv1spv2_spinor") * c1**2  # SL SL

    int2e_res[:n2c, n2c:, n2c:, :n2c] = mol.intor(
        "int2e_vsp1spv2_spinor") * c1**2  # LS SL
    int2e_res[n2c:, :n2c, :n2c, n2c:] = mol.intor(
        "int2e_spv1vsp2_spinor") * c1**2  # SL LS

    tmp = mol.intor("int2e_vsp1spsp2_spinor") * c1**3
    int2e_res[:n2c, n2c:, n2c:, n2c:] = tmp  # LS SS
    int2e_res[n2c:, n2c:, :n2c, n2c:] = tmp.transpose(2, 3, 0, 1)  # SS LS

    tmp = mol.intor("int2e_spv1spsp2_spinor") * c1**3
    int2e_res[n2c:, :n2c,  n2c:, n2c:] = tmp  # SL SS
    int2e_res[n2c:, n2c:, n2c:, :n2c] = tmp.transpose(2, 3, 0, 1)  # SS SL

    int2e_res[n2c:, n2c:, n2c:, n2c:] = mol.intor(
        "int2e_spsp1spsp2_spinor") * c1**4  # SS SS

    int2e_full = numpy.einsum("ijkl,ip->pjkl", int2e_res, mo_coeff_pes.conj())
    int2e_full = numpy.einsum("pjkl,jq->pqkl", int2e_full, mo_coeff_pes)
    int2e_full = numpy.einsum("pqkl,kr->pqrl", int2e_full, mo_coeff_pes.conj())
    int2e_full = numpy.einsum("pqrl,ls->pqrs", int2e_full, mo_coeff_pes)

    energy_core = mol.get_enuc()

    nmo = n2c // 2
    nelec = mol.nelectron
    ms = 0
    tol = 1e-10
    nuc = energy_core
    float_format = tools.fcidump.DEFAULT_FLOAT_FORMAT

    orbsym_ID = []
    for _ in range(nmo):
        orbsym_ID.append(0)

    with open(filename, 'w') as fout:  # 4-fold symmetry
        tools.fcidump.write_head(fout, nmo, nelec, ms, orbsym_ID)
        output_format = float_format + float_format + ' %4d %4d %4d %4d\n'
        for i in range(n2c):
            for j in range(n2c):
                for k in range(n2c):
                    for l in range(n2c):
                        if abs(int2e_full[i][j][k][l]) > tol:
                            fout.write(output_format % (
                                int2e_full[i][j][k][l].real, int2e_full[i][j][k][l].imag, i+1, j+1, k+1, l+1))

        output_format = float_format + float_format + ' %4d %4d  0  0\n'
        for i in range(n2c):
            for j in range(n2c):
                if abs(h1e[i, j]) > tol:
                    fout.write(output_format %
                               (h1e[i, j].real, h1e[i, j].imag, i+1, j+1))

        output_format = float_format + ' 0.0  0  0  0  0\n'
        fout.write(output_format % nuc)

if __name__ == "__main__":
    mol = gto.M(atom='H 0 0 0; H 0 0 1; O 0 1 0', basis='sto-3g', verbose=5)
    # mf = scf.RHF(mol)
    # mf.kernel()
    # mf.analyze()
    mf = scf.dhf.RDHF(mol)
    mf.kernel()

    FCIDUMP_Rela4C(mol, mf, "FCIDUMP_4C")