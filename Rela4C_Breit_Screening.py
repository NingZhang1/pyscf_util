from pyscf import gto, scf, lib
import numpy
from functools import reduce
from pyscf import lib
from pyscf import tools
import pyscf

import re

from pyscf import __config__


if __name__ == "__main__":
    mol = gto.M(atom='F 0 0 0', basis='unc-cc-pvdz-dk', verbose=5,
                charge=-1, spin=0, symmetry="d2h")
    mol.build()
    mf = scf.dhf.RDHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    n2c = mol.nao_2c()
    n4c = 2 * n2c

    c1 = .5 / lib.param.LIGHT_SPEED
    int2e_breit = numpy.zeros(
        (n4c, n4c, n4c, n4c), dtype=numpy.complex128)

    tmp = mol.intor("int2e_breit_ssp1ssp2_spinor") * c1**2
    int2e_breit[:n2c, n2c:, :n2c, n2c:] = tmp
    tmp = mol.intor("int2e_breit_sps1sps2_spinor") * c1**2
    int2e_breit[n2c:, :n2c, n2c:, :n2c] = tmp
    ##### (LS|SL) and (SL|LS) #####
    tmp2 = mol.intor("int2e_breit_ssp1sps2_spinor") * c1**2
    int2e_breit[:n2c, n2c:, n2c:, :n2c] = tmp2  # (LS|SL)
    tmp2 = mol.intor("int2e_breit_sps1ssp2_spinor") * c1**2
    int2e_breit[n2c:, :n2c, :n2c, n2c:] = tmp2  # (SL|LS)

    # print("Check Breit Term screening condition")

    # for i in range(n4c):
    #     for j in range(n4c):
    #         for k in range(n4c):
    #             for l in range(n4c):
    #                 int_now = int2e_breit[i, j, k, l]
    #                 int_approx = numpy.sqrt(
    #                     abs(int2e_breit[i, j, j, i] * int2e_breit[k, l, l, k]))
    #                 if abs(int_now) > abs(int_approx) + 1e-8:
    #                     print("Breit Term does not satisfy the screening condition (ijji) %15.8e > %15.8e" % (
    #                         abs(int_now), abs(int_approx)))   # this approx seems to be no problem!
    #                 int_approx_2 = numpy.sqrt(
    #                     abs(int2e_breit[i, j, i, j] * int2e_breit[k, l, k, l]))
    #                 if abs(int_now) > abs(int_approx_2) + 1e-8:
    #                     print("Breit Term does not satisfy the screening condition (ijij) %15.8e > %15.8e" % (
    #                         abs(int_now), abs(int_approx_2)))
    # else:
    #     print("%15.8e < %15.8e" % (int_now, int_approx))

    print("Breit Term")

    for i in range(n4c):
        for j in range(n4c):
            print("i j j i with i = %d, j = %d: %15.8e %15.8e" %
                  (i, j, int2e_breit[i, j, j, i].real, int2e_breit[i, j, j, i].imag))
            print("i j i j with i = %d, j = %d: %15.8e %15.8e" %
                  (i, j, int2e_breit[i, j, i, j].real, int2e_breit[i, j, i, j].imag))

    print("Coulomb Term")

    int2e_res = numpy.zeros((n4c, n4c, n4c, n4c), dtype=numpy.complex128)
    c1 = .5 / lib.param.LIGHT_SPEED
    int2e_res[:n2c, :n2c, :n2c, :n2c] = mol.intor("int2e_spinor")  # LL LL
    tmp = mol.intor("int2e_spsp1_spinor") * c1**2
    int2e_res[n2c:, n2c:, :n2c, :n2c] = tmp  # SS LL
    int2e_res[:n2c, :n2c, n2c:, n2c:] = tmp.transpose(2, 3, 0, 1)  # LL SS
    int2e_res[n2c:, n2c:, n2c:, n2c:] = mol.intor(
        "int2e_spsp1spsp2_spinor") * c1**4  # SS SS

    # print("Check Coulomb Term screening condition")

    # for i in range(n4c):
    #     for j in range(n4c):
    #         for k in range(n4c):
    #             for l in range(n4c):
    #                 int_now = int2e_res[i, j, k, l]
    #                 int_approx = numpy.sqrt(
    #                     abs(int2e_res[i, j, j, i] * int2e_res[k, l, l, k]))
    #                 if abs(int_now) > abs(int_approx) + 1e-8:
    #                     print("Coulomb Term does not satisfy the screening condition (ijji) %15.8e > %15.8e" % (
    #                         abs(int_now), abs(int_approx)))   # this approx seems to be no problem!
    #                 int_approx_2 = numpy.sqrt(
    #                     abs(int2e_res[i, j, i, j] * int2e_res[k, l, k, l]))
    #                 if abs(int_now) > abs(int_approx_2) + 1e-8:
    #                     print("Coulomb Term does not satisfy the screening condition (ijij) %15.8e > %15.8e" % (
    #                         abs(int_now), abs(int_approx_2)))

    for i in range(n4c):
        for j in range(n4c):
            print("i j j i with i = %d, j = %d: %15.8e %15.8e" %
                  (i, j, int2e_res[i, j, j, i].real, int2e_res[i, j, j, i].imag))
            print("i j i j with i = %d, j = %d: %15.8e %15.8e" %
                  (i, j, int2e_res[i, j, i, j].real, int2e_res[i, j, i, j].imag))

    import sys
    sys.stdout.flush()

    # debug for the screening condition

    from pyscf.ao2mo import _ao2mo

    ao2mopt = _ao2mo.AO2MOpt(mol, "int2e_spinor", 'CVHFrkbllll_schwarz_cond',
                             'CVHFrkbllll_direct_scf_debug')

    ao2mopt = _ao2mo.AO2MOpt(mol, "int2e_spsp1spsp2_spinor", 'CVHFrkbssss_schwarz_cond',
                             'CVHFrkbssss_direct_scf_debug')
