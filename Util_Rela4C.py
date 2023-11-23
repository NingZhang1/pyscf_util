from pyscf import gto, scf, lib
# import libzquatev
import sys
import pickle

import zquatev
import numpy
from functools import reduce
from pyscf import lib
from pyscf import tools
import pyscf

import re


def read(filename, verbose=True):
    '''Parse FCIDUMP.  Return a dictionary to hold the integrals and
    parameters with keys:  H1, H2, ECORE, NORB, NELEC, MS, ORBSYM, ISYM

    Kwargs:
        molpro_orbsym (bool): Whether the orbsym in the FCIDUMP file is in
            Molpro orbsym convention as documented in
            https://www.molpro.net/info/current/doc/manual/node36.html
            In return, orbsym is converted to pyscf symmetry convention
        verbose (bool): Whether to print debugging information
    '''
    if verbose:
        print('Parsing %s' % filename)
    finp = open(filename, 'r')

    data = []
    for i in range(10):
        line = finp.readline().upper()
        data.append(line)
        if '&END' in line:
            break
    else:
        raise RuntimeError('Problematic FCIDUMP header')

    result = {}
    tokens = ','.join(data).replace('&FCI', '').replace('&END', '')
    tokens = tokens.replace(' ', '').replace('\n', '').replace(',,', ',')
    for token in re.split(',(?=[a-zA-Z])', tokens):
        key, val = token.split('=')
        if key in ('NORB', 'NELEC', 'MS2', 'ISYM'):
            result[key] = int(val.replace(',', ''))
        elif key in ('ORBSYM',):
            result[key] = [int(x) for x in val.replace(',', ' ').split()]
        else:
            result[key] = val

    # Convert to Molpro orbsym convert_orbsym
    if 'ORBSYM' in result:
        if min(result['ORBSYM']) < 0:
            raise RuntimeError('Unknown orbsym convention')

    norb = result['NORB']
    n2c = norb * 2
    norb_pair = norb * (norb+1) // 2
    h1e = numpy.zeros((n2c, n2c), dtype=numpy.complex128)
    h2e = numpy.zeros((n2c, n2c, n2c, n2c), dtype=numpy.complex128)
    dat = finp.readline().split()
    while dat:
        i, j, k, l = [int(x) for x in dat[2:6]]
        if k != 0:
            h2e[i][j][k][l] = complex(float(dat[0]), float(dat[1]))
        elif k == 0:
            if j != 0:
                h1e[i-1, j-1] = float(dat[0])
            else:
                result['ECORE'] = float(dat[0])
        dat = finp.readline().split()

    idx, idy = numpy.tril_indices(norb, -1)
    if numpy.linalg.norm(h1e[idy, idx]) == 0:
        h1e[idy, idx] = h1e[idx, idy]
    elif numpy.linalg.norm(h1e[idx, idy]) == 0:
        h1e[idx, idy] = h1e[idy, idx]
    result['H1'] = h1e
    result['H2'] = h2e
    finp.close()
    return result


def FCIDUMP_Rela4C(mol, my_RDHF, with_breit=False, filename="fcidump", mode="incore", debug=False):

    assert mode in ["original", "incore", "outcore"]

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
            int2e_breit[n2c:, :n2c, n2c:,
                        :n2c] = tmp.conj().transpose(1, 0, 3, 2)
            ##### (LS|SL) and (SL|LS) #####
            tmp2 = mol.intor("int2e_breit_ssp1sps2_spinor") * c1**2
            int2e_breit[:n2c, n2c:, n2c:, :n2c] = tmp2
            int2e_breit[n2c:, :n2c, :n2c, n2c:] = tmp2.transpose(2, 3, 0, 1)
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
        raise NotImplementedError("outcore mode is not implemented yet")

    else:
        raise ValueError("Unknown mode %s" % mode)

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
        if int2e_coulomb.ndim == 4:
            for i in range(n2c):
                # for j in range(n2c):
                for j in range(i + 1):
                    # for k in range(n2c):
                    for k in range(i+1):
                        for l in range(n2c):
                            if abs(int2e_coulomb[i][j][k][l]) > tol:
                                fout.write(output_format % (
                                    int2e_coulomb[i][j][k][l].real, int2e_coulomb[i][j][k][l].imag, i+1, j+1, k+1, l+1))
                            if abs(int2e_breit[i][j][l][k]) > tol:
                                fout.write(output_format % (
                                    int2e_breit[i][j][l][k].real, int2e_coulomb[i][j][l][k].imag, n2c+i+1, n2c+j+1, n2c+l+1, n2c+k+1))
        elif int2e_coulomb.ndim == 2:
            raise NotImplementedError("2-fold symmetry is not implemented yet")
            npair = n2c * (n2c + 1) // 2
            assert (int2e_coulomb.size == npair * npair)
            ij = 0
            for i in range(n2c):
                for j in range(0, i+1):
                    kl = 0
                    for k in range(0, n2c):
                        for l in range(0, k+1):
                            if abs(int2e_coulomb[ij, kl]) > tol:
                                fout.write(output_format %
                                           (int2e_coulomb[ij, kl].real, int2e_coulomb[ij, kl].imag, i+1, j+1, k+1, l+1))
                            kl += 1
                    ij += 1
        else:
            raise ValueError("Unknown int2e_coulomb.ndim %d" %
                             int2e_coulomb.ndim)

        output_format = float_format + float_format + ' %4d %4d  0  0\n'
        for i in range(n2c):
            # for j in range(n2c):
            for j in range(i+1):
                if abs(h1e[i, j]) > tol:
                    fout.write(output_format %
                               (h1e[i, j].real, h1e[i, j].imag, i+1, j+1))

        output_format = float_format + ' 0.0  0  0  0  0\n'
        fout.write(output_format % nuc)

    if debug:
        return int2e_coulomb, int2e_breit


if __name__ == "__main__":
    mol = gto.M(atom='H 0 0 0; H 0 0 1; O 0 1 0', basis='sto-3g', verbose=5)
    # mol = gto.M(atom='F 0 0 0', basis='cc-pvdz', verbose=5, charge=-1, spin=0)
    # mf = scf.RHF(mol)
    # mf.kernel()
    # mf.analyze()
    mf = scf.dhf.RDHF(mol)
    mf.conv_tol = 1e-12
    # mf.kernel()

    # mf.with_gaunt = True
    # mf.kernel()

    mf.with_breit = True
    mf.kernel()

    int2e1, breit_1 = FCIDUMP_Rela4C(
        mol, mf, with_breit=True, filename="FCIDUMP_4C", mode="original", debug=True)

    int2e2, breit_2 = FCIDUMP_Rela4C(
        mol, mf, with_breit=True, filename="FCIDUMP_4C_incore", mode="incore", debug=True)

    # FCIDUMP_Rela4C(mol, mf, filename="FCIDUMP_4C_incore2", mode="incore")

    print("diff = ", numpy.linalg.norm(int2e1 - int2e2))

    if breit_1 is not None:
        print("breit diff = ", numpy.linalg.norm(breit_1 - breit_2))

    # mf = scf.hf.RHF(mol)

    # mf.kernel()

    # nao = mf.mo_coeff.shape[1]
    # print(mf.mo_energy[nao//2:])
    # for i in range(nao//2, nao):
    #     print("nao %d" % i)
    #     for j in range(nao):
    #         if abs(mf.mo_coeff[i, j]) > 1e-6:
    #             print("%4d %15.8f %15.8f" % (j, mf.mo_coeff[i, j].real, mf.mo_coeff[i, j].imag))

    #### check the time-reversal symmetry of the orbitals ####

    nao = mol.nao

    for i in range(nao):
        for j in range(nao):
            for k in range(nao):
                for l in range(nao):

                    t1 = abs(int2e1[2*i, 2*j, 2*k, 2*l] - int2e1[2*i, 2*j, 2*l+1, 2*k+1])
                    t2 = abs(int2e1[2*j+1, 2*i+1, 2*k, 2*l] - int2e1[2*j+1, 2*i+1, 2*l+1, 2*k+1])
                    t3 = abs(int2e1[2*i, 2*j, 2*k, 2*l] - int2e1[2*j+1, 2*i+1, 2*k, 2*l])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Coulomb AAAA group is not time-reversal symmetric")
                        print(int2e1[2*i, 2*j, 2*k, 2*l], int2e1[2*i, 2*j, 2*l+1, 2*k+1], int2e1[2*j+1, 2*i+1, 2*k, 2*l], int2e1[2*j+1, 2*i+1, 2*l+1, 2*k+1])
                    
                    t1 = abs(breit_1[2*i, 2*j, 2*k, 2*l] + breit_1[2*i, 2*j, 2*l+1, 2*k+1])
                    t2 = abs(breit_1[2*j+1, 2*i+1, 2*k, 2*l] + breit_1[2*j+1, 2*i+1, 2*l+1, 2*k+1])
                    t3 = abs(breit_1[2*i, 2*j, 2*k, 2*l] + breit_1[2*j+1, 2*i+1, 2*k, 2*l])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Breit AAAA group is not time-reversal symmetric")
                        print(breit_1[2*i, 2*j, 2*k, 2*l+1], -breit_1[2*i, 2*j, 2*l+1, 2*k+1], -breit_1[2*j+1, 2*i+1, 2*k, 2*l], breit_1[2*j+1, 2*i+1, 2*l+1, 2*k+1])

                    t1 = abs(int2e1[2*i, 2*j, 2*k, 2*l+1] + int2e1[2*i, 2*j, 2*l, 2*k+1])
                    t2 = abs(int2e1[2*j+1, 2*i+1, 2*k, 2*l+1] + int2e1[2*j+1, 2*i+1, 2*l, 2*k+1])
                    t3 = abs(int2e1[2*i, 2*j, 2*k, 2*l+1] - int2e1[2*j+1, 2*i+1, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Coulomb AAAB group is not time-reversal symmetric")
                        print(int2e1[2*i, 2*j, 2*k, 2*l+1], -int2e1[2*i, 2*j, 2*l, 2*k+1], int2e1[2*j+1, 2*i+1, 2*k, 2*l+1], -int2e1[2*j+1, 2*i+1, 2*l, 2*k+1])

                    t1 = abs(breit_1[2*i, 2*j, 2*k, 2*l+1] - breit_1[2*i, 2*j, 2*l, 2*k+1])
                    t2 = abs(breit_1[2*j+1, 2*i+1, 2*k, 2*l+1] - breit_1[2*j+1, 2*i+1, 2*l, 2*k+1])
                    t3 = abs(breit_1[2*i, 2*j, 2*k, 2*l+1] + breit_1[2*j+1, 2*i+1, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Breit AAAB group is not time-reversal symmetric")
                        print(breit_1[2*i, 2*j, 2*k, 2*l+1], breit_1[2*i, 2*j, 2*l, 2*k+1], -breit_1[2*j+1, 2*i+1, 2*k, 2*l+1], -breit_1[2*j+1, 2*i+1, 2*l, 2*k+1])

                    
                    t1 = abs(int2e1[2*i, 2*j+1, 2*k, 2*l+1] + int2e1[2*i, 2*j+1, 2*l, 2*k+1])
                    t2 = abs(int2e1[2*j, 2*i+1, 2*k, 2*l+1] + int2e1[2*j, 2*i+1, 2*l, 2*k+1])
                    t3 = abs(int2e1[2*i, 2*j+1, 2*k, 2*l+1] + int2e1[2*j, 2*i+1, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Coulomb ABAB group is not time-reversal symmetric")
                        print(int2e1[2*i, 2*j+1, 2*k, 2*l+1], -int2e1[2*i, 2*j+1, 2*l, 2*k+1], -int2e1[2*j, 2*i+1, 2*k, 2*l+1], int2e1[2*j, 2*i+1, 2*l, 2*k+1])

                    t1 = abs(breit_1[2*i, 2*j+1, 2*k, 2*l+1] - breit_1[2*i, 2*j+1, 2*l, 2*k+1])
                    t2 = abs(breit_1[2*j, 2*i+1, 2*k, 2*l+1] - breit_1[2*j, 2*i+1, 2*l, 2*k+1])
                    t3 = abs(breit_1[2*i, 2*j+1, 2*k, 2*l+1] - breit_1[2*j, 2*i+1, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Breit ABAB group is not time-reversal symmetric")
                        print(breit_1[2*i, 2*j+1, 2*k, 2*l+1], breit_1[2*i, 2*j+1, 2*l, 2*k+1], breit_1[2*j, 2*i+1, 2*k, 2*l+1], breit_1[2*j, 2*i+1, 2*l, 2*k+1])

                    t1 = abs(int2e1[2*i+1, 2*j, 2*k, 2*l+1] + int2e1[2*i+1, 2*j, 2*l, 2*k+1])
                    t2 = abs(int2e1[2*j+1, 2*i, 2*k, 2*l+1] + int2e1[2*j+1, 2*i, 2*l, 2*k+1])
                    t3 = abs(int2e1[2*i+1, 2*j, 2*k, 2*l+1] + int2e1[2*j+1, 2*i, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Coulomb BAAB group is not time-reversal symmetric")
                        print(int2e1[2*i+1, 2*j, 2*k, 2*l+1], -int2e1[2*i+1, 2*j, 2*l, 2*k+1], -int2e1[2*j+1, 2*i, 2*k, 2*l+1], int2e1[2*j+1, 2*i, 2*l, 2*k+1])

                    t1 = abs(breit_1[2*i+1, 2*j, 2*k, 2*l+1] - breit_1[2*i+1, 2*j, 2*l, 2*k+1])
                    t2 = abs(breit_1[2*j+1, 2*i, 2*k, 2*l+1] - breit_1[2*j+1, 2*i, 2*l, 2*k+1])
                    t3 = abs(breit_1[2*i+1, 2*j, 2*k, 2*l+1] - breit_1[2*j+1, 2*i, 2*k, 2*l+1])
                    if t1 > 1e-8 or t2 > 1e-8 or t3 > 1e-8:
                        print("Breit BAAB group is not time-reversal symmetric")
                        print(breit_1[2*i+1, 2*j, 2*k, 2*l+1], breit_1[2*i+1, 2*j, 2*l, 2*k+1], breit_1[2*j+1, 2*i, 2*k, 2*l+1], breit_1[2*j+1, 2*i, 2*l, 2*k+1])