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


def FCIDUMP_Rela4C(mol, my_RDHF, filename, mode, debug=False):

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
        int2e_full = numpy.einsum(
            "ijkl,ip->pjkl", int2e_res, mo_coeff_pes.conj())
        int2e_full = numpy.einsum("pjkl,jq->pqkl", int2e_full, mo_coeff_pes)
        int2e_full = numpy.einsum(
            "pqkl,kr->pqrl", int2e_full, mo_coeff_pes.conj())
        int2e_full = numpy.einsum("pqrl,ls->pqrs", int2e_full, mo_coeff_pes)
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
        int2e_full = int2e_res
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
        if int2e_full.ndim == 4:
            for i in range(n2c):
                # for j in range(n2c):
                for j in range(i + 1):
                    # for k in range(n2c):
                    for k in range(i+1):
                        for l in range(n2c):
                            if abs(int2e_full[i][j][k][l]) > tol:
                                fout.write(output_format % (
                                    int2e_full[i][j][k][l].real, int2e_full[i][j][k][l].imag, i+1, j+1, k+1, l+1))
        elif int2e_full.ndim == 2:
            npair = n2c * (n2c + 1) // 2
            assert (int2e_full.size == npair * npair)
            ij = 0
            for i in range(n2c):
                for j in range(0, i+1):
                    kl = 0
                    for k in range(0, n2c):
                        for l in range(0, k+1):
                            if abs(int2e_full[ij, kl]) > tol:
                                fout.write(output_format %
                                           (int2e_full[ij, kl].real, int2e_full[ij, kl].imag, i+1, j+1, k+1, l+1))
                            kl += 1
                    ij += 1
        else:
            raise ValueError("Unknown int2e_full.ndim %d" % int2e_full.ndim)

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
        return int2e_full


if __name__ == "__main__":
    # mol = gto.M(atom='H 0 0 0; H 0 0 1; O 0 1 0', basis='sto-3g', verbose=5)
    mol = gto.M(atom='F 0 0 0', basis='cc-pvdz', verbose=5, charge=-1, spin=0)
    # mf = scf.RHF(mol)
    # mf.kernel()
    # mf.analyze()
    mf = scf.dhf.RDHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    # mf.with_gaunt = True
    # mf.kernel()

    # mf.with_breit = True
    # mf.kernel()

    int2e1 = FCIDUMP_Rela4C(mol, mf, "FCIDUMP_4C", mode="original", debug=True)

    int2e2 = FCIDUMP_Rela4C(mol, mf, "FCIDUMP_4C_incore",
                            mode="incore", debug=True)

    print("diff = ", numpy.linalg.norm(int2e1 - int2e2))

    # mf = scf.hf.RHF(mol)

    # mf.kernel()

    # nao = mf.mo_coeff.shape[1]
    # print(mf.mo_energy[nao//2:])
    # for i in range(nao//2, nao):
    #     print("nao %d" % i)
    #     for j in range(nao):
    #         if abs(mf.mo_coeff[i, j]) > 1e-6:
    #             print("%4d %15.8f %15.8f" % (j, mf.mo_coeff[i, j].real, mf.mo_coeff[i, j].imag))
