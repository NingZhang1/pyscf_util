import pyscf
import numpy
import Driver_SCF
import Util_Mole
from functools import reduce
from pyscf import tools

A1_ID = 0
A2_ID = 1
E1x_ID = 2
E1y_ID = 3
E2x_ID = 10
E2y_ID = 11
E3x_ID = 12
E3y_ID = 13
E4x_ID = 20
E4y_ID = 21
E5x_ID = 22
E5y_ID = 23

E1_up_ID = 2
E1_dn_ID = 3
E2_up_ID = 10
E2_dn_ID = 11
E3_up_ID = 12
E3_dn_ID = 13
E4_up_ID = 20
E4_dn_ID = 21
E5_up_ID = 22
E5_dn_ID = 23


def get_symmetry_adapted_basis_Coov(mol, coeff):
    orbsym_ID, _ = Util_Mole.get_orbsym(mol, coeff)
    return _get_symmetry_adapted_basis_Coov(orbsym_ID)


def _get_symmetry_adapted_basis_Coov(orbsym_ID):

    # orbsym_ID, _ = Util_Mole.get_orbsym(mol, coeff)

    A1 = [i for i, x in enumerate(orbsym_ID) if x is A1_ID]
    A2 = [i for i, x in enumerate(orbsym_ID) if x is A2_ID]

    E1_x = [i for i, x in enumerate(orbsym_ID) if x is E1x_ID]
    E1_y = [i for i, x in enumerate(orbsym_ID) if x is E1y_ID]

    E2_x = [i for i, x in enumerate(orbsym_ID) if x is E2x_ID]
    E2_y = [i for i, x in enumerate(orbsym_ID) if x is E2y_ID]

    E3_x = [i for i, x in enumerate(orbsym_ID) if x is E3x_ID]
    E3_y = [i for i, x in enumerate(orbsym_ID) if x is E3y_ID]

    E4_x = [i for i, x in enumerate(orbsym_ID) if x is E4x_ID]
    E4_y = [i for i, x in enumerate(orbsym_ID) if x is E4y_ID]

    E5_x = [i for i, x in enumerate(orbsym_ID) if x is E5x_ID]
    E5_y = [i for i, x in enumerate(orbsym_ID) if x is E5y_ID]

    factor = 1.0/numpy.sqrt(2)

    basis_trans = numpy.identity(len(orbsym_ID), dtype=numpy.complex128)

    Lz = numpy.zeros(len(orbsym_ID), dtype=numpy.int32)

    for orbx, orby in zip(E1_x, E1_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = factor*(0+1j)
        basis_trans[orby, orby] = -factor*(0+1j)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 1
        Lz[orby] = -1

    for orbx, orby in zip(E2_x, E2_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = factor*(0+1j)
        basis_trans[orby, orby] = -factor*(0+1j)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 2
        Lz[orby] = -2

    for orbx, orby in zip(E3_x, E3_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = factor*(0+1j)
        basis_trans[orby, orby] = -factor*(0+1j)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 3
        Lz[orby] = -3

    for orbx, orby in zip(E4_x, E4_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = factor*(0+1j)
        basis_trans[orby, orby] = -factor*(0+1j)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 4
        Lz[orby] = -4

    for orbx, orby in zip(E5_x, E5_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = factor*(0+1j)
        basis_trans[orby, orby] = -factor*(0+1j)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 5
        Lz[orby] = -5

    return numpy.matrix(basis_trans), Lz


def _get_time_reversal_half_pair(orbsym_ID):

    E1_up = [i for i, x in enumerate(orbsym_ID) if x is E1_up_ID]
    E2_up = [i for i, x in enumerate(orbsym_ID) if x is E2_up_ID]
    E3_up = [i for i, x in enumerate(orbsym_ID) if x is E3_up_ID]
    E4_up = [i for i, x in enumerate(orbsym_ID) if x is E4_up_ID]
    E5_up = [i for i, x in enumerate(orbsym_ID) if x is E5_up_ID]

    Res = []
    Res.extend(E1_up)
    Res.extend(E2_up)
    Res.extend(E3_up)
    Res.extend(E4_up)
    Res.extend(E5_up)

    return Res


def _get_time_reversal_pair(orbsym_ID):

    E1_up = [i for i, x in enumerate(orbsym_ID) if x is E1_up_ID]
    E1_dn = [i for i, x in enumerate(orbsym_ID) if x is E1_dn_ID]

    E2_up = [i for i, x in enumerate(orbsym_ID) if x is E2_up_ID]
    E2_dn = [i for i, x in enumerate(orbsym_ID) if x is E2_dn_ID]

    E3_up = [i for i, x in enumerate(orbsym_ID) if x is E3_up_ID]
    E3_dn = [i for i, x in enumerate(orbsym_ID) if x is E3_dn_ID]

    E4_up = [i for i, x in enumerate(orbsym_ID) if x is E4_up_ID]
    E4_dn = [i for i, x in enumerate(orbsym_ID) if x is E4_dn_ID]

    E5_up = [i for i, x in enumerate(orbsym_ID) if x is E5_up_ID]
    E5_dn = [i for i, x in enumerate(orbsym_ID) if x is E5_dn_ID]

    Res = range(0, len(orbsym_ID))

    for i in range(len(E1_up)):
        up_ID = E1_up[i]
        dn_ID = E1_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E2_up)):
        up_ID = E2_up[i]
        dn_ID = E2_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E3_up)):
        up_ID = E3_up[i]
        dn_ID = E3_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E4_up)):
        up_ID = E4_up[i]
        dn_ID = E4_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E5_up)):
        up_ID = E5_up[i]
        dn_ID = E5_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    return Res


def _check_orb_range_valid(time_reversal_pair, begin_orb, end_orb):
    for i in range(begin_orb, end_orb):
        if time_reversal_pair[i] < begin_orb or time_reversal_pair[i] >= end_orb:
            raise RuntimeError


def tranform_rdm1_adapted_2_xy(orbsym_ID, rdm1, begin_orb, end_orb):

    time_reversal_pair = _get_time_reversal_pair(orbsym_ID)
    _check_orb_range_valid(time_reversal_pair, begin_orb, end_orb)

    basis_trans, _ = _get_symmetry_adapted_basis_Coov(orbsym_ID)
    basis_trans = basis_trans.H
    basis_trans = basis_trans[begin_orb:end_orb, begin_orb:end_orb]

    Res = rdm1
    Res = numpy.einsum("ij,ip->pj", Res, basis_trans.conj())
    Res = numpy.einsum("pj,jq->pq", Res, basis_trans)

    return Res


def tranform_rdm1_xy_2_adapted(orbsym_ID, rdm1, begin_orb, end_orb):

    time_reversal_pair = _get_time_reversal_pair(orbsym_ID)
    _check_orb_range_valid(time_reversal_pair, begin_orb, end_orb)

    basis_trans, _ = _get_symmetry_adapted_basis_Coov(orbsym_ID)
    basis_trans = basis_trans[begin_orb:end_orb, begin_orb:end_orb]

    Res = rdm1
    Res = numpy.einsum("ij,ip->pj", Res, basis_trans.conj())
    Res = numpy.einsum("pj,jq->pq", Res, basis_trans)

    return Res


def tranform_rdm2_adapted_2_xy(orbsym_ID, rdm2, begin_orb, end_orb):

    time_reversal_pair = _get_time_reversal_pair(orbsym_ID)
    _check_orb_range_valid(time_reversal_pair, begin_orb, end_orb)

    basis_trans, _ = _get_symmetry_adapted_basis_Coov(orbsym_ID)
    basis_trans = basis_trans.H
    basis_trans = basis_trans[begin_orb:end_orb, begin_orb:end_orb]

    Res = rdm2

    Res = numpy.einsum("ijkl,ip->pjkl", Res, basis_trans.conj())
    Res = numpy.einsum("pjkl,jq->pqkl", Res, basis_trans.conj())
    Res = numpy.einsum("pqkl,kr->pqrl", Res, basis_trans)
    Res = numpy.einsum("pqrl,ls->pqrs", Res, basis_trans)

    return Res


def tranform_rdm2_xy_2_adapted(orbsym_ID, rdm2, begin_orb, end_orb):

    time_reversal_pair = _get_time_reversal_pair(orbsym_ID)
    _check_orb_range_valid(time_reversal_pair, begin_orb, end_orb)

    basis_trans, _ = _get_symmetry_adapted_basis_Coov(orbsym_ID)
    basis_trans = basis_trans[begin_orb:end_orb, begin_orb:end_orb]

    Res = rdm2

    Res = numpy.einsum("ijkl,ip->pjkl", Res, basis_trans.conj())
    Res = numpy.einsum("pjkl,jq->pqkl", Res, basis_trans.conj())
    Res = numpy.einsum("pqkl,kr->pqrl", Res, basis_trans)
    Res = numpy.einsum("pqrl,ls->pqrs", Res, basis_trans)

    return Res


def symmetrize_rdm1(orbsym_ID, rdm1, begin_orb, end_orb, xy_basis=False):

    assert(rdm1.shape[0] == rdm1.shape[1])
    assert(rdm1.shape[0] == (end_orb-begin_orb))

    rdm1_tmp = rdm1

    if xy_basis:
        rdm1_tmp = tranform_rdm1_xy_2_adapted(
            orbsym_ID, rdm1_tmp, begin_orb, end_orb)

    rdm1_tmp_ = numpy.zeros(rdm1_tmp.shape)

    # spatial symmetry

    for i in range(end_orb-begin_orb):
        for j in range(end_orb-begin_orb):
            if orbsym_ID[i+begin_orb] == orbsym_ID[j+begin_orb]:
                rdm1_tmp_[i, j] = rdm1_tmp[i, j]
            else:
                if abs(rdm1_tmp[i, j]) > 1e-6:
                    print("Warning %d %d rdm1 %.4e too large" %
                          (i, j, rdm1_tmp[i, j]))

    # time-reversal symmetry

    time_reversal_half_pair = _get_time_reversal_half_pair(orbsym_ID)
    time_reversal_pair = _get_time_reversal_pair(orbsym_ID)

    for orbi in range(begin_orb, end_orb):
        if orbi in time_reversal_half_pair:
            for orbj in range(begin_orb, end_orb):
                if orbsym_ID[orbi] == orbsym_ID[orbj]:
                    orbi_pair = time_reversal_pair[orbi]
                    orbj_pair = time_reversal_pair[orbj]
                    i = orbi-begin_orb
                    j = orbj-begin_orb
                    i_pair = orbi_pair-begin_orb
                    j_pair = orbj_pair-begin_orb
                    tmp = (rdm1_tmp_[i, j]+rdm1_tmp_[i_pair, j_pair])/2
                    rdm1_tmp_[i, j] = tmp
                    rdm1_tmp_[i_pair, j_pair] = tmp

    if xy_basis:
        rdm1_tmp = tranform_rdm2_adapted_2_xy(
            orbsym_ID, rdm1_tmp_, begin_orb, end_orb)
    else:
        rdm1_tmp = rdm1_tmp_

    return rdm1_tmp.real

# figure out the convention for RDM2 first !


def symmetrize_rdm2(orbsym_ID, rdm2, begin_orb, end_orb, xy_basis=False):

    assert(rdm2.shape[0] == rdm2.shape[1])
    assert(rdm2.shape[0] == rdm2.shape[2])
    assert(rdm2.shape[0] == rdm2.shape[3])
    assert(rdm2.shape[0] == (end_orb-begin_orb))

    rdm2_tmp = rdm2

    if xy_basis:
        rdm2_tmp = tranform_rdm2_xy_2_adapted(
            orbsym_ID, rdm2_tmp, begin_orb, end_orb)

    rdm2_tmp_ = numpy.zeros(rdm2_tmp.shape)

    _, Lz = _get_symmetry_adapted_basis_Coov(orbsym_ID)

    # spatial symmetry

    for orbi in range(begin_orb, end_orb):
        for orbj in range(begin_orb, end_orb):
            for orbk in range(begin_orb, end_orb):
                for orbl in range(begin_orb, end_orb):
                    pass

    # time-reversal symmetry

    time_reversal_half_pair = _get_time_reversal_half_pair(orbsym_ID)
    time_reversal_pair = _get_time_reversal_pair(orbsym_ID)

    if xy_basis:
        rdm2_tmp = tranform_rdm2_adapted_2_xy(
            orbsym_ID, rdm2_tmp_, begin_orb, end_orb)
    else:
        rdm2_tmp = rdm2_tmp_

    return rdm2_tmp.real

def FCIDUMP_Coov(mol, my_scf, filename):
    orbsym_ID, _ = Util_Mole.get_orbsym(mol, my_scf.mo_coeff)
    basis_trans, _ = get_symmetry_adapted_basis_Coov(
        mol, my_scf.mo_coeff)
    
    # construct h1e and h2e

    h1e = reduce(numpy.dot, (my_scf.mo_coeff.T,
                 my_scf.get_hcore(), my_scf.mo_coeff))
    
    h1e_adapted = reduce(numpy.dot, (basis_trans.H,
                                     h1e, basis_trans))
    
    int2e_full = pyscf.ao2mo.full(
        eri_or_mol=mol, mo_coeff=my_scf.mo_coeff, aosym='1').reshape((mol.nao, mol.nao, mol.nao, mol.nao))

    int2e_full = numpy.einsum("ijkl,ip->pjkl", int2e_full, basis_trans.conj())
    int2e_full = numpy.einsum("pjkl,jq->pqkl", int2e_full, basis_trans)
    int2e_full = numpy.einsum("pqkl,kr->pqrl", int2e_full, basis_trans.conj())
    int2e_full = numpy.einsum("pqrl,ls->pqrs", int2e_full, basis_trans)

    int2e_full = int2e_full.real
    h1e_adapted = h1e_adapted.real
    energy_core = mol.get_enuc()
    
    nmo = my_scf.mo_coeff.shape[1]
    nelec = mol.nelectron
    ms = 0
    tol = 1e-10
    nuc = energy_core
    float_format = tools.fcidump.DEFAULT_FLOAT_FORMAT

    with open(filename, 'w') as fout: # 4-fold symmetry
        tools.fcidump.write_head(fout, nmo, nelec, ms, orbsym_ID)
        output_format = float_format + ' %4d %4d %4d %4d\n'
        for i in range(nmo):
            for j in range(i+1):
                for k in range(i+1):
                    if i>k:
                        for l in range(i+1):
                            if abs(int2e_full[i][j][k][l]) > tol:
                                fout.write(output_format % (int2e_full[i][j][k][l], i+1, j+1, k+1, l+1))
                    else:
                        for l in range(j+1):
                            if abs(int2e_full[i][j][k][l]) > tol:
                                fout.write(output_format % (int2e_full[i][j][k][l], i+1, j+1, k+1, l+1))

        tools.fcidump.write_hcore(fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)


if __name__ == "__main__":
    mol = pyscf.gto.M(
        verbose=4,
        atom='''
C   0.000000000000       0.000000000000      -0.621265
O   0.000000000000       0.000000000000       0.621265
''',
        basis={'C': 'cc-pvdz', 'O': 'cc-pvdz'},
        spin=0,
        charge=0,
        symmetry='coov',
    )
    mol.build()

    my_scf = Driver_SCF.Run_SCF(mol)
    Driver_SCF.Analysis_SCF(mol, my_scf)

    orbsym_ID, orbsym = Util_Mole.get_orbsym(mol, my_scf.mo_coeff)

    print(orbsym_ID)
    print(orbsym)

    basis_trans, Lz = get_symmetry_adapted_basis_Coov(
        mol, my_scf.mo_coeff)

    print(Lz)
    # print(parity)

    print(basis_trans)

    h1e = reduce(numpy.dot, (my_scf.mo_coeff.T,
                 my_scf.get_hcore(), my_scf.mo_coeff))

    for i in range(mol.nao):
        for j in range(mol.nao):
            if abs(h1e[i, j]) > 1e-10:
                print(i, j, h1e[i, j])

    h1e_adapted = reduce(numpy.dot, (basis_trans.H,
                                     h1e, basis_trans))

    for i in range(mol.nao):
        for j in range(mol.nao):
            if abs(h1e_adapted[i, j]) > 1e-10:
                print(i, j, h1e_adapted[i, j], Lz[i], Lz[j])
                print(i, j, h1e[i,j])
                assert(Lz[i] == Lz[j])
                # assert(parity[i] == parity[j])

    int2e_full = pyscf.ao2mo.full(
        eri_or_mol=mol, mo_coeff=my_scf.mo_coeff, aosym='1').reshape((mol.nao, mol.nao, mol.nao, mol.nao))

    print(int2e_full.shape)

    # (1*1|2*2)

    int2e_full = numpy.einsum("ijkl,ip->pjkl", int2e_full, basis_trans.conj())
    int2e_full = numpy.einsum("pjkl,jq->pqkl", int2e_full, basis_trans)
    int2e_full = numpy.einsum("pqkl,kr->pqrl", int2e_full, basis_trans.conj())
    int2e_full = numpy.einsum("pqrl,ls->pqrs", int2e_full, basis_trans)

    # print(int2e_full)

    for i in range(mol.nao):
        for j in range(mol.nao):
            for k in range(mol.nao):
                for l in range(mol.nao):
                    if abs(int2e_full[i, j, k, l]) > 1e-10:
                        # real integrals
                        assert(abs(int2e_full[i, j, k, l].imag) < 1e-10)
                        assert(abs(Lz[i]+Lz[k]-Lz[j]-Lz[l])
                               < 1e-10)  # conservation of Lz
                        # assert(parity[i]*parity[j]*parity[k]*parity[l]==1)

    int2e_full = int2e_full.real
    h1e_adapted = h1e_adapted.real
    energy_core = mol.get_enuc()

    tools.fcidump.from_integrals(filename="FCIDUMP_CO",
                                 h1e=h1e_adapted,
                                 h2e=int2e_full,
                                 nuc=energy_core,
                                 nmo=my_scf.mo_coeff.shape[1],
                                 nelec=mol.nelectron, 
                                 tol=1e-10,
                                 orbsym=orbsym_ID)
    
    filename = "FCIDUMP_CO_FULL"
    nmo = my_scf.mo_coeff.shape[1]
    nelec = mol.nelectron
    ms = 0
    tol = 1e-10
    nuc = energy_core
    float_format = tools.fcidump.DEFAULT_FLOAT_FORMAT
    with open(filename, 'w') as fout:
        tools.fcidump.write_head(fout, nmo, nelec, ms, orbsym_ID)
        output_format = float_format + ' %4d %4d %4d %4d\n'
        for i in range(nmo):
            for j in range(i+1):
                for k in range(i+1):
                    if i>k:
                        for l in range(i+1):
                            if abs(int2e_full[i][j][k][l]) > tol:
                                fout.write(output_format % (int2e_full[i][j][k][l], i+1, j+1, k+1, l+1))
                    else:
                        for l in range(j+1):
                            if abs(int2e_full[i][j][k][l]) > tol:
                                fout.write(output_format % (int2e_full[i][j][k][l], i+1, j+1, k+1, l+1))

        tools.fcidump.write_hcore(fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)

