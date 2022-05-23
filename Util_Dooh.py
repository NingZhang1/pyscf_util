import pyscf
import numpy
import Driver_SCF
import Util_Mole
from functools import reduce

A1g_ID = 0
A2g_ID = 1
A1u_ID = 5
A2u_ID = 4
E1gx_ID = 2
E1gy_ID = 3
E1ux_ID = 7
E1uy_ID = 6
E2gx_ID = 10
E2gy_ID = 11
E2ux_ID = 15
E2uy_ID = 14
E3gx_ID = 12
E3gy_ID = 13
E3ux_ID = 17
E3uy_ID = 16
E4gx_ID = 20
E4gy_ID = 21
E4ux_ID = 25
E4uy_ID = 24
E5gx_ID = 22
E5gy_ID = 23
E5ux_ID = 27
E5uy_ID = 26

E1g_up_ID = 2
E1g_dn_ID = 3
E1u_up_ID = 7
E1u_dn_ID = 6
E2g_up_ID = 10
E2g_dn_ID = 11
E2u_up_ID = 15
E2u_dn_ID = 14
E3g_up_ID = 12
E3g_dn_ID = 13
E3u_up_ID = 17
E3u_dn_ID = 16
E4g_up_ID = 20
E4g_dn_ID = 21
E4u_up_ID = 25
E4u_dn_ID = 24
E5g_up_ID = 22
E5g_dn_ID = 23
E5u_up_ID = 27
E5u_dn_ID = 26


def get_symmetry_adapted_basis_Dooh(mol, coeff):
    orbsym_ID, _ = Util_Mole.get_orbsym(mol, coeff)
    return _get_symmetry_adapted_basis_Dooh(orbsym_ID)


def _get_symmetry_adapted_basis_Dooh(orbsym_ID):

    # orbsym_ID, _ = Util_Mole.get_orbsym(mol, coeff)

    E1g_x = [i for i, x in enumerate(orbsym_ID) if x is E1gx_ID]
    E1g_y = [i for i, x in enumerate(orbsym_ID) if x is E1gy_ID]
    E1u_x = [i for i, x in enumerate(orbsym_ID) if x is E1ux_ID]
    E1u_y = [i for i, x in enumerate(orbsym_ID) if x is E1uy_ID]

    E2g_x = [i for i, x in enumerate(orbsym_ID) if x is E2gx_ID]
    E2g_y = [i for i, x in enumerate(orbsym_ID) if x is E2gy_ID]
    E2u_x = [i for i, x in enumerate(orbsym_ID) if x is E2ux_ID]
    E2u_y = [i for i, x in enumerate(orbsym_ID) if x is E2uy_ID]

    E3g_x = [i for i, x in enumerate(orbsym_ID) if x is E3gx_ID]
    E3g_y = [i for i, x in enumerate(orbsym_ID) if x is E3gy_ID]
    E3u_x = [i for i, x in enumerate(orbsym_ID) if x is E3ux_ID]
    E3u_y = [i for i, x in enumerate(orbsym_ID) if x is E3uy_ID]

    E4g_x = [i for i, x in enumerate(orbsym_ID) if x is E4gx_ID]
    E4g_y = [i for i, x in enumerate(orbsym_ID) if x is E4gy_ID]
    E4u_x = [i for i, x in enumerate(orbsym_ID) if x is E4ux_ID]
    E4u_y = [i for i, x in enumerate(orbsym_ID) if x is E4uy_ID]

    E5g_x = [i for i, x in enumerate(orbsym_ID) if x is E5gx_ID]
    E5g_y = [i for i, x in enumerate(orbsym_ID) if x is E5gy_ID]
    E5u_x = [i for i, x in enumerate(orbsym_ID) if x is E5ux_ID]
    E5u_y = [i for i, x in enumerate(orbsym_ID) if x is E5uy_ID]

    factor = 1.0/numpy.sqrt(2)

    basis_trans = numpy.identity(len(orbsym_ID), dtype=numpy.complex128)

    Lz = numpy.zeros(len(orbsym_ID), dtype=numpy.int)

    Parity = numpy.ones(len(orbsym_ID), dtype=numpy.int)

    for orbx, orby in zip(E1g_x, E1g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 1
        Lz[orby] = -1

    for orbx, orby in zip(E1u_x, E1u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 1
        Lz[orby] = -1
        Parity[orbx] = -1
        Parity[orby] = -1

    for orbx, orby in zip(E2g_x, E2g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 2
        Lz[orby] = -2

    for orbx, orby in zip(E2u_x, E2u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 2
        Lz[orby] = -2
        Parity[orbx] = -1
        Parity[orby] = -1

    for orbx, orby in zip(E3g_x, E3g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 3
        Lz[orby] = -3

    for orbx, orby in zip(E3u_x, E3u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 3
        Lz[orby] = -3
        Parity[orbx] = -1
        Parity[orby] = -1

    for orbx, orby in zip(E4g_x, E4g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 4
        Lz[orby] = -4

    for orbx, orby in zip(E4u_x, E4u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 4
        Lz[orby] = -4
        Parity[orbx] = -1
        Parity[orby] = -1

    for orbx, orby in zip(E5g_x, E5g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 5
        Lz[orby] = -5

    for orbx, orby in zip(E5u_x, E5u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 5
        Lz[orby] = -5
        Parity[orbx] = -1
        Parity[orby] = -1

    return numpy.matrix(basis_trans), Lz, Parity


def _get_time_reversal_half_pair(orbsym_ID):

    E1g_up = [i for i, x in enumerate(orbsym_ID) if x is E1g_up_ID]
    E1u_up = [i for i, x in enumerate(orbsym_ID) if x is E1u_up_ID]
    E2g_up = [i for i, x in enumerate(orbsym_ID) if x is E2g_up_ID]
    E2u_up = [i for i, x in enumerate(orbsym_ID) if x is E2u_up_ID]
    E3g_up = [i for i, x in enumerate(orbsym_ID) if x is E3g_up_ID]
    E3u_up = [i for i, x in enumerate(orbsym_ID) if x is E3u_up_ID]
    E4g_up = [i for i, x in enumerate(orbsym_ID) if x is E4g_up_ID]
    E4u_up = [i for i, x in enumerate(orbsym_ID) if x is E4u_up_ID]
    E5g_up = [i for i, x in enumerate(orbsym_ID) if x is E5g_up_ID]
    E5u_up = [i for i, x in enumerate(orbsym_ID) if x is E5u_up_ID]

    Res = []
    Res.extend(E1g_up)
    Res.extend(E2g_up)
    Res.extend(E3g_up)
    Res.extend(E4g_up)
    Res.extend(E5g_up)
    Res.extend(E1u_up)
    Res.extend(E2u_up)
    Res.extend(E3u_up)
    Res.extend(E4u_up)
    Res.extend(E5u_up)

    return Res


def _get_time_reversal_pair(orbsym_ID):

    E1g_up = [i for i, x in enumerate(orbsym_ID) if x is E1g_up_ID]
    E1g_dn = [i for i, x in enumerate(orbsym_ID) if x is E1g_dn_ID]
    E1u_up = [i for i, x in enumerate(orbsym_ID) if x is E1u_up_ID]
    E1u_dn = [i for i, x in enumerate(orbsym_ID) if x is E1u_dn_ID]

    E2g_up = [i for i, x in enumerate(orbsym_ID) if x is E2g_up_ID]
    E2g_dn = [i for i, x in enumerate(orbsym_ID) if x is E2g_dn_ID]
    E2u_up = [i for i, x in enumerate(orbsym_ID) if x is E2u_up_ID]
    E2u_dn = [i for i, x in enumerate(orbsym_ID) if x is E2u_dn_ID]

    E3g_up = [i for i, x in enumerate(orbsym_ID) if x is E3g_up_ID]
    E3g_dn = [i for i, x in enumerate(orbsym_ID) if x is E3g_dn_ID]
    E3u_up = [i for i, x in enumerate(orbsym_ID) if x is E3u_up_ID]
    E3u_dn = [i for i, x in enumerate(orbsym_ID) if x is E3u_dn_ID]

    E4g_up = [i for i, x in enumerate(orbsym_ID) if x is E4g_up_ID]
    E4g_dn = [i for i, x in enumerate(orbsym_ID) if x is E4g_dn_ID]
    E4u_up = [i for i, x in enumerate(orbsym_ID) if x is E4u_up_ID]
    E4u_dn = [i for i, x in enumerate(orbsym_ID) if x is E4u_dn_ID]

    E5g_up = [i for i, x in enumerate(orbsym_ID) if x is E5g_up_ID]
    E5g_dn = [i for i, x in enumerate(orbsym_ID) if x is E5g_dn_ID]
    E5u_up = [i for i, x in enumerate(orbsym_ID) if x is E5u_up_ID]
    E5u_dn = [i for i, x in enumerate(orbsym_ID) if x is E5u_dn_ID]

    Res = range(0, len(orbsym_ID))

    for i in range(len(E1g_up)):
        up_ID = E1g_up[i]
        dn_ID = E1g_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E1u_up)):
        up_ID = E1u_up[i]
        dn_ID = E1u_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E2g_up)):
        up_ID = E2g_up[i]
        dn_ID = E2g_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E2u_up)):
        up_ID = E2u_up[i]
        dn_ID = E2u_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E3g_up)):
        up_ID = E3g_up[i]
        dn_ID = E3g_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E3u_up)):
        up_ID = E3u_up[i]
        dn_ID = E3u_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E4g_up)):
        up_ID = E4g_up[i]
        dn_ID = E4g_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E4u_up)):
        up_ID = E4u_up[i]
        dn_ID = E4u_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E5g_up)):
        up_ID = E5g_up[i]
        dn_ID = E5g_dn[i]
        Res[up_ID] = dn_ID
        Res[dn_ID] = up_ID

    for i in range(len(E5u_up)):
        up_ID = E5u_up[i]
        dn_ID = E5u_dn[i]
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

    basis_trans, _, _ = _get_symmetry_adapted_basis_Dooh(orbsym_ID)
    basis_trans = basis_trans.H
    basis_trans = basis_trans[begin_orb:end_orb, begin_orb:end_orb]

    Res = rdm1
    Res = numpy.einsum("ij,ip->pj", Res, basis_trans.conj())
    Res = numpy.einsum("pj,jq->pq", Res, basis_trans)

    return Res


def tranform_rdm1_xy_2_adapted(orbsym_ID, rdm1, begin_orb, end_orb):

    time_reversal_pair = _get_time_reversal_pair(orbsym_ID)
    _check_orb_range_valid(time_reversal_pair, begin_orb, end_orb)

    basis_trans, _, _ = _get_symmetry_adapted_basis_Dooh(orbsym_ID)
    basis_trans = basis_trans[begin_orb:end_orb, begin_orb:end_orb]

    Res = rdm1
    Res = numpy.einsum("ij,ip->pj", Res, basis_trans.conj())
    Res = numpy.einsum("pj,jq->pq", Res, basis_trans)

    return Res


def tranform_rdm2_adapted_2_xy(orbsym_ID, rdm2, begin_orb, end_orb):

    time_reversal_pair = _get_time_reversal_pair(orbsym_ID)
    _check_orb_range_valid(time_reversal_pair, begin_orb, end_orb)

    basis_trans, _, _ = _get_symmetry_adapted_basis_Dooh(orbsym_ID)
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

    basis_trans, _, _ = _get_symmetry_adapted_basis_Dooh(orbsym_ID)
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

    _, Lz, Parity = _get_symmetry_adapted_basis_Dooh(orbsym_ID)

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


if __name__ == "__main__":
    mol = pyscf.gto.M(
        verbose=4,
        atom='''
C   0.000000000000       0.000000000000      -0.625265
C   0.000000000000       0.000000000000       0.625265
''',
        basis={'C': 'ccpvtz', 'H': 'ccpvdz'},
        spin=0,
        charge=0,
        symmetry='dooh',
    )
    mol.build()

    my_scf = Driver_SCF.Run_SCF(mol)
    Driver_SCF.Analysis_SCF(mol, my_scf)

    orbsym_ID, orbsym = Util_Mole.get_orbsym(mol, my_scf.mo_coeff)

    # print([i for i, x in enumerate(orbsym_ID) if x is 2])
    # print([i for i, x in enumerate(orbsym_ID) if x is 7])
    # print([i for i, x in enumerate(orbsym_ID) if x is 50])
    # print([i for i, x in enumerate(orbsym_ID) if x is 55])

    basis_trans, Lz, parity = get_symmetry_adapted_basis_Dooh(
        mol, my_scf.mo_coeff)

    print(Lz)
    print(parity)

    h5e = reduce(numpy.dot, (my_scf.mo_coeff.T,
                 my_scf.get_hcore(), my_scf.mo_coeff))

    for i in range(mol.nao):
        for j in range(mol.nao):
            if abs(h5e[i, j]) > 1e-10:
                print(i, j, h5e[i, j])

    h5e_adapted = reduce(numpy.dot, (basis_trans.H,
                                     h5e, basis_trans))

    for i in range(mol.nao):
        for j in range(mol.nao):
            if abs(h5e_adapted[i, j]) > 1e-10:
                print(i, j, h5e_adapted[i, j])

    int2e_full = pyscf.ao2mo.full(
        eri_or_mol=mol, mo_coeff=my_scf.mo_coeff, aosym='1').reshape((mol.nao, mol.nao, mol.nao, mol.nao))

    # for i in range(mol.nao):
    #     for j in range(mol.nao):
    #         for k in range(mol.nao):
    #             for l in range(mol.nao):
    #                 if abs(int2e_full[i, j, k, l]) > 1e-10:
    #                     pass
    #                     # print(i, j, k, l, int2e_full[i, j, k, l])

    print(int2e_full.shape)

    # (1*1|2*2)

    # int2e_full = numpy.einsum("ijkl,ip->pjkl", int2e_full, basis_trans.conj())
    # int2e_full = numpy.einsum("pjkl,jq->pqkl", int2e_full, basis_trans.conj())
    # int2e_full = numpy.einsum("pqkl,kr->pqrl", int2e_full, basis_trans)
    # int2e_full = numpy.einsum("pqrl,ls->pqrs", int2e_full, basis_trans)
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
                        # print(i, j, k, l,
                        #       int2e_full[i, j, k, l].real)
                        # real integrals
                        assert(abs(int2e_full[i, j, k, l].imag) < 1e-10)
                        assert(abs(Lz[i]+Lz[k]-Lz[j]-Lz[l])
                               < 1e-10)  # conservation of Lz

    # print(h5e_adapted)
    # print(numpy.dot(basis_trans.H, basis_trans))
