import functools
import pyscf
from pyscf import gto, symm, scf, lo
import numpy
import Util_Math
import Util_Mole
import copy
from functools import reduce


def pairing_orb_diatomic_mole(Mol, orb, bond_begin, bond_end):
    nao = Mol.nao
    nao_atom = Mol.nao // 2
    atm_ovlp = Mol.intor("int1e_ovlp")[:nao_atom, :nao_atom]
    res = []

    irrep_ids = symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb, orb)

    print(irrep_ids)

    # print(bond_begin,bond_end)
    for bond_id in range(bond_begin, bond_end):
        orb_coeff = orb[:, bond_id:bond_id+1].reshape(-1)
        id_pair = -1
        norm_pair = 100.0
        atm_coeff_1 = orb_coeff[:nao_atom]
        atm_coeff_2 = orb_coeff[nao_atom:]
        # for pair_id in range(0, nao):
        for pair_id in range(bond_begin, bond_end):
            print(bond_id, irrep_ids[bond_id], pair_id, irrep_ids[pair_id])
            # symmetry allowed!
            if (bond_id == pair_id) or (irrep_ids[bond_id] != irrep_ids[pair_id]):
                continue
            orb_pair_coeff = orb[:, pair_id:pair_id+1].reshape(-1)
            pair_coeff_1 = orb_pair_coeff[:nao_atom]
            pair_coeff_2 = orb_pair_coeff[nao_atom:]
            # ovlp1 = abs(reduce(numpy.dot,(atm_coeff_1.T,atm_ovlp,pair_coeff_1)))
            # ovlp2 = abs(reduce(numpy.dot,(atm_coeff_2.T,atm_ovlp,pair_coeff_2)))
            atm1_plus = atm_coeff_1+pair_coeff_1
            atm1_plus_norm = reduce(
                numpy.dot, (atm1_plus.T, atm_ovlp, atm1_plus))
            atm1_minus = atm_coeff_1-pair_coeff_1
            atm1_minus_norm = reduce(
                numpy.dot, (atm1_minus.T, atm_ovlp, atm1_minus))
            atm2_plus = atm_coeff_2+pair_coeff_2
            atm2_plus_norm = reduce(
                numpy.dot, (atm2_plus.T, atm_ovlp, atm2_plus))
            atm2_minus = atm_coeff_2-pair_coeff_2
            atm2_minus_norm = reduce(
                numpy.dot, (atm2_minus.T, atm_ovlp, atm2_minus))
            # norm1 = min(numpy.linalg.norm(atm_coeff_1+pair_coeff_1),
            #             numpy.linalg.norm(atm_coeff_1-pair_coeff_1))
            norm1 = numpy.sqrt(min(atm1_plus_norm, atm1_minus_norm))
            # norm2 = min(numpy.linalg.norm(atm_coeff_2+pair_coeff_2),
            #             numpy.linalg.norm(atm_coeff_2-pair_coeff_2))
            norm2 = numpy.sqrt(min(atm2_plus_norm, atm2_minus_norm))
            norm = norm1+norm2
            if norm < norm_pair:
                norm_pair = norm
                id_pair = pair_id

        # if (id_pair < bond_id):
        #     res.append([id_pair, bond_id])
        res.append([bond_id, id_pair])
        print("pair %d %d %f" % (bond_id, id_pair, norm_pair))
        # print(orb_coeff)
        # print(orb[:, id_pair])

    return res


def loc_mo_given_pairing(orb, pairing):  # only for dooh
    Res = copy.deepcopy(orb)
    for pair in pairing:
        orb1 = Res[:, pair[0]]
        orb2 = Res[:, pair[1]]
        orb1_new = (orb1+orb2)*numpy.sqrt(0.5)
        orb2_new = (orb1-orb2)*numpy.sqrt(0.5)
        Res[:, pair[0]] = orb1_new
        Res[:, pair[1]] = orb2_new
    return Res


def get_stat_irrep(Mol, orb, begin_indx, end_indx):

    act_orb = orb[:, begin_indx:end_indx]
    print("begindx ", begin_indx, " end_indx ", end_indx)
    irrep_ids = symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb, act_orb)

    stat_irrep = {}

    for id, irrep in enumerate(irrep_ids):
        if irrep not in stat_irrep.keys():
            stat_irrep[irrep] = []
        stat_irrep[irrep].append(id+begin_indx)

    return stat_irrep


def split_loc_given_range(Mol, orb, begin_indx, end_indx, loc_method=lo.Boys):
    stat_irrep = get_stat_irrep(Mol, orb, begin_indx, end_indx)
    print(stat_irrep)
    Res = copy.deepcopy(orb)
    for irrep in stat_irrep.keys():
        orb_indx = stat_irrep[irrep]
        orb_tmp = Res[:, orb_indx]
        loc_orb = loc_method(Mol, orb_tmp).kernel()
        Res[:, orb_indx] = loc_orb
    return Res


def split_loc_given_range_NoSymm(Mol, orb, begin_indx, end_indx, loc_method=lo.Boys):
    # stat_irrep = get_stat_irrep(Mol, orb, begin_indx, end_indx)
    # print(stat_irrep)
    Res = copy.deepcopy(orb)
    # for irrep in stat_irrep.keys():
    orb_tmp = Res[:, begin_indx:end_indx]
    loc_orb = loc_method(Mol, orb_tmp).kernel()
    Res[:, begin_indx:end_indx] = loc_orb
    return Res


def get_orbsym_act_given_atmorb(Mol, atm_indx):
    indx = Mol.search_ao_label(atm_indx)

    init_mo = numpy.zeros((Mol.nao, len(indx)))
    for loc, id in enumerate(indx):
        init_mo[id, loc] = 1.0

    init_mo = Util_Math._orthogonalize(init_mo, Mol.intor("int1e_ovlp"))
    act_orb = symm.symmetrize_space(Mol, init_mo)
    irrep_ids = symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb, act_orb)
    # print('Occupied orbital symmetry: %s' % irrep_ids)

    stat_irrep = {}

    for irrep in irrep_ids:
        if irrep not in stat_irrep.keys():
            stat_irrep[irrep] = 0
        stat_irrep[irrep] += 1

    return stat_irrep


def reorder_BDF_orb(Mol, mo_coeff, mo_energy, mo_occ, NFZC, NACT, NVIR, check=False):

    stat_irrep = get_stat_irrep(Mol, mo_coeff, 0, Mol.nao)
    order = []

    for key in stat_irrep.keys():
        order.extend(stat_irrep[key][:NFZC[key]])
    nfzc = len(order)
    for key in stat_irrep.keys():
        order.extend(stat_irrep[key][NFZC[key]:NFZC[key]+NACT[key]])
    nact = len(order) - nfzc
    for key in stat_irrep.keys():
        order.extend(stat_irrep[key][NFZC[key]+NACT[key]
                     :NFZC[key]+NACT[key]+NVIR[key]])
    nvir = len(order) - nact - nfzc

    def takeSecond(elem):
        return elem[1]

    fzc_frag = list(zip(order[:nfzc], mo_energy[order][:nfzc]))
    fzc_frag.sort(key=takeSecond)
    act_frag = list(zip(order[nfzc:nfzc+nact], mo_occ[order][nfzc:nfzc+nact]))
    act_frag.sort(key=takeSecond, reverse=True)
    vir_frag = list(zip(order[nfzc+nact:], mo_energy[order][nfzc+nact:]))
    vir_frag.sort(key=takeSecond)

    if check:
        print(mo_energy[order][:nfzc])
        print(mo_energy[order][nfzc:nfzc+nact])
        print(mo_energy[order][nfzc+nact:])
        print(mo_occ[order][:nfzc])
        print(mo_occ[order][nfzc:nfzc+nact])
        print(mo_occ[order][nfzc+nact:])

        print(fzc_frag)
        print(act_frag)
        print(vir_frag)

    order = []
    for x in fzc_frag:
        order.append(x[0])
    for x in act_frag:
        order.append(x[0])
    for x in vir_frag:
        order.append(x[0])

    return order, mo_coeff[:, order], nfzc, nact, nvir


def _add_new_pair(Mol, mo_coeff, split_region, pairing):

    paired_tmp = []
    for i in range(mo_coeff.shape[1]):
        paired_tmp.append(False)

    for pair in pairing:
        paired_tmp[pair[0]] = True
        paired_tmp[pair[1]] = True

    Res = copy.deepcopy(pairing)

    for i in range(len(split_region)-1):
        begin_indx = split_region[i]
        end_indx = split_region[i+1]

        stat_irrep = get_stat_irrep(Mol, mo_coeff, begin_indx, end_indx)

        for irrep in stat_irrep.keys():
            orb_unpaired = []
            orb_indx = stat_irrep[irrep]
            for orb in orb_indx:
                if paired_tmp[orb] == False:
                    orb_unpaired.append(orb)
            if len(orb_unpaired) == 2:
                print(orb_unpaired[0], orb_unpaired[1], "is added!\n")
                Res.append([orb_unpaired[0], orb_unpaired[1]])

    return Res


#  localize dooh orb

def localize_diatomic_orb(Mol, mo_coeff, split_region, group_orb=True, molden_filename=None):

    # print(get_stat_irrep(Mol, mo_coeff, 0, Mol.nao))
    stat_irrep = get_stat_irrep(Mol, mo_coeff, 0, Mol.nao)

    # pairing = pairing_orb_diatomic_mole(
    #     Mol, mo_coeff, split_region[0], split_region[-1])

    pairing = []
    for i in range(len(split_region)-1):
        pairing.extend(pairing_orb_diatomic_mole(
            Mol, mo_coeff, split_region[i], split_region[i+1]))
    nao_atm = Mol.nao//2

    pairing.sort()
    print(pairing)

    # return

    # check pairing

    pairing_res = []
    symmetry = []

    paired = []
    for i in range(Mol.nao):
        paired.append(False)
        symmetry.append('A1')

    for key in stat_irrep.keys():
        for id in stat_irrep[key]:
            symmetry[id] = key

    for pair in pairing:
        orb_i = pair[0]
        orb_j = pair[1]
        if (pairing[orb_j][1] != orb_i) or (symmetry[orb_i] != symmetry[orb_j]):
            print(pair, " is abandon.")
        else:
            paired[orb_i] = True
            paired[orb_j] = True
            if orb_i < orb_j:
                pairing_res.append(pair)

    print(pairing_res)
    # pairing = pairing_res
    pairing = _add_new_pair(Mol, mo_coeff, split_region, pairing_res)
    print(pairing)

    # localize orb

    orb_pair_loc = loc_mo_given_pairing(mo_coeff, pairing)

    # reorder

    loc_res = copy.deepcopy(orb_pair_loc)

    n_paired_per_region = []

    for i in range(len(split_region)-1):
        paired_loc = split_region[i]
        unpaired_loc = split_region[i+1]-1
        n_paired = 0
        for j in range(split_region[i], split_region[i+1]):
            if paired[j]:
                loc_res[:, paired_loc] = orb_pair_loc[:, j]
                paired_loc += 1
                n_paired += 1
            else:
                loc_res[:, unpaired_loc] = orb_pair_loc[:, j]
                unpaired_loc -= 1
        n_paired_per_region.append(n_paired)
        print(i, n_paired)

    orb_pair_loc = loc_res

    # localize

    for i in range(len(split_region)-1):
        orb_pair_loc = split_loc_given_range(
            Mol, orb_pair_loc, split_region[i], split_region[i+1])

    loc_res = copy.deepcopy(orb_pair_loc)

    if group_orb:

        # loc_res = numpy.zeros((Mol.nao, Mol.nao))

        for i in range(len(split_region)-1):
            # orb_num = split_region[i+1] - split_region[i]
            orb_num = n_paired_per_region[i]
            if (orb_num % 2) != 0:
                print("Fatal Error ", orb_num,
                      " is not even")
                return None
            offset_A = split_region[i]
            offset_B = split_region[i] + orb_num // 2

            print("n_paired_per_region of ", i, " is equal to ", orb_num)

            for j in range(split_region[i], split_region[i]+orb_num):
                norm_A = numpy.linalg.norm(orb_pair_loc[:nao_atm, j])
                norm_B = numpy.linalg.norm(orb_pair_loc[nao_atm:, j])
                print("norm_A %15.8f norm_B %15.8f" % (norm_A, norm_B))
                if norm_A > norm_B:
                    loc_res[:, offset_A] = orb_pair_loc[:, j]
                    offset_A += 1
                else:
                    loc_res[:, offset_B] = orb_pair_loc[:, j]
                    offset_B += 1

    if molden_filename is not None:
        pyscf.tools.molden.from_mo(Mol, molden_filename, loc_res)

    return loc_res


def get_macrocfg(criterion_array, tol=1e-4, rela_tol=1e-4):

    criterion_now = criterion_array[0]
    norb_group = 1

    Res = [1]
    nmo = len(criterion_array)

    for i in range(1, nmo):

        new_group = True

        # print(criterion_array[i], criterion_now, abs(
        #     (criterion_array[i]-criterion_now)/(criterion_now)))

        if (abs(criterion_array[i]-criterion_now) < tol) and ((abs((criterion_array[i]-criterion_now)/(criterion_now))) < rela_tol):
            new_group = False

        if new_group:
            criterion_now = criterion_array[i]
            norb_group += 1
            Res.append(1)
        else:
            Res[norb_group-1] += 1

    return Res

# 对称性


Pair = {
    'IR0': None,
    'IR1': None,
    'IR2': None,
    'IR3': None,
    'IR4': None,
    'IR5': None,
    'IR6': None,
    'IR7': None,
    'A1g': None,
    'A1u': None,
    'E1gx': 'E1gy',
    'E1gy': 'E1gx',
    'E1uy': 'E1ux',
    'E1ux': 'E1uy',
    'E2gx': 'E2gy',
    'E2gy': 'E2gx',
    'E2uy': 'E2ux',
    'E2ux': 'E2uy',
    'E3gx': 'E3gy',
    'E3gy': 'E3gx',
    'E3uy': 'E3ux',
    'E3ux': 'E3uy',
    'E4gx': 'E4gy',
    'E4gy': 'E4gx',
    'E4uy': 'E4ux',
    'E4ux': 'E4uy',
    'E5gx': 'E5gy',
    'E5gy': 'E5gx',
    'E5uy': 'E5ux',
    'E5ux': 'E5uy',
    'E6gx': 'E6gy',
    'E6gy': 'E6gx',
    'E6uy': 'E6ux',
    'E6ux': 'E6uy', }


def OrbSymInfo(Mol, cmoao, s=None):
    OrbSym = pyscf.symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb,
                                       cmoao, s)
    return OrbSym


def sort_rdm1(Mol, Occ, cmoao, s=None):
    assert (len(Occ) == cmoao.shape[1])

    orbsym = OrbSymInfo(Mol, cmoao, s)

    tmp = []

    def cmpr(A, B):
        if abs(A[0] - B[0]) > 1e-8:
            if A[0] < B[0]:
                return 1
            else:
                return -1
        if A[1] < B[1]:
            return 1
        else:
            return -1

    for i in range(len(Occ)):
        tmp.append([Occ[i], orbsym[i], cmoao[:, i]])

    tmp.sort(reverse=False, key=functools.cmp_to_key(cmpr))

    Res = numpy.zeros(cmoao.shape)
    OccRes = numpy.zeros(len(Occ))

    for id, data in enumerate(tmp):
        Res[:, id] = data[2]
        OccRes[id] = data[0]

    return OccRes, Res


def sort_rdm1_given_orbsym(orbsym, Occ, cmoao):
    assert (len(Occ) == cmoao.shape[1])

    tmp = []

    def cmpr(A, B):
        if abs(A[0] - B[0]) > 1e-8:
            if A[0] < B[0]:
                return 1
            else:
                return -1
        if A[1] < B[1]:
            return 1
        else:
            return -1

    for i in range(len(Occ)):
        tmp.append([Occ[i], orbsym[i], cmoao[:, i]])

    tmp.sort(reverse=False, key=functools.cmp_to_key(cmpr))

    Res = numpy.zeros(cmoao.shape)
    OccRes = numpy.zeros(len(Occ))

    for id, data in enumerate(tmp):
        Res[:, id] = data[2]
        OccRes[id] = data[0]
        orbsym[id] = data[1]

    return OccRes, Res, orbsym


def sort_rdm1_with_sym(Mol, Occ, cmoao, s=None):
    assert (len(Occ) == cmoao.shape[1])

    orbsym = OrbSymInfo(Mol, cmoao, s)

    tmp = []

    for i in range(len(Occ)):
        tmp.append([orbsym[i], Occ[i], cmoao[:, i]])

    def cmpr(A, B):
        if A[0] != B[0]:
            if A[0] < B[0]:
                return 1
            else:
                return -1
        if A[1] < B[1]:
            return 1
        else:
            return -1

    tmp.sort(reverse=False, key=functools.cmp_to_key(cmpr))

    Res = numpy.zeros(cmoao.shape)
    OccRes = numpy.zeros(len(Occ))

    for id, data in enumerate(tmp):
        Res[:, id] = data[2]
        OccRes[id] = data[1]

    return OccRes, Res


def sort_rdm1_with_sym_given_orbsym(orbsym, Occ, cmoao):
    assert (len(Occ) == cmoao.shape[1])

    tmp = []

    for i in range(len(Occ)):
        tmp.append([orbsym[i], Occ[i], cmoao[:, i]])

    def cmpr(A, B):
        if A[0] != B[0]:
            if A[0] < B[0]:
                return 1
            else:
                return -1
        if A[1] < B[1]:
            return 1
        else:
            return -1

    tmp.sort(reverse=False, key=functools.cmp_to_key(cmpr))

    Res = numpy.zeros(cmoao.shape)
    OccRes = numpy.zeros(len(Occ))

    for id, data in enumerate(tmp):
        Res[:, id] = data[2]
        OccRes[id] = data[1]

    return OccRes, Res


def _align_phase(indx1, indx2, rdm1, cmoao):
    assert (len(indx1) == len(indx2))
    rdm1_bench = rdm1[indx1, :][:, indx1]
    rdm2_align = rdm1[indx2, :][:, indx2]

    length = len(indx2)

    # determine which line

    bench_line = 0

    for i in range(length):
        if rdm1_bench[i, i] < 1.9999:
            bench_line = i
            break

    for i in range(length):
        if rdm1_bench[bench_line, i] * rdm2_align[bench_line, i] < 0.0:
            cmoao[:, indx2[i]] *= -1.0
            rdm1[:, indx2[i]] *= -1.0
            rdm1[indx2[i], :] *= -1.0
            rdm2_align = rdm1[indx2, :][:, indx2]

    for i in range(length):
        for j in range(length):
            if rdm1_bench[i, j] * rdm2_align[i, j] < 0:
                print(i, j)
                print(rdm1_bench)
                print(rdm2_align)
                exit(1)


SO3_IREEP = {
    's': ['s+0'],
    'p': ['p+0', 'p-1', 'p+1'],
    'd': ['d+0', 'd-2', 'd-1', 'd+1', 'd+2'],
    'f': ['f+0', 'f-3', 'f-2', 'f-1', 'f+1', 'f+2', 'f+3'],
    'g': ['g+0', 'g-4', 'g-3', 'g-2', 'g-1', 'g+1', 'g+2', 'g+3', 'g+4'],
    'h': ['h+0', 'h-5', 'h-4', 'h-3', 'h-2', 'h-1', 'h+1', 'h+2', 'h+3', 'h+4', 'h+5'],
    'i': ['i+0', 'i-6', 'i-5', 'i-4', 'i-3', 'i-2', 'i-1', 'i+1', 'i+2', 'i+3', 'i+4', 'i+5', 'i+6'],
}


def _diag_rdm1_SO3(mol, cmoao, rdm1, s=None):

    orbsym = pyscf.symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, cmoao)

    nmo = cmoao.shape[1]
    Occ = numpy.zeros(cmoao.shape[1])
    mo_trans = numpy.zeros((nmo, nmo))

    # align phase

    for l in ['p', 'd', 'f', 'g', 'h', 'i']:
        indx = numpy.where(orbsym == SO3_IREEP[l][0])[0]
        for i in range(1, len(SO3_IREEP[l])):
            indx2 = numpy.where(orbsym == SO3_IREEP[l][i])[0]
            _align_phase(indx, indx2, rdm1, cmoao)

    # diag rdm

    for l in ['s', 'p', 'd', 'f', 'g', 'h', 'i']:
        indx = numpy.where(orbsym == SO3_IREEP[l][0])[0]
        rdm_tmp = rdm1[indx, :][:, indx]
        for i in range(1, len(SO3_IREEP[l])):
            indx2 = numpy.where(orbsym == SO3_IREEP[l][i])[0]
            rdm_tmp += rdm1[indx2, :][:, indx2]
        rdm_tmp /= len(SO3_IREEP[l])

        e11, h11 = numpy.linalg.eigh(rdm_tmp)

        e = numpy.zeros(e11.shape)
        h = numpy.zeros(h11.shape)
        for i in range(len(e11)):
            e[i] = e11[len(e11)-1-i]
            h[:, i] = h11[:, len(e11)-1-i]
        e11 = e
        h11 = h

        for i in range(0, len(SO3_IREEP[l])):
            indx2 = numpy.where(orbsym == SO3_IREEP[l][i])[0]
            Occ[indx2] = e11
            for id2, irow in enumerate(indx2):
                mo_trans[irow][indx2] = h11[id2, :]

    return Occ, numpy.dot(cmoao, mo_trans)


def Diag_Rdm1_with_Sym(mol, cmoao, rdm1, s=None):
    assert (cmoao.shape[1] == rdm1.shape[1])

    if mol.groupname == 'SO3':
        return _diag_rdm1_SO3(mol, cmoao, rdm1, s)

    nmo = cmoao.shape[1]
    Occ = numpy.zeros(cmoao.shape[1])
    mo_trans = numpy.zeros((nmo, nmo))

    orbsym = OrbSymInfo(mol, cmoao, s)

    for irrep_name in mol.irrep_name:
        if Pair[irrep_name] is not None:
            orb_id = [i for i in range(nmo) if orbsym[i] == irrep_name]
            orb_id_pair = [i for i in range(
                nmo) if orbsym[i] == Pair[irrep_name]]
            rdm1_tmp1 = rdm1[:, orb_id]
            rdm1_tmp1 = rdm1_tmp1[orb_id, :]
            rdm1_tmp2 = rdm1[:, orb_id_pair]
            rdm1_tmp2 = rdm1_tmp2[orb_id_pair, :]
            rdm1_tmp0 = (rdm1_tmp1+rdm1_tmp2)/2
            e11, h11 = numpy.linalg.eigh(rdm1_tmp0)

            e = numpy.zeros(e11.shape)
            h = numpy.zeros(h11.shape)
            for i in range(len(e11)):
                e[i] = e11[len(e11)-1-i]
                h[:, i] = h11[:, len(e11)-1-i]
            e11 = e
            h11 = h

            Occ[orb_id] = e11
            Occ[orb_id_pair] = e11
            for id2, irow in enumerate(orb_id):
                mo_trans[irow][orb_id] = h11[id2, :]
            for id2, irow in enumerate(orb_id_pair):
                mo_trans[irow][orb_id_pair] = h11[id2, :]
        else:
            orb_id = [i for i in range(nmo) if orbsym[i] == irrep_name]
            rdm1_tmp0 = rdm1[:, orb_id]
            rdm1_tmp0 = rdm1_tmp0[orb_id, :]
            e11, h11 = numpy.linalg.eigh(rdm1_tmp0)

            e = numpy.zeros(e11.shape)
            h = numpy.zeros(h11.shape)
            for i in range(len(e11)):
                e[i] = e11[len(e11)-1-i]
                h[:, i] = h11[:, len(e11)-1-i]
            e11 = e
            h11 = h

            Occ[orb_id] = e11
            for id2, irow in enumerate(orb_id):
                mo_trans[irow][orb_id] = h11[id2, :]

    return Occ, numpy.dot(cmoao, mo_trans)


def Diag_Rdm1_with_Sym_given_orbsym(orbsym, cmoao, rdm1):
    assert (cmoao.shape[1] == rdm1.shape[1])

    nmo = cmoao.shape[1]
    Occ = numpy.zeros(cmoao.shape[1])
    mo_trans = numpy.zeros((nmo, nmo))

    for irrep_name in [0, 1, 2, 3, 4, 5, 6, 7]:

        orb_id = [i for i in range(nmo) if orbsym[i] == irrep_name]
        rdm1_tmp0 = rdm1[:, orb_id]
        rdm1_tmp0 = rdm1_tmp0[orb_id, :]
        e11, h11 = numpy.linalg.eigh(rdm1_tmp0)

        e = numpy.zeros(e11.shape)
        h = numpy.zeros(h11.shape)
        for i in range(len(e11)):
            e[i] = e11[len(e11)-1-i]
            h[:, i] = h11[:, len(e11)-1-i]
        e11 = e
        h11 = h

        Occ[orb_id] = e11
        for id2, irow in enumerate(orb_id):
            mo_trans[irow][orb_id] = h11[id2, :]

    return Occ, numpy.dot(cmoao, mo_trans)

# 分析轨道成分


# atm_indx is 1-BASED 

def _construct_atm_bas(mol, atm_bas_info, atm_label, atm_indx=None, orthogonalize=False):

    if atm_indx is not None:
        assert (atm_indx > 0)

    nao = mol.nao

    Res = numpy.zeros((nao, 0))

    nbas_tmp = 0

    indx_tmp = 0

    for orb_info in atm_label:
        tmp = orb_info.split(".")
        atm = tmp[0]
        orb = tmp[1]

        # print(atm, orb)

        loc_tmp = 0
        atm_bas_begin_loc = []

        for i in range(mol.natm):
            if mol.atom_pure_symbol(i) == atm:
                if atm_indx is None:
                    atm_bas_begin_loc.append(loc_tmp)
                else:
                    indx_tmp += 1
                    if indx_tmp == atm_indx:
                        atm_bas_begin_loc.append(loc_tmp)
            loc_tmp += atm_bas_info[mol.atom_pure_symbol(i)]["nao"]

        # print(atm_bas_begin_loc)

        nbas_added = len(atm_bas_begin_loc) * len(atm_bas_info[atm][orb])

        # print(nbas_added)

        atm_nao = atm_bas_info[atm]["nao"]

        # if Res is None:
        #     Res = numpy.zeros((nao, nbas_added))
        # else:

        ResTmp = copy.deepcopy(Res)
        Res = numpy.zeros((nao, nbas_tmp + nbas_added))
        Res[:, :nbas_tmp] = ResTmp

        for begin_loc in atm_bas_begin_loc:
            for colindx in atm_bas_info[atm][orb]:
                Res[begin_loc:begin_loc+atm_nao, nbas_tmp:nbas_tmp +
                    1] = atm_bas_info[atm]["cmoao"][:, colindx:colindx+1]
                nbas_tmp += 1

        if orthogonalize:
            Res = Util_Math._orthogonalize(Res, mol.intor("int1e_ovlp"))

    return Res


def _construct_orb_comp_analysis(mol, atm_bas_info, with_distinct_atm=False):

    if with_distinct_atm:
        Res = {}

        natm = {}

        for i in range(mol.natm):
            atm_label = mol.atom_pure_symbol(i)
            if atm_label not in natm.keys():
                natm[atm_label] = 1
            else:
                natm[atm_label] += 1

        for atm in atm_bas_info.keys():
            if atm in natm.keys():
                for id in range(1, natm[atm]+1):
                    for key in atm_bas_info[atm].keys():
                        if key not in ["nao", "cmoao", "basis"]:
                            Res["%s_%d.%s" % (atm, id, key)] = _construct_atm_bas(
                                mol, atm_bas_info, ["%s.%s" % (atm, key)], id, True)

    else:

        Res = {}

        for atm in atm_bas_info.keys():
            for key in atm_bas_info[atm].keys():
                if key not in ["nao", "cmoao", "basis"]:
                    Res["%s.%s" % (atm, key)] = _construct_atm_bas(
                        mol, atm_bas_info, ["%s.%s" % (atm, key)], True)

    return Res


def Analysis_Orb_Comp(mol, cmoao, indx_begin, indx_end, atm_bas_info, tol = 0.1, with_distinct_atm=False):

    ovlp = mol.intor("int1e_ovlp")

    orb_comp_analysis_bas = _construct_orb_comp_analysis(
        mol, atm_bas_info, with_distinct_atm)

    for i in range(indx_begin, indx_end):
        for key in orb_comp_analysis_bas.keys():
            norm = numpy.linalg.norm(reduce(
                numpy.dot, (cmoao[:, i:i+1].T, ovlp, orb_comp_analysis_bas[key])))
            if norm > tol:
                print("orbindx %3d Comp %s Contr %15.8f" % (i, key, norm*norm))
        print("--------------------------------------")

import Util_File

if __name__ == "__main__":


    atm_bas = {
    "H": {
        "1s": [0],
        "2s": [1],
        "2p": [2, 3, 4],
        "nao": 5,
        "basis": "ccpvdz-dk",
        "cmoao": None,
    },
    "C": {
        "1s": [0],
        "2s": [1],
        "2p": [2, 3, 4],
        "3p": [5, 6, 7],
        "3s": [8],
        "3d": [9, 10, 11, 12, 13],
        "nao": 14,
        "basis": "ccpvdz-dk",
        "cmoao": None,
    },
    "S": {
        "1s": [0],
        "2s": [1],
        "2p": [2, 3, 4],
        "3s": [5],
        "3p": [6, 7, 8],
        "4p": [9, 10, 11],
        "4s": [12],
        "3d": [13, 14, 15, 16, 17],
        "5p": [18, 19, 20],
        "5s": [21, ],
        "4d": [22, 23, 24, 25, 26],
        "nao": 27,
        "basis": "aug-ccpvdz-dk",
        "cmoao": None,
    },
    "Fe": {
        "1s": [0],
        "2s": [1],
        "2p": [2, 3, 4],
        "3s": [5],
        "3p": [6, 7, 8],
        "4s": [9],
        "3d": [10, 11, 12, 13, 14],
        "4p": [15, 16, 17],
        "5s": [18],
        "4d": [19, 20, 21, 22, 23],
        "5p": [24, 25, 26],
        "6s": [27],
        "6p": [28, 29, 30],
        "5d": [31, 32, 33, 34, 35],
        "4f": [36, 37, 38, 39, 40, 41, 42],
        "7p": [43, 44, 45],
        "7s": [46],
        "6d": [47, 48, 49, 50, 51],
        "5f": [52, 53, 54, 55, 56, 57, 58],
        "nao": 59,
        "basis": "aug-ccpvdz-dk",
        "cmoao": None,
    },
}

    Mol = pyscf.gto.Mole()
    Mol.atom = '''
    Fe  0.000000  0.000000  1.312783
    Fe  0.000000  0.000000 -1.312783
    S   0.757448 -1.576802  0.000000
    S  -0.757448  1.576802  0.000000
    S   1.583457  0.881621  2.761494
    S  -1.583457 -0.881621  2.761494
    S  -1.583457 -0.881621 -2.761494
    S   1.583457  0.881621 -2.761494
    C  -2.006854  0.653551  3.669027
    C   2.006854 -0.653551  3.669027
    C   2.006854 -0.653551 -3.669027
    C  -2.006854  0.653551 -3.669027
    H  -2.501125  1.369319  2.995471
    H   2.501125 -1.369319  2.995471
    H   2.501125 -1.369319 -2.995471
    H  -2.501125  1.369319 -2.995471
    H  -2.683553  0.421356  4.508610
    H   2.683553 -0.421356  4.508610
    H   2.683553 -0.421356 -4.508610
    H  -2.683553  0.421356 -4.508610
    H  -1.086873  1.121991  4.050762
    H   1.086873 -1.121991  4.050762
    H   1.086873 -1.121991 -4.050762
    H  -1.086873  1.121991 -4.050762
        '''
    Mol.basis = {'Fe': 'aug-ccpvdz-dk', 'S': 'aug-ccpvdz-dk',
                 'C': 'ccpvdz-dk', 'H': 'ccpvdz-dk'}
    Mol.symmetry = "C1"
    Mol.spin = 10
    Mol.charge = -2
    Mol.verbose = 4
    Mol.unit = 'angstorm'
    Mol.build()


    dirname = "./Examples"

    for atom in ["H", "C", "S", "Fe"]:
        atm_bas[atom]["cmoao"] = Util_File.ReadIn_Cmoao(
            dirname+"/"+"%s_0_%s" % (atom, atm_bas[atom]["basis"]), atm_bas[atom]["nao"])
    
    cmoao_fe2s2 = Util_File.ReadIn_Cmoao("Fe2S2_22_26_Init", 396)

    Analysis_Orb_Comp(Mol, cmoao_fe2s2, 0, Mol.nao,
                      atm_bas, tol=0.3, with_distinct_atm=False)
    Analysis_Orb_Comp(Mol, cmoao_fe2s2, 0, Mol.nao,
                      atm_bas, tol=0.3, with_distinct_atm=True)