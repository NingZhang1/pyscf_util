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
    # print(bond_begin,bond_end)
    for bond_id in range(bond_begin, bond_end):
        orb_coeff = orb[:, bond_id:bond_id+1].reshape(-1)
        id_pair = -1
        norm_pair = 100.0
        atm_coeff_1 = orb_coeff[:nao_atom]
        atm_coeff_2 = orb_coeff[nao_atom:]
        # for pair_id in range(0, nao):
        for pair_id in range(bond_begin, bond_end):
            if bond_id == pair_id:
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
    pairing = pairing_res

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
