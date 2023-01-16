import pyscf
import numpy
import Util_Math
# from pyscf.symm.Dmatrix import *


def get_orbsym(mol, mocoeff):

    OrbSym = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                       mocoeff)
    OrbSymID = [pyscf.symm.irrep_name2id(mol.groupname, x) for x in OrbSym]

    return OrbSymID, OrbSym


def get_mol(xyz, charge=0, spin=0, basis='6-31G(d)', symmetry="", print_verbose=0, unit='angstorm'):
    mol = pyscf.gto.M(
        verbose=print_verbose,
        atom=xyz,
        basis=basis,
        spin=spin,
        charge=charge,
        symmetry=symmetry,
        unit=unit
    )
    mol.build()
    return mol


def get_mole_info_for_chem_bond_analysis(mol):

    # 获得每一个原子独特的label

    atom_num = {
        "H": 0,
        "C": 0,
        "O": 0,
        "N": 0,
        "F": 0,
    }

    atom_label = []

    for atom_id in range(mol.natm):
        atom_type = mol.atom_pure_symbol(atom_id)
        atom_num[atom_type] += 1
        atom_label.append(atom_type+str(atom_num[atom_type]))

    res = {}

    atom_id = 0
    for label in atom_label:
        res[label] = {
            'atom_id': atom_id,
            # 'atom_orb_loc': None,
            # 'atom_id_bonded': None,
            # 'atom_label_bonded': None,
            # 'bond_level': numpy.zeros(mol.natm),  # an array 0 --> not bonded,
            # 'chemical bond': None,  # [loc_orb_id,atom_id]
            # 'non-bonding orb_id': None,
        }

    return res, atom_label

def get_atm_symbol(mol):
    res = []
    for id in range(mol.natm):
        res.append(mol.atom_symbol(id))
    return res

def get_atm_symbol(mol):
    res = []
    for id in range(mol.natm):
        res.append(mol.atom_symbol(id))
    return res


def get_rotated_mol_coord(mol, rot_center, alpha, beta, gamma):
    coord = mol.atom_coords()
    if mol.unit == 'angstorm':
        coord = coord * 0.52917720859
    coord -= rot_center
    rot_mat = Util_Math.get_rotation_matrix_euler_angle_ZYZ(alpha, beta, gamma)
    coord = numpy.dot(rot_mat, coord.T).T
    coord += rot_center
    res = []
    for i in range(mol.natm):
        res.append([mol.atom_symbol(i), coord[i, :]])
    return res

def get_xyz_list_format(mol):
    coord = mol.atom_coords()
    if mol.unit == 'angstorm':
        coord = coord * 0.52917720859
    return coord

def get_xyz_list_format(mol):
    coord = mol.atom_coords()
    if mol.unit == 'angstorm':
        coord = coord * 0.52917720859
    return coord


def get_mol_xyz_list_format(mol):
    coord = mol.atom_coords()
    if mol.unit == 'angstorm':
        coord = coord * 0.52917720859
    res = []
    for i in range(mol.natm):
        res.append([mol.atom_symbol(i), coord[i, :]])
    return res


def get_mol_geometric_center(mol):
    xyz_list = get_mol_xyz_list_format(mol)
    res = xyz_list[0][1]
    for i in range(1, mol.natm):
        res += xyz_list[i][1]
    return res/mol.natm


# def get_bas_rotate_matrix(mol, alpha, beta, gamma):
#     res = numpy.zeros((mol.nao, mol.nao), dtype=numpy.float64)
#     loc = 0
#     for i in range(mol.nbas):
#         l = mol.bas_angular(i)
#         dmat = Dmatrix(l, alpha, beta, gamma, reorder_p=True)
#         for _ in range(mol.bas_nctr(i)):
#             res[loc:loc+2*l+1, loc:loc+2*l + 1] = dmat
#             loc += 2*l+1
#     return numpy.matrix(res)


# local gauge problem

def get_atm_bas_in_mole_fix_local_gauge_problem(mol, mole_graph):
    basis = mol.basis
    mole_geometric_center = get_mol_geometric_center(mol)
    xyz_list = get_mol_xyz_list_format(mol)

    # print(mole_geometric_center)

    res = numpy.zeros((mol.nao, mol.nao))
    occ = numpy.zeros((mol.nao))

    loc_res = 0

    for id_atm in range(mol.natm):

        # 获取与给原子编号相同的原子

        bonded = []
        for id in range(0, mol.natm):
            if id == id_atm:
                continue
            if mole_graph[id_atm, id] > 0:
                bonded.append(id)

        # 构造局域分子

        xyz_partial = [[xyz_list[id_atm][0]+'1', xyz_list[id_atm][1]]]

        for id in bonded:
            xyz_partial.append(xyz_list[id])

        xyz_partial.append(['X', mole_geometric_center])

        basis_list = {
            xyz_list[id_atm][0]+'1': pyscf.gto.basis.load(basis, xyz_list[id_atm][0]),
            'C': 'sto-3g',
            'H': 'sto-3g',
            'O': 'sto-3g',
            'N': 'sto-3g',
            'F': 'sto-3g',
            'X': pyscf.gto.basis.load('sto-3g', 'H')
        }

        mol_partial = get_mol(
            xyz_partial, spin=None, basis=basis_list)

        # print(get_mol_xyz_list_format(mol_partial))

        # make rdm1

        scf = pyscf.scf.ROHF(mol_partial)
        scf.kernel()
        dma, dmb = scf.make_rdm1(scf.mo_coeff, scf.mo_occ)
        dm1 = dma + dmb

        # print("etot = ",scf.e_tot)
        # print("nao  = ",mol_partial.nao)

        # 抽出局域原子基组, 用了 HF 可能很慢，暂时先看看可行性, expanded over atomic HF ?

        atom = get_mol([xyz_partial[0]], spin=None, basis=basis_list)
        Nao = atom.nao

        dm_atm = dm1[:Nao, :Nao]
        atom_orb_rotated = numpy.zeros((Nao, Nao))
        atom_occ = numpy.zeros((Nao))

        loc_now = 0

        for i in range(atom.nbas):
            # print('shell %d on atom %d l = %s has %d contracted GTOs' %
            #       (i, atom.bas_atom(i), atom.bas_angular(i), atom.bas_nctr(i)))
            for _ in range(atom.bas_nctr(i)):
                loc_end = loc_now + 2*atom.bas_angular(i)+1
                if atom.bas_angular(i) == 0:
                    atom_orb_rotated[loc_now:loc_end,
                                     loc_now:loc_end] = 1.0  # s function
                    atom_occ[loc_now:loc_end] = dm_atm[loc_now:loc_end,
                                                       loc_now:loc_end]
                else:
                    dm_tmp = dm_atm[loc_now:loc_end, loc_now:loc_end]
                    e, m = numpy.linalg.eigh(dm_tmp)  # ascending order
                    if numpy.linalg.det(m) < 0.0:
                        m *= -1.0
                    # print(e)
                    atom_orb_rotated[loc_now:loc_end,
                                     loc_now:loc_end] = m  # m 的相位问题靠 ovlp 矩阵消除
                    atom_occ[loc_now:loc_end] = e
                loc_now = loc_end

        res[loc_res:loc_res+Nao, loc_res:loc_res+Nao] = atom_orb_rotated
        occ[loc_res:loc_res+Nao] = atom_occ
        loc_res += Nao

    return res, occ

# 给定一组 d2h irrep 还原 SO(3)


def _is_s(irrep_d2h):
    return irrep_d2h[0] == 0


def _is_p(irrep_d2h):
    irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
    for irrep in irrep_d2h:
        irrep_tmp[irrep] += 1

    return irrep_tmp[4] == 0 and irrep_tmp[5] == 1 and irrep_tmp[6] == 1 and irrep_tmp[7] == 1


def _is_d(irrep_d2h):
    # print("In is_d", irrep_d2h)
    irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
    for irrep in irrep_d2h:
        irrep_tmp[irrep] += 1

    return irrep_tmp[0] == 2 and irrep_tmp[1] == 1 and irrep_tmp[2] == 1 and irrep_tmp[3] == 1


def _is_f(irrep_d2h):
    irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
    for irrep in irrep_d2h:
        irrep_tmp[irrep] += 1

    return irrep_tmp[4] == 1 and irrep_tmp[5] == 2 and irrep_tmp[6] == 2 and irrep_tmp[7] == 2


def _is_g(irrep_d2h):
    irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
    for irrep in irrep_d2h:
        irrep_tmp[irrep] += 1

    return irrep_tmp[0] == 3 and irrep_tmp[1] == 2 and irrep_tmp[2] == 2 and irrep_tmp[3] == 2


def _is_h(irrep_d2h):
    irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
    for irrep in irrep_d2h:
        irrep_tmp[irrep] += 1

    return irrep_tmp[4] == 2 and irrep_tmp[5] == 3 and irrep_tmp[6] == 3 and irrep_tmp[7] == 3


def _is_i(irrep_d2h):
    irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
    for irrep in irrep_d2h:
        irrep_tmp[irrep] += 1

    return irrep_tmp[0] == 4 and irrep_tmp[1] == 3 and irrep_tmp[2] == 3 and irrep_tmp[3] == 3


def _is_type(irrep_d2h):
    length = len(irrep_d2h)
    if length == 1:
        return _is_s(irrep_d2h)
    elif length == 3:
        return _is_p(irrep_d2h)
    elif length == 5:
        return _is_d(irrep_d2h)
    elif length == 7:
        return _is_f(irrep_d2h)
    elif length == 9:
        return _is_g(irrep_d2h)
    elif length == 11:
        return _is_h(irrep_d2h)
    elif length == 13:
        return _is_i(irrep_d2h)


def _determine_nshell_g(irrep_d2h):
    irrep_tmp = irrep_d2h
    # for irrep in irrep_d2h:
    #     irrep_tmp[irrep] += 1
    nshell = irrep_tmp[0] - irrep_tmp[1]
    Res = {
        's': 0,
        'd': 0,
        'g': 0,
        'i': 0,
    }
    # print(irrep_d2h)
    # print(irrep_tmp)
    # print(nshell)
    find_solution = False
    for ns in range(nshell+1):
        nd_max = (irrep_tmp[0] - ns)//2
        for nd in range(nd_max+1):
            ng_max = (irrep_tmp[0] - ns - nd*2)//3
            for ng in range(ng_max+1):
                ni_max = (irrep_tmp[0] - ns - nd*2 - ng*3)//4
                for ni in range(ni_max+1):
                    if ((ns + nd*2 + ng * 3 + ni*4) == irrep_tmp[0]) and ((nd*1 + ng * 2 + ni*3) == irrep_tmp[1]):
                        if find_solution == False:
                            Res = {
                                's': ns,
                                'd': nd,
                                'g': ng,
                                'i': ni,
                            }
                            find_solution = True
                        else:
                            # raise RuntimeError
                            return None
    return Res


def _determine_nshell_u(irrep_d2h):
    irrep_tmp = irrep_d2h
    # for irrep in irrep_d2h:
    #     irrep_tmp[irrep] += 1
    nshell = irrep_tmp[5] - irrep_tmp[4]

    Res = {
        'p': 0,
        'f': 0,
        'h': 0,
    }
    # print(irrep_tmp)
    # print(nshell)
    find_solution = False
    for np in range(nshell+1):
        nf_max = irrep_tmp[4]
        for nf in range(nf_max+1):
            nh_max = (irrep_tmp[4] - nf)//2
            for nh in range(nh_max+1):
                if ((nf*1 + nh * 2) == irrep_tmp[4]) and ((np+nf*2 + nh * 3) == irrep_tmp[5]):
                    if find_solution == False:
                        Res = {
                            'p': np,
                            'f': nf,
                            'h': nh,
                        }
                        find_solution = True
                    else:
                        # raise RuntimeError
                        return None
    return Res


sphere_2_norb = {
    's': 1,
    'p': 3,
    'd': 5,
    'f': 7,
    'g': 9,
    'h': 11,
    'i': 13,
}


def _determine_macrocfg_g(nshell, irrep_d2h):
    # print(irrep_d2h)
    # print("In _determine_macrocfg_g")
    # print(nshell)
    # print(irrep_d2h)
    length = len(irrep_d2h)
    loc_now = 0
    Res = []
    HasLeft = True
    while (HasLeft):
        HasLeft = False
        for sphere in ['i', 'g', 'd', 's']:
            if nshell[sphere] > 0:
                # print(sphere, loc_now)
                HasLeft = True
                if (loc_now + sphere_2_norb[sphere]) > length:
                    raise RuntimeError
                if _is_type(irrep_d2h[loc_now:loc_now+sphere_2_norb[sphere]]):
                    Res.append(sphere_2_norb[sphere])
                    nshell[sphere] -= 1
                    loc_now += sphere_2_norb[sphere]
                    break
    return Res


def _determine_macrocfg_u(nshell, irrep_d2h):
    # print(nshell)
    length = len(irrep_d2h)
    loc_now = 0
    Res = []
    HasLeft = True
    while (HasLeft):
        HasLeft = False
        for sphere in ['h', 'f', 'p']:
            if nshell[sphere] > 0:
                HasLeft = True
                if (loc_now + sphere_2_norb[sphere]) > length:
                    raise RuntimeError
                if _is_type(irrep_d2h[loc_now:loc_now+sphere_2_norb[sphere]]):
                    Res.append(sphere_2_norb[sphere])
                    nshell[sphere] -= 1
                    loc_now += sphere_2_norb[sphere]
                    break
    return Res


def get_macro_given_SO3_irrep(irrep):
    length = len(irrep)
    loc = 0

    Res = []

    while (loc < length):
        L = irrep[loc][0]
        tmp = sphere_2_norb[L]
        loc += tmp
        Res.append(tmp)

    return Res


def get_macro_given_d2h_irrep(irrep_d2h):
    symmetry_now = None
    loc_begin = None

    Res = []

    irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]

    for id, irrep in enumerate(irrep_d2h):
        if symmetry_now is None:
            if irrep >= 4:
                symmetry_now = 'u'
                loc_begin = 0
                irrep_tmp[irrep] += 1
            else:
                symmetry_now = 'g'
                loc_begin = 0
                irrep_tmp[irrep] += 1
        else:
            if irrep >= 4 and symmetry_now == 'g':

                nshell = _determine_nshell_g(irrep_tmp)
                if nshell == None:
                    Res.append(id-loc_begin)
                else:
                    Res.extend(_determine_macrocfg_g(
                        nshell, irrep_d2h[loc_begin:id]))

                symmetry_now = 'u'
                loc_begin = id
                irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
                irrep_tmp[irrep] += 1

            else:
                if irrep < 4 and symmetry_now == 'u':

                    nshell = _determine_nshell_u(irrep_tmp)
                    if nshell == None:
                        Res.append(id-loc_begin)
                    else:
                        Res.extend(_determine_macrocfg_u(
                            nshell, irrep_d2h[loc_begin:id]))

                    symmetry_now = 'g'
                    loc_begin = id
                    irrep_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
                    irrep_tmp[irrep] += 1
                else:
                    # print(id, irrep)
                    irrep_tmp[irrep] += 1

    # judge the last

    if symmetry_now == 'u':
        nshell = _determine_nshell_u(irrep_tmp)
        if nshell == None:
            Res.append(len(irrep_d2h)-loc_begin)
        else:
            Res.extend(_determine_macrocfg_u(
                nshell, irrep_d2h[loc_begin:]))
    else:
        nshell = _determine_nshell_g(irrep_tmp)
        if nshell == None:
            Res.append(len(irrep_d2h)-loc_begin)
        else:
            Res.extend(_determine_macrocfg_g(
                nshell, irrep_d2h[loc_begin:]))

    return Res
