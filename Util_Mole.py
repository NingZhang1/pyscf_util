import pyscf
import numpy
import Util_Math
from pyscf.symm.Dmatrix import *


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


def get_rotated_mol_coord(mol, rot_center, alpha, beta, gamma):
    coord = mol.atom_coords() * 0.52917720859
    coord -= rot_center
    rot_mat = Util_Math.get_rotation_matrix_euler_angle_ZYZ(alpha, beta, gamma)
    coord = numpy.dot(rot_mat, coord.T).T
    coord += rot_center
    res = []
    for i in range(mol.natm):
        res.append([mol.atom_symbol(i), coord[i, :]])
    return res


def get_bas_rotate_matrix(mol, alpha, beta, gamma):
    res = numpy.zeros((mol.nao, mol.nao), dtype=numpy.float64)
    loc = 0
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        dmat = Dmatrix(l, alpha, beta, gamma, reorder_p=True)
        for _ in range(mol.bas_nctr(i)):
            res[loc:loc+2*l+1, loc:loc+2*l + 1] = dmat
            loc += 2*l+1
    return numpy.matrix(res)
