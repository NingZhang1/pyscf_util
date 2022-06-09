import os
from tqdm import tqdm
import torch
import get_atom_orb
import Chem_Bond_Analysis

# load_path = '/home/ningzhang/GitHubPackage/deepdf/data/qm9_100'
load_path = '/home/wenshi/data/qm9'


def check_locMO(atm_bas='6-31G(d)'):

    atom_bas = get_atom_orb.atom_min_cas_bas(
        ["C", "H", "O", "N", "F"], basis=atm_bas, print_verbose=0)

    file_list = sorted(os.listdir(load_path))[:2048]
    print("length of file_list", len(file_list))
    res = []
    id = 0
    for fname in tqdm(file_list):
        with open(os.path.join(load_path, fname), 'r') as fp:
            lines = fp.readlines()
            xyz = ''.join(lines[2:])
            # mol = get_mol_by(xyz)
            chem_bond_analyzer = Chem_Bond_Analysis.ChemBondAnalyzer(
                xyz=xyz, print_verbose=0, basis=atm_bas)  # bugs!
            chem_bond_analyzer.atom_bas = atom_bas
            etot = chem_bond_analyzer.e_tot
            dm = chem_bond_analyzer.dm
            mo_coeff = chem_bond_analyzer.rohf.mo_coeff
            # print(etot, occ_vir_etot, occ_etot)
            res.append({"xyz": xyz,
                        "etot": etot,
                        "dm": dm,
                        "mo_coeff": mo_coeff,
                        })
            id += 1
            if (id % 128) == 1:
                torch.save(res, "dm" + "_" + atm_bas + ".data")
                print("finish ", id)
    torch.save(res, "dm" + "_" + atm_bas + ".data")


if __name__ == "__main__":
    check_locMO()
    check_locMO(atm_bas="ccpvtz")
