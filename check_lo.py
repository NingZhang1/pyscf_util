import os
from tqdm import tqdm
import torch
import get_atom_orb
import Chem_Bond_Analysis

load_path = '/home/ningzhang/GitHubPackage/deepdf/data/qm9_100'


def check_locMO(atm_bas='6-31G(d)', vir_label=Chem_Bond_Analysis.vir_label_min, vir_name="minimal"):

    atom_bas = get_atom_orb.atom_min_cas_bas(
        ["C", "H", "O", "N", "F"], basis=atm_bas, print_verbose=0)

    file_list = sorted(os.listdir(load_path))
    res = []
    for fname in tqdm(file_list):
        with open(os.path.join(load_path, fname), 'r') as fp:
            lines = fp.readlines()
            xyz = ''.join(lines[2:])
            # mol = get_mol_by(xyz)
            chem_bond_analyzer = Chem_Bond_Analysis.ChemBondAnalyzer(
                xyz=xyz, print_verbose=0, basis=atm_bas)  # bugs!
            chem_bond_analyzer.atom_bas = atom_bas
            etot = chem_bond_analyzer.e_tot
            occ_vir_etot = chem_bond_analyzer.check_cnvg_orb_proj_atm_occ_vir()
            occ_etot = chem_bond_analyzer.check_cnvg_orb_proj_atm_occ()
            # print(etot, occ_vir_etot, occ_etot)
            res.append({"xyz": xyz,
                        "etot": etot,
                        "occ": occ_etot,
                        "occvir": occ_vir_etot,
                        })
    torch.save(res, "loc_mo_analysis" + "_" +
               atm_bas + "_" + vir_name + ".data")


if __name__ == "__main__":
    check_locMO()
    check_locMO(atm_bas="ccpvdz")
    check_locMO(atm_bas="ccpvtz")
    check_locMO(atm_bas="ccpvqz")
    check_locMO(atm_bas="aug-ccpvdz")

    check_locMO(vir_label=Chem_Bond_Analysis.vir_label, vir_name="large")
    check_locMO(atm_bas="ccpvdz",
                vir_label=Chem_Bond_Analysis.vir_label, vir_name="large")
    check_locMO(atm_bas="ccpvtz",
                vir_label=Chem_Bond_Analysis.vir_label, vir_name="large")
    check_locMO(atm_bas="ccpvqz",
                vir_label=Chem_Bond_Analysis.vir_label, vir_name="large")
    check_locMO(atm_bas="aug-ccpvdz",
                vir_label=Chem_Bond_Analysis.vir_label, vir_name="large")
