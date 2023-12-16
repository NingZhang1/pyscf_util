import Util_Orb
import pyscf
import pickle

# basically the procedure consists of first HF, then reoder, then perform ICIPT2

APP = "/home/nzhangcaltech/iCI_CXX/bin/iCIPT2_D2h_CSF_CoreExt.exe"


def _perform_hf_then_localize_orb(mol, CoreOrb: list, debug=False):

    # perform HF
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    if debug:
        print("HF finished")
        print("HF energy: ", mf.e_tot)
        print(mf.mo_energy)

    # split-localization of orbitals

    mo = mf.mo_coeff

    for core_orbs_info in CoreOrb:
        loc = core_orbs_info["loc"]
        mo = Util_Orb.split_loc_given_range(mol, mo, loc[0], loc[1])

    mf.mo_coeff = mo

    if debug:
        for core_orbs_info in CoreOrb:
            loc = core_orbs_info["loc"]
            print(mo[:, loc[0]:loc[1]])

    return mf


def _dump_CoreExt_FCIDUMP(mol, hf, CoreOrb: list, prefix, debug=False):

    # dump the CoreExt FCIDUMP
    # CoreOrb: [{"loc": [0, 2], "cas": [4, 4], "type": "C1s"}, ...]
    # prefix: prefix of the FCIDUMP file

    mo = hf.mo_coeff
    CoreOrb_CAS = []

    nocc = mol.nelectron // 2

    Res = []

    for core_orbs_info in CoreOrb:
        loc = core_orbs_info["loc"]
        cas = core_orbs_info["cas"]
        assert (loc[1] + cas[0] <= nocc)
        if loc[1] + cas[0] == nocc and loc[0] == 0:
            orb_order = list(range(mol.nao))
        else:
            orb_order = list(range(loc[0]))  # the core that not involved
            # the nor-core that not involved
            orb_order.extend(range(loc[1], nocc - cas[0]))
            # the core that involved
            orb_order.extend(list(range(loc[0], loc[1])))
            # the nor-core that involved
            orb_order.extend(list(range(nocc - cas[0], mol.nao)))

        if debug:
            print("orb_order: ", orb_order)

        import copy
        mo = copy.deepcopy(hf.mo_coeff)

        mo = mo[:, orb_order]
        orbsym = pyscf.symm.label_orb_symm(
            mol, mol.irrep_id, mol.symm_orb, mo)

        ### dump the FCIDUMP ###

        pyscf.tools.fcidump.from_mo(
            mol, prefix + core_orbs_info["type"], mo, orbsym)

        Res.append({"loc": loc, "cas": cas, "type": core_orbs_info["type"],
                    "orb_order": orb_order, "orbsym": orbsym, "mo": mo})
    
    return Res


if __name__ == "__main__":

    ### test C2H2 ###

    CoreOrb_C2H4 = [
        {"loc": [0, 2],
         "cas": [4, 4],
         "type": "C1s"},
    ]

    mol = pyscf.gto.M(
        verbose=4,
        atom='''
C  0.0000000  -0.6654000  0.0000000
H  0.9220909  -1.2282467  0.0000000
H -0.9220909  -1.2282467  0.0000000
C  0.0000000   0.6654000  0.0000000
H -0.9220909   1.2282467  0.0000000
H  0.9220909   1.2282467  0.0000000
''',
        basis='cc-pvdz',
        spin=0,
        charge=0,
        symmetry='D2h',
    )
    mol.build()

    hf = _perform_hf_then_localize_orb(mol, CoreOrb_C2H4, debug=True)

    _dump_CoreExt_FCIDUMP(mol, hf, CoreOrb_C2H4, "FCIDUMP_C2H4_", debug=True)
