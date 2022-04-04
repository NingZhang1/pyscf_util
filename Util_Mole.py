import pyscf


def get_orbsym(mol, mocoeff):

    OrbSym = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                       mocoeff)
    OrbSymID = [pyscf.symm.irrep_name2id(mol.groupname, x) for x in OrbSym]

    return OrbSymID, OrbSym

