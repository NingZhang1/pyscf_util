import pyscf
from pyscf import tools
import numpy as np


def dump_heff_casci(_mol, _mcscf, _core_coeff, _mocoeff, _filename='FCIDUMP'):
    loc1 = 0
    if _core_coeff is not None:
        loc1 = _core_coeff.shape[1]
    else:
        loc1 = 0
    norb = loc1 + _mocoeff.shape[1]
    nao = _mocoeff.shape[0]
    mocoeff = np.zeros((nao, norb))
    core_indx = list(range(0, loc1))
    act_indx = list(range(loc1, norb))
    mocoeff[:, core_indx] = _core_coeff
    mocoeff[:, act_indx] = _mocoeff
    int2e_full = pyscf.ao2mo.full(eri_or_mol=_mol, mo_coeff=mocoeff, aosym='1')
    int2e_full = pyscf.ao2mo.restore(1, int2e_full.copy(), mocoeff.shape[1])
    # Get integrals
    int2e_res = int2e_full[loc1:norb, loc1:norb, loc1:norb, loc1:norb]
    int2e_res = pyscf.ao2mo.restore(8, int2e_res.copy(), norb-loc1)
    int1e_res, energy_core = pyscf.mcscf.casci.h1e_for_cas(
        _mcscf, mo_coeff=mocoeff, ncas=_mocoeff.shape[1], ncore=loc1)
    # get orbsym
    OrbSym = pyscf.symm.label_orb_symm(_mol, _mol.irrep_name, _mol.symm_orb,
                                       _mocoeff)
    OrbSymID = [pyscf.symm.irrep_name2id(_mol.groupname, x) for x in OrbSym]
    # DUMP
    tools.fcidump.from_integrals(filename=_filename,
                                 h1e=int1e_res,
                                 h2e=int2e_res,
                                 nuc=energy_core,
                                 nmo=_mocoeff.shape[1],
                                 nelec=_mol.nelectrons - 2 *
                                 _core_coeff.shape[1],  # Useless
                                 tol=1e-10,
                                 orbsym=OrbSymID)

