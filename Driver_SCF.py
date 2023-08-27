from functools import reduce
import TEST_CONFIG

import pyscf
import Util_Mole
import numpy as np
from pyscf import fci, mcscf


def Run_SCF(mol, sfx1e=False,  newton=False):
    my_hf = pyscf.scf.ROHF(mol)
    if sfx1e:
        my_hf = pyscf.scf.sfx2c(my_hf)
    if newton:
        my_hf = pyscf.scf.newton(my_hf)
    my_hf.kernel()
    return my_hf


def Run_SCF_customized(mol, mo_coeff, sfx1e=False,  newton=False):
    my_hf = pyscf.scf.ROHF(mol)
    if sfx1e:
        my_hf = pyscf.scf.sfx2c(my_hf)
    if newton:
        my_hf = pyscf.scf.newton(my_hf)

    ovlp = mol.intor("int1e_ovlp")
    ovlp = reduce(np.dot, (mo_coeff.T, ovlp, mo_coeff))
    hcore = my_hf.get_hcore()
    hcore = reduce(np.dot, (mo_coeff.T, hcore, mo_coeff))
    eri = pyscf.ao2mo.full(eri_or_mol=mol, mo_coeff=mo_coeff, aosym='4')

    my_hf.get_hcore = lambda *args: hcore
    my_hf.get_ovlp = lambda *args: ovlp
    my_hf._eri = eri

    my_hf.kernel()
    return my_hf


def mocoeff_phase_canonicalize(mo_coeff):
    for i in range(mo_coeff.shape[1]):
        for j in range(mo_coeff.shape[0]):
            if abs(mo_coeff[j, i]) > 1e-4:
                if mo_coeff[j, i] < 0:
                    mo_coeff[:, i] *= -1
                break
    return mo_coeff


def Analysis_SCF(mol, my_hf):
    mo_energy = my_hf.mo_energy
    orbsym_ID, orbsym = Util_Mole.get_orbsym(mol, my_hf.mo_coeff)
    mo_occ = my_hf.mo_occ

    print("\n\n***********************************************")
    print("******************* Detail  *******************")
    print("***********************************************")
    print("**** Basic Atom Info ****")
    print("Remark : Coordinate in Bohr")
    for i in range(mol.natm):
        print('%8s charge %6.2f xyz %s' % (mol.atom_pure_symbol(i),
                                           mol.atom_charge(i),
                                           mol.atom_coord(i)))
    print("--------------------------------------------------------------------------------")
    print("**** Basic MO Info ****")
    print("--------------------------------------------------------------------------------")
    print("%6s | %16s | %16s | %5s | %5s | %5s |" %
          ("OrbId", "OrbEne", "OrbEne(eV)", "SymID", "Sym", "Occ"))
    for i in range(mol.nao):
        print("%6d | %16.8f | %16.8f | %5d | %5s | %5d |" %
              (i, mo_energy[i], mo_energy[i]*27.2114, orbsym_ID[i], orbsym[i], mo_occ[i]))
    print("--------------------------------------------------------------------------------")
    my_hf.analyze()
    print("------------------------------     END     ------------------------------------")


def Run_MCSCF(_mol, _rohf, _nelecas, _ncas,
              _frozen=None,
              _mo_init=None,
              _cas_list=None,
              _mc_conv_tol=1e-7,
              _mc_max_macro=128,
              _iCI=False,
              _pyscf_state=None,  # [spintwo, irrep, nstates]
              _iCI_State=None,  # [spintwo, irrep, nstates]
              _cmin=1e-4,  # for iCI
              _tol=1e-6,  # for iCI
              _do_pyscf_analysis=False,
              _internal_rotation=False,
              _run_mcscf=True,):
    # Generate MCSCF object
    my_mc = pyscf.mcscf.CASSCF(_rohf, nelecas=_nelecas, ncas=_ncas)
    my_mc.conv_tol = _mc_conv_tol
    my_mc.max_cycle_macro = _mc_max_macro
    my_mc.frozen = _frozen
    # print("In RUN_MCSCF", my_mc.mol)
    # Sort MO
    mo_init = _rohf.mo_coeff
    my_mc.mo_coeff = _rohf.mo_coeff  # in case _run_mcscf = False,
    if _cas_list is not None:
        if (isinstance(_cas_list, dict)):
            mo_init = pyscf.mcscf.sort_mo_by_irrep(my_mc, mo_init, _cas_list)
        else:
            mo_init = pyscf.mcscf.sort_mo(my_mc, mo_init, _cas_list)
    # determine FCIsolver
    nelecas = _nelecas
    if (isinstance(_nelecas, tuple)):
        nelecas = _nelecas[0] + _nelecas[1]
    if _ncas > 12 or nelecas > 12:  # Large Cas Space
        _iCI = True
    if not _iCI:
        if _pyscf_state is not None:
            # [spintwo, irrep, nstates]
            # Only spin averaged is supported
            solver_all = []
            nstates = 0
            for state in _pyscf_state:
                if state[0] % 2 == 1:
                    solver = fci.direct_spin1_symm.FCI(_mol)
                    solver.wfnsym = state[1]
                    solver.nroots = state[2]
                    solver.spin = state[0]
                    solver_all.append(solver)
                else:
                    solver = fci.direct_spin0_symm.FCI(_mol)
                    solver.wfnsym = state[1]
                    solver.nroots = state[2]
                    solver.spin = state[0]
                    solver_all.append(solver)
                nstates += state[2]
            my_mc = mcscf.state_average_mix_(
                my_mc, solver_all, (np.ones(nstates)/nstates))
    # else:
    #     if _iCI_State is None:
    #         if _pyscf_state is None:
    #             _iCI_State = [[_nelecas % 2, 0, 1], ]
    #         else:
    #             _iCI_State = []
    #             for state in _pyscf_state:
    #                 state_iCI = [state[0], pyscf.symm.irrep_name2id(
    #                     _mol.groupname, state[1]) % 10, state[2]]
    #                 _iCI_State.append(state_iCI)
    #             print(_iCI_State)
    #             # if _mol.groupname == 'Dooh' or _mol.groupname == 'Coov':
    #             #     raise ValueError("Not support group Dooh or Coov")
    #     my_mc.fcisolver = iCISCF.iCI(mol=_mol,
    #                                  cmin=_cmin,
    #                                  state=_iCI_State,
    #                                  tol=_tol,
    #                                  mo_coeff=mo_init)
    # Run
    if _iCI:
        my_mc.internal_rotation = _internal_rotation
    if _run_mcscf:
        # print("In RUN_MCSCF", my_mc.mol)
        my_mc.kernel(mo_init)
    # Analysis
    if _do_pyscf_analysis:
        my_mc.analyze()
    # 分析成分
    # ao_labels, ao_basis = _get_unique_aolabels(_mol)
    # analysis_mo_contribution(_mol, ao_labels, ao_basis, my_mc.mo_coeff)
    if _run_mcscf:
        return my_mc
    else:
        return [my_mc, mo_init]
