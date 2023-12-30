# coding=UTF-8

from pyscf import ao2mo
from functools import reduce
from pyscf import __config__
import pyscf
import os
import sys
from pyscf import tools
from pyscf import symm
import numpy as np
import iCISCF
import copy

import pickle

from pyscf import fci, mcscf


def get_state_averaged_CASSCF(_mol, my_mc_input, _pyscf_state):
    solver_all = []
    nstates = 0
    for state in _pyscf_state:
        if state[0] % 2 == 1:
            solver = fci.direct_spin1_symm.FCI(_mol)
            solver.wfnsym = state[1]
            solver.nroots = state[2]
            solver.spin = 1
            solver_all.append(solver)
        else:
            solver = fci.direct_spin0_symm.FCI(_mol)
            solver.wfnsym = state[1]
            solver.nroots = state[2]
            solver.spin = 0
            solver_all.append(solver)
        nstates += state[2]
    my_mc = mcscf.state_average_mix_(
        my_mc_input, solver_all, (np.ones(nstates)/nstates))
    return my_mc


TaskInfo = [

    {
        "atom": "H",
        "charge": 0,
        "spin": 1,
        "state": [[1, 's+0', 1], ],
    },

    {
        "atom": "Li",
        "charge": 0,
        "spin": 1,
        "state": [[1, 's+0', 1], ],
    },

    {
        "atom": "Na",
        "charge": 0,
        "spin": 1,
        "state": [[1, 's+0', 1], ],
    },

    {
        "atom": "Be",
        "charge": 0,
        "spin": 0,
        "state": [[0, 's+0', 1], ],
    },

    {
        "atom": "B",
        "charge": 0,
        "spin": 1,

        "state": [[1, 'b1u', 1],
                  [1, 'b2u', 1],
                  [1, 'b3u', 1],
                  ],

        # "iCI_state": [[1, 5, 1, [1]], [1, 6, 1, [1]], [1, 7,  1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 1,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 2,
        },

        "fakescf": True,
        "fake_charge": 1,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 4},
    },

    {
        "atom": "Al",
        "charge": 0,
        "spin": 1,

        "state": [[1, 'b1u', 1],
                  [1, 'b2u', 1],
                  [1, 'b3u', 1],
                  ],

        # "iCI_state": [[1, 5, 1, [1]], [1, 6, 1, [1]], [1, 7,  1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 1,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 3,
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "fakescf": True,
        "fake_charge": 1,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 6, 'p+1': 2, 'p-1': 2, 'p+0': 2},
    },

    {
        "atom": "C",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "iCI_state": [[2, 1, 1, [1]], [2, 2, 1, [1]], [2, 3,  1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 2,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 2,
        },

        "fakescf": True,
        "fake_charge": 2,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 4},
    },

    {
        "atom": "Si",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "iCI_state": [[2, 1, 1, [1]], [2, 2, 1, [1]], [2, 3,  1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 2,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 3,
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "fakescf": True,
        "fake_charge": 2,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 6, 'p-1': 2, 'p+0': 2, 'p+1': 2},
    },

    {
        "atom": "O",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "iCI_state": [[2, 1, 1, [1]], [2, 2, 1, [1]], [2, 3,  1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 4,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 2,
        },

        "fakescf": True,
        "fake_charge": -2,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 4,
                             'p-1': 2, 'p+0': 2, 'p+1': 2, },
    },

    {
        "atom": "S",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "iCI_state": [[2, 1, 1, [1]], [2, 2, 1, [1]], [2, 3,  1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 4,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 3,
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "fakescf": True,
        "fake_charge": -2,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 6,
                             'p-1': 4, 'p+0': 4, 'p+1': 4, },
    },

    {
        "atom": "N",
        "charge": 0,
        "spin": 3,

        "state": [[3, 's+0', 1],
                  ],

        "iCI_state": [[3, 4, 1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 3,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 2,
        },

        "fakescf": True,
        "fake_charge": 3,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 4},
    },

    {
        "atom": "P",
        "charge": 0,
        "spin": 3,

        "state": [[3, 's+0', 1],
                  ],

        "iCI_state": [[3, 4, 1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 3,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 3,
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "fakescf": True,
        "fake_charge": 3,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 6, 'p-1': 2, 'p+0': 2, 'p+1': 2, },
    },

    {
        "atom": "F",
        "charge": 0,
        "spin": 1,

        "state": [[1, 'p+0', 1],
                  [1, 'p+1', 1],
                  [1, 'p-1', 1],
                  ],

        "iCI_state": [[1, 5, 1, [1]], [1, 6, 1, [1]], [1, 7,  1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 5,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 2,
        },

        "fakescf": True,
        "fake_charge": -1,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 4,
                             'p-1': 2, 'p+0': 2, 'p+1': 2, },
    },

    {
        "atom": "Cl",
        "charge": 0,
        "spin": 1,

        "state": [[1, 'p+0', 1],
                  [1, 'p+1', 1],
                  [1, 'p-1', 1],
                  ],

        "iCI_state": [[1, 5, 1, [1]], [1, 6, 1, [1]], [1, 7,  1, [1]]],

        "minimal_cas": {
            "norb": 3,
            "nelec": 5,
        },

        "cas_symm":  {
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "core_symm":  {
            's+0': 3,
            'p-1': 1, 'p+0': 1, 'p+1': 1,
        },

        "fakescf": True,
        "fake_charge": -1,
        "fake_spin": 0,
        "fake_irrep_nelec": {'s+0': 6,
                             'p-1': 4, 'p+0': 4, 'p+1': 4, },
    },

    #### 过渡金属 ####

    # Cr

    {
        "atom": "Cr",
        "charge": 0,
        "spin": 6,

        "state": [[6, 's+0', 1],],

        "iCI_state": [[6, 0, 1, [1]]],

        "minimal_cas": {
            "norb": 6,
            "nelec": 6,
        },

        "cas_symm":  {
            's+0': 0,
            'd-2': 1, 'd-1': 1, 'd+0': 1, 'd+1': 1, 'd+2': 1,
        },

        "core_symm":  {
            's+0': 3,
            'p-1': 2, 'p+0': 2, 'p+1': 2,
        },

        "fakescf": False,
    },

    # Fe

    {
        "atom": "Fe",
        "charge": 0,
        "spin": 4,

        "state": [[4, 'd+0', 1], [4, 'd+1', 1], [4, 'd+2', 1], [4, 'd-1', 1], [4, 'd-2', 1],],

        "iCI_state": [[4, 0, 2, [1, 1]], [4, 1, 1, [1]], [4, 2, 1, [1]], [4, 3, 1, [1]]],

        "minimal_cas": {
            "norb": 5,
            "nelec": 6,
        },

        "cas_symm":  {
            'd-2': 1, 'd-1': 1, 'd+0': 1, 'd+1': 1, 'd+2': 1,
        },

        "core_symm":  {
            's+0': 4,
            'p-1': 2, 'p+0': 2, 'p+1': 2,
        },

        "fakescf": True,
        "fake_charge": 2,
        "fake_spin": 6,
        "fake_irrep_nelec": {'s+0': 7,
                             'p-1': 4, 'p+0': 4, 'p+1': 4,
                             'd-2': 1, 'd-1': 1, 'd+0': 1, 'd+1': 1, 'd+2': 1, },
    },
]


BASIS = ["ccpvdz",         "ccpvtz",         "ccpvqz",
         "aug-ccpvdz",     "aug-ccpvtz",     "aug-ccpvqz",
         "unc-ccpvdz",     "unc-ccpvtz",     "unc-ccpvqz",
         "unc-aug-ccpvdz", "unc-aug-ccpvtz", "unc-aug-ccpvqz", ]

# BASIS = [
#     "ccpvdz"
# ]

DEFAULT_FLOAT_FORMAT = getattr(__config__, 'fcidump_float_format', ' %.16g')
TOL = getattr(__config__, 'fcidump_write_tol', 1e-15)


def dump_ints(filename, mf, mo_coeff, orbsym, tol=TOL):
    # mol = mf.mol

    h1e = reduce(np.dot, (mo_coeff.T, mf.get_hcore(), mo_coeff))
    if mf._eri is None:
        if getattr(mf, 'exxdiv', None):  # PBC system
            eri = mf.with_df.ao2mo(mo_coeff)
        else:
            eri = ao2mo.full(mf.mol, mo_coeff)
    else:  # Handle cached integrals or customized systems
        eri = ao2mo.full(mf._eri, mo_coeff)
    # orbsym = getattr(mo_coeff, 'orbsym', None)
    # if molpro_orbsym and orbsym is not None:
    #     orbsym = [ORBSYM_MAP[mol.groupname][i] for i in orbsym]
    nuc = mf.energy_nuc()
    tools.fcidump.from_integrals(
        filename, h1e, eri, h1e.shape[0], mf.mol.nelec, nuc, 0, orbsym, tol, DEFAULT_FLOAT_FORMAT)


def OrbSymInfo_MO(Mol, mo_coeff):
    IRREP_MAP = {}
    nsym = len(Mol.irrep_name)
    for i in range(nsym):
        IRREP_MAP[Mol.irrep_name[i]] = i
    print(IRREP_MAP)

    OrbSym = pyscf.symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb,
                                       mo_coeff)
    IrrepOrb = []
    for i in range(len(OrbSym)):
        IrrepOrb.append(symm.irrep_name2id(Mol.groupname, OrbSym[i]))
    return IrrepOrb


if __name__ == '__main__':

    Res = {}

    for task in TaskInfo:

        Res[task["atom"]] = {}

        for basis in BASIS:

            Mol_D2h = pyscf.gto.Mole()
            Mol_D2h.atom = '''
            %s     0.0000      0.0000  0.0000
            ''' % (task["atom"])
            Mol_D2h.basis = basis
            Mol_D2h.symmetry = 'd2h'
            Mol_D2h.spin = task["spin"]
            Mol_D2h.charge = task["charge"]
            Mol_D2h.verbose = 1
            Mol_D2h.unit = 'angstorm'
            Mol_D2h.build()

            Mol_Dooh = pyscf.gto.Mole()
            Mol_Dooh.atom = '''
            %s     0.0000      0.0000  0.0000
            ''' % (task["atom"])
            Mol_Dooh.basis = basis
            Mol_Dooh.symmetry = 'dooh'
            Mol_Dooh.spin = task["spin"]
            Mol_Dooh.charge = task["charge"]
            Mol_Dooh.verbose = 1
            Mol_Dooh.unit = 'angstorm'
            Mol_Dooh.build()

            Mol = pyscf.gto.Mole()
            Mol.atom = '''
            %s     0.0000      0.0000  0.0000
            ''' % (task["atom"])
            Mol.basis = basis
            Mol.symmetry = True
            Mol.spin = task["spin"]
            Mol.charge = task["charge"]
            Mol.verbose = 1
            Mol.unit = 'angstorm'
            Mol.build()

            SCF = pyscf.scf.newton(pyscf.scf.ROHF(Mol))
            SCF.max_cycle = 32
            SCF.conv_tol = 1e-12
            SCF.run()

            print("SCF is called")

            if task["atom"] in ['H', "Li", "Na", "Be", "Cr"]:

                SCF = pyscf.scf.ROHF(Mol)
                SCF.max_cycle = 128
                SCF.conv_tol = 1e-12
                SCF.run()

                DumpFileName = task["atom"] + "_" + basis
                # pyscf_util.dump_cmoao(DumpFileName, SCF.mo_coeff)
                orbsym = OrbSymInfo_MO(Mol_D2h, SCF.mo_coeff)
                DumpFileName = "FCIDUMP_" + task["atom"] + "_" + basis
                # Res[task["atom"]][basis] = SCF.e_tot
                pyscf.tools.fcidump.from_mo(
                    Mol_D2h, DumpFileName, SCF.mo_coeff, orbsym)

                print("atm %s basis %s energy = %15.8f" % (
                      task["atom"], basis, SCF.e_tot))

                continue

            if "fakescf" in task.keys():
                if task["fakescf"]:
                    Mol_fake = pyscf.gto.Mole()
                    Mol_fake.atom = '''
            %s     0.0000      0.0000  0.0000
            ''' % (task["atom"])
                    Mol_fake.basis = basis
                    Mol_fake.symmetry = True
                    Mol_fake.spin = task["fake_spin"]
                    Mol_fake.charge = task["fake_charge"]
                    Mol_fake.verbose = 1
                    Mol_fake.unit = 'angstorm'
                    Mol_fake.build()
                    SCF_fake = pyscf.scf.newton(pyscf.scf.ROHF(Mol_fake))
                    SCF_fake.irrep_nelec = task["fake_irrep_nelec"]
                    SCF_fake.max_cycle = 32
                    SCF_fake.conv_tol = 1e-9
                    SCF_fake.run()
                    print("FAKED is called!")
                    SCF.mo_coeff = copy.deepcopy(SCF_fake.mo_coeff)

            norb = task["minimal_cas"]["norb"]
            nelec = task["minimal_cas"]["nelec"]
            iCISCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)

            mo_init = pyscf.mcscf.sort_mo_by_irrep(
                iCISCF_Driver, SCF.mo_coeff, task["cas_symm"], task["core_symm"])  # right!

            # print(mo_init.shape)

            if "iCI_state" in task.keys():
                iCISCF_Driver.fcisolver = iCISCF.iCI(
                    mol=Mol_D2h, cmin=0.0, state=task["iCI_state"],  mo_coeff=mo_init)
            else:
                iCISCF_Driver = get_state_averaged_CASSCF(
                    Mol_D2h, iCISCF_Driver, task["state"])

            iCISCF_Driver.internal_rotation = False
            iCISCF_Driver.conv_tol = 1e-12
            iCISCF_Driver.max_cycle_macro = 32
            iCISCF_Driver.kernel(mo_coeff=mo_init)

            print("After kernel iciscf")

            DumpFileName = task["atom"] + "_" + basis
            # pyscf_util.dump_cmoao(DumpFileName, iCISCF_Driver.mo_coeff)

            # dump FCIDUMP

            SCF.mo_coeff = iCISCF_Driver.mo_coeff
            orbsym = OrbSymInfo_MO(Mol_D2h, iCISCF_Driver.mo_coeff)
            DumpFileName = "FCIDUMP_" + task["atom"] + "_" + basis
            pyscf.tools.fcidump.from_mo(
                Mol_D2h, DumpFileName, SCF.mo_coeff, orbsym)

            print("atm %s basis %s energy = %15.8f",
                  task["atom"], basis, iCISCF_Driver.e_tot)

            # Res[task["atom"]][basis] = iCISCF_Driver.e_tot

    with open("atm.HF.data", "wb") as f:
        pickle.dump(Res, f)

    print(Res)
