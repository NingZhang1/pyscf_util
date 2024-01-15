# coding=UTF-8

import pyscf
import os
import sys
from pyscf import tools
from pyscf import symm
import numpy as np
from Util_File import Dump_Cmoao
import iCISCF
import copy

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
        "basis":'ccpvdz',
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

        "basis":'ccpvdz',

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
        "atom": "N",
        "charge": 0,
        "spin": 3,

        "state": [[3, 's+0', 1],
                  ],

        "iCI_state": [[3, 4, 1, [1]]],

        "basis":'ccpvdz',

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
        "atom": "O",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "iCI_state": [[2, 1, 1, [1]], [2, 2, 1, [1]], [2, 3,  1, [1]]],

        "basis":'ccpvdz',

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
        "fake_irrep_nelec": {'s+0': 4, 'p+0': 2, 'p-1': 2, 'p+1': 2, },
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

        "basis":'ccpvdz',

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
        "fake_irrep_nelec": {'s+0': 4, 'p+0': 2, 'p-1': 2, 'p+1': 2, },
    },

    {
        "atom": "Cr",
        "charge": 0,
        "spin": 6,

        "state": [[6, 's+0', 1],],

        "iCI_state": [[6, 0, 1, [1]]],

        "basis":'ccpvdz-dk',

        "minimal_cas": {
            "norb": 6,
            "nelec": 6,
        },

        "cas_symm":  {
            's+0': 1, 'd+0': 1, 'd+1': 1, 'd-1': 1, 'd+2': 1, 'd-2': 1,
        },

        "core_symm":  {
            's+0': 3,
            'p+0': 2, 'p-1': 2, 'p+1': 2,
        },

        "fakescf": False,
        # "fake_charge": 0,
        # "fake_spin": 0,
        # "fake_irrep_nelec": {'s+0': 4, 'p+0': 2, 'p-1': 2, 'p+1': 2, },
    },
]

if __name__ == '__main__':

    for task in TaskInfo:

        # if task["atom"] not in ["C", "H"]:
        if task["atom"] not in ["Cr"]:
            continue

        Mol_Dooh = pyscf.gto.Mole()
        Mol_Dooh.atom = '''
        %s     0.0000      0.0000  0.0000
        ''' % (task["atom"])
        Mol_Dooh.basis = task["basis"]
        Mol_Dooh.symmetry = 'dooh'
        Mol_Dooh.spin = task["spin"]
        Mol_Dooh.charge = task["charge"]
        Mol_Dooh.verbose = 5
        # Mol.verbose = 1
        # Mol.verbose = 0
        Mol_Dooh.unit = 'angstorm'
        Mol_Dooh.build()

        Mol = pyscf.gto.Mole()
        Mol.atom = '''
        %s     0.0000      0.0000  0.0000
        ''' % (task["atom"])
        Mol.basis = task["basis"]
        Mol.symmetry = True
        Mol.spin = task["spin"]
        Mol.charge = task["charge"]
        Mol.verbose = 5
        # Mol.verbose = 1
        # Mol.verbose = 0
        Mol.unit = 'angstorm'
        Mol.build()

        SCF = pyscf.scf.newton(pyscf.scf.sfx2c(pyscf.scf.ROHF(Mol)))
        SCF.max_cycle = 32
        SCF.conv_tol = 1e-9
        SCF.run()

        SCF.analyze()

        if task["atom"] == 'H':
            DumpFileName = task["atom"] + "_" + str(task["charge"]) + "_" +\
                task["basis"]
            Dump_Cmoao(DumpFileName, SCF.mo_coeff)
            continue

        if "fakescf" in task.keys():
            if task["fakescf"]:
                Mol_fake = pyscf.gto.Mole()
                Mol_fake.atom = '''
        %s     0.0000      0.0000  0.0000
        ''' % (task["atom"])
                Mol_fake.basis = task["basis"]
                Mol_fake.symmetry = True
                Mol_fake.spin = task["fake_spin"]
                Mol_fake.charge = task["fake_charge"]
                Mol_fake.verbose = 5
                Mol_fake.unit = 'angstorm'
                Mol_fake.build()
                SCF_fake = pyscf.scf.newton(
                    pyscf.scf.sfx2c(pyscf.scf.ROHF(Mol_fake)))
                SCF_fake.irrep_nelec = task["fake_irrep_nelec"]
                SCF_fake.max_cycle = 32
                SCF_fake.conv_tol = 1e-9
                SCF_fake.run()
                SCF.mo_coeff = copy.deepcopy(SCF_fake.mo_coeff)

        # print("HF      energy", BondLength, 0, SCF.energy_tot())

        norb = task["minimal_cas"]["norb"]
        nelec = task["minimal_cas"]["nelec"]
        iCISCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)

        # mo_init = SCF.mo_coeff
        mo_init = copy.deepcopy(SCF.mo_coeff)
        # mo_init[:, 9] = SCF.mo_coeff[:, 14]
        # mo_init[:, 14] = SCF.mo_coeff[:, 9]
        # mo_init = pyscf.mcscf.caslst_by_irrep(
        #     iCISCF_Driver, SCF.mo_coeff, task["cas_symm"], task["core_symm"])  # right!

        print(mo_init.shape)

        if "iCI_state" in task.keys():
            iCISCF_Driver.fcisolver = iCISCF.iCI(
                mol=Mol_Dooh, cmin=0.0, state=task["iCI_state"],  mo_coeff=mo_init)
        else:
            iCISCF_Driver = get_state_averaged_CASSCF(
                Mol, iCISCF_Driver, task["state"])

        iCISCF_Driver.internal_rotation = False
        iCISCF_Driver.conv_tol = 1e-12
        iCISCF_Driver.max_cycle_macro = 128
        iCISCF_Driver.kernel(mo_coeff=mo_init)

        print("CASSCF 6 energy ", 0, iCISCF_Driver.e_tot)

        DumpFileName = task["atom"] + "_" + str(task["charge"]) + "_" +\
            task["basis"]
        Dump_Cmoao(DumpFileName, iCISCF_Driver.mo_coeff)
