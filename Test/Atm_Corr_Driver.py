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


APP = None


#### the initial guess must be HF ####

TaskInfo = [

    {
        "atom": "Li",
        "charge": 0,
        "spin": 1,
        "state": [[1, 's+0', 1], ],

        "ici_task": {
            "full":
            {
                "segment": "0 1 1 0 %d 0",
                "left": 2,
                "nvalelec": 1,
                "task": "1 0 1 1",
            }
        },
    },

    {
        "atom": "Na",
        "charge": 0,
        "spin": 1,
        "state": [[1, 's+0', 1], ],

        "ici_task": {
            "full":
            {
                "segment": "0 5 1 0 %d 0",
                "left": 6,
                "nvalelec": 1,
                "task": "1 0 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 4 1 0 %d 0",
                "left": 6,
                "nvalelec": 1,
                "task": "1 0 1 1",
            }
        },
    },

    {
        "atom": "Be",
        "charge": 0,
        "spin": 0,
        "state": [[0, 's+0', 1], ],

        "ici_task": {
            "full":
            {
                "segment": "0 1 1 0 %d 0",
                "left": 2,
                "nvalelec": 2,
                "task": "0 0 1 1",
            }
        },
    },

    {
        "atom": "B",
        "charge": 0,
        "spin": 1,

        "state": [[1, 'b1u', 1],
                  [1, 'b2u', 1],
                  [1, 'b3u', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 2 1 2 %d 0",
                "left": 5,
                "nvalelec": 1,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            }
        },
    },

    {
        "atom": "Al",
        "charge": 0,
        "spin": 1,

        "state": [[1, 'b1u', 1],
                  [1, 'b2u', 1],
                  [1, 'b3u', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 6 1 2 %d 0",
                "left": 9,
                "nvalelec": 1,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 5 1 2 %d 0",
                "left": 9,
                "nvalelec": 1,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            },
            "fzc_1s2s2p":
            {
                "segment": "5 1 1 2 %d 0",
                "left": 9,
                "nvalelec": 1,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            }
        },

    },

    {
        "atom": "C",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 2 1 2 %d 0",
                "left": 5,
                "nvalelec": 2,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 1 1 2 %d 0",
                "left": 5,
                "nvalelec": 2,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
        },
    },

    {
        "atom": "Si",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 6 1 2 %d 0",
                "left": 9,
                "nvalelec": 2,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 5 1 2 %d 0",
                "left": 9,
                "nvalelec": 2,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
            "fzc_1s2s2p":
            {
                "segment": "5 1 1 2 %d 0",
                "left": 9,
                "nvalelec": 2,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
        },
    },

    {
        "atom": "O",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 2 1 2 %d 0",
                "left": 5,
                "nvalelec": 4,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 1 1 2 %d 0",
                "left": 5,
                "nvalelec": 4,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
        },
    },

    {
        "atom": "S",
        "charge": 0,
        "spin": 2,

        "state": [[2, 'p+0', 1],
                  [2, 'p+1', 1],
                  [2, 'p-1', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 6 1 2 %d 0",
                "left": 9,
                "nvalelec": 4,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 5 1 2 %d 0",
                "left": 9,
                "nvalelec": 4,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            },
            "fzc_1s2s2p":
            {
                "segment": "5 1 1 2 %d 0",
                "left": 9,
                "nvalelec": 4,
                "task": "2 1 1 1 2 2 1 1 2 3 1 1",
            }
        },
    },

    {
        "atom": "N",
        "charge": 0,
        "spin": 3,

        "state": [[3, 's+0', 1],],

        "ici_task": {
            "full":
            {
                "segment": "0 2 1 2 %d 0",
                "left": 5,
                "nvalelec": 3,
                "task": "3 0 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 1 1 2 %d 0",
                "left": 5,
                "nvalelec": 3,
                "task": "3 0 1 1",
            },
        },
    },

    {
        "atom": "P",
        "charge": 0,
        "spin": 3,

        "state": [[3, 's+0', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 6 1 2 %d 0",
                "left": 9,
                "nvalelec": 3,
                "task": "3 0 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 5 1 2 %d 0",
                "left": 9,
                "nvalelec": 3,
                "task": "3 0 1 1",
            },
            "fzc_1s2s2p":
            {
                "segment": "5 1 1 2 %d 0",
                "left": 9,
                "nvalelec": 3,
                "task": "3 0 1 1",
            }
        },
    },

    {
        "atom": "F",
        "charge": 0,
        "spin": 1,

        "state": [[1, 'p+0', 1],
                  [1, 'p+1', 1],
                  [1, 'p-1', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 2 1 2 %d 0",
                "left": 5,
                "nvalelec": 5,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 1 1 2 %d 0",
                "left": 5,
                "nvalelec": 5,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            }
        },
    },

    {
        "atom": "Cl",
        "charge": 0,
        "spin": 1,

        "state": [[1, 'p+0', 1],
                  [1, 'p+1', 1],
                  [1, 'p-1', 1],
                  ],

        "ici_task": {
            "full":
            {
                "segment": "0 6 1 2 %d 0",
                "left": 9,
                "nvalelec": 5,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 5 1 2 %d 0",
                "left": 9,
                "nvalelec": 5,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            },
            "fzc_1s2s2p":
            {
                "segment": "5 1 1 2 %d 0",
                "left": 9,
                "nvalelec": 5,
                "task": "1 5 1 1 1 6 1 1 1 7 1 1",
            }
        },
    },

    #### 过渡金属 ####

    # Cr

    {
        "atom": "Cr",
        "charge": 0,
        "spin": 6,

        "state": [[6, 's+0', 1],],

        "ici_task": {
            "full":
            {
                "segment": "0 9 3 3 %d 0",
                "left": 15,
                "nvalelec": 6,
                "task": "6 0 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 8 3 3 %d 0",
                "left": 15,
                "nvalelec": 6,
                "task": "6 0 1 1",
            },
            "fzc_1s2s2p":
            {
                "segment": "5 4 3 3 %d 0",
                "left": 15,
                "nvalelec": 6,
                "task": "6 0 1 1",
            },
            "fzc_1s2s2p3s3p":
            {
                "segment": "9 0 3 3 %d 0",
                "left": 15,
                "nvalelec": 6,
                "task": "6 0 1 1",
            },
        }
    },

    # Fe

    {
        "atom": "Fe",
        "charge": 0,
        "spin": 4,

        "state": [[4, 'd+0', 1], [4, 'd+1', 1], [4, 'd+2', 1], [4, 'd-1', 1], [4, 'd-2', 1],],

        "ici_task": {
            "full":
            {
                "segment": "0 9 3 3 %d 0",
                "left": 15,
                "nvalelec": 8,
                "task": "4 0 2 1 1 4 1 1 1 4 2 1 1 4 3 1 1",
            },
            "fzc_1s":
            {
                "segment": "1 8 3 3 %d 0",
                "left": 15,
                "nvalelec": 8,
                "task": "4 0 2 1 1 4 1 1 1 4 2 1 1 4 3 1 1",
            },
            "fzc_1s2s2p":
            {
                "segment": "5 4 3 3 %d 0",
                "left": 15,
                "nvalelec": 8,
                "task": "4 0 2 1 1 4 1 1 1 4 2 1 1 4 3 1 1",
            },
            "fzc_1s2s2p3s3p":
            {
                "segment": "9 0 3 3 %d 0",
                "left": 15,
                "nvalelec": 8,
                "task": "4 0 2 1 1 4 1 1 1 4 2 1 1 4 3 1 1",
            },
        }
    },
]


BASIS = ["ccpvdz",         "ccpvtz",         "ccpvqz",
         "aug-ccpvdz",     "aug-ccpvtz",     "aug-ccpvqz",
         "unc-ccpvdz",     "unc-ccpvtz",     "unc-ccpvqz",
         "unc-aug-ccpvdz", "unc-aug-ccpvtz", "unc-aug-ccpvqz", ]


def _Generate_InputFile_SiCI(File,
                             Segment,
                             nelec_val,
                             rotatemo=0,  # HF orbitals!
                             cmin=0.0,
                             perturbation=1,
                             dumprdm=0,
                             relative=0,
                             Task=None,
                             tol=1e-8,
                             selection=1
                             ):
    """Generate the input file
    """
    with open(File, "w") as f:
        f.write("nsegment=%s\n" % Segment)
        f.write("nvalelec=%d\n" % nelec_val)
        f.write("rotatemo=%d\n" % rotatemo)
        f.write("cmin=%s\n" % cmin)
        f.write("perturbation=%d 0\n" % perturbation)
        f.write("dumprdm=%d\n" % dumprdm)
        f.write("relative=%d\n" % relative)
        f.write("task=%s\n" % Task)
        f.write("etol=%e\n" % tol)
        f.write("selection=%d\n" % selection)


if __name__ == '__main__':

    Res = {}

    for task in TaskInfo:

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

            nao = Mol_D2h.nao_nr()

            DumpFileName = "FCIDUMP_" + task["atom"] + "_" + basis

            for taskname in task["ici_task"].keys():
                InputName = task["atom"] + "_" + \
                    basis + "_" + taskname + ".inp"
                OutputName = task["atom"] + "_" + \
                    basis + "_" + taskname + ".out"

                segment = task["ici_task"][taskname]["segment"]
                left = task["ici_task"][taskname]["left"]
                nvalelec = task["ici_task"][taskname]["nvalelec"]
                ici_task_info = task["ici_task"][taskname]["task"]

                _Generate_InputFile_SiCI(InputName,
                                         Segment=segment % (nao - left),
                                         nelec_val=nvalelec,
                                         cmin="1e-5",
                                         Task=ici_task_info)

                # performan calculation

                os.system("%s %s %s 1>%s 2>%s.err" %
                          (APP, InputName, DumpFileName, OutputName, OutputName))
