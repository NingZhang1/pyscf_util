import copy
from pyscf import tools
import pyscf
import Util_CoreExt

Mole_Geometry = {
    "SiH4":     '''
Si  0.000000  0.000000  0.000000
H   0.854400  0.854400  0.854400
H  -0.854400 -0.854400  0.854400
H  -0.854400  0.854400 -0.854400
H   0.854400 -0.854400 -0.854400
''',
    "PH3":     '''
P   0.00000000 0.59312580  0.76512855
H   1.02732402 0.00000000  0.00000000
H   0.00000000 1.77937741  0.00000000
H  -1.02732402 0.00000000  0.00000000
''',
    "H2S":      '''
S  0.000000  0.000000  -0.1855710
H  0.000000  0.9608222  0.7422841
H  0.000000 -0.9608222  0.7422841
''',
    "HCl":     '''
Cl  0.000000 0.000000 0.000000
H   0.000000 0.000000 1.274400
''',
}

##### GEOMETRY OPTIMIZED VIA B3LYP aug-ccpvtz in Bohr #####

Mole_Geometry_Bohr = {
    "SiH4":     '''
Si -1.53358317e-13  4.83937748e-13 -4.36697869e-13
H   1.61747223e+00  1.61747223e+00  1.61747223e+00
H  -1.61747223e+00 -1.61747223e+00  1.61747223e+00
H  -1.61747223e+00  1.61747223e+00 -1.61747223e+00
H   1.61747223e+00 -1.61747223e+00 -1.61747223e+00
''',
    "PH3":     '''
P   8.31537687e-14  1.12084654e+00  1.44917485e+00
H   1.95612313e+00 -8.52643878e-03 -1.09740995e-03
H  -1.53388980e-13  3.37958764e+00 -1.09662522e-03
H  -1.95612313e+00 -8.52643878e-03 -1.09740995e-03
''',
    "H2S":      '''
S  0.          0.         -0.35232488
H  0.          1.83662056  1.40353691
H  0.         -1.83662056  1.40353691
''',
    "HCl":     '''
Cl  0.          0.         -0.00868048
H   0.          0.          2.41694745
''',
}

BASIS = [
    "aug-cc-pVDZ-DK",
    # "aug-cc-pVTZ-DK",
    # "unc-aug-cc-pVDZ-DK",
    # "unc-aug-cc-pVTZ-DK",
]

basis_test = "cc-pVDZ"

#### describe the task ####


def _get_gt_nsegment(mol, cas_orb):
    """Get the nsegment for the ground state
    """

    nocc = mol.nelectron // 2

    nsegment = "0 %d %d %d %d 0" % (
        nocc-cas_orb[0], cas_orb[0], cas_orb[1], mol.nao - nocc - cas_orb[1])

    return nsegment


def _dump_FCIDUMP(mol, hf, filename, with_breit=False):

    # n2c = hf.mo_coeff.shape[1] // 2
    # mo = copy.deepcopy(hf.mo_coeff)
    # orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
    # tools.fcidump.from_mo(mol, filename, mo, orbsym)

    from Util_Rela4C import FCIDUMP_Rela4C

    FCIDUMP_Rela4C(mol, hf, with_breit=with_breit, filename=filename, mode="outcore")

Vert_Ext = {
    "SiH4": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "Si1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "0 0 1 1",
            "type": "Si1s",
        },
    },
    "PH3": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "P1s": {
            "loc": [0, 1],
            "cas": [4, 5],
            "task": "0 0 2 1 1 0 1 1 1",
            "type": "P1s",
        },
    },
    "H2S": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "S1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            # "task": "0 0 1 1 0 3 1 1 2 0 1 1 2 3 1 1", # seems to be problematic
            "task": "0 0 3 1 1 1 0 2 1 1 0 3 2 1 1",
            "type": "S1s",
        },
    },
    "HCl": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "Cl1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            # "task": "0 0 2 1 1 0 2 1 1 0 3 1 1 2 0 2 1 1 2 2 1 1 2 3 1 1",
            "task": "0 0 2 1 1 0 2 1 1 0 3 1 1",
            "type": "Cl1s",
        },
    },
}

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


CMIN = "1e-4"

from pyscf import scf

if __name__ == "__main__":

    ### test the vert ext ###

    for mole, info in Vert_Ext.items():

        # if mole != "H2S":
        #     continue

        for basis in BASIS:
            mol = pyscf.gto.M(
                verbose=10,
                atom=Mole_Geometry_Bohr[mole],
                basis=basis,
                spin=0,
                charge=0,
                symmetry=True,
                unit="Bohr",
            )
            mol.build()

            if mol.groupname == "Dooh":
                mol.symmetry = "D2h"
                mol.build()
            elif mol.groupname == "Coov":
                mol.symmetry = "C2v"
                mol.build()

            ######## Run 4C with Coulomb Only ########

            mf = scf.dhf.RDHF(mol)
            mf.conv_tol = 1e-10
            mf.kernel()

            n2c = mf.mo_coeff.shape[1] // 2

            print(mf.mo_energy[n2c:])

            for task, subinfo in info.items():
                if task == "gt":

                    _dump_FCIDUMP(mol, mf, "FCIDUMP_%s_gt_%s_Vert" %
                                  (mole, basis))

                    # generate the input file

                    _Generate_InputFile_SiCI(
                        "input_%s_gt_%s_Vert" % (mole, basis),
                        _get_gt_nsegment(mol, subinfo["cas"]),
                        subinfo["cas"][0] * 2,
                        Task=subinfo["task"],
                        cmin=CMIN,
                    )

                else:

                    res = Util_CoreExt._dump_CoreExt_FCIDUMP(
                        mol, mf, [subinfo], "FCIDUMP_Vert_%s_%s_" % (mole, basis), Rela4C=True)
                    nsegment = res[0]["nsegment"]

                    ncore = subinfo["loc"][1] - subinfo["loc"][0]

                    _Generate_InputFile_SiCI(
                        "input_%s_%s_%s_Vert" % (
                            mole, subinfo["type"], basis),
                        nsegment,
                        subinfo["cas"][0] * 2 + ncore * 2,
                        Task=subinfo["task"],
                        cmin=CMIN,
                    )
            
            ######## Run 4C with Breit Only ########