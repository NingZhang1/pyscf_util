# coding=UTF-8

import pyscf
from pyscf import tools
from Util_File import ReadIn_Cmoao, Dump_Cmoao
from Util_Orb import Analysis_Orb_Comp, _construct_atm_bas

# construct ao

atm_bas = {
    "H": {
        "1s": [0],
        "2s": [1],
        "2p": [2, 3, 4],
        "nao": 5,
        "basis": "ccpvdz",
        "cmoao": None,
    },
    "C": {
        "1s": [0],
        "2s": [1],
        # "2p": [2, 3, 4],
        "2px": [4],
        "2py": [2],
        "2pz": [3],
        "3p": [5, 6, 7],
        "3s": [8],
        "3d": [9, 10, 11, 12, 13],
        "nao": 14,
        "basis": "ccpvdz",
        "cmoao": None,
    },
}

dirname = "/home/nzhangcaltech/GitHub_Repo/pyscf_util/Test/AtmOrb"

for atom in ["H", "C"]:
    atm_bas[atom]["cmoao"] = ReadIn_Cmoao(
        dirname+"/"+"%s_0_%s" % (atom, atm_bas[atom]["basis"]), atm_bas[atom]["nao"])

print(atm_bas["H"]["cmoao"])
print(atm_bas["C"]["cmoao"])
print(atm_bas["C"]["2pz"])

if __name__ == '__main__':
    Mol = pyscf.gto.Mole()
    Mol.atom = '''
C     0.0000    1.396792    0.0000
C     0.0000    -1.396792    0.0000
C     1.209657    0.698396    0.0000
C     -1.209657    -0.698396    0.0000
C    -1.209657    0.698396    0.0000
C     1.209657    -0.698396    0.0000
H     0.0000    2.484212    0.0000
H     2.151390    1.242106    0.0000
H     -2.151390    -1.242106    0.0000
H     -2.151390    1.242106    0.0000
H     2.151390    -1.242106    0.0000
H     0.0000    -2.484212    0.0000
'''

    Mol.basis = 'ccpvdz'
    Mol.symmetry = 'D2h'
    Mol.spin = 0
    Mol.build()
    Mol.verbose = 4
    SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
    SCF.run()
    # extract info from SCF
    # TraInt.AO_output("AoInt.dat", Mol)
    # Result = TraInt.AO_ReadIn("AoInt.dat")
    # DumpFileName = "FCIDUMP"
    # tools.fcidump.from_scf(SCF,DumpFileName,1e-10)

    SCF.analyze()

    # reorder MO

    # generate ### molden

    pyscf.tools.molden.from_mo(Mol, "benzene.molden", SCF.mo_coeff)

    #Analysis_Orb_Comp(Mol, SCF.mo_coeff, 0, Mol.nelectron//2,
    #                  atm_bas, tol=0.3, with_distinct_atm=True)
    # Analysis_Orb_Comp(Mol, SCF.mo_coeff, Mol.nelectron//2, Mol.nao,
    #                   atm_bas, tol=0.3, with_distinct_atm=True)
    # pz_basis = _construct_atm_bas(Mol, atm_bas, ["C.2pz"], orthogonalize=True)
    # ovlp = SCF.get_ovlp()
    # mo_virtual = SCF.mo_coeff[:, Mol.nelectron:]

    OCC_ORDER = [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 17, 18, 16, 19, 20
    ]

    mo_coeff = SCF.mo_coeff[:, :Mol.nelectron//2]
    mo_coeff = mo_coeff[:, OCC_ORDER]
    SCF.mo_coeff[:, :Mol.nelectron//2] = mo_coeff

    ## dump cmoao 

    Dump_Cmoao("benzene_ccpvdz_hf", SCF.mo_coeff)
    orbsym = pyscf.symm.label_orb_symm(Mol, Mol.irrep_id, Mol.symm_orb, SCF.mo_coeff)

    tools.fcidump.from_mo(Mol, "FCIDUMP", SCF.mo_coeff, orbsym, 1e-8)

    # calculate the overlap between virtual orbitals and pz orbitals

    # from functools import reduce
    # import numpy
    # ovlp_pz_vir = reduce(numpy.dot, (pz_basis.T, ovlp, mo_virtual))
    # print(reduce(numpy.dot, (pz_basis.T, ovlp, pz_basis)))
    # ## perform SVD and transform the virtual orbitals to pz basis
    # u, s, vh = numpy.linalg.svd(ovlp_pz_vir)
    # print(s)
    # print(vh.shape)
    # mo_virtual_pz = numpy.dot(mo_virtual, vh.T)
    # SCF.mo_coeff[:, Mol.nelectron:] = mo_virtual_pz
    # Analysis_Orb_Comp(Mol, SCF.mo_coeff, Mol.nelectron//2, Mol.nao,
    #                     atm_bas, tol=0.3, with_distinct_atm=True)
    # pyscf.tools.molden.from_mo(Mol, "benzene2.molden", SCF.mo_coeff)
