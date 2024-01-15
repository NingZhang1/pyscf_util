# coding=UTF-8

import pyscf
from pyscf import tools
from Util_File import ReadIn_Cmoao, Dump_Cmoao, ReadIn_SpinRDM1
from Util_Orb import Analysis_Orb_Comp, _construct_atm_bas
from Util_Pic import draw_heatmap

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
        "2p": [2, 3, 4],
        # "2px": [4],
        # "2py": [2],
        # "2pz": [3],
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
# print(atm_bas["C"]["2pz"])

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

    mo_coeff_hf = ReadIn_Cmoao("benzene_ccpvdz_hf", Mol.nao)

    dm1 = ReadIn_SpinRDM1("rdm1.csv", Mol.nao, 1, True)

    no_mocoeff = ReadIn_Cmoao("cmoao", Mol.nao)

    import numpy

    print(numpy.diag(dm1))

    MO_ORDER = [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 29, 23, 24, 25, 26, 27, 28
    ]

    no_mocoeff_ordered = no_mocoeff[:, :30]
    no_mocoeff_ordered = no_mocoeff_ordered[:, MO_ORDER]
    no_mocoeff[:, :30] = no_mocoeff_ordered

    no_ao_coeff = numpy.dot(mo_coeff_hf, no_mocoeff)

    # check orthogonality

    ovlp = SCF.get_ovlp()

    from functools import reduce

    print(numpy.allclose(
        reduce(numpy.dot, (no_ao_coeff.T, ovlp, no_ao_coeff)), numpy.eye(Mol.nao)))

    Analysis_Orb_Comp(Mol, no_ao_coeff, Mol.nelectron//2-3, Mol.nelectron//2+3,
                      atm_bas, tol=0.1, with_distinct_atm=True)

    pyscf.tools.molden.from_mo(Mol, "benzene2.molden", no_ao_coeff)

    ### localization ###

    Mol.symmetry = False
    Mol.build()

    from Util_Orb import split_loc_given_range_NoSymm

    ### generate a random guess ###

    nbas = Mol.nelectron//2 - 3 - 6
    random_uni = numpy.random.uniform(-1, 1, (nbas, nbas))
    uni, _ = numpy.linalg.qr(random_uni)

    no_ao_coeff[:, 6:Mol.nelectron//2 -
                3] = numpy.dot(no_ao_coeff[:, 6:Mol.nelectron//2-3], uni)

    # no_ao_coeff = split_loc_given_range_NoSymm(Mol, no_ao_coeff, 0, 6) # C 1s
    no_ao_coeff = split_loc_given_range_NoSymm(
        Mol, no_ao_coeff, 0, Mol.nelectron//2-3)  # 局域化的有问题 !
    no_ao_coeff = split_loc_given_range_NoSymm(
        Mol, no_ao_coeff, Mol.nelectron//2+3, Mol.nao)

    print(numpy.allclose(
        reduce(numpy.dot, (no_ao_coeff.T, ovlp, no_ao_coeff)), numpy.eye(Mol.nao)))

    mo_occ = numpy.zeros(Mol.nao, dtype=numpy.int)
    for i in range(Mol.nelectron//2):
        mo_occ[i] = 2

    dm1 = pyscf.scf.hf.make_rdm1(no_ao_coeff, mo_occ)

    SCF.mo_coeff = no_ao_coeff

    fock = SCF.get_fock(dm=dm1)  # in AO basis

    fock_mo = reduce(numpy.dot, (no_ao_coeff.T, fock, no_ao_coeff))

    print(numpy.diag(fock_mo))

    ### reordering by fock ###

    fock_mo = numpy.diag(fock_mo)
    fock_mo = numpy.diag(fock_mo)

    e, c = numpy.linalg.eigh(fock_mo)

    no_ao_coeff = numpy.dot(no_ao_coeff, c)

    # build fock again

    dm1 = pyscf.scf.hf.make_rdm1(no_ao_coeff, mo_occ)
    fock = SCF.get_fock(dm=dm1)  # in AO basis
    

    print(numpy.diag(fock_mo))
    # draw_heatmap(numpy.log(numpy.abs(fock_mo)), list(range(Mol.nao)), list(range(Mol.nao)), x_label="MO", y_label="MO", vmax=1.3, vmin=-10)
    fock_mo = reduce(numpy.dot, (no_ao_coeff.T, fock, no_ao_coeff))

    pyscf.tools.molden.from_mo(Mol, "benzene3.molden", no_ao_coeff)

    Analysis_Orb_Comp(Mol, no_ao_coeff, 0, Mol.nelectron//2-3,
                      atm_bas, tol=0.1, with_distinct_atm=True)

    Res = Analysis_Orb_Comp(Mol, no_ao_coeff, Mol.nelectron//2+3, Mol.nao,
                            atm_bas, tol=0.1, with_distinct_atm=True)

    print(Res)

    List = {
        "H_1": [],
        "H_2": [],
        "H_3": [],
        "H_4": [],
        "H_5": [],
        "H_6": [],
        "C_1": [],
        "C_2": [],
        "C_3": [],
        "C_4": [],
        "C_5": [],
        "C_6": [],
        "anti_bonding"  : [],
    }

    for data in Res:
        if isinstance(data['key'], list):
            List["anti_bonding"].append(data['orbindx'])
        else:
            List[data['key'][:3]].append(data['orbindx'])
    
    Reordering = []
    Reordering.extend(List["anti_bonding"])
    Reordering.extend(List["H_1"])
    Reordering.extend(List["H_2"])
    Reordering.extend(List["H_3"])
    Reordering.extend(List["H_4"])
    Reordering.extend(List["H_5"])
    Reordering.extend(List["H_6"])
    Reordering.extend(List["C_1"])
    Reordering.extend(List["C_2"])
    Reordering.extend(List["C_3"])
    Reordering.extend(List["C_4"])
    Reordering.extend(List["C_5"])
    Reordering.extend(List["C_6"])

    print(Reordering)

    no_ao_coeff[:, Mol.nelectron//2+3:] = no_ao_coeff[:, Reordering]
    fock_mo = reduce(numpy.dot, (no_ao_coeff.T, fock, no_ao_coeff))
    print(numpy.diag(fock_mo))
    draw_heatmap(numpy.log(numpy.abs(fock_mo)), list(range(Mol.nao)), list(range(Mol.nao)), x_label="MO", y_label="MO", vmax=1.3, vmin=-10)
    # fock_mo = reduce(numpy.dot, (no_ao_coeff.T, fock, no_ao_coeff))