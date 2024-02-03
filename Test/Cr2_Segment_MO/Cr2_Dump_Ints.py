# coding=UTF-8

import pyscf
from pyscf import tools
from Util_File import ReadIn_Cmoao, Dump_Cmoao
from Util_Orb import Analysis_Orb_Comp, _construct_atm_bas
from pyscf import symm
import iCISCF
import Integrals_Manager
import numpy

# construct ao

atm_bas = {
    "Cr": {
        # "1s": [0],
        # "2s": [1],
        # "2p": [2, 3, 4],
        # "3s": [5],
        # "3p": [6, 7, 8],
        # "4s": [9],
        # "3d": [10, 11, 12, 13, 14],
        # "4p": [15, 16, 17],
        # "5s": [18],
        # "5p": [19, 20, 21],
        # "4d": [22, 23, 24, 25, 26],
        # "6p": [27, 28, 29],
        # "5d": [30, 31, 32, 33, 34],
        # "6s": [35],
        # "4f": [36, 37, 38, 39, 40, 41, 42],
        "all": list(range(43)),
        "nao": 43,
        "basis": "ccpvdz-dk",
        "cmoao": None,
    },
}

dirname = "/home/ningzhang/GitHub_Repo/pyscf_util/Test/AtmOrb"

for atom in ["Cr"]:
    atm_bas[atom]["cmoao"] = ReadIn_Cmoao(
        dirname+"/"+"%s_0_%s" % (atom, atm_bas[atom]["basis"]), atm_bas[atom]["nao"])


def Generate_InputFile_SiCI(inputfilename, spin, OrbSymInfo, nstates, symmetry,
                            nelec_val, rotatemo, cmin):
    inputfile = open(inputfilename, "w")
    inputfile.write("NSTATES=%d\nunpair=%d\nIRREP=%d\n" %
                    (nstates, spin, symmetry))
    inputfile.write("nsegment=%d %d %d %d %d %d\n" % (OrbSymInfo[0], OrbSymInfo[1], OrbSymInfo[2],
                                                      OrbSymInfo[3], OrbSymInfo[4], 0))
    inputfile.write(
        "nvalelec=%d\nETOL=0.0003\nCMIN=%s\nROTATEMO=%d\n" %
        (nelec_val, cmin, rotatemo))
    inputfile.write("perturbation=1 0\n")  # Do SDSPT2


def OrbSymInfo(Mol, SCF):
    IRREP_MAP = {}
    nsym = len(Mol.irrep_name)
    for i in range(nsym):
        IRREP_MAP[Mol.irrep_name[i]] = i
    # print(IRREP_MAP)

    OrbSym = pyscf.symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb,
                                       SCF.mo_coeff)
    IrrepOrb = []
    for i in range(len(OrbSym)):
        IrrepOrb.append(symm.irrep_name2id(Mol.groupname, OrbSym[i]))
    return IrrepOrb


def get_sym(IrrepMap, Occ):
    res = 0
    for i in range(len(Occ)):
        if (Occ[i] == 1):
            res ^= (IrrepMap[i] % 10)
    return res


cas_space_symmetry = {
    'A1u': 2,
    'A1g': 2,
    'E1ux': 1,
    'E1gy': 1,
    'E1gx': 1,
    'E1uy': 1,
    'E2gy': 1,
    'E2gx': 1,
    'E2uy': 1,
    'E2ux': 1
}

SEGMENT = {
    168: {
        "CAS":list(range(0, 12)),
        "Cr1":list(range(12, 49)),
        "Cr2":list(range(49, 86)),
    },
    250: {
        "CAS":list(range(0, 12)),
        "Cr1":list(range(12, 49)),
        "Cr2":list(range(49, 86)),
    },
    280: {
        "CAS":list(range(0, 12)),
        "Cr1":list(range(12, 49)),
        "Cr2":list(range(49, 86)),
    },
    320: {
        "CAS":list(range(0, 12)),
        "Cr1":list(range(12, 49)),
        "Cr2":list(range(49, 86)),
    },
}

ATM_LOC_NELEC = (48 - 12) // 2
ATM_LOC_NOCCORB = ATM_LOC_NELEC // 2

if __name__ == '__main__':

    bondlength = [1.68, 2.5, 2.8, 3.2]
    # bondlength = [1.68]
    # bondlength = [3.2]

    for BondLength in bondlength:

        Mol = pyscf.gto.Mole()
        Mol.atom = '''
Cr     0.0000      0.0000  %f 
Cr     0.0000      0.0000  -%f 
''' % (BondLength / 2, BondLength/2)
        Mol.basis = 'ccpvdz-dk'
        Mol.symmetry = True
        Mol.spin = 2
        Mol.charge = 0
        Mol.verbose = 4
        Mol.unit = 'angstorm'
        Mol.build()
        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 1e-9
        SCF.run()

        norb = 12
        nelec = 12
        iCISCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
        # mo_init = pyscf.mcscf.sort_mo_by_irrep(
        #     iCISCF_Driver, iCISCF_Driver.mo_coeff, cas_space_symmetry)  # right!

        bond_int = int(BondLength * 100)
        # cmoao = ReadIn_Cmoao("Cr2_%d_LO_cmoao" % (bond_int), Mol.nao)

        Mol.symmetry = "C2V"
        Mol.build()

        ## dump CAS + Cr1 

        cmoao_CAS = ReadIn_Cmoao("Cr2_%d_LO_cmoao_%s" % (bond_int, "CAS"), Mol.nao, len(SEGMENT[bond_int]["CAS"]))
        cmoao_Cr1 = ReadIn_Cmoao("Cr2_%d_LO_cmoao_%s" % (bond_int, "Cr1"), Mol.nao, len(SEGMENT[bond_int]["Cr1"]))
        cmoao_Cr2 = ReadIn_Cmoao("Cr2_%d_LO_cmoao_%s" % (bond_int, "Cr2"), Mol.nao, len(SEGMENT[bond_int]["Cr2"]))
        
        print(cmoao_CAS.shape)
        print(cmoao_Cr1.shape)
        print(cmoao_Cr2.shape)

        act_cmoao = numpy.concatenate([cmoao_Cr1[:, :ATM_LOC_NOCCORB], cmoao_CAS, cmoao_Cr1[:, ATM_LOC_NOCCORB:]], axis=1)
        print(act_cmoao.shape)
        core_cmoao = cmoao_Cr2[:, :ATM_LOC_NOCCORB]

        Integrals_Manager.dump_heff_casci(Mol, iCISCF_Driver, core_cmoao, act_cmoao, "Cr2_%d_%s" % (bond_int, "Cr1"))

        # Integrals_Manager.dump_heff_casci(Mol, iCISCF_Driver, , )

        act_cmoao = numpy.concatenate([cmoao_Cr2[:, :ATM_LOC_NOCCORB], cmoao_CAS, cmoao_Cr2[:, ATM_LOC_NOCCORB:]], axis=1)
        print(act_cmoao.shape)
        core_cmoao = cmoao_Cr1[:, :ATM_LOC_NOCCORB]

        Integrals_Manager.dump_heff_casci(Mol, iCISCF_Driver, core_cmoao, act_cmoao, "Cr2_%d_%s" % (bond_int, "Cr2"))
