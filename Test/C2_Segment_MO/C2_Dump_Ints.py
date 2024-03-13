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
    "C": {
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
        "all": list(range(14)),
        "nao": 14,
        "basis": "ccpvdz",
        "cmoao": None,
    },
}

dirname = "/home/ningzhangcaltech/Github_Repo/pyscf_util/Test/AtmOrb"

for atom in ["C"]:
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
    'Ag': 2,
    'B1g':1,
    'B2g':1,
    'B3g':1,
    'Au': 0,
    'B1u':2,
    'B2u':1,
    'B3u':1,
}

SEGMENT = {
    "C1":list(range(0, 10)),
    "C1CAS":list(range(10, 14)),
    "C2CAS":list(range(14, 18)),
    "C2":list(range(18, 28)),
}

ATM_LOC_NELEC = (12 - 8) // 2
ATM_LOC_NOCCORB = ATM_LOC_NELEC // 2

if __name__ == '__main__':

    # bondlength = [1.68, 2.5, 2.8, 3.2]
    # bondlength = [1.68]
    # bondlength = [3.2]

    bondlength = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0]

    for BondLength in bondlength:

        Mol = pyscf.gto.Mole()
        Mol.atom = '''
C     0.0000      0.0000  %f 
C     0.0000      0.0000  -%f 
''' % (BondLength / 2, BondLength/2)
        Mol.basis = 'ccpvdz-dk'
        Mol.symmetry = "D2h"
        Mol.spin = 0
        Mol.charge = 0
        Mol.verbose = 2
        # Mol.unit = 'angstorm'
        Mol.build()
        SCF = pyscf.scf.RHF(Mol)
        SCF.max_cycle = 32
        SCF.conv_tol = 1e-9

        norb = 8
        nelec = 8
        iCISCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
        # mo_init = pyscf.mcscf.sort_mo_by_irrep(
        #     iCISCF_Driver, iCISCF_Driver.mo_coeff, cas_space_symmetry)  # right!

        bond_int = int(BondLength * 100)
        # cmoao = ReadIn_Cmoao("Cr2_%d_LO_cmoao" % (bond_int), Mol.nao)

        Mol.symmetry = "C2V"
        Mol.build()

        ## dump CAS + Cr1 

        cmoao_C1 = ReadIn_Cmoao("../C2_%d_LO_cmoao_%s" % (bond_int, "C1"), Mol.nao, len(SEGMENT["C1"]))
        cmoao_C1CAS = ReadIn_Cmoao("../C2_%d_LO_cmoao_%s" % (bond_int, "C1CAS"), Mol.nao, len(SEGMENT["C1CAS"]))
        cmoao_C2 = ReadIn_Cmoao("../C2_%d_LO_cmoao_%s" % (bond_int, "C2"), Mol.nao, len(SEGMENT["C2"]))
        cmoao_C2CAS = ReadIn_Cmoao("../C2_%d_LO_cmoao_%s" % (bond_int, "C2CAS"), Mol.nao, len(SEGMENT["C2CAS"]))
        
        
        # print(cmoao_CAS.shape)
        # print(cmoao_C1.shape)
        # print(cmoao_C2.shape)

        # act_cmoao = numpy.concatenate([cmoao_C1[:, :ATM_LOC_NOCCORB], cmoao_CAS, cmoao_C1[:, ATM_LOC_NOCCORB:]], axis=1)
        act_cmoao = numpy.concatenate([cmoao_C1CAS, cmoao_C2CAS, cmoao_C1], axis=1)
        print(act_cmoao.shape)
        core_cmoao = cmoao_C2[:, :ATM_LOC_NOCCORB]

        Integrals_Manager.dump_heff_casci(Mol, iCISCF_Driver, core_cmoao, act_cmoao, "C2_%d_%s" % (bond_int, "C1"))

        # Integrals_Manager.dump_heff_casci(Mol, iCISCF_Driver, , )

        # act_cmoao = numpy.concatenate([cmoao_C2[:, :ATM_LOC_NOCCORB], cmoao_CAS, cmoao_C2[:, ATM_LOC_NOCCORB:]], axis=1)
        act_cmoao = numpy.concatenate([cmoao_C1CAS, cmoao_C2CAS, cmoao_C2], axis=1)
        print(act_cmoao.shape)
        core_cmoao = cmoao_C1[:, :ATM_LOC_NOCCORB]

        Integrals_Manager.dump_heff_casci(Mol, iCISCF_Driver, core_cmoao, act_cmoao, "C2_%d_%s" % (bond_int, "C2"))
        
        cmoao = numpy.concatenate([cmoao_C1, cmoao_C1CAS, cmoao_C2CAS, cmoao_C2], axis=1)
        
        SCF = pyscf.scf.RHF(Mol)
        SCF.max_cycle = 32
        SCF.conv_tol = 1e-9

        SCF.mo_coeff = cmoao
        orbsym = OrbSymInfo(Mol, SCF)
        pyscf.tools.fcidump.from_mo(Mol, "C2_Hmat_%d" % bond_int,cmoao,orbsym)
