# coding=UTF-8

import pyscf
from pyscf import tools
from Util_File import ReadIn_Cmoao, Dump_Cmoao
from Util_Orb import Analysis_Orb_Comp, _construct_atm_bas
from pyscf import symm
import iCISCF
import Integrals_Manager
import numpy
import os 

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

def _Generate_InputFile_SiCI(File,
                             Segment,
                             nelec_val,
                             cmin=0.0,
                             perturbation=1,
                             Task=None,
                             tol=1e-8,
                             selection=1
                             ):
    """Generate the input file
    """
    with open(File, "w") as f:
        f.write("nsegment=%s\n" % Segment)
        f.write("nvalelec=%d\n" % nelec_val)
        f.write("rotatemo=0\n")
        f.write("cmin=%s\n" % cmin)
        f.write("perturbation=%d 0\n" % perturbation)
        f.write("dumprdm=0\n")
        f.write("relative=0\n")
        f.write("task=%s\n" % Task)
        f.write("etol=%e\n" % tol)
        f.write("selection=%d\n" % selection)

def _Generate_InputFile_RDM(File,
                             Segment,
                             nelec_val,
                             orbsym,
                             cmin=0.0,
                             perturbation=1,
                             Task=None,
                             tol=1e-8,
                             selection=1,
                             orb_segment="0 4 8 18",
                             weightcutoff=1e-5
                             ):
    """Generate the input file
    """
    with open(File, "w") as f:
        f.write("nsegment=%s\n" % Segment)
        f.write("nvalelec=%d\n" % nelec_val)
        f.write("rotatemo=0\n")
        f.write("cmin=%s\n" % cmin)
        f.write("perturbation=%d 0\n" % perturbation)
        f.write("dumprdm=0\n")
        f.write("relative=0\n")
        f.write("task=%s\n" % Task)
        f.write("etol=%e\n" % tol)
        f.write("selection=%d\n" % selection)
        f.write("orbsegment=%s\n" % orb_segment)
        f.write("diagrdm=1\n")
        f.write("orbsym=%s\n" % orbsym)
        f.write("weightcutoff=%e\n" % weightcutoff)
        
dirname = "/home/ningzhang/GitHub_Repo/pyscf_util/Test/AtmOrb"

for atom in ["C"]:
    atm_bas[atom]["cmoao"] = ReadIn_Cmoao(
        dirname+"/"+"%s_0_%s" % (atom, atm_bas[atom]["basis"]), atm_bas[atom]["nao"])


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

iCI_ProgramName = "ICI_CPP"
executable = os.getenv(iCI_ProgramName)

rdm_executable="/home/ningzhang/iCIPT2_CXX/Tools_Program/Reduced_DM.exe"

ORBSYM ={
    180:"0 0 3 2 0 0 3 2 0 3 2 0 0 1 0 2 3 0",
    200:"0 0 3 2 0 0 3 2 0 0 2 3 0 3 1 0 2 0",
    220:"0 3 0 2 0 3 0 2 0 0 2 3 0 3 2 1 0 0",
    240:"0 3 0 2 0 3 0 2 0 0 2 3 0 3 2 1 0 0",
    260:"0 3 0 2 0 3 0 2 0 0 2 3 0 3 2 1 0 0",
    280:"0 3 0 2 0 3 0 2 0 3 0 0 2 2 1 0 0 3",
    300:"0 3 0 2 0 3 0 2 0 3 2 0 0 2 1 0 0 3",
    350:"0 3 0 2 0 3 0 2 0 3 2 0 0 0 0 1 2 3",
    400:"0 3 0 2 0 3 0 2 0 3 2 0 0 0 1 2 0 3",
}

CMIN=1e-5
WEIGHTCUTOFF = 1e-4

import shutil

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

        #### first segment

        FCIDUMP_FILE1 = "C2_%d_%s" % (bond_int, "C1")
        PREFIX_C1 = "C2_%d_C1" % bond_int
        
        ##### generate input file for iCI
        
        _Generate_InputFile_SiCI("input_C1", "0 0 4 5 9 0", 10, cmin=CMIN, perturbation=1, Task="0 0 1 1", tol=1e-8, selection=1)
        
        os.system("%s input_C1 %s 1>%s.out 2>%s.err" % (executable, FCIDUMP_FILE1, PREFIX_C1, PREFIX_C1))
        
        ##### generate input file for RDM
        
        _Generate_InputFile_RDM("input_C1_RDM", "0 0 4 5 9 0", 10, ORBSYM[bond_int], cmin=CMIN, perturbation=1, Task="0 0 1 1", tol=1e-8, selection=1, orb_segment="0 4 8 18", weightcutoff=WEIGHTCUTOFF)
        
        os.system("%s input_C1_RDM 1>%s.rdm.out 2>%s.rdm.err" % (rdm_executable, PREFIX_C1, PREFIX_C1))
        
        ##### 找到所有以 Block 开头 以 .PrimeSpace 结尾的文件，将其添加尾缀并移动到指定目录 
        
        new_dir = './bond_%d' % bond_int
        os.makedirs(new_dir, exist_ok=True)
        
        for filename in os.listdir('.'):
            if (filename.startswith('Block') and filename.endswith('.PrimeSpace')) or filename.endswith(".dat") or filename.endswith(".out"):
                new_filename = filename + '.C1'
                src_file = os.path.join('.', filename)
                dst_file = os.path.join(new_dir, new_filename)
                shutil.move(src_file, dst_file)
        
        #### 删除所有 开头 以 .PrimeSpace 结尾的文件
        
        for filename in os.listdir('.'):
            if filename.endswith('.PrimeSpace'):
                os.remove(filename)
                
        os.remove("input_C1")
        os.remove("input_C1_RDM")
        os.remove("input_C1.enedat")
                
        #### second segment
        
        FCIDUMP_FILE2 = "C2_%d_%s" % (bond_int, "C2")
        PREFIX_C2 = "C2_%d_C2" % bond_int
        
        ##### generate input file for iCI
        
        _Generate_InputFile_SiCI("input_C2", "0 0 4 5 9 0", 10, cmin=CMIN, perturbation=1, Task="0 0 1 1", tol=1e-8, selection=1)
        
        os.system("%s input_C2 %s 1>%s.out 2>%s.err" % (executable, FCIDUMP_FILE2, PREFIX_C2, PREFIX_C2))
        
        ##### generate input file for RDM
        
        _Generate_InputFile_RDM("input_C2_RDM", "0 0 4 5 9 0", 10, ORBSYM[bond_int], cmin=CMIN, perturbation=1, Task="0 0 1 1", tol=1e-8, selection=1, orb_segment="0 4 8 18", weightcutoff=WEIGHTCUTOFF)
        
        os.system("%s input_C2_RDM 1>%s.rdm.out 2>%s.rdm.err" % (rdm_executable, PREFIX_C2, PREFIX_C2))
        
        ##### 找到所有以 Block 开头 以 .PrimeSpace 结尾的文件，将其添加尾缀并移动到指定目录
        
        for filename in os.listdir('.'):
            if (filename.startswith('Block') and filename.endswith('.PrimeSpace')) or filename.endswith(".dat") or filename.endswith(".out"):
                new_filename = filename + '.C2'
                src_file = os.path.join('.', filename)
                dst_file = os.path.join(new_dir, new_filename)
                shutil.move(src_file, dst_file)
        
        #### 删除所有 开头 以 .PrimeSpace 结尾的文件
        
        # exit(1)
        
        for filename in os.listdir('.'):
            if filename.endswith('.PrimeSpace'):
                os.remove(filename)
            
        os.remove("input_C2")
        os.remove("input_C2_RDM")
        os.remove("input_C2.enedat")