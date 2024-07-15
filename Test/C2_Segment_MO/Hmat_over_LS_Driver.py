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

def _Generate_InputFile_Hmat(File,
                             Segment,
                             nelec_val,
                             nblock,
                             blocknorb,
                             subspace,
                             nelec,
                             cmin=0.0,
                             perturbation=1,
                             Task=None,
                             tol=1e-8,
                             selection=1,
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
        f.write("blocknorb=%s\n" % blocknorb)
        f.write("subspace=%s\n" % subspace)
        f.write("nelec=%s\n" % nelec)
        f.write("nblock=%s\n" % nblock)
        
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

rdm_executable="/home/ningzhang/iCIPT2_CXX/Tools_Program/Hmat_over_LocalStates.exe"

ORBSYM ={
    180:"0 0 3 2",
    200:"0 0 3 2",
    220:"0 3 0 2",
    240:"0 3 0 2",
    260:"0 3 0 2",
    280:"0 3 0 2",
    300:"0 3 0 2",
    350:"0 3 0 2",
    400:"0 3 0 2",
}

CMIN=1e-5
WEIGHTCUTOFF = 1e-5

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

        work_dir  = './bond_%d_LS' % bond_int

        #### copy FCIDUMP 
        
        src_file = os.path.join("./", "C2_Hmat_%d" % bond_int)
        dst_file = os.path.join("./", "FCIDUMP")
        shutil.copy(src_file, dst_file)

        #### 分析文件名
        
        subspace_str = ""
        
        for filename in os.listdir(work_dir):
            split1 = filename.split('.')
            # assert split1[-1] in ["C1", "C2"]
            # print(split1[0])
            # print(split1[-1])
            if "Block" not in split1[0]:
                continue
            split2 = split1[0].split('_')
            # print(split2)
            block_id = split2[1]
            nelec    = split2[3]
            sztwo    = split2[5]
            irrep    = split2[7]
            nstate   = 0
       
            with open(work_dir+"/"+filename, "r") as f:
                lines = f.readlines()
                line = lines[1].split(" ")
                # print(line)
                
                nfloat = 0
                for data in line:
                    try:
                        float(data)
                        nfloat += 1
                    except:
                        pass
                # print(nfloat)
                
                nstate = nfloat - 2
            
            print("Block %s, nelec %s, sztwo %s, irrep %s, nstate %d" % (block_id, nelec, sztwo, irrep, nstate))
            
            subspace_str += "%d %d %d %d %d " % (int(block_id), int(nelec), int(sztwo), int(irrep), nstate)
            
            src_file = os.path.join(work_dir, filename)
            dst_file = os.path.join('./', filename)
            
            shutil.copy(src_file, dst_file)
        
        print(subspace_str)
        
        _Generate_InputFile_Hmat(
            "input_C2_Hmat", 
            Segment="0 2 4 4 18 0", 
            nelec_val=8,
            nblock=4,
            blocknorb="10 4 4 10",
            subspace=subspace_str,
            cmin=CMIN,
            Task="0 0 1 1",
            nelec=12
        )
        
        PREFIX_C2 = "C2_%d" % bond_int
        os.system("%s input_C2_Hmat 1>%s.Hmat.out 2>%s.Hmat.err" % (rdm_executable, PREFIX_C2, PREFIX_C2)) 
        
        FCIDUMP_LS = "FCIDUMP_LS"
        FCIDUMP_Hmat = "FCIDUMP_Hmat_%d" % bond_int
        
        shutil.copy(FCIDUMP_LS, FCIDUMP_Hmat)
        
        for filename in os.listdir('.'):
            if filename.endswith('.C1') or filename.endswith('.C2') or filename.endswith('.PrimeSpace') or filename.startswith('.dat'):
                os.remove(filename)
        
                
        
        