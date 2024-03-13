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

def _Generate_InputFile_RDM(File,
                             Segment,
                             nelec_val,
                             orbsym,
                             subspace1,
                             subspace2,
                             cmin=0.0,
                             perturbation=1,
                             Task=None,
                             tol=1e-8,
                             selection=1,
                             weightcutoff=1e-5,
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
        f.write("subspace1=%s\n" % subspace1)
        f.write("subspace2=%s\n" % subspace2)
        f.write("orbsym=%s\n" % orbsym)
        f.write("weightcutoff=%e\n" % weightcutoff)
        
dirname = "/home/ningzhangcaltech/Github_Repo/pyscf_util/Test/AtmOrb"

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

rdm_executable="/home/ningzhangcaltech/Github_Repo/iCIPT2_CXX/Tools_Program/MixRDM.exe"

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

        work_dir = './bond_%d' % bond_int
        new_dir  = './bond_%d_LS' % bond_int
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        #### 分析文件名
        
        block_0_subspace1_str = ""
        block_0_subspace2_str = ""
        block_1_subspace1_str = ""
        block_1_subspace2_str = ""
        
        for filename in os.listdir(work_dir):
            split1 = filename.split('.')
            assert split1[-1] in ["C1", "C2"]
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
            
            if int(block_id) == 2:
                if split1[-1] == "C1":
                    new_filename = "Block_%d_Nelec_%d_SzTwo_%d_Irrep_%d.PrimeSpace" % (0, int(nelec), int(sztwo), int(irrep))
                else:
                    new_filename = "Block_%d_Nelec_%d_SzTwo_%d_Irrep_%d.PrimeSpace" % (3, int(nelec), int(sztwo), int(irrep))
                src_file = os.path.join(work_dir, filename)
                dst_file = os.path.join(new_dir, new_filename)
                ## copy file
                shutil.copy(src_file, dst_file)
            else:
                
                ## copy to the current dir 
                
                if split1[-1] == "C1":
                    if int(block_id) == 0:
                        block_0_subspace1_str += "%d %d %d %d " % (int(nelec), int(sztwo), int(irrep), nstate)
                    else:
                        block_1_subspace1_str += "%d %d %d %d " % (int(nelec), int(sztwo), int(irrep), nstate)
                else:
                    if int(block_id) == 0:
                        block_0_subspace2_str += "%d %d %d %d " % (int(nelec), int(sztwo), int(irrep), nstate)
                    else:
                        block_1_subspace2_str += "%d %d %d %d " % (int(nelec), int(sztwo), int(irrep), nstate)
                
                src_file = os.path.join(work_dir, filename)
                dst_file = os.path.join('./', filename) 
                
                shutil.copy(src_file, dst_file)
        
        print("block_0_subspace1_str", block_0_subspace1_str)
        print("block_0_subspace2_str", block_0_subspace2_str)
        print("block_1_subspace1_str", block_1_subspace1_str)   
        print("block_1_subspace2_str", block_1_subspace2_str)
        
        # exit(1)
        
        # do mix RDM for block0
        
        ##### block 0 
        
        filename = "block0_rdm_eigenvalue.dat.C1"
        filename_new = "block0_rdm_eigenvalue.dat"
        
        src_file = os.path.join(work_dir, filename)
        dst_file = os.path.join('./', filename_new)
        
        shutil.copy(src_file, dst_file)
        
        filename = "block0_rdm_eigenvalue.dat.C2"
        filename_new = "block1_rdm_eigenvalue.dat"
        
        src_file = os.path.join(work_dir, filename)
        dst_file = os.path.join('./', filename_new)
        
        shutil.copy(src_file, dst_file)
        
        _Generate_InputFile_RDM("input_C2_RDM", "0 0 2 2 0 0", 4, ORBSYM[bond_int], cmin=CMIN, perturbation=1, 
                                subspace1=block_0_subspace1_str,
                                subspace2=block_0_subspace2_str,
                                Task="0 0 1 1", tol=1e-8, selection=1, weightcutoff=WEIGHTCUTOFF)
        
        for filename in os.listdir("./"):
            if (filename.endswith('.C1') or filename.endswith('C2')) and filename.startswith('Block'):
                split1 = filename.split('.')
                split2 = split1[0].split('_')
                # print(split2)
                block_id = split2[1]
                nelec    = split2[3]
                sztwo    = split2[5]
                irrep    = split2[7]
                nstate   = 0
                
                if block_id != "0":
                    continue
                
                if filename.endswith('.C1'):
                    filename_new = "Block_%d_Nelec_%d_SzTwo_%d_Irrep_%d.PrimeSpace" % (0, int(nelec), int(sztwo), int(irrep))
                else:
                    filename_new = "Block_%d_Nelec_%d_SzTwo_%d_Irrep_%d.PrimeSpace" % (1, int(nelec), int(sztwo), int(irrep))
        
                src_file = os.path.join('./', filename)
                dst_file = os.path.join('./', filename_new)
                # print("src_file", src_file)
                # print("dst_file", dst_file)
                ## copy file
                shutil.copy(src_file, dst_file)
        
        
        PREFIX_C1 = "C2_%d_C1" % bond_int
        os.system("%s input_C2_RDM 1>%s.mixrdm.out 2>%s.mixrdm.err" % (rdm_executable, PREFIX_C1, PREFIX_C1))
        
        # exit(1)
        
        for filename in os.listdir("./"):
            if filename.endswith('.PrimeSpace') and filename.startswith('Block'):
                split1 = filename.split('.')
                split2 = split1[0].split('_')
                # print(split2)
                block_id = split2[1]
                nelec    = split2[3]
                sztwo    = split2[5]
                irrep    = split2[7]
                
                if block_id != "2":
                    continue
                
                filename_new = "Block_%d_Nelec_%d_SzTwo_%d_Irrep_%d.PrimeSpace" % (1, int(nelec), int(sztwo), int(irrep))

                src_file = os.path.join('./', filename)
                dst_file = os.path.join(new_dir, filename_new)
                
                ## copy file
                shutil.copy(src_file, dst_file)

        # exit(1)

        ##### block 1 
        
        filename = "block1_rdm_eigenvalue.dat.C1"
        filename_new = "block0_rdm_eigenvalue.dat"
        
        src_file = os.path.join(work_dir, filename)
        dst_file = os.path.join('./', filename_new)
        
        shutil.copy(src_file, dst_file)
        
        filename = "block1_rdm_eigenvalue.dat.C2"
        filename_new = "block1_rdm_eigenvalue.dat"
        
        src_file = os.path.join(work_dir, filename)
        dst_file = os.path.join('./', filename_new)
        
        shutil.copy(src_file, dst_file)
        
        _Generate_InputFile_RDM("input_C2_RDM", "0 0 2 2 0 0", 4, ORBSYM[bond_int], cmin=CMIN, perturbation=1,
                                subspace1=block_1_subspace1_str,
                                subspace2=block_1_subspace2_str,
                                Task="0 0 1 1", tol=1e-8, selection=1, weightcutoff=WEIGHTCUTOFF)
        
        for filename in os.listdir("./"):
            if (filename.endswith('.C1') or filename.endswith('C2')) and filename.startswith('Block'):
                split1 = filename.split('.')
                split2 = split1[0].split('_')
                # print(split2)
                block_id = split2[1]
                nelec    = split2[3]
                sztwo    = split2[5]
                irrep    = split2[7]
                nstate   = 0
                
                if block_id != "1":
                    continue
                
                if filename.endswith('.C1'):
                    filename_new = "Block_%d_Nelec_%d_SzTwo_%d_Irrep_%d.PrimeSpace" % (0, int(nelec), int(sztwo), int(irrep))
                else:
                    filename_new = "Block_%d_Nelec_%d_SzTwo_%d_Irrep_%d.PrimeSpace" % (1, int(nelec), int(sztwo), int(irrep))
        
                src_file = os.path.join('./', filename)
                dst_file = os.path.join('./', filename_new)
                # print("src_file", src_file)
                # print("dst_file", dst_file)
                ## copy file
                shutil.copy(src_file, dst_file)
        
        PREFIX_C2 = "C2_%d_C2" % bond_int
        os.system("%s input_C2_RDM 1>%s.mixrdm.out 2>%s.mixrdm.err" % (rdm_executable, PREFIX_C2, PREFIX_C2))
        
        # exit(1)
        
        for filename in os.listdir("./"):
            if filename.endswith('.PrimeSpace') and filename.startswith('Block'):
                split1 = filename.split('.')
                split2 = split1[0].split('_')
                # print(split2)
                block_id = split2[1]
                nelec    = split2[3]
                sztwo    = split2[5]
                irrep    = split2[7]
                
                if block_id != "2":
                    continue
                
                filename_new = "Block_%d_Nelec_%d_SzTwo_%d_Irrep_%d.PrimeSpace" % (2, int(nelec), int(sztwo), int(irrep))

                src_file = os.path.join('./', filename)
                dst_file = os.path.join(new_dir, filename_new)
                
                ## copy file
                shutil.copy(src_file, dst_file)
        
        ### 删除当前目录下所有 以 C1, C2 结尾的文件 
        
        for filename in os.listdir('.'):
            if filename.endswith('.C1') or filename.endswith('.C2') or filename.endswith('.PrimeSpace') or filename.startswith('.dat'):
                os.remove(filename)
                
        
        