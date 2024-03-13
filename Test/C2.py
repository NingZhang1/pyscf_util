# coding=UTF-8

import pyscf
from pyscf import tools
from Util_File import ReadIn_Cmoao, Dump_Cmoao
from Util_Orb import Analysis_Orb_Comp, _construct_atm_bas
from pyscf import symm
import iCISCF

# construct ao

atm_bas = {
    "C": {
        "1s": [0],
        "2s": [1],
        "2p": [2, 3, 4],
        "3p": [5, 6, 7],
        "3s": [8],
        # "4s": [9],
        "3d": [9, 10, 11, 12, 13],
        # "4p": [15, 16, 17],
        # "5s": [18],
        # "5p": [19, 20, 21],
        # "4d": [22, 23, 24, 25, 26],
        # "6p": [27, 28, 29],
        # "5d": [30, 31, 32, 33, 34],
        # "6s": [35],
        # "4f": [36, 37, 38, 39, 40, 41, 42],
        "nao": 14,
        "basis": "ccpvdz",
        "cmoao": None,
    },
}

dirname = "/home/ningzhangcaltech/Github_Repo/pyscf_util/Test/AtmOrb"

for atom in ["C"]:
    atm_bas[atom]["cmoao"] = ReadIn_Cmoao(
        dirname+"/"+"%s_0_%s" % (atom, atm_bas[atom]["basis"]), atm_bas[atom]["nao"])

# print(atm_bas["H"]["cmoao"])
# print(atm_bas["C"]["cmoao"])
# print(atm_bas["C"]["2pz"])


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

# Cmin = "1e-4 9e-5 7e-5 5e-5 4e-5 3e-5 2e-5 1.5e-5"

# Cmin = [1e-4,9e-5,7e-5,5e-5,4e-5,3e-5,2e-5,1.5e-5,1e-5]


# cas_space_symmetry = {
#     'A1u': 2,        5
#     'A1g': 2,        0
#     'E1ux': 1,       7
#     'E1gy': 1,       3
#     'E1gx': 1,       2
#     'E1uy': 1,       6
#     'E2gy': 1,       1
#     'E2gx': 1,       0
#     'E2uy': 1,       4 
#     'E2ux': 1        5
# }

cas_space_symmetry = {
    'Ag': 2,
    'B1g':0,
    'B2g':1,
    'B3g':1,
    'Au': 0,
    'B1u':2,
    'B2u':1,
    'B3u':1,
}


if __name__ == '__main__':

    # bondlength = [1.68, 2.5, 2.8, 3.2]
    # bondlength = [1.68]
    # bondlength = [3.2]
    
    bondlength = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0]

    for BondLength in bondlength[-2:]:

        Mol = pyscf.gto.Mole()
        Mol.atom = '''
C     0.0000      0.0000  %f 
C     0.0000      0.0000  -%f 
''' % (BondLength / 2, BondLength/2)
        Mol.basis = 'ccpvdz-dk'
        Mol.symmetry = "D2h"
        Mol.spin = 0
        Mol.charge = 0
        Mol.verbose = 4
        # Mol.unit = 'angstorm'
        Mol.build()
        # SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF = pyscf.scf.RHF(Mol)
        SCF.max_cycle = 32
        SCF.conv_tol = 1e-9
        SCF.run()
        print(SCF.mo_energy)
        print(SCF.mo_occ)
        print(OrbSymInfo(Mol, SCF))
        print(Mol.groupname)
        print(get_sym(OrbSymInfo(Mol, SCF), SCF.mo_occ))

        DumpFileName = "FCIDUMP" + "_" + "C2" + "_" + \
            'ccpvdz' + "_" + str(int(BondLength*100)) + "_SCF"

        tools.fcidump.from_scf(SCF, DumpFileName, 1e-10)

        # tools.fcidump.from_mo(Mol, DumpFileName, iCISCF_Driver.mo_coeff, 1e-10)

        bond_int = int(BondLength*100)
        pyscf.tools.molden.from_mo(
            Mol, "C2_%d_SCF.molden" % (bond_int), SCF.mo_coeff)
        Dump_Cmoao("C2_%d_SCF_cmoao" % (bond_int), SCF.mo_coeff)

        Mol.spin = 0
        Mol.build()

        # iCISCF

        # the driver

        # norb = 8
        # nelec = 8
        # iCISCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
        # mo_init = pyscf.mcscf.sort_mo_by_irrep(
        #     iCISCF_Driver, iCISCF_Driver.mo_coeff, cas_space_symmetry)  # right!
        # cmin_now = 0.0
        # # iCISCF_Driver.fcisolver = iCISCF.iCI(mol=Mol, cmin=cmin_now, state=[
        # #                                      [0, 0, 1, [1]]],  mo_coeff=mo_init)
        # iCISCF_Driver.internal_rotation = True
        # iCISCF_Driver.conv_tol = 1e-8
        # iCISCF_Driver.max_cycle_macro = 128
        # iCISCF_Driver.canonicalization = True
        # iCISCF_Driver.kernel(mo_coeff=mo_init)
        
        iCISCF_Driver = SCF

        core_cmoao = iCISCF_Driver.mo_coeff[:, :Mol.nelectron//2-4]
        act_cmoao = iCISCF_Driver.mo_coeff[:, Mol.nelectron//2-4:Mol.nelectron//2+4]

        import Integrals_Manager

        Integrals_Manager.dump_heff_casci(Mol, iCISCF_Driver, core_cmoao, act_cmoao, "C2_%d_%s" % (bond_int, "CASSCF"))

        # exit(1)

        Res = Analysis_Orb_Comp(Mol, iCISCF_Driver.mo_coeff, Mol.nelectron//2-4, Mol.nelectron//2+4,
                                atm_bas, tol=0.1, with_distinct_atm=True)

        # continue

        # pyscf_util.dump_cmoao(DumpFileName, iCISCF_Driver.mo_coeff)

        mo_ene = iCISCF_Driver.mo_energy  # diagonal generalized fock
        mo_coeff = iCISCF_Driver.mo_coeff
        ovlp = SCF.get_ovlp()

        # dump molden

        bond_int = int(BondLength*100)
        pyscf.tools.molden.from_mo(
            Mol, "C2_%d_MCSCF.molden" % (bond_int), mo_coeff)
        Dump_Cmoao("C2_%d_MCSCF_cmoao" % (bond_int), mo_coeff)

        Mol.symmetry = "C2V"
        Mol.build()

        # localization

        import numpy
        import copy
        from Util_Orb import split_loc_given_range_NoSymm, split_loc_given_range
        from functools import reduce

        fock_canonicalized_mo = numpy.diag(mo_ene)

        lo_coeff = copy.deepcopy(mo_coeff)

        # generate random matrix

        # nbas = Mol.nelectron//2-6
        # random_uni = numpy.random.uniform(-1, 1, (nbas, nbas))
        # uni, _ = numpy.linalg.qr(random_uni)
        # lo_coeff[:, :Mol.nelectron//2-6] = numpy.dot(lo_coeff[:, :Mol.nelectron//2-6], uni)
        # nbas = Mol.nao - (Mol.nelectron//2+6)
        # random_uni = numpy.random.uniform(-1, 1, (nbas, nbas))
        # uni, _ = numpy.linalg.qr(random_uni)
        # lo_coeff[:, Mol.nelectron//2+6:] = numpy.dot(lo_coeff[:, Mol.nelectron//2+6:], uni)

        small_rot = numpy.sqrt(1.0/2.0) * numpy.array([[1, -1], [1, 1]])

        lo_coeff[:, :2] = numpy.dot(lo_coeff[:, :2], small_rot)

        lo_coeff = split_loc_given_range(
            Mol, lo_coeff, 0, Mol.nelectron//2-4)  # 局域化的有问题 !
        lo_coeff = split_loc_given_range(
            Mol, lo_coeff, Mol.nelectron//2-4, Mol.nelectron//2+4, random=True)
        lo_coeff = split_loc_given_range(
            Mol, lo_coeff, Mol.nelectron//2+4, Mol.nao, random=True)

        lo_mocoeff = reduce(numpy.dot, (mo_coeff.T, ovlp, lo_coeff))

        print(lo_mocoeff[:, 0])

        fock_lo = reduce(
            numpy.dot, (lo_mocoeff.T, fock_canonicalized_mo, lo_mocoeff))

        fock_lo = numpy.diag(numpy.diag(fock_lo))

        print("e = ", numpy.diag(fock_lo))

        # the core part 
        
        ncore = Mol.nelectron//2-4
        e, h = numpy.linalg.eigh(fock_lo[:ncore, :ncore])
        print("e = ", e)
        lo_mocoeff[:, :ncore] = numpy.dot(lo_mocoeff[:, :ncore], h)  # should be a reordering

        ## active space 

        e, h = numpy.linalg.eigh(fock_lo[ncore:ncore+8, :][:, ncore:ncore+8])
        print("e = ", e)
        lo_mocoeff[:, ncore:ncore+8] = numpy.dot(lo_mocoeff[:, ncore:ncore+8], h)

        vir_begin = Mol.nelectron//2+4
        e, h = numpy.linalg.eigh(fock_lo[vir_begin:, vir_begin:])
        print("e = ", e)
        lo_mocoeff[:, vir_begin:] = numpy.dot(lo_mocoeff[:, vir_begin:], h)

        lo_coeff = numpy.dot(mo_coeff, lo_mocoeff)

        ovlp = SCF.get_ovlp()
        
        lo_ovlp = reduce(numpy.dot, (lo_coeff.T, ovlp, lo_coeff))
        assert numpy.allclose(lo_ovlp, numpy.eye(lo_ovlp.shape[0]))

        ############# for the core first try to localize the core orbitals #############

        Res = Analysis_Orb_Comp(Mol, lo_coeff, 0, Mol.nelectron//2-4,
                                atm_bas, tol=0.1, with_distinct_atm=True)

        List = {
            "C_1": [],
            "C_2": [],
            "bonding": [],
            "C_1_act": [],
            "C_2_act": [],
        }

        Res = Analysis_Orb_Comp(Mol, lo_coeff, 0, Mol.nelectron//2-4,
                                atm_bas, tol=0.1, with_distinct_atm=True)

        for data in Res:
            print(data)
            if isinstance(data['key'], list):
                # judge whether the same orbitals
                same = True
                for i in range(len(data['key'])):
                    if data['key'][i][:3] != data['key'][0][:3]:
                        same = False
                        break
                if same:
                    List[data['key'][0][:3]].append(data['orbindx'])
                else:
                    List["bonding"].append(data['orbindx'])
            else:
                List[data['key'][:3]].append(data['orbindx'])

        # List["bonding"].extend(
        #     list(range(Mol.nelectron//2-4, Mol.nelectron//2+4)))
        
        Res = Analysis_Orb_Comp(Mol, lo_coeff, Mol.nelectron//2-4, Mol.nelectron//2+4,
                                atm_bas, tol=0.1, with_distinct_atm=True)
        
        for data in Res:
            print(data)
            if isinstance(data['key'], list):
                # judge whether the same orbitals
                same = True
                for i in range(len(data['key'])):
                    if data['key'][i][:3] != data['key'][0][:3]:
                        same = False
                        break
                if same:
                    List[data['key'][0][:3]+"_act"].append(data['orbindx'])
                else:
                    List["bonding"].append(data['orbindx'])
            else:
                List[data['key'][:3]+"_act"].append(data['orbindx'])
        

        Res = Analysis_Orb_Comp(Mol, lo_coeff, Mol.nelectron//2+4, Mol.nao,
                                atm_bas, tol=0.1, with_distinct_atm=True)

        for data in Res:
            print(data)
            if isinstance(data['key'], list):
                # judge whether the same orbitals
                same = True
                for i in range(len(data['key'])):
                    if data['key'][i][:3] != data['key'][0][:3]:
                        same = False
                        break
                if same:
                    List[data['key'][i][:3]].append(data['orbindx'])
                else:
                    List["bonding"].append(data['orbindx'])
            else:
                List[data['key'][:3]].append(data['orbindx'])

        List["C_1"].sort()
        List["C_2"].sort()
        List["bonding"].sort()
        List["C_1_act"].sort()
        List["C_2_act"].sort()
        print(List["bonding"])
        assert len(List["bonding"]) == 0

        Reordering = []
        # Reordering.extend(List["bonding"])
        Reordering.extend(List["C_1"])
        Reordering.extend(List["C_1_act"])
        Reordering.extend(List["C_2_act"])
        Reordering.extend(List["C_2"])

        print(Reordering)

        lo_coeff[:, :] = lo_coeff[:, Reordering]

        pyscf.tools.molden.from_mo(
            Mol, "C2_%d_LO.molden" % (bond_int), lo_coeff)
        Dump_Cmoao("C2_%d_LO_cmoao" % (bond_int), lo_coeff)

        