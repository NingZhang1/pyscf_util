# coding=UTF-8

import pyscf
from pyscf import tools
from Util_File import ReadIn_Cmoao, Dump_Cmoao
from Util_Orb import Analysis_Orb_Comp, _construct_atm_bas
from pyscf import symm
import iCISCF

# construct ao

atm_bas = {
    "Cr": {
        "1s": [0],
        "2s": [1],
        "2p": [2, 3, 4],
        "3s": [5],
        "3p": [6, 7, 8],
        "4s": [9],
        "3d": [10, 11, 12, 13, 14],
        "4p": [15, 16, 17],
        "5s": [18],
        "5p": [19, 20, 21],
        "4d": [22, 23, 24, 25, 26],
        "6p": [27, 28, 29],
        "5d": [30, 31, 32, 33, 34],
        "6s": [35],
        "4f": [36, 37, 38, 39, 40, 41, 42],
        "nao": 43,
        "basis": "ccpvdz-dk",
        "cmoao": None,
    },
}

dirname = "/home/nzhangcaltech/GitHub_Repo/pyscf_util/Test/AtmOrb"

for atom in ["Cr"]:
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
        Mol.verbose = 2
        Mol.unit = 'angstorm'
        Mol.build()
        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 1e-9
        SCF.run()
        print(SCF.mo_energy)
        print(SCF.mo_occ)
        print(OrbSymInfo(Mol, SCF))
        print(Mol.groupname)
        print(get_sym(OrbSymInfo(Mol, SCF), SCF.mo_occ))

        DumpFileName = "FCIDUMP" + "_" + "CR2" + "_" + \
            'ccpvdz-dk' + "_" + str(int(BondLength*100)) + "_SCF"

        # tools.fcidump.from_scf(SCF, DumpFileName, 1e-10)

        # tools.fcidump.from_mo(Mol, DumpFileName, iCISCF_Driver.mo_coeff, 1e-10)

        bond_int = int(BondLength*100)
        pyscf.tools.molden.from_mo(
            Mol, "Cr2_%d_SCF.molden" % (bond_int), SCF.mo_coeff)
        Dump_Cmoao("Cr2_%d_SCF_cmoao" % (bond_int), SCF.mo_coeff)

        Mol.spin = 0
        Mol.build()

        # iCISCF

        # the driver

        norb = 12
        nelec = 12
        iCISCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
        mo_init = pyscf.mcscf.sort_mo_by_irrep(
            iCISCF_Driver, iCISCF_Driver.mo_coeff, cas_space_symmetry)  # right!

        cmin_now = 0.0
        iCISCF_Driver.fcisolver = iCISCF.iCI(mol=Mol, cmin=cmin_now, state=[
                                             [0, 0, 1, [1]]],  mo_coeff=mo_init)
        iCISCF_Driver.internal_rotation = True
        iCISCF_Driver.conv_tol = 1e-8
        iCISCF_Driver.max_cycle_macro = 128
        iCISCF_Driver.canonicalization = True
        iCISCF_Driver.kernel(mo_coeff=mo_init)

        core_cmoao = iCISCF_Driver.mo_coeff[:, :Mol.nelectron//2-6]
        act_cmoao = iCISCF_Driver.mo_coeff[:, Mol.nelectron//2-6:Mol.nelectron//2+6]

        import Integrals_Manager

        Integrals_Manager.dump_heff_casci(Mol, iCISCF_Driver, core_cmoao, act_cmoao, "Cr2_%d_%s" % (bond_int, "CASSCF"))

        # exit(1)

        Res = Analysis_Orb_Comp(Mol, iCISCF_Driver.mo_coeff, Mol.nelectron//2-6, Mol.nelectron//2+6,
                                atm_bas, tol=0.1, with_distinct_atm=True)

        # continue

        # pyscf_util.dump_cmoao(DumpFileName, iCISCF_Driver.mo_coeff)

        mo_ene = iCISCF_Driver.mo_energy  # diagonal generalized fock
        mo_coeff = iCISCF_Driver.mo_coeff
        ovlp = SCF.get_ovlp()

        # dump molden

        bond_int = int(BondLength*100)
        pyscf.tools.molden.from_mo(
            Mol, "Cr2_%d_MCSCF.molden" % (bond_int), mo_coeff)
        Dump_Cmoao("Cr2_%d_MCSCF_cmoao" % (bond_int), mo_coeff)

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

        lo_coeff = split_loc_given_range(
            Mol, lo_coeff, 0, Mol.nelectron//2-6)  # 局域化的有问题 !
        lo_coeff = split_loc_given_range(
            Mol, lo_coeff, Mol.nelectron//2+6, Mol.nao)

        lo_mocoeff = reduce(numpy.dot, (mo_coeff.T, ovlp, lo_coeff))

        print(lo_mocoeff[:, 0])

        fock_lo = reduce(
            numpy.dot, (lo_mocoeff.T, fock_canonicalized_mo, lo_mocoeff))

        fock_lo = numpy.diag(numpy.diag(fock_lo))

        print("e = ", numpy.diag(fock_lo))

        # the core part 
        
        ncore = Mol.nelectron//2-6
        e, h = numpy.linalg.eigh(fock_lo[:ncore, :ncore])
        print("e = ", e)
        lo_mocoeff[:, :ncore] = numpy.dot(lo_mocoeff[:, :ncore], h)  # should be a reordering

        vir_begin = Mol.nelectron//2+6
        e, h = numpy.linalg.eigh(fock_lo[vir_begin:, vir_begin:])
        print("e = ", e)
        lo_mocoeff[:, vir_begin:] = numpy.dot(lo_mocoeff[:, vir_begin:], h)

        lo_coeff = numpy.dot(mo_coeff, lo_mocoeff)

        ############# for the core first try to localize the core orbitals #############

        Res = Analysis_Orb_Comp(Mol, lo_coeff, 0, Mol.nelectron//2-6,
                                atm_bas, tol=0.1, with_distinct_atm=True)

        List = {
            "Cr_1": [],
            "Cr_2": [],
            "bonding": [],
        }

        Res = Analysis_Orb_Comp(Mol, lo_coeff, 0, Mol.nelectron//2-6,
                                atm_bas, tol=0.1, with_distinct_atm=True)

        for data in Res:
            print(data)
            if isinstance(data['key'], list):
                # judge whether the same orbitals
                same = True
                for i in range(len(data['key'])):
                    if data['key'][i][:4] != data['key'][0][:4]:
                        same = False
                        break
                if same:
                    List[data['key'][0][:4]].append(data['orbindx'])
                else:
                    List["bonding"].append(data['orbindx'])
            else:
                List[data['key'][:4]].append(data['orbindx'])

        List["bonding"].extend(
            list(range(Mol.nelectron//2-6, Mol.nelectron//2+6)))

        Res = Analysis_Orb_Comp(Mol, lo_coeff, Mol.nelectron//2+6, Mol.nao,
                                atm_bas, tol=0.1, with_distinct_atm=True)

        for data in Res:
            print(data)
            if isinstance(data['key'], list):
                # judge whether the same orbitals
                same = True
                for i in range(len(data['key'])):
                    if data['key'][i][:4] != data['key'][0][:4]:
                        same = False
                        break
                if same:
                    List[data['key'][i][:4]].append(data['orbindx'])
                else:
                    List["bonding"].append(data['orbindx'])
            else:
                List[data['key'][:4]].append(data['orbindx'])

        Reordering = []
        Reordering.extend(List["bonding"])
        Reordering.extend(List["Cr_1"])
        Reordering.extend(List["Cr_2"])

        print(Reordering)

        lo_coeff[:, :] = lo_coeff[:, Reordering]

        pyscf.tools.molden.from_mo(
            Mol, "Cr2_%d_LO.molden" % (bond_int), lo_coeff)
        Dump_Cmoao("Cr2_%d_LO_cmoao" % (bond_int), lo_coeff)

        