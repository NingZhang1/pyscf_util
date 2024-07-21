from pyscf import gto, scf, lib
import numpy
from functools import reduce
from pyscf import lib
from pyscf import tools
import pyscf

import re

from pyscf import __config__

from Util_File import *
from Util_Rela4C import *

IOBLK_SIZE = getattr(__config__, 'ao2mo_outcore_ioblk_size', 256)  # 256 MB
IOBUF_WORDS = getattr(__config__, 'ao2mo_outcore_iobuf_words', 1e8)  # 1.6 GB
IOBUF_ROW_MIN = getattr(__config__, 'ao2mo_outcore_row_min', 160)
MAX_MEMORY = getattr(__config__, 'ao2mo_outcore_max_memory', 4000)  # 4GB

def _pair_correction(mol, pes_dm1, pes_dm2, mo_ene, U_mo, g_ppnp, g_pppn):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    
    assert pes_dm1.shape == (n2c, n2c)
    assert pes_dm2.shape == (n2c, n2c, n2c, n2c)
    assert mo_ene.shape == (n4c,)
    assert U_mo.shape == (n4c, n4c)
    assert g_ppnp.shape == (n2c, n2c, n2c, n2c)
    assert g_pppn.shape == (n2c, n2c, n2c, n2c)
    
    res = 0.0
    
    ### the first term ###
    
    term1 = numpy.einsum("is,ti,ts->it", U_mo[:n2c, n2c:], U_mo[n2c:, :n2c], pes_dm1, optimize=True)
    for i in range(n2c):
        for t in range(n2c):
            res += -term1[i, t] / (mo_ene[i] - mo_ene[t+n2c])
            #print("term     = ", term1[i, t])
            #print("ene_diff = ", (mo_ene[i] - mo_ene[t+n2c]))

    #print("res = ", res)

    ### the second term ###
    
    term2 = numpy.einsum("pris,ti,prts->it", g_ppnp, U_mo[n2c:, :n2c], pes_dm2, optimize=True)
    for i in range(n2c):
        for t in range(n2c):
            res += term2[i, t] / (mo_ene[i] - mo_ene[t+n2c])
    
    ### the third term ###
    
    term3 = numpy.einsum("it,prsi,prst->rips", U_mo[:n2c, n2c:], g_pppn, pes_dm2, optimize=True)
    for r in range(n2c):
        for i in range(n2c):
            for p in range(n2c):
                for s in range(n2c):
                    res += term3[r, i, p, s] / (mo_ene[r+n2c] + mo_ene[i] - mo_ene[p+n2c] - mo_ene[s+n2c])

    return res


def _r_outcore_Coulomb_g(mol, my_RDHF, prefix, mo_coeffs, max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE):

    from pyscf.ao2mo import r_outcore

    n2c = mol.nao_2c()
    
    #mo_coeff = my_RDHF.mo_coeff
    #mo_coeff_mat = numpy.matrix(mo_coeff)
    #mo_coeff_pes = mo_coeff_mat[:, n2c:]
    #mo_coeff_L = mo_coeff_pes[:n2c, :]
    #mo_coeff_S = mo_coeff_pes[n2c:, :]

    r_outcore.general(mol, (
        mo_coeffs[0][:n2c, :], 
        mo_coeffs[1][:n2c, :], 
        mo_coeffs[2][:n2c, :], 
        mo_coeffs[3][:n2c, :]), coulomb_LLLL %
                      prefix, intor="int2e_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (
        mo_coeffs[0][n2c:, :], 
        mo_coeffs[1][n2c:, :], 
        mo_coeffs[2][n2c:, :], 
        mo_coeffs[3][n2c:, :]), coulomb_SSSS %
                      prefix, intor="int2e_spsp1spsp2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (
        mo_coeffs[0][:n2c, :], 
        mo_coeffs[1][:n2c, :], 
        mo_coeffs[2][n2c:, :], 
        mo_coeffs[3][n2c:, :]), coulomb_LLSS %
                      prefix, intor="int2e_spsp2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (
        mo_coeffs[0][n2c:, :], 
        mo_coeffs[1][n2c:, :], 
        mo_coeffs[2][:n2c, :], 
        mo_coeffs[3][:n2c, :]), coulomb_SSLL %
                      prefix, intor="int2e_spsp1_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')

def _r_outcore_Breit_g(mol, my_RDHF, prefix, mo_coeffs, max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE):

    from pyscf.ao2mo import r_outcore

    n2c = mol.nao_2c()
    
    #mo_coeff = my_RDHF.mo_coeff
    #mo_coeff_mat = numpy.matrix(mo_coeff)
    #mo_coeff_pes = mo_coeff_mat[:, n2c:]
    #mo_coeff_L = mo_coeff_pes[:n2c, :]
    #mo_coeff_S = mo_coeff_pes[n2c:, :]

    r_outcore.general(mol, (
        mo_coeffs[0][:n2c, :], 
        mo_coeffs[1][n2c:, :], 
        mo_coeffs[2][:n2c, :], 
        mo_coeffs[3][n2c:, :]), breit_LSLS %
                      prefix, intor="int2e_breit_ssp1ssp2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (
        mo_coeffs[0][n2c:, :], 
        mo_coeffs[1][:n2c, :], 
        mo_coeffs[2][n2c:, :], 
        mo_coeffs[3][:n2c, :]), breit_SLSL %
                      prefix, intor="int2e_breit_sps1sps2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (
        mo_coeffs[0][:n2c, :], 
        mo_coeffs[1][n2c:, :], 
        mo_coeffs[2][n2c:, :], 
        mo_coeffs[3][:n2c, :]), breit_LSSL %
                      prefix, intor="int2e_breit_ssp1sps2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')
    r_outcore.general(mol, (
        mo_coeffs[0][n2c:, :], 
        mo_coeffs[1][:n2c, :], 
        mo_coeffs[2][:n2c, :], 
        mo_coeffs[3][n2c:, :]), breit_SLLS %
                      prefix, intor="int2e_breit_sps1ssp2_spinor", max_memory=max_memory, ioblk_size=ioblk_size, aosym='s1')

    
def _get_g(mol, my_RDHF, mo_coeff, type, incore=True):
    
    assert type in ["ppnp", "pppn"]
    
    n2c = mol.nao_2c()
    n4c = n2c * 2
    assert mo_coeff.shape == (n4c, n4c)
    
    mo_coeffs = []
    
    if type == "ppnp":
        mo_coeffs.append(mo_coeff[:, n2c:])
        mo_coeffs.append(mo_coeff[:, n2c:])
        mo_coeffs.append(mo_coeff[:, :n2c])
        mo_coeffs.append(mo_coeff[:, n2c:])
    else:
        mo_coeffs.append(mo_coeff[:, n2c:])
        mo_coeffs.append(mo_coeff[:, n2c:])
        mo_coeffs.append(mo_coeff[:, n2c:])
        mo_coeffs.append(mo_coeff[:, :n2c])
    
    
    with_breit = my_RDHF.with_breit
    with_gaunt = my_RDHF.with_gaunt
    
    if with_breit:
        INT_LSLS_name = "int2e_breit_ssp1ssp2_spinor"
        INT_SLSL_name = "int2e_breit_sps1sps2_spinor"
        INT_LSSL_name = "int2e_breit_ssp1sps2_spinor"
        INT_SLLS_name = "int2e_breit_sps1ssp2_spinor"
    else:
        if with_gaunt:
            INT_LSLS_name = "int2e_ssp1ssp2_spinor"
            INT_SLSL_name = "int2e_sps1sps2_spinor"
            INT_LSSL_name = "int2e_ssp1sps2_spinor"
            INT_SLLS_name = "int2e_sps1ssp2_spinor"
    
    if incore:
        int2e_res = numpy.zeros((n4c, n4c, n4c, n4c), dtype=numpy.complex128)
        c1 = .5 / lib.param.LIGHT_SPEED
        int2e_res[:n2c, :n2c, :n2c, :n2c] = mol.intor("int2e_spinor")  # LL LL
        tmp = mol.intor("int2e_spsp1_spinor") * c1**2
        int2e_res[n2c:, n2c:, :n2c, :n2c] = tmp  # SS LL
        int2e_res[:n2c, :n2c, n2c:, n2c:] = tmp.transpose(2, 3, 0, 1)  # LL SS
        int2e_res[n2c:, n2c:, n2c:, n2c:] = mol.intor("int2e_spsp1spsp2_spinor") * c1**4  # SS SS
        int2e_res = lib.einsum("ijkl,ip->pjkl", int2e_res, mo_coeffs[0].conj())
        int2e_res = lib.einsum("pjkl,jq->pqkl", int2e_res, mo_coeffs[1])
        int2e_res = lib.einsum("pqkl,kr->pqrl", int2e_res, mo_coeffs[2].conj())
        int2e_res = lib.einsum("pqrl,ls->pqrs", int2e_res, mo_coeffs[3])
    
        if with_breit or with_gaunt:
            int2e_bg = numpy.zeros((n4c, n4c, n4c, n4c), dtype=numpy.complex128)
            ##### (LS|LS) and (SL|SL) #####
            tmp = mol.intor(INT_LSLS_name) * c1**2
            int2e_bg[:n2c, n2c:, :n2c, n2c:] = tmp
            tmp = mol.intor(INT_SLSL_name) * c1**2
            int2e_bg[n2c:, :n2c, n2c:, :n2c] = tmp
            ##### (LS|SL) and (SL|LS) #####
            tmp2 = mol.intor(INT_LSSL_name) * c1**2
            int2e_bg[:n2c, n2c:, n2c:, :n2c] = tmp2  # (LS|SL)
            tmp2 = mol.intor(INT_SLLS_name) * c1**2
            int2e_bg[n2c:, :n2c, :n2c, n2c:] = tmp2  # (SL|LS)
            ###############################
            int2e_bg = lib.einsum("ijkl,ip->pjkl", int2e_bg, mo_coeffs[0].conj())
            int2e_bg = lib.einsum("pjkl,jq->pqkl", int2e_bg, mo_coeffs[1])
            int2e_bg = lib.einsum("pqkl,kr->pqrl", int2e_bg, mo_coeffs[2].conj())
            int2e_bg = lib.einsum("pqrl,ls->pqrs", int2e_bg, mo_coeffs[3])
            int2e_res += int2e_bg
        
    else:
        
        PREFIX = "RELA_4C_%d" % (numpy.random.randint(1, 19951201+1))
        
        _r_outcore_Coulomb_g(mol, my_RDHF, PREFIX, mo_coeffs)
        
        import h5py, os
        
        c1 = .5 / lib.param.LIGHT_SPEED
        
        feri_coulomb_LLLL = h5py.File(coulomb_LLLL % PREFIX, 'r')
        feri_coulomb_LLSS = h5py.File(coulomb_LLSS % PREFIX, 'r')
        feri_coulomb_SSLL = h5py.File(coulomb_SSLL % PREFIX, 'r')
        feri_coulomb_SSSS = h5py.File(coulomb_SSSS % PREFIX, 'r')
        
        int2e_res = numpy.array(feri_coulomb_LLLL['eri_mo'])
        int2e_res+= numpy.array(feri_coulomb_LLSS['eri_mo']) * c1**2
        int2e_res+= numpy.array(feri_coulomb_SSLL['eri_mo']) * c1**2
        int2e_res+= numpy.array(feri_coulomb_SSSS['eri_mo']) * c1**4
        
        #print("int2e_res.shape = ", int2e_res.shape)
        
        os.remove(coulomb_LLLL % PREFIX)
        os.remove(coulomb_LLSS % PREFIX)
        os.remove(coulomb_SSLL % PREFIX)
        os.remove(coulomb_SSSS % PREFIX)
        
        if with_breit:
            _r_outcore_Breit_g(mol, my_RDHF, PREFIX, mo_coeffs)

            feri_breit_LSLS = h5py.File(breit_LSLS % PREFIX, 'r')
            feri_breit_SLSL = h5py.File(breit_SLSL % PREFIX, 'r')
            feri_breit_LSSL = h5py.File(breit_LSSL % PREFIX, 'r')
            feri_breit_SLLS = h5py.File(breit_SLLS % PREFIX, 'r')

            int2e_res += numpy.array(feri_breit_LSLS['eri_mo']) * c1**2
            int2e_res += numpy.array(feri_breit_SLSL['eri_mo']) * c1**2
            int2e_res += numpy.array(feri_breit_LSSL['eri_mo']) * c1**2
            int2e_res += numpy.array(feri_breit_SLLS['eri_mo']) * c1**2
        
            os.remove(breit_LSLS % PREFIX)
            os.remove(breit_SLSL % PREFIX)
            os.remove(breit_LSSL % PREFIX)
            os.remove(breit_SLLS % PREFIX)
            
        else:
            if with_gaunt:
                raise NotImplementedError
        
        #print("int2e_res.shape = ", int2e_res.shape)
        
        int2e_res = int2e_res.reshape((n2c, n2c, n2c, n2c))
        
    return int2e_res

def _Generate_InputFile_SiCI(File,
                             Segment,
                             nelec_val,
                             cmin=0.0,
                             perturbation=1,
                             Task=None,
                             tol=1e-8,
                             selection=1,
                             dumprdm=0
                             ):
    """Generate the input file
    """
    with open(File, "w") as f:
        f.write("nsegment=%s\n" % Segment)
        f.write("nvalelec=%d\n" % nelec_val)
        f.write("rotatemo=%d\n" % 0)
        f.write("cmin=%s\n" % cmin)
        f.write("perturbation=%d 0\n" % perturbation)
        f.write("dumprdm=%d\n" % dumprdm)
        f.write("task=%s\n" % Task)
        f.write("etol=%e\n" % tol)
        f.write("selection=%d\n" % selection)
        f.write("print=12\n")

APP = "/home/ningzhangcaltech/Github_Repo/iCIPT2_CXX/bin/iCIPT2_Spinor.exe"

def _pack_fzc(dm1, dm2, norb, nfzc):
    
    ### rdm1 ### 
    
    for i in range(2*nfzc):
        dm1[i,i] = 1.0
    
    ### rdm2 pure core ###
    
    ### iiii
    
    for i in range(nfzc):
        dm2[2*i,2*i,2*i+1,2*i+1] = 1.0 
        dm2[2*i,2*i+1,2*i+1,2*i] = 1.0 
        dm2[2*i+1,2*i,2*i,2*i+1] = 1.0 
        dm2[2*i+1,2*i+1,2*i,2*i] = 1.0 

    ### iijj
    
    for i in range(nfzc):
        for j in range(nfzc):
            if i == j:
                continue
            
            dm2[2*i,2*i,2*j,2*j] = 1.0
            dm2[2*i,2*i,2*j+1,2*j+1] = 1.0
            dm2[2*i+1,2*i+1,2*j,2*j] = 1.0
            dm2[2*i+1,2*i+1,2*j+1,2*j+1] = 1.0
            
            dm2[2*i,2*j,2*j,2*i] = -1.0
            dm2[2*i,2*j+1,2*j+1,2*i] = -1.0
            dm2[2*i+1,2*j,2*j,2*i+1] = -1.0
            dm2[2*i+1,2*j+1,2*j+1,2*i+1] = -1.0

    ### rdm2 semicore

    for i in range(2*nfzc):
        dm2[i,i,2*nfzc:2*norb,2*nfzc:2*norb] = dm1[2*nfzc:2*norb,2*nfzc:2*norb]
        dm2[2*nfzc:2*norb,2*nfzc:2*norb,i,i] = dm1[2*nfzc:2*norb,2*nfzc:2*norb]
        
        dm2[i,2*nfzc:2*norb,2*nfzc:2*norb,i] = -dm1[2*nfzc:2*norb,2*nfzc:2*norb].T
        dm2[2*nfzc:2*norb,i,i,2*nfzc:2*norb] = -dm1[2*nfzc:2*norb,2*nfzc:2*norb].T
    
    return dm1, dm2

if __name__ == "__main__":
    # mol = gto.M(atom='H 0 0 0; H 0 0 1; O 0 1 0', basis='sto-3g', verbose=5)
    mol = gto.M(atom='F 0 0 0', basis='cc-pvdz-dk', verbose=5,
                charge=-1, spin=0, symmetry="d2h")
    mol.build()
    mf = scf.dhf.RDHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    mf.with_breit = True
    mf.kernel()
    
    mo_coeff = mf.mo_coeff
    
    print("mo_coeff.shape = ", mo_coeff.shape)
    
    Dump_Cmoao_4C("F_4C", mo_coeff)    
    
    hcore = mf.get_hcore()
    print("hcore.shape = ", hcore.shape)
    dm = mf.make_rdm1()
    print("dm.shape = ", dm.shape)
    
    fock = mf.get_fock(dm=dm)
    ovlp = mf.get_ovlp()
    e, mo_coeff = mf._eigh(fock, ovlp)
    print("mo_ene = ", e)

    hcore_mo = reduce(numpy.dot, (mo_coeff.conj().T, hcore, mo_coeff))
    fock_mo  = reduce(numpy.dot, (mo_coeff.conj().T, fock, mo_coeff))
    U_mo     = fock_mo - hcore_mo
    
    print("fock_mo[4]  = ", fock_mo[4])
    print("hcore_mo[4] = ", hcore_mo[4])
    print("U_mo[4]     = ", U_mo[4])
    
    g_pppn = _get_g(mol, mf, mo_coeff, "pppn", incore=False)
    g_ppnp = _get_g(mol, mf, mo_coeff, "ppnp", incore=False)
    
    int2e2, breit_2 = FCIDUMP_Rela4C(
        mol, mf, with_breit=None, filename="FCIDUMP_4C_incore", mode="incore", debug=False)

    _Generate_InputFile_SiCI(
        File="input",
        Segment="1 0 4 4 0 %d" % (mol.nao-9),
        nelec_val=7,
        cmin="0.0",
        perturbation=0,
        Task="1 0 6 1 1 1 1 1 1",
        dumprdm=7
    )
    
    import os
    os.system("%s %s %s 1>%s.out 2>%s.err" % (APP, "input", "FCIDUMP_4C_incore", "F", "F"))
    
    rdm1 = ReadIn_RDM1_4C("SpinTwo_%d_Irrep_%d_rdm1.csv" % (1,0), mol.nao * 2, 6)
    rdm2 = ReadIn_RDM2_4C("SpinTwo_%d_Irrep_%d_rdm2.csv" % (1,0), mol.nao * 2, 6)
    
    print("rdm1.shape = ", rdm1.shape)
    print("rdm2.shape = ", rdm2.shape)
    
    istate1 = 4 
    istate2 = 2
    
    rdm1_average = (rdm1[0] + rdm1[1] + rdm1[2] + rdm1[3]) / 4
    rdm2_average = (rdm2[0] + rdm2[1] + rdm2[2] + rdm2[3]) / 4
    
    rdm1_average, rdm2_average = _pack_fzc(rdm1_average, rdm2_average, 1, 9)
    
    print(rdm1_average[1])
    
    print(_pair_correction(mol, rdm1_average, rdm2_average, mf.mo_energy, U_mo, g_ppnp, g_pppn))
    
    rdm1_average = (rdm1[4] + rdm1[5]) / 2
    rdm2_average = (rdm2[4] + rdm2[5]) / 2
    
    rdm1_average, rdm2_average = _pack_fzc(rdm1_average, rdm2_average, 1, 9)
    
    print(rdm1_average[1])
    
    print(_pair_correction(mol, rdm1_average, rdm2_average, mf.mo_energy, U_mo, g_ppnp, g_pppn))
    