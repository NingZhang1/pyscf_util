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
from Util_Rela4C_PC import _get_g, _Generate_InputFile_SiCI, _pair_correction, _pack_fzc

APP = "/home/ningzhangcaltech/Github_Repo/iCIPT2_CXX/bin/iCIPT2_Spinor.exe"

TASK = {
    ###### F ######
    "F":{
        'charge':-1,
        'spin':0,
        'with_breit':True,
        "basis":"unc-ccpvtz-dk",
        "state":"1 0 6 1 1 1 1 1 1",
        "degenerate_state":[4,2],
        "nvalelec":7,
        'task':[
        {
            "name":"fzc_1s_corr_minimal",
            "segment":"1 0 2 2 0 %d",
            "cmin":"0.0",
            "nleft":5,
            "nfzc": 1,
            "norb": 5, # none for all vir
        },
        {
            "name":"fzc_1s_corr_sp",
            "segment":"1 0 4 4 0 %d",
            "cmin":"0.0",
            "nleft":9,
            "nfzc": 1,
            "norb": 9, # none for all vir
        },
        {
            "name":"fzc_1s_corr_allvir",
            "segment":"1 0 4 4 %d 0",
            "cmin":"5e-6",
            "nleft":9,
            "nfzc": 1,
            "norb": None, # none for all vir
        },
        ]
    },
    ###### Cl ######
    "Cl":{
        'charge':-1,
        'spin':0,
        'with_breit':True,
        "basis":"unc-ccpvtz-dk",
        "state":"1 0 6 1 1 1 1 1 1",
        "degenerate_state":[4,2],
        "nvalelec":7,
        'task':[
        {
            "name":"fzc_1s2s2p_corr_minimal",
            "segment":"5 0 2 2 0 %d",
            "cmin":"0.0",
            "nleft":9,
            "nfzc": 5,
            "norb": 9, # none for all vir
        },
        {
            "name":"fzc_1s2s2p_corr_sp",
            "segment":"5 0 4 4 0 %d",
            "cmin":"0.0",
            "nleft":13,
            "nfzc": 5,
            "norb": 13, # none for all vir
        },
        {
            "name":"fzc_1s2s2p_corr_allvir",
            "segment":"5 0 4 4 %d 0",
            "cmin":"5e-6",
            "nleft":13,
            "nfzc": 5,
            "norb": None, # none for all vir
        },
        ]
    },
    ###### Br ######
    "Br":{
        'charge':-1,
        'spin':0,
        'with_breit':False,
        "basis":"unc-ccpvtz-dk",
        "state":"1 0 6 1 1 1 1 1 1",
        "degenerate_state":[4,2],
        "nvalelec":7,
        'task':[
        {
            "name":"fzc_1s2s2p_corr_minimal",
            "segment":"14 0 2 2 0 %d",
            "cmin":"0.0",
            "nleft":18,
            "nfzc": 14,
            "norb": 18, # none for all vir
        },
        {
            "name":"fzc_1s2s2p_corr_sp",
            "segment":"14 0 4 4 0 %d",
            "cmin":"0.0",
            "nleft":22,
            "nfzc": 14,
            "norb": 22, # none for all vir
        },
        {
            "name":"fzc_1s2s2p_corr_allvir",
            "segment":"14 0 4 4 %d 0",
            "cmin":"5e-6",
            "nleft":22,
            "nfzc": 14,
            "norb": None, # none for all vir
        },
        ]
    },
    ###### I ######
    "I":{
        'charge':-1,
        'spin':0,
        'with_breit':False,
        "basis":"unc-ccpvtz-dk",
        "state":"1 0 6 1 1 1 1 1 1",
        "degenerate_state":[4,2],
        "nvalelec":7,
        'task':[
        {
            "name":"fzc_1s2s2p_corr_minimal",
            "segment":"23 0 2 2 0 %d",
            "cmin":"0.0",
            "nleft":27,
            "nfzc": 23,
            "norb": 27, # none for all vir
        },
        {
            "name":"fzc_1s2s2p_corr_sp",
            "segment":"23 0 4 4 0 %d",
            "cmin":"0.0",
            "nleft":31,
            "nfzc": 23,
            "norb": 31, # none for all vir
        },
        {
            "name":"fzc_1s2s2p_corr_allvir",
            "segment":"23 0 4 4 %d 0",
            "cmin":"5e-6",
            "nleft":31,
            "nfzc": 23,
            "norb": None, # none for all vir
        },
        ]
    },
}

if __name__ == "__main__":
    
    for task in TASK:
        
        mol = gto.M(atom='F 0 0 0', 
                    basis=TASK[task]["basis"], 
                    verbose=5,
                    charge=TASK[task]['charge'], 
                    spin=TASK[task]['spin'], 
                    symmetry="d2h")
        mol.build()
        
        mf = scf.dhf.RDHF(mol)
        mf.conv_tol = 1e-12
        if TASK[task]['with_breit']:
            mf.with_breit = True
        mf.kernel()
        
        ####### build U_mo #######
        
        hcore    = mf.get_hcore()
        dm       = mf.make_rdm1()
        fock     = mf.get_fock(dm=dm)
        mo_coeff = mf.mo_coeff
        hcore_mo = reduce(numpy.dot, (mo_coeff.conj().T, hcore, mo_coeff))
        fock_mo  = reduce(numpy.dot, (mo_coeff.conj().T, fock, mo_coeff))
        U_mo     = fock_mo - hcore_mo
        
        g_pppn = _get_g(mol, mf, mo_coeff, "pppn")
        g_ppnp = _get_g(mol, mf, mo_coeff, "ppnp")
        
        FCIDUMP_Rela4C(mol, mf, with_breit=None, filename="FCIDUMP_4C_incore", mode="incore", debug=False)
        
        n2c = mol.nao * 2
        
        for _task_ in TASK[task]['task']:
            _Generate_InputFile_SiCI(
                File="input.%s" % _task_['name'],
                Segment=_task_['segment'] % (mol.nao-_task_['nleft']),
                nelec_val=TASK[task]['nvalelec'],
                cmin=_task_['cmin'],
                perturbation=0,
                Task=TASK[task]['state'],
                dumprdm=7
            )
            
            outname = "%s_%s" % (task, _task_['name'])
            
            import os
            os.system("%s %s %s 1>%s.out 2>%s.err" % (APP, "input.%s" % _task_['name'], "FCIDUMP_4C_incore", outname, outname)) 
            
            #####################################################
            
            ## read rdm ## 
            
            nstate_tot = numpy.sum(TASK[task]['degenerate_state'])
            
            rdm1 = ReadIn_RDM1_4C("SpinTwo_%d_Irrep_%d_rdm1.csv" % (TASK[task]['nvalelec']%2,0), mol.nao * 2, nstate_tot)
            rdm2 = ReadIn_RDM2_4C("SpinTwo_%d_Irrep_%d_rdm2.csv" % (TASK[task]['nvalelec']%2,0), mol.nao * 2, nstate_tot)
            
            ## pc ##
            
            istate = 0
            
            for ideg in range(len(TASK[task]['degenerate_state'])):
                rdm1_tmp = numpy.zeros((n2c, n2c), dtype=numpy.complex128)
                rdm2_tmp = numpy.zeros((n2c, n2c, n2c, n2c), dtype=numpy.complex128)
                for i in range(istate, istate+TASK[task]['degenerate_state'][ideg]):
                    rdm1_tmp += rdm1[i]
                    rdm2_tmp += rdm2[i]
                istate += TASK[task]['degenerate_state'][ideg]
                
                rdm1_tmp /= TASK[task]['degenerate_state'][ideg]
                rdm2_tmp /= TASK[task]['degenerate_state'][ideg]
                
                if _task_['norb'] is None:
                    norb = mol.nao
                else:
                    norb = _task_['norb']
                rdm1_tmp, rdm2_tmp = _pack_fzc(rdm1_tmp, rdm2_tmp, _task_['nfzc'], norb)
                
                PC = _pair_correction(mol, rdm1_tmp, rdm2_tmp, mf.mo_energy, U_mo, g_ppnp, g_pppn)                
                print("pair correction for atm %s task %s state %d is %12.8e" % (task, _task_['name'], ideg, PC))
            
            