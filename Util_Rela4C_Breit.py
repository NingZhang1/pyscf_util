from pyscf import gto, scf, lib
# import libzquatev
import sys
import pickle

import zquatev
import numpy
from functools import reduce
from pyscf import lib
from pyscf import tools
import pyscf

import re

import Util_Rela4C

from pyscf.scf import _vhf


def time_reversal_matrix(mol, mat):
    ''' T(A_ij) = A[T(i),T(j)]^*
    '''
    tao = numpy.asarray(mol.time_reversal_map())
    n2c = tao.size
    # tao(i) = -j  means  T(f_i) = -f_j
    # tao(i) =  j  means  T(f_i) =  f_j
    idx = abs(tao) - 1  # -1 for C indexing convention
    # :signL = [(1 if x>0 else -1) for x in tao]
    # :sign = numpy.hstack((signL, signL))

    # :tmat = numpy.empty_like(mat)
    # :for j in range(mat.__len__()):
    #:    for i in range(mat.__len__()):
    #:        tmat[idx[i],idx[j]] = mat[j,i] * sign[i]*sign[j]
    # :return tmat.conjugate()
    sign_mask = tao < 0
    if mat.shape[0] == n2c * 2:
        idx = numpy.hstack((idx, idx+n2c))
        sign_mask = numpy.hstack((sign_mask, sign_mask))

    tmat = mat[idx[:, None], idx]
    tmat[sign_mask, :] *= -1
    tmat[:, sign_mask] *= -1
    return tmat.conj()


def _ensure_time_reversal_symmetry(mol, mat):
    if mat.ndim == 2:
        mat = [mat]
    for m in mat:
        if abs(m - time_reversal_matrix(mol, m)).max() > 1e-9:
            raise RuntimeError('Matrix does have time reversal symmetry')


DEBUG = False


def _call_veff_gaunt_breit(mol, dm, hermi=1, mf_opt=None, with_breit=False):
    if with_breit:
        # integral function int2e_breit_ssp1ssp2_spinor evaluates
        # -1/2[alpha1*alpha2/r12 + (alpha1*r12)(alpha2*r12)/r12^3]
        intor_prefix = 'int2e_breit_'
    else:
        # integral function int2e_ssp1ssp2_spinor evaluates only
        # alpha1*alpha2/r12. Minus sign was not included.
        intor_prefix = 'int2e_'

    if hermi == 0 and DEBUG:
        _ensure_time_reversal_symmetry(mol, dm)

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n_dm = 1
        n2c = dm.shape[0] // 2
        dmls = dm[:n2c, n2c:].copy()
        dmsl = dm[n2c:, :n2c].copy()
        dmll = dm[:n2c, :n2c].copy()
        dmss = dm[n2c:, n2c:].copy()
        dms = [dmsl, dmsl, dmls, dmll, dmss]
    else:
        n_dm = len(dm)
        n2c = dm[0].shape[0] // 2
        dmll = [dmi[:n2c, :n2c].copy() for dmi in dm]
        dmls = [dmi[:n2c, n2c:].copy() for dmi in dm]
        dmsl = [dmi[n2c:, :n2c].copy() for dmi in dm]
        dmss = [dmi[n2c:, n2c:].copy() for dmi in dm]
        dms = dmsl + dmsl + dmls + dmll + dmss

    # vj = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex128)
    # vk = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex128)

    jks = ('lk->s1ij',) * n_dm \
        + ('jk->s1il',) * n_dm
    vx = _vhf.rdirect_bindm(intor_prefix+'ssp1ssp2_spinor', 's1', jks, dms[:n_dm*2], 1,
                            mol._atm, mol._bas, mol._env, mf_opt)
    # vj[:, :n2c, n2c:] = vx[:n_dm, :, :]
    # vk[:, :n2c, n2c:] = vx[n_dm:, :, :]

    vj_LS_1 = vx[:n_dm, :, :]
    vk_LS = vx[n_dm:, :, :]

    jks = ('lk->s1ij',) * n_dm \
        + ('li->s1kj',) * n_dm \
        + ('jk->s1il',) * n_dm
    vx = _vhf.rdirect_bindm(intor_prefix+'ssp1sps2_spinor', 's1', jks, dms[n_dm*2:], 1,
                            mol._atm, mol._bas, mol._env, mf_opt)
    # vj[:, :n2c, n2c:] += vx[:n_dm, :, :]
    # vk[:, n2c:, n2c:] = vx[n_dm:n_dm*2, :, :]
    # vk[:, :n2c, :n2c] = vx[n_dm*2:, :, :]

    vj_LS_2 = vx[:n_dm, :, :]
    vk_SS = vx[n_dm:n_dm*2, :, :]
    vk_LL = vx[n_dm*2:, :, :]

    # if hermi == 1:
    #     vj[:,n2c:,:n2c] = vj[:,:n2c,n2c:].transpose(0,2,1).conj()
    #     vk[:,n2c:,:n2c] = vk[:,:n2c,n2c:].transpose(0,2,1).conj()
    # elif hermi == 2:
    #     vj[:,n2c:,:n2c] = -vj[:,:n2c,n2c:].transpose(0,2,1).conj()
    #     vk[:,n2c:,:n2c] = -vk[:,:n2c,n2c:].transpose(0,2,1).conj()
    # else:
    #     raise NotImplementedError

    # vj = vj.reshape(dm.shape)
    # vk = vk.reshape(dm.shape)
    c1 = .5 / lib.param.LIGHT_SPEED

    vj_LS_1 = vj_LS_1.reshape((n2c, n2c))
    vk_LS = vk_LS.reshape((n2c, n2c))
    vj_LS_2 = vj_LS_2.reshape((n2c, n2c))
    vk_SS = vk_SS.reshape((n2c, n2c))
    vk_LL = vk_LL.reshape((n2c, n2c))

    if with_breit:
        # vj *= c1**2
        # vk *= c1**2

        vj_LS_1 *= c1**2
        vk_LS *= c1**2
        vj_LS_2 *= c1**2
        vk_SS *= c1**2
        vk_LL *= c1**2

    else:
        # vj *= -c1**2
        # vk *= -c1**2

        vj_LS_1 *= -c1**2
        vk_LS *= -c1**2
        vj_LS_2 *= -c1**2
        vk_SS *= -c1**2
        vk_LL *= -c1**2

    return vj_LS_1, vj_LS_2, vk_LS, vk_SS, vk_LL


if __name__ == "__main__":
    mol = gto.M(atom='H 0 0 0; H 0 0 1; O 0 1 0', basis='sto-3g', verbose=5)
    # mol = gto.M(atom='F 0 0 0', basis='cc-pvdz', verbose=5, charge=-1, spin=0)
    # mf = scf.RHF(mol)
    # mf.kernel()
    # mf.analyze()
    mf = scf.dhf.RDHF(mol)
    mf.conv_tol = 1e-12
    # mf.kernel()

    # mf.with_gaunt = True
    # mf.kernel()

    mf.with_breit = True
    mf.kernel()

    ###### test the contraction of dm with int2e_full ######

    dm1 = mf.make_rdm1()

    n2c = dm1.shape[0] // 2

    dmls = dm1[:n2c, n2c:].copy()
    dmsl = dm1[n2c:, :n2c].copy()
    dmll = dm1[:n2c, :n2c].copy()
    dmss = dm1[n2c:, n2c:].copy()

    for i in range(n2c):
        for j in range(n2c):
            if abs(dmls[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, dmls[i, j])

    for i in range(n2c):
        for j in range(n2c):
            if abs(dmsl[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, dmsl[i, j])
    
    for i in range(n2c):
        for j in range(n2c):
            if abs(dmll[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, dmll[i, j])
    
    for i in range(n2c):
        for j in range(n2c):
            if abs(dmss[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, dmss[i, j])

    print("dm1.shape = ", dm1.shape)
    # print(dm1)

    c1 = .5 / lib.param.LIGHT_SPEED

    tmp1 = mol.intor("int2e_breit_ssp1ssp2_spinor") * c1**2 ## (LS|LS)

    n2c = mf.mo_coeff.shape[1] // 2
    pes_mo = mf.mo_coeff[:, n2c:]

    tmp2 = mol.intor("int2e_breit_ssp1sps2_spinor") * c1**2 ## (LS|SL)

    vj_LS_1, vj_LS_2, vk_LS, vk_SS, vk_LL = _call_veff_gaunt_breit(
        mol, dm1, with_breit=True)

    print("vj_LS_1.shape = ", vj_LS_1.shape)
    print("vj_LS_2.shape = ", vj_LS_2.shape)
    print("vk_LS.shape = ", vk_LS.shape)
    print("vk_SS.shape = ", vk_SS.shape)
    print("vk_LL.shape = ", vk_LL.shape)

    vj_LS_1_test = numpy.einsum("ijkl,kl->ij", tmp1, dmls)

    print("diff = ", numpy.linalg.norm(vj_LS_1 - vj_LS_1_test))

    for i in range(n2c):
        for j in range(n2c):
            if abs(vj_LS_1[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, vj_LS_1[i, j], vj_LS_1_test[i, j])

    vj_LS_2_test = numpy.einsum("ijkl,kl->ij", tmp2, dmsl)

    print("diff = ", numpy.linalg.norm(vj_LS_2 - vj_LS_2_test))
    
    for i in range(n2c):
        for j in range(n2c):
            if abs(vj_LS_2[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, vj_LS_2[i, j], vj_LS_2_test[i, j])

    vk_LS_test = numpy.einsum("ijkl,kj->il", tmp1, dmls.conj()) ## ? 
    vk_LS_test_2 = numpy.einsum("ijkl,jk->il", tmp1, dmsl) ## ? 

    print("diff = ", numpy.linalg.norm(vk_LS - vk_LS_test))
    print("diff = ", numpy.linalg.norm(vk_LS - vk_LS_test_2))

    for i in range(n2c):
        for j in range(n2c):
            if abs(vk_LS[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, vk_LS[i, j], vk_LS_test[i, j])

    tmp3 = tmp2.transpose(2, 3, 0, 1) ## (SL|LS)

    vk_SS_test = numpy.einsum("ijkl,kj->il", tmp3, dmll.conj())

    print("diff = ", numpy.linalg.norm(vk_SS - vk_SS_test))

    for i in range(n2c):
        for j in range(n2c):
            if abs(vk_SS[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, vk_SS[i, j], vk_SS_test[i, j])

    vk_LL_test = numpy.einsum("ijkl,kj->il", tmp2, dmss.conj())

    print("diff = ", numpy.linalg.norm(vk_LL - vk_LL_test))

    for i in range(n2c):
        for j in range(n2c):
            if abs(vk_LL[i, j]) > 1e-8:
                print("i = ", i, "j = ", j, vk_LL[i, j], vk_LL_test[i, j])