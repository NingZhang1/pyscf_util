# import SOC_Driver
# import RelInt
import pyscf
from pyscf import gto, lib
from functools import reduce
import numpy
from pyscf.data import nist
import copy

epsilon = numpy.zeros((3, 3, 3))
epsilon[0][1][2] = 1.0
epsilon[0][2][1] = -1.0
epsilon[1][2][0] = 1.0
epsilon[1][0][2] = -1.0
epsilon[2][0][1] = 1.0
epsilon[2][1][0] = -1.0


def Frobenius_Norm(_mat):
    return numpy.trace(numpy.dot(_mat.T, _mat))


def Check_AntiSymmetry(_mat):
    return Frobenius_Norm(_mat+_mat.T)


def _get_r(s, snesc):
    # R^dag \tilde{S} R = S
    # R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}
    w, v = numpy.linalg.eigh(s)
    idx = w > 1e-14
    v = v[:, idx]
    w_sqrt = numpy.sqrt(w[idx])
    w_invsqrt = 1 / w_sqrt

    # eigenvectors of S as the new basis
    snesc = reduce(numpy.dot, (v.conj().T, snesc, v))
    r_mid = numpy.einsum('i,ij,j->ij', w_invsqrt, snesc, w_invsqrt)
    w1, v1 = numpy.linalg.eigh(r_mid)
    idx1 = w1 > 1e-14
    v1 = v1[:, idx1]
    r_mid = numpy.dot(v1/numpy.sqrt(w1[idx1]), v1.conj().T)
    r = numpy.einsum('i,ij,j->ij', w_invsqrt, r_mid, w_sqrt)
    # Back transform to AO basis
    r = reduce(numpy.dot, (v, r, v.conj().T))
    return r


def _get_rmat(_sfx1c, x=None):
    '''The matrix (in AO basis) that changes metric from NESC metric to NR metric'''
    xmol = _sfx1c.with_x2c.get_xmol()[0]
    if x is None:
        x = _sfx1c.with_x2c.get_xmat(xmol)
    c = lib.param.LIGHT_SPEED
    s = xmol.intor_symmetric('int1e_ovlp')
    t = xmol.intor_symmetric('int1e_kin')
    s1 = s + reduce(numpy.dot, (x.conj().T, t, x)) * (.5/c**2)
    return _get_r(s, s1)


def get_X_R_dm_LL_LS_SS(_mol, _sfX2C, _cmoao, _dm1):
    # with uncontracted orbitals, more precise!
    xmol, contr_coeff = _sfX2C.with_x2c.get_xmol(_mol)
    # use xmol
    nao = xmol.nao
    X0 = _sfX2C.with_x2c.get_xmat()
    R0 = _get_rmat(_sfX2C)
    nao = xmol.nao
    # Get RDM
    dm1_AO = reduce(numpy.dot, (_cmoao, _dm1 / 2.0, _cmoao.T))
    if _sfX2C.with_x2c.xuncontract and contr_coeff is not None:
        dm1_AO_uncontracted = reduce(
            numpy.dot, (contr_coeff, dm1_AO, contr_coeff.T))
    else:
        dm1_AO_uncontracted = dm1_AO.copy()
    dm1_LL = reduce(
        numpy.dot, (R0, dm1_AO_uncontracted, R0.T))  # R0 is real
    dm1_LS = numpy.dot(dm1_LL, X0.T)
    # dm1_SL = numpy.dot(X0, dm1_LL)
    dm1_SS = reduce(numpy.dot, (X0, dm1_LL, X0.T))

    return xmol, contr_coeff, X0, R0, dm1_LL, dm1_LS, dm1_SS


def fetch_soc2e_SOMF_BP(_mol, _cmoao, _dm1, _SOMF=True):
    nao = _mol.nao
    # if nao >= DO_SHELL_BY_SHELL_CRITERION:
    #     return fetch_soc2e_SOMF_BP_shell_by_shell(_mol, _cmoao, _dm1)
    # RDM-1 in the atomic basis set
    cmoao = numpy.matrix(_cmoao)
    # rdm1_atom = reduce(numpy.dot, (cmoao.T.I, _dm1, cmoao.I)) Wrong , Totally Wrong !
    # Copy from pyscf/mcscf/addons.py make_rdm12()
    rdm1_atom = reduce(numpy.dot, (cmoao, _dm1, cmoao.T)
                       )  # I am so stupid here !
    # 2e part, Spin-Orbit Mean-Field, Eqn (10) -- (16)
    # hso2e = _mol.intor('int2e_p1vxp1').reshape(3, nao, nao, nao, nao)
    # SOMF
    hSOMF_atom = _mol.intor('int1e_pnucxp', 3)
    if _SOMF:
        vj, vk, vk2 = pyscf.scf.jk.get_jk(_mol, [rdm1_atom, rdm1_atom, rdm1_atom], [
        'ijkl,kl->ij', 'ijkl,jk->il', 'ijkl,li->kj'], intor='int2e_p1vxp1', comp=3)
        hSOMF_atom += vj - 1.5 * vk - 1.5 * vk2

    # hSOMF_atom += _mol.intor('int1e_pnucxp', 3)

    # Check AntiSymmetry
    for x in hSOMF_atom:
        print("Check Antisymmetry in fetch_soc2e_SOMF_BP %e" %
              (Check_AntiSymmetry(x)))

    return hSOMF_atom


def _Get_Wsd(_V):
    vx = _V[0]
    vy = _V[1]
    vz = _V[2]
    return numpy.bmat([[vz, vx+vy*complex(0, -1)], [vx+vy*complex(0, 1), -vz]]) * complex(0, 1)


def _Mat_GetReal(_M):
    nao = _M.shape[0]
    # print(_M.shape)
    # print(_M)
    M_Real = numpy.zeros([nao, nao])
    M_Imag = numpy.zeros([nao, nao])
    for i in range(0, nao):
        M_Real[i] = _M[i].real
        M_Imag[i] = _M[i].imag
    # print("In _Mat_GetReal pure real ", numpy.allclose(
    #     M_Imag, numpy.zeros([nao, nao])))
    return M_Real


def _Get_Wsf_Wxyz(_W):
    # print(_W)
    W = copy.deepcopy(_W)
    W *= complex(0, -1)
    nao = int(W.shape[0]/2)
    # print(nao)
    W11 = W[:nao, :nao]
    W12 = W[:nao, nao:]
    W21 = W[nao:, :nao]
    W22 = W[nao:, nao:]

    # print("Wsf get real")
    Wsf = _Mat_GetReal((W11+W22) / complex(0.0, -2.0))  # bug here !
    wz = _Mat_GetReal((W11-W22) / 2.0)
    wx = _Mat_GetReal((W12+W21) / 2.0)
    wy = _Mat_GetReal((W12-W21) / complex(0.0, -2.0))
    return Wsf, numpy.asarray([wx, wy, wz])


def fetch_X2C_soDKH1(_mol, _sfX2C, _cmoao, _dm1, _test=False, _get_1e=True, _get_2e=True):
    # with uncontracted orbitals, more precise!
    xmol, contr_coeff = _sfX2C.with_x2c.get_xmol(_mol)
    # get rdm in AO base
    dm1_AO = reduce(numpy.dot, (_cmoao, _dm1 / 2.0, _cmoao.T))
    if _sfX2C.with_x2c.xuncontract and contr_coeff is not None:
        dm1_AO_uncontracted = reduce(
            numpy.dot, (contr_coeff, dm1_AO, contr_coeff.T))
    else:
        dm1_AO_uncontracted = dm1_AO.copy()
    # print('dm1_AO = dm1_AO_uncontracted',
    #       numpy.allclose(dm1_AO, dm1_AO_uncontracted))
    # Get X matrix and R matrix
    X0 = _sfX2C.with_x2c.get_xmat()
    # R0 = _sfX2C.with_x2c._get_rmat()
    R0 = _get_rmat(_sfX2C)
    if _test:
        X0 = numpy.eye(xmol.nao)
        R0 = numpy.eye(xmol.nao)
    nao = xmol.nao
    ZeroMat = numpy.zeros((nao, nao))
    # print("ZeroMat Size", ZeroMat.shape)
    # print("ZeroMat Size", X0.shape)
    # print("ZeroMat Size", R0.shape)
    X_plus_Full = numpy.bmat([[X0, ZeroMat], [ZeroMat, X0]])
    R_plus_Full = numpy.bmat([[R0, ZeroMat], [ZeroMat, R0]])
    # Get Wsd 1e part
    E_plus_1_sd = numpy.zeros((3, nao, nao))
    if _get_1e:
        Wsd = xmol.intor('int1e_pnucxp', 3)
        WsdMat = _Get_Wsd(Wsd)
        E_plus_1 = reduce(numpy.dot, (R_plus_Full.conj().T,
                                      X_plus_Full.conj().T, WsdMat, X_plus_Full, R_plus_Full))
        E_plus_1_sf, E_plus_1_sd = _Get_Wsf_Wxyz(
            E_plus_1)  # E_plus_1_sf should be zero matrix !
        for x in E_plus_1_sd:
            print("Check Antisymmetry in fetch_X2C_soDKH1 get 1e %e" %
                  (Check_AntiSymmetry(x)))
    g = numpy.zeros((3, nao, nao))
    if _get_2e:
        # Get RDM
        dm1_LL = reduce(
            numpy.dot, (R0, dm1_AO_uncontracted, R0.T))  # R0 is real
        dm1_LS = numpy.dot(dm1_LL, X0.T)
        # dm1_SL = numpy.dot(X0, dm1_LL)
        dm1_SS = reduce(numpy.dot, (X0, dm1_LL, X0.T))

        g_LL_1, g_LS_1, g_LS_2, g_SS_1, g_SS_2, g_SS_3 = pyscf.scf.jk.get_jk(xmol,
                                                                             [dm1_SS, dm1_LS, dm1_LS,
                                                                              dm1_LL, dm1_LL, dm1_LL],
                                                                             ['abcd,ac->bd',
                                                                                 'abcd,bc->ad',
                                                                                 'abcd,ac->bd',
                                                                                 'abcd,dc->ab',
                                                                                 'abcd,cd->ab',
                                                                                 'abcd,bd->ac',
                                                                              ],
                                                                             intor='int2e_ip1ip2',
                                                                             comp=9)
        g_LL = -2.0 * g_LL_1
        g_LS = -1.0 * g_LS_1 - 1.0 * g_LS_2
        g_SS = -2.0 * g_SS_1 - 2.0 * g_SS_2 + 2.0 * g_SS_3

        g_LL = g_LL.reshape(3, 3, xmol.nao, xmol.nao)
        g_LS = g_LS.reshape(3, 3, xmol.nao, xmol.nao)
        g_SS = g_SS.reshape(3, 3, xmol.nao, xmol.nao)

        # print(g_LL.shape)

        g_LL = numpy.einsum("lmn,mnab->lab", epsilon, g_LL)
        g_LS = numpy.einsum("lmn,mnab->lab", epsilon, g_LS)
        g_SS = numpy.einsum("lmn,mnab->lab", epsilon, g_SS)

        g_SL = numpy.asarray([-x.T for x in g_LS])

        # contract X
        g_LS = numpy.asarray(
            [reduce(numpy.dot, (x, X0)) for x in g_LS])
        g_SL = numpy.asarray(
            [reduce(numpy.dot, (X0.T, x)) for x in g_SL])
        g_SS = numpy.asarray(
            [reduce(numpy.dot, (X0.T, x, X0)) for x in g_SS])
        # construct g
        g = g_SS + g_SL + g_LS + g_LL

        for x in g:
            print("Check Antisymmetry in fetch_X2C_soDKH1 get 2e %e" %
                  (Check_AntiSymmetry(x)))

    # Result and Return

    res = E_plus_1_sd + g

    if _sfX2C.with_x2c.xuncontract and contr_coeff is not None:
        res = numpy.asarray(
            [reduce(numpy.dot, (contr_coeff.T, x, contr_coeff)) for x in res])

    return res
