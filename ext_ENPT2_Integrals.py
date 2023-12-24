import pyscf
import numpy
from functools import reduce
from pyscf import tools
import Util_Orb
import tempfile
import h5py


def _combine2(a, b):  # 8-fold 0-based
    if a > b:
        return a*(a+1)//2 + b
    else:
        return b*(b+1)//2 + a


def _combine4(a, b, c, d):  # 8-fold 0-based
    return _combine2(_combine2(a, b), _combine2(c, d))


def dump_FCIDUMP_extPT_outcore(mol, scf, mo_coeff, nfzc, nact, nvir, filename="FCIDUMP", tol = 1e-8):
    nmo = nfzc+nact+nvir
    assert (nmo <= mol.nao)
    # nmo = my_scf.mo_coeff.shape[1]
    nelec = mol.nelectron
    ms = 0
    # tol = 1e-10
    nuc = mol.get_enuc()
    float_format = tools.fcidump.DEFAULT_FLOAT_FORMAT

    h1e = reduce(numpy.dot, (mo_coeff.T,
                 scf.get_hcore(), mo_coeff))
    h1e = h1e[:nmo, :nmo]

    ftmp = tempfile.NamedTemporaryFile()
    print('MO integrals are saved in file  %s  under dataset "eri_mo"' % ftmp.name)
    pyscf.ao2mo.kernel(mol, mo_coeff[:, :nmo], ftmp.name)

    OrbSym = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                       mo_coeff[:, :nmo])
    OrbSymID = [pyscf.symm.irrep_name2id(mol.groupname, x) for x in OrbSym]

    # dump

    with open(filename, 'w') as fout:  # 8-fold symmetry
        tools.fcidump.write_head(fout, nmo, nelec, ms, OrbSymID)
        output_format = float_format + ' %4d %4d %4d %4d\n'

        # eri

        with h5py.File(ftmp.name) as eri_file:
            for p in range(nmo):

                ncore_p = 0
                nvirt_p = 0
                if p < nfzc:
                    ncore_p += 1
                if p >= (nfzc+nact):
                    nvirt_p += 1

                for q in range(p+1):

                    ncore_q = ncore_p
                    nvirt_q = nvirt_p
                    if q < nfzc:
                        ncore_q += 1
                    if q >= (nfzc+nact):
                        nvirt_q += 1

                    eri_pq = eri_file['eri_mo'][_combine2(p, q)]

                    # print("shape %d %d " % (p, q), eri_pq.shape)

                    for r in range(p+1):

                        ncore_r = ncore_q
                        nvirt_r = nvirt_q
                        if r < nfzc:
                            ncore_r += 1
                        if r >= (nfzc+nact):
                            nvirt_r += 1

                        if (p == q) or (p == r) or (q == r):
                            if p > r:
                                for s in range(r+1):
                                    indx = _combine2(r, s)
                                    if abs(eri_pq[indx]) > tol:
                                        fout.write(output_format % (
                                            eri_pq[indx], p+1, q+1, r+1, s+1))
                            else:
                                for s in range(q+1):
                                    indx = _combine2(r, s)
                                    if abs(eri_pq[indx]) > tol:
                                        fout.write(output_format % (
                                            eri_pq[indx], p+1, q+1, r+1, s+1))
                        else:

                            if (ncore_r > 2) or (nvirt_r > 2):
                                continue

                            if p > r:
                                for s in range(r+1):
                                    ncore_s = ncore_r
                                    nvirt_s = nvirt_r
                                    if s < nfzc:
                                        ncore_s += 1
                                    if s >= (nfzc+nact):
                                        nvirt_s += 1
                                    indx = _combine2(r, s)
                                    if ((p == s) or (q == s) or (r == s)):
                                        if abs(eri_pq[indx]) > tol:
                                            fout.write(output_format % (
                                                eri_pq[indx], p+1, q+1, r+1, s+1))
                                    else:
                                        if (ncore_s <= 2) and (nvirt_s <= 2):
                                            if abs(eri_pq[indx]) > tol:
                                                fout.write(output_format % (
                                                    eri_pq[indx], p+1, q+1, r+1, s+1))
                                        else:
                                            if (nvirt_s > 2):
                                                break
                            else:
                                for s in range(q+1):
                                    ncore_s = ncore_r
                                    nvirt_s = nvirt_r
                                    if s < nfzc:
                                        ncore_s += 1
                                    if s >= (nfzc+nact):
                                        nvirt_s += 1
                                    indx = _combine2(r, s)
                                    if ((p == s) or (q == s) or (r == s)):
                                        if abs(eri_pq[indx]) > tol:
                                            fout.write(output_format % (
                                                eri_pq[indx], p+1, q+1, r+1, s+1))
                                    else:
                                        if (ncore_s <= 2) and (nvirt_s <= 2):
                                            if abs(eri_pq[indx]) > tol:
                                                fout.write(output_format % (
                                                    eri_pq[indx], p+1, q+1, r+1, s+1))
                                        else:
                                            if (nvirt_s > 2):
                                                break

        # h1e and nuc

        tools.fcidump.write_hcore(
            fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)


def dump_FCIDUMP_extPT(mol, scf, mo_coeff, nfzc, nact, nvir, filename="FCIDUMP", tol = 1e-8):
    nmo = nfzc+nact+nvir
    assert (nmo <= mol.nao)
    # nmo = my_scf.mo_coeff.shape[1]
    nelec = mol.nelectron
    ms = 0
    # tol = 1e-10
    nuc = mol.get_enuc()
    float_format = tools.fcidump.DEFAULT_FLOAT_FORMAT

    h1e = reduce(numpy.dot, (mo_coeff.T,
                 scf.get_hcore(), mo_coeff))
    h1e = h1e[:nmo, :nmo]

    # print(h1e)

    int2e_full = pyscf.ao2mo.full(
        eri_or_mol=mol, mo_coeff=mo_coeff[:, :nmo], aosym='4')
    int2e_full = pyscf.ao2mo.restore(8, int2e_full.copy(), nmo)

    OrbSym = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                       mo_coeff[:, :nmo])
    OrbSymID = [pyscf.symm.irrep_name2id(mol.groupname, x) for x in OrbSym]

    with open(filename, 'w') as fout:  # 8-fold symmetry
        tools.fcidump.write_head(fout, nmo, nelec, ms, OrbSymID)
        output_format = float_format + ' %4d %4d %4d %4d\n'
        for p in range(nmo):

            ncore_p = 0
            nvirt_p = 0
            if p < nfzc:
                ncore_p += 1
            if p >= (nfzc+nact):
                nvirt_p += 1

            for q in range(p+1):

                ncore_q = ncore_p
                nvirt_q = nvirt_p
                if q < nfzc:
                    ncore_q += 1
                if q >= (nfzc+nact):
                    nvirt_q += 1

                for r in range(p+1):

                    ncore_r = ncore_q
                    nvirt_r = nvirt_q
                    if r < nfzc:
                        ncore_r += 1
                    if r >= (nfzc+nact):
                        nvirt_r += 1

                    if (p == q) or (p == r) or (q == r):
                        if p > r:
                            for s in range(r+1):
                                indx = _combine4(p, q, r, s)
                                if abs(int2e_full[indx]) > tol:
                                    fout.write(output_format % (
                                        int2e_full[indx], p+1, q+1, r+1, s+1))
                        else:
                            for s in range(q+1):
                                indx = _combine4(p, q, r, s)
                                if abs(int2e_full[indx]) > tol:
                                    fout.write(output_format % (
                                        int2e_full[indx], p+1, q+1, r+1, s+1))
                    else:

                        if (ncore_r > 2) or (nvirt_r > 2):
                            continue

                        if p > r:
                            for s in range(r+1):
                                ncore_s = ncore_r
                                nvirt_s = nvirt_r
                                if s < nfzc:
                                    ncore_s += 1
                                if s >= (nfzc+nact):
                                    nvirt_s += 1
                                indx = _combine4(p, q, r, s)
                                if ((p == s) or (q == s) or (r == s)):
                                    if abs(int2e_full[indx]) > tol:
                                        fout.write(output_format % (
                                            int2e_full[indx], p+1, q+1, r+1, s+1))
                                else:
                                    if (ncore_s <= 2) and (nvirt_s <= 2):
                                        if abs(int2e_full[indx]) > tol:
                                            fout.write(output_format % (
                                                int2e_full[indx], p+1, q+1, r+1, s+1))
                                    else:
                                        if (nvirt_s > 2):
                                            break
                        else:
                            for s in range(q+1):
                                ncore_s = ncore_r
                                nvirt_s = nvirt_r
                                if s < nfzc:
                                    ncore_s += 1
                                if s >= (nfzc+nact):
                                    nvirt_s += 1
                                indx = _combine4(p, q, r, s)
                                if ((p == s) or (q == s) or (r == s)):
                                    if abs(int2e_full[indx]) > tol:
                                        fout.write(output_format % (
                                            int2e_full[indx], p+1, q+1, r+1, s+1))
                                else:
                                    if (ncore_s <= 2) and (nvirt_s <= 2):
                                        if abs(int2e_full[indx]) > tol:
                                            fout.write(output_format % (
                                                int2e_full[indx], p+1, q+1, r+1, s+1))
                                    else:
                                        if (nvirt_s > 2):
                                            break

        tools.fcidump.write_hcore(
            fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)

# Fock Operator


def construct_Fock_Operator(mol, scf, mo_coeff, nfzc, nact, nvir, rdm1):

    nmo = nfzc+nact+nvir
    assert (nmo <= mol.nao)

    h1e = reduce(numpy.dot, (mo_coeff.T,
                 scf.get_hcore(), mo_coeff))
    FOCK = h1e[:nmo, :nmo]

    # ftmp = tempfile.NamedTemporaryFile()
    # print('MO integrals are saved in file  %s  under dataset "eri_mo"' % ftmp.name)
    # pyscf.ao2mo.kernel(mol, mo_coeff[:, :nmo], ftmp.name)

    int2e_full = pyscf.ao2mo.full(
        eri_or_mol=mol, mo_coeff=mo_coeff[:, :nmo], aosym='1')
    # int2e_full = pyscf.ao2mo.restore(8, int2e_full.copy(), nmo)
    int2e_full = int2e_full.reshape(nmo, nmo, nmo, nmo)

    # 转动积分

    OrbSym = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                       mo_coeff[:, :nmo])
    OrbSymID = [pyscf.symm.irrep_name2id(mol.groupname, x) for x in OrbSym]

    print(int2e_full.shape)
    print(rdm1.shape)
    FOCK += numpy.einsum("pquv,uv->pq",
                         int2e_full, rdm1)
    FOCK -= 0.5 * numpy.einsum(
        "puvq,uv->pq", int2e_full, rdm1)

    return FOCK

    with h5py.File(ftmp.name) as eri_file:
        for p in range(nmo):
            for q in range(p+1):
                eri_pq = eri_file['eri_mo'][_combine2(p, q)]

                for u in range(nfzc+nact):
                    for v in range(nfzc+nact):
                        FOCK[p, q] += eri_pq[_combine2(u, v)] * rdm1[u, v]

                u = q
                for q_prime in range(nmo):
                    for v in range(nfzc+nact):
                        FOCK[p,
                             q_prime] -= eri_pq[_combine2(v, q_prime)] * rdm1[u, v]

        for p in range(nmo):
            for q in range(p+1, nmo):
                FOCK[p, q] = FOCK[q, p]

    return FOCK


def BDF_molden_2_extPT_FCIDUMP(mol, scf, molden_file, NFZC, NACT, NVIR, fcidump_file):
    _, mo_energy, mo_coeff, mo_occ, _, _ = tools.molden.load(
        molden_file)

    order, mo_coeff_new, nfzc, nact, nvir = Util_Orb.reorder_BDF_orb(
        mol, mo_coeff, mo_energy, mo_occ, NFZC, NACT, NVIR)

    if (nfzc+nact+nvir) <= 128:
        dump_FCIDUMP_extPT(mol, scf, mo_coeff_new, nfzc,
                           nact, nvir, fcidump_file)
    else:
        dump_FCIDUMP_extPT_outcore(
            mol, scf, mo_coeff_new, nfzc, nact, nvir, fcidump_file)


if __name__ == "__main__":

    from pyscf import scf
    from functools import reduce

    b = 1.24253

    # C2 6-31G as an example

    # Mol

    Mol = pyscf.gto.Mole()
    Mol.atom = [
        ['C', (0.000000,  0.000000, -b/2)],
        ['C', (0.000000,  0.000000,  b/2)], ]
    Mol.basis = 'cc-pvtz'
    Mol.symmetry = 'd2h'
    Mol.spin = 0
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = 'angstorm'
    Mol.build()

    # HF

    mf = scf.RHF(Mol).run()

    # FCIDUMP

    DumpFileName = "FCIDUMP_FULL"
    tools.fcidump.from_scf(mf, DumpFileName, 1e-10)

    # only ext

    dump_FCIDUMP_extPT(Mol, mf, mf.mo_coeff, 6,
                       8, Mol.nao-14, filename="FCIDUMP_EXTPT")

    dump_FCIDUMP_extPT_outcore(Mol, mf, mf.mo_coeff, 6,
                               8, Mol.nao-14, filename="FCIDUMP_EXTPT_OUTCORE")
