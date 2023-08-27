from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc import tools
import numpy

from pyscf import tools as origin_tools

def get_iCI_irrep_id_and_kmesh(cell, kpts, dimension):
    assert (dimension in [1, 2, 3])
    kmesh = tools.k2gamma.kpts_to_kmesh(cell, kpts)[:dimension]
    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    scaled_k = [x[:dimension] for x in scaled_k]

    # get irrep
    irrep_id_iCI = []

    if dimension == 1:
        nkx = kmesh[0]
        for scaled_kpt in scaled_k:
            px = int(nkx*scaled_kpt[0])
            while px < 0:
                px += nkx
            irrep_id_iCI.append(px)

    if dimension == 2:
        nkx = kmesh[0]
        nky = kmesh[1]
        for scaled_kpt in scaled_k:
            px = int(nkx*scaled_kpt[0])
            py = int(nky*scaled_kpt[1])
            while px < 0:
                px += nkx
            while py < 0:
                py += nky
            irrep_id_iCI.append(px * nky + py)

    if dimension == 3:
        nkx = kmesh[0]
        nky = kmesh[1]
        nkz = kmesh[2]
        for scaled_kpt in scaled_k:
            px = int(nkx*scaled_kpt[0])
            py = int(nky*scaled_kpt[1])
            pz = int(nkz*scaled_kpt[2])
            while px < 0:
                px += nkx
            while py < 0:
                py += nky
            while pz < 0:
                pz += nkz
            irrep_id_iCI.append(px * nky * nkz + py * nkz + pz)

    return kmesh, scaled_k, irrep_id_iCI


def get_orbmap_pyscf_2_iCI(mf):
    nkpts = len(mf.mo_coeff)
    nmo = mf.mo_coeff[0].shape[1]
    mo_energy_tmp = []

    for k in range(nkpts):
        for iorb in range(nmo):
            mo_energy_tmp.append([k, iorb, mf.mo_energy[k][iorb]])

    # print(mo_energy_tmp)

    def get_elmt(input):
        return input[-1]

    mo_energy_tmp.sort(key=get_elmt)
    for data in mo_energy_tmp:
        print(data)

    pyscf_to_iCI = numpy.zeros((nkpts, nmo), dtype=numpy.int)

    for id, data in enumerate(mo_energy_tmp):
        pyscf_to_iCI[data[0]][data[1]] = id

    print(pyscf_to_iCI)

    return pyscf_to_iCI

# dump SOLID FCIDUMP


def _write_head(fout, nmo, nelec, ms, orbsym, kpts, dimension):
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    if dimension == 1:
        fout.write('  IRREPSIZE=%d\n' % (kpts[0]))
    if dimension == 2:
        fout.write('  IRREPSIZE=%d,%d\n' % (kpts[0], kpts[1]))
    if dimension == 3:
        fout.write('  IRREPSIZE=%d,%d,%d\n' % (kpts[0], kpts[1], kpts[2]))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')

from functools import reduce

def _write_hcore(fout, mf, pyscf_2_iCI, tol=1e-10, output_format="%.15e %.15e %5d %5d %5d %5d\n"):
    hcore = mf.get_hcore()  # include nuc!
    nkpts = hcore.shape[0]
    nmo = hcore.shape[1]

    mo_coeff = mf.mo_coeff
    h1e_dump = numpy.zeros(hcore.shape, dtype=hcore.dtype)
    for kpts in range(nkpts):
        h1e_dump[kpts] = reduce(numpy.dot, (mo_coeff[kpts].conj().T,hcore[kpts],mo_coeff[kpts]))


    for kpts in range(nkpts):
        for i in range(nmo):
            for j in range(0, i+1):
                if abs(h1e_dump[kpts][i, j]) > tol:
                    # print(hcore[kpts][i, j])
                    # print(hcore[kpts][i, j].real)
                    # print(hcore[kpts][i, j].imag)
                    fout.write(output_format % (
                        h1e_dump[kpts][i, j].real, h1e_dump[kpts][i, j].imag, pyscf_2_iCI[kpts][i]+1, pyscf_2_iCI[kpts][j]+1, 0, 0))


def _write_eri(fout, cell, mf, kpts, nmo, pyscf_2_iCI, tol=1e-10, output_format="%.15e %.15e %5d %5d %5d %5d\n"):
    kconserv = tools.get_kconserv(cell, kpts)
    nkpts = len(kpts)

    print(len(mf.mo_coeff))

    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp, kq, kr]
                print(kp,kq,kr,ks)
                print(mf.mo_coeff[kp].shape)
                eri_kpt = mf.with_df.ao2mo([mf.mo_coeff[i] for i in (kp, kq, kr, ks)],
                                           [kpts[i] for i in (kp, kq, kr, ks)])
                eri_kpt = eri_kpt.reshape([nmo]*4)
                for p in range(nmo):
                    id_p = pyscf_2_iCI[kp][p]
                    for q in range(nmo):
                        id_q = pyscf_2_iCI[kq][q]
                        for r in range(nmo):
                            id_r = pyscf_2_iCI[kr][r]
                            for s in range(nmo):
                                id_s = pyscf_2_iCI[ks][s]
                                if abs(eri_kpt[p][q][r][s]) > tol:
                                    fout.write(output_format % (
                                        eri_kpt[p][q][r][s].real / 4, eri_kpt[p][q][r][s].imag / 4,
                                        id_p+1, id_q+1, id_r+1, id_s+1))

from pyscf import ao2mo

def dump_supercell_gamma(mf, filename="FCIDUMP"):
    hcore = mf.get_hcore()
    h1e = reduce(numpy.dot,(mf.mo_coeff.T, hcore, mf.mo_coeff))

    eri = mf.with_df.ao2mo(mf.mo_coeff)
    eri = ao2mo.restore(8, eri, mf.mo_coeff.shape[1])
    
    # ao_eri = mf.with_df.get_ao_eri(mf.kpt,compact=True)
    # ao_eri = ao2mo.restore(1, ao_eri, mf.mo_coeff.shape[1])
    # mo_eri = numpy.einsum("ijab,ip->pjab", ao_eri, mf.mo_coeff)
    # mo_eri = numpy.einsum("pjab,jq->pqab", mo_eri, mf.mo_coeff)
    # mo_eri = numpy.einsum("pqab,ar->pqrb", mo_eri, mf.mo_coeff)
    # mo_eri = numpy.einsum("pqrb,bs->pqrs", mo_eri, mf.mo_coeff)
    # eri = ao2mo.restore(8, mo_eri, mf.mo_coeff.shape[1])

    super_cell = mf.cell
    madelung = tools.pbc.madelung(super_cell, [mf.kpt]) * super_cell.nelectron * -0.5
    print(madelung)
    nuc = mf.energy_nuc()
    orbsym = numpy.zeros((super_cell.nao), dtype=numpy.int)
    origin_tools.fcidump.from_integrals(filename, h1e, eri, super_cell.nao, super_cell.nelectron, nuc + madelung, 0, orbsym, tol=1e-10)

def dump(cell, mf, kpts, dimension, filename="FCIDUMP"):

    # 1 get basic info

    kpts = cell.make_kpts(kpts)
    nmo = mf.mo_coeff[0].shape[1]
    nkpts = len(kpts)
    kmesh, scaled_k, irrep_id_kpts = get_iCI_irrep_id_and_kmesh(
        cell, kpts, dimension)
    print(irrep_id_kpts)
    for data in scaled_k:
        print(data)
    pyscf_to_iCI = get_orbmap_pyscf_2_iCI(mf)

    irrep_id = numpy.zeros((nmo * nkpts), numpy.int)

    # for id, data in enumerate(irrep_id_kpts):
    #     for orb_id in data:
    #         irrep_id[orb_id] = irrep_id_kpts[id]

    for kpt_id in range(nkpts):
        for orb_id in range(nmo):
            irrep_id[pyscf_to_iCI[kpt_id][orb_id]] = irrep_id_kpts[kpt_id]

    # 2. dump

    nelectron = cell.nelectron
    with open(filename, 'w') as fout:
        # dump head
        _write_head(fout, nmo * nkpts, nelectron * nkpts,
                    0, irrep_id, kmesh, dimension)
        # dump int1e
        _write_hcore(fout, mf, pyscf_to_iCI)
        # dump eri
        _write_eri(fout, cell, mf, kpts, nmo, pyscf_to_iCI)
        # nuc + madelung 
        nelectron = float(mf.cell.tot_electrons(nkpts)) 
        madelung = tools.pbc.madelung(mf.cell, [mf.kpts]) * -0.5
        print(madelung, cell.nelectron)
        nuc = mf.energy_nuc()
        print(nuc)
        fout.write("%.15e %.15e %5d %5d %5d %5d\n" % ((nuc + madelung * cell.nelectron) * nk * nk, 0.0, 0, 0, 0, 0))

# test_case


if __name__ == "__main__":

    # graphene

    nk = 2
    kpts = [nk, nk, 1]
    Lz = 25  # Smallest Lz value for ~1e-6 convergence in absolute energy
    a = 1.42  # bond length in graphene
    fft_ke_cut = 300
    # Much smaller mesh needed for AFTDF with the setting cell.low_dim_ft_type='inf_vacuum'
    aft_mesh = [30, 30, 40]
    e = []
    t = []
    pseudo = 'gth-pade'

    cell = pbcgto.Cell()
    cell.build(unit='B',
               a=[[4.6298286730500005, 0.0, 0.0], [-2.3149143365249993,
                                                   4.009549246030899, 0.0], [0.0, 0.0, Lz]],
               atom='C 0 0 0; C 0 2.67303283 0',
               ke_cutoff=fft_ke_cut,
               dimension=2,
               pseudo=pseudo,
               verbose=7,
               precision=1e-6,
               basis='gth-szv')
    # mf = pbchf.KRHF(cell, exxdiv='ewald')
    mf = pbchf.KRHF(cell.copy())
    # mf.with_df = pdf.FFTDF(cell)
    mf.kpts = cell.make_kpts(kpts)
    mf.conv_tol = 1e-6
    mf.kernel()

    dump(cell, mf, kpts, 2, "FCIDUMP_graphene")

    # supercell 

    super_cell = tools.super_cell(cell, [nk,nk,1])
    mf = pbchf.RHF(super_cell.copy())
    ehf = mf.kernel()

    dump_supercell_gamma(mf, "FCIDUMP_graphene_supercell_gamma")
