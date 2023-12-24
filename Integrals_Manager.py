import pyscf
from pyscf import tools
import numpy as np
from pyscf.lib import logger
from pyscf import lib
from pyscf import tools
from pyscf import ao2mo
from pyscf import mcscf, fci
import pyscf
from pyscf import tools
import os


def read(filename, verbose=True):
    '''Parse FCIDUMP.  Return a dictionary to hold the integrals and
    parameters with keys:  H1, H2, ECORE, NORB, NELEC, MS, ORBSYM, ISYM

    Kwargs:
        molpro_orbsym (bool): Whether the orbsym in the FCIDUMP file is in
            Molpro orbsym convention as documented in
            https://www.molpro.net/info/current/doc/manual/node36.html
            In return, orbsym is converted to pyscf symmetry convention
        verbose (bool): Whether to print debugging information
    '''
    if verbose:
        print('Parsing %s' % filename)
    finp = open(filename, 'r')

    data = []
    for i in range(10):
        line = finp.readline().upper()
        data.append(line)
        if '&END' in line:
            break
    else:
        raise RuntimeError('Problematic FCIDUMP header')

    result = {}
    tokens = ','.join(data).replace('&FCI', '').replace('&END', '')
    tokens = tokens.replace(' ', '').replace('\n', '').replace(',,', ',')
    for token in re.split(',(?=[a-zA-Z])', tokens):
        key, val = token.split('=')
        if key in ('NORB', 'NELEC', 'MS2', 'ISYM'):
            result[key] = int(val.replace(',', ''))
        elif key in ('ORBSYM',):
            result[key] = [int(x) for x in val.replace(',', ' ').split()]
        else:
            result[key] = val

    # Convert to Molpro orbsym convert_orbsym
    if 'ORBSYM' in result:
        if min(result['ORBSYM']) < 0:
            raise RuntimeError('Unknown orbsym convention')

    norb = result['NORB']
    n2c = norb * 2
    norb_pair = norb * (norb+1) // 2
    h1e = numpy.zeros((n2c, n2c), dtype=numpy.complex128)
    h2e = numpy.zeros((n2c, n2c, n2c, n2c), dtype=numpy.complex128)
    dat = finp.readline().split()
    while dat:
        i, j, k, l = [int(x) for x in dat[2:6]]
        if k != 0:
            h2e[i][j][k][l] = complex(float(dat[0]), float(dat[1]))
        elif k == 0:
            if j != 0:
                h1e[i-1, j-1] = float(dat[0])
            else:
                result['ECORE'] = float(dat[0])
        dat = finp.readline().split()

    idx, idy = numpy.tril_indices(norb, -1)
    if numpy.linalg.norm(h1e[idy, idx]) == 0:
        h1e[idy, idx] = h1e[idx, idy]
    elif numpy.linalg.norm(h1e[idx, idy]) == 0:
        h1e[idx, idy] = h1e[idy, idx]
    result['H1'] = h1e
    result['H2'] = h2e
    finp.close()
    return result


def dump_heff_casci(_mol, _mcscf, _core_coeff, _mocoeff, _filename='FCIDUMP'):
    loc1 = 0
    if _core_coeff is not None:
        loc1 = _core_coeff.shape[1]
    else:
        loc1 = 0
    norb = loc1 + _mocoeff.shape[1]
    nao = _mocoeff.shape[0]
    mocoeff = np.zeros((nao, norb))
    core_indx = list(range(0, loc1))
    act_indx = list(range(loc1, norb))
    mocoeff[:, core_indx] = _core_coeff
    mocoeff[:, act_indx] = _mocoeff
    int2e_full = pyscf.ao2mo.full(eri_or_mol=_mol, mo_coeff=mocoeff, aosym='1')
    int2e_full = pyscf.ao2mo.restore(1, int2e_full.copy(), mocoeff.shape[1])
    # Get integrals
    int2e_res = int2e_full[loc1:norb, loc1:norb, loc1:norb, loc1:norb]
    int2e_res = pyscf.ao2mo.restore(8, int2e_res.copy(), norb-loc1)
    int1e_res, energy_core = pyscf.mcscf.casci.h1e_for_cas(
        _mcscf, mo_coeff=mocoeff, ncas=_mocoeff.shape[1], ncore=loc1)
    # get orbsym
    OrbSym = pyscf.symm.label_orb_symm(_mol, _mol.irrep_name, _mol.symm_orb,
                                       _mocoeff)
    OrbSymID = [pyscf.symm.irrep_name2id(_mol.groupname, x) for x in OrbSym]
    # DUMP
    if _filename == None:
        return int1e_res, int2e_res, energy_core, _mocoeff.shape[1], _mol.nelectron - 2 * _core_coeff.shape[1], OrbSymID
    else:
        tools.fcidump.from_integrals(filename=_filename,
                                     h1e=int1e_res,
                                     h2e=int2e_res,
                                     nuc=energy_core,
                                     nmo=_mocoeff.shape[1],
                                     nelec=_mol.nelectron - 2 *
                                     _core_coeff.shape[1],  # Useless
                                     tol=1e-10,
                                     orbsym=OrbSymID)

# permutation symmetry


def _combine2(a, b):
    if a > b:
        return a*(a+1)//2 + b
    else:
        return b*(b+1)//2 + a


def _combine4(a, b, c, d):
    return _combine2(_combine2(a, b), _combine2(c, d))


def permutate_integrals(h1e, h2e, norb, map_old_2_new):
    h1e_new = np.zeros((norb, norb))
    h2e_new = np.zeros(len(h2e))

    # loop 1e

    for p in range(norb):
        for q in range(norb):
            h1e_new[map_old_2_new[p], map_old_2_new[q]] = h1e[p, q]

    # loop 2e

    indx = 0

    for p in range(norb):
        for q in range(p+1):
            for r in range(p+1):
                end_s = 0
                if p == r:
                    end_s = q+1
                else:
                    end_s = r+1
                for s in range(end_s):
                    indx_new = _combine4(
                        map_old_2_new[p], map_old_2_new[q], map_old_2_new[r], map_old_2_new[s])
                    h2e_new[indx_new] = h2e[indx]
                    indx += 1

    return h1e_new, h2e_new


def particle_2_hole(h1e, h2e, energy_core, norb):

    h1e_new = np.zeros((norb, norb))
    energy_core_new = energy_core

    for i in range(norb):
        for j in range(norb):
            if i == j:
                h1e_new[i, i] = - h1e[i, i] - h2e[_combine4(i, i, i, i)]
                for k in range(norb):
                    if k == i:
                        continue
                    h1e_new[i, i] += (-2 * h2e[_combine4(i, i, k, k)
                                               ] + h2e[_combine4(i, k, k, i)])
            else:
                h1e_new[j, i] = - h1e[i, j] - \
                    h2e[_combine4(i, i, i, j)] - h2e[_combine4(i, j, j, j)]
                for k in range(norb):
                    if k == i or k == j:
                        continue
                    h1e_new[j, i] += (h2e[_combine4(i, k, k, j)] -
                                      2*h2e[_combine4(i, j, k, k)])

    for i in range(norb):
        energy_core_new += 2*h1e[i, i] + h2e[_combine4(i, i, i, i)]

    for i in range(norb):
        for j in range(i+1, norb):
            energy_core_new += 4 * \
                h2e[_combine4(i, i, j, j)] - 2 * h2e[_combine4(i, j, j, i)]

    return h1e_new, energy_core_new


APP = os.getenv("ICI_CPP")

if __name__ == "__main__":

    from pyscf import gto, scf
    # import SOC_Driver
    from functools import reduce

    b = 1.24253

    # C2 6-31G as an example

    # Mol

    Mol = pyscf.gto.Mole()
    Mol.atom = [
        ['C', (0.000000,  0.000000, -b/2)],
        ['C', (0.000000,  0.000000,  b/2)], ]
    Mol.basis = 'sto-3G'
    Mol.symmetry = True
    Mol.spin = 0
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = 'angstorm'
    Mol.build()

    nelec = 12
    nval = 8
    norb = Mol.nao

    task = "0 0 1 1"

    # HF

    mf = scf.RHF(Mol).run()

    # Run iCI, CASCI

    nvir = norb - 10
    Segment = "%d %d %d %d %d %d" % (2, 0, 4, 4, 0, nvir)

    # FCIDUMP

    DumpFileName = "FCIDUMP"
    tools.fcidump.from_scf(mf, DumpFileName, 1e-10)

    exit(1)

    # inputfile

    File = "C2_HF_ORB_CASCI" + ".inp"
    Out = "C2_HF_ORB_CASCI" + ".out"
    SOC_Driver._Generate_InputFile_SiCI(File,
                                        Segment=Segment,
                                        nelec_val=nval,
                                        rotatemo=0,  # HF orbitals!
                                        cmin=0.0,
                                        perturbation=0,
                                        dumprdm=0,
                                        relative=0,
                                        Task=task,
                                        inputocfg=0,
                                        tol=1e-8,
                                        selection=0
                                        )

    # Run

    os.system("%s %s > %s" % (APP, File, Out))

    os.system("cp FCIDUMP FCIDUMP_BENCH")

    # Run iCI Orb_Perm

    permute = [0, 1, 9, 8, 7, 6, 5, 4, 3, 2]
    for id in range(norb-1, 9, -1):
        permute.append(id)
    print(permute)

    mo_coeff = mf.mo_coeff

    mo_coeff_new = np.zeros((Mol.nao, Mol.nao))

    for i in range(norb):
        mo_coeff_new[:, i] = mo_coeff[:, permute[i]]

    OrbSym = pyscf.symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb,
                                       mo_coeff_new)
    OrbSymID = [pyscf.symm.irrep_name2id(Mol.groupname, x) for x in OrbSym]
    orb_sym = OrbSymID
    tools.fcidump.from_mo(
        Mol, "FCIDUMP", mo_coeff=mo_coeff_new, orbsym=orb_sym, tol=1e-10)

    File = "C2_HF_ORB_CASCI_PERM_BENCH" + ".inp"
    Out = "C2_HF_ORB_CASCI_PERM_BENCH" + ".out"
    SOC_Driver._Generate_InputFile_SiCI(File,
                                        Segment=Segment,
                                        nelec_val=nval,
                                        rotatemo=0,  # HF orbitals!
                                        cmin=0.0,
                                        perturbation=0,
                                        dumprdm=0,
                                        relative=0,
                                        Task=task,
                                        inputocfg=0,
                                        tol=1e-8,
                                        selection=0
                                        )

    # RUN

    os.system("%s %s > %s" % (APP, File, Out))

    # TEST _PERMUTATION

    h1e = reduce(np.dot, (mo_coeff.T, mf.get_hcore(), mo_coeff))
    h2e = ao2mo.full(Mol, mo_coeff, verbose=0)
    # print(h2e.shape)
    h2e = pyscf.ao2mo.restore(8, h2e, norb)
    energy_core = mf.energy_nuc()

    h1e_new, h2e_new = permutate_integrals(h1e, h2e, norb, permute)

    tools.fcidump.from_integrals(
        "FCIDUMP", h1e_new, h2e_new, norb, mf.mol.nelec, energy_core, 0, orb_sym, 1e-10)

    File = "C2_HF_ORB_CASCI_PERM" + ".inp"
    Out = "C2_HF_ORB_CASCI_PERM" + ".out"
    SOC_Driver._Generate_InputFile_SiCI(File,
                                        Segment=Segment,
                                        nelec_val=nval,
                                        rotatemo=0,  # HF orbitals!
                                        cmin=0.0,
                                        perturbation=0,
                                        dumprdm=0,
                                        relative=0,
                                        Task=task,
                                        inputocfg=0,
                                        tol=1e-8,
                                        selection=0
                                        )

    os.system("%s %s > %s" % (APP, File, Out))

    # permute and particle --> hole

    OrbSym = pyscf.symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb,
                                       mo_coeff)
    OrbSymID = [pyscf.symm.irrep_name2id(Mol.groupname, x) for x in OrbSym]
    orb_sym = OrbSymID

    permute = []
    orb_sym_new = []

    for i in range(norb):
        permute.append(norb - 1 - i)
        orb_sym_new.append(orb_sym[norb - 1 - i])

    h1e_new, h2e_new = permutate_integrals(h1e, h2e, norb, permute)

    h1e_hole, energy_core_hole = particle_2_hole(
        h1e_new, h2e_new, energy_core, norb)

    tools.fcidump.from_integrals(
        "FCIDUMP", h1e_hole, h2e_new, norb, mf.mol.nelec, energy_core_hole, 0, orb_sym_new, 1e-10)

    # Run iCI

    Segment = "%d %d %d %d %d %d" % (nvir, 0, 4, 4, 0, 2)

    File = "C2_HF_ORB_CASCI_HOLE" + ".inp"
    Out = "C2_HF_ORB_CASCI_HOLE" + ".out"
    SOC_Driver._Generate_InputFile_SiCI(File,
                                        Segment=Segment,
                                        nelec_val=nval,
                                        rotatemo=0,  # HF orbitals!
                                        cmin=0.0,
                                        perturbation=0,
                                        dumprdm=0,
                                        relative=0,
                                        Task=task,
                                        inputocfg=0,
                                        tol=1e-8,
                                        selection=0
                                        )

    os.system("%s %s > %s" % (APP, File, Out))

    # Final test on the core term

    Mol = pyscf.gto.Mole()
    Mol.atom = [
        ['C', (0.000000,  0.000000, -b/2)],
        ['C', (0.000000,  0.000000,  b/2)], ]
    Mol.basis = 'sto-3G'
    Mol.symmetry = True
    Mol.spin = 0
    Mol.charge = -8
    Mol.verbose = 4
    Mol.unit = 'angstorm'
    Mol.build()

    mf = scf.RHF(Mol).run()
