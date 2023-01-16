import pyscf
import Driver_SCF

# WARNNING: USELESS

Atom_Info = {
    'C': {
        "charge_fake": 2,
        "spin": 2,
        "state": [[0, 'b1g', 1], [0, 'b2g', 1], [0, 'b3g', 1]],
        "nelec": (2, 0)
    },
    'Si': {
        "charge_fake": 2,
        "spin": 2,
        "state": [[0, 'b1g', 1], [0, 'b2g', 1], [0, 'b3g', 1]],
        "nelec": (2, 0)
    },
    'N': {
        "charge_fake": 3,
        "spin": 3,
        "state": [[1, 'au', 1]],
        "nelec": (3, 0)
    },
    'P': {
        "charge_fake": 3,
        "spin": 3,
        "state": [[1, 'au', 1]],
        "nelec": (3, 0)
    },
    'O': {
        "charge_fake": 4,
        "spin": 2,
        "state": [[0, 'b1g', 1], [0, 'b2g', 1], [0, 'b3g', 1]],
        "nelec": (3, 1)
    },
    'S': {
        "charge_fake": 4,
        "spin": 2,
        "state": [[0, 'b1g', 1], [0, 'b2g', 1], [0, 'b3g', 1]],
        "nelec": (3, 1)
    },
    'F': {
        "charge_fake": -1,
        "spin": 1,
        "state": [[1, 'b1u', 1], [1, 'b2u', 1], [1, 'b3u', 1]],
        "nelec": (3, 2)
    },
    'Cl': {
        "charge_fake": -1,
        "spin": 1,
        "state": [[1, 'b1u', 1], [1, 'b2u', 1], [1, 'b3u', 1]],
        "nelec": (3, 2)
    },
}


def _remove_orb_other_than_sp(energy, start_indx):
    energy_now = 1e10
    multi = 0
    orb_indx = []
    res = []
    for indx in range(start_indx, len(energy)):
        if abs(energy[indx]-energy_now) > 1e-8:
            if multi >= 5:
                res.extend(orb_indx)
            energy_now = energy[indx]
            multi = 1
            orb_indx = [indx]
        else:
            orb_indx.append(indx)
            multi += 1

    if multi >= 5:
        res.extend(orb_indx)

    return res


def _label_atom_orb(energy):

    orb_info = [["s", 0], ["p", 1], ["d", 2], ["f", 3], ["g", 4], ["h", 5]]

    energy_now = 1e10
    multi = 0
    res = {}
    orb_indx = []
    for indx in range(0, len(energy)):
        if abs(energy[indx]-energy_now) > 1e-8 and  multi != 0:
            l = (multi-1)//2
            orb_info[l][1] += 1
            res[str(orb_info[l][1])+orb_info[l][0]] = orb_indx
            # print(res)
            energy_now = energy[indx]
            multi = 1
            orb_indx = [indx]
        else:
            orb_indx.append(indx)
            multi += 1

    l = (multi-1)//2
    orb_info[l][1] += 1
    res[str(orb_info[l][1])+orb_info[l][0]] = orb_indx

    return res


def _atom_min_cas(atom_label, basis='6-31G(d)', print_verbose=0):
    if atom_label == "H":
        mol = pyscf.gto.M(
            verbose=print_verbose,
            atom='''
H   0.000000000000       0.000000000000      0.000000000000  
''',
            basis=basis,
            spin=1,
            charge=0,
            symmetry='d2h',
        )
        mol.build()
        scf = pyscf.scf.ROHF(mol)
        scf.kernel()
        return [scf.mo_energy, scf.mo_coeff, _label_atom_orb(scf.mo_energy)]

    else:
        mol = pyscf.gto.M(
            verbose=print_verbose,
            atom='''
%s   0.000000000000       0.000000000000      0.000000000000  
''' % (atom_label),
            basis=basis,
            spin=Atom_Info[atom_label]["spin"],
            charge=0,
            symmetry='d2h',
        )
        mol.build()
        mol_fake = pyscf.gto.M(
            verbose=print_verbose,
            atom='''
%s   0.000000000000       0.000000000000      0.000000000000  
''' % (atom_label),
            basis=basis,
            spin=0,
            charge=Atom_Info[atom_label]["charge_fake"],
            symmetry='d2h',
        )
        mol_fake.build()

        scf_fake = Driver_SCF.Run_SCF(mol_fake)
        scf = pyscf.scf.ROHF(mol)
        scf.mo_coeff = scf_fake.mo_coeff

        if print_verbose >= 10:
            print(scf.mo_coeff[:, -2])

        frozen = _remove_orb_other_than_sp(scf_fake.mo_energy, 5)

        if (len(frozen) == 0):
            frozen = None

        min_cas = Driver_SCF.Run_MCSCF(mol, scf, Atom_Info[atom_label]["nelec"], 3,
                                       _pyscf_state=Atom_Info[atom_label]["state"], _frozen=frozen)

        if print_verbose >= 10:
            print(min_cas.mo_coeff[:, -2])

        scf.mo_coeff = min_cas.mo_coeff
        min_cas_ = Driver_SCF.Run_MCSCF(mol, scf, Atom_Info[atom_label]["nelec"], 3,  _mc_max_macro=0,
                                        _pyscf_state=Atom_Info[atom_label]["state"])

        if print_verbose >= 10:
            print(min_cas_.mo_coeff[:, -2])

        return [min_cas_.mo_energy, min_cas_.mo_coeff, _label_atom_orb(min_cas_.mo_energy)]


def atom_min_cas_bas(atom_label_list, basis='6-31G(d)', print_verbose=0):
    res = {}
    res['basis'] = basis
    for atom in atom_label_list:
        a, b, c = _atom_min_cas(atom, basis, print_verbose)
        res[atom] = b
        res[atom+"_bas_label"] = c 
        if print_verbose >= 10:
            print(a)
            print(c)
    return res


if __name__ == "__main__":
    # print(_atom_min_cas("C")[1][:,2:5])
    # print(_atom_min_cas("H")[1][:,2:5])
    # print(_atom_min_cas("O")[1][:,2:5])
    # print(_atom_min_cas("N")[1][:,2:5])

    a, b = _atom_min_cas("C", basis='ccpvqz', print_verbose=0)
    print(a, b[:, :5])
    a, b = _atom_min_cas("H", basis='ccpvqz', print_verbose=0)
    print(a, b[:, :5])
    a, b = _atom_min_cas("O", basis='ccpvqz', print_verbose=0)
    print(a, b[:, :5])
    a, b = _atom_min_cas("N", basis='ccpvqz', print_verbose=0)
    print(a, b[:, :5])
