import pyscf
from pyscf import lo
import Driver_SCF
import get_atom_orb
import Util_Mole
# import Util_Pic
from Util_Pic import draw_heatmap
from functools import reduce
import numpy


def generate_atom_basis(mol, atom_min_cas_bas):
    loc = [0]
    res = numpy.zeros((mol.nao, mol.nao))
    nocc = 0
    nval = 0
    for i in range(mol.natm):
        atom_min_cas_bas_ = atom_min_cas_bas[mol.atom_pure_symbol(i)]
        loc_begin = loc[-1]
        loc_end = loc_begin + atom_min_cas_bas_.shape[0]
        loc.append(loc_end)
        res[loc_begin:loc_end, loc_begin:loc_end] = atom_min_cas_bas_
        if mol.atom_pure_symbol(i) != "H":
            nocc += 5
            nval += 4
        else:
            nocc += 1
            nval += 1
    # print(nocc)
    loc2 = [0]
    loc3 = [0]
    res2 = numpy.zeros((mol.nao, nocc))
    res3 = numpy.zeros((mol.nao, nval))
    loc_now = 0
    loc_now_3 = 0
    for i in range(mol.natm):
        if mol.atom_pure_symbol(i) != "H":
            res2[loc[i]:loc[i+1], loc_now:loc_now +
                 5] = atom_min_cas_bas[mol.atom_pure_symbol(i)][:, :5]
            res3[loc[i]:loc[i+1], loc_now_3:loc_now_3 +
                 4] = atom_min_cas_bas[mol.atom_pure_symbol(i)][:, 1:5]
            loc_now += 5
            loc_now_3 += 4
            loc2.append(loc_now)
            loc3.append(loc_now_3)
        else:
            res2[loc[i]:loc[i+1], loc_now:loc_now +
                 1] = atom_min_cas_bas[mol.atom_pure_symbol(i)][:, :1]
            res3[loc[i]:loc[i+1], loc_now_3:loc_now_3 +
                 1] = atom_min_cas_bas[mol.atom_pure_symbol(i)][:, :1]
            loc_now += 1
            loc_now_3 += 1
            loc2.append(loc_now)
            loc3.append(loc_now_3)
    return loc, res, loc2, res2, loc3, res3


def analysis_mole_occ_orb(xyz, atom_bas, basis="6-31G(d)", charge=0, spin=0, verbose=0, latex=False, print_data=False):

    mol = Util_Mole.get_mol(xyz=xyz, basis=basis, spin=spin,
                            charge=charge, print_verbose=verbose)

    rohf = pyscf.scf.ROHF(mol)
    rohf.kernel()

    assert(atom_bas['basis'] == mol.basis)

    # atom_bas = get_atom_orb.atom_min_cas_bas(["C", "H", "O", "N"], basis=basis)

    # construct atomic basis for mole basis

    _, bas, loc_occ, bas_occ, _, _ = generate_atom_basis(mol, atom_bas)

    ovlp = mol.intor("int1e_ovlp")
    bas = numpy.matrix(bas)

    nocc = numpy.sum(rohf.mo_occ > 0)

    mole_orb_occ = rohf.mo_coeff[:, :nocc]

    # print()

    indx = []
    atom = []

    for i in range(nocc):
        indx.append(str(i))
    for i in range(mol.natm):
        atom.append(mol.atom_pure_symbol(i))

    # analysis orb 成分

    print("canonicalized orbitals")

    comp_orb = numpy.zeros((mol.natm, nocc))

    ovlp_atom_occ_mole_occ = reduce(numpy.dot, (bas_occ.T, ovlp, mole_orb_occ))
    ovlp_atom_occ_mole_occ = numpy.square(ovlp_atom_occ_mole_occ)
    for i in range(mol.natm):
        tmp = ovlp_atom_occ_mole_occ[loc_occ[i]:loc_occ[i+1], :]
        tmp = numpy.sum(tmp, axis=0)
        comp_orb[i, :] = tmp * 100
        # print(tmp.shape)
        if print_data:
            print("%2s " % (mol.atom_pure_symbol(i)), end="")
            for comp in tmp:
                print("%8.2f " % (comp*100), end="")
            print("")

    draw_heatmap(comp_orb, indx, atom)

    # latex

    if latex:
        for i in range(mol.natm):
            tmp = ovlp_atom_occ_mole_occ[loc_occ[i]:loc_occ[i+1], :]
            tmp = numpy.sum(tmp, axis=0)
            # print(tmp.shape)
            print("%2s &" % (mol.atom_pure_symbol(i)), end="")
            for comp in tmp:
                print("%8.2f &" % (comp*100), end="")
            print("\\\\\midrule")

    print("localized orbitals")
    loc_orb = lo.Boys(mol, mole_orb_occ).kernel()
    # mole_orb_occ[:,:] = loc_orb

    ovlp_atom_occ_mole_occ = reduce(numpy.dot, (bas_occ.T, ovlp, loc_orb))
    ovlp_atom_occ_mole_occ = numpy.square(ovlp_atom_occ_mole_occ)

    comp_orb = numpy.zeros((mol.natm, nocc))

    orb_label = ["1s", "2s", "2p", "2p", "2p"]

    for i in range(mol.natm):
        tmp = ovlp_atom_occ_mole_occ[loc_occ[i]:loc_occ[i+1], :]
        tmp = numpy.sum(tmp, axis=0)
        comp_orb[i, :] = tmp * 100
        if print_data:
            print("%2s " % (mol.atom_pure_symbol(i)), end="")
            for comp in tmp:
                print("%8.2f " % (comp*100), end="")
            print("")

    draw_heatmap(comp_orb, indx, atom)

    if latex:
        for i in range(mol.natm):
            tmp = ovlp_atom_occ_mole_occ[loc_occ[i]:loc_occ[i+1], :]
            tmp = numpy.sum(tmp, axis=0)
            print("%2s &" % (mol.atom_pure_symbol(i)), end="")
            for comp in tmp:
                print("%8.2f &" % (comp*100), end="")
            print("\\\\\midule")

    for i in range(mol.natm):
        tmp = ovlp_atom_occ_mole_occ[loc_occ[i]:loc_occ[i+1], :]
        nocc_atom = 5
        if mol.atom_pure_symbol(i) == "H":
            nocc_atom = 1
        if print_data:
            for j in range(nocc_atom):
                print("%2s %2s " %
                      (mol.atom_pure_symbol(i), orb_label[j]), end="")
                for comp in tmp[j]:
                    print("%8.2f " % (comp*100), end="")
                print("")

    return mol, rohf, loc_orb, comp_orb

# chemical bond analyzer


class ChemBondAnalyzer:
    def __init__(self, xyz=None, charge=0, spin=0, basis='6-31G(d)', symmetry="", print_verbose=0) -> None:

        # basic info

        self._mol = None
        self._atom_bas = None
        self.basis = basis

        # 辅助信息 -- 原子基组

        self.ovlp = None
        self.canonical_mo_occ = None
        self.nocc = 0

        self.bas_loc = None
        self.bas = None
        self.occ_loc = None
        self.occ_bas = None
        self.val_loc = None
        self.val_bas = None

        # 辅助信息 -- 成键分析

        self.rohf = None
        self.comp_occ_orb = None

        # 辅助信息 -- 分子

        if xyz is not None:
            self._get_mol(xyz, charge, spin, basis, symmetry, print_verbose)

    # setter and getter

    @property
    def mol(self):
        return self._mol

    @mol.setter
    def mol(self, input_mol):
        if self._mol == None:
            self._mol = input_mol
        else:
            raise RuntimeError

    @property
    def atom_bas(self):
        return self._atom_bas

    @atom_bas.setter
    def atom_bas(self, input):
        if self._atom_bas == None:
            assert(self.basis == input['basis'])
            self._atom_bas = input
        else:
            raise RuntimeError

    # build subobject

    def _get_mol(self, xyz=None, charge=0, spin=0, basis='6-31G(d)', symmetry="", print_verbose=0):
        self._mol = Util_Mole.get_mol(
            xyz, charge, spin, basis, symmetry, print_verbose)

    def _get_atom_bas(self):
        get_atom_orb.atom_min_cas_bas(["C", "H", "O", "N"], basis=self.basis)

    def _get_atom_basis_for_mol(self):

        if self._atom_bas is None:
            self._atom_bas = self._get_atom_bas()

        # do the work

        if self.bas_loc is None:
            self.bas_loc, self.bas, self.occ_loc, self.occ_bas, self.val_loc, self.val_bas = generate_atom_basis(
                self.mol, self.atom_bas)
            self.bas = numpy.matrix(self.bas)

    def _run_scf(self):
        if self.rohf is None:
            self.rohf = pyscf.scf.ROHF(self.mol)
            self.rohf.kernel()
            self.ovlp = self.mol.intor("int1e_ovlp")
            self.nocc = numpy.sum(self.rohf.mo_occ > 0)
            self.canonical_mo_occ = self.rohf.mo_coeff[:, :self.nocc]

    # do the work

    def analysis_mole_occ_orb(self):

        self._get_atom_basis_for_mol()
        self._run_scf()
