import pyscf
from pyscf import lo
from torch import float64
import Driver_SCF
import get_atom_orb
import Util_Mole
# import Util_Pic
from Util_Pic import draw_heatmap
from functools import reduce
import numpy
import Util_Math

occ_label = {
    "H": ["1s"],
    "C": ["1s", "2s", "2p"],
    "N": ["1s", "2s", "2p"],
    "O": ["1s", "2s", "2p"],
    "F": ["1s", "2s", "2p"],
}
val_label = {
    "H": ["1s"],
    "C": ["2s", "2p"],
    "N": ["2s", "2p"],
    "O": ["2s", "2p"],
    "F": ["2s", "2p"],
}
vir_label_min = {
    "H": ["2s", "2p"],
    "C": ["3s", "3p", "3d"],
    "N": ["3s", "3p", "3d"],
    "O": ["3s", "3p", "3d"],
    "F": ["3s", "3p", "3d"],
}
vir_label = {
    "H": ["2s", "2p", "3s", "3p", "3d"],
    "C": ["3s", "3p", "3d", "4s", "4p", "4d"],
    "N": ["3s", "3p", "3d", "4s", "4p", "4d"],
    "O": ["3s", "3p", "3d", "4s", "4p", "4d"],
    "F": ["3s", "3p", "3d", "4s", "4p", "4d"],
}


def generate_atm_bas_given_label(mol, atom_min_cas_bas, label):

    atom_count = {
        "H": 0,
        "C": 0,
        "N": 0,
        "O": 0,
        "F": 0,
    }

    nbas = 0
    for i in range(mol.natm):
        atom_symbol = mol.atom_pure_symbol(i)
        for orb_symbol in label[atom_symbol]:
            if orb_symbol in atom_min_cas_bas[atom_symbol+"_bas_label"].keys():
                nbas += len(atom_min_cas_bas[atom_symbol +
                            "_bas_label"][orb_symbol])
    res = numpy.zeros((mol.nao, nbas))

    row_loc_begin = 0
    col_loc_begin = 0

    orb_info = {}

    for i in range(mol.natm):
        atom_symbol = mol.atom_pure_symbol(i)
        atom_min_cas_bas_ = atom_min_cas_bas[mol.atom_pure_symbol(i)]
        row_loc_end = row_loc_begin+atom_min_cas_bas_.shape[0]

        atom_count[atom_symbol] += 1
        atom_symbol_unique = atom_symbol + str(atom_count[atom_symbol])

        for orb_symbol in label[atom_symbol]:
            if orb_symbol in atom_min_cas_bas[atom_symbol+"_bas_label"].keys():
                nbas_tmp = len(
                    atom_min_cas_bas[atom_symbol+"_bas_label"][orb_symbol])
                col_loc_end = col_loc_begin + nbas_tmp
                res[row_loc_begin:row_loc_end, col_loc_begin:col_loc_end] = atom_min_cas_bas_[
                    :, atom_min_cas_bas[atom_symbol+"_bas_label"][orb_symbol]]
                orb_info[atom_symbol_unique+"_" +
                         orb_symbol] = list(range(col_loc_begin, col_loc_end))
                col_loc_begin = col_loc_end

        row_loc_begin = row_loc_end

    return res, orb_info


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
    def __init__(self, xyz=None, charge=0, spin=0, basis='6-31G(d)', symmetry="", print_verbose=0, vir_label=vir_label_min) -> None:

        # basic info

        self._mol = None
        self._atom_bas = None
        self.basis = basis
        self.verbose = print_verbose

        # 辅助信息 -- 原子基组

        self.ovlp = None
        self.canonical_mo_occ = None
        self.nocc = 0
        self.atm_occ_bas = None
        self.vir_label = vir_label
        self.mole_atm_bas = None
        self.mole_atm_loc = None

        # 辅助信息 -- 成键分析

        self.rohf = None
        # self.comp_occ_orb = None

        # 辅助信息 -- 分子

        if xyz is not None:
            self._get_mol(xyz, charge, spin, basis, symmetry, print_verbose)

    # setter and getter

    @property
    def mol(self):
        return self._mol

    @property
    def e_tot(self):
        if self.rohf is None:
            self._run_scf()
        return self.rohf.e_tot

    @mol.setter
    def mol(self, input_mol):
        if self._mol == None:
            self._mol = input_mol
            self.basis = input_mol.basis
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

    @property
    def dm(self):
        if self.rohf is None:
            self._run_scf()
        return numpy.dot(self.canonical_mo_occ, self.canonical_mo_occ.T) * 2.0

    @property
    def dm_mole_atom_bas(self):
        C_mole_atom_bas = numpy.dot(self.mole_atm_bas.I,self.canonical_mo_occ)
        return numpy.dot(C_mole_atom_bas, C_mole_atom_bas.T) * 2.0

    # build subobject

    def _get_mol(self, xyz=None, charge=0, spin=0, basis='6-31G(d)', symmetry="", print_verbose=0):
        self._mol = Util_Mole.get_mol(
            xyz, charge, spin, basis, symmetry, print_verbose)

    def _get_atom_bas(self):
        self._atom_bas = get_atom_orb.atom_min_cas_bas(
            ["C", "H", "O", "N", "F"], basis=self.basis)

        if self.verbose > 10:
            print(self.atom_bas)

    def _get_atom_basis_for_mol(self):

        if self._atom_bas is None:
            self._get_atom_bas()

        if self.atm_occ_bas is None:

            self.atm_occ_bas, self.atm_occ_label = generate_atm_bas_given_label(
                self.mol, self.atom_bas, occ_label)
            self.atm_nocc = self.atm_occ_bas.shape[1]

            atm_vir_bas, atm_vir_label = generate_atm_bas_given_label(
                self.mol, self.atom_bas, self.vir_label)

            self.atm_n_large = self.atm_nocc + atm_vir_bas.shape[1]
            self.atm_large_bas = numpy.zeros((self.mol.nao, self.atm_n_large))
            for key in atm_vir_label.keys():
                atm_vir_label[key] = [
                    x+self.atm_nocc for x in atm_vir_label[key]]
            self.atm_large_bas_label = {}
            for key in self.atm_occ_label:
                self.atm_large_bas_label[key] = self.atm_occ_label[key]
            for key in atm_vir_label:
                self.atm_large_bas_label[key] = atm_vir_label[key]
            self.atm_large_bas[:, :self.atm_nocc] = self.atm_occ_bas
            self.atm_large_bas[:,
                               self.atm_nocc:self.atm_n_large] = atm_vir_bas

            self.mole_atm_loc, self.mole_atm_bas,  _, _, _, _ = generate_atom_basis(
                self.mol, self.atom_bas)
            self.mole_atm_bas = numpy.matrix(self.mole_atm_bas)

            if self.verbose > 10:
                print(self.atm_large_bas)
                print(self.atm_large_bas_label)

    def _run_scf(self):
        if self.rohf is None:
            self.rohf = pyscf.scf.ROHF(self.mol)
            self.rohf.kernel()
            self.ovlp = self.mol.intor("int1e_ovlp")
            self.nocc = numpy.sum(self.rohf.mo_occ > 0)
            self.canonical_mo_occ = self.rohf.mo_coeff[:, :self.nocc]
            self.loc_mo = lo.Boys(self.mol, self.canonical_mo_occ).kernel()

            self._get_atom_basis_for_mol()
            self.atm_occ_bas_orth = Util_Math._orthogonalize(
                self.atm_occ_bas, self.ovlp)
            self.atm_large_bas_orth = Util_Math._orthogonalize(
                self.atm_large_bas, self.ovlp)

    def _check_input_mo(self, mo_occ, opt=False):
        self._run_scf()
        dm = numpy.dot(mo_occ, mo_occ.T) * 2.0
        # dm = numpy.dot(self.canonical_mo_occ, self.canonical_mo_occ.T) * 2.0
        # dm = pyscf.scf.hf.make_rdm1(self.rohf.mo_coeff, self.rohf.mo_occ)
        # print("dm shape", dm.shape)
        # res = 0.0
        # for i in range(dm.shape[0]):
        #     res += dm[i, i]
        # print("nelec = %15.8f", res)
        if opt is False:
            rohf_tmp = pyscf.scf.ROHF(self.mol)
            rohf_tmp.max_cycle = -1
            rohf_tmp.kernel(dm)
            print("the energy of HF with input MO is %15.8f, with diff %15.8e" %
                  (rohf_tmp.e_tot, rohf_tmp.e_tot - self.rohf.e_tot))
            return (float)(rohf_tmp.e_tot)
        else:
            rohf_tmp = Driver_SCF.Run_SCF_customized(self.mol, mo_occ)
            print("the energy of HF with opted MO is %15.8f, with diff %15.8e" %
                  (rohf_tmp.e_tot, rohf_tmp.e_tot - self.rohf.e_tot))
            return (float)(rohf_tmp.e_tot)
        # rohf_tmp = pyscf.scf.ROHF(self.mol)
        # rohf_tmp.kernel(dm)

    # do the work

    def analysis_mole_occ_orb(self):

        self._get_atom_basis_for_mol()
        self._run_scf()

    def check_cnvg_orb_proj_atm_occ(self, opt=False):
        self._run_scf()
        cnvg_orb = self.canonical_mo_occ
        coeff = reduce(numpy.dot, (cnvg_orb.T, self.ovlp,
                       self.atm_occ_bas_orth))  # (nocc, nocc_atm)
        orb_proj = numpy.dot(self.atm_occ_bas_orth, coeff.T)  # (nato,nocc)
        orb_proj = Util_Math._orthogonalize(orb_proj, self.ovlp)
        return self._check_input_mo(orb_proj, opt)

    def check_cnvg_orb_proj_atm_occ_vir(self, opt=False):
        self._run_scf()
        cnvg_orb = self.canonical_mo_occ
        coeff = reduce(numpy.dot, (cnvg_orb.T, self.ovlp,
                       self.atm_large_bas_orth))  # (nocc, nocc_atm)
        orb_proj = numpy.dot(self.atm_large_bas_orth, coeff.T)  # (nato,nocc)
        orb_proj = Util_Math._orthogonalize(orb_proj, self.ovlp)
        return self._check_input_mo(orb_proj, opt)
