from certifi import where
import pyscf
import numpy
import Driver_SCF
import Util_Mole
from pyscf import tools
from functools import reduce


def get_symmetry_adapted_basis_Dooh(mol, coeff):

    orbsym_ID, _ = Util_Mole.get_orbsym(mol, coeff)

    E1g_x = [i for i, x in enumerate(orbsym_ID) if x is 2]
    E1g_y = [i for i, x in enumerate(orbsym_ID) if x is 3]
    E1u_x = [i for i, x in enumerate(orbsym_ID) if x is 7]
    E1u_y = [i for i, x in enumerate(orbsym_ID) if x is 6]

    E2g_x = [i for i, x in enumerate(orbsym_ID) if x is 10]
    E2g_y = [i for i, x in enumerate(orbsym_ID) if x is 11]
    E2u_x = [i for i, x in enumerate(orbsym_ID) if x is 15]
    E2u_y = [i for i, x in enumerate(orbsym_ID) if x is 14]

    E3g_x = [i for i, x in enumerate(orbsym_ID) if x is 12]
    E3g_y = [i for i, x in enumerate(orbsym_ID) if x is 13]
    E3u_x = [i for i, x in enumerate(orbsym_ID) if x is 17]
    E3u_y = [i for i, x in enumerate(orbsym_ID) if x is 16]

    E4g_x = [i for i, x in enumerate(orbsym_ID) if x is 20]
    E4g_y = [i for i, x in enumerate(orbsym_ID) if x is 21]
    E4u_x = [i for i, x in enumerate(orbsym_ID) if x is 25]
    E4u_y = [i for i, x in enumerate(orbsym_ID) if x is 24]

    E5g_x = [i for i, x in enumerate(orbsym_ID) if x is 22]
    E5g_y = [i for i, x in enumerate(orbsym_ID) if x is 23]
    E5u_x = [i for i, x in enumerate(orbsym_ID) if x is 27]
    E5u_y = [i for i, x in enumerate(orbsym_ID) if x is 26]

    factor = 1.0/numpy.sqrt(2)

    basis_trans = numpy.identity(coeff.shape[0], dtype=numpy.complex128)

    Lz = numpy.zeros(mol.nao)

    for orbx, orby in zip(E1g_x, E1g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 1
        Lz[orby] = -1

    for orbx, orby in zip(E1u_x, E1u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 1
        Lz[orby] = -1

    for orbx, orby in zip(E2g_x, E2g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 2
        Lz[orby] = -2

    for orbx, orby in zip(E2u_x, E2u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 2
        Lz[orby] = -2

    for orbx, orby in zip(E3g_x, E3g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 3
        Lz[orby] = -3

    for orbx, orby in zip(E3u_x, E3u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 3
        Lz[orby] = -3

    for orbx, orby in zip(E4g_x, E4g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 4
        Lz[orby] = -4

    for orbx, orby in zip(E4u_x, E4u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 4
        Lz[orby] = -4

    for orbx, orby in zip(E5g_x, E5g_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 5
        Lz[orby] = -5

    for orbx, orby in zip(E5u_x, E5u_y):
        basis_trans[orbx, orbx] = factor
        basis_trans[orby, orbx] = -factor*numpy.complex(0, 1)
        basis_trans[orby, orby] = factor*numpy.complex(0, 1)
        basis_trans[orbx, orby] = factor
        Lz[orbx] = 5
        Lz[orby] = -5

    # print(basis_trans)

    return numpy.matrix(basis_trans), Lz


if __name__ == "__main__":
    mol = pyscf.gto.M(
        verbose=4,
        atom='''
C   0.000000000000       0.000000000000      -0.621265
C   0.000000000000       0.000000000000       0.621265
''',
        basis={'C': 'ccpvqz', 'H': 'ccpvdz'},
        spin=0,
        charge=0,
        symmetry='dooh',
    )
    mol.build()

    my_scf = Driver_SCF.Run_SCF(mol)
    Driver_SCF.Analysis_SCF(mol, my_scf)

    orbsym_ID, orbsym = Util_Mole.get_orbsym(mol, my_scf.mo_coeff)

    print([i for i, x in enumerate(orbsym_ID) if x is 2])
    print([i for i, x in enumerate(orbsym_ID) if x is 7])
    print([i for i, x in enumerate(orbsym_ID) if x is 10])
    print([i for i, x in enumerate(orbsym_ID) if x is 15])

    basis_trans, Lz = get_symmetry_adapted_basis_Dooh(mol, my_scf.mo_coeff)

    print(Lz)

    h1e = reduce(numpy.dot, (my_scf.mo_coeff.T,
                 my_scf.get_hcore(), my_scf.mo_coeff))

    for i in range(mol.nao):
        for j in range(mol.nao):
            if abs(h1e[i, j]) > 1e-10:
                print(i, j, h1e[i, j])

    h1e_adapted = reduce(numpy.dot, (basis_trans.H,
                                     h1e, basis_trans))

    for i in range(mol.nao):
        for j in range(mol.nao):
            if abs(h1e_adapted[i, j]) > 1e-10:
                print(i, j, h1e_adapted[i, j])

    int2e_full = pyscf.ao2mo.full(
        eri_or_mol=mol, mo_coeff=my_scf.mo_coeff, aosym='1').reshape((mol.nao, mol.nao, mol.nao, mol.nao))

    for i in range(mol.nao):
        for j in range(mol.nao):
            for k in range(mol.nao):
                for l in range(mol.nao):
                    if abs(int2e_full[i, j, k, l]) > 1e-10:
                        pass
                        # print(i, j, k, l, int2e_full[i, j, k, l])

    print(int2e_full.shape)

    int2e_full = numpy.einsum("ijkl,ip->pjkl", int2e_full, basis_trans.conj())
    int2e_full = numpy.einsum("pjkl,jq->pqkl", int2e_full, basis_trans.conj())
    int2e_full = numpy.einsum("pqkl,kr->pqrl", int2e_full, basis_trans)
    int2e_full = numpy.einsum("pqrl,ls->pqrs", int2e_full, basis_trans)
    # print(int2e_full)

    for i in range(mol.nao):
        for j in range(mol.nao):
            for k in range(mol.nao):
                for l in range(mol.nao):
                    if abs(int2e_full[i, j, k, l]) > 1e-10:
                        # print(i, j, k, l,
                        #       int2e_full[i, j, k, l].real)
                        assert(abs(int2e_full[i, j, k, l].imag) < 1e-10)
                        assert(abs(Lz[i]+Lz[j]-Lz[k]-Lz[l]) < 1e-10)

    # print(h1e_adapted)
    # print(numpy.dot(basis_trans.H, basis_trans))
