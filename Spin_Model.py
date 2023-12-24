import numpy
from functools import reduce

"""

Spin_Model.py is a module for generating spin model Hamiltonian.

Contents
--------

::

        get_spin_coupling_term
        get_S2_matrix

"""

def _get_S_plus(spintwo):
    '''Get S+ matrix for a given spin
    '''
    res = numpy.zeros((spintwo+1, spintwo+1))
    for id, mtwo in enumerate(range(-spintwo+2, spintwo+1, 2)):
        res[id, id+1] = numpy.sqrt(spintwo*(spintwo+2)/4-mtwo*(mtwo-2)/4)
    return res


def _get_S_minus(spintwo):
    '''Get S- matrix for a given spin
    '''
    res = numpy.zeros((spintwo+1, spintwo+1))
    for id, mtwo in enumerate(range(-spintwo, spintwo-1, 2)):
        res[id+1, id] = numpy.sqrt(spintwo*(spintwo+2)/4-mtwo*(mtwo+2)/4)
    return res


def _get_Sz(spintwo):
    '''Get Sz matrix for a given spin
    '''
    res = numpy.zeros((spintwo+1, spintwo+1))
    for id, i in enumerate(range(-spintwo, spintwo+1, 2)):
        res[id, id] = i/2.0
    return res


def get_empty_matrix(spintwo_list):
    local_mat = []
    for spintwo in spintwo_list:
        local_mat.append(numpy.zeros((spintwo+1, spintwo+1)))
    return reduce(numpy.kron, local_mat)


def _get_identity_matrix_list(spintwo_list):
    res = []
    for spintwo in spintwo_list:
        res.append(numpy.identity(spintwo+1))
    return res


def get_spin_coupling_term(spintwo_list, A_loc, B_loc, J):
    '''Get spin coupling term for a given spin list
    Args:
        spintwo_list: a list of spin
        A_loc: the location of A spin
        B_loc: the location of B spin
        J: the coupling constant

    Kwargs:

    Returns:
        res: the spin coupling matrix
    '''
    res = get_empty_matrix(spintwo_list)

    # + - term
    tmp = _get_identity_matrix_list(spintwo_list)
    if A_loc == B_loc:
        tmp[A_loc] = numpy.dot(_get_S_plus(
            spintwo_list[A_loc]), _get_S_minus(spintwo_list[A_loc]))
    else:
        tmp[A_loc] = _get_S_plus(spintwo_list[A_loc])
        tmp[B_loc] = _get_S_minus(spintwo_list[B_loc])
    res += reduce(numpy.kron, tmp) * 0.5
    # - + term
    tmp = _get_identity_matrix_list(spintwo_list)
    if A_loc == B_loc:
        tmp[A_loc] = numpy.dot(_get_S_minus(
            spintwo_list[A_loc]), _get_S_plus(spintwo_list[A_loc]))
    else:
        tmp[A_loc] = _get_S_minus(spintwo_list[A_loc])
        tmp[B_loc] = _get_S_plus(spintwo_list[B_loc])
    res += reduce(numpy.kron, tmp) * 0.5
    # Sz Sz term
    tmp = _get_identity_matrix_list(spintwo_list)
    if A_loc == B_loc:
        tmp[A_loc] = numpy.dot(_get_Sz(spintwo_list[A_loc]),
                               _get_Sz(spintwo_list[A_loc]))
    else:
        tmp[A_loc] = _get_Sz(spintwo_list[A_loc])
        tmp[B_loc] = _get_Sz(spintwo_list[B_loc])
    res += reduce(numpy.kron, tmp)

    return res * J


def get_S2_matrix(spintwo_list):
    res = get_empty_matrix(spintwo_list)
    for id_i in range(len(spintwo_list)):
        for id_j in range(len(spintwo_list)):
            res += get_spin_coupling_term(spintwo_list, id_i, id_j, 1.0)
    return res

# 计算 prop


if __name__ == "__main__":
    print(_get_Sz(2))
    print(_get_S_plus(2))
    print(_get_S_minus(2))
    print(_get_Sz(5))
    print(_get_S_plus(5))
    print(_get_S_minus(5))
    print(get_empty_matrix([2, 3, 4]).shape)  # should be 60

    print(get_spin_coupling_term([2, 3], 0, 1, 1.0))

    mat = get_spin_coupling_term([2, 3], 0, 1, 1.0)

    e, m = numpy.linalg.eigh(mat)
    print(e)
