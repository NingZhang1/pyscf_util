import numpy
from functools import reduce
from pyscf.symm.Dmatrix import *
from torch import cos_

def _orthogonalize(_vec, _ovlp):
    # ndim = _vec.shape[0]
    nvec = _vec.shape[1]
    for i in range(nvec):
        veci = _vec[:, i]
        norm = numpy.sqrt(reduce(numpy.dot, (veci.T, _ovlp, veci)))
        # print("norm %e" % norm)
        _vec[:, i] /= norm
        for j in range(i+1, nvec):
            vecj = _vec[:, j]
            # print(veci)
            # print(vecj)
            inner_product = reduce(numpy.dot, (veci.T, _ovlp, vecj))
            # print("inner_product %e" % inner_product)
            _vec[:, j] -= inner_product * _vec[:, i]
    return _vec.copy()
    
def get_rotation_matrix(theta, u):

    # if theta > numpy.pi and theta < 2*numpy.pi:
    #     theta -= numpy.pi
    #     u = [x*-1.0 for x in u]

    # assert(theta >= 0)
    # assert(theta <= numpy.pi)

    u /= numpy.linalg.norm(u, ord=2)
    res = numpy.eye(3)
    tmp = numpy.zeros((3, 3), dtype=numpy.float64)
    tmp[0, 1] = -u[2]
    tmp[0, 2] = u[1]
    tmp[1, 0] = u[2]
    tmp[1, 2] = -u[0]
    tmp[2, 0] = -u[1]
    tmp[2, 1] = u[0]
    res += numpy.sin(theta) * tmp + (1-numpy.cos(theta)) * numpy.dot(tmp, tmp)
    return res


def get_euler_angle(rot_mat):
    cos_theta = rot_mat[2, 2]
    if abs(cos_theta-1) > 1e-8 and abs(cos_theta+1) > 1e-8:
        sin_theta_square = 1 - cos_theta * cos_theta
        sin_psi_sin_phi = rot_mat[0, 2]*rot_mat[2, 0] / (1-sin_theta_square)
        cos_psi = (rot_mat[1, 1] + sin_psi_sin_phi) / cos_theta
        cos_psi_sin_phi = -rot_mat[0, 2]*rot_mat[2, 1] / (1-sin_theta_square)
        sin_psi = (rot_mat[0, 1]/cos_theta) - cos_psi_sin_phi
        sin_theta = -rot_mat[2, 1] / cos_psi
        sin_phi = rot_mat[0, 2]/sin_theta
        cos_phi = rot_mat[1, 2]/sin_theta
        print("sin_psi",sin_psi)
        print("sin_phi",sin_phi)
        print("sin_theta",sin_theta)
        print("cos_psi",cos_psi)
        print("cos_phi",cos_phi)
        print("cos_theta",cos_theta)
        psi = numpy.arccos(cos_psi)
        theta = numpy.arccos(cos_theta)
        phi = numpy.arccos(cos_phi)
        if sin_psi < 0.0:
            psi += numpy.pi
        if sin_theta < 0.0:
            theta += numpy.pi
        if sin_phi < 0.0:
            phi += numpy.pi
        return [psi, theta, phi]
    else:
        phi = 0.0
        theta = 0.0
        if abs(cos_theta+1) <= 1e-8:
            theta = numpy.pi
        cos_psi = rot_mat[0,0]
        sin_psi = rot_mat[0,1]/cos_theta
        psi = numpy.arccos(cos_psi)
        if sin_psi < 0.0:
            psi += numpy.pi
        return [psi, theta, phi]

def get_rotation_matrix_euler_angle_ZYZ(psi, theta, phi):
    return reduce(numpy.dot, (get_rotation_matrix(psi, [0, 0, 1]),
                              get_rotation_matrix(theta, [0, 1, 0]),
                              get_rotation_matrix(phi, [0, 0, 1])))


if __name__ == "__main__":
    pass
