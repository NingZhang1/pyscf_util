import numpy

# dump file


def Dump_Cmoao(TaskName, mocoeff):
    filename = TaskName + ".csv"
    FILE = open(filename, "w")
    FILE.write("i,j,mocoeff\n")
    for i in range(mocoeff.shape[0]):
        for j in range(mocoeff.shape[1]):
            FILE.write("%d,%d,%20.12e\n" % (i, j, mocoeff[i][j]))
    FILE.close()

def Dump_Cmoao_4C(TaskName, mocoeff):
    filename = TaskName + ".csv"
    FILE = open(filename, "w")
    FILE.write("i,j,mocoeff_real,mocoeff_imag\n")
    for i in range(mocoeff.shape[0]):
        for j in range(mocoeff.shape[1]):
            FILE.write("%d,%d,%20.12e,%20.12e\n" % (i, j, mocoeff[i][j].real, mocoeff[i][j].imag))
    FILE.close()


def Dump_Relint_csv(TaskName, relint):
    filename = TaskName + ".csv"
    FILE = open(filename, "w")
    FILE.write("type,i,j,integrals\n")
    for i in range(relint.shape[0]):
        for j in range(relint.shape[1]):
            for k in range(relint.shape[2]):
                if abs(relint[i][j][k]) > 1e-8:
                    FILE.write("%d,%d,%d,%20.12e\n" %
                               (i, j, k, relint[i][j][k]))
    FILE.close()


def Dump_Relint_iCI(filename, relint, nao):
    FILE = open(filename, "w")
    for k in [0, 1, 2, 3]:
        for i in range(0, nao):
            for j in range(0, nao):
                if (abs(relint[k][i][j]) > 1e-12):
                    FILE.write("%.15f %d %d %d\n" %
                               (relint[k][i][j], i, j, k))  # 0 : x; 1 : y; 2 : z; 3: sf
    FILE.close()


def Dump_SpinRDM1(filename, rdm1):
    norb = rdm1.shape[1]
    nstates = rdm1.shape[0]
    file = open(filename, "w")
    file.write("istate,i,j,rdm1\n")
    for istate in range(nstates):
        for i in range(norb):
            for j in range(norb):
                if (abs(rdm1[istate][i][j]) > 1e-10):
                    file.write("%d,%d,%d,%20.12e\n" %
                               (istate, i, j, rdm1[istate][i][j]))
    file.close()


def Dump_SpinRDM2(filename, rdm2):
    norb = rdm2.shape[1]
    nstates = rdm2.shape[0]
    file = open(filename, "w")
    file.write("istate,p,q,r,s,rdm2\n")
    for istate in range(nstates):
        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        if (abs(rdm2[istate][p][q][r][s]) > 1e-10):
                            file.write("%d,%d,%d,%d,%d,%20.12e\n" %
                                       (istate, p, q, r, s, rdm2[istate][p][q][r][s]))
    file.close()

# ReadIn


def ReadIn_Relint_csv(TaskName, nao, skiprows=1):
    filename = TaskName + ".csv"
    i, j, k, val = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,d'),
                                 delimiter=',', skiprows=skiprows, unpack=True)
    relint = numpy.zeros((4, nao, nao))
    relint[i, j, k] = val
    return relint


def ReadIn_Cmoao(TaskName, nao, nmo=None, skiprows=1):
    filename = TaskName + ".csv"
    i, j, val = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,d'),
                              delimiter=',', skiprows=skiprows, unpack=True)
    if nmo is None:
        nmo = nao
    cmoao = numpy.zeros((nao, nmo))
    cmoao[i, j] = val
    return cmoao


def ReadIn_SpinRDM1(filename, norb, nstates, IsAveraged=False):
    if IsAveraged:
        i, j, val = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,d'),
                                  delimiter=',', skiprows=1, unpack=True)
        rdm1 = numpy.zeros((norb, norb))
        rdm1[i, j] = val
        return rdm1
    else:
        istate, i, j, val = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,d'),
                                          delimiter=',', skiprows=1, unpack=True)
        rdm1 = numpy.zeros((nstates, norb, norb))
        rdm1[istate, i, j] = val
        return rdm1

def ReadIn_RDM1_4C(filename, norb, nstates, IsAveraged=False):
    if IsAveraged:
        i, j, val_real, val_imag = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,d,d'),
                                  delimiter=',', skiprows=1, unpack=True)
        rdm1 = numpy.zeros((norb, norb), dtype=numpy.complex128)
        rdm1[i, j] = val_real + 1j * val_imag
        return rdm1
    else:
        istate, i, j, val_real, val_imag = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,d,d'),
                                          delimiter=',', skiprows=1, unpack=True)
        rdm1 = numpy.zeros((nstates, norb, norb), dtype=numpy.complex128)
        rdm1[istate, i, j] = val_real + 1j * val_imag
        return rdm1

def ReadIn_RDM1_4C_SU2(filename, norb, nstates, IsAveraged=False):
    if IsAveraged:
        i, j, val_real= numpy.loadtxt(filename, dtype=numpy.dtype('i,i,d'),
                                  delimiter=',', skiprows=1, unpack=True)
        rdm1 = numpy.zeros((norb, norb))
        rdm1[i, j] = val_real
        return rdm1
    else:
        istate, i, j, val_real= numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,d'),
                                          delimiter=',', skiprows=1, unpack=True)
        rdm1 = numpy.zeros((nstates, norb, norb))
        rdm1[istate, i, j] = val_real
        return rdm1
    
def ReadIn_RDM2_4C(filename, norb, nstates, IsAveraged=False):
    if IsAveraged:
        i, j, k, l, val_real, val_imag = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,i,d,d'),
                                  delimiter=',', skiprows=1, unpack=True)
        rdm2 = numpy.zeros((norb, norb, norb, norb), dtype=numpy.complex128)
        rdm2[i, j, k, l] = val_real + 1j * val_imag
        return rdm2
    else:
        istate, i, j, k, l, val_real, val_imag = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,i,i,d,d'),
                                          delimiter=',', skiprows=1, unpack=True)
        rdm2 = numpy.zeros((nstates, norb, norb, norb, norb), dtype=numpy.complex128)
        rdm2[istate, i, j, k, l] = val_real + 1j * val_imag
        return rdm2

def ReadIn_RDM2_4C_SU2(filename, norb, nstates, IsAveraged=False):
    if IsAveraged:
        i, j, k, l, val_real = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,i,d'),
                                  delimiter=',', skiprows=1, unpack=True)
        rdm2 = numpy.zeros((norb, norb, norb, norb))
        rdm2[i, j, k, l] = val_real
        return rdm2
    else:
        istate, i, j, k, l, val_real = numpy.loadtxt(filename, dtype=numpy.dtype('i,i,i,i,i,d'),
                                          delimiter=',', skiprows=1, unpack=True)
        rdm2 = numpy.zeros((nstates, norb, norb, norb, norb))
        rdm2[istate, i, j, k, l] = val_real
        return rdm2
    
if __name__ == "__main__":
    pass
