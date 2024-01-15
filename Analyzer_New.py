import numpy

import os
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

#### deal with filename and path ####


#### struct to record data ####

class iCI_RawData:
    def __init__(self,
                 spintwo,
                 symmetry,
                 cmin,
                 ncfg,
                 ncsf,
                 istate,
                 evar,
                 ept,
                 etot) -> None:
        self.spintwo = spintwo
        self.symmetry = symmetry
        self.cmin = cmin
        self.ncfg = ncfg
        self.ncsf = ncsf
        self.istate = istate
        self.evar = evar
        self.ept = ept
        self.etot = etot

    def __str__(self) -> str:
        return "spintwo: %d, symmetry: %d, ncfg: %d, ncsf: %d, istate: %d, evar: %f, ept: %f, etot: %f" % (self.spintwo, self.symmetry, self.ncfg, self.ncsf, self.istate, self.evar, self.ept, self.etot)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        return self.spintwo == o.spintwo and self.symmetry == o.symmetry and self.cmin == o.cmin and self.istate == o.istate

    def __lt__(self, o: object) -> bool:
        ## first spintwo then symmetry then istate then cmin

        if self.spintwo == o.spintwo:
            if self.symmetry == o.symmetry:
                if self.istate == o.istate:
                    return self.cmin > o.cmin
                else:
                    return self.istate < o.istate
            else:
                return self.symmetry < o.symmetry
        else:
            return self.spintwo < o.spintwo

#### deal with info ####


CMIN_TAG = "(CMIN_SCHEDULE)"


def _fetch_cmin(filename):
    '''
    fetch info in lines like:  Cmin Schedule                (CMIN_SCHEDULE) = 1.0000e-04 7.0000e-05 5.0000e-05 3.0000e-05 1.5000e-05 
    '''

    file = open(filename)
    lines = file.readlines()

    cmin = []

    for line in lines:
        if CMIN_TAG in line:
            cmin = line.split("=")[1].split()
            cmin = [float(x) for x in cmin]
            break

    file.close()

    return cmin

### extract pt info ###


def Extract_NonRela_Pt_Info_New(filename: str):

    SPINTWO_LOC = 0
    SYMMETRY_LOC = 1
    NCFG_LOC = 2
    NCSF_LOC = 3
    E_VAR_LOC = 4
    E_PT_LOC = 5
    E_TOT_LOC = 6

    lines = None
    try:
        file = open(filename)
        lines = file.readlines()
        file.close()
    except:
        return None, False, False, 0

    Res = []

    CMIN = _fetch_cmin(filename)
    ITASK = 0

    for i in range(len(lines)):
        if "iCI_ENPT(2)_NonRela::Info" in lines[i]:
            begin = i
            end = i+1
            for j in range(begin+1, len(lines)):
                if "********************************" in lines[j]:
                    end = j
                    break

            DataTmp = []

            spintwo = None
            symmetry = None
            ncfg = None
            ncsf = None
            istate = None
            evar = None
            ept = None
            etot = None

            cmin = CMIN[ITASK]
            ITASK += 1

            for j in range(begin, end):
                if '_______________________________________________________________________________________________' in lines[j]:
                    for k in range(j+3, end):
                        if '_______________________________________________________________________________________________' in lines[k]:
                            break

                        str_tmp = lines[k].split("|")
                        try:
                            spintwo = int(str_tmp[SPINTWO_LOC])
                            istate = 0
                        except:
                            istate += 1
                            assert spintwo is not None

                        try:
                            symmetry = int(str_tmp[SYMMETRY_LOC])
                        except:
                            assert symmetry is not None

                        try:
                            ncfg = int(str_tmp[NCFG_LOC])
                        except:
                            assert ncfg is not None

                        try:
                            ncsf = int(str_tmp[NCSF_LOC])
                        except:
                            assert ncsf is not None

                        evar = float(str_tmp[E_VAR_LOC])
                        ept = float(str_tmp[E_PT_LOC])
                        etot = float(str_tmp[E_TOT_LOC])
                        DataTmp.append(iCI_RawData(
                            spintwo, symmetry, cmin, ncfg, ncsf, istate, evar, ept, etot))
                    break

            # print("DataTmp: ", len(DataTmp))
            Res.append(DataTmp)

    return Res


if __name__ == "__main__":

    lines_test = "(CMIN_SCHEDULE) = 1.0000e-04 7.0000e-05 5.0000e-05 3.0000e-05 1.5000e-05"
    cmin = lines_test.split("=")[1].split()
    cmin = [float(x) for x in cmin]

    print(cmin)

    filename = "/home/ningzhang/HPC_Task/CoreExtTest/pt_coreext/input_SO2_S2p_aug-cc-pVTZ_Vert.out"
    cmin = _fetch_cmin(filename)
    print(cmin)
    pt_info = Extract_NonRela_Pt_Info_New(filename)

    for array in pt_info:
        for item in array:
            print(item)
        print("**********")

    ## flat array
        
    pt_info_flat = []
    for array in pt_info:
        pt_info_flat.extend(array)
    
    ## sort array
        
    pt_info_flat.sort()

    ## print array

    for item in pt_info_flat:
        print(item)
    
    ## 文件名匹配，提取信息，根据文件名提取信息
        
    FILENAME = "input_SO2_S2p_aug-cc-pVTZ_Vert.out"
    FILENAME = FILENAME.split(".")[0]
    print(FILENAME)
    print(FILENAME.split("_"))
