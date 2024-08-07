import numpy

import os
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

"""

Analyzer.py: A collection of functions for analyzing the outfile generated by iCI 

Contents
--------

::

    is_iCI_Output
    Print_Rela_Pt_Info
    Extract_NonRela_Selection_Info_New
    Extract_NonRela_Pt_Info
    Extract_NonRela_Pt_Info_New
    Extract_NonRela_Pt_Info_Curve
    pack_data
    Extract_NonRela_Pt_Info_New_Curve
    Extract_point_Info_Curve
    Extract_Curve_Info
    LinearRegression_EstimateError
    draw_extra_pic

Driver 

::

    _generate_empty_res_
    _generate_empty_res
    _extract_Heff_Rela
    load_data_diff_file
    load_data_same_file
    load_data_selection_info

"""


def is_iCI_Output(filename):
    ''' Check if the file is generated by iCI
    Args:
        filename: the name of the file

    Kwargs:

    Returns:
        True or False
    '''
    file = open(filename)
    lines = file.readlines()
    # for i in range(min(10, len(lines))):
    for i in range(len(lines)):
        if "Iteractive Configuration Interaction with Selection" in lines[i]:
            return True
    file.close()
    return False


def Print_Rela_Pt_Info(filename):
    file = open(filename)
    lines = file.readlines()

    begin = 0
    end = 0
    for i in range(len(lines)):
        if "Begin ENPT2 Relativity" in lines[i]:
            begin = i
        if "End ENPT2 Relativity" in lines[i]:
            end = i+1
            break
    # for i in range(begin,end):
    #     print(lines[i])

    begin_ene = 0
    end_ene = 0
    for i in range(begin, end):
        if "------" in lines[i]:
            if (begin_ene == 0):
                begin_ene = i + 2
            else:
                end_ene = i
                break
    Res = []
    for i in range(begin_ene, end_ene):
        segment = lines[i].split("/")
        # print(float(segment[1]))
        # print(float(segment[2][:-1]))
        Res.append([float(segment[1]), float(segment[2][:-1])])
    # print(Res)

    def get_key(a):
        return a[1]

    Res.sort(key=get_key)

    for item in Res:
        print("%.8f,%.8f," % (item[0], item[1]))
    print("")

    file.close()


def Extract_NonRela_Selection_Info_New(filename: str, skip=1, macrocfg=True):

    lines = None

    try:
        file = open(filename)
        lines = file.readlines()
        file.close()
    except:
        return None, 0, 0

    Res = []

    nstate = None
    n_spacetype = None

    for i in range(len(lines)):

        if "BEGIN PRINT Important cfgspace INFO" in lines[i]:
            begin = i+3
            end = i+4
            for j in range(begin+1, len(lines)):
                if "Begin PRINT NONRELA CFGSPACE" in lines[j]:
                    end = j-1
                    break

            nstate_tmp = 0
            n_spacetype_tmp = 0

            for j in range(begin, end):
                if '|' not in lines[j]:
                    break
                res_str = lines[j].split("|")
                res = [float(x) for x in res_str[:-1]]
                tmp = res_str[-1].split(",")
                # print(tmp)
                # print(len(tmp))
                res.extend([float(x) for x in tmp[:-1]])
                restmp = [int(res[3]), int(res[4])]
                restmp.extend([float(x) for x in res[5:]])
                Res.append(restmp)
                nstate_tmp += (len(restmp) - 2)
                n_spacetype_tmp += 1

            if nstate == None:
                nstate = nstate_tmp
            if n_spacetype == None:
                n_spacetype = n_spacetype_tmp

            assert (nstate_tmp == nstate)
            assert (n_spacetype == n_spacetype_tmp)

    Res = Res[skip:]

    if macrocfg:
        nblock = len(Res) // n_spacetype
        assert (nblock % 2 == 0)
        ResNew = []
        for iblock in range(nblock):
            if iblock % 2 == 1:
                for j in range(n_spacetype):
                    ResNew.append(Res[iblock*n_spacetype+j])
        Res = ResNew

    return Res, nstate, n_spacetype


def Extract_NonRela_Pt_Info(filename: str, has_norm: bool):
    file = open(filename)
    lines = file.readlines()
    file.close()

    Res = []

    for i in range(len(lines)):
        if "ENPT2 Info" in lines[i]:
            begin = i
            end = i+1
            for j in range(begin+1, len(lines)):
                if "********************************" in lines[j]:
                    end = j
                    break
            # print(begin,end)

            for j in range(begin, end):
                if 'perturbation' in lines[j]:
                    for k in range(j+1, end):
                        if '---------------------------------------------------' in lines[k]:
                            break
                        # print(lines[k].split("/"))
                        res = lines[k].split("/")
                        res = [float(x) for x in res]
                        Res.append(res)
                        print(res)

    return Res


def Extract_NonRela_Pt_Info_New(filename: str):

    lines = None
    try:
        file = open(filename)
        lines = file.readlines()
        file.close()
    except:
        return None, False, False, 0

    Res = []

    find_iCIPT2 = False
    find_iCIext_PT2 = False

    nstate = None

    for i in range(len(lines)):
        if "iCI_ENPT(2)_NonRela::Info" in lines[i]:
            begin = i
            end = i+1
            for j in range(begin+1, len(lines)):
                if "********************************" in lines[j]:
                    end = j
                    break
            # print(begin,end)

            find_iCIPT2 = True
            restmp = None
            ncsf = 0
            ncfg = 0

            nstate_tmp = 0

            for j in range(begin, end):
                if '_______________________________________________________________________________________________' in lines[j]:
                    for k in range(j+3, end):
                        if '_______________________________________________________________________________________________' in lines[k]:
                            break
                        # print(lines[k].split("/"))
                        res = lines[k].split("|")
                        # res = [float(x) for x in res]
                        # restmp = [int(res[0]), int(res[1])]
                        try:
                            res = [float(x) for x in res]
                            restmp = [int(res[0]), int(res[1])]
                            ncsf = int(res[1])
                            ncfg = int(res[0])
                            restmp.extend([float(x) for x in res[2:]])

                        except:
                            # res = [0, 0]
                            # res.extend([float(x) for x in res[2:]])
                            restmp = [ncfg, ncsf]
                            restmp.extend([float(x) for x in res[2:]])

                        Res.append(restmp)
                        nstate_tmp += 1
                        # print(res)
                    break

            if nstate == None:
                nstate = nstate_tmp

            assert (nstate_tmp == nstate)

        if "iCI_ext_ENPT2_NonRela::Info" in lines[i]:
            begin = i
            end = i+1
            for j in range(begin+1, len(lines)):
                if "********************************" in lines[j]:
                    end = j
                    break
            # print(begin,end)

            find_iCIext_PT2 = True
            restmp = None
            ncsf = 0
            ncfg = 0

            for j in range(begin, end):
                if '_______________________________________________________________________________________________' in lines[j]:
                    for k in range(j+3, end):
                        if '_______________________________________________________________________________________________' in lines[k]:
                            break
                        # print(lines[k].split("/"))
                        res = lines[k].split("|")
                        # res = [float(x) for x in res]

                        try:
                            res = [float(x) for x in res]
                            restmp = [int(res[0]), int(res[1])]
                            ncsf = int(res[1])
                            ncfg = int(res[0])
                            restmp.extend([float(x) for x in res[2:]])

                        except:
                            # res = [0, 0]
                            # res.extend([float(x) for x in res[2:]])
                            restmp = [ncfg, ncsf]
                            restmp.extend([float(x) for x in res[2:]])

                        Res.append(restmp)
                        # print(res)
                    break

    return Res, find_iCIPT2, find_iCIext_PT2, nstate


# extract info

def Extract_NonRela_Pt_Info_Curve(DirName: str, FILE_F, BondLength, HAS_NORM=True):

    Res = []

    for id, bondlength in enumerate(BondLength):
        # print(id,bondlength)
        if os.path.isfile(os.path.join(DirName, FILE_F % (bondlength * 100))) == False:
            print("%s not foound " %
                  (os.path.join(DirName, FILE_F % (bondlength * 100))))
            continue
        data = Extract_NonRela_Pt_Info(
            os.path.join(DirName, FILE_F % (bondlength * 100)), has_norm=HAS_NORM)

        ResTmp = {}
        DataPoint = len(data)
        ResTmp['BondLength'] = bondlength

        # print(data)

        evar = []
        ept = []
        norm = []

        for data_id in range(DataPoint):
            DATA = {}
            # DATA['ncsf'] = data[2*data_id][0]
            # DATA['ncfg'] = data[2*data_id][1]
            DATA['evar'] = data[data_id][0]
            DATA['ept'] = data[data_id][1]
            ept.append(data[data_id][1])
            evar.append(data[data_id][0])
            if HAS_NORM:
                DATA['norm'] = data[data_id][2]
                norm.append(data[data_id][2])
            # ResTmp['Data_%d' % (data_id)] = DATA

        ResTmp['ept'] = ept
        ResTmp['evar'] = evar
        ResTmp['norm'] = norm

        Res.append(ResTmp)

    return Res


def pack_data(data, HAS_NORM=True):
    DataPoint = len(data)//2
    ResTmp = {}

    evar = []
    ept = []
    norm = []
    norm_ext = []
    ept_ext = []

    for data_id in range(DataPoint):
        ept.append(data[2*data_id][3])
        evar.append(data[2*data_id][2])
        ept_ext.append(data[2*data_id+1][3])
        if HAS_NORM:
            norm.append(data[2*data_id][5])
            norm_ext.append(data[2*data_id+1][4])
        # ResTmp['Data_%d' % (data_id)] = DATA

    ResTmp['ept'] = ept
    ResTmp['evar'] = evar
    ResTmp['norm'] = norm
    ResTmp['norm_ext'] = norm_ext
    ResTmp['ept_ext'] = ept_ext
    ResTmp['etot'] = [x+y for x, y in zip(ept, evar)]

    return ResTmp


def Extract_NonRela_Pt_Info_New_Curve(DirName: str, FILE_F, BondLength, HAS_NORM=True):

    Res = []

    for id, bondlength in enumerate(BondLength):
        # print(id,bondlength)
        if os.path.isfile(os.path.join(DirName, FILE_F % (bondlength * 100))) == False:
            continue
        data = Extract_NonRela_Pt_Info_New(
            os.path.join(DirName, FILE_F % (bondlength * 100)))

        ResTmp = {}
        DataPoint = len(data)//2
        ResTmp['BondLength'] = bondlength

        for data_id in range(DataPoint):
            DATA = {}
            DATA['ncsf'] = data[2*data_id][0]
            DATA['ncfg'] = data[2*data_id][1]
            DATA['evar'] = data[2*data_id][2]
            DATA['ept'] = data[2*data_id][3]
            DATA['ept_ext'] = data[2*data_id+1][3]
            if HAS_NORM:
                DATA['norm'] = data[2*data_id][5]
                DATA['norm_ext'] = data[2*data_id+1][4]
            ResTmp['Data_%d' % (data_id)] = DATA

        Res.append(ResTmp)

    return Res


def Extract_point_Info_Curve(Res, data_pnt: int = 6):
    Res1 = []
    for data in Res:
        ResTmp = {}
        ResTmp['BondLength'] = data['BondLength']
        keys = list(data['Data_0'].keys())
        # print(data)
        for key in keys:
            info = []
            for id in range(data_pnt):
                try:
                    info.append(data['Data_%d' % (id)][key])
                except:
                    continue
            ResTmp[key] = info
        Res1.append(ResTmp)

    return Res1


def Extract_Curve_Info(Res, BondLength, data_pnt: int = 6):
    Res2 = {}

    keys = list(Res[0]['Data_0'].keys())

    for id in range(data_pnt):
        info = {}
        info['bondlength'] = BondLength
        for key in keys:
            info_tmp = []
            for data in Res:
                info_tmp.append(data['Data_%d' % (id)][key])
            info[key] = info_tmp
        Res2['Data_%d' % (id)] = info

    return Res2

from scipy.optimize import curve_fit

def quadratic_fit_estimate_error(x, y, print_verbose=False):
    # Define the form of the function we want to fit
    def func(x, a, b, c):
        return a * x**2 + b * x + c

    x = np.asarray(x)
    y = np.asarray(y)

    # Use curve_fit to fit the function to our data. popt will contain the fitted parameters
    popt, pcov = curve_fit(func, x, y)

    # Generate predicted y values from our fitted function
    y_pred = func(x, *popt)

    # Calculate the root mean square error between the predicted and actual y values
    c = popt[2]
    c_error = np.sqrt(pcov[2][2])

    if print_verbose:
        print("Quadratic Regression : \n")
        print("a                 : %15.8f\n" % popt[0])
        print("b                 : %15.8f\n" % popt[1])
        print("c                 : %15.8f\n" % popt[2])
        print("c_error           : %15.8f\n" % c_error)

    return c, c_error

def LinearRegression_EstimateError(x, y, print_verbose=False):

    if (len(x) != len(y)):
        print("Error Different Length\n")
        return

    t_25 = [0, 12.7062, 4.3027, 3.1824, 2.7764,
            2.5706, 2.4469, 2.3646, 2.3060,
            2.2622, 2.2281, 2.2010, 2.1788,
            2.1604, 2.1448, 2.1315, 2.1199]
    Mean_x = np.mean(x)
    Mean_y = np.mean(y)
    Sxx = 0
    Syy = 0
    Sxy = 0
    for i in range(len(x)):
        Sxx = Sxx + (x[i]-Mean_x)**2
        Syy = Syy + (y[i]-Mean_y)**2
        Sxy = Sxy + (x[i]-Mean_x) * (y[i]-Mean_y)
    N = len(x)
    a = stats.linregress(x, y)
    sigma = 0.0
    if N > 2:
        sigma = (1.0/(N-2))*(Syy - a[0] * Sxy)
    sigma = np.sqrt(sigma)
    # print(sigma)
    # print(t_25[N-2])
    b = t_25[N-2]*sigma*np.sqrt(1.0/N+Mean_x*Mean_x/Sxx)
    if print_verbose:
        print("Linear Regression : \n")
        print("Slope             : %16.8e\n" % a[0])
        print("Intercept         : %16.8e\n" % a[1])
        print("R Value           : %16.8e\n" % a[2])
        print("R Square          : %16.8e\n" % a[2]**2)
        print("0.95 Interval x=0 : %16.8f +- %16.8e\n" % (a[1], b))
    # return [a[0], a[1], a[2]**2, (a[1]-y[-1])*10**6, b, a[2]**2]
    return [a[0], a[1], a[2], b/t_25[N-2]]


def draw_extra_pic(x: list,
                   y: list,
                   legend: list,
                   line_prop: list,
                   xlabel: str = '$E_{pt}^{(2)}/E_H$',
                   ylabel: str = 'E_{tot}/E_H',
                   title=""):
    plt.figure(figsize=(16, 9))
    for id, x in enumerate(x):
        plt.plot(x, y[id], marker=line_prop[id]['marker'], markersize=line_prop[id]
                 ['markersize'], linewidth=line_prop[id]['linewidth'], label=legend[id])
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=18)
    plt.show()


# Driver for analysis


def _generate_empty_res_():
    Res = {
        'ncsf': None,
        'ncfg': None,
        'eiCI': None,
        'ept': None,
        'ept_ext': None,
        'ept_norm': None,
        'ept_ext_norm': None,
    }
    return Res


def _generate_empty_res(nroot):
    Res = {'Heff_Rela': None, }
    for i in range(nroot):
        Res["root_%d" % (i)] = _generate_empty_res_()
        for key in Res["root_%d" % (i)].keys():
            Res["root_%d" % (i)][key] = []
    return Res


def _extract_Heff_Rela(dirname, filename, complex=False):

    filepath = os.path.join(dirname, filename)
    Res = []
    lines = None

    try:
        file = open(filepath)
        lines = file.readlines()
        file.close()
    except:
        return None

    for i in range(len(lines)):
        if "The effective Hamiltonian matrix elements of" in lines[i]:
            begin = i
            end = i+1
            for j in range(begin+1, len(lines)):
                if "-------------------------------------------------------" in lines[j]:
                    end = j
                    break
            if end == (i+1):
                Res.append(None)
            else:
                Heff = []
                for j in range(begin, end):
                    if "Row" in lines[j]:
                        file_str = lines[j].split(" ")
                        # print(file_str)
                        tmp = []
                        for data in file_str[:-1]:
                            try:
                                tmp.append(float(data))
                            except:
                                pass
                        tmp.append(float(file_str[-1][:-1]))
                        tmp = tmp[1:]
                        tmp_complex = []
                        # print(tmp)
                        if complex:
                            for k in range(len(tmp)//2):
                                tmp_complex.append(
                                    numpy.complex(tmp[2*k], tmp[2*k+1]))
                            # print(k, numpy.complex(tmp[2*k], tmp[2*k+1]))
                            Heff.append(tmp_complex)
                        else:
                            Heff.append(tmp)
                Heff = numpy.array(Heff)
                # print(Heff.shape)
                for i in range(Heff.shape[1]):
                    Heff[i, i] = 0.0
                Res.append(Heff)

    return Res


def load_data_diff_file(dirname, file_FORMAT, id_list,
                        do_extpt=False, do_pt_with_norm=True):

    Res = None

    for id in id_list:
        filepath = os.path.join(dirname, file_FORMAT % (id))
        ResTmp, find_pt, find_expt, nstate = Extract_NonRela_Pt_Info_New(
            filepath)

        if ResTmp == None:
            find_expt = do_extpt
            continue

        assert (do_extpt == find_expt)

        if Res == None:
            Res = _generate_empty_res(nstate)

        if find_expt:
            for id, data in enumerate(ResTmp):
                root_id = id % nstate
                batch_id = id // (nstate)
                if batch_id % 2 == 0:
                    Res["root_%d" % (root_id)]["ncsf"].append(data[1])
                    Res["root_%d" % (root_id)]["ncfg"].append(data[0])
                    Res["root_%d" % (root_id)]["eiCI"].append(data[2])
                    Res["root_%d" % (root_id)]["ept"].append(data[3])
                    if do_pt_with_norm:
                        Res["root_%d" % (root_id)]["ept_norm"].append(data[5])
                if batch_id % 2 == 1:
                    Res["root_%d" % (root_id)]["ept_ext"].append(data[3])
                    if do_pt_with_norm:
                        Res["root_%d" %
                            (root_id)]["ept_ext_norm"].append(data[4])
        else:
            for id, data in enumerate(ResTmp):
                root_id = id % nstate
                batch_id = id // (nstate)
                Res["root_%d" % (root_id)]["ncsf"].append(data[1])
                Res["root_%d" % (root_id)]["ncfg"].append(data[0])
                Res["root_%d" % (root_id)]["eiCI"].append(data[2])
                Res["root_%d" % (root_id)]["ept"].append(data[3])
                Res["root_%d" % (root_id)]["ept_norm"].append(data[5])

    return Res


def load_data_same_file(dirname, filename, do_extpt=False, do_pt_with_norm=True):

    filepath = os.path.join(dirname, filename)
    ResTmp, find_pt, find_expt, nstate = Extract_NonRela_Pt_Info_New(
        filepath)

    if ResTmp == None:
        return None

    assert (find_pt)
    assert (do_extpt == find_expt)

    Res = _generate_empty_res(nstate)

    if find_expt:
        for id, data in enumerate(ResTmp):
            root_id = id % nstate
            batch_id = id // (nstate)
            if batch_id % 2 == 0:
                Res["root_%d" % (root_id)]["ncsf"].append(data[1])
                Res["root_%d" % (root_id)]["ncfg"].append(data[0])
                Res["root_%d" % (root_id)]["eiCI"].append(data[2])
                Res["root_%d" % (root_id)]["ept"].append(data[3])
                if do_pt_with_norm:
                    Res["root_%d" % (root_id)]["ept_norm"].append(data[5])
            if batch_id % 2 == 1:
                Res["root_%d" % (root_id)]["ept_ext"].append(data[3])
                if do_pt_with_norm:
                    Res["root_%d" %
                        (root_id)]["ept_ext_norm"].append(data[4])
    else:
        for id, data in enumerate(ResTmp):
            root_id = id % nstate
            batch_id = id // (nstate)
            Res["root_%d" % (root_id)]["ncsf"].append(data[1])
            Res["root_%d" % (root_id)]["ncfg"].append(data[0])
            Res["root_%d" % (root_id)]["eiCI"].append(data[2])
            Res["root_%d" % (root_id)]["ept"].append(data[3])
            if do_pt_with_norm:
                Res["root_%d" % (root_id)]["ept_norm"].append(data[5])

    return Res


def load_data_selection_info(dirname, filename, skip=1, macrocfg=False):

    filepath = os.path.join(dirname, filename)
    ResTmp, nstate, nspace_type = Extract_NonRela_Selection_Info_New(
        filepath, skip, macrocfg)

    if ResTmp == None:
        return None

    Res = _generate_empty_res(nstate)

    root_id = None
    batch_id = None

    for id, data in enumerate(ResTmp):
        # space_id = id % nspace_type

        if batch_id == None:
            batch_id = id // nspace_type
            root_id = 0

        if batch_id != (id // nspace_type):
            root_id = 0
            batch_id = id // nspace_type

        # root_id = id % nstate
        # batch_id = id // (nstate)

        nstate_tmp = len(data) - 2
        for id in range(nstate_tmp):
            # Res["root_%d" % (root_id)]["ncsf"].append(data[1])
            # Res["root_%d" % (root_id)]["ncfg"].append(data[0])
            Res["root_%d" % (root_id)]["ncsf"].append(data[0])
            Res["root_%d" % (root_id)]["ncfg"].append(data[1])
            Res["root_%d" % (root_id)]["eiCI"].append(data[id+2])
            root_id += 1

    return Res
