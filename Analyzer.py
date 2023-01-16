import TEST_CONFIG

import os
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# iCI 输出文件分析 

# 判断是不是 iCI　的输出文件


def is_iCI_Output(filename):
    file = open(filename)
    lines = file.readlines()
    for i in range(min(10, len(lines))):
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
                        # print(res)

    return Res


def Extract_NonRela_Pt_Info_New(filename: str):
    file = open(filename)
    lines = file.readlines()
    file.close()

    Res = []

    for i in range(len(lines)):
        if "iCI_ENPT(2)_NonRela::Info" in lines[i]:
            begin = i
            end = i+1
            for j in range(begin+1, len(lines)):
                if "********************************" in lines[j]:
                    end = j
                    break
            # print(begin,end)

            for j in range(begin, end):
                if '_______________________________________________________________________________________________' in lines[j]:
                    for k in range(j+3, end):
                        if '_______________________________________________________________________________________________' in lines[k]:
                            break
                        # print(lines[k].split("/"))
                        res = lines[k].split("|")
                        res = [float(x) for x in res]
                        restmp = [int(res[0]), int(res[1])]
                        restmp.extend([float(x) for x in res[2:]])
                        Res.append(restmp)
                        # print(res)
                    break

        if "iCI_ext_ENPT2_NonRela::Info" in lines[i]:
            begin = i
            end = i+1
            for j in range(begin+1, len(lines)):
                if "********************************" in lines[j]:
                    end = j
                    break
            # print(begin,end)

            for j in range(begin, end):
                if '_______________________________________________________________________________________________' in lines[j]:
                    for k in range(j+3, end):
                        if '_______________________________________________________________________________________________' in lines[k]:
                            break
                        # print(lines[k].split("/"))
                        res = lines[k].split("|")
                        res = [float(x) for x in res]
                        restmp = [int(res[0]), int(res[1])]
                        restmp.extend([float(x) for x in res[2:]])
                        Res.append(restmp)
                        # print(res)
                    break

    return Res


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
    return [a[0], a[1], a[2]**2, (a[1]-y[-1])*10**6, b, a[2]**2]


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
