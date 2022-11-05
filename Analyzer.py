import TEST_CONFIG

import os
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

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