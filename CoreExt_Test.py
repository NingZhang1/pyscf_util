import copy
from pyscf import tools
import pyscf
import pickle
import Util_CoreExt

Mole_Geometry = {
    "CO":  '''
C 0.000000 0.000000  0.000000
O 0.000000 0.000000  1.128400
''',
    "CO2":  '''
C 0.000000 0.000000  0.000000
O 0.000000 0.000000  1.160000
O 0.000000 0.000000 -1.160000
''',
    "CH4":     '''
C   0.0000000  0.0000000  0.0000000
H   0.0000000  0.0000000  1.0861000
H   0.0000000 -1.0239849 -0.3620333
H  -0.8867969  0.5119924 -0.3620333
H   0.8867969  0.5119924 -0.3620333
''',
    "NH3":     '''
N   0.00000000 0.46869816  0.37922759
H   0.81180903 0.00000000  0.00000000
H   0.00000000 1.40609449  0.00000000
H  -0.81180903 0.00000000  0.00000000
''',
    "H2O":         '''
O  0.000000  0.0000000 -0.1172519
H  0.000000 -0.7571643  0.4690074
H  0.000000  0.7571643  0.4690074
''',
    "HF":     '''
F  0.000000 0.000000 0.000000
H  0.000000 0.000000 0.916800
''',
    "C2H2":     '''
C 0.000000 0.000000 -0.601500
C 0.000000 0.000000  0.601500
H 0.000000 0.000000 -1.663000
H 0.000000 0.000000  1.663000
''',
    "H2CO":     '''
C  0.0000000   0.0000000	 -0.5296279
O  0.0000000   0.0000000	  0.6741721
H  0.0000000   0.9361475	 -1.1078044
H  0.0000000  -0.9361475     -1.1078044
''',
    "N2":     '''
N 0.000000 0.000000 0.000000
N 0.000000 0.000000 1.097600
''',
    "O2":     '''
O 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.207700
''',  # NOTE: 基态是个三重态！
    "SiH4":     '''
Si  0.000000  0.000000  0.000000
H   0.854400  0.854400  0.854400
H  -0.854400 -0.854400  0.854400
H  -0.854400  0.854400 -0.854400
H   0.854400 -0.854400 -0.854400
''',
    "PH3":     '''
P   0.00000000 0.59312580  0.76512855
H   1.02732402 0.00000000  0.00000000
H   0.00000000 1.77937741  0.00000000
H  -1.02732402 0.00000000  0.00000000
''',
    "H2S":      '''
S  0.000000  0.000000  -0.1855710
H  0.000000  0.9608222  0.7422841
H  0.000000 -0.9608222  0.7422841
''',
    "HCl":     '''
Cl  0.000000 0.000000 0.000000
H   0.000000 0.000000 1.274400
''',
    "SO2":     '''
S  0.0000000    0.0000000    -0.3618316
O  0.0000000    1.2359241     0.3618316
O  0.0000000   -1.2359241     0.3618316
''',
    "H3COH":     '''
C  -0.0503000   0.6685000    0.0000000
O  -0.0503000  -0.7585000    0.0000000
H  -1.0807000   1.0417000    0.0000000
H   0.4650000   1.0417000    0.8924000
H   0.4650000   1.0417000   -0.8924000
H   0.8544000  -1.0417000    0.0000000
''',
    "Cl2":     '''
Cl 0.000000 0.000000 0.000000
Cl 0.000000 0.000000 1.987900
''',
    "NNO":     '''
    N 0.0000    0.0000    0.0000
    N 1.0256    0.0000    0.0000
    O 2.0000    0.0000    0.0000
''',
    "CH3CN": '''
   N  1.2608    0.0000    0.0000 
   C -1.3613    0.0000    0.0000 
   C  0.1006    0.0000    0.0000 
   H -1.7500   -0.8301    0.5974 
   H -1.7501   -0.1022   -1.0175 
   H -1.7500    0.9324    0.4202 
   ''',
    "HCN": '''
  N -0.5800    0.0000    0.0000 
  C  0.5800    0.0000    0.0000 
  H  1.6450    0.0000    0.0000 
''',
    "O3": '''
  O -0.0950   -0.4943    0.0000 
  O  1.1489    0.2152    0.0000 
  O -1.0540    0.2791    0.0000 
'''
}

##### GEOMETRY OPTIMIZED VIA B3LYP aug-ccpvtz in Bohr #####

Mole_Geometry_Bohr = {
    "CO":  '''
C 0.         0.         0.002402  
O 0.         0.         2.12996496
''',
    "CO2":  '''
C 0.          0.          0.        
O 0.          0.          2.19297627
O 0.          0.         -2.19297627
''',
    "CH4":     '''
C   4.68249694e-14 -1.01868188e-06 -2.69392358e-07
H   7.35241212e-14 -1.90735245e-07  2.05654391e+00
H  -1.30449729e-13 -1.93892934e+00 -6.85514063e-01
H  -1.67916106e+00  9.69465181e-01 -6.85514693e-01
H   1.67916106e+00  9.69465181e-01 -6.85514693e-01
''',
    "NH3":     '''
N  -1.45354658e-14  8.85711161e-01  7.10648982e-01
H   1.54081995e+00 -3.88178152e-03  1.99576692e-03
H   3.56290566e-14  2.66489705e+00  1.99576837e-03
H  -1.54081995e+00 -3.88178152e-03  1.99576692e-03
''',
    "H2O":         '''
O  0.          0.         -0.22000133
H  0.         -1.44281777  0.88550921
H  0.          1.44281777  0.88550921
''',
    "HF":     '''
F  0.          0.         -0.00692527
H  0.          0.          1.73942619
''',
    "C2H2":     '''
C 0.          0.         -1.13045612
C 0.          0.          1.13045612
H 0.          0.         -3.13661966
H 0.          0.          3.13661966
''',
    "H2CO":     '''
C  0.          0.         -0.99260976
O  0.          0.          1.27567854
H  0.          1.77341306 -2.09840683
H  0.         -1.77341306 -2.09840683
''',
    "N2":     '''
N 0.         0.        -1.0309951
N 0.         0.         1.0309951
''',
    "O2":     '''
O 0.          0.         -1.13791786
O 0.          0.          1.13791786
''',  # NOTE: 基态是个三重态！
    "SiH4":     '''
Si -1.53358317e-13  4.83937748e-13 -4.36697869e-13
H   1.61747223e+00  1.61747223e+00  1.61747223e+00
H  -1.61747223e+00 -1.61747223e+00  1.61747223e+00
H  -1.61747223e+00  1.61747223e+00 -1.61747223e+00
H   1.61747223e+00 -1.61747223e+00 -1.61747223e+00
''',
    "PH3":     '''
P   8.31537687e-14  1.12084654e+00  1.44917485e+00
H   1.95612313e+00 -8.52643878e-03 -1.09740995e-03
H  -1.53388980e-13  3.37958764e+00 -1.09662522e-03
H  -1.95612313e+00 -8.52643878e-03 -1.09740995e-03
''',
    "H2S":      '''
S  0.          0.         -0.35232488
H  0.          1.83662056  1.40353691
H  0.         -1.83662056  1.40353691
''',
    "HCl":     '''
Cl  0.          0.         -0.00868048
H   0.          0.          2.41694745
''',
    "SO2":     '''
S  0.          0.         -0.71086447
O  0.          2.35338623  0.69731355
O  0.         -2.35338623  0.69731355
''',
    "H3COH":     '''
C  -0.06153042  1.27161433  0.        
O  -0.11262059 -1.41864888  0.        
H  -2.01894885  1.90072611  0.        
H   0.87002457  2.02742646  1.6835658 
H   0.87002457  2.02742646 -1.6835658 
H   1.59274454 -2.04156444  0.        
''',
    "Cl2":     '''
Cl 0.          0.         -1.91175365
Cl 0.          0.          1.91175365
''',
    "NNO":     '''
    N -2.52146955e-01  0.00000000e+00 0.00000000e+00
    N  1.86622219e+00  0.00000000e+00 0.00000000e+00
    O  4.10348013e+00  0.00000000e+00 0.00000000e+00
''',
    "CH3CN": '''
   N  2.34428892e+00 -3.52234270e-06 -2.96776429e-05 
   C -2.57636409e+00  3.46414736e-05  4.38525144e-05 
   C  1.72212306e-01  9.57945287e-06 -2.08898454e-05 
   H -3.28705613e+00 -1.56847662e+00  1.12879623e+00 
   H -3.28712060e+00 -1.93192614e-01 -1.92267752e+00 
   H -3.28702256e+00  1.76181751e+00  7.94076978e-01 
   ''',
    "HCN": '''
  N -1.07874586e+00  0.00000000e+00 -2.39529697e-16 
  C  1.08686968e+00  0.00000000e+00  2.41333548e-16 
  H  3.10047565e+00  0.00000000e+00  6.88443891e-16 
''',
    "O3": '''
  O -0.03647225 -0.81030057  0.        
  O  2.05158405  0.31377831  0.        
  O -2.01530078  0.49652226  0.        
'''
}

BASIS = [
    "aug-cc-pVDZ",
    # "aug-cc-pVTZ",
    # "aug-cc-pVDZ-DK",
    # "aug-cc-pVTZ-DK",
]

basis_test = "cc-pVDZ"

#### describe the task ####


def _get_gt_nsegment(mol, cas_orb):
    """Get the nsegment for the ground state
    """

    nocc = mol.nelectron // 2

    nsegment = "0 %d %d %d %d 0" % (
        nocc-cas_orb[0], cas_orb[0], cas_orb[1], mol.nao - nocc - cas_orb[1])

    return nsegment


def _dump_FCIDUMP(mol, hf, filename):

    mo = copy.deepcopy(hf.mo_coeff)
    orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
    tools.fcidump.from_mo(mol, filename, mo, orbsym)


Vert_Ext = {
    "CO": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [1, 2],
            "cas": [4, 4],
            # "task": "0 0 2 1 1 0 2 1 1 0 3 1 1 2 0 2 1 1 2 2 1 1 2 3 1 1",
            "task": "0 0 2 1 1 0 2 1 1 0 3 1 1",
            "type": "C1s",
        },
        "O1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            # "task": "0 0 2 1 1 0 2 1 1 0 3 1 1 2 0 2 1 1 2 2 1 1 2 3 1 1",
            "task": "0 0 2 1 1 0 2 1 1 0 3 1 1",
            "type": "O1s",
        },
    },
    "CO2": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [2, 3],
            "cas": [4, 4],
            # "task": "0 0 1 1 0 5 1 1 0 6 1 1 0 7 1 1 2 0 1 1 2 5 1 1 2 6 1 1 2 7 1 1",
            "task": "0 0 1 1 0 5 1 1 0 6 1 1 0 7 1 1",
            "type": "C1s",
        }
    },
    "CH4": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            # "task": "0 0 1 1 0 1 1 1 0 2 1 1 0 3 1 1 2 0 1 1 2 1 1 1 2 2 1 1 2 3 1 1",
            "task": "0 0 1 1 0 1 1 1 0 2 1 1 0 3 1 1",
            "type": "C1s",
        }
    },
    "C2H2": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [0, 2],
            "cas": [4, 4],
            # "task": "0 0 2 1 1 0 5 2 1 1 2 0 2 1 1 2 5 2 1 1", # TODO: not correct!
            "task": "0 0 2 1 1 0 2 2 1 1 0 3 2 1 1 0 5 2 1 1 0 6 2 1 1 0 7 2 1 1",
            "type": "C1s",
        }
    },
    "H2CO": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [1, 2],
            "cas": [4, 4],
            # "task": "0 0 2 1 1 0 2 1 1 0 3 1 1 2 0 2 1 1 2 2 1 1 2 3 1 1",
            "task": "0 0 2 1 1 0 2 1 1 0 3 1 1",
            "type": "C1s",
        },
        "O1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            # "task": "0 0 2 1 1 0 2 1 1 0 3 1 1 2 0 2 1 1 2 2 1 1 2 3 1 1",
            "task": "0 0 2 1 1 0 2 1 1 0 3 1 1",
            "type": "O1s",
        }
    },
    "N2": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "N1s": {
            "loc": [0, 2],
            "cas": [4, 4],
            # "task": "0 0 2 1 1 0 2 1 1 0 3 1 1 0 5 2 1 1 0 6 1 1 0 7 1 1 2 0 2 1 1 2 2 1 1 2 3 1 1 2 5 2 1 1 2 6 1 1 2 7 1 1",
            "task": "0 0 3 1 1 1 0 2 2 1 1 0 3 2 1 1 0 5 3 1 1 1 0 6 2 1 1 0 7 2 1 1",
            "type": "N1s",
        }
    },
    "O2": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "O1s": {
            "loc": [0, 2],
            "cas": [4, 4],
            # "task": "0 0 2 1 1 0 3 1 1 0 5 2 1 1 0 6 1 1 2 0 2 1 1 2 3 1 1 2 5 2 1 1 2 6 1 1", # TODO: not correct!
            "task": "0 0 3 1 1 1 0 2 2 1 1 0 3 2 1 1 0 5 3 1 1 1 0 6 2 1 1 0 7 2 1 1",
            "type": "O1s",
        }
    },
    "SiH4": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "Si1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "0 0 1 1 2 0 1 1",
            "type": "Si1s",
        },
        # "Si2s": {
        #     "loc": [1, 2],
        #     "cas": [4, 4],
        #     "task": "0 0 1 1 2 0 1 1",
        #     "type": "Si2s",
        # },
        # "Si2p": {
        #     "loc": [2, 5],
        #     "cas": [4, 4],
        #     "task": "0 1 1 1 0 2 1 1 0 3 1 1 2 1 1 1 2 2 1 1 2 3 1 1",
        #     "type": "Si2p",
        # },
    },
    "PH3": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "P1s": {
            "loc": [0, 1],
            "cas": [4, 5],
            "task": "0 0 3 1 1 1 0 1 2 1 1 2 0 3 1 1 1 2 1 2 1 1",
            "type": "P1s",
        },
        # "P2p": {
        #     "loc": [2, 5],
        #     "cas": [4, 5],
        #     "task": "0 0 8 1 1 1 1 1 1 1 1 0 1 7 1 1 1 1 1 1 1 2 0 8 1 1 1 1 1 1 1 1 2 1 7 1 1 1 1 1 1 1",
        #     "type": "P2p",
        # },
    },
    "H2S": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "S1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            # "task": "0 0 1 1 0 3 1 1 2 0 1 1 2 3 1 1", # seems to be problematic
            "task": "0 0 3 1 1 1 0 2 1 1 0 3 2 1 1",
            "type": "S1s",
        },
        # "S2s": {
        #     "loc": [1, 2],
        #     "cas": [4, 4],
        #     "task": "0 0 1 1 0 3 1 1 2 0 1 1 2 3 1 1",
        #     "type": "S2s",
        # },
        # "S2p": {
        #     "loc": [2, 5],
        #     "cas": [4, 4],
        #     "task": "0 0 2 1 1 0 1 1 1 0 2 1 1 0 3 2 1 1 2 0 2 1 1 2 1 1 1 2 2 1 1 2 3 2 1 1",
        #     "type": "S2p",
        # },
    },
    "HCl": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "Cl1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "0 0 2 1 1 0 2 1 1 0 3 1 1 2 0 2 1 1 2 2 1 1 2 3 1 1",
            "type": "Cl1s",
        },
        # "Cl2s": {
        #     "loc": [1, 2],
        #     "cas": [4, 4],
        #     "task": "0 0 2 1 1 0 2 1 1 0 3 1 1 2 0 2 1 1 2 2 1 1 2 3 1 1",
        #     "type": "Cl2s",
        # },
        # "Cl2p": {
        #     "loc": [2, 5],
        #     "cas": [4, 4],
        #     "task": "0 0 4 1 1 1 1 0 1 2 1 1 0 2 3 1 1 1 0 3 3 1 1 1 2 0 4 1 1 1 1 2 1 2 1 1 2 2 3 1 1 1 2 3 3 1 1 1",
        #     "type": "Cl2p",
        # },
    },
    "SO2": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [3, 3],
        },
        "S1s": {
            "loc": [0, 1],
            "cas": [3, 3],
            "task": "0 0 1 1 0 2 1 1 0 3 1 1 2 0 1 1 2 2 1 1 2 3 1 1",
            "type": "S1s",
        },
        # "S2p": {
        #     "loc": [4, 7],
        #     "cas": [3, 3],
        #     "task": "0 0 3 1 1 1 0 1 2 1 1 0 2 2 1 1 0 3 2 1 1 2 0 3 1 1 1 2 1 2 1 1 2 2 2 1 1 2 3 2 1 1",
        #     "type": "S2p",
        # },
    },
    "Cl2": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 3],
        },
        "Cl1s": {
            "loc": [0, 2],
            "cas": [4, 3],
            "task": "0 0 1 1 0 5 1 1 2 0 1 1 2 5 1 1",
            "type": "Cl1s",
        },
        # "Cl2s": {
        #     "loc": [2, 4],
        #     "cas": [4, 3],
        #     "task": "0 0 1 1 0 5 1 1 2 0 1 1 2 5 1 1",
        #     "type": "Cl2s",
        # },
        # "Cl2p": {
        #     "loc": [4, 10],
        #     "cas": [4, 3],
        #     "task": "0 0 1 1 0 2 1 1 0 3 1 1 0 5 1 1 0 6 1 1 0 7 1 1 2 0 1 1 2 2 1 1 2 3 1 1 2 5 1 1 2 6 1 1 2 7 1 1",
        #     "type": "Cl2p",
        # },
    },
    "NNO": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [3, 4],
        },
        "N1s": {
            "loc": [1, 3],
            "cas": [3, 4],
            "task": "0 0 4 1 1 1 1 0 2 2 1 1 0 3 2 1 1",
            "type": "N1s",
        },
    },
    "O3": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [1, 2],
        },
        "O1s": {
            "loc": [0, 3],
            "cas": [1, 4],
            "task": "0 0 6 1 1 1 1 1 1 0 1 3 1 1 1",
            "type": "O1s",
        },
    },
}

Ion_Ene = {
    "H3COH": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [1, 2],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "C1s",
        },
        "O1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "O1s",
        },
    },
    "CO": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [1, 2],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "C1s",
        },
        "O1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "O1s",
        },
    },
    "H2CO": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [1, 2],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "C1s",
        },
        "O1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "O1s",
        },
    },
    "CH4": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "C1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "C1s",
        },
    },
    "NH3": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "N1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "N1s",
        },
    },
    "H2O": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "O1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "O1s",
        },
    },
    "HF": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [4, 4],
        },
        "F1s": {
            "loc": [0, 1],
            "cas": [4, 4],
            "task": "1 0 1 1",
            "type": "F1s",
        },
    },
    "CH3CN": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [2, 3],
        },
        "C1s": {
            "loc": [1, 3],
            "cas": [2, 3],
            "task": "1 0 2 1 1",
            "type": "C1s",
        },
        "N1s": {
            "loc": [0, 1],
            "cas": [2, 3],
            "task": "1 0 1 1",
            "type": "N1s",
        },
    },
    "HCN": {
        "gt": {
            "task": "0 0 1 1",
            "cas": [2, 3],
        },
        "C1s": {
            "loc": [1, 2],
            "cas": [2, 3],
            "task": "1 0 1 1",
            "type": "C1s",
        },
        "N1s": {
            "loc": [0, 1],
            "cas": [2, 3],
            "task": "1 0 1 1",
            "type": "N1s",
        },
    },
}


def _Generate_InputFile_SiCI(File,
                             Segment,
                             nelec_val,
                             rotatemo=0,  # HF orbitals!
                             cmin=0.0,
                             perturbation=1,
                             dumprdm=0,
                             relative=0,
                             Task=None,
                             tol=1e-8,
                             selection=1
                             ):
    """Generate the input file
    """
    with open(File, "w") as f:
        f.write("nsegment=%s\n" % Segment)
        f.write("nvalelec=%d\n" % nelec_val)
        f.write("rotatemo=%d\n" % rotatemo)
        f.write("cmin=%s\n" % cmin)
        f.write("perturbation=%d 0\n" % perturbation)
        f.write("dumprdm=%d\n" % dumprdm)
        f.write("relative=%d\n" % relative)
        f.write("task=%s\n" % Task)
        f.write("etol=%e\n" % tol)
        f.write("selection=%d\n" % selection)


CMIN = "1e-4"

if __name__ == "__main__":

    #### test the orb ####

    for mole, geometry in Mole_Geometry.items():
        continue
        mol = pyscf.gto.M(
            verbose=1,
            atom=geometry,
            basis=basis_test,
            spin=0,
            charge=0,
            symmetry=True,
        )
        mol.build()

        print(mole, " has symmetry ", mol.groupname)

        if mol.groupname == "Dooh":
            mol.symmetry = "D2h"
            mol.build()
        elif mol.groupname == "Coov":
            mol.symmetry = "C2v"
            mol.build()

        scf = pyscf.scf.RHF(mol).x2c()
        scf.kernel()

        mo_energy = scf.mo_energy
        # print(mo_energy)

        # label the symmetry of each orbital

        mo_symm = pyscf.symm.label_orb_symm(
            mol, mol.irrep_id, mol.symm_orb, scf.mo_coeff)

        for x, y in zip(mo_energy, mo_symm):
            print("%15.8f %2d" % (x, y))

    # exit(1)

    ### test the vert ext ###

    for mole, info in Vert_Ext.items():

        for basis in BASIS:
            mol = pyscf.gto.M(
                verbose=4,
                atom=Mole_Geometry[mole],
                basis=basis,
                spin=0,
                charge=0,
                symmetry=True,
            )
            if mole == "O2":
                mol.spin = 2
            mol.build()

            if mol.groupname == "Dooh":
                mol.symmetry = "D2h"
                mol.build()
            elif mol.groupname == "Coov":
                mol.symmetry = "C2v"
                mol.build()

            scf = pyscf.scf.RHF(mol).x2c()
            scf.kernel()

            for task, subinfo in info.items():
                if task == "gt":

                    _dump_FCIDUMP(mol, scf, "FCIDUMP_%s_gt_%s_Vert" %
                                  (mole, basis))

                    # generate the input file

                    _Generate_InputFile_SiCI(
                        "input_%s_gt_%s_Vert" % (mole, basis),
                        _get_gt_nsegment(mol, subinfo["cas"]),
                        subinfo["cas"][0] * 2,
                        Task=subinfo["task"],
                        cmin=CMIN,
                    )

                else:

                    res = Util_CoreExt._dump_CoreExt_FCIDUMP(
                        mol, scf, [subinfo], "FCIDUMP_Vert_%s_%s_" % (mole, basis))
                    nsegment = res[0]["nsegment"]

                    ncore = subinfo["loc"][1] - subinfo["loc"][0]

                    _Generate_InputFile_SiCI(
                        "input_%s_%s_%s_Vert" % (
                            mole, subinfo["type"], basis),
                        nsegment,
                        subinfo["cas"][0] * 2 + ncore * 2,
                        Task=subinfo["task"],
                        cmin=CMIN,
                    )

    ### test the Ion_Ene ###

    for mole, info in Ion_Ene.items():

        for basis in BASIS:
            mol = pyscf.gto.M(
                verbose=4,
                atom=Mole_Geometry[mole],
                basis=basis,
                spin=0,
                charge=0,
                symmetry=True,
            )
            mol.build()

            if mol.groupname == "Dooh":
                mol.symmetry = "D2h"
                mol.build()
            elif mol.groupname == "Coov":
                mol.symmetry = "C2v"
                mol.build()

            scf = pyscf.scf.RHF(mol).x2c()
            scf.kernel()

            for task, subinfo in info.items():
                if task == "gt":

                    _dump_FCIDUMP(mol, scf, "FCIDUMP_%s_gt_%s_Ion" %
                                  (mole, basis))

                    # generate the input file

                    _Generate_InputFile_SiCI(
                        "input_%s_gt_%s_Ion" % (mole, basis),
                        _get_gt_nsegment(mol, subinfo["cas"]),
                        subinfo["cas"][0] * 2,
                        Task=subinfo["task"],
                        cmin=CMIN,
                    )

                else:
                    # print("subinfo: ", subinfo)

                    res = Util_CoreExt._dump_CoreExt_FCIDUMP(
                        mol, scf, [subinfo], "FCIDUMP_Ion_%s_%s_" % (mole, basis))
                    nsegment = res[0]["nsegment"]

                    ncore = subinfo["loc"][1] - subinfo["loc"][0]

                    _Generate_InputFile_SiCI(
                        "input_%s_%s_%s_Ion" % (mole, subinfo["type"], basis),
                        nsegment,
                        subinfo["cas"][0] * 2 + ncore * 2 - 1,
                        Task=subinfo["task"],
                        cmin=CMIN,
                    )
