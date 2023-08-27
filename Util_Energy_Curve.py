import pyscf
import numpy
import Util_Math
import pyscf_util


def _Generate_InputFile_SiCI(inputfilename,
                             Segment,
                             nelec_val,
                             rotatemo,
                             cmin,
                             perturbation,
                             dumprdm,
                             relative,
                             Task,
                             inputocfg,
                             relativeAddSingle=0,
                             tol=1e-5,
                             selection=1,
                             doublegroup="",
                             nstaterela=""):
    inputfile = open(inputfilename, "w")
    inputfile.write("nsegment=%s\n" % (Segment))
    inputfile.write(
        "nvalelec=%d\nETOL=%e\nCMIN=%s\nROTATEMO=%d\n" %
        (nelec_val, tol, cmin, rotatemo))
    inputfile.write("perturbation=%d 0\ndumprdm=%d\nrelative=%d %d\n" %
                    (perturbation, dumprdm, relative, relativeAddSingle))
    if Task is not None:
        inputfile.write("task=%s\n" % (Task))
    inputfile.write("inputcfg=%s\n" % (inputocfg))
    inputfile.write("selection=%s\n" % (selection))
    inputfile.write("print=11\n")

    if doublegroup != "":
        inputfile.write("doublegroup=%s\n" % (doublegroup))
    if nstaterela != "":
        inputfile.write("nstaterela=%s\n" % (nstaterela))
    inputfile.close()


# SEARCH CASSCF

iCISCF_Init_mode = ["CASSCF",
                    "OtherBasis",
                    "Neighborhood",
                    'Default'
                    ]


DEFAULT_CASSCF_CONV_TOL = 1e-8
DEFAULT_CASSCF_CONV_TOL_GRAD = 1e-4
DEFAULT_CASSCF_MX_MACRO = 128

DEFAULT_ICISCF_CONV_TOL = 1e-5
DEFAULT_ICISCF_CONV_TOL_GRAD = 2e-2


def _get_mx_macro(id):
    res = [128, 48, 32]
    if id < 3:
        res[id]
    else:
        return 32


def iCISCF_scan(AtomA: str,
                AtomB: str,
                BondLength: list,
                Mol_config,
                SCF_config,
                CASSCF_config=None,
                iCISCF_config=None):

    Res = []  # bondlength, energy, mo_coeff

    for id, bondlength in enumerate(BondLength):

        # construct Mol

        Mol = pyscf.gto.Mole()
        Mol.atom = '''
        %s     0.0000      0.0000  %f
        %s     0.0000      0.0000  -%f
        ''' % (AtomA, bondlength / 2, AtomB, bondlength/2)
        Mol.unit = 'angstorm'

        if 'symmetry' not in Mol_config.keys():
            Mol.symmetry = True
        else:
            Mol.symmetry = Mol_config['symmetry']

        if 'spin' not in Mol_config.keys():
            Mol.spin = None
        else:
            Mol.spin = Mol_config['spin']

        if 'basis' not in Mol_config.keys():
            raise RuntimeError
        Mol.basis = Mol_config['basis']

        if 'charge' not in Mol_config.keys():
            Mol.charge = 0
        else:
            Mol.charge = Mol_config['charge']

        if 'verbose' in Mol_config.keys():
            Mol.verbose = Mol_config['verbose']

        Mol.build()

        # RUN SCF

        sfx1e = False
        do_pyscf_analysis = False
        newton = False

        if 'sfx1e' in SCF_config.keys():
            sfx1e = SCF_config['sfx1e']

        if 'newton' in SCF_config.keys():
            newton = SCF_config['newton']

        SCF = pyscf_util.RUN_SCF(Mol, sfx1e, do_pyscf_analysis, newton)

        # RUN MCSCF

        casscf = None

        if CASSCF_config is not None:

            # config

            cas_list = None
            mc_conv_tol = DEFAULT_CASSCF_CONV_TOL
            mc_conv_tol_grad = DEFAULT_CASSCF_CONV_TOL_GRAD
            mc_max_macro = DEFAULT_CASSCF_MX_MACRO
            state = None
            nelecas = None
            ncas = None

            # check config

            if 'cas_list' in CASSCF_config.keys():
                cas_list = CASSCF_config['cas_list']

            if 'mc_conv_tol' in CASSCF_config.keys():
                mc_conv_tol = CASSCF_config['mc_conv_tol']

            if 'mc_conv_tol_grad' in CASSCF_config.keys():
                mc_conv_tol_grad = CASSCF_config['mc_conv_tol_grad']

            if 'mc_max_macro' in CASSCF_config.keys():
                mc_max_macro = CASSCF_config['mc_max_macro']

            if 'state' not in CASSCF_config.keys():
                raise RuntimeError
            state = CASSCF_config['state']

            if 'nelecas' not in CASSCF_config.keys():
                raise RuntimeError
            nelecas = CASSCF_config['nelecas']

            if 'ncas' not in CASSCF_config.keys():
                raise RuntimeError
            ncas = CASSCF_config['ncas']

            # run config

            casscf = pyscf_util.RUN_MCSCF(
                Mol, SCF, nelecas, ncas, cas_list, mc_conv_tol, mc_conv_tol_grad, mc_max_macro, False, state, None)

        if iCISCF_config is not None:

            pass

            # check whether

            cmin_schedule = None
            input_mode = 'Default'
            mc_conv_tol = DEFAULT_ICISCF_CONV_TOL
            mc_conv_tol_grad = DEFAULT_ICISCF_CONV_TOL_GRAD

            # process mo_init

            mo_init = None

            # RUN iCISCF

            iciscf = None

            for id_cmin, cmin in enumerate(cmin_schedule):
                pass

            ResTmp = {}
            ResTmp['bondlength'] = bondlength
            ResTmp['etot'] = iciscf.e_tot
            ResTmp['mo_coeff'] = iciscf.mo_coeff
            Res.append(ResTmp)

        else:

            if casscf is not None:
                ResTmp = {}
                ResTmp['bondlength'] = bondlength
                ResTmp['etot'] = casscf.e_tot
                ResTmp['mo_coeff'] = casscf.mo_coeff
                Res.append(ResTmp)
            else:
                ResTmp = {}
                ResTmp['bondlength'] = bondlength
                ResTmp['etot'] = SCF.energy_tot()
                ResTmp['mo_coeff'] = SCF.mo_coeff
                Res.append(ResTmp)

    return Res
