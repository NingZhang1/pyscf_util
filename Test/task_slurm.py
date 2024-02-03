from ToolKitNZ.src.Slurm.tool import generate_slurm_input
# HUBBARD_2D_iCIPT2 = "/home/ningzhang/iCIPT2_CXX/bin/Hubbard_2D.exe"
import os

################## task driver ##################

__FILENAME__ = "Task_%d.sh"
__JOB_NAME__ = "C_Test"
__OUTPUT_NAME__ = "C_Test.out"
__ERROR_NAME__ = "C_Test.err"
__COMMOND__ = "python3 -u /home/ningzhang/GitHub_Repo/pyscf_util/Test/Cr2.py"

time = "5-0:00:00"
partition = "serial,parallel"

ID = 0

generate_slurm_input(
    __FILENAME__ % (ID),
    __JOB_NAME__,
    __OUTPUT_NAME__,
    errorname=__ERROR_NAME__,
    partition=partition,
    time=time,
    commond=__COMMOND__
)

for i in range(0, 1):
    os.system("sbatch %s " % (__FILENAME__ % (i)))
