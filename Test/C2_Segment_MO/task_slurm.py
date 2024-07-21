from ToolKitNZ.src.Slurm.tool import generate_slurm_input
import os

################## task driver ##################

__FILENAME__ = "Task_%d.sh"
__JOB_NAME__ = "void"
__OUTPUT_NAME__ = "Hmat_over_LS_Driver.out"
__ERROR_NAME__ = "Hmat_over_LS_Driver.err"
__COMMOND__ = "python3 -u ./Hmat_over_LS_Driver.py"

time = "1-0:00:00"
partition = "serial,parallel"

ID = 0

generate_slurm_input(
    __FILENAME__ % (ID),
    __JOB_NAME__,
    __OUTPUT_NAME__,
    errorname=__ERROR_NAME__,
    partition=partition,
    time=time,
    # memory=500,
    commond=__COMMOND__
)

for i in range(0, 1):
    os.system("sbatch %s " % (__FILENAME__ % (i)))
