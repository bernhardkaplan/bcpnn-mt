import os
import numpy as np
sn = "NetworkSimModule.py"


for ws in [5e-5, 1e-4, 5e-4, 1e-3]:
    os.system('mpirun -np 8 python  %s %f'  % (sn, ws))

