import os
import numpy as np

for sweep_parameter in [0.05, 0.10, 0.15]:
    os.system('mpirun -np 8 python NetworkSimModule.py %f'  % (sweep_parameter))
    # on 1 core
#    os.system('python NetworkSimModule.py %f'  % (sweep_parameter))

