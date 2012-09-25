import os
import numpy as np

w_sigma_x_start = 0.05
w_sigma_x_stop = 0.8
w_sigma_x_step = 0.1

#w_sigma_v = 0.20

w_sigma_range = np.arange(w_sigma_x_start, w_sigma_x_stop, w_sigma_x_step)


os.system('python prepare_tuning_prop.py')
for i_, w_sigma_x in enumerate(w_sigma_range):
    w_sigma_v = w_sigma_x
    os.system('mpirun -np 8 python prepare_connections.py %f %f' % (w_sigma_x, w_sigma_v))
    os.system('python analyse_connectivity.py %f %f %d' % (w_sigma_x, w_sigma_v, i_))
