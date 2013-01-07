import os
import numpy as np

blur_x_start = 0.02
blur_x_stop = 0.5
blur_x_step = 0.02
blur_range = np.arange(blur_x_start, blur_x_stop, blur_x_step)

idx = 0
blur_v = 0.05
#for i_, blur_v in enumerate(blur_range):
for j_, blur_x in enumerate(blur_range):
    os.system('python prepare_tuning_prop.py %f %f' % (blur_x, blur_v))
    os.system('mpirun -np 8 python prepare_spike_trains.py %f %f' % (blur_x, blur_v))
    os.system('python analyse_input.py %f %f %d' % (blur_x, blur_v, idx))
    idx += 1
#        os.system('python plot_input_spiketrains.py %f %f' % (blur_x, blur_v))
