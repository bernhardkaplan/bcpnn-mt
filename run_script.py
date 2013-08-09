import numpy as np
import os

#tau_zis = [10, 50, 100, 200, 300, 400, 500, 750]
#tau_zis += range(1000, 7500, 500)

tau_zis = [10, 100, 250, 500, 1000, 2000, 5000]

script_name = 'toy_experiment.py'

# sweep dx and tau_zi
#for dx in np.arange(0.05, 0.75, 0.05):
#    for tau_zi in tau_zis:
#        command = 'python %s %d %f' % (script_name, tau_zi, dx)
#        os.system(command)
#dvs = np.arange(.0, 1., 0.2)
dvs = np.array([1])

it_cnt = 0 
#v_stims = np.arange(0.2, 2.0, 0.2)
v_stims = np.arange(0.1, 2.0, 0.3)
for v_stim in v_stims:
    for tau_zi in tau_zis:
        command = 'python %s %d %f' % (script_name, tau_zi, v_stim)
        os.system(command)
        print '\n\nIteration: %d / %d\n' % (it_cnt, len(tau_zis) * dvs.size * v_stims.size)
#        print '\n\nIteration: %d / %d\n' % (it_cnt, len(tau_zis) * dvs.size * v_stims.size)
        it_cnt += 1
