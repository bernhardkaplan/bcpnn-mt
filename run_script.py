import numpy as np
import os

#tau_zis = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, \
#        1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
#tau_zis += range(2100, 5100, 100)
tau_zis = [10, 50, 100, 200, 300, 400, 500, 750]
tau_zis += range(1000, 7500, 500)

script_name = 'toy_experiment.py'

# sweep dx and tau_zi
#for dx in np.arange(0.05, 0.75, 0.05):
#    for tau_zi in tau_zis:
#        command = 'python %s %d %f' % (script_name, tau_zi, dx)
#        os.system(command)

for tau_zi in tau_zis:
    for dv in np.arange(.1, 2., 0.2):
        for v_stim in np.arange(0.1, 2.0, 0.1):
            command = 'python %s %d %f %f' % (script_name, tau_zi, dv, v_stim)
            os.system(command)
