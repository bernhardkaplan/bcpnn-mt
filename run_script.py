import numpy as np
import os

#tau_zis = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, \
#        1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
tau_zis = np.arange(3100, 5100, 100)

script_name = 'toy_experiment.py'

for tau_zi in tau_zis:
    command = 'python %s %d' % (script_name, tau_zi)
    os.system(command)
