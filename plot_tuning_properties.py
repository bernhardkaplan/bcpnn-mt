import os
import simulation_parameters
import pylab
import re
import numpy as np
#import matplotlib
#import matplotlib.patches as mpatches
#from matplotlib.collections import PatchCollection

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

fn = params['tuning_prop_means_fn']
d = np.loadtxt(fn)

n_cells = d[:, 0].size

patches = []
fig = pylab.figure()
ax = fig.add_subplot(111)

scale = 20.
for i in xrange(n_cells):
    x, y, u, v = d[i, :]
#    lw = np.sqrt(u**2 + v**2)
    ax.plot((x*scale, x*scale+u), (y*scale, y*scale+v), 'k')


ax.set_xlabel('$RF_x$')
ax.set_ylabel('$RF_y$')
ax.set_title('Preferred directions\n space was scaled by factor %.1f' % scale)

pylab.show()
