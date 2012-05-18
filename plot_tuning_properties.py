import os
import simulation_parameters
import pylab
import re
import numpy as np
import utils
#import matplotlib
#import matplotlib.patches as mpatches
#from matplotlib.collections import PatchCollection

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

fn = params['tuning_prop_means_fn']
d = np.loadtxt(fn)

n_cells = d[:, 0].size
scale = 20.
ms = 3 # markersize for scatterplots

fig = pylab.figure()
pylab.subplots_adjust(hspace=.6)
pylab.subplots_adjust(wspace=.3)

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
#ax4 = fig.add_subplot(224)

thetas = np.zeros(n_cells)
for i in xrange(n_cells):
    x, y, u, v = d[i, :]
    # calculate the color from tuning angle theta
    thetas[i] = np.arctan2(v, u)
    h = ((thetas[i] + np.pi) / (2 * np.pi)) * 360. # theta determines h, h must be [0, 360)
    l = np.sqrt(u**2 + v**2) / np.sqrt(2 * params['v_max']**2) # lightness [0, 1]
    s = 1. # saturation
    assert (0 <= h and h < 360)
    assert (0 <= l and l <= 1)
    assert (0 <= s and s <= 1)
    (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
    ax1.plot(x, y, 'o', c=(r,g,b), markersize=ms)
    # plot velocity
    ax2.plot(u, v, 'o', color='k', markersize=ms)

ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_title('Spatial receptive fields\n preferred directions color coded')

ax2.set_xlabel('$u$')
ax2.set_ylabel('$v$')
ax2.set_ylim((d[:, 3].min() * 1.05, d[:, 3].max() * 1.05))
ax2.set_xlim((d[:, 2].min() * 1.05, d[:, 2].max() * 1.05))
ax2.set_title('Receptive fields for speed')

scale = 20.
for i in xrange(n_cells):
    x, y, u, v = d[i, :]
    ax3.plot((x*scale, x*scale+u), (y*scale, y*scale+v), 'k')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$y$')
ax3.set_title('Preferred directions\n (space was scaled by factor %d)' % scale)
ax3.set_ylim((d[:, 1].min() * 1.05 * scale, d[:, 1].max() * 1.05 * scale))
ax3.set_xlim((d[:, 0].min() * 1.05 * scale, d[:, 0].max() * 1.05 * scale))


print "Saving to ... ", params['tuning_prop_fig_fn']
pylab.savefig(params['tuning_prop_fig_fn'])
#pylab.show()
