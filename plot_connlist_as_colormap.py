import pylab
import numpy as np
import sys
import os
import utils
import simulation_parameters

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
sim_cnt = 0
if (len(sys.argv) < 2):
    fn = params['conn_list_ee_fn_base'] + '%d.dat' % sim_cnt
    print "Plotting default file:", fn
else:
    fn = sys.argv[1]

n_cells = params['n_exc']
w, delays = utils.convert_connlist_to_matrix(fn, n_cells)
print 'Weights: min %.2e median %.2e max %.2e mean %.2e st %.2e' % (w.min(), w.max(), np.median(w), w.mean(), w.std())
fig = pylab.figure()
ax = fig.add_subplot(111)
print "plotting ...."
title = 'Connection matrix with w_sigma_x(v): %.1f (%.1f)' % (params['w_sigma_x'], params['w_sigma_v'])
ax.set_title(title)
cax = ax.pcolormesh(w)#, edgecolor='k', linewidths='1')
ax.set_ylim(0, w.shape[0])
ax.set_xlim(0, w.shape[1])
pylab.colorbar(cax)
output_fig = params['figures_folder'] + 'conn_mat_wsigmaxv_%.1f_%.1f.png' % (params['w_sigma_x'], params['w_sigma_v'])
print 'Saving fig to', output_fig
pylab.savefig(output_fig)
pylab.show()
