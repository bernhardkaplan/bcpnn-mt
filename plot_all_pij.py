import os
import simulation_parameters
import pylab
import numpy as np
import utils
import sys
import matplotlib

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

if len(sys.argv) > 1:
#    fn = sys.argv[1]
    iteration = int(sys.argv[1])
else:
#    fn = raw_input('Filename to plot:\n')
#    fn = params['bcpnntrace_folder'] + 'all_pij.dat'
    iteration = 0


fn = 'Abstract/TrainingResults_%d/all_wij_%d.dat'  % (iteration, iteration)
d = np.loadtxt(fn)

n_cells = params['n_exc']
wij_matrix = np.zeros((n_cells, n_cells))
bias_matrix = np.zeros(n_cells)


w_min = -6.
w_max =  6
norm = matplotlib.mpl.colors.Normalize(vmin=w_min, vmax=w_max)

#pre_id	    post_id	    pij[-1]	    w_ij[-1]	bias
for line in xrange(d[:, 0].size):
    i, j, pij_, wij, bias_j = d[line, :]
#    print 'debug i, j, w_ij, bias_j', i, j, wij, bias_j
    bias_matrix[j] = bias_j
    wij_matrix[i, j] = wij
#    wij_matrix[j, i] = wij

plot_matrix = wij_matrix + bias_matrix

#plot_matrix2 = wij_matrix + bias_matrix
#plot_matrix = plot_matrix2[:-1, :-1]

#fig = pylab.figure()
#ax = fig.add_subplot(111)
#cax = ax.pcolormesh(plot_matrix)
#ax.set_title('weights + bias')
#ax.set_ylim(0, n_cells)
#ax.set_xlim(0, n_cells)
#pylab.colorbar(cax)

fig = pylab.figure()
ax = fig.add_subplot(111)
cax_weights= ax.pcolormesh(wij_matrix, cmap='seismic', norm=norm)
ax.set_title('Weights after iteration %d' % iteration)
ax.set_ylim(0, n_cells)
ax.set_xlim(0, n_cells)
pylab.colorbar(cax_weights)

#fig = pylab.figure()
#ax = fig.add_subplot(111)
#cax_bias = ax.pcolormesh(bias_matrix)
#ax.set_title('bias')
#ax.set_ylim(0, n_cells)
#ax.set_xlim(0, n_cells)
#pylab.colorbar(cax_bias)

output_fn = params['folder_name'] + 'TrainingResults_%d/' % iteration + 'wij_matrix_%d.dat' % (iteration)
print 'Saving to:', output_fn
np.savetxt(output_fn, wij_matrix)

output_fn_bias = params['folder_name'] + 'TrainingResults_%d/' % iteration + 'bias_%d.dat' % (iteration)
np.savetxt(output_fn_bias, bias_matrix)
output_fn = params['figures_folder'] + 'weight_matrix_%d.png' % (iteration)
print 'Saving figure to:', output_fn
pylab.savefig(output_fn)
#pylab.show()
