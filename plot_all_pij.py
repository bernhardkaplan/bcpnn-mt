import os
import simulation_parameters
import pylab
import numpy as np
import utils
import sys

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
#    fn = raw_input('Filename to plot:\n')
    fn = params['bcpnntrace_folder'] + 'all_pij.dat'

d = np.loadtxt(fn)

n_cells = params['n_exc']
wij_matrix = np.zeros((n_cells, n_cells))
bias_matrix = np.zeros((n_cells, n_cells))

#pre_id	    post_id	    pij[-1]	    w_ij[-1]	bias
for line in xrange(d[:, 0].size):
    i, j, pij_, wij, bias_j = d[line, :]
#    print 'debug i, j, w_ij, bias_j', i, j, wij, bias_j
    bias_matrix[:, j] = bias_j
    wij_matrix[i, j] = wij
    wij_matrix[j, i] = wij

print 'debug bias[:, 0]', bias_matrix[:, 0]
plot_matrix = wij_matrix + bias_matrix

#plot_matrix2 = wij_matrix + bias_matrix
#plot_matrix = plot_matrix2[:-1, :-1]

fig = pylab.figure()
ax = fig.add_subplot(111)

cax = ax.pcolormesh(plot_matrix)
ax.set_title('weights + bias')
ax.set_ylim(0, n_cells)
ax.set_xlim(0, n_cells)
pylab.colorbar(cax)

fig = pylab.figure()
ax = fig.add_subplot(111)
cax_weights= ax.pcolormesh(wij_matrix)
ax.set_title('weights')
ax.set_ylim(0, n_cells)
ax.set_xlim(0, n_cells)
pylab.colorbar(cax_weights)

fig = pylab.figure()
ax = fig.add_subplot(111)
cax_bias = ax.pcolormesh(bias_matrix)
ax.set_title('bias')
ax.set_ylim(0, n_cells)
ax.set_xlim(0, n_cells)
pylab.colorbar(cax_bias)

print 'Saving to:', params['weight_matrix_abstract'], '\n\t', params['bias_matrix_abstract']
np.savetxt(params['weight_matrix_abstract'], wij_matrix)
np.savetxt(params['bias_matrix_abstract'], bias_matrix)

pylab.show()
