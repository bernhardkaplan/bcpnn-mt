import numpy as np
import utils
import pylab
import sys

import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tp = np.loadtxt(params['tuning_prop_means_fn'])
mp = params['motion_params']
print "Motion parameters", mp

conn_list_fn = params['conn_list_ee_fn_base'] + '0.dat'
print "Loading connectivity data from ", conn_list_fn
conn_mat, delays = utils.convert_connlist_to_matrix(conn_list_fn, params['n_exc'])

gids = np.loadtxt('Testing/Parameters/gids_to_record.dat')
n_good = int(params['n_exc'] * 0.05)
good_gids = gids[0:n_good]
src_cell = int(sys.argv[1])
tgts = conn_mat[src_cell, :].nonzero()[0]
srcs = conn_mat[:, src_cell].nonzero()[0]
weights_out = conn_mat[src_cell, tgts]
weights_in = conn_mat[srcs, src_cell]
#print "Target cells:", tgts
#print "Weights out:", weights_out
#print "Source cells (projecting to gid %d):" % src_cell, srcs
#print "Weights:", weights_in

print 'src\ttgt\tw\tdelay'
for gid in gids:
    dist_to_stim, spatial_dist = utils.get_min_distance_to_stim(mp, tp[gid, :])
#    print 'gid %d dist_to_stim %.2f  t_min %.1f' % (gid, dist_to_stim, np.argmin(spatial_dist) / 100. * params['t_sim']), tp[gid, :]
    print '%d\t%d\t%.2e\t%.2f' % (src_cell, gid, conn_mat[src_cell, gid], delays[src_cell, gid])

# analyse connectivity: make histogram of all incoming connections
n_cells = params['n_exc']
w_in = np.zeros(n_cells)
w_in_avg = np.zeros(n_cells)
for i in xrange(n_cells):
    w_in[i] = conn_mat[:, i].sum()
    n_in = conn_mat[:, i].nonzero()[0].size
    w_in_avg[i] = conn_mat[:, i].sum() / n_in
w_in_mean = w_in_avg.mean()
w_in_std = w_in_avg.std()
print 'Mean value for incoming weights: %.2e +- %.2e [uS]' % (w_in_mean, w_in_std)

fig = pylab.figure()
ax = fig.add_subplot(111)
x = np.arange(0, n_cells)
container = ax.bar(x, w_in)
ax.set_xlim((0, n_cells))
fs = 18
ax.set_title('Distribution of incoming weights', fontsize=fs)
ax.set_xlabel('Excitatory cell GID', fontsize=fs)
ax.set_ylabel('Sum of all incoming weights into cell [uS]', fontsize=fs)
for i in xrange(len(container)):
    if i in good_gids:
        container[i].set_facecolor('r')
ax.legend( (container[0], container[int(good_gids[0])]), ('Normal cells', 'Well-tuned cells') )


# plot the average weight per incoming connection
fig2 = pylab.figure()
ax2 = fig2.add_subplot(111)
x = np.arange(0, n_cells)
container = ax2.bar(x, w_in_avg)
ax2.set_xlim((0, n_cells))
fs = 18
ax2.set_title('Distribution of incoming weights per connection\nw_in=%.2e +- %.2e [uS]' % (w_in_mean, w_in_std), fontsize=fs)
ax2.set_xlabel('Excitatory cell GID', fontsize=fs)
ax2.set_ylabel('Average incoming weights per connection [uS]', fontsize=fs)
for i in xrange(len(container)):
    if i in good_gids:
        container[i].set_facecolor('r')
ax2.legend( (container[0], container[int(good_gids[0])]), ('Normal cells', 'Well-tuned cells') )

pylab.show()
