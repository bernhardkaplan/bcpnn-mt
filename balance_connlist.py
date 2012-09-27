import numpy as np
import utils
import matplotlib
matplotlib.use('Agg')
import pylab
import sys
import os

import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tp = np.loadtxt(params['tuning_prop_means_fn'])
mp = params['motion_params']
print "Motion parameters", mp

conn_list_fn = params['merged_conn_list_ee']
if not os.path.exists(conn_list_fn):
    tmp_fn = 'delme_tmp_%d' % (np.random.randint(0, 1e7))
    cat = 'cat %s* > %s' % (params['conn_list_ee_fn_base'], tmp_fn)
    sort = 'sort -gk 2 -gk 1 %s > %s' % (tmp_fn, conn_list_fn)
    del_tmp_fn = 'rm %s' % (tmp_fn)
    print cat
    os.system(cat)
    print sort
    os.system(sort)
    print del_tmp_fn
    os.system(del_tmp_fn)

conn_list_balanced_fn = params['conn_list_ee_balanced_fn']
print "Loading connectivity data from ", conn_list_fn
conn_mat, delays = utils.convert_connlist_to_matrix(conn_list_fn, params['n_exc'])
conn_mat_balanced = np.zeros((params['n_exc'], params['n_exc']))

w_in_sum = np.zeros(params['n_exc'])
for i in xrange(params['n_exc']):
    w_in_sum[i] = conn_mat[:, i].sum()

w_in_sum_mean = w_in_sum.mean()
#balancing_factor = (w_in_sum_mean - w_in_sum) / w_in_sum_mean + w_in_sum.max() / w_in_sum_mean
balancing_factor = w_in_sum.mean() / w_in_sum
print 'Every cell gets on average a sum of %.2e +- %.2e [uS] weights, min %.2e max %.2e' % (w_in_sum.mean(), w_in_sum.std(), w_in_sum.min(), w_in_sum.max())
#print 'balancing_factor = %.2e +- %.2e' % (balancing_factor.mean(), balancing_factor.std())
#print balancing_factor

conn_list = np.loadtxt(conn_list_fn)
#output = np.zeros((conn_list[:, 0].size, 4))
output = ''
#for i in xrange(20):
for i in xrange(conn_list[:, 0].size):
    src, tgt, w, delay = conn_list[i, :]
    w_new = w * balancing_factor[tgt]
    conn_mat_balanced[src, tgt] = w_new
    if w_new > 0.:
#    output[i, 0], output[i, 1], output[i, 2], output[i, 3] = src, tgt, w_new, delay
        output += '%d\t%d\t%.4e\t%.2e\n' % (src, tgt, w_new, delay)

print 'Saving balanced conn_list to', conn_list_balanced_fn
output_file = open(conn_list_balanced_fn, 'w')
output_file.write(output)
output_file.close()
#np.savetxt(conn_list_balanced_fn, output, fmt='%d\t%d\t%.4e\t%.2e')


# analyse connectivity: make histogram of all incoming connections
n_cells = params['n_exc']
# summed incoming weights
w_in = np.zeros(n_cells)
w_in_balanced = np.zeros(n_cells)
# average weight per incoming connection
w_in_avg = np.zeros(n_cells)
w_in_avg_balanced = np.zeros(n_cells)
for i in xrange(n_cells):
    w_in[i] = conn_mat[:, i].sum()
    w_in_balanced[i] = conn_mat_balanced[:, i].sum()

    n_in = conn_mat[:, i].nonzero()[0].size
    w_in_avg[i] = conn_mat[:, i].sum() / n_in
    w_in_avg_balanced[i] = conn_mat_balanced[:, i].sum() / conn_mat_balanced[:, i].nonzero()[0].size

# plotting sums
x = np.arange(0, n_cells)
fig = pylab.figure()
# plot the distribution of 'normal' sums of incoming weights
ax = fig.add_subplot(221)
ax.bar(x, w_in)
fs = 12
ax.set_title('Sum of NORMAL \nincoming weights [uS]', fontsize=fs)
ax.set_ylabel('w_in_sum [uS]')
ax.set_xlabel('GID of exc cell')
ax.set_xlim((0, n_cells))

# plot the distribution of balanced incoming weights
ax = fig.add_subplot(223)
ax.bar(x, w_in_balanced)
ax.set_title('Sum of BALANCED \nincoming weights [uS]', fontsize=fs)
ax.set_ylabel('w_in_sum [uS]')
ax.set_xlabel('GID of exc cell')
ax.set_xlim((0, n_cells))

# plotting averages
ax = fig.add_subplot(222)
ax.bar(x, w_in_avg)
ax.set_title('Average NORMAL \nincoming connection weight [uS]', fontsize=fs)
ax.set_ylabel('average w_in [uS]')
ax.set_xlabel('GID of exc cell')
ax.set_xlim((0, n_cells))

# balanced
ax = fig.add_subplot(224)
ax.bar(x, w_in_avg_balanced)
ax.set_title('Average BALANCED \nincoming connection weight [uS]', fontsize=fs)
ax.set_ylabel('average w_in [uS]')
ax.set_xlabel('GID of exc cell')
ax.set_xlim((0, n_cells))

pylab.subplots_adjust(hspace=0.5)
#pylab.show()

output_fn = params['figures_folder'] + 'distributions_of_incoming_weights.png'
print 'Output_fig:', output_fn
pylab.savefig(output_fn)

