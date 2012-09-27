import matplotlib
matplotlib.use('Agg')
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
    fn = params['merged_conn_list_ee']
    print "Plotting default file:", fn
    if not os.path.exists(fn):
        print 'Merging files %s*' % (params['conn_list_ee_fn_base'])
        tmp_fn = 'delme_tmp_%d' % (np.random.randint(0, 1e7))
        cat = 'cat %s* > %s' % (params['conn_list_ee_fn_base'], tmp_fn)
        sort = 'sort -gk 2 -gk 1 %s > %s' % (tmp_fn, params['merged_conn_list_ee'])
        del_tmp_fn = 'rm %s' % (tmp_fn)
        print cat
        os.system(cat)
        print sort
        os.system(sort)
        print del_tmp_fn
        os.system(del_tmp_fn)

else:
    fn = sys.argv[1]

n_cells = params['n_exc']
tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])
conn_list = np.loadtxt(fn)

def map_weights_to_tuning_properties(tp, merged_conn_fn):
    n_tp_bins = 20
    tp_index = 2  # 0 - x, 1 - y, 2 - v_x, 3 - v_y
    tp_grid = np.linspace(tuning_prop[:, tp_index].min(), tuning_prop[:, tp_index[:, tp_index].max())
    w_in = np.zeros(n_tp_bins)

    for i in xrange(conn_list[:, 0].size):
        src, tgt, w, delay = conn_list[i, :]





print 'debug', fn
w, delays = utils.convert_connlist_to_matrix(fn, n_cells)
#connmat_fn = params['connections_folder'] + 'conn_mat_wsigmaxv_%.1f_%.1f.dat' % (params['w_sigma_x'], params['w_sigma_v'])
#print 'Saving connection matrix to', connmat_fn
#np.savetxt(connmat_fn, w)
#delay_fn = params['connections_folder'] + 'delay_mat_wsigmaxv_%.1f_%.1f.dat' % (params['w_sigma_x'], params['w_sigma_v'])
#print 'Saving delay matrix to', delay_fn
#np.savetxt(delay_fn, delays)


print 'Weights: min %.2e median %.2e max %.2e mean %.2e st %.2e' % (w.min(), w.max(), np.median(w), w.mean(), w.std())
fig = pylab.figure()
ax = fig.add_subplot(111)
print "plotting ...."
title = 'Connection matrix with w_sigma_x(v): %.1f (%.1f)' % (params['w_sigma_x'], params['w_sigma_v'])
ax.set_title(title)
cax = ax.pcolormesh(w)#, edgecolor='k', linewidths='1')
ax.set_ylim(0, w.shape[0])
ax.set_xlim(0, w.shape[1])
ax.set_ylabel('Target')
ax.set_xlabel('Source')
pylab.colorbar(cax)


#max_incoming_weights = np.zeros(params['n_exc'])
#for i in xrange(params['n_exc']):
#    sorted_idx = w[:, i].argsort()
#    print 'max weights', w[:, i].max(), w[sorted_idx[-6:], i]
#    print 'sorted idx', w[:, i].argmax(), sorted_idx[-3:]
#    max_incoming_weights[i] = w[:, i].max()
#    print w[:, i].max()

#count, bins = np.histogram(max_incoming_weights, bins=20)
#bin_width = bins[1] - bins[0]
#ax = fig.add_subplot(212)
#ax.bar(bins[:-1], count, width=bin_width)

output_fig = params['figures_folder'] + fn.rsplit('/')[-1].rsplit('.dat')[0] + '.png'
#output_fig = fn.rsplit('.dat')[0] + '.png'
print 'Saving fig to', output_fig
pylab.savefig(output_fig)
#pylab.show()


