import numpy as np
import utils
import pylab
import sys
import re
import os


import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

params['w_sigma_x'], params['w_sigma_v'] = float(sys.argv[1]), float(sys.argv[2])
file_count = int(sys.argv[3])
print 'w_sigma', params['w_sigma_x'], params['w_sigma_v']

fn = params['conn_list_ee_conv_constr_fn_base'] + 'merged.dat'

conn_list = np.loadtxt(fn)
w = conn_list[:, 2]

#M, delays = utils.convert_connlist_to_matrix(fn, params['n_exc'])
#w_in = np.zeros(params['n_exc'])
#w_out = np.zeros(params['n_exc'])
#for i in xrange(params['n_exc']):
#    w_in[i] = M[:, i].sum()
#    idx = M[i, :].nonzero()[0]
#    w_out[i] =  M[i, idx].mean()
#print 'w_out_mean_all', w_out.mean()

n_bins = 100
n, bins = np.histogram(w, bins=n_bins)

w_min, w_max, w_mean, w_std, w_median = w.min(), w.max(), w.mean(), w.std(), np.median(w)
label_txt = 'w_ee min, max = %.2e %.2e\nmean, std = %.2e, %.2e\nmedian=%.2e   w_sum=%.2e\nmax count=%d for w[%d]=(%.1e-%.1e)' % (w_min, w_max, w_mean, w_std, w_median, w.sum(), np.max(n), np.argmax(n), bins[np.argmax(n)], bins[np.argmax(n)+1])
print 'Info:', label_txt

fig = pylab.figure()
ax1 = fig.add_subplot(111)
bar = ax1.bar(bins[:-1], n, width=bins[1]-bins[0])


#x = np.arange(n)

#ax1.set_xlabel('Cells sorted by num input spikes')
#ax1.set_ylabel('Number of input spikes')
#ax1.set_xlim((params['w_min'], params['w_max']* 1.02))
#ax1.set_xlim((w_min, w_max * 1.02))
#ax1.set_ylim((0, 17000))

title = 'Distribution of all weights\nInput parameters:\n w_sigma_x(v)=%.1e (%.1e)\nn_exc=%d n_inh=%d' % (params['w_sigma_x'], params['w_sigma_v'], params['n_exc'], params['n_inh'])
ax1.set_title(title)
pylab.subplots_adjust(top=0.8)

(text_pos_x, text_pos_y) = ax1.get_xlim()[1] * 0.45, ax1.get_ylim()[1] * 0.75
pylab.text(text_pos_x, text_pos_y, label_txt, bbox=dict(pad=5.0, ec="k", fc="none"))


#output_fig = 'Figures_WsigmaSweepV/' + 'fig_wsigmaXV%.1e_%.1e.png' % (params['w_sigma_x'], params['w_sigma_v'])
#print 'Saving to:', output_fig
#pylab.savefig(output_fig)

output_fig = 'Figures_WsigmaSweep_TransformedBound/%d.png' % (file_count)
print 'Saving to:', output_fig
pylab.savefig(output_fig)
#pylab.show()
