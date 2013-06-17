import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import pylab
import simulation_parameters
import sys
from scipy.optimize import leastsq
import os
import utils

# parse command line arguments (conn_type and folder
conn_types = ['ee', 'ei', 'ie', 'ii']
conn_type = None
params = None
for arg in sys.argv:
    try: 
        param_fn = arg
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)

    except:
        params = None

    print arg
    if arg in conn_types:
        conn_type = arg

# if not set yet, set to defaults
if params == None:
    # load simulation parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
if conn_type == None:
    conn_type = 'ee'


def get_incoming_connection_numbers(conn_data, n_tgt):
    n_in = np.zeros(n_tgt)
    for i in xrange(conn_data[:, 0].size):
        src, tgt, w, delay = conn_data[i, :4]
        n_in[tgt] += 1

    return n_in


fn = params['merged_conn_list_%s' % conn_type] 
if not os.path.exists(fn):
    os.system('python merge_connlists.py %s' % params['folder_name'])
output_fn = params['figures_folder'] + 'weights_and_delays_%s.png' % (conn_type)

d = np.loadtxt(fn)
print 'debug', d.shape, fn

(n_src, n_tgt, syn_type) = utils.resolve_src_tgt(conn_type, params)
n_in = get_incoming_connection_numbers(d, n_tgt)
string = 'n_%s = %.2f +- %.2f' % (conn_type, n_in.mean(), n_in.std())
string += '\nn_%s_min = %.2f' % (conn_type, n_in.min())
string += '\nn_%s_max = %.2f' % (conn_type, n_in.max())
out_fn = params['data_folder'] + 'nconn_%s.txt' % (conn_type)
f = open(out_fn, 'w')
f.write(string)
f.flush()
f.close()
print 'Writing to:', out_fn 
print string

weights = d[:, 2]
delays = d[:, 3]
w_mean, w_std = weights.mean(), weights.std()
d_mean, d_std = delays.mean(), delays.std()
n_weights = weights.size
n_possible = params['n_exc']**2

n_bins = 50
n_w, bins_w = np.histogram(weights, bins=n_bins, normed=False)
#fig = pylab.figure()
#ax1 = fig.add_subplot(111)
#pylab.hist(delays, bins=n_bins)
#pylab.show()
#n_w = n_w / float(n_w.sum())

print "bins_w", bins_w, '\nn_w', n_w
n_d, bins_d = np.histogram(delays, bins=n_bins, normed=False)
#n_d = n_d / float(n_d.sum())
print "bins_d", bins_d, '\nn_d', n_d

def residuals_exp_dist(p, y, x):
    return y - eval_exp_dist(x, p)

def eval_exp_dist(x, p):
    return p[0] * np.exp(- x * p[0])
#    return p[0] * np.exp(- x / p[1])

def residuals_delay_dist(p, y, x):
    return y - eval_delay_dist(x, p)

def eval_delay_dist(x, p):
    return x * p[1] * np.exp(- x / p[0])

def residuals_gaussian(p, y, x):
    return y - eval_gaussian(x, p)

def eval_gaussian(x, p):
    return 1. / (np.sqrt(2 * np.pi) * p[1]) * np.exp(-(x - p[0])**2 / (2 *p[1]**2))



print "Fitting function to weight distribution"
guess_params = (5e-2) # (w[0], w_tau)
#guess_params = (0.5, 5e-4) # (w[0], w_tau)
#opt_params = leastsq(residuals_exp_dist, guess_params, args=(n_w, bins_w[:-1]), maxfev=1000)
guess_params = (0.001, 0.001)
opt_params = leastsq(residuals_gaussian, guess_params, args=(n_w, bins_w[:-1]), maxfev=1000)[0]
#opt_w0 = opt_params[0][0]
#print "Optimal parameters: w_0 %.2e w_tau %.2e" % (opt_w0, opt_wtau)
#opt_wtau= opt_params[0]#[0]
opt_wmean= opt_params[0]#[0]
opt_wsigma= opt_params[1]#[0]

p_ee = float(n_weights) / n_possible
print 'P_ee: %.3e' % p_ee
print 'w_min: %.2e w_max %.2e w_mean: %.2e  w_std: %.2e' % (weights.min(), weights.max(), weights.mean(), weights.std())
print 'd_min: %.2e d_max %.2e d_mean: %.2e  d_std: %.2e' % (delays.min(), delays.max(), delays.mean(), delays.std())
#print "Optimal parameters: w_lambda %.5e" % (opt_wtau)
print "Optimal parameters: w_mu %.2e w_sigma = %.2e" % (opt_wmean, opt_wsigma)

print "Fitting function to delay distribution"
guess_params = (5., 10.)
opt_params_delay = leastsq(residuals_delay_dist, guess_params, args=(n_d, bins_d[:-1]), maxfev=1000)
print 'Opt delay params:', opt_params_delay
opt_d0 = opt_params_delay[0][0]
opt_d1 = opt_params_delay[0][1]

print "Plotting ..."
fig = pylab.figure()
ax1 = fig.add_subplot(211)
bin_width = bins_w[1] - bins_w[0]
ax1.bar(bins_w[:-1]-.5*bin_width, n_w, width=bin_width, label='$w_{mean} = %.2e \pm %.2e$' % (w_mean, w_std))
ax1.plot(bins_w[:-1], eval_gaussian(bins_w[:-1], opt_params), 'r--', label='Fit: gaussian $\mu_{w}=%.2e \quad \sigma_{w}=%.2e$' % (opt_wmean, opt_wsigma))
#ax1.plot(bins_w[:-1], eval_exp_dist(bins_w[:-1], opt_params), 'r--', label='Fit: $(%.2e) * exp(-(%.2e) \cdot w)$' % (opt_wtau, opt_wtau))
#ax1.plot(bins_w[:-1], eval_exp_dist(bins_w[:-1], opt_params[0]), 'r--', label='Fit: $(%.1e) * exp(-w / (%.1e))$' % (opt_w0, opt_wtau))
ax1.set_xlabel('Weights')
ax1.set_ylabel('Count')
ax1.set_xlim((weights.min()-.5*bin_width, weights.max()))
title = 'Weight profile for %s connections\n$\sigma_{X(V)} = %.1f (%.1f)$' % (conn_type, params['w_sigma_x'], params['w_sigma_v'])
ax1.set_title(title)
ax1.legend()

ax2 = fig.add_subplot(212)
bin_width = bins_d[1] - bins_d[0]
ax2.bar(bins_d[:-1]-.5*bin_width, n_d, width=bin_width, label='$\delta_{mean} = %.1e \pm %.1e$' % (d_mean, d_std))
ax2.plot(bins_d[:-1], eval_delay_dist(bins_d[:-1], (opt_d0, opt_d1)), 'r--', label='Fit: $\delta \cdot exp(-\delta / (%.1e))$' % (opt_d0))
ax2.set_xlabel('Delays')
ax2.set_ylabel('Count')
ax2.set_xlim((0. - .5 * bin_width, delays.max() + 2 * bin_width))
#ax2.set_xlim((0. - .5 * bin_width, 20))
#ax2.set_xlim((delays.min()-.5*bin_width, delays.max()))
ax2.legend()


print "Saving to:", output_fn
pylab.savefig(output_fn)
#pylab.show()


