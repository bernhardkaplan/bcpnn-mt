# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import simulation_parameters
import utils
import numpy as np
import set_tuning_properties as stp
GP = simulation_parameters.parameter_storage()
params = GP.params
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pylab
# <codecell>

n_rf_x_log = params['n_rf_x'] - params['n_rf_x_fovea']
xpos_log = stp.get_xpos_log_distr(params, n_x=n_rf_x_log, x_min=0.1, x_max=0.45)
#print 'xpos log\n', xpos_log, '\n', xpos_log.size
#idx_half = n_rf_x_log * .5
# lower_half = xpos_log[:idx_half]
#rf_sizes = lower_half[1:] - lower_half[:-1]
# lower_half + rf_sizes

# <codecell>

xpos_const = np.linspace(.5 - params['x_min_tp'], .5 + params['x_min_tp'], params['n_rf_x_fovea'])
#print 'xpos_const\n', xpos_const
xpos = np.zeros(params['n_rf_x'])
xpos[:params['n_rf_x_log'] / 2] = xpos_log[:params['n_rf_x_log'] / 2]
xpos[-params['n_rf_x_log'] / 2:] = xpos_log[params['n_rf_x_log'] / 2:]
xpos[params['n_rf_x_log'] / 2: params['n_rf_x_log'] / 2 + params['n_rf_x_fovea']]= xpos_const
#print xpos
rf_sizes_x = np.zeros(params['n_rf_x'])
xpos_diff = xpos[1:] - xpos[:-1]
n_x = xpos.size
xpos_diff_lower = xpos_diff[:n_x/2 - 1]
xpos_diff_upper = list(xpos_diff_lower)
xpos_diff_upper.reverse()
print 'xpos_diff_lower', xpos_diff_lower
print 'xpos_diff_upper', xpos_diff_upper
print 'nx', n_x
rf_sizes_x[:n_x / 2 - 1] = xpos_diff_lower
rf_sizes_x[n_x / 2 - 1] = xpos_diff_lower[-1]
rf_sizes_x[n_x / 2] = xpos_diff_upper[0]
rf_sizes_x[n_x / 2 + 1:] = xpos_diff_upper
rf_sizes_x[-1] = rf_sizes_x[0]
for i_ in xrange(rf_sizes_x.size):
    print '%.3f\t%.3f\t%.2f' % (xpos[i_], rf_sizes_x[i_], xpos[i_] + rf_sizes_x[i_] * .5)

# print rf_sizes

# <codecell>

n_v = params['n_v']
v_rho_half = np.logspace(np.log(params['v_min_tp']) / np.log(params['log_scale']),\
                         np.log(params['v_max_tp']) / np.log(params['log_scale']), num=n_v/2,
                            endpoint=True, base=params['log_scale'])
print 'v_rho_half:', v_rho_half
v_rho_neg = list(-1. * v_rho_half)
v_rho_neg.reverse()
v_rho = np.zeros(n_v)
v_rho[:n_v/2] = v_rho_neg
v_rho[n_v/2:] = v_rho_half
print 'v_rho', v_rho
rf_sizes_v = np.zeros(n_v)
v_diff = np.zeros(n_v)
#rf_sizes_v[:n_v/2 + 1] = v_rho[1:] - v_rho[:n_v/2 + 2]
v_diff[:-1] = v_rho[1:] - v_rho[:-1]
#v_diff[-1] = rf_sizes_v[0]
print 'v_diff', v_diff
v_diff_lower = v_diff[:n_v/2 - 1]
v_diff_upper = list(v_diff_lower)
v_diff_upper.reverse()
print 'v_diff_lower', v_diff_lower
print 'v_diff_upper', v_diff_upper
#rf_sizes_v[:n_v/2 - 1] = v_diff[:n_v/2 - 1]
#rf_sizes_v[n_v/2-1] = v_diff[n_v/2 - 2]

rf_sizes_v[:n_v/2 - 1] = v_diff_lower
rf_sizes_v[n_v/2 - 1] = v_diff_lower[-1]
rf_sizes_v[n_v/2] = v_diff_upper[0]
rf_sizes_v[n_v/2 + 1:] = v_diff_upper
#rf_sizes_v[n_v/2:] = v_diff[:n_v/2 - 1]

for i_ in xrange(n_v):
    print '%.3f\t%.3f' % (v_rho[i_], rf_sizes_v[i_])

print 'rf_sizes_v', rf_sizes_v
print 'n_v', n_v

# <codecell>

tp, rfs = stp.set_tuning_prop_1D_with_const_fovea(params)
#tp  = np.loadtxt(params['tuning_prop_means_fn'])
fig = pylab.figure()
ax = fig.add_subplot(111)
#ax.plot(tp[:, 0], tp[:, 2], 'o', ls='')
patches = []
tp = np.zeros((params['n_exc'], 4))
for i_hc in xrange(params['n_hc']):
    for i_mc in xrange(n_v):
        for i_cell in xrange(params['n_exc_per_mc']):
            i_ = i_hc * params['n_exc_per_hc'] + i_mc * params['n_exc_per_mc'] + i_cell
            tp[i_, 0] = xpos[i_hc] # + noise
            tp[i_, 1] = .5
            tp[i_, 2] = v_rho[i_mc]
            ellipse = mpatches.Ellipse((tp[i_, 0], tp[i_, 2]), rf_sizes_x[i_hc], rf_sizes_v[i_mc])
            # print 'debug', tp[i_, 0], tp[i_, 2], rf_sizes_x[i_hc], rf_sizes_v[i_mc]
            patches.append(ellipse)
collection = PatchCollection(patches, alpha=0.2, facecolor='b', edgecolor='k')
ax.add_collection(collection)
ax.set_xlim((0., 1.))
ax.set_ylim((-1., 1.))

# <codecell>


# <codecell>

pylab.show()
