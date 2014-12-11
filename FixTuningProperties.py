import simulation_parameters
import utils
import numpy as np
import set_tuning_properties as stp
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pylab


def get_tuning_properties_and_rfs_const_fovea(params):
    tp = np.zeros((params['n_exc'], 4))
    rfs = np.zeros((params['n_exc'], 4))
    RNG = np.random.RandomState(params['tp_seed'])

    n_rf_x_log = params['n_rf_x'] - params['n_rf_x_fovea']
    xpos_log = stp.get_xpos_log_distr(params, n_x=n_rf_x_log, x_min=0.1, x_max=0.45)
    xpos_const = np.linspace(.5 - params['x_min_tp'], .5 + params['x_min_tp'], params['n_rf_x_fovea'])

    xpos = np.zeros(params['n_rf_x'])
    xpos[:params['n_rf_x_log'] / 2] = xpos_log[:params['n_rf_x_log'] / 2]
    xpos[-params['n_rf_x_log'] / 2:] = xpos_log[params['n_rf_x_log'] / 2:]
    xpos[params['n_rf_x_log'] / 2: params['n_rf_x_log'] / 2 + params['n_rf_x_fovea']]= xpos_const
    rf_sizes_x = np.zeros(params['n_rf_x'])
    xpos_diff = xpos[1:] - xpos[:-1]
    n_x = xpos.size
    xpos_diff_lower = xpos_diff[:n_x/2 - 1]
    xpos_diff_upper = list(xpos_diff_lower)
    xpos_diff_upper.reverse()
    rf_sizes_x[:n_x / 2 - 1] = xpos_diff_lower
    rf_sizes_x[n_x / 2 - 1] = xpos_diff_lower[-1]
    rf_sizes_x[n_x / 2] = xpos_diff_upper[0]
    rf_sizes_x[n_x / 2 + 1:] = xpos_diff_upper
    rf_sizes_x[-1] = rf_sizes_x[0]

    n_v = params['n_v']
    v_rho_half = np.logspace(np.log(params['v_min_tp']) / np.log(params['log_scale']),\
                             np.log(params['v_max_tp']) / np.log(params['log_scale']), num=n_v/2,
                                endpoint=True, base=params['log_scale'])
    v_rho_neg = list(-1. * v_rho_half)
    v_rho_neg.reverse()
    v_rho = np.zeros(n_v)
    v_rho[:n_v/2] = v_rho_neg
    v_rho[n_v/2:] = v_rho_half
    rf_sizes_v = np.zeros(n_v)
    v_diff = np.zeros(n_v)
    v_diff[:-1] = v_rho[1:] - v_rho[:-1]
    v_diff_lower = v_diff[:n_v/2 - 1]
    v_diff_upper = list(v_diff_lower)
    v_diff_upper.reverse()

    rf_sizes_v[:n_v/2 - 1] = v_diff_lower
    rf_sizes_v[n_v/2 - 1] = v_diff_lower[-1]
    rf_sizes_v[n_v/2] = v_diff_upper[0]
    rf_sizes_v[n_v/2 + 1:] = v_diff_upper

    for i_hc in xrange(params['n_hc']):
        for i_mc in xrange(n_v):
            for i_cell in xrange(params['n_exc_per_mc']):
                i_ = i_hc * params['n_exc_per_hc'] + i_mc * params['n_exc_per_mc'] + i_cell
                tp[i_, 0] = xpos[i_hc] # + noise
                tp[i_, 1] = .5
                tp[i_, 2] = v_rho[i_mc]
                rfs[i_, 0] = rf_sizes_x[i_hc]
                rfs[i_, 2] = rf_sizes_v[i_mc]

    return tp, rfs

if __name__ == '__main__':
    GP = simulation_parameters.parameter_storage()
    params = GP.params
    
    tp, rfs = get_tuning_properties_and_rfs_const_fovea(params)
#    tp, rfs = stp.set_tuning_prop_1D_with_const_fovea(params)
    #tp  = np.loadtxt(params['tuning_prop_exc_fn'])
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    #ax.plot(tp[:, 0], tp[:, 2], 'o', ls='')
    patches = []
    for i_hc in xrange(params['n_hc']):
        for i_mc in xrange(params['n_v']):
            for i_cell in xrange(params['n_exc_per_mc']):
                i_ = i_hc * params['n_exc_per_hc'] + i_mc * params['n_exc_per_mc'] + i_cell
#                tp[i_, 0] = tp[i_, 0][i_hc] # + noise
#                tp[i_, 0] = xpos[i_hc] # + noise
#                tp[i_, 1] = .5
#                tp[i_, 2] = v_rho[i_mc]
                ellipse = mpatches.Ellipse((tp[i_, 0], tp[i_, 2]), rfs[i_, 0], rfs[i_, 2])
#                ellipse = mpatches.Ellipse((tp[i_, 0], tp[i_, 2]), rf_sizes_x[i_hc], rf_sizes_v[i_mc])
                # print 'debug', tp[i_, 0], tp[i_, 2], rf_sizes_x[i_hc], rf_sizes_v[i_mc]
                patches.append(ellipse)
    collection = PatchCollection(patches, alpha=0.2, facecolor='b', edgecolor='k')
    ax.add_collection(collection)
    ax.set_xlim((0., 1.))
    ax.set_ylim((-1., 1.))


pylab.show()
