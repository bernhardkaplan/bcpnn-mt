import numpy as np
import pylab
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy.random as rnd
import utils
import PlottingScripts.FigureCreator
from PlottingScripts.FigureCreator import plot_params


def get_xpos_exponential_distr(params):
    """
    Returns n_hc positions
    """
    n_hc = params['n_hc']
    eps_0 = params['xpos_hc_0'] # the position of the first HC index
    eps_half = params['rf_x_center_distance'] # distance for HC with index (n_hc / 2 - 1) from center 0.5 (the -1 is because HC counting starts at 0 and ends at n_hc - 1)
    assert (n_hc % 2 == 0), 'because there should be an equal number of HC for left and right part of the visual field'
    assert (0 <= eps_0)
    assert (0 < eps_half)
    assert (.5 > eps_0)
    assert (.5 > eps_half)
    C = params['rf_x_distribution_steepness']# 'free' parameter determining the steep-ness of the exponential distribution for x-pos
    z = (n_hc / 2 - 1.) # half of the indices
    A = (.5 - eps_half - eps_0) / (1. - np.exp(-C * z))
    # idx = hypercolumn indices for the first half
    idx = np.arange(0, n_hc / 2)    
    idx_2 = np.arange(n_hc / 2, n_hc)    
    # f_idx = the x position of the HC depending on the index of the HC
    f_idx = A * (1. - np.exp(-C * idx)) + eps_0
    B = (.5 - eps_0 - eps_half) / (np.exp(C * (n_hc / 2 - 1)) - 1)
    D = .5 + eps_half - B
    g_idx = B * np.exp(C * idx) + D
    x_pos = np.zeros(n_hc)
    x_pos[:n_hc / 2] = f_idx
    x_pos[-n_hc / 2:] = g_idx
    return x_pos


def get_xpos_log_distr(params, n_x=None, x_min=None, x_max=None):
    """
    Returns the n_hc positions
    """
    if n_x == None:
        n_x = params['n_hc']
    if x_min == None:
        x_min = params['rf_x_center_distance']
    if x_max == None:
        x_max = .5 - params['xpos_hc_0']
    logscale = params['log_scale']
#    logspace = np.logspace(np.log(x_min) / np.log(logscale), np.log(x_max) / np.log(logscale), n_x / 2, base=logscale)
    logspace = np.logspace(np.log(x_max) / np.log(logscale), np.log(x_min) / np.log(logscale), n_x / 2, base=logscale, endpoint=False)
#    print 'x logscpace_pos:', logspace
#    logspace = list(logspace)
#    logspace.reverse()
    x_lower = .5 - np.array(logspace)
#    logspace = np.logspace(np.log(x_min) / np.log(logscale), np.log(x_max) / np.log(logscale), n_x / 2, base=logscale)
    logspace = np.logspace(np.log(x_max) / np.log(logscale), np.log(x_min) / np.log(logscale), n_x / 2, base=logscale, endpoint=False)
    logspace = list(logspace)
    logspace.reverse()

    x_upper =  logspace + np.array(.5)
    x_rho = np.zeros(n_x)
    x_rho[:n_x/2] = x_lower
    x_rho[n_x/2:] = x_upper
    return x_rho



def get_speed_tuning(params):
    """
    Returns n_mc_per_hc preferred speeds.
    """
    n_v = params['n_mc_per_hc']
    v_max = params['v_max_tp']
    v_min = params['v_min_tp']
    logscale = params['log_scale']
    v_rho_pos = np.logspace(np.log(v_min) / np.log(logscale), np.log(v_max) / np.log(logscale), n_v / 2, base=logscale)
    v_rho_neg = -1. * v_rho_pos
    v_rho = np.zeros(n_v)
    v_rho[:n_v/2] = v_rho_pos
    v_rho[n_v/2:] = v_rho_neg
    return v_rho


def get_receptive_field_sizes_v(params, v_rho):

    rf_size_v = np.zeros(v_rho.size)
    v_half = v_rho[:params['n_mc_per_hc'] / 2]
    dv_rho_half = np.zeros(params['n_mc_per_hc'] / 2)
    dv_rho_half[:-1] = v_half[1:] - v_half[:-1]
    dv_rho_half[-1] = v_half[-1] - v_half[-2]
#    dv_rho_half[-1] = 1.5 * dv_rho_half[-2]
    dv_rho_half[0] = params['v_min_tp']
    rf_size_v[:params['n_mc_per_hc'] / 2] = dv_rho_half
    rf_size_v[params['n_mc_per_hc'] / 2:] = dv_rho_half
    rf_size_v *= .9

#    rf_size_v[:-1] = v_rho[1:] - v_rho[:-1]
#    rf_size_v[-1] = rf_size_v[-2]
#    print 'debug rf_size_v', rf_size_v
    return rf_size_v


def get_receptive_field_sizes_x(params, x_pos):

    rf_size_x = np.zeros(x_pos.size)
    x_half = x_pos[:params['n_hc'] / 2]
    dx_pos_half = np.zeros(params['n_hc'] / 2)
    dx_pos_half[:-1] = x_half[1:] - x_half[:-1]
    dx_pos_half[-1] = dx_pos_half[-2]

#    dx_pos_half[0] = self.params['rf_x_center_distance']

#    dx_pos_half[1:] = x_half[1:] - x_half[:-1]
#    dx_pos_half[0] = x_half[1] - x_half[0]
#    print 'debug x_pos', x_pos
#    print 'debug x_half', x_half
#    print 'debug dx_pos_half', dx_pos_half
#    dx_pos_half[-1] = .5 * dx_pos_half[-2]
    rf_size_x[:params['n_hc'] / 2] = dx_pos_half
    dx_pos_upper_half = list(dx_pos_half)
    dx_pos_upper_half.reverse()
#    print 'debug dx_pos_upper_half', len(dx_pos_upper_half), len(rf_size_x[params['n_hc'] / 2:]), len(rf_size_x)
    rf_size_x[params['n_hc'] / 2:] = dx_pos_upper_half
#    rf_size_x *= 1.5
#    rf_size_x *= 0.8
#    rf_size_x[params['n_hc'] / 2:] = dx_pos_half

#    rf_size_x[:-1] = x_pos[1:] - x_pos[:-1]
#    rf_size_x[-1] = rf_size_x[-2]
#    print 'debug rf_size_x', rf_size_x
    return rf_size_x


def get_relative_distance_error(rf_centers):

    rel = np.zeros(rf_centers.size - 1)
    for i_ in xrange(1, rf_centers.size):
        diff = rf_centers[i_] - rf_centers[i_-1]
        rel[i_-1] = np.abs(diff) / np.abs(rf_centers[i_-1])
    return rel


def set_tuning_properties(params):
    tuning_prop = np.zeros((params['n_exc'], 4))
    rfs = np.zeros((params['n_exc'], 4))
    x_pos = get_xpos_log_distr(params)
    v_rho = get_speed_tuning(params)
    rf_size_x = get_receptive_field_sizes_x(params, x_pos)
    rf_size_v = get_receptive_field_sizes_v(params, v_rho)
    index = 0
    for i_mc in xrange(params['n_mc_per_hc']):
        print 'DEBUG rf_size_v[%d] = %.3e' % (i_mc, rf_size_v[i_mc])
    for i_hc in xrange(params['n_hc']):
        print 'DEBUG rf_size_x[%d] = %.3e' % (i_hc, rf_size_x[i_hc])
        for i_mc in xrange(params['n_mc_per_hc']):
            x, u = x_pos[i_hc], v_rho[i_mc]
            for i_exc in xrange(params['n_exc_per_mc']):
                x, u = x_pos[i_hc], v_rho[i_mc]
                tuning_prop[index, 0] = (x + np.abs(x - .5) / .5 * rnd.uniform(-params['sigma_rf_pos'] , params['sigma_rf_pos'])) % params['torus_width']
                tuning_prop[index, 1] = 0.5 
                tuning_prop[index, 2] = u * (1. + rnd.uniform(-params['sigma_rf_speed'] , params['sigma_rf_speed']))
                tuning_prop[index, 3] = 0. 
                rfs[index, 0] = rf_size_x[i_hc]
                rfs[index, 2] = rf_size_v[i_mc]
                index += 1
    return tuning_prop, rfs


def get_receptive_field_sizes_x_const_fovea(params, rf_x):
    idx = np.argsort(rf_x)
    rf_size_x = np.zeros(rf_x.size)
    pos_idx = (rf_x[idx] > 0.5).nonzero()[0]
    neg_idx = (rf_x[idx] < 0.5).nonzero()[0]
    dx_pos_half = np.zeros(pos_idx.size)
    dx_neg_half = np.zeros(neg_idx.size)
    dx_pos_half = rf_x[idx][pos_idx][1:] - rf_x[idx][pos_idx][:-1]
    dx_neg_half = rf_x[idx][neg_idx][1:] - rf_x[idx][neg_idx][:-1]
    rf_size_x[:neg_idx.size-1] = dx_neg_half
    rf_size_x[neg_idx.size] = dx_neg_half[-1]
    if params['n_rf_x'] % 2:
        rf_size_x[pos_idx.size+2:] = dx_pos_half # for 21
    else:
        rf_size_x[pos_idx.size+1:] = dx_pos_half # for 20
    rf_size_x[pos_idx.size] = dx_pos_half[0]
    rf_size_x[idx.size / 2 - 1] = dx_pos_half[0]
#    rf_size_x *= params['rf_size_x_multiplicator']
    return rf_size_x


def set_tuning_properties_and_rfs_const_fovea(params):
    tp = np.zeros((params['n_exc'], 4))
    rfs = np.zeros((params['n_exc'], 4))
    RNG = np.random.RandomState(params['tp_seed'])

    n_rf_x_log = params['n_rf_x'] - params['n_rf_x_fovea']
    xpos_log = get_xpos_log_distr(params, n_x=n_rf_x_log, x_min=0.1, x_max=0.45)
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
                noise_x = np.abs(xpos[i_hc] - .5) / .5 * RNG.normal(0., params['sigma_rf_pos'])
                tp[i_, 0] = min(1., max(xpos[i_hc] + noise_x, 0.))
                tp[i_, 1] = .5
                tp[i_, 2] = v_rho[i_mc] * RNG.normal(1., params['sigma_rf_speed'])
                rfs[i_, 0] = rf_sizes_x[i_hc]
                rfs[i_, 2] = rf_sizes_v[i_mc]

    return tp, rfs


def set_tuning_prop_1D_with_const_fovea(params, cell_type='exc'):
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_v = params['n_v']
        n_rf_x = params['n_hc']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_v = params['n_v_inh']
        n_rf_x = params['n_rf_x_inh']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    if params['log_scale']==1:
        v_rho_half = np.linspace(v_min, v_max, num=n_v/2, endpoint=True)
    else:
        v_rho_half = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_v/2,
                        endpoint=True, base=params['log_scale'])
    RNG = np.random.RandomState(params['tp_seed'])

    rf_sizes = np.zeros((n_cells, 4))
    v_rho = np.zeros(n_v)
    v_rho[:n_v/2] = -v_rho_half
    v_rho[n_v/2:] = v_rho_half
    
    n_rf_x_log = params['n_rf_x'] - params['n_rf_x_fovea']
#    RF_x_log = get_xpos_log_distr_const_fovea(params['log_scale'], n_rf_x_log, x_min=params['x_min_tp'], x_max=params['x_max_tp'])
    RF_x_log = get_xpos_log_distr(params, n_rf_x_log, x_min=params['x_min_tp'], x_max=params['x_max_tp'])
#    RF_x_log = get_xpos_log_distr(params, n_rf_x_log)
    RF_x_const = np.linspace(.5 - params['x_min_tp'], .5 + params['x_min_tp'], params['n_rf_x_fovea'])
    RF_x = np.zeros(n_rf_x)
    idx_upper = n_rf_x_log / 2 + params['n_rf_x_fovea']
    RF_x[:n_rf_x_log / 2] = RF_x_log[:n_rf_x_log / 2]
    RF_x[idx_upper:] = RF_x_log[n_rf_x_log / 2:]
    RF_x[n_rf_x_log / 2 : n_rf_x_log / 2 + params['n_rf_x_fovea']] = RF_x_const

#    print '------------------------------\nDEBUG'
#    print 'v_rho:', v_rho
#    print 'v_rho_half:', v_rho_half
#    print 'n_rf_x: ', n_rf_x
#    print 'n_rf_x_log: ', n_rf_x_log
#    print 'n_rf_x_fovea: ', params['n_rf_x_fovea']
#    print 'RF_x_const:', RF_x_const
#    print 'RF_x_log:', RF_x_log
#    print 'RF_x:', RF_x

    index = 0
    tuning_prop = np.zeros((n_cells, 4))
    rf_sizes_x = get_receptive_field_sizes_x_const_fovea(params, RF_x)
    rf_sizes_v = get_receptive_field_sizes_v(params, v_rho)
    for i_x in xrange(n_rf_x):
        for i_v_rho, rho in enumerate(v_rho):
            for i_in_mc in xrange(params['n_exc_per_mc']):
                x = RF_x[i_x]
#                tuning_prop[index, 0] = RF_x[i_x]
#                tuning_prop[index, 0] += RNG.normal(.0, params['sigma_rf_pos'] / 2) # add some extra noise to the neurons representing the fovea (because if their noise is only a percentage of their distance from the center, it's too small
#                tuning_prop[index, 0] = tuning_prop[index, 0] % 1.0
                pm = utils.get_plus_minus(RNG)
                tuning_prop[index, 0] = RF_x[i_x] + pm * RNG.normal(.0, params['sigma_rf_pos'])
                tuning_prop[index, 1] = 0.5 # i_x / float(n_rf_x) # y-pos 
#                    tuning_prop[index, 2] = (-1)**(i_v_rho % 2) * rho * (1. + params['sigma_rf_speed'] * np.random.randn())
                tuning_prop[index, 2] = rho * (1. + params['sigma_rf_speed'] * np.random.randn())
                tuning_prop[index, 3] = 0. 
                rf_sizes[index, 0] = rf_sizes_x[i_x]
                rf_sizes[index, 2] = rf_sizes_v[i_v_rho]
                index += 1

    assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
#        exit(1)
    return tuning_prop, rf_sizes


def set_tuning_prop_1D_with_const_fovea_columns(params, cell_type='exc'):
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_v = params['n_v']
        n_rf_x = params['n_hc']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_v = params['n_v_inh']
        n_rf_x = params['n_rf_x_inh']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    if params['log_scale']==1:
        v_rho_half = np.linspace(v_min, v_max, num=n_v/2, endpoint=True)
    else:
        v_rho_half = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_v/2,
                        endpoint=True, base=params['log_scale'])
    RNG = np.random.RandomState(params['tp_seed'])

    rf_sizes = np.zeros((n_cells, 4))
    v_rho = np.zeros(n_v)
    v_rho[:n_v/2] = -v_rho_half
    v_rho[n_v/2:] = v_rho_half
    
    n_rf_x_log = params['n_rf_x'] - params['n_rf_x_fovea']
    RF_x_log = get_xpos_log_distr(params, n_rf_x_log, x_min=params['x_min_tp'], x_max=params['x_max_tp'])
    RF_x_const = np.linspace(.5 - params['x_min_tp'], .5 + params['x_min_tp'], params['n_rf_x_fovea'])
    RF_x = np.zeros(n_rf_x)
    idx_upper = n_rf_x_log / 2 + params['n_rf_x_fovea']
    RF_x[:n_rf_x_log / 2] = RF_x_log[:n_rf_x_log / 2]
    RF_x[idx_upper:] = RF_x_log[n_rf_x_log / 2:]
    RF_x[n_rf_x_log / 2 : n_rf_x_log / 2 + params['n_rf_x_fovea']] = RF_x_const

    print '------------------------------\nDEBUG'
    print 'v_rho:', v_rho
    print 'v_rho_half:', v_rho_half
    print 'n_rf_x: ', n_rf_x
    print 'n_rf_x_log: ', n_rf_x_log
    print 'n_rf_x_fovea: ', params['n_rf_x_fovea']
    print 'RF_x_const:', RF_x_const
    print 'RF_x_log:', RF_x_log
    print 'RF_x:', RF_x

    index = 0
    tuning_prop = np.zeros((n_cells, 4))
    rf_sizes_x = get_receptive_field_sizes_x_const_fovea(params, RF_x)
    rf_sizes_v = get_receptive_field_sizes_v(params, v_rho)
    for i_RF in xrange(n_rf_x):
        for i_v_rho, rho in enumerate(v_rho):
            for i_in_mc in xrange(params['n_exc_per_mc']):
                x = RF_x[i_RF]
                tuning_prop[index, 0] = RF_x[i_RF]
                tuning_prop[index, 0] += RNG.normal(.0, params['sigma_rf_pos'] / 2) # add some extra noise to the neurons representing the fovea (because if their noise is only a percentage of their distance from the center, it's too small
                tuning_prop[index, 0] = tuning_prop[index, 0] % 1.0
                tuning_prop[index, 1] = 0.5 # i_RF / float(n_rf_x) # y-pos 
#                    tuning_prop[index, 2] = (-1)**(i_v_rho % 2) * rho * (1. + params['sigma_rf_speed'] * np.random.randn())
                tuning_prop[index, 2] = rho * (1. + params['sigma_rf_speed'] * np.random.randn())
                tuning_prop[index, 3] = 0. 
                rf_sizes[index, 0] = rf_sizes_x[i_RF]
                rf_sizes[index, 2] = rf_sizes_v[i_v_rho]
                index += 1

    assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
#        exit(1)
    return tuning_prop, rf_sizes


def plot_tuning_prop(params):
    """
    without constant fovea
    """
    x_pos = get_xpos_log_distr(params)
    x_pos_exp = get_xpos_exponential_distr(params)
    v_rho = get_speed_tuning(params)
    rf_size_v = get_receptive_field_sizes_v(params, v_rho)
    rf_size_x = get_receptive_field_sizes_x(params, x_pos)

    rel_v = get_relative_distance_error(v_rho)
    rel_x_log = get_relative_distance_error(np.abs(x_pos-.5))
    rel_x_exp= get_relative_distance_error(np.abs(x_pos_exp-.5))

    fig = pylab.figure(figsize=utils.get_figsize(800, portrait=False))
    ax = fig.add_subplot(211)
    ax.plot(range(rel_v.size), rel_v, 'o', markersize=5)
    ax.set_title('relative error for v')
    ax2 = fig.add_subplot(212)
    ax2.plot(range(rel_x_log.size), rel_x_log, 'o', c='b', markersize=5, label='log distr')
    ax2.plot(range(rel_x_exp.size), rel_x_exp, 'o', c='r', markersize=5, label='exp distr')
    ax2.set_title('relative error for x')
    ax2.legend()

    fig = pylab.figure(figsize=utils.get_figsize(800, portrait=False))
    ax = fig.add_subplot(111)
    rnd.seed(0)
    tuning_prop = np.zeros((params['n_exc'], 4))
    patches_mc = []
    patches_ = []
    index = 0
    for i_mc in xrange(params['n_mc_per_hc']):
        print 'DEBUG rf_size_v[%d] = %.3e' % (i_mc, rf_size_v[i_mc])
    for i_hc in xrange(params['n_hc']):
        print 'DEBUG rf_size_x[%d] = %.3e' % (i_hc, rf_size_x[i_hc])
        for i_mc in xrange(params['n_mc_per_hc']):
            x, u = x_pos[i_hc], v_rho[i_mc]
            p_mc, = ax.plot(x, u, 'o', c='r', markersize=5, markeredgewidth=0)
            ellipse = mpatches.Ellipse((x, u), rf_size_x[i_hc], rf_size_v[i_mc])
            patches_mc.append(ellipse)
            for i_exc in xrange(params['n_exc_per_mc']):
                x, u = x_pos[i_hc], v_rho[i_mc]
                tuning_prop[index, 0] = (x + np.abs(x - .5) / .5 * rnd.uniform(-params['sigma_rf_pos'] , params['sigma_rf_pos'])) % params['torus_width']
                tuning_prop[index, 1] = 0.5 
                tuning_prop[index, 2] = u * (1. + rnd.uniform(-params['sigma_rf_speed'] , params['sigma_rf_speed']))
                tuning_prop[index, 3] = 0. 
                p_cell, = ax.plot(tuning_prop[index, 0], tuning_prop[index, 2], 'o', c='k', markersize=2)
                ellipse = mpatches.Ellipse((tuning_prop[index, 0], tuning_prop[index, 2]), rf_size_x[i_hc], rf_size_v[i_mc])
                patches_.append(ellipse)
                index += 1
    plots = [p_mc, p_cell]
    labels = ['MC center without noise', 'Cell']
            
    ax.set_title('Distribution of tuning properties for learning anisotropic\nconnectivity for motion-based prediction through BCPNN')
    ax.legend(plots, labels, numpoints=1)
    collection = PatchCollection(patches_mc, alpha=0.2, facecolor='r', edgecolor='k', linewidth=4)
    ax.add_collection(collection)
    collection = PatchCollection(patches_, alpha=0.1, facecolor='b', edgecolor=None)
    ax.add_collection(collection)

    ylim = ax.get_ylim()
    ax.set_ylim((1.1 * ylim[0], 1.1 * ylim[1]))
    ax.set_xlabel('Receptive field position x')
    ax.set_ylabel('Preferred speed $v_x$')
    pylab.savefig('tuning_properties.png', dpi=200)
    pylab.show()


def plot_tuning_prop_const_fovea(params):


    fig = pylab.figure(figsize=utils.get_figsize(800, portrait=False))
    ax = fig.add_subplot(111)
    tuning_prop, rf_sizes = set_tuning_prop_1D_with_const_fovea(params)
    rf_size_x = rf_sizes[:, 0]
    rf_size_v = rf_sizes[:, 2]

    patches_ = []
    index = 0
    for i_mc in xrange(params['n_mc_per_hc']):
        print 'DEBUG rf_size_v[%d] = %.3e' % (i_mc, rf_size_v[i_mc])
    for i_hc in xrange(params['n_hc']):
        print 'DEBUG rf_size_x[%d] = %.3e' % (i_hc, rf_size_x[i_hc])
        for i_mc in xrange(params['n_mc_per_hc']):
            for i_exc in xrange(params['n_exc_per_mc']):
                p_cell, = ax.plot(tuning_prop[index, 0], tuning_prop[index, 2], 'o', c='k', markersize=2)
                ellipse = mpatches.Ellipse((tuning_prop[index, 0], tuning_prop[index, 2]), rf_size_x[i_hc], rf_size_v[i_mc])
                patches_.append(ellipse)
                index += 1
    plots = [p_cell]
    labels = ['MC center without noise', 'Cell']
            
    ax.set_title('Distribution of tuning properties for learning anisotropic\nconnectivity for motion-based prediction through BCPNN')
    ax.legend(plots, labels, numpoints=1)
    collection = PatchCollection(patches_, alpha=0.1, facecolor='b', edgecolor=None)
    ax.add_collection(collection)
    pylab.show()


def set_tuning_prop_1D_with_const_fovea_and_const_velocity(params, cell_type='exc'):
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_v = params['n_v']
        n_rf_x = params['n_rf_x']
        n_rf_v = params['n_v']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_v = params['n_v_inh']
        n_rf_x = params['n_rf_x_inh']
        n_rf_v = params['n_rf_v_inh']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    RNG_tp = np.random.RandomState(params['tp_seed'])
    rf_sizes = np.zeros((n_cells, 4))
    n_rf_v_log = params['n_v'] - params['n_rf_v_fovea']
    RF_v_const = np.linspace(-params['v_min_tp'], params['v_min_tp'], params['n_rf_v_fovea'] + 1, endpoint=False)[1:]
    RF_v = np.zeros(n_rf_v)
    idx_upper = n_rf_v_log / 2 + params['n_rf_v_fovea']
    if params['log_scale']==1:
        v_rho_half = np.linspace(v_min, v_max, num=n_rf_v_log/2, endpoint=True)
    else:
        v_rho_half = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_rf_v_log/2,
                        endpoint=True, base=params['log_scale'])
    v_rho_half_ = list(v_rho_half)
    v_rho_half_.reverse()
    RF_v[:n_rf_v_log / 2] = v_rho_half_
#    print 'debug RF_v:', RF_v, v_rho_half
    RF_v[idx_upper:] = -v_rho_half
    RF_v[n_rf_v_log / 2 : n_rf_v_log / 2 + params['n_rf_v_fovea']] = RF_v_const

    n_rf_x_log = params['n_rf_x'] - params['n_rf_x_fovea']
    RF_x_log = get_xpos_log_distr(params, n_rf_x_log, x_min=params['x_min_tp'], x_max=params['x_max_tp'])

    RF_x_const = np.linspace(.5 - params['x_min_tp'], .5 + params['x_min_tp'], params['n_rf_x_fovea'])
    RF_x = np.zeros(n_rf_x)
    idx_upper = n_rf_x_log / 2 + params['n_rf_x_fovea']
    RF_x[:n_rf_x_log / 2] = RF_x_log[:n_rf_x_log / 2]
    RF_x[idx_upper:] = RF_x_log[n_rf_x_log / 2:]
    RF_x[n_rf_x_log / 2 : n_rf_x_log / 2 + params['n_rf_x_fovea']] = RF_x_const
    index = 0
    tuning_prop = np.zeros((n_cells, 4))
    rf_sizes_x = get_receptive_field_sizes_x(params, RF_x)
    rf_sizes_v = get_receptive_field_sizes_v_const_fovea(params, RF_v)
    for i_RF in xrange(n_rf_x):
        for i_v_rho, rho in enumerate(RF_v):
            for i_exc in xrange(params['n_exc_per_mc']):
                tuning_prop[index, 0] = RF_x[i_RF]
                tuning_prop[index, 0] += RNG_tp.normal(.0, params['sigma_rf_pos'] / 2) # add some extra noise to the neurons representing the fovea (because if their noise is only a percentage of their distance from the center, it's too small
                tuning_prop[index, 0] = tuning_prop[index, 0] % 1.0
                tuning_prop[index, 1] = 0.5 # i_RF / float(n_rf_x) # y-pos 
                tuning_prop[index, 2] = rho * (1. + params['sigma_rf_speed'] * RNG_tp.randn())
                tuning_prop[index, 3] = 0. 
                rf_sizes[index, 0] = rf_sizes_x[i_RF]
                rf_sizes[index, 2] = rf_sizes_v[i_v_rho]
                index += 1

    assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
    return tuning_prop, rf_sizes


def get_receptive_field_sizes_v_const_fovea(params, rf_v):
    if (params['n_v'] >= 4):
        idx = np.argsort(rf_v)
        rf_size_v = np.zeros(rf_v.size)
        pos_idx = (rf_v[idx] > 0.0).nonzero()[0]
        neg_idx = (rf_v[idx] < 0.0).nonzero()[0]
        dv_pos_half = np.zeros(pos_idx.size)
        dv_neg_half = np.zeros(neg_idx.size)
        dv_pos_half = rf_v[idx][pos_idx][1:] - rf_v[idx][pos_idx][:-1]
        dv_neg_half = np.abs(rf_v[idx][neg_idx][1:] - rf_v[idx][neg_idx][:-1])
        dv_neg_reverse = list(dv_neg_half)
        dv_neg_reverse.reverse()
        rf_size_v[:neg_idx.size-1] = dv_neg_half
        rf_size_v[neg_idx.size] = dv_neg_half[-1]
        if params['n_v'] % 2:
            rf_size_v[pos_idx.size+2:] = dv_pos_half # for 21
        else:
            rf_size_v[pos_idx.size+1:] = dv_pos_half # for 20
        rf_size_v[pos_idx.size] = dv_pos_half[0]
        rf_size_v[idx.size / 2 - 1] = dv_pos_half[0]
    #    rf_size_v *= params['rf_size_v_multiplicator']
    #    print 'rf_size_v', rf_size_v
        return rf_size_v
    else:
        rf_size_v = np.zeros(rf_v.size)
        print 'rf_v:', rf_v
        return rf_v


if __name__ == '__main__':

    plot_params.update({'figure.subplot.left':.12,
                  'figure.subplot.bottom':.12, 
                  'figure.subplot.right':.95,
                  'figure.subplot.top':.88})
    pylab.rcParams.update(plot_params)

    import simulation_parameters
    param_tool = simulation_parameters.parameter_storage()
    params = param_tool.params
    plot_tuning_prop_const_fovea(params)

