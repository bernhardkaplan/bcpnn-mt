import numpy as np
import pylab
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy.random as rnd
import utils
import PlottingScripts.FigureCreator
from PlottingScripts.FigureCreator import plot_params

plot_params.update({'figure.subplot.left':.12,
              'figure.subplot.bottom':.12, 
              'figure.subplot.right':.95,
              'figure.subplot.top':.88})
pylab.rcParams.update(plot_params)

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


def get_xpos_log_distr(params):
    """
    Returns the n_hc positions
    """
    n_x = params['n_mc_per_hc']
    x_min = params['rf_x_center_distance']
    x_max = .5 - params['xpos_hc_0']
    logscale = params['log_scale']
    logspace = np.logspace(np.log(x_min) / np.log(logscale), np.log(x_max) / np.log(logscale), n_x / 2, base=logscale)
#    print 'x logscpace_pos:', logspace
    logspace = list(logspace)
    logspace.reverse()
    x_lower = .5 - np.array(logspace)
    logspace = np.logspace(np.log(x_min) / np.log(logscale), np.log(x_max) / np.log(logscale), n_x / 2, base=logscale)
    x_upper =  logspace + .5
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
    print 'debug x_pos', x_pos
#    print 'debug x_half', x_half
    print 'debug dx_pos_half', dx_pos_half
#    dx_pos_half[-1] = .5 * dx_pos_half[-2]
    rf_size_x[:params['n_hc'] / 2] = dx_pos_half
    dx_pos_upper_half = list(dx_pos_half)
    dx_pos_upper_half.reverse()
    print 'debug dx_pos_upper_half', dx_pos_upper_half
    rf_size_x[params['n_hc'] / 2:] = dx_pos_upper_half
    rf_size_x *= 1.5
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


if __name__ == '__main__':
    import simulation_parameters
    param_tool = simulation_parameters.parameter_storage()
    params = param_tool.params

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

