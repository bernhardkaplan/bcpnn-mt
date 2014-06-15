import numpy as np
import pylab
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def get_xpos(params):
    """
    Returns n_hc positions
    """
    n_hc = params['n_hc']
    eps_0 = 0.05     # the position of the first HC index
    eps_half = params['rf_x_center_distance'] # distance for HC with index (n_hc / 2 - 1) from center 0.5 (the -1 is because HC counting starts at 0 and ends at n_hc - 1)
    assert (n_hc % 2 == 0), 'because there should be an equal number of HC for left and right part of the visual field'
    assert (0 <= eps_0)
    assert (0 < eps_half)
    assert (.5 > eps_0)
    assert (.5 > eps_half)

    C = .3          # 'free' parameter determining the steep-ness of the exponential distribution for x-pos
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

    dv_rho = np.zeros(v_rho.size)
    v_half = v_rho[:params['n_mc_per_hc'] / 2]
    dv_rho_half = np.zeros(params['n_mc_per_hc'] / 2)
    dv_rho_half[:-1] = v_half[1:] - v_half[:-1]
    dv_rho_half[-1] = dv_rho_half[-2]
    dv_rho_half[0] = params['rf_x_center_distance']
    dv_rho[:params['n_mc_per_hc'] / 2] = dv_rho_half
    dv_rho[params['n_mc_per_hc'] / 2:] = dv_rho_half

#    dv_rho[:-1] = v_rho[1:] - v_rho[:-1]
#    dv_rho[-1] = dv_rho[-2]
    return dv_rho


def get_receptive_field_sizes_x(params, x_pos):

    dx_pos = np.zeros(x_pos.size)
    x_half = x_pos[:params['n_hc'] / 2]
    dx_pos_half = np.zeros(params['n_hc'] / 2)
    dx_pos_half[:-1] = x_half[1:] - x_half[:-1]
    dx_pos_half[-1] = dx_pos_half[-2]
    dx_pos[:params['n_hc'] / 2] = dx_pos_half
    dx_pos_upper_half = list(dx_pos_half)
    dx_pos_upper_half.reverse()
    dx_pos[params['n_hc'] / 2:] = dx_pos_upper_half
#    dx_pos[params['n_hc'] / 2:] = dx_pos_half

#    dx_pos[:-1] = x_pos[1:] - x_pos[:-1]
#    dx_pos[-1] = dx_pos[-2]
    print 'debug dx_pos', dx_pos
    return dx_pos



if __name__ == '__main__':
    import simulation_parameters
    param_tool = simulation_parameters.parameter_storage()
    params = param_tool.params

    x_pos = get_xpos(params)
    v_rho = get_speed_tuning(params)
    rf_size_v = get_receptive_field_sizes_v(params, v_rho)
    rf_size_x = get_receptive_field_sizes_x(params, x_pos)
#    rf_size_x = .05 * np.ones(params['n_hc'])

#    fig = pylab.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(range(params['n_hc']), x_pos, 'o')

    fig = pylab.figure()
    ax = fig.add_subplot(111)


    patches = []
    for i_hc in xrange(params['n_hc']):
        for i_mc in xrange(params['n_mc_per_hc']):
            x, y = x_pos[i_hc], v_rho[i_mc]
            ax.plot(x, y, 'o', c='b')
            ellipse = mpatches.Ellipse((x, y), rf_size_x[i_hc], rf_size_v[i_mc])
            patches.append(ellipse)

    collection = PatchCollection(patches, alpha=0.1)
    ax.add_collection(collection)

    pylab.show()

