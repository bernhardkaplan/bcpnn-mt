"""
    This file contains a bunch of helper functions (in alphabetic order).
"""

import numpy as np
import numpy.random as rnd
import os
from scipy.spatial import distance
import copy


def filter_spike_train(spikes, dt=1., tau=30., t_max=None):
    """
    spikes: list or array of spikes
    """
    if t_max == None:
        t_max = spikes[-1] + tau
    t_vec = np.arange(0, t_max, dt)
    y = np.zeros(t_vec.size)

    spike_idx = []
    for spike in spikes:
        spike_idx.append((t_vec < spike).nonzero()[0][-1])

    for i_, spike in enumerate(spikes):
        y[spike_idx[i_]:] += np.exp(-(t_vec[spike_idx[i_]:] - spike) / tau)

    return t_vec, y


def convert_connlist_to_matrix(fn, n_src, n_tgt):
    """
    Convert the connlist which is in format (src, tgt, weight, delay) to a weight matrix.
    """
    conn_list = np.loadtxt(fn)
    m = np.zeros((n_src, n_tgt))
    delays = np.zeros((n_src, n_tgt))
    print 'utils.convert_connlist_to_matrix(%s, %d, %d)' % (fn, n_src, n_tgt)
#    print 'utils.convert_connlist_to_matrix(%s, %d, %d) conn_list size: %d' % (fn, n_src, n_tgt, conn_list[:, 0].size)
    for i in xrange(conn_list[:,0].size):
        src = conn_list[i, 0]
        tgt = conn_list[i, 1]
        m[src, tgt] = conn_list[i, 2]
        delays[src, tgt] = conn_list[i, 3]
    return m, delays


def convert_connlist_to_adjlist_srcidx(fn, n_src):
    """
    Convert the connlist which is in format (src, tgt, weight, delay) to an
    adjacency list:
    src : [tgt_0, ..., tgt_n]
    """
    conn_list = np.loadtxt(fn)
    print 'utils.convert_connlist_to_adjlist(%s, %d, %d)' % (fn, n_src)
    adj_list = [[] for i in xrange(n_src)]
    for src in xrange(n_src):
        targets = get_targets(conn_list, src)
        adj_list[src] = targets[:, 1:].tolist()
    return adj_list


def convert_connlist_to_adjlist_tgtidx(fn, n_tgt):
    """
    Convert the connlist which is in format (src, tgt, weight, delay) to an
    adjacency list:
    src : [tgt_0, ..., tgt_n]
    """
    conn_list = np.loadtxt(fn)
    print 'utils.convert_connlist_to_adjlist(%s, %d, %d)' % (fn, n_tgt)
    adj_list = [[] for i in xrange(n_tgt)]
    for tgt in xrange(n_src):
        targets = get_targets(conn_list, tgt)
        adj_list[tgt] = targets[:, [0, 2, 3]].tolist()
    return adj_list


def extract_trace(d, gid):
    """
    d : voltage trace from a saved with compatible_output=False
    gid : cell_gid
    """
    mask = gid * np.ones(d[:, 0].size)
    indices = mask == d[:, 0]
    time_axis, volt = d[indices, 1], d[indices, 2]
    return time_axis, volt

def convert_spiketrain_to_trace(st, n):
    """
    st: spike train in the format [time, id]
    n : size of the trace to be returned
    To be used with spike train inputs.
    Returns a np.array with st[i] = 1 if i in st[:, 0], st[i] = 0 else.
    """
    trace = np.zeros(n)
    for i in st:
        trace[int(i)] = 1
    return trace


def low_pass_filter(trace, tau=10, initial_value=0.001, dt=1., spike_height=1.):
    """
    trace can be e.g. a spike train trace, i.e. all elements are 0 except spike times = 1,
    """

    eps = 0.0001
    n = len(trace)
    zi = np.ones(n) * initial_value
    for i in xrange(1, n):
        # pre-synaptic trace zi follows trace 
        dzi = dt * (trace[i] * spike_height - zi[i-1] + eps) / tau
        zi[i] = zi[i-1] + dzi
    return zi

def create_spike_trains_for_motion(tuning_prop, params, contrast=.9, my_units=None, seed=None):
    """
    This function writes spike trains to a dedicated path specified in the params dict
    Spike trains are generated for each unit / minicolumn based on the function's arguments the following way:
    The strength of stimulation for a column is computed based on the motion parameters and the tuning properties of the column.
    This strength determines the envelope the non-homogeneous Poisson process to create the spike train.

    Arguments:
        tuning_prop = np.array((n_cells, 4))
            tp[:, 0] : x-position
            tp[:, 1] : y-position
            tp[:, 2] : u-position (speed in x-direction)
            tp[:, 3] : v-position (speed in y-direction)

        params:  dictionary storing all simulation parameters
        my_units: tuple of integers (start, begin), in case of parallel execution each processor creates spike trains for its own units or columns

    """

    if seed == None:
        seed = params['input_spikes_seed']
    rnd.seed(seed)
    dt = params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 

#    time = np.arange(0, params['t_stimulus'], dt)
    time = np.arange(0, params['t_sim'], dt)
    blank_idx = np.arange(1./dt * params['t_stimulus'], 1. / dt * (params['t_stimulus'] + params['t_blank']))

    if (my_units == None):
        my_units = xrange(tp.shape[0])
    else:
        my_units = xrange(my_units[0], my_units[1])

    n_cells = len(my_units)
    L_input = np.zeros((n_cells, time.shape[0]))
    for i_time, time_ in enumerate(time):
        if (i_time % 100 == 0):
            print "t:", time_
        L_input[:, i_time] = get_input(tuning_prop[my_units, :], params, time_/params['t_stimulus'])
        L_input[:, i_time] *= params['f_max_stim']

    for i_time in blank_idx:
        L_input[:, i_time] = 0.


    for i_, unit in enumerate(my_units):
        rate_of_t = np.array(L_input[i_, :]) 
        output_fn = params['input_rate_fn_base'] + str(unit) + '.npy'
        np.save(output_fn, rate_of_t)
        # each cell will get its own spike train stored in the following file + cell gid
        n_steps = rate_of_t.size
        st = []
        for i in xrange(n_steps):
            r = rnd.rand()
            if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                st.append(i * dt) 
#        output_fn = tgt_fn_base + str(column)
        output_fn = params['input_st_fn_base'] + str(unit) + '.npy'
        np.save(output_fn, np.array(st))



def get_plus_minus(rnd):
    """
    Returns either -1., or +1. as float.
    rnd -- should be your numpy.random RNG
    """
    return (rnd.randint(-1, 1) + .5) * 2

# 
# def get_input_delay(tuning_prop, params, t, motion_params=None, delay=.00, motion='dot'):
#     """
#     Similar to get_input but simpler + with a delay
# 
#     """
#     t -= delay
#     if motion_params == None:
#         motion_params = params['motion_params']
#     n_cells = tuning_prop[:, 0].size
#     blur_X, blur_V = params['blur_X'], params['blur_V'] #0.5, 0.5
#     # get the current stimulus parameters
#     x0, y0, u0, v0 = motion_params[0], motion_params[1], motion_params[2], motion_params[3]
#     x_stim = (x0 + u0 * t) % params['torus_width']
#     y_stim = (y0 + v0 * t) % params['torus_height']
#     if params['n_grid_dimensions'] == 2:
#         d_ij = torus_distance2D_vec(tuning_prop[:, 0], x_stim * np.ones(n_cells), tuning_prop[:, 1], y_stim * np.ones(n_cells))
#         L = np.exp(-.5 * (d_ij)**2 / blur_X**2
#                 -.5 * (tuning_prop[:, 2] - u0)**2 / blur_V**2
#                 -.5 * (tuning_prop[:, 3] - v0)**2 / blur_V**2)
#     else:
# #            d_ij = np.sqrt((tuning_prop[:, 0] - x * np.ones(n_cells))**2)
#         d_ij = torus_distance_array(tuning_prop[:, 0], x_stim * np.ones(n_cells))
#         L = np.exp(-.5 * (d_ij)**2 / blur_X**2 \
#                    -.5 * (tuning_prop[:, 2] - u0)**2 / blur_V**2)
#     return L
#
#
def get_input(tuning_prop, params, t, motion_params=None, delay=0., delay_compensation=0., contrast=.9, motion='dot'):
    """
    This function computes the input to each cell for one point in time t based on the given tuning properties.
    Knowing the velocity one can estimate the analytical response to
     - motion energy detectors
     - to a gaussian blob
    as a function of the distance between
     - the center of the receptive fields,
     - the current position of the blob.

    # TODO : prove this analytically to disentangle the different blurs (size of RF / size of dot)

    L range between 0 and 1
    Arguments:
        tuning_prop: 2-dim np.array;
            dim 0 is number of cells
            tuning_prop[:, 0] : x-position
            tuning_prop[:, 1] : y-position
            tuning_prop[:, 2] : u-position (speed in x-direction)
            tuning_prop[:, 3] : v-position (speed in y-direction)
        t: time (NOT in [ms]) in the period (not restricted to 0 .. 1)
        motion: type of motion
    """
#     t -= delay

    if motion_params == None:
        motion_params = params['motion_params']
    n_cells = tuning_prop[:, 0].size
    blur_X, blur_V = params['blur_X'], params['blur_V']
    # get the current stimulus parameters
    x0, y0, u0, v0 = motion_params[0], motion_params[1], motion_params[2], motion_params[3]
    # compensate for sensory delay
    x_stim = (x0 + u0 * t + tuning_prop[:, 2] * delay_compensation ) % params['torus_width']
    y_stim = (y0 + v0 * t + tuning_prop[:, 3] * delay_compensation ) % params['torus_height']
    if motion=='dot':

        if params['n_grid_dimensions'] == 2:
            d_ij = torus_distance2D_vec(tuning_prop[:, 0], x_stim * np.ones(n_cells), tuning_prop[:, 1], y_stim * np.ones(n_cells))
            L = np.exp(-.5 * (d_ij)**2 / blur_X**2
                    -.5 * (tuning_prop[:, 2] - u0)**2 / blur_V**2
                    -.5 * (tuning_prop[:, 3] - v0)**2 / blur_V**2)
        else:
#            d_ij = np.sqrt((tuning_prop[:, 0] - x * np.ones(n_cells))**2)
            d_ij = torus_distance_array(tuning_prop[:, 0], x_stim * np.ones(n_cells))
            L = np.exp(-.5 * (d_ij)**2 / blur_X**2 \
                       -.5 * (tuning_prop[:, 2] - u0)**2 / blur_V**2)
    elif motion=='bar':
        blur_theta = params['blur_theta']
        if params['n_grid_dimensions'] == 2:
            d_ij = torus_distance2D_vec(tuning_prop[:, 0], x_stim * np.ones(n_cells), tuning_prop[:, 1], y_stim * np.ones(n_cells))
        else:
            d_ij = torus_distance_array(tuning_prop[:, 0], x_stim * np.ones(n_cells))
        # compute the motion energy input to all cells
        # then for all cells we have to check if they get stimulate by any x and y on the bar
        L = np.exp(-.5 * (d_ij)**2 / blur_X**2
                -.5 * (tuning_prop[:, 2] - u0)**2 / blur_V**2
                -.5 * (tuning_prop[:, 3] - v0)**2 / blur_V**2
                -.5 * (tuning_prop[:, 4] - orientation)**2 / blur_theta**2)
    else: # to be implemented: 'oriented dot' (bar)
        print 'Unspecified motion in get_input:', motion
    return L

        # OLD
        # define the parameters of the motion
#        x0, y0, u0, v0 = motion_params
#        blur_X, blur_V = params['blur_X'], params['blur_V'] #0.5, 0.5
#        x, y = (x0 + u0*t) % params['torus_width'], (y0 + v0*t) % params['torus_height'] # current position of the blob at time t assuming a perfect translation
        # compute the motion energy input to all cells
#        L = np.exp(-.5 * ((torus_distance2D_vec(tuning_prop[:, 0], x*np.ones(n_cells), tuning_prop[:, 1], y*np.ones(n_cells)))**2 / blur_X**2)
#                -.5 * (tuning_prop[:, 2] - u0)**2 / blur_V**2
#                -.5 * (tuning_prop[:, 3] - v0)**2 / blur_V**2
#                )
#    L = (1. - contrast) + contrast * L


def set_receptive_fields(self, params, tuning_prop):
    """
    Can be called only after set_tuning_prop.
    Receptive field sizes increase linearly depending on their relative position.
    """
#    rfs[:, 0] = params['rf_size_x_gradient'] * np.abs(tuning_prop[:, 0] - .5) + params['rf_size_x_min']
#    rfs[:, 1] = params['rf_size_y_gradient'] * np.abs(tuning_prop[:, 1] - .5) + params['rf_size_y_min']
    rfs = np.zeros((tuning_prop[:, 0].size, 4))
    rfs[:, 0] = params['blur_X']
    rfs[:, 1] = params['blur_X']
    rfs[:, 2] = params['rf_size_vx_gradient'] * np.abs(tuning_prop[:, 2])# + params['rf_size_vx_min']
    rfs[:, 3] = params['rf_size_vy_gradient'] * np.abs(tuning_prop[:, 3])# + params['rf_size_vy_min']
    rfs[rfs[:, 2] < params['rf_size_vx_min'], 2] = params['rf_size_vx_min']
    rfs[rfs[:, 3] < params['rf_size_vy_min'], 3] = params['rf_size_vy_min']
    return rfs


def select_well_tuned_cells_2D_with_orientation(tp, mp, n_cells):
    """
    mp -- [x, y, vx, vy, orientation]
    """
    w_pos = 10.
    x_diff = (tp[:, 0] - mp[0])**2 * w_pos + (tp[:, 1] - mp[1])**2 * w_pos + (tp[:, 2] - mp[2])**2 + (tp[:, 3] - mp[3])**2 + (tp[:, 4] - mp[4])**2
    idx_sorted = np.argsort(x_diff)
    return idx_sorted[:n_cells]


def select_well_tuned_cells_1D(tp, mp, n_cells):
    """
    mp -- [x, y, vx, vy, orientation]
    """
    w_pos = 1.
    x_diff = torus_distance_array(tp[:, 0], mp[0]) * w_pos + np.abs(tp[:, 2] - mp[2])
    idx_sorted = np.argsort(x_diff)
    return idx_sorted[:n_cells]


def distribute_list(l, n_proc, pid):
    """
    l: list of elements to be distributed among n_proc processors
    pid: (int) process id of the process calling this function
    n_proc: total number of processors
    Returns a list to be assigned to the processor with id pid
    """
    n_files = len(l)
    n_files_per_proc = int(n_files / n_proc)
    R = n_files % n_proc
    offset = min(pid, R)
    file_id_min = int(pid * n_files_per_proc + offset)
    if (pid < R):
        file_id_max = file_id_min + n_files_per_proc + 1
    else:
        file_id_max = file_id_min + n_files_per_proc
    sublist = [l[i] for i in range(file_id_min, file_id_max)]
    return sublist


def distribute_n(n, n_proc, pid):
    """
    l: list of elements to be distributed among n_proc processors
    pid: (int) process id of the process calling this function
    n_proc: total number of processors
    Returns the min and max index to be assigned to the processor with id pid
    """
    n_per_proc = int(n / n_proc)
    R = n % n_proc
    offset = min(pid, R)
    n_min = int(pid * n_per_proc + offset)
    if (pid < R):
        n_max = int(n_min + n_per_proc + 1)
    else:
        n_max = int(n_min + n_per_proc)
    return (n_min, n_max)


def euclidean(x, y):
    return distance.euclidean(x, y)

def gauss(x, mu, sigma):
    return np.exp( - (x - mu)**2 / (2 * sigma ** 2))

def get_time_of_max_stim(tuning_prop, motion_params):
    """
    This function assumes motion with constant velocity, starting at x0 y0.
    Based on the spatial receptive field (RF: mu_x, mu_y) of the cell (column) the time when the stimulus is closest
    to the RF.
    t_min = (mu_x * u0 + mu_y * v0 - v0 * y0 + u0 * x0) / (v0**2 + u0**2)
    """
    x_i, y_i, u_i, v_i = tuning_prop
    x_stim, y_stim, u_stim, v_stim = motion_params
    t_min = (u_stim * (x_i - x_stim) + v_stim * (y_i - y_stim)) / (u_stim**2 + v_stim**2)
    return t_min


def get_time_of_max_response(spikes, range=None, n_binsizes=1):
    """
    For n_binsizes the average max response will be computed.
    Average max response is the mean of those bins that have the maximum number of spikes in it.
    """
    binsizes = np.linspace(5, 50, n_binsizes)
    t_max_depending_on_binsize = np.zeros((n_binsizes, 2))
    for i in xrange(n_binsizes):
        n_bins = binsizes[i]
        n, bins = np.histogram(spikes, bins=n_bins, range=range)
        binsize = round(bins[1] - bins[0])
        bins_with_max_height = (n == n.max()).nonzero()[0]
        times_with_max_response = binsize * bins_with_max_height + .5 * binsize
        t_max, t_max_std = times_with_max_response.mean(), times_with_max_response.std()
        t_max_depending_on_binsize[i, 0] = t_max
        t_max_depending_on_binsize[i, 1] = t_max_std
    return t_max_depending_on_binsize[:, 0].mean(), t_max_depending_on_binsize[:, 1].mean()


def set_limited_tuning_properties(params, y_range=(0, 1.), x_range=(0, 1.), u_range=(0, 1.), v_range=(0, 1.), cell_type='exc'):
    """
    This function uses the same algorithm as set_tuning_prop, but discards those cells
    that are out of the given parameter range.
    Purpose of this is to simulate a sub-network of cells with only limited tuning properties 
    and tune this network.
    """

    rnd.seed(params['tuning_prop_seed'])
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_theta = params['N_theta']
        n_v = params['N_V']
        n_rf_x = params['N_RF_X']
        n_rf_y = params['N_RF_Y']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_theta = params['N_theta_inh']
        n_v = params['N_V_INH']
        n_rf_x = params['N_RF_X_INH']
        n_rf_y = params['N_RF_Y_INH']
        if n_v == 1:
            v_min = params['v_min_tp'] + .5 * (params['v_max_tp'] - params['v_min_tp'])
            v_max = v_min
        else:
            v_max = params['v_max_tp']
            v_min = params['v_min_tp']

    tuning_prop = np.zeros((n_cells, 4))
    if params['log_scale']==1:
        v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
    else:
        v_rho = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_v,
                        endpoint=True, base=params['log_scale'])
    v_theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    parity = np.arange(params['N_V']) % 2

    RF = np.zeros((2, n_rf_x * n_rf_y))
    X, Y = np.mgrid[0:1:1j*(n_rf_x+1), 0:1:1j*(n_rf_y+1)]

    # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
    X, Y = X[1:, 1:], Y[1:, 1:]
    # Add to every even Y a half RF width to generate hex grid
    Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./N_RF
    RF[0, :] = X.ravel()
    RF[1, :] = Y.ravel()

    # wrapping up:
    index = 0
    random_rotation = 2*np.pi*rnd.rand(n_rf_x * n_rf_y) * params['sigma_RF_direction']
    neuron_in_range = np.zeros((n_cells, 1), dtype=bool)
    for i_RF in xrange(n_rf_x * n_rf_y):
        for i_v_rho, rho in enumerate(v_rho):
            for i_theta, theta in enumerate(v_theta):
                x_pos = (RF[0, i_RF] + params['sigma_RF_pos'] * rnd.randn()) % params['torus_width']
                y_pos = (RF[1, i_RF] + params['sigma_RF_pos'] * rnd.randn()) % params['torus_height']
                v_x = np.cos(theta + random_rotation[i_RF] + parity[i_v_rho] * np.pi / n_theta) \
                        * rho * (1. + params['sigma_RF_speed'] * rnd.randn())
                v_y = np.sin(theta + random_rotation[i_RF] + parity[i_v_rho] * np.pi / n_theta) \
                        * rho * (1. + params['sigma_RF_speed'] * rnd.randn())

                tuning_prop[index, 0] = x_pos 
                tuning_prop[index, 1] = y_pos
                tuning_prop[index, 2] = v_x
                tuning_prop[index, 3] = v_y
                if ((x_pos > x_range[0]) and (x_pos <= x_range[1]) \
                        and (y_pos > y_range[0]) and (y_pos <= y_range[1]) \
                        and (v_x > u_range[0]) and (v_x <= u_range[1]) \
                        and (v_y > v_range[0]) and (v_y <= v_range[1])):
                    neuron_in_range[index] = True
                index += 1

    n_cells_in_range = neuron_in_range.nonzero()[0].size
    tp_good = np.zeros((n_cells_in_range, 4))
    tp_good = tuning_prop[neuron_in_range.nonzero()[0], :]

    idx_out_of_range = np.ones((n_cells, 1), dtype=int)
    idx_out_of_range -= neuron_in_range
    n_cells_out_of_range = idx_out_of_range.nonzero()[0].size
    assert (n_cells_out_of_range + n_cells_in_range == n_cells), 'Number of cells in/out of range do not sum to one'
    tp_out_of_range = np.zeros((n_cells_out_of_range, 4))
    tp_out_of_range = tuning_prop[idx_out_of_range, :]
    
    return tp_good, tp_out_of_range





def set_tuning_prop(params, mode, cell_type):
    if params['n_grid_dimensions'] == 2:
        return set_tuning_prop_2D(params, mode, cell_type)
    else:
        return set_tuning_prop_1D(params, cell_type)


def set_tuning_prop_1D(params, cell_type='exc'):

    rnd.seed(params['tuning_prop_seed'])
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_v = params['N_V']
        n_rf_x = params['N_RF_X']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_v = params['N_V_INH']
        n_rf_x = params['N_X_INH']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']

    tuning_prop = np.zeros((n_cells, 4))
    if params['log_scale']==1:
        v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
    else:
        v_rho = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_v,
                        endpoint=True, base=params['log_scale'])

    n_orientation = params['n_orientation']
    orientations = np.linspace(0, np.pi, n_orientation, endpoint=False)
    xlim = (0, params['torus_width'])

    RF = np.linspace(0, params['torus_width'], n_rf_x, endpoint=False)
    index = 0
    random_rotation_for_orientation = np.pi*rnd.rand(n_rf_x * n_v * n_orientation) * params['sigma_RF_orientation']

    for i_RF in xrange(n_rf_x):
        for i_v_rho, rho in enumerate(v_rho):
            for orientation in orientations:
                for i_cell in xrange(params['n_exc_per_mc']):
                    tuning_prop[index, 0] = (RF[i_RF] + params['sigma_RF_pos'] * rnd.randn()) % params['torus_width']
                    tuning_prop[index, 1] = 0. 
                    tuning_prop[index, 2] = get_plus_minus(rnd) * rho * (1. + params['sigma_RF_speed'] * rnd.randn())
                    tuning_prop[index, 3] = 0. 
                    index += 1

    if index != n_cells:
        print '\nWARNING\n: \tutils.set_tuning_prop_1D mismatch between n_cells=%d and index=%d of tuning properties set for cell type %s\n\n' % (n_cells, index, cell_type)

    return tuning_prop


def set_tuning_prop_2D(params, mode='hexgrid', cell_type='exc'):
    """
    Place n_exc excitatory cells in a 4-dimensional space by some mode (random, hexgrid, ...).
    The position of each cell represents its excitability to a given a 4-dim stimulus.
    The radius of their receptive field is assumed to be constant (TODO: one coud think that it would depend on the density of neurons?)

    return value:
        tp = set_tuning_prop(params)
        tp[:, 0] : x-position
        tp[:, 1] : y-position
        tp[:, 2] : u-position (speed in x-direction)
        tp[:, 3] : v-position (speed in y-direction)

    All x-y values are in range [0..1]. Positios are defined on a torus and a dot moving to a border reappears on the other side (as in Pac-Man)
    By convention, velocity is such that V=(1,0) corresponds to one horizontal spatial period in one temporal period.
    This implies that in one frame, a translation is of  ``1. / N_frame`` in cortical space.
    """

    rnd.seed(params['tuning_prop_seed'])
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_theta = params['N_theta']
        n_v = params['N_V']
        n_rf_x = params['N_RF_X']
        n_rf_y = params['N_RF_Y']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_theta = params['N_theta_inh']
        n_v = params['N_V_INH']
        n_rf_x = params['N_RF_X_INH']
        n_rf_y = params['N_RF_Y_INH']
        if n_v == 1:
            v_min = params['v_min_tp'] + .5 * (params['v_max_tp'] - params['v_min_tp'])
            v_max = v_min
        else:
            v_max = params['v_max_tp']
            v_min = params['v_min_tp']

    tuning_prop = np.zeros((n_cells, 4))
    if mode=='random':
        # place the columns on a grid with the following dimensions
        x_max = int(round(np.sqrt(n_cells)))
        y_max = int(round(np.sqrt(n_cells)))
        if (params['n_cells'] > x_max * y_max):
            x_max += 1

        for i in xrange(params['n_cells']):
            tuning_prop[i, 0] = (i % x_max) / float(x_max)   # spatial rf centers are on a grid
            tuning_prop[i, 1] = (i / x_max) / float(y_max)
            tuning_prop[i, 2] = v_max * rnd.randn() + v_min
            tuning_prop[i, 3] = v_max * rnd.randn() + v_min

    elif mode=='hexgrid':
        if params['log_scale']==1:
            v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
        else:
            v_rho = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                            np.log(v_max)/np.log(params['log_scale']), num=n_v,
                            endpoint=True, base=params['log_scale'])
        v_theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        parity = np.arange(params['N_V']) % 2


        xlim = (0, params['torus_width'])
        ylim = (0, np.sqrt(3) * params['torus_height'])

        RF = np.zeros((2, n_rf_x * n_rf_y))
        X, Y = np.mgrid[0:1:1j*(n_rf_x+1), 0:1:1j*(n_rf_y+1)]
        X, Y = np.mgrid[xlim[0]:xlim[1]:1j*(n_rf_x+1), ylim[0]:ylim[1]:1j*(n_rf_y+1)]
    
        # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
        X, Y = X[1:, 1:], Y[1:, 1:]
        if n_rf_y > 1:
            # Add to every even Y a half RF width to generate hex grid
            Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./N_RF
        RF[0, :] = X.ravel()
        RF[1, :] = Y.ravel() 
        RF[1, :] /= np.sqrt(3) # scale to get a regular hexagonal grid
    
        # wrapping up:
        index = 0
        random_rotation = 2*np.pi*rnd.rand(n_rf_x * n_rf_y * n_v * n_theta) * params['sigma_RF_direction']
            # todo do the same for v_rho?
        for i_RF in xrange(n_rf_x * n_rf_y):
            for i_v_rho, rho in enumerate(v_rho):
                for i_theta, theta in enumerate(v_theta):
                    # for plotting this looks nicer, and due to the torus property it doesn't make a difference
                    tuning_prop[index, 0] = (RF[0, i_RF] + params['sigma_RF_pos'] * rnd.randn())# % params['torus_width']
                    tuning_prop[index, 1] = (RF[1, i_RF] + params['sigma_RF_pos'] * rnd.randn())# % params['torus_height']
                    tuning_prop[index, 2] = np.cos(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta) \
                            * rho * (1. + params['sigma_RF_speed'] * rnd.randn())
                    tuning_prop[index, 3] = np.sin(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta) \
                            * rho * (1. + params['sigma_RF_speed'] * rnd.randn())
                    index += 1

    return tuning_prop

def set_hexgrid_positions(params, NX, NY):

    RF = np.zeros((2, NX*NY))
    X, Y = np.mgrid[0:1:1j*(NX+1), 0:1:1j*(NY+1)]

    # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
    X, Y = X[1:, 1:], Y[1:, 1:]
    # Add to every even Y a half RF width to generate hex grid
    Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./N_RF
    RF[0, :] = X.ravel()
    RF[1, :] = Y.ravel()
    for i in xrange(RF[0, :].size):
        RF[0, i] *= (1. + params['sigma_RF_pos'] * rnd.randn())
        RF[1, i] *= (1. + params['sigma_RF_pos'] * rnd.randn())

    return RF.transpose()


def get_predicted_stim_pos(tp):
    """
    For each cell this function calculates the target position based on the tuning_prop of the cell:
    x_predicted = (x_0 + v_0) % 1
    """
    n_pos = tp[:, 0].size
    pos = np.zeros((n_pos, 2))
    for i in xrange(n_pos):
        x_predicted = (tp[i, 0] + tp[i, 2]) % 1
        y_predicted = (tp[i, 1] + tp[i, 3]) % 1
        pos[i, 0], pos[i, 1] = x_predicted, y_predicted
    return pos

    


def spatial_readout(particles, N_X, N_Y, hue, hue_zoom, fig_width, width, ywidth, display=True):
    """
    Reads-out particles into a probability density function in spatial space.

    Instead of a quiver plot, it makes an histogram of the density of particles by: 1) transforming a particle set in a 3 dimensional (x,y, \theta) density (let's forget about speed norm), (2) showing direction spectrum as hue and spatial density as transparency

    Marginalization over all speeds.

    Input
    -----
    N particles as a particles.shape array

    Output
    ------
    a position PDF

    """

    x = particles[0, ...].ravel()
    y = particles[1, ...].ravel()
    # we weight the readout by the weight of the particles
    weights = particles[4, ...].ravel()

    x_edges = np.linspace(0., 1., N_X)
    y_edges = np.linspace(0., 1., N_Y)

    if hue:
        N_theta_=3 # the 3 RGB channels
        # TODO : velocity angle histogram as hue
        u = particles[3, ...].ravel()
        v = particles[2, ...].ravel()   
        # TODO:  rotate angle because we are often going on the horizontal on the right /// check in other plots, hue = np.arctan2(u+v, v-u)/np.pi/2 + .5 /// np.arctan2 is in the [np.pi, np.pi] range, cm.hsv takes an argument in [0, 1]
        v_theta = np.arctan2(u+v, v-u)
        if hue_zoom:
            v_theta_edges = np.linspace(-np.pi/4-np.pi/8, -np.pi/4+np.pi/8, N_theta_ + 1 )
        else:
            v_theta_edges = np.linspace(-np.pi, np.pi, N_theta_ + 1 )# + pi/N_theta
            
        sample = np.hstack((x[:,np.newaxis], y[:,np.newaxis], v_theta[:,np.newaxis]))
#        print v_theta.shape, sample.shape
        bin_edges = (x_edges, y_edges, v_theta_edges)
#        print np.histogramdd(sample, bins =bin_edges, normed=True, weights=weights)
        v_hist, edges_ = np.histogramdd(sample, bins=bin_edges, normed=True, weights=weights)
        v_hist /= v_hist.sum()
                             
    else:        
        v_hist, x_edges_, y_edges_ = np.histogram2d(x, y, (x_edges, y_edges), normed=True, weights=weights)
        v_hist /= v_hist.sum()

#    print fig_width * np.float(N_Y) / N_X, width * np.float(N_Y) / N_X
    ywidth = width * np.float(N_Y) / N_X
    if display:
#        print fig_width, fig_width * np.float(N_Y) / N_X
#        fig = pylab.figure()#figsize=(fig_width, fig_width * np.float(N_Y) / N_X))
        # BK: the following code has been modified to remove import pylab from the utils.py file in order to run it on the cluster
        fig = None
        a = fig.add_axes([0., 0., 1., 1.])
        if hue:
# TODO : overlay image and use RGB(A) information
#            print v_hist, v_hist.min(), v_hist.max(), np.flipud( np.fliplr(im_).T
#            a.imshow(-np.log(np.rot90(v_hist)+eps_hist), interpolation='nearest')
            a.imshow(np.fliplr(np.rot90(v_hist/v_hist.max(),3)), interpolation='nearest', origin='lower', extent=(-width/2, width/2, -ywidth/2., ywidth/2.))#, vmin=0., vmax=v_hist.max())
#            pylab.axis('image')
        else:
#            a.pcolor(x_edges, y_edges, v_hist, cmap=pylab.bone(), vmin=0., vmax=v_hist.max(), edgecolor='k')
            pass 
        a.axis([-width/2, width/2, -ywidth/2., ywidth/2.])

        return fig, a
    else:
        return v_hist, x_edges, y_edges


def threshold_weights(connection_matrix, w_thresh):
    """
    Elements in connection_matrix below w_thresh will be set to zero.
    """
    for i in xrange(connection_matrix[:, 0].size):
        for j in xrange(connection_matrix[0, :].size):
            if connection_matrix[i, j] < w_thresh:
                connection_matrix[i, j] = 0.0
    return connection_matrix


def get_spiketimes(all_spikes, gid, gid_idx=0, time_idx=1):
    """
    Returns the spikes fired by the cell with gid the
    all_spikes: 2-dim array containing all spiketimes (raw data, if NEST: gids are 1-aligned)
    gid_idx: is the column index in the all_spikes array containing GID information
    time_idx: is the column index in the all_spikes array containing time information
    """
    if all_spikes.size == 0:
        return np.array([])
    else:
        idx_ = (all_spikes[:, gid_idx] == gid).nonzero()[0]
        spiketimes = all_spikes[idx_, time_idx]
        return spiketimes

def get_nspikes(spiketimes_fn_merged, n_cells=0, get_spiketrains=False):
    """
    Returns an array with the number of spikes fired by each cell.
    nspikes[gid]
    if n_cells is not given, the length of the array will be the highest gid (not advised!)
    """
    d = np.loadtxt(spiketimes_fn_merged)
    if (n_cells == 0):
        n_cells = 1 + int(np.max(d[:, 1]))# highest gid
    nspikes = np.zeros(n_cells)
    spiketrains = [[] for i in xrange(n_cells)]
    if (d.size == 0):
        if get_spiketrains:
            return nspikes, spiketrains
        else:
            return spiketrains
    # seperate spike trains for all the cells
    if d.shape == (2,):
        nspikes[int(d[1])] = 1
        spiketrains[int(d[1])] = [d[0]]
    else:
        for i in xrange(d[:, 0].size):
            spiketrains[int(d[i, 1])].append(d[i, 0])
        for gid in xrange(n_cells):
            nspikes[gid] = len(spiketrains[gid])
    if get_spiketrains:
        return nspikes, spiketrains
    else:
        return nspikes


def get_connection_center_of_mass(connection_gids, weights, tp):
    cms = [0., 0.] 
    tp_conn = tp[connection_gids, :]
    M = np.sum(weights)
    cms[0] = np.sum(tp_conn[:, 0] * weights) / M
    cms[1] = np.sum(tp_conn[:, 2] * weights) / M
    return cms


def get_sources(conn_list, target_gid):
    idx = conn_list[:, 1] == target_gid
    sources = conn_list[idx, :]
    return sources


def get_targets(conn_list, source_gid):
    idx = conn_list[:, 0] == source_gid
    targets = conn_list[idx, :]
    return targets


def get_cond_in(nspikes, conn_list, target_gid):
    cond_in = 0.
    srcs = get_sources(conn_list, target_gid)
    for i in xrange(len(srcs)):
        src_id = srcs[i, 0]
        cond_in += nspikes[src_id] * srcs[i, 2]
    return cond_in


def get_spiketrains(spiketimes_fn_or_array, n_cells=0):
    """
    Returns an list of spikes fired by each cell
    if n_cells is not given, the length of the array will be the highest gid (not recommended!)
    """
    if type(spiketimes_fn_or_array) == type(np.array([])):
        d = spiketimes_fn_or_array
    else:
        d = np.loadtxt(spiketimes_fn_or_array)
    if (n_cells == 0):
        n_cells = 1 + np.max(d[:, 1])# highest gid
    spiketrains = [[] for i in xrange(n_cells)]
    # seperate spike trains for all the cells
    if d.size == 0:
        return spiketrains
    elif d.shape == (2,):
        spiketrains[int(d[1])] = [d[0]]
    else:
        for i in xrange(d[:, 0].size):
            spiketrains[int(d[i, 1])].append(d[i, 0])
    return spiketrains


def get_grid_pos(x0, y0, xedges, yedges):
    x_index, y_index = len(xedges)-1, len(yedges)-1
    for (ix, x) in enumerate(xedges[1:]):
        if x0 <= x:
            x_index = ix
            break
            
    for (iy, y) in enumerate(yedges[1:]):
        if y0 <= y:
            y_index = iy
            break
    return (x_index, y_index)

def get_grid_pos_1d(x0, xedges):

    x_index = len(xedges)-1
    for (ix, x) in enumerate(xedges[1:]):
        if x0 <= x:
            x_index = ix
            break
            
    return x_index

def convert_hsl_to_rgb(h, s, l):
    """
    h : [0, 360) degree
    s : [0, 1]
    l : [0, 1]

    returns (r,g,b) tuple with values in range [0, 1]
    Source of the formula: http://en.wikipedia.org/wiki/HSV_color_space#Conversion_from_RGB_to_HSL_or_HSV
    """
    c = (1. - np.abs(2*l - 1.)) * s # c = chroma
    h_ = h / 60.
    x = c * (1 -  np.abs(h_ % 2 - 1))

    if 0 <= h_ and h_ < 1:
        r, g, b = c, x, 0
    elif 1 <= h_ and h_ < 2:
        r, g, b = x, c, 0
    elif 2 <= h_ and h_ < 3:
        r, g, b = 0, c, x
    elif 3 <= h_ and h_ < 4:
        r, g, b = 0, x, c
    elif 4 <= h_ and h_ < 5:
        r, g, b = x, 0, c
    elif 5 <= h_ and h_ < 6:
        r, g, b = c, 0, x
    else: 
        r, g, b = 0, 0, 0

    # match lightness
    m = l - .5 * c
    r_, g_, b_ = r+m, g+m, b+m
    # avoid negative values due to precision problems
    r_ = max(r_, 0)
    g_ = max(g_, 0)
    b_ = max(b_, 0)
    return (r_, g_, b_)


def sort_gids_by_distance_to_stimulus(tp, mp, params, local_gids=None):
    """
    This function return a list of gids sorted by the distances between cells and the stimulus.
    It calculates the minimal distances between the moving stimulus and the spatial receptive fields of the cells 
    and adds the distances between the motion_parameters and the tuning_properties of each cell.

    Arguments:
        tp: tuning_properties array 
        tp[:, 0] : x-pos
        tp[:, 1] : y-pos
        tp[:, 2] : x-velocity
        tp[:, 3] : y-velocity

        mp: motion_parameters (x0, y0, u0, v0)

    """
    if local_gids == None: 
        n_cells = tp[:, 0].size
    else:
        n_cells = len(local_gids)
    x_dist = np.zeros(n_cells) # stores minimal distance in space between stimulus and cells
    for i in xrange(n_cells):
        x_dist[i], spatial_dist = get_min_distance_to_stim(mp, tp[i, :], params)

    cells_closest_to_stim_pos = x_dist.argsort()
    if local_gids != None:
        gids_closest_to_stim = local_gids[cells_closest_to_stim_pos]
        return gids_closest_to_stim, x_dist[cells_closest_to_stim_pos]#, cells_closest_to_stim_velocity
    else:
        return cells_closest_to_stim_pos, x_dist[cells_closest_to_stim_pos]#, cells_closest_to_stim_velocity


def get_min_distance_to_stim(mp, tp_cell, params):
    """
    mp : motion_parameters (x,y,u,v)
    tp_cell : same format as mp
    n_steps: steps for calculating the motion path
    """
    try:
        if params['abstract'] == False:
            time = np.arange(0, params['t_sim'], params['dt_rate'])
        else:
            time = np.arange(0, params['t_sim'], 50 * params['dt_rate'])
    except:# use larger time step to numerically find minimum distance --> faster
        time = np.arange(0, params['t_sim'], 50 * params['dt_rate'])
    spatial_dist = np.zeros(time.shape[0])
    x_pos_stim = mp[0] + mp[2] * time / params['t_stimulus']
    y_pos_stim = mp[1] + mp[3] * time / params['t_stimulus']
    spatial_dist = torus_distance_array(tp_cell[0], x_pos_stim)**2 + torus_distance_array(tp_cell[1], y_pos_stim)**2
    min_spatial_dist = np.sqrt(np.min(spatial_dist))
    velocity_dist = np.sqrt((tp_cell[2] - mp[2])**2 + (tp_cell[3] - mp[3])**2)
    dist =  min_spatial_dist + velocity_dist
    return dist, min_spatial_dist
    

#def torus_distance(x0, x1):
#    return x0 - x1

def torus(x, w=1.):
    """
    center x in the range [-w/2., w/2.]
    To see what this does, try out:
    >> x = np.linspace(-4,4,100)
    >> pylab.plot(x, torus(x, 2.))
    """
    return np.mod(x + w/2., w) - w/2.

def torus_distance_array(x0, x1, w=1.):
    """
    Compute the 1-D distance on a torus for arrays
    w -- torus width
    """
    return np.minimum(np.abs(x0 - x1), w - np.abs(x0 - x1))

def torus_distance(x0, x1):
    """
    1-D torus like distance
    """
    return min(abs(x0 - x1), 1. - abs(x0 - x1))

def torus_distance2D(x1, x2, y1, y2, w=1., h=1.):
    """
    w and h are the width (x) and height (y) of the grid, respectively.
    """
    return np.sqrt(np.min(np.abs(x1 - x2), np.abs(w - np.abs(x1 - x2)))**2 + np.min(np.abs(y1 - y2), np.abs(h - np.abs(y1-y2)))**2)
    # if not on a torus:
#    return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2)

def torus_distance2D_vec(x1, x2, y1, y2, w=1., h=1.):
    """
    w and h are the width (x) and height (y) of the grid, respectively.
    """
    return np.sqrt(np.minimum(np.abs(x1 - x2), np.abs(w - np.abs(x1 - x2)))**2 + np.minimum(np.abs(y1 - y2), np.abs(h - np.abs(y1-y2)))**2)

def gather_conn_list(comm, data, n_total, output_fn):
    """
    This function makes all processes with pc_id > 1 send their data to process 0.
    pc_id: process id of the calling process
    n_proc: total number of processes
    data: data to be sent
    n_total: total number of elements to be stored
    """

    pc_id, n_proc = comm.rank, comm.size
    # receiving data
    if (pc_id == 0):
        output_data = np.zeros((n_total, 4))
        # copy the data computed by pc_id 0
        line_pnt = data[:,0].size
        output_data[0:line_pnt, :] = data
        for sender in xrange(1, n_proc):
            # each process sends a list with four elements: [(src, tgt, w, d), ... ]
            data = comm.recv(source=sender, tag=sender)
            print "Master receives from proc %d" % sender, data
            new_line_pnt = line_pnt + data[:, 0].size
            # write received data to output buffer
            output_data[line_pnt:new_line_pnt, :] = data
            line_pnt = new_line_pnt

        print "DEBUG, Master proc saves weights to", output_fn
        np.savetxt(output_fn, output_data)
            
    # sending data
    elif (pc_id != 0):
#            print  pc_id, "sending data to master"
        comm.send(data, dest=0, tag=pc_id)


def gather_bias(comm, data, n_total, output_fn):
    """
    This function makes all processes with pc_id > 1 send their data to process 0.
    pc_id: process id of the calling process
    n_proc: total number of processes
    data: data to be sent; here: dictionary = { gid : bias_value }
    n_total: total number of elements to be stored
    """
    pc_id, n_proc = comm.rank, comm.size

    # receiving data
    if (pc_id == 0):
        output_data = np.zeros(n_total)
        # copy the data computed by pc_id 0
        for gid in data.keys():
            if (data[gid] != None):
                output_data[gid] = data[gid]
        for sender in xrange(1, n_proc):
            # each process sends a list with four elements: [(src, tgt, w, d), ... ]
            data = comm.recv(source=sender, tag=sender)
            for gid in data.keys():
                if (data[gid] != None):
                    output_data[gid] = data[gid]
#            print "Master receives data of from %d of shape: " % sender, data.shape
            # write received data to output buffer
#            print "debug,", output_data[line_pnt:new_line_pnt, :].shape, data.shape

        print "DEBUG, Master proc saves bias to", output_fn
        np.savetxt(output_fn, output_data)
        
    # sending data
    elif (pc_id != 0):
#            print pc_id, "sending data to master"
        comm.send(data, dest=0, tag=pc_id)


def get_conn_dict(params, conn_fn, comm=None):
    """
    Returns a dictionary of dictionaries with target cell_gid as keys:
        conn_dict = { cell_gid : { 'sources' : [], 'w_in' : []}
        e.g.
        conn_dict[i] = { # i = target cell gid
                'sources'   : [j, k, x] # list of all cells connecting to cell i
                'w_in'      : [w_ji, w_ki, w_xi]
                }
            
    Currently, this can be used only for the exc-exc connections since those are stored in a file.
    TODO: Parallelize it
    """
    if comm != None:
        pc_id, n_proc = comm.rank, comm.size
    else:
        pc_id, n_proc = 0, 1 

    conn_dict = {}
    empty_dict = {'sources' : [], 'w_in' : []}
    for gid in xrange(params['n_exc']):
        conn_dict[gid] = copy.deepcopy(empty_dict)

    conns = np.loadtxt(conn_fn) # src tgt weight delay
    for row in xrange(conns[:,0].size):
        src = int(conns[row, 0])
        tgt = int(conns[row, 1])
        w = conns[row, 2]
        conn_dict[tgt]['sources'].append(src)
        conn_dict[tgt]['w_in'].append(w)

    return conn_dict


def get_incoming_connections(d, tgt_gids):
#    d = conn_list
    c_in = [[] for i in xrange(len(tgt_gids))]
    for i in xrange(d[:, 0].size):
        if d[i, 1] == tgt_gid:
            c_in.append(d[i, :])
    return c_in

def get_outgoing_connections(d, src_gid):
#    d = conn_list
    c_out = []
    for i in xrange(d[:, 0].size):
        if d[i, 0] == src_gid:
            c_out.append(d[i, :])
    return c_out


def linear_transformation(x, y_min, y_max):
    """
    x : the range to be transformed
    y_min, y_max : lower and upper boundaries for the range into which x
                   is transformed to
    Returns y = f(x), f(x) = m * x + b
    """
    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max:
        x_max = x_min * 1.0001
    return (y_min + (y_max - y_min) / (x_max - x_min) * (x - x_min))

def merge_files(input_fn_base, output_fn):

    cmd = 'cat %s* > %s' % (input_fn_base, output_fn)
    os.system(cmd)


def sort_cells_by_distance_to_stimulus(n_cells, verbose=True):
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
    tp = np.loadtxt(params['tuning_prop_means_fn'])
    mp = params['motion_params']
    indices, distances = sort_gids_by_distance_to_stimulus(tp , mp, params) # cells in indices should have the highest response to the stimulus
    print 'Motion parameters', mp
    print 'GID\tdist_to_stim\tx\ty\tu\tv\t\t'
    if verbose:
        for i in xrange(n_cells):
            gid = indices[i]
            print gid, '\t', distances[i], tp[gid, :]
    return indices, distances


def get_pmax(p_effective, w_sigma, conn_type):
    """
    When using isotropic connectivity, the connections are drawn based upon
    this formula:
        p_ij = p_max * np.exp(-d_ij / (2 * w_sigma_x**2))

    This function return the p_max to use in order to get a desired p_effective

    p_effective vs p_max has been simulated for different w_sigma values
    --> p_max is linearly dependent on p_effective, and the gradient is dependent on w_sigma (exponential decay works ok)
    Because inh and exc have different densities, they have different parameters to achieve the same p_eff
    
    The data have been acquired by fitting the gradient (p_max vs p_eff) versus w_sigma_x for w_sigma_x > 0.06
    """

    if conn_type == 'ee':
        fit_wsigma = [  8.69759077e+35,   1.15523952e-02,   1.68865787e+00, 5.21448789e-02]
    elif conn_type == 'ei':
        fit_wsigma = [  1.71248794e+32,   1.28067624e-02,   1.74747161e+00, 5.85147960e-02]
    elif conn_type == 'ie':
        fit_wsigma = [5.47304944e+43,   9.56954239e-03,   1.84786854e+00, 4.26839514e-02]
    elif conn_type == 'ii':
        fit_wsigma = [2.21668319e+46,   9.05343215e-03,   1.76483061e+00, 4.01129051e-02]
    gradient  = fit_wsigma[0] * np.exp( - w_sigma**fit_wsigma[3] / fit_wsigma[1]) + fit_wsigma[2]
    print 'debug utils.get_pmax gradient for %s ws %.1e: %.3e' % (conn_type, w_sigma, gradient)
    p_max = gradient * p_effective

    return p_max
    


def scale_input_frequency(x):
    """
    How these optimal values come about:
      - run run_input_analysis.py, analyse_input.py to get different the average number of input
      spike into a cell for different blur_x/v values
      - run get_input_scaling_factor.py with the file written by analyse_input.py 
      to find the fitted function and these parameters

      Purpose of all this is to have similar input excitation into the whole network for different blur_x/v values
    """
    p = [2.64099116e-01,   3.27055672e-02,  9.66385641e-03,   3.24742098e-03, -4.62469854e-05,  -1.34801304e-06]
    y = p[0] + p[1] / x + p[2] / x**2 + p[3] / x**3 + p[4] / x**4 + p[5] / x**5
    return y



def resolve_src_tgt(conn_type, params):
    """
    Deliver the correct source and target parameters based on conn_type
    """

    if conn_type == 'ee':
        n_src, n_tgt = params['n_exc'], params['n_exc']
#        tp_src = tuning_prop_exc
#        tp_tgt = tuning_prop_exc
        syn_type = 'excitatory'

    elif conn_type == 'ei':
        n_src, n_tgt = params['n_exc'], params['n_inh']
#        tp_src = tuning_prop_exc
#        tp_tgt = tuning_prop_inh
        syn_type = 'excitatory'

    elif conn_type == 'ie':
        n_src, n_tgt = params['n_inh'], params['n_exc']
#        tp_src = tuning_prop_inh
#        tp_tgt = tuning_prop_exc
        syn_type = 'inhibitory'

    elif conn_type == 'ii':
        n_src, n_tgt = params['n_inh'], params['n_inh']
#        tp_src = tuning_prop_inh
#        tp_tgt = tuning_prop_inh
        syn_type = 'inhibitory'

    return (n_src, n_tgt, syn_type)
#    return (n_src, n_tgt, tp_src, tp_tgt, syn_type)


def resolve_src_tgt_with_tp(conn_type, params):
    """
    Deliver the correct source and target parameters based on conn_type
    """
    if conn_type == 'ee':
        n_src, n_tgt = params['n_exc'], params['n_exc']
        tuning_prop_exc = np.loadtxt(params['tuning_prop_means_fn'])
        tp_src = tuning_prop_exc
        tp_tgt = tuning_prop_exc
        syn_type = 'excitatory'

    elif conn_type == 'ei':
        n_src, n_tgt = params['n_exc'], params['n_inh']
        tuning_prop_exc = np.loadtxt(params['tuning_prop_means_fn'])
        tuning_prop_inh = np.loadtxt(params['tuning_prop_inh_fn'])
        tp_src = tuning_prop_exc
        tp_tgt = tuning_prop_inh
        syn_type = 'excitatory'

    elif conn_type == 'ie':
        n_src, n_tgt = params['n_inh'], params['n_exc']
        tuning_prop_exc = np.loadtxt(params['tuning_prop_means_fn'])
        tuning_prop_inh = np.loadtxt(params['tuning_prop_inh_fn'])
        tp_src = tuning_prop_inh
        tp_tgt = tuning_prop_exc
        syn_type = 'inhibitory'

    elif conn_type == 'ii':
        n_src, n_tgt = params['n_inh'], params['n_inh']
        tuning_prop_inh = np.loadtxt(params['tuning_prop_inh_fn'])
        tp_src = tuning_prop_inh
        tp_tgt = tuning_prop_inh
        syn_type = 'inhibitory'

    return (n_src, n_tgt, tp_src, tp_tgt)



def convert_to_url(fn):
    p = os.path.realpath('.')
    s = 'file://%s/%s' % (p, fn)
    return s

def get_figsize(fig_width_pt, portrait=True):
    """
    For getting a figure with an 'aesthetic' ratio
    """
    inches_per_pt = 1.0 / 72.0 # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0) / 2.0 # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt # width in inches
    fig_height = fig_width * golden_mean # height in inches
    if portrait:
        fig_size = (fig_height, fig_width) # exact figsize
    else:
        fig_size = (fig_width, fig_height) # exact figsize
    return fig_size
