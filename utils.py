"""
    This file contains a bunch of helper functions (in alphabetic order).
"""

import numpy as np
import numpy.random as rnd
import os
import copy
import re
import json
import itertools

def compute_stim_time(stim_params):
    """
    Based on the stim params, one the training stimulus takes different time to stimulate
    """
    if stim_params[2] > 0: # rightward movement
        xlim = 1.
    else:
        xlim = 0.
    dx = np.abs(stim_params[0] - xlim)
    t_exit = dx / np.abs(stim_params[2]) * 1000.
    return t_exit 


def get_gids_near_stim_nest(mp, tp_cells, n=1, ndim=1):
    """
    Get the cell GIDS (0 - aligned) from n cells,
    that are closest to the stimulus with parameters mp
    mp : target parameters (x, y, u, v)
    tp_cells : same format as mp
    ndim: number of spatial dimensions in the model
    if ndim == 1: use only the x-components to calculate the distance
    """
#    dy = (utils.torus_distance_array(tp_cells[:, 1], mp[1]))**2
#    dx = (utils.torus_distance_array(tp_cells[:, 0], mp[0]))**2
    if ndim == 1:
        dx = np.abs(tp_cells[:, 0] - mp[0])
        velocity_dist = np.abs(tp_cells[:, 2] - mp[2])
        summed_dist = dx + velocity_dist
    elif ndim == 2:
        dy = (tp_cells[:, 1] - mp[1])**2
        dx = np.abs(tp_cells[:, 0] - mp[0])
        velocity_dist = np.sqrt((tp_cells[:, 2] - mp[2])**2 + (tp_cells[:, 3] - mp[3])**2)
        summed_dist = dx + dy + velocity_dist
    gids_sorted = np.argsort(summed_dist)[:n] # 0 .. n-1
    nest_gids = gids_sorted + 1
    return nest_gids, summed_dist[gids_sorted]
    


def set_vx_tau_transformation_params(params, vmin, vmax):
    tau_max, tau_min = params['tau_zi_max'], params['tau_zi_min']
    if params['tau_vx_transformation_mode'] == 'linear':
        beta = (tau_max - tau_min * vmin / vmax) / (1. - vmin / vmax)
        params['tau_vx_param1'] = beta
        params['tau_vx_param2'] = (tau_min - beta) / vmax
    else:
        alpha = (tau_min * vmax - tau_max * vmin) / (tau_max - tau_min)
        beta = tau_max * (alpha - vmin)
        params['tau_vx_param1'] = beta
        params['tau_vx_param2'] = (tau_min - beta) / vmax


def load_params(param_fn):
    if os.path.isdir(param_fn):
        param_fn = os.path.abspath(param_fn) + '/Parameters/simulation_parameters.json'
    params = json.load(file(param_fn, 'r')) 
    return params


def convert_adjacency_list_to_connlist(adj_list, src_tgt='tgt'):
    """
    src_tgt -- if 'tgt' the keys in adj_list stand for the target GID
    """

    gids_key = []
    gids_ = []
    weights = []
    for gid_key in adj_list.keys():
        for i_, (gid_, w) in enumerate(adj_list[gid_key]):
            gids_key.append(gid_key)
            gids_.append(gid_)
            weights.append(w)
    output_array = np.zeros((len(gids_), 3)) 
    if src_tgt == 'tgt':
        output_array[:, 1] = np.array(gids_key)
        output_array[:, 0] = np.array(gids_)
    else:
        output_array[:, 0] = np.array(gids_key)
        output_array[:, 1] = np.array(gids_)
    output_array[:, 2] = np.array(weights)
    return output_array


def convert_adjacency_lists(params, iteration=0, verbose=False):
    """
    Convert the adjacency lists from target-indexed to source-indexed.
    """

    output_fn = params['adj_list_src_fn_base'] + 'merged.json'
    if os.path.exists(output_fn):
        print 'Loading data from:', output_fn
        f = file(output_fn, 'r')
        d = json.load(f)
        return d
    else:
        fns = []
        conn_mat_fn = params['adj_list_tgt_fn_base'] + 'AS_(\d)_(\d+).json'
        to_match = conn_mat_fn.rsplit('/')[-1]
        for fn in os.listdir(os.path.abspath(params['connections_folder'])):
            m = re.match(to_match, fn)
            if m:
                it_ = int(m.groups()[0])
                if it_ == iteration:
                    fn_abs_path = params['connections_folder'] +  fn
                    fns.append(fn_abs_path)
        adj_list_tgt = {}
        if not verbose:
            print 'utils.convert_adjacency_lists: Loading %d adjacency lists' % len(fns)
        for fn in fns:
            f = file(fn, 'r')
            if verbose:
                print 'utils.convert_adjacency_lists: Loading weights:', fn
            d = json.load(f)
            adj_list_tgt.update(d)
        
        adj_list_src = {}
        for tgt in adj_list_tgt.keys():
            for (src, w) in adj_list_tgt[tgt]:
                if int(src) not in adj_list_src.keys():
                    adj_list_src[int(src)] = []
                else:
                    adj_list_src[int(src)].append([int(tgt), float(w)])
        output_fn = params['adj_list_src_fn_base'] + 'merged.json'
        print 'Writing source - indexed adjacency list to :', output_fn
        f = file(output_fn, 'w')
        json.dump(adj_list_src, f, indent=2)
        f.flush()
        f.close()
        return adj_list_src


def get_adj_list_src_indexed(params):
    return convert_adjacency_lists(params)


def remove_empty_files(folder):
    for fn in os.listdir(folder):
        path = os.path.abspath(folder) + '/%s' % fn
        file_size = os.path.getsize(path)
        if file_size == 0:
            rm_cmd = 'rm %s' % (path)
            os.system(rm_cmd)

def remove_files_from_folder(folder):
    print 'Removing all files from folder:', folder
    path =  os.path.abspath(folder)
    cmd = 'rm  %s/*' % path
    print cmd
    os.system(cmd)


def get_spikes_within_interval(d, t0, t1, time_axis=1, gid_axis=0):
    """
    d -- spike data (each row contains one spike with (GID, time) as default format)
    """
    spikes = np.array(([], []))
    idx_0 = (d[:, time_axis] > t0).nonzero()[0]
    idx_1 = (d[:, time_axis] <= t1).nonzero()[0]
    idx_ = list(set(idx_0).intersection(set(idx_1)))
    idx = np.zeros(len(idx_), dtype=np.int)
    idx = idx_

    spikes = d[idx, time_axis]
    gids = d[idx, gid_axis]
    return (spikes, gids)

def get_spikes_for_gid(spike_data, gid, t_range=None):
    if t_range == None:
        idx = np.nonzero(gid == spike_data[:, 0])[0]
        return spike_data[idx, 1]
    else: 
        (spikes, gids) = get_spikes_within_interval(spike_data, t_range[0], t_range[1], time_axis=1, gid_axis=0)
        idx = np.nonzero(gid == gids)[0]
        return spikes[idx]


def transform_tauzi_from_vx(vx, params):
    """
    tau_zi ~ 1 / v_min
    """
    if params['tau_vx_transformation_mode'] == 'linear':
        tau_zi = params['tau_vx_param2'] * vx + params['tau_vx_param1']
    else:
        tau_zi = params['tau_vx_param1'] / (vx + params['tau_vx_param2']) # beta / (alpha + x)

    return tau_zi


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


def get_spike_fns(folder, spike_fn_pattern='exc_spikes'):
    fns = []
    for fn in os.listdir(folder):
        if fn.find(spike_fn_pattern) != -1:
            fns.append(fn)
    return fns


def get_spiketimes(all_spikes, nest_gid, gid_idx=0, time_idx=1):
    """
    Returns the spikes fired by the cell with gid the 
    all_spikes: 2-dim array containing all spiketimes (raw data, gids are 1-aligned thanks to NEST)
    gid_idx: is the column index in the all_spikes array containing GID information
    time_idx: is the column index in the all_spikes array containing time information
    if pynest == True: subtract 1 from gids
    """
    
    if all_spikes.size == 0:
        return np.array([])
    else:
        idx_ = (all_spikes[:, gid_idx] == nest_gid).nonzero()[0]
        spiketimes = all_spikes[idx_, time_idx]
        return spiketimes



def get_grid_index_mapping(values, bins):
    """
    Returns a 2-dim array (gid, grid_pos) mapping with values.size length, i.e. the indices of values 
    and the bin index to which each value belongs.
    values -- the values to be put in a grid
    bins -- list or array with the 1-dim grid bins 
    """

    bin_idx = np.zeros((len(values), 2), dtype=np.int)
    for i_, b in enumerate(bins):
        idx_in_b = (values > b).nonzero()[0]
        bin_idx[idx_in_b, 0] = idx_in_b
        bin_idx[idx_in_b, 1] = i_
    return bin_idx


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


def convert_spiketrain_to_trace(st, t_max, dt=0.1, spike_width=1):
    """Converts a single spike train into a binary trace
    Keyword arguments: 
    st --  spike train in the format [time, id]
    n  --  size of the trace to be returned
    spike_width -- number of time steps (in dt) for which the trace is set to 1
    Returns a np.array with st[i] = 1 if i in st[:, 0], st[i] = 0 else.
    """
    n = np.int(t_max / dt)
    trace = np.zeros(n)
    for t in st:
        idx_0 = np.int(t / dt)
        idx_1 = np.int(t / dt) + spike_width
        trace[idx_0:idx_1] = 1
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

def create_spike_trains_for_motion(tuning_prop, params, my_units=None, seed=None, protocol='congruent'):
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
        protocol: trajectory of stimulus is by default congruent. Protocol type is used in (motion_type = bar), congruent trajectory corresponds to 
        trajectory type in which orientation of bar outside and inside of CRF (classical receptive field is the same)
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
        L_input[:, i_time] = get_input(tuning_prop[my_units, :], params, time_/params['t_stimulus'], protocol = protocol)
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



def get_input(tuning_prop, rfs, params, predictor_params, motion='dot'):
    """
    This function computes the input to each cell for one point in time t based on the given tuning properties.

    Arguments:
        tuning_prop: 2-dim np.array; 
            dim 0 is number of cells
            tuning_prop[:, 0] : x-position
            tuning_prop[:, 1] : y-position
            tuning_prop[:, 2] : u-position (speed in x-direction)
            tuning_prop[:, 3] : v-position (speed in y-direction)
        t: time in the period (not restricted to 0 .. 1) NOT IN MS!
        predictor_params : x, y, y, v, theta of the current stimulus
        motion: type of motion (TODO: filename to movie, ... ???)

    """
    n_cells = tuning_prop[:, 0].size
    blur_X, blur_V = params['blur_X'], params['blur_V'] #0.5, 0.5
    rfs_x = rfs[:, 0]
    rfs_v = rfs[:, 2]
    blur_theta = params['blur_theta']
    # get the current stimulus parameters
    x_stim, y_stim, u_stim, v_stim, orientation = predictor_params[0], predictor_params[1], predictor_params[2], predictor_params[3], predictor_params[4]
        
    if motion=='dot':
        # compute the motion energy input to all cells
        """
            Knowing the velocity one can estimate the analytical response to 
             - motion energy detectors
             - to a gaussian blob
            as a function of the distance between 
             - the center of the receptive fields,
             - the current position of the blob.
             
            # TODO : prove this analytically to disentangle the different blurs (size of RF / size of dot)
            
            L range between 0 and 1
        """
        # to translate the initial static line at each time step with motion parameters
        if params['n_grid_dimensions'] == 2:
            d_ij = torus_distance2D_vec(tuning_prop[:, 0], x_stim * np.ones(n_cells), tuning_prop[:, 1], y_stim * np.ones(n_cells))
            L = np.exp(-.5 * (d_ij)**2 / blur_X**2 
                    -.5 * (tuning_prop[:, 2] - u_stim)**2 / (blur_V**2 + rfs_v**2)
                    -.5 * (tuning_prop[:, 3] - v_stim)**2 / (blur_V**2 + rfs_v**2))
        else:
#            print 'Debug', tuning_prop[:, 0].shape, x_stim, x_stim.shape, n_cells
#            d_ij = torus_distance_array(tuning_prop[:, 0], x_stim * np.ones(n_cells))
            d_ij = np.sqrt((tuning_prop[:, 0] - x_stim * np.ones(n_cells))**2)
            L = np.exp(-.5 * (d_ij)**2 / (blur_X**2 + rfs_x**2)\
                       -.5 * (tuning_prop[:, 2] - u_stim)**2 / (blur_V**2 + rfs_v**2))


#    if motion=='bar':
#        if params['n_grid_dimensions'] == 2:
#            d_ij = torus_distance2D_vec(tuning_prop[:, 0], x_stim * np.ones(n_cells), tuning_prop[:, 1], y_stim * np.ones(n_cells))
#        else:
#            d_ij = torus_distance_array(tuning_prop[:, 0], x_stim * np.ones(n_cells))
#        L = np.exp(-.5 * (d_ij)**2 / blur_X**2
#                -.5 * (tuning_prop[:, 2] - u_stim)**2 / blur_V**2
#                -.5 * (tuning_prop[:, 3] - v_stim)**2 / blur_V**2
#                -.5 * (tuning_prop[:, 4] - orientation)**2 / blur_theta**2)

        # ######## if bar is composed of several dots
#        x_init = np.round(np.linspace(0, 0.2, 5), decimals=2)# to control the height of bar with x_init range
#        y_init = np.arctan(orientation) * x_init
#        x, y = (x_init + u0*t) % params['torus_width'], (y_init + v0*t) % params['torus_height'] # current position of the blob at time t assuming a perfect translation
#        L = np.zeros(n_cells)
#        for x_i ,y_i in zip(x, y):
#            L_ = np.exp(-.5 * ((torus_distance2D_vec(tuning_prop[:, 0], x_i * np.ones(n_cells), tuning_prop[:, 1], y_i * np.ones(n_cells)))**2/blur_X**2)
#                -.5 * (tuning_prop[:, 2] - u0)**2/blur_V**2 
#                -.5 * (tuning_prop[:, 3] - v0)**2/blur_V**2
#                -.5 * (tuning_prop[:, 4] - orientation)**2 / blur_theta**2)
#            L += L_
                          
    return L



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


def set_tuning_prop(params, mode, cell_type):
    if params['n_grid_dimensions'] == 2:
        return set_tuning_prop_2D(params, mode, cell_type)
    else:
        if params['regular_tuning_prop']:
            return set_tuning_prop_1D_regular(params, cell_type)
        else:
            return set_tuning_prop_1D(params, cell_type)


def set_tuning_prop_1D(params, cell_type='exc'):

    rnd.seed(params['tuning_prop_seed'])
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_v = params['n_v']
        n_rf_x = params['n_rf_x']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_v = params['n_v_inh']
        n_rf_x = params['n_rf_x_inh']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']

    tuning_prop = np.zeros((n_cells, 5))
    rfs = np.zeros((n_cells, 2)) # receptive field sizes

    if params['log_scale']==1:
        v_rho_pos = np.linspace(v_min, v_max, num=n_v / 2, endpoint=True)
    else:
        v_rho_pos = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_v / 2,
                        endpoint=True, base=params['log_scale'])
    v_rho = np.zeros(n_v)
    v_rho[:n_v/2] = v_rho_pos
    v_rho[-(n_v/2):] = -v_rho_pos

    n_orientation = params['n_orientation']
    orientations = np.linspace(0, np.pi, n_orientation, endpoint=False)
    xlim = (0, params['torus_width'])
    RF = np.linspace(0, params['torus_width'], n_rf_x, endpoint=False)
    index = 0
    random_rotation_for_orientation = np.pi*rnd.rand(n_rf_x * n_v * n_orientation) * params['sigma_rf_orientation']
    for i_RF in xrange(n_rf_x):
        for i_v_rho, rho in enumerate(v_rho):
            plus_minus = get_plus_minus(rnd) # sign for the speed
            for orientation in orientations:
                for i_cell in xrange(params['n_exc_per_mc']):
                    tuning_prop[index, 0] = (RF[i_RF] + params['sigma_rf_pos'] * rnd.randn()) % params['torus_width']
                    tuning_prop[index, 1] = 0. # i_RF / float(n_rf_x) # y-pos 
                    tuning_prop[index, 2] = rho * (1. + params['sigma_rf_speed'] * rnd.randn()) 
#                    tuning_prop[index, 2] = (-1) ** (i_v_rho % 2) * rho * (1. + params['sigma_rf_speed'] * rnd.randn()) 
#                    tuning_prop[index, 2] = plus_minus * rho * (1. + params['sigma_rf_speed'] * rnd.randn())
                    tuning_prop[index, 3] = 0. # np.sin(theta + random_rotation[index]) * rho * (1. + params['sigma_rf_speed'] * rnd.randn())
                    tuning_prop[index, 4] = (orientation + random_rotation_for_orientation[index / params['n_exc_per_mc']]) % np.pi
                    index += 1
    return tuning_prop


def set_tuning_prop_1D_regular(params, cell_type='exc'):
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_v = params['n_v']
        n_rf_x = params['n_rf_x']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh_spec']
        n_v = params['n_v_inh']
        n_rf_x = params['n_rf_x_inh']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
#    v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
    v_rho = np.linspace(-v_max, v_max, num=n_v, endpoint=True)

    RF = np.linspace(0., 1., n_rf_x, endpoint=True)
    index = 0
    tuning_prop = np.zeros((n_cells, 4))
    for i_RF in xrange(n_rf_x):
        for i_v_rho, rho in enumerate(v_rho):
            for i_in_mc in xrange(params['n_exc_per_mc']):
                tuning_prop[index, 0] = RF[i_RF] + rnd.uniform(-params['sigma_rf_pos'] , params['sigma_rf_pos'])
                tuning_prop[index, 1] = 0.5 
                tuning_prop[index, 2] = rho + rnd.uniform(-params['sigma_rf_speed'] , params['sigma_rf_speed'])
                tuning_prop[index, 3] = 0. 
                index += 1
    assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
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
        n_theta = params['n_theta']
        n_v = params['n_v']
        n_rf_x = params['n_rf_x']
        n_rf_y = params['n_rf_y']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
        n_per_mc = params['n_exc_per_mc']
    else:
        print 'Not supported with modular structure (yet)'
        # possible solution: distribute tuning properties among the n_inh_per_mc neurons 

#        n_cells = params['n_inh']
#        n_theta = params['n_theta_inh']
#        n_v = params['n_v_inh']
#        n_rf_x = params['n_rf_x_inh']
#        n_rf_y = params['n_rf_y_inh']
#        n_per_mc = 1
#        if n_v == 1:
#            v_min = params['v_min_tp'] + .5 * (params['v_max_tp'] - params['v_min_tp'])
#            v_max = v_min
#        else:
#            v_max = params['v_max_tp']
#            v_min = params['v_min_tp']

    tuning_prop = np.zeros((n_cells, 5))
    if params['log_scale']==1:
        v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
    else:
        v_rho = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_v,
                        endpoint=True, base=params['log_scale'])
    v_theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    n_orientation = params['n_orientation']
    orientations = np.linspace(0, np.pi, n_orientation, endpoint=False)
#    orientations = np.linspace(-.5 * np.pi, .5 * np.pi, n_orientation)

    parity = np.arange(params['n_v']) % 2


    xlim = (0, params['torus_width'])
    ylim = (0, np.sqrt(3) * params['torus_height'])

    RF = np.zeros((2, n_rf_x * n_rf_y))
    X, Y = np.mgrid[xlim[0]:xlim[1]:1j*(n_rf_x+1), ylim[0]:ylim[1]:1j*(n_rf_y+1)]

    # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
    X, Y = X[1:, 1:], Y[1:, 1:]
    # Add to every even Y a half RF width to generate hex grid
    Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./n_RF
    RF[0, :] = X.ravel()
    RF[1, :] = Y.ravel() 
    RF[1, :] /= np.sqrt(3) # scale to get a regular hexagonal grid

    # wrapping up:
    index = 0
    random_rotation = 2*np.pi*rnd.rand(n_rf_x * n_rf_y * n_v * n_theta*n_orientation) * params['sigma_rf_direction']
    random_rotation_for_orientation = np.pi*rnd.rand(n_rf_x * n_rf_y * n_v * n_theta * n_orientation) * params['sigma_rf_orientation']

        # todo do the same for v_rho?
    for i_RF in xrange(n_rf_x * n_rf_y):
        for i_v_rho, rho in enumerate(v_rho):
            for i_theta, theta in enumerate(v_theta):
                for orientation in orientations:
                    for i_cell in xrange(params['n_exc_per_mc']):
                    # for plotting this looks nicer, and due to the torus property it doesn't make a difference
                        tuning_prop[index, 0] = (RF[0, i_RF] + params['sigma_rf_pos'] * rnd.randn()) % params['torus_width']
                        tuning_prop[index, 1] = (RF[1, i_RF] + params['sigma_rf_pos'] * rnd.randn()) % params['torus_height']
                        tuning_prop[index, 2] = np.cos(theta + random_rotation[index / n_per_mc] + parity[i_v_rho] * np.pi / n_theta) \
                                * rho * (1. + params['sigma_rf_speed'] * rnd.randn())
                        tuning_prop[index, 3] = np.sin(theta + random_rotation[index / n_per_mc] + parity[i_v_rho] * np.pi / n_theta) \
                                * rho * (1. + params['sigma_rf_speed'] * rnd.randn())
                        tuning_prop[index, 4] = (orientation + random_rotation_for_orientation[index / n_per_mc]) % np.pi
                        index += 1

    return tuning_prop

def set_hexgrid_positions(params, NX, NY):

    RF = np.zeros((2, NX*NY))
    X, Y = np.mgrid[0:1:1j*(NX+1), 0:1:1j*(NY+1)]

    # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
    X, Y = X[1:, 1:], Y[1:, 1:]
    # Add to every even Y a half RF width to generate hex grid
    Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./n_RF
    RF[0, :] = X.ravel()
    RF[1, :] = Y.ravel()
    for i in xrange(RF[0, :].size):
        RF[0, i] *= (1. + params['sigma_rf_pos'] * rnd.randn())
        RF[1, i] *= (1. + params['sigma_rf_pos'] * rnd.randn())

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


def get_nspikes(spiketimes_fn_or_array, n_cells=0, cell_offset=0, get_spiketrains=False, pynest=True):
    """
    Returns an array with the number of spikes fired by each cell.
    nspikes[gid]
    if n_cells is not given, the length of the array will be the highest gid (not advised!)
    """
    if pynest == True:
        gid_axis = 0
        time_axis = 1
        gid_offset = 0 + cell_offset
    else: # it's likely PyNN
        gid_axis = 1
        time_axis = 0
        gid_offset = cell_offset

    if (type(spiketimes_fn_or_array) == type ('')) or (type(spiketimes_fn_or_array) == type(unicode('a'))):
        print 'debug utils.get_nspikes loads:', spiketimes_fn_or_array
        d = np.loadtxt(spiketimes_fn_or_array)
    else:
        d = spiketimes_fn_or_array
    if (n_cells == 0):
        if pynest:
            n_cells = int(np.max(d[:, gid_axis]))# highest gid
        else:
            n_cells = 1 + int(np.max(d[:, gid_axis]))# highest gid

    nspikes = np.zeros(n_cells)
    spiketrains = [np.array([]) for i in xrange(n_cells)]

    if (d.size == 0):
        if get_spiketrains:
            return nspikes, spiketrains
        else:
            return spiketrains
    # seperate spike trains for all the cells
    if d.shape == (2,):
        nspikes[int(d[gid_axis]) - gid_offset] = 1
        spiketrains[int(d[gid_axis]) - gid_offset] = [d[time_axis]]
    else:
        gids = np.unique(d[:, gid_axis])
        for i_, gid in enumerate(gids):
            indices = (d[:, gid_axis] == gid).nonzero()[0]
            idx = int(gid - 1 - cell_offset) # if not pynest --> remove -1
            spiketrains[idx] = d[indices, time_axis] 
            nspikes[idx] = spiketrains[idx].size

    if get_spiketrains:
        return nspikes, spiketrains
    else:
        return nspikes

def get_random_connections(n_src, n_tgt, p_conn, allow_autapses=True, RNG=None):
    """

    RNG -- an instance of numpy.random.RandomState
    if the source population == target population, you should set allow_autapse=False
    if n_src != n_tgt, allow_autapses is ignored (because source population != target population)
    """
    if RNG == None:
        RNG = np.random

    if (n_src == n_tgt) and allow_autapses == False:
        invalid_idx = [(n_src + 1) * i_ for i_ in xrange(n_tgt - 1)]
        n_conn = np.int(np.round((n_src**2 - n_src) * p_conn))
        n_conn_possible = n_src ** 2 - n_src
    else:
        invalid_idx = []
        n_conn = np.int(np.round((n_src * n_tgt) * p_conn))
        n_conn_possible = n_src * n_tgt

    if n_conn == 0:
        return []
    print 'invalid_idx', invalid_idx
    list_of_connections = []
    combinations = itertools.product(range(n_src), range(n_tgt))
    n_idx = 0
#    while (n_idx != n_conn):
#    for i_ in xrange(n_conn):
#        rnd_idx = RNG.random_integers(0, len(n_conn_possible - 1), 1)


    randomly_selected_connection_idx = RNG.random_integers(0, n_conn_possible - 1, n_conn)
    n_idx = np.unique(randomly_selected_connection_idx).size
    print 'n_idx = ', n_idx, n_conn, n_conn_possible


    valid_idx = np.array(list(set(range(n_conn_possible)).difference(invalid_idx)))
    print 'n_conn_possible:', n_conn_possible
    print 'valid_idx', valid_idx

    conn_idx = np.sort(randomly_selected_connection_idx)
    print 'selected idx:', conn_idx
    print 'selected conn idx:', valid_idx[conn_idx]
    i_cnt = 0
    i_conn = 0
    for conn in combinations:
#        print 'debug', list_of_connections, i_conn, len(valid_idx)
        if i_cnt == valid_idx[conn_idx][i_conn]:
            list_of_connections.append((conn[0], conn[1]))
            i_conn += 1
            if i_conn == len(conn_idx):
                break
        i_cnt += 1
    return list_of_connections
#    i_cnt = 0
#    print 'n_idx', n_idx, p_conn, p_conn * n_conn_possible
#    print 'n_conn_possible', n_conn_possible
#    print 'conn_idx', conn_idx, conn_idx.size
#    print 'invalid_idx', invalid_idx

#    i_conn = 0
#    for i_ in xrange(n_src):
#        for j_ in xrange(n_tgt):
#            if len(invalid_idx) > 0:
#                if i_cnt != invalid_idx[0]:

#            if i_cnt == conn_idx[i_conn]:
#                print 'conn', i_, j_, i_cnt, i_conn
#                i_conn += 1
#            else:
#                print 'no conn', i_, j_, i_cnt

#            i_cnt += 1
#    

def get_sources(conn_list, target_gid):
    n = conn_list[:, 0].size 
    target = target_gid * np.ones(n)
    mask = conn_list[:, 1] == target
    sources = conn_list[mask, :]
    return sources


def get_targets(conn_list, source_gid):
    n = conn_list[:, 0].size 
#    source = source_gid * np.ones(n)
    mask = (conn_list[:, 0] == source_gid).nonzero()[0]
    targets = conn_list[mask, :]
    return targets


def get_cond_in(nspikes, conn_list, target_gid):
    cond_in = 0.
    srcs = get_sources(conn_list, target_gid)
    for i in xrange(len(srcs)):
        src_id = srcs[i, 0]
        cond_in += nspikes[src_id] * srcs[i, 2]
    return cond_in


#def get_spiketrains(spiketimes_fn_or_array, n_cells=0, pynest=True):
#    """
#    Returns a list of spikes fired by each cell
#    This function should be used for the format 
#        time    GID
#    if n_cells is not given, the length of the array will be the highest gid (not recommended!)
#    """
#    if pynest == True:
#        gid_axis = 0
#        time_axis = 1
#    else: # it's likely PyNN
#        gid_axis = 1
#        time_axis = 0
#    if type(spiketimes_fn_or_array) == type(np.array([])):
#        d = spiketimes_fn_or_array
#    else:
#        d = np.loadtxt(spiketimes_fn_or_array)
#    if (n_cells == 0):
#        n_cells = 1 + np.int(np.max(d[:, gid_axis]))# highest gid
#    spiketrains = [[] for i in xrange(n_cells)]
#    if d.size == 0:
#        return spiketrains
#    elif d.shape == (2,):
#        spiketrains[int(d[gid_axis])] = [d[time_axis]]
#    else:
#        for i in xrange(d[:, time_axis].size):
#            spiketrains[int(d[i, gid_axis])].append(d[i, time_axis])
#    return spiketrains

def find_files(folder, to_match):
    """
    Use re module to find files in folder and return list of files matching the 'to_match' string
    Arguments:
    folder -- string to folder
    to_match -- a string (regular expression) to match all files in folder
    """
    assert (to_match != None), 'utils.find_files got invalid argument'
    list_of_files = []
    for fn in os.listdir(folder):
        m = re.match(to_match, fn)
        if m:
            list_of_files.append(fn)
    return list_of_files


def get_filenames(folder, to_match, to_match_contains_folder=True, return_abspath=True):
    """
    Keyword arguments:
    folder -- the folder where to look for files
    to_match -- the filename to look for (re.match operation)
    to_match_contains_folder -- if to_match contains the folder in the name, the folder is seperated from the to_match string
    """
    fns = []
    # find files written by different processes
    if to_match_contains_folder:
        to_match = to_match.rsplit('/')[-1]
    for fn in os.listdir(os.path.abspath(folder)):
        m = re.match(to_match, fn)
        if m:
            path = os.path.abspath(folder) + '/' + fn
            fns.append(path)
    return fns


def get_spiketrains(spiketimes_fn_or_array, n_cells=0, pynest=True):
    """
    Returns a list of spikes fired by each cell
    This function should be used for the format 
        GID     time
    if n_cells is not given, the length of the array will be the highest gid (not recommended!)
    """
    if pynest == True:
        gid_axis = 0
        time_axis = 1
        gid_offset = 0
    else: # it's likely PyNN
        gid_axis = 1
        time_axis = 0
        gid_offset = 1

    if type(spiketimes_fn_or_array) == type(np.array([])):
        d = spiketimes_fn_or_array
    else:
        d = np.loadtxt(spiketimes_fn_or_array)
    

    # seperate spike trains for all the cells
    if d.size == 0:
        if n_cells == 0:
            return []
        else:
            spiketrains = [[] for i in xrange(n_cells)]
            return spiketrains
    elif d.shape == (2,):
        if (n_cells == 0):
            spiketrains = [d[gid_axis]]
        else:
            spiketrains = [[] for i in xrange(n_cells)]
            spiketrains[int(d[0])] = [d[1]]
    else:
        if (n_cells == 0):
            n_cells = 1 + np.int(np.max(d[:, gid_axis]))# highest gid
        spiketrains = [[] for i in xrange(n_cells)]
        for gid in xrange(gid_offset, n_cells + gid_offset):
            idx = (d[:, gid_axis] == gid).nonzero()[0]
            spiketrains[gid] = d[idx, time_axis]
#        for i in xrange(d[:, 0].size):
#            spiketrains[int(d[i, gid_axis])].append(d[i, time_axis])
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

        mp: motion_parameters (x0, y0, u0, v0, orientation)

    """
    if local_gids == None: 
        n_cells = tp[:, 0].size
    else:
        n_cells = len(local_gids)
    x_dist = np.zeros(n_cells) # stores minimal distance between stimulus and cells
    # it's a linear sum of spatial distance, direction-tuning distance and orientation tuning distance
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
    mp : motion_parameters (x, y, u, v, orientation)
    tp_cell : same format as mp
    n_steps: steps for calculating the motion path
    """
    time = np.arange(0, params['t_sim'], 50 * params['dt_rate'])
    spatial_dist = np.zeros(time.shape[0])
    x_pos_stim = mp[0] + mp[2] * time / params['t_stimulus']
    y_pos_stim = mp[1] + mp[3] * time / params['t_stimulus']
    spatial_dist = torus_distance_array(tp_cell[0], x_pos_stim)**2 + torus_distance_array(tp_cell[1], y_pos_stim)**2
    min_spatial_dist = np.sqrt(np.min(spatial_dist))

    velocity_dist = np.sqrt((tp_cell[2] - mp[2])**2 + (tp_cell[3] - mp[3])**2)

    if params['motion_type'] == 'bar':
        orientation_dist = np.sqrt((tp_cell[4] - mp[4])**2)
        dist =  min_spatial_dist + (velocity_dist + orientation_dist) * .1
    else:
        dist =  min_spatial_dist + velocity_dist
    return dist, min_spatial_dist
    

def torus(x, w=1.):
    """
    center x in the range [-w/2., w/2.]
    To see what this does, try out:
    >> x = np.linspace(-4,4,100)
    >> pylab.plot(x, torus(x, 2.))
    """
    return np.mod(x + w/2., w) - w/2.

def torus_distance_array(x0, x1):
    """
    Compute the 1-D distance on a torus for arrays
    """
    return np.minimum(np.abs(x0 - x1), 1. - np.abs(x0 - x1))

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


#def linear_transformation(x, y_min, y_max):
# renamed to:
def transform_linear(x, y_range, x_range=None):
    """
    x: single value or x-range to be linearly mapped into y_range
    y_min, y_max : lower and upper boundaries for the range into which x
                   is transformed to
    if x_range == None:
        x must be a list or array with more than one element, other wise no mapping is possible into y-range

    Returns y = f(x), f(x) = m * x + b
    """
    error_txt = 'Error: can not map a single value without x_range into the given y_range. \n \
            Please give x_range or use an array (or list) for the x parameter when calling utils.transform_linear!'
    if x_range == None:
        x_min = np.min(x)
        x_max = np.max(x)
        assert (x_min != x_max), error_txt
    else:
        x_min = np.min(x_range)
        x_max = np.max(x_range)
    y_min, y_max = y_range
    assert (x_min != x_max), error_txt
    return (y_min + (y_max - y_min) / (x_max - x_min) * (x - x_min))


def transform_quadratic(x, a, y_range, x_range=None):
    """
    Returns the function   f(x) = a * x**2 + b * x + c    for a value or interval.
    (however, this function internally works with the vertex form f(x) = a * (x - h)**2 + k (where (h, k) are the vertex' (x, y) coordinates
    The vertex coordinates are derived depending on x (or x_range), y_range and 'a'

    x -- either a list or array of x-values to be mapped, or a single value
        if x is a single value x_range can not be None
    a -- 'pos' or 'neg' (if 
        if a == 'pos': parabola is open upwards --> a > 0
           a == 'neg': parabola is open downwards --> a < 0
        if a > 0 and y_range[0] < y_range[1] --> quadratic increase from left (x_range[0]) to right x_range[1] (parabola open 'upwards')
        if a < 0 and y_range[0] < y_range[1] --> quadratic approach from x_range[0] to x_range[1] (parabola open 'downwards')
        if a > 0 and y_range[0] > y_range[1] --> quadratic decrease from left to right (parabola open upwards)
        if a < 0 and y_range[0] > y_range[1] --> quadratic decrease from left to right (parabola open downwards)
    """
    
    if a != 'pos' and a != 'neg':
        raise ValueError('The parameter \'a\' must be either a \'neg\' or \'pos\' and determines whether the parabola implementing your quadratic fit is open upwards or downwards')
    error_txt = 'Error: can not map a single value without x_range into the given y_range. \n \
            Please give x_range or use an array (or list) for the x parameter when calling utils.transform_linear!'
    if x_range != None:
        assert x_range[0] < x_range[1], 'Error: please give x_range as tuple with the smaller element first, e.g.  x_range = (0, 1) and NOT (1, 0)'
    else:
        x_range = (np.min(x), np.max(x))
    assert (x_range[0] != x_range[1]), error_txt

    assert a != 0, 'if you want a == 0, you should use utils.transform_linear'
    # determine the vertex and the other point of the parabola
    if a == 'neg' and y_range[0] < y_range[1]:
        vertex = (x_range[1], y_range[1])
        x0 = x_range[0]
        y0 = y_range[0]
    elif a == 'neg' and y_range[0] > y_range[1]:
        vertex = (x_range[0], y_range[0])
        x0 = x_range[1]
        y0 = y_range[1]
    elif a == 'pos' and y_range[0] < y_range[1]:
        vertex = (x_range[0], y_range[0])
        x0 = x_range[1]
        y0 = y_range[1]
    elif a == 'pos' and y_range[0] > y_range[1]:
        vertex = (x_range[1], y_range[1])
        x0 = x_range[0]
        y0 = y_range[0]
            
    alpha = (y0 - vertex[1]) / (x0 - vertex[0])**2
    f_x = alpha * (x - vertex[0])**2 + vertex[1]
    return f_x


def merge_connection_files(params, conn_type='ee', iteration=None):
    #conn_list_fn = params['merged_conn_list_%s' % conn_type]
    if iteration==None:
        merge_pattern = params['conn_list_%s_fn_base' % conn_type]
    else:
        merge_pattern = params['conn_list_%s_fn_base' % conn_type] + 'it%04d_' % iteration

#        params['conn_list_ii_fn_base'] = '%sconn_list_ii_' % (params['connections_folder'])
#        params['merged_conn_list_ii'] = '%smerged_conn_list_ii.dat' % (params['connections_folder'])
#    if not os.path.exists(conn_list_fn):
#        print 'Merging default connection files...'
    if iteration==None:
        merge_pattern = params['conn_list_%s_fn_base' % conn_type]
    else:
        merge_pattern = params['conn_list_%s_fn_base' % conn_type] + 'it%04d_' % iteration
    fn_out = params['merged_conn_list_%s' % conn_type]
#    merge_files(merge_pattern, fn_out)
    merge_and_sort_files(merge_pattern, fn_out)

def merge_spike_files_exc(params):
    cell_type = 'exc'
#    fn = params['exc_spiketimes_fn_merged']
    merge_and_sort_files(params['%s_spiketimes_fn_base' % cell_type], params['%s_spiketimes_fn_merged' % cell_type])

def merge_files(input_fn_base, output_fn):
    cmd = 'cat %s* > %s' % (input_fn_base, output_fn)
    os.system(cmd)

def merge_and_sort_files(merge_pattern, fn_out):
    rnd_nr1 = np.random.randint(0,10**8)
    rnd_nr2 = rnd_nr1 + 1
    # merge files from different processors
    tmp_file = "tmp_%d" % (rnd_nr2)
    cmd = "cat %s* > %s" % (merge_pattern, tmp_file)
    os.system(cmd)
    # sort according to cell id
    os.system("sort -gk 1 %s > %s" % (tmp_file, fn_out))
    os.system("rm %s" % (tmp_file))


def sort_cells_by_distance_to_stimulus(n_cells, verbose=False):
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    mp = params['mp_select_cells']
    indices, distances = sort_gids_by_distance_to_stimulus(tp, mp, params) # cells in indices should have the highest response to the stimulus
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
        tuning_prop_exc = np.loadtxt(params['tuning_prop_exc_fn'])
        tp_src = tuning_prop_exc
        tp_tgt = tuning_prop_exc
        syn_type = 'excitatory'

    elif conn_type == 'ei':
        n_src, n_tgt = params['n_exc'], params['n_inh']
        tuning_prop_exc = np.loadtxt(params['tuning_prop_exc_fn'])
        tuning_prop_inh = np.loadtxt(params['tuning_prop_inh_fn'])
        tp_src = tuning_prop_exc
        tp_tgt = tuning_prop_inh
        syn_type = 'excitatory'

    elif conn_type == 'ie':
        n_src, n_tgt = params['n_inh'], params['n_exc']
        tuning_prop_exc = np.loadtxt(params['tuning_prop_exc_fn'])
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



def get_plus_minus(rnd):
    """
    Returns either -1., or +1. as float.
    rnd -- should be your numpy.random RNG
    """
    return (rnd.randint(-1, 1) + .5) * 2


def convert_to_url(fn):
    p = os.path.realpath('.')
    s = 'file://%s/%s' % (p, fn)
    return s


def select_well_tuned_cells(tp, mp, n_cells, w_pos=5.):
#    w_pos = 10.
    x_diff = (tp[:, 0] - mp[0])**2 * w_pos + (tp[:, 1] - mp[1])**2 * w_pos + (tp[:, 2] - mp[2])**2 + (tp[:, 3] - mp[3])**2# + (tp[:, 4] - mp[4])**2
    idx_sorted = np.argsort(x_diff)
    return idx_sorted[:n_cells]
    


def select_well_tuned_cells_1D(tp, mp, n_cells, w_pos=1.):
#    w_pos = 10.
    x_diff = (tp[:, 0] - mp[0])**2 * w_pos + (tp[:, 2] - mp[2])**2 
    idx_sorted = np.argsort(x_diff)
    return idx_sorted[:n_cells]
    

def select_well_tuned_cells_trajectory(tp, mp, params, n_cells, n_pop):
    """
    tp -- array storing the tuning properties of the cells
    mp -- the motion parameters for the cells should be 'optimally' tuned
    params
    n_cells -- (int) number of cells to be selected
    n_pop -- n_cells is being split up in n_pop populations sorted by x-position
    """
    gids, dist = sort_gids_by_distance_to_stimulus(tp, mp, params)
    selected_gids = gids[:n_cells]
    x_pos = tp[selected_gids, 0]
    x_pos_srt = np.argsort(x_pos)
    gids_sorted = selected_gids[x_pos_srt]

    pops = []
    n_per_pop = int(round(n_cells / n_pop))
    for i in xrange(n_pop):
        sublist = distribute_list(gids_sorted, n_pop, i)
        pops.append(sublist)
    return gids_sorted, pops


# recording parameters for anticipatory mode
def all_anticipatory_gids(params):
    ex_cells = params['n_exc']
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    selected_gids = []
    cells = np.arange(ex_cells)
    for cell in cells:
#        print cell
#        i_cell = cells.tolist().index(cell)
        if (tp[cell,1]>0.3 and tp[cell,1]<0.7):
            selected_gids.append(cell)
    return selected_gids       
            
def pop_anticipatory_gids(params):
    ex_cells = params['n_exc']
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    selected_gids = all_anticipatory_gids(params)
    pop1, pop2, pop3, pop4, pop5 = [],[],[],[],[]
    for gid in selected_gids:
        if (tp[gid,0]>0 and tp[gid,0]<0.20):
            pop1.append(gid)
        elif (tp[gid,0]>0.2 and tp[gid,0]<0.4):
            pop2.append(gid)
    
        elif (tp[gid,0]>0.4 and tp[gid,0]<0.6):
            pop3.append(gid)
    
        elif (tp[gid,0]>0.6 and tp[gid,0]<0.8):
            pop4.append(gid)
    
        elif (tp[gid,0]>0.8 and tp[gid,0]<1):
            pop5.append(gid)
    pops = [pop1,pop2,pop3,pop4,pop5]
    return pops 


def CRF_anticipatory_gids(params, RF_xrange = np.arange(0.7,1,0.1)):
    ex_cells = params['n_exc']
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    selected_gids = all_anticipatory_gids(params)
    CRF_pop = []
    for gid in selected_gids:
        if (tp[gid,0] > RF_xrange[0] and tp[gid,0] < RF_xrange[-1]):
            CRF_pop.append(gid)    
    return CRF_pop



def get_figsize(fig_width_pt, portrait=True):
    """
    For getting a figure with an 'aesthetic' ratio
    """
    inches_per_pt = 1.0 / 72.0                # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    if portrait:
        fig_size =  (fig_height, fig_width)      # exact figsize
    else:
        fig_size =  (fig_width, fig_height)      # exact figsize
    return fig_size


def get_colorlist(n_colors=17):
    colorlist = ['k', 'b', 'r', 'g', 'm', 'c', 'y', \
            '#00FF99', \
            #light green
            '#FF6600', \
                    #orange
            '#CCFFFF', \
                    #light turquoise
            '#FF00FF', \
                    #light pink
            '#0099FF', \
                    #light blue
            '#CCFF00', \
                    #yellow-green
            '#D35F8D', \
                    #mauve
            '#808000', \
                    #brown-green
            '#bb99ff', \
                    # light violet
            '#7700ff', \
                    # dark violet
                    ]

    if n_colors > 17:
        r = lambda: rnd.randint(0,255)
        for i_ in xrange(n_colors - 17):
            colorlist.append('#%02X%02X%02X' % (r(),r(),r()))

    return colorlist


def convert_to_NEST_conform_dict(json_dict):
    testing_params = {}
    for k in json_dict.keys():
        if type(json_dict[k]) == type({}):
            d = json_dict[k]
            d_new = {}
            for key in d.keys():
                d_new[str(key)] = d[key]
            testing_params[k] = d_new
        elif type(json_dict[k]) == unicode:
            testing_params[str(k)] = str(json_dict[k])
        else:
            testing_params[str(k)] = json_dict[k]
    return testing_params


def extract_weight_from_connection_list(conn_list, pre_gid, post_gid, idx=None):
    """
    Extract the weight that connects the pre_gid to the post_gid
    """
    if idx == None:
        idx = 2
    pre_idx = set((conn_list[:, 0] == pre_gid).nonzero()[0])
    post_idx = set((conn_list[:, 1] == post_gid).nonzero()[0])
    valid_idx = list(pre_idx.intersection(post_idx))
    if len(valid_idx) == 0:
        return 0.
    return float(conn_list[valid_idx, idx])


def get_indices_for_gid(params, gid):
    """Returns the HC, MC, and within MC index for the gid
    """
    n_per_hc = params['n_mc_per_hc'] * params['n_exc_per_mc']
    mc_idx = (gid - 1) / params['n_exc_per_mc']
    hc_idx = (gid - 1) / n_per_hc
    mc_idx_in_hc = mc_idx - hc_idx * params['n_mc_per_hc']
    idx_in_mc = (gid - 1) - mc_idx * params['n_exc_per_mc']
    return hc_idx, mc_idx_in_hc, idx_in_mc


def get_mc_index_for_gid(params, gid):
    mc_idx = (gid - 1) / params['n_exc_per_mc']
    return mc_idx



