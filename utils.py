"""
    This file contains a bunch of helper functions (in alphabetic order).
"""

import numpy as np
import numpy.random as rnd
import os
from scipy.spatial import distance
from NeuroTools import signals as nts


def convert_connlist_to_matrix(fn, n_cells):
    """
    Convert the connlist which is in format (src, tgt, weight, delay) to a weight matrix.

    """
    conn_list = np.loadtxt(fn)
    m = np.zeros((n_cells, n_cells))
    for i in xrange(conn_list[:,0].size):
        src = conn_list[i, 0]
        tgt = conn_list[i, 1]
        m[src, tgt] = conn_list[i, 2]
    return m

def convert_motion_energy_to_spike_trains(tuning_prop, n_steps=100, tgt_fn_base='input_st_'):
    """
    Based on the time dependent motion energy stored in the input vectors, 
    this function writes input spike trains to files starting with 'tgt_fn_base'
    Arguments:
        tuning_prop: 2 dimensional array with shape (n_cells, 4)
            tp[:, 0] : x-position
            tp[:, 1] : y-position
            tp[:, 2] : u-position (speed in x-direction)
            tp[:, 3] : v-position (speed in y-direction)
        dt: time step in ms, should be very small (Poisson process)
        tgt_fn_base: output will be stored in: output_fn = tgt_fn_base + str(cell) + '.dat'
    """
    n_cells = tuning_prop.size[0]
    dt = 1./n_steps

    st = []
    for cell in xrange(n_cells):
        st.append( [] )


    for i in xrange(n_steps):
        input_vec = get_input(tuning_prop, np.float(i)/n_steps, motion='dot')
        for cell in xrange(n_cells):
            r = rnd.rand()
            if (r <= (input_vec[cell] * dt)):
                st[cell].append(i * dt)

    # TODO : use NeuroTools' SpikeList class and test!
    spklist = nts.SpikeList(st, range(n_cells), t_start=0, t_stop=n_steps)

    for cell in xrange(n_cells):
        output_fn = tgt_fn_base + str(cell) + '.dat'
        # TODO test and check if compatible with the reading spike trains
#        spklist[0].save(StandPickleFile(output_fn)
        np.savetxt(output_fn, np.array(st))

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




def create_spike_trains_for_motion(tuning_prop, motion_params, params, my_units=None):
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

        motion: (x0, y0, u0, v0)
            x0 x-position at start
            y0 y-position at start
            u0 velocity in x-direction
            v0 velocity in y-direction

        params:  dictionary storing all simulation parameters
        my_units: tuple of integers (start, begin), in case of parallel execution each processor creates spike trains for its own units or columns

    """

    # each cell will get its own spike train stored in the following file + cell gid
    tgt_fn_base = os.path.abspath(params['input_st_fn_base'])
    n_units = tuning_prop.shape[0]
    x0, y0, u0, v0 = motion_params

    dt = 0.01 # [ms] time step for the non-homogenous Poisson process 
    time = np.arange(0, params['t_sim'], dt)

    if (my_units == None):
        my_units = xrange(n_units)
    else:
        my_units = xrange(my_units[0], my_units[1])

    for column in my_units:
#        for cell in xrange(params['n_exc_per_mc']):
#        gid = column * params['n_exc_per_mc'] + cell
        mu_x = tuning_prop[column, 0]
        mu_y = tuning_prop[column, 1]
        mu_u = tuning_prop[column, 2]
        mu_v = tuning_prop[column, 3]

        # Shape the spike train: where is the max and how large is the stimulus pulse?
        # decide on the length of the stimulus: faster motion -> shorter stimulus package (arbitrary choice) TODO: something better?
        width_of_stim = params['stim_dur_sigma'] * (1 - np.sqrt(u0**2 + v0**2))

        # TODO: receptive field, e.g.  width_of_stim, max_of_stim = rf(tuning_prop, motion_params)
        dist_from_rf = distance.euclidean((mu_u, mu_v), (u0, v0))
        time_of_max_stim = params['t_sim'] * get_time_of_max_stim(tuning_prop[column, :], motion_params)
        print "Creating input for column %d. t_stim = (%.1f, %.1f)" % (column, time_of_max_stim, width_of_stim)

        width_of_stim *= dist_from_rf
        max_of_stim = dist_from_rf * params['f_max_stim']

        rate_of_t = max_of_stim * gauss(time, time_of_max_stim, width_of_stim)

        n_steps = rate_of_t.size
        st = []
        for i in xrange(n_steps):
            r = rnd.rand()
            if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                st.append(i * dt)
        output_fn = tgt_fn_base + str(column)
        np.save(output_fn, np.array(st)) # to be changed to binary np.save
        output_fn = params['input_folder'] + 'rate_' + str(column)
        np.save(output_fn, rate_of_t) # to be changed to binary np.save



def distance_to_rf_center(cell_tuning, motion_params, t_stop, t_start=0., dt=0.01):
    """
    This function returns an array containing the distance between 
    a dot and the center of a cell's receptive field.

    parameters:
        mu_x = cell_tuning[0]
        mu_y = cell_tuning[1]
        mu_u = cell_tuning[2]
        mu_v = cell_tuning[3]
        motion_params:
            x0, y0: starting point of the moving dot
            u0, v0: velocity in x (y) direction of the moving dot
    returns:
        value between 0 and 1
    This class contains the simulation parameters in a dictionary called params.
    
    TODO: use the NeuroTools.parameter class that does exactly that?
    """
    return distance.euclidean(cell_tuning, motion_params)

#    mu_x = cell_tuning[0]
#    mu_y = cell_tuning[1]
#    mu_u = cell_tuning[2]
#    mu_v = cell_tuning[3]
#    x0 = motion_params[0]
#    y0 = motion_params[1]
#    u0 = motion_params[2]
#    v0 = motion_params[3]
#    spatial_distance = np.sqrt((mu_x - x0 - u0 * time)**2 + (mu_y - y0 - v0 * time)**2)
#    velocity_component = np.sqrt((mu_u - u0)**2 + (mu_v - v0)**2)
#    return velocity_component * spatial_distance

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




def get_input(tuning_prop, t, motion='dot'):
    """
    This function computes the input to each cell based on the given tuning properties.

    Arguments:
        tuning_prop: 2-dim np.array; 
            dim 0 is number of cells
            tuning_prop[:, 0] : x-position
            tuning_prop[:, 1] : y-position
            tuning_prop[:, 2] : u-position (speed in x-direction)
            tuning_prop[:, 3] : v-position (speed in y-direction)
        t: time in the period is between 0 (included) and 1. (excluded)
        motion: type of motion (TODO: filename to movie, ... ???)
    """
    n_cells = tuning_prop[:, 0].size
#    L = np.zeros((n_cells, time_steps)) # maybe use sparse matrices or adjacency lists instead

    L = np.zeros(n_cells)
    if motion=='dot':
        # define the parameters of the motion
        X_0, Y_0 = .25, .5 #
        V_X, V_Y = .5, 0.0
        blur_X, blur_V = 1.0, 1.0
        # compute the motion energy input to all cells
        """

            Knowing the velocity one can estimate the analytical response to 
             - motion energy detectors
             - to a gaussian blob
            as a function of the distance between 
             - the center of the receptive fields,
             - the currentpostio of the blob.
             
            # TODO : prove this analytically to disentangle the different blurs (size of RF / size of dot)
            
            L range between 0 and 1
    
        """
        X, Y = X_0 + V_X*t, Y_0 + V_Y*t # current position of the blob at timet assuming a perfect translation

    pedestal = .2 # TODO: there arebetter ways to describe the adaptativity of MT to global inut
    for cell in xrange(n_cells): # todo: vectorize
        L[cell] = np.exp( -.5 * (tuning_prop[cell, 0] - X)**2/blur_X**2
                          -.5 * (tuning_prop[cell, 1] - Y)**2/blur_X**2
                          -.5 * (tuning_prop[cell, 2] - V_X)**2/blur_V**2
                          -.5 * (tuning_prop[cell, 3] - V_Y)**2/blur_V**2
                          )
        L[cell] *= (1-pedestal)
        L[cell] += pedestal
    return L



def get_time_of_max_stim(tuning_prop, motion_params):
    """
    This function assumes motion with constant velocity, starting at x0 y0.
    Based on the spatial receptive field (RF: mu_x, mu_y) of the cell (column) the time when the stimulus is closest
    to the RF.
    t_min = (mu_x * u0 + mu_y * v0 - v0 * y0 + u0 * x0) / (v0**2 + u0**2)
    """
    mu_x = tuning_prop[0] #+ np.sign(rnd.uniform(-1, 1)) * sigma_x
    mu_y = tuning_prop[1] #+ np.sign(rnd.uniform(-1, 1)) * sigma_y
    x0, y0, u0, v0 = motion_params
    t_min = (mu_x * u0 + mu_y * v0 - v0 * y0 + u0 * x0) / (u0**2 + v0**2)
    return t_min


def set_tuning_prop(params, mode='hexgrid', v_max=2.0):
    """
    Place n_cells in a 4-dimensional space by some mode (random, hexgrid, ...).
    The position of each cell represents its excitability to a given a 4-dim stimulus.
    The radius of their receptive field is assumed to be constant (TODO: one coud think that it would depend on the density of neurons?)

    return value:
        tp = set_tuning_prop(n_cells)
        tp[:, 0] : x-position
        tp[:, 1] : y-position
        tp[:, 2] : u-position (speed in x-direction)
        tp[:, 3] : v-position (speed in y-direction)

    All x-y values are in range [0..1]. Positios are defined on a torus and a dot moving to a border reappears on the other side (as in Pac-Man)
    By convention, velocity is such that V=(1,0) corresponds to one horizontal spatial period in one temporal period.
    This implies that in one frame, a translation is of  ``1. / N_frame`` in cortical space.
    """

    tuning_prop = np.zeros((params['n_exc'], 4))
    if mode=='random':
        # place the columns on a grid with the following dimensions
        x_max = int(round(np.sqrt(params['n_cells'])))
        y_max = int(round(np.sqrt(params['n_cells'])))
        if (params['n_cells'] > x_max * y_max):
            x_max += 1

        for i in xrange(params['n_cells']):
            tuning_prop[i, 0] = (i % x_max) / float(x_max)   # spatial rf centers are on a grid
            tuning_prop[i, 1] = (i / x_max) / float(y_max)
            tuning_prop[i, 2] = v_max * rnd.randn()
            tuning_prop[i, 3] = v_max * rnd.randn()

    elif mode=='hexgrid':

        if params['log_scale']==1:
            v_rho = np.linspace(v_max/params['N_V'], v_max, num=params['N_V'], endpoint=True)
        else:
            v_rho = np.logspace(np.log(v_max/params['N_V'])/np.log(params['log_scale']),
                            np.log(v_max)/np.log(params['log_scale']), num=params['N_V'],
                            endpoint=True, base=params['log_scale'])
        v_theta = np.linspace(0, 2*np.pi, params['N_theta'], endpoint=False)
        parity = np.arange(params['N_V']) % 2

        RF = np.zeros((2, params['N_RF_X']*params['N_RF_Y']))
        X, Y = np.mgrid[0:1:1j*(params['N_RF_X']+1), 0:1:1j*(params['N_RF_Y']+1)]
    
        # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
        X, Y = X[1:, 1:], Y[1:, 1:]
        # Add to every even Y a half RF width to generate hex grid
        Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./N_RF
        RF[0, :] = X.ravel()
        RF[1, :] = Y.ravel()
    
        # wrapping up:


        index = 0
        for i_v_rho, rho in enumerate(v_rho):
            for i_theta, theta in enumerate(v_theta):
                for i_RF in xrange(params['N_RF_X']*params['N_RF_Y']):
                    tuning_prop[index, 0] = RF[0, i_RF]
                    tuning_prop[index, 1] = RF[1, i_RF]
                    tuning_prop[index, 2] = np.cos(theta + parity[i_v_rho] * np.pi / params['N_theta']) * rho
                    tuning_prop[index, 3] = np.sin(theta + parity[i_v_rho] * np.pi / params['N_theta']) * rho
                    index += 1

    return tuning_prop


def threshold_weights(connection_matrix, w_thresh):
    """
    Elements in connection_matrix below w_thresh will be set to zero.
    """
    for i in xrange(connection_matrix[:, 0].size):
        for j in xrange(connection_matrix[0, :].size):
            if connection_matrix[i, j] < w_thresh:
                connection_matrix[i, j] = 0.0
    return connection_matrix
