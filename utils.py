"""
    This file contains a bunch of helper functions (in alphabetic order).
"""

import numpy as np
import numpy.random as rnd
import os
from scipy.spatial import distance
from NeuroTools import signals as nts
import pylab


def convert_connlist_to_matrix(fn, n_cells):
    """
    Convert the connlist which is in format (src, tgt, weight, delay) to a weight matrix.
    """
    conn_list = np.loadtxt(fn)
    m = np.zeros((n_cells, n_cells))
    delays = np.zeros((n_cells, n_cells))
    for i in xrange(conn_list[:,0].size):
        src = conn_list[i, 0]
        tgt = conn_list[i, 1]
        m[src, tgt] = conn_list[i, 2]
        delays[src, tgt] = conn_list[i, 3]
    return m, delays


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




def create_spike_trains_for_motion(tuning_prop, params, contrast=.9, my_units=None):
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
    motion_params = params['motion_params']
    # each cell will get its own spike train stored in the following file + cell gid
    tgt_fn_base = os.path.abspath(params['input_st_fn_base'])
    n_units = tuning_prop.shape[0]
    n_cells = params['n_exc'] # each unit / column can contain several cells
    dt = 0.1 # [ms] time step for the non-homogenous Poisson process 
    time = np.arange(0, params['t_sim'], dt)

    if (my_units == None):
        my_units = xrange(n_units)
    else:
        my_units = xrange(my_units[0], my_units[1])

    L_input = np.empty((n_cells, time.shape[0]))
    mv = np.zeros((time.shape[0], 5))
    for i_time, time_ in enumerate(time):
        print "t:", time_
#        print i_time, time.shape[0]
#        L_input[:, i_time] = get_input(tuning_prop, params, time_/params['t_sim'], contrast=contrast) * params['f_max_stim']
        L_input[:, i_time], (x, y, x_, y_) = get_input(tuning_prop, params, time_/params['t_sim'], contrast=contrast)
        L_input[:, i_time] *= params['f_max_stim']
        mv[i_time, 0] = time_
        mv[i_time, 1] = x
        mv[i_time, 2] = y
        mv[i_time, 3] = x_
        mv[i_time, 4] = y_
    
    print "Debug, saving motion to:", params['motion_fn']
    np.savetxt(params['motion_fn'], mv)

    for column in my_units:
        print "cell:", column
        rate_of_t = np.array(L_input[column, :]) #  max_of_stim * gauss(time, time_of_max_stim, width_of_stim)

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


def get_input(tuning_prop, params, t, contrast=.9, motion='dot'):
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

    motion_params = params['motion_params']
    n = params['N_RF_X']
    m = params['N_RF_Y']
    L = np.zeros(n_cells)
    if motion=='dot':
        # define the parameters of the motion
        x0, y0, u0, v0 = motion_params

        blur_X, blur_V = params['blur_X'], params['blur_V'] #0.5, 0.5
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
        x, y = x0 + u0*t, y0 + v0*t # current position of the blob at time t assuming a perfect translation
        x_, y_ = x, y

        # modify position of dot to match torus constraints
#        x_lim, y_lim = 1, 1 # half the circumcircle of the torus --> parametrize ?
#        ax = (np.int(x) / x_lim + 1) % 2
#        not_ax = (np.int(x) / x_lim) % 2
#        b = x % x_lim
#        c = x_lim - ax * b
#        x = ax * c + not_ax * b

#        ay = (np.int(y) / y_lim + 1) % 2
#        not_ay = (np.int(y) / y_lim) % 2
#        b = y % y_lim
#        c = y_lim - ay * b
#        y = ay * c + not_ay * b

        x, y = np.mod(x, 1.), np.mod(y, 1.) # we are on a torus

    for cell in xrange(n_cells): # todo: vectorize
        L[cell] = np.exp( -.5 * (tuning_prop[cell, 0] - x)**2/blur_X**2
                          -.5 * (tuning_prop[cell, 1] - y)**2/blur_X**2
                          -.5 * (tuning_prop[cell, 2] - u0)**2/blur_V**2
                          -.5 * (tuning_prop[cell, 3] - v0)**2/blur_V**2
                          )
    
#    L = (1. - contrast) + contrast * L
        
    return L, (x, y, x_, y_)




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
    mu_x = tuning_prop[0] #+ np.sign(rnd.uniform(-1, 1)) * sigma_x
    mu_y = tuning_prop[1] #+ np.sign(rnd.uniform(-1, 1)) * sigma_y
    x0, y0, u0, v0 = motion_params
    t_min = (mu_x * u0 + mu_y * v0 - v0 * y0 + u0 * x0) / (u0**2 + v0**2)
    return t_min


def set_tuning_prop(params, mode='hexgrid', v_max=1.0):
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
        random_rotation = 2*np.pi*rnd.rand(params['N_RF_X']*params['N_RF_Y'])
        # todo do the same for v_rho?
        for i_v_rho, rho in enumerate(v_rho):
            for i_theta, theta in enumerate(v_theta):
                for i_RF in xrange(params['N_RF_X']*params['N_RF_Y']):
                    tuning_prop[index, 0] = RF[0, i_RF] + params['sigma_RF_pos'] * rnd.randn()
                    tuning_prop[index, 1] = RF[1, i_RF] + params['sigma_RF_pos'] * rnd.randn()
                    tuning_prop[index, 2] = np.cos(theta + random_rotation[i_RF] + parity[i_v_rho] * np.pi / params['N_theta']) \
                            * rho * (1. + params['sigma_RF_speed'] * rnd.randn())
                    tuning_prop[index, 3] = np.sin(theta + random_rotation[i_RF] + parity[i_v_rho] * np.pi / params['N_theta']) \
                            * rho * (1. + params['sigma_RF_speed'] * rnd.randn())
                    index += 1

    return tuning_prop

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
        fig = pylab.figure()#figsize=(fig_width, fig_width * np.float(N_Y) / N_X))
        a = fig.add_axes([0., 0., 1., 1.])
        if hue:
# TODO : overlay image and use RGB(A) information
#            print v_hist, v_hist.min(), v_hist.max(), np.flipud( np.fliplr(im_).T
#            a.imshow(-np.log(np.rot90(v_hist)+eps_hist), interpolation='nearest')
            a.imshow(np.fliplr(np.rot90(v_hist/v_hist.max(),3)), interpolation='nearest', origin='lower', extent=(-width/2, width/2, -ywidth/2., ywidth/2.))#, vmin=0., vmax=v_hist.max())
#            pylab.axis('image')
        else:
            a.pcolor(x_edges, y_edges, v_hist, cmap=pylab.bone(), vmin=0., vmax=v_hist.max(), edgecolor='k')
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
    # seperate spike trains for all the cells
    for i in xrange(d[:, 0].size):
        spiketrains[int(d[i, 1])].append(d[i, 0])
    for gid in xrange(n_cells):
        nspikes[gid] = len(spiketrains[gid])
    if get_spiketrains:
        return nspikes, spiketrains
    else:
        return nspikes


def get_spiketrains(spiketimes_fn_merged, n_cells=0):
    """
    Returns an array with the number of spikes fired by each cell.
    nspikes[gid]
    if n_cells is not given, the length of the array will be the highest gid (not advised!)
    """
    d = np.loadtxt(spiketimes_fn_merged)
    if (n_cells == 0):
        n_cells = 1 + np.max(d[:, 1])# highest gid
    nspikes = np.zeros(n_cells)
    spiketrains = [[] for i in xrange(n_cells)]
    # seperate spike trains for all the cells
    for i in xrange(d[:, 0].size):
        spiketrains[int(d[i, 1])].append(d[i, 0])
    return spiketrains

def get_grid_pos(x0, y0, xedges, yedges):

    x_index, y_index = 0, 0
    for (ix, x) in enumerate(xedges[1:]):
        if x0 < x:
            x_index = ix
            break
            
    for (iy, y) in enumerate(yedges[1:]):
        if y0 < y:
            y_index = iy
            break
    return (x_index, y_index)

def get_grid_pos_1d(x0, xedges):

    x_index, y_index = 0, 0
    for (ix, x) in enumerate(xedges[1:]):
        if x0 < x:
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


def sort_gids_by_distance_to_stimulus(tp, mp, sorting_index=0):
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

        sorting_index: integer [0, 3] or string ('x', 'y', 'u', 'v')
    """
    if sorting_index == 'x':
        sorting_index = 0
    elif sorting_index == 'y':
        sorting_index = 1
    elif sorting_index == 'u':
        sorting_index = 2
    elif sorting_index == 'v':
        sorting_index = 3

    n_cells = tp[:, 0].size
#    v_dist = np.zeros(n_cells) # distance in 'velocity space' between stimulus (u, v) and cells' tuning_properties
    x_dist = np.zeros(n_cells) # stores minimal distance in space between stimulus and cells

    n_steps = 100 # the of 
    x_pos_stim = np.array([mp[0] + mp[2] * i * 1./n_steps for i in xrange(n_steps)])
    y_pos_stim = np.array([mp[1] + mp[3] * i * 1./n_steps for i in xrange(n_steps)])
    for i in xrange(n_cells):
#        v_dist[i] = np.sqrt((tp[i, 2] - mp[2])**2 + (tp[i,3] - mp[3])**2)
        x_dist[i] = np.sqrt(np.min((tp[i, 0] - x_pos_stim)** 2 + (tp[i, 1] - y_pos_stim)**2) + (tp[i, 2] - mp[2])**2 + (tp[i,3] - mp[3])**2)

#    return x_dist
    cells_closest_to_stim_pos = x_dist.argsort()
#    cells_closest_to_stim_velocity = v_dist.argsort()
    return cells_closest_to_stim_pos, x_dist[cells_closest_to_stim_pos]#, cells_closest_to_stim_velocity

def torus_distance(x0, x1):
    dx = x0 - x1
    x_lim, y_lim = 1, 1 # half the circumcircle of the torus --> parametrize ?
    ax = (np.int(dx) / x_lim + 1) % 2
    not_ax = (np.int(dx) / x_lim) % 2
    b = dx % x_lim
    c = x_lim - ax * b
    dx = ax * c + not_ax * b

    return dx

