import numpy
import numpy.random as rnd
import os
from scipy.spatial import distance

def set_tuning_prop(n_columns):
    """
    Define the receptive field of a cell by a gaussian in a 4-dim space (x, y, u, v) (2dim for position, 2 for speed).
    The position of each cell represents its excitability to a given a 4-dim stimulus.
    n_columns is the number of cells or minicolumns for which the tuning properties are to be set.

    return value:
        tp = numpy.array((n_columns, 4, 2))
        tp[:, 0, 0] : mu_x
        tp[:, 0, 1] : sigma_x
        tp[:, 1, 0] : mu_y
        tp[:, 1, 1] : sigma_y
        tp[:, 2, 0] : mu_u (speed in x-direction)
        tp[:, 2, 1] : sigma_u
        tp[:, 3, 0] : mu_v (speed in y-direction)
        tp[:, 3, 1] : sigma_v

        All values are in range [0..1]
    """

    tuning_prop = numpy.zeros((n_columns, 4, 2))

    # place the columns on a grid with the following dimensions
    x_max = int(round(numpy.sqrt(n_columns)))
    y_max = int(round(numpy.sqrt(n_columns)))
    if (n_columns > x_max * y_max):
        x_max += 1

    for i in xrange(n_columns):
        tuning_prop[i, 0, 0] = (i % x_max) / float(x_max)   # spatial rf centers are on a grid
        tuning_prop[i, 1, 0] = (i / x_max) / float(y_max)
        tuning_prop[i, 2, 0] = rnd.rand()  # velocity rf are randomly set
        tuning_prop[i, 3, 0] = rnd.rand()
        tuning_prop[i, :, 1] = rnd.rand(4) * 0.001 # sigma values are in the range [0, 0.1]
    return tuning_prop

def create_spike_trains_for_motion(tuning_prop, motion_params, params):
    """
    This function writes spike trains to a dedicated path specified in the params dict
    Spike trains are generated for each unit / minicolumn based on the function's arguments the following way:
    The strength of stimulation for a column is computed based on the motion parameters and the tuning properties of the column.
    This strength determines the envelope the non-homogeneous Poisson process to create the spike train.

    Arguments:
        tuning_prop = numpy.array((n_cells, 4, 2))
            tuning_prop[:, 0, 0] : mu_x
            tuning_prop[:, 0, 1] : sigma_x
            tuning_prop[:, 1, 0] : mu_y
            tuning_prop[:, 1, 1] : sigma_y
            tuning_prop[:, 2, 0] : mu_u (speed in x-direction)
            tuning_prop[:, 2, 1] : sigma_u
            tuning_prop[:, 3, 0] : mu_v (speed in y-direction)
            tuning_prop[:, 4, 1] : sigma_v

        motion: (x0, y0, u0, v0)
            x0 x-position at start
            y0 y-position at start
            u0 velocity in x-direction
            v0 velocity in y-direction

        params:  dictionary storing all simulation parameters

    """

    # each cell will get its own spike train stored in the following file + cell gid
    tgt_fn_base = os.path.abspath(params['input_st_fn_base'])
    n_units = tuning_prop.shape[0]
    n_cells = params['n_exc_per_mc'] # each unit / column can contain several cells
    x0, y0, u0, v0 = motion_params

    dt = 0.01 # [ms] time step for the non-homogenous Poisson process 
    time = numpy.arange(0, params['t_sim'], dt)

    for column in xrange(n_units):
#        for cell in xrange(params['n_exc_per_mc']):
#        gid = column * params['n_exc_per_mc'] + cell
        mu_x = tuning_prop[column, 0, 0]
        mu_y = tuning_prop[column, 1, 0]
        mu_u = tuning_prop[column, 2, 0]
        mu_v = tuning_prop[column, 3, 0]

        # Shape the spike train: where is the max and how large is the stimulus pulse?
        # decide on the length of the stimulus: faster motion -> shorter stimulus package (arbitrary choice) TODO: something better?
        width_of_stim = params['stim_dur_sigma'] * (1 - numpy.sqrt(u0**2 + v0**2))

        # TODO: receptive field, e.g.  width_of_stim, max_of_stim = rf(tuning_prop, motion_params)
        dist_from_rf = distance.euclidean(tuning_prop[column,:, 0], motion_params)
        time_of_max_stim = params['t_sim'] * get_time_of_max_stim(tuning_prop[column, :, :], motion_params)
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
        numpy.save(output_fn, numpy.array(st)) # to be changed to binary numpy.save
        output_fn = params['input_folder'] + 'rate_' + str(column)
        numpy.save(output_fn, rate_of_t) # to be changed to binary numpy.save

def get_time_of_max_stim(tuning_prop, motion_params):
    """
    This function assumes motion with constant velocity, starting at x0 y0.
    Based on the spatial receptive field (RF: mu_x, mu_y) of the cell (column) the time when the stimulus is closest
    to the RF.
    t_min = (mu_x * u0 + mu_y * v0 - v0 * y0 + u0 * x0) / (v0**2 + u0**2)
    """
    sigma_x = tuning_prop[0, 1]
    sigma_y = tuning_prop[1, 1]
    mu_x = tuning_prop[0, 0] #+ numpy.sign(rnd.uniform(-1, 1)) * sigma_x
    mu_y = tuning_prop[1, 0] #+ numpy.sign(rnd.uniform(-1, 1)) * sigma_y
    x0, y0, u0, v0 = motion_params
    t_min = (mu_x * u0 + mu_y * v0 - v0 * y0 + u0 * x0) / (u0**2 + v0**2)
    return t_min

def get_input_for_constant_velocity(tuning_prop, motion_params):
    """
    This function returns a list of parameters describing the time course of the stimulation elicited
    by a moving dot starting at (x0, y0) moving with constant velocity (u0, v0=0).
    The shape (radius) of the dot is not considered.
    
    As we assume receptive fields shaped as Gauss functions and constant velocities,
    the time course of the stimulus has Gaussian shape.
    -> Return value: for each cell (dim 0) a tuple for mu and sigma of the Gauss function.

    Arguments:
        tuning_prop = numpy.array((n_cells, 4, 2))
            tuning_prop[:, 0, 0] : mu_x
            tuning_prop[:, 0, 1] : sigma_x
            tuning_prop[:, 1, 0] : mu_y
            tuning_prop[:, 1, 1] : sigma_y
            tuning_prop[:, 2, 0] : mu_u (speed in x-direction)
            tuning_prop[:, 2, 1] : sigma_u
            tuning_prop[:, 3, 0] : mu_v (speed in y-direction)
            tuning_prop[:, 4, 1] : sigma_v

        motion: (x0, y0, u0, v0)
            x0 x-position at start
            y0 y-position at start
            u0 velocity in x-direction
            v0 velocity in y-direction
    """
    pass 





def get_input(tuning_prop, time_steps, motion):
    """
    This function computes the input to each cell based on the given tuning properties.

    Arguments:
        tuning_prop: 3-dim numpy.array; 
            dim 0 is number of cells
        time_steps: number of simulation time steps
        motion: type of motion, maybe filename to movie, ... ???
    """
    n_cells = tuning_prop[:, 0].size
    L = numpy.zeros((n_cells, time_steps)) # maybe use sparse matrices or adjacency lists instead

    # compute the motion energy input to all cells
    """
        ...

         ... Laurent makes some very cool stuff here ...

        ...

    """

    return L



def convert_motion_energy_to_spike_trains(rate_of_t, dt=1, tgt_fn_base='input_st_'):
    """
    Based on the time dependent motion energy stored in the input vectors, 
    this function writes input spike trains to files starting with 'tgt_fn_base'
    Arguments:
        rate_of_t : callback function to derive the temporal development of the rate
#        input_vecs : time dependent input
#            input_vecs[i, :] : input vector into cell i
#            input_vecs[:, t] : input at time t
        dt: time step in ms, should be very small (Poisson process)
        tgt_fn_base: output will be stored in: output_fn = tgt_fn_base + str(cell) + '.dat'
    """
    n_cells = input_vecs[:, 0].size
    n_steps = input_vecs[0, :].size

    for cell in xrange(n_cells):
        st = []
        for i in xrange(n_steps):
            r = rnd.rand()
            if (r <= (input_vecs[cell, i] * dt)):
                st.append(i * dt)

        output_fn = tgt_fn_base + str(cell) + '.dat'
        numpy.save(output_fn, numpy.array(st)) # to be changed to binary numpy.save


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
#    spatial_distance = numpy.sqrt((mu_x - x0 - u0 * time)**2 + (mu_y - y0 - v0 * time)**2)
#    velocity_component = numpy.sqrt((mu_u - u0)**2 + (mu_v - v0)**2)
#    return velocity_component * spatial_distance


def gauss(x, mu, sigma):
    return numpy.exp( - (x - mu)**2 / (2 * sigma ** 2))


def euclidean(x, y):
    return distance.euclidean(x, y)
