import numpy
import numpy.random as rnd


def set_tuning_prop(n_cells):
    """
    Place n_cells in a 4-dimensional space by random.
    The position of each cell represents its excitability to a given a 4-dim stimulus.

    return value:
        tp = numpy.array((n_cells, 4))
        tp[:, 0] : x-position
        tp[:, 1] : y-position
        tp[:, 2] : u-position (speed in x-direction)
        tp[:, 3] : v-position (speed in y-direction)

        All values are in range [0..1]
    """
    tuning_prop = numpy.zeros((n_cells, 4))
    for cell in xrange(n_cells):
        tuning_prop[cell, 0] = rnd.rand()
        tuning_prop[cell, 1] = rnd.rand()
        tuning_prop[cell, 2] = rnd.rand()
        tuning_prop[cell, 3] = rnd.rand()
    return tuning_prop


def get_input(tuning_prop, time_steps, motion):
    """
    This function computes the input to each cell based on the given tuning properties.

    Arguments:
        tuning_prop: 2-dim numpy.array; 
            dim 0 is number of cells
            tuning_prop[:, 0] : x-position
            tuning_prop[:, 1] : y-position
            tuning_prop[:, 2] : u-position (speed in x-direction)
            tuning_prop[:, 3] : v-position (speed in y-direction)
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



def convert_motion_energy_to_spike_trains(input_vecs, dt=1, tgt_fn_base='input_st_'):
    """
    Based on the time dependent motion energy stored in the input vectors, 
    this function writes input spike trains to files starting with 'tgt_fn_base'
    Arguments:
        input_vecs : time dependent input
            input_vecs[i, :] : input vector into cell i
            input_vecs[:, t] : input at time t
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
        numpy.savetxt(output_fn, numpy.array(st)) # to be changed to binary numpy.save



class network_parameters(object):
    """
    This class contains the simulation parameters in a dictionary called params.
    """

    def __init__(self):
        self.params = {}
        self.params['n_exc' ] = 12      # number of excitatory cells
        self.params['n_inh' ] = 3       # number of inhibitory cells
        self.params['seed'] = 12345
        self.params['t_sim'] = 100.     #
        rnd.seed(self.params['seed'])


    def load_params(self):
        """
        return the simulation parameters in a dictionary
        """
        return self.params

