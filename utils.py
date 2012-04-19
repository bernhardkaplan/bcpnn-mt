import numpy as np
import numpy.random as rnd


def set_tuning_prop(n_cells, mode='hexgrid', v_max=2.0):
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
    tuning_prop = np.zeros((n_cells, 4))
    if mode=='random':
        for cell in xrange(n_cells):
            tuning_prop[cell, 0] = rnd.rand()
            tuning_prop[cell, 1] = rnd.rand()
            tuning_prop[cell, 2] = v_max * rnd.randn()
            tuning_prop[cell, 3] = v_max * rnd.randn()
    elif mode=='hexgrid':
        N_V, N_theta = 8, 16 # resolution in velocity norm and direction
        log_scale = 1. # base of the logarithmic tiling of particle_grid; linear if equal to one

        v_rho = np.logspace(np.log(v_max/N_V)/np.log(log_scale),
                            np.log(v_max)/np.log(log_scale), num=N_V,
                            endpoint=True, base=log_scale)
        v_theta = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
        parity = np.arange(N_V) % 2

        N_RF = np.int(n_cells/N_V)
        # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of N_RF**2 dots?"
        N_RF_X = np.int(np.sqrt(N_RF*np.sqrt(3)))
        N_RF_Y = np.int(np.sqrt(N_RF/np.sqrt(3)))
        RF = np.zeros((2, N_RF_X*N_RF_Y))
        X, Y = np.mgrid[0:1:1j*(N_RF_X+1), 0:1:1j*(N_RF_Y+1)]
    
        # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
        X, Y = X[1:, 1:], Y[1:, 1:]
        # Add to every even Y a half RF width to generate hex grid
        Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./N_RF
        RF[0, :] = X.ravel()
        RF[1, :] = Y.ravel()
    
        # wrapping up:
        assert N_V * N_theta * N_RF_X*N_RF_Y==n_cells

        index = 0
        for i_v_rho, rho in enumerate(v_rho):
            for i_theta, theta in enumerate(v_theta):
                for i_RF in range(N_RF_X*N_RF_Y):
                    tuning_prop[index, 0] = RF[0, i_RF]
                    tuning_prop[index, 1] = RF[1, i_RF]
                    tuning_prop[index, 2] = np.cos(theta + parity[i_v_rho] * np.pi / N_theta) * rho
                    tuning_prop[index, 3] = np.sin(theta + parity[i_v_rho] * np.pi / N_theta) * rho
                    index += 1
         
    return tuning_prop



def get_input(tuning_prop, t, motion='dot'):
    """
    This function computes the input to each cell based on the given tuning properties.

    Arguments:
        tuning_prop: 2-dim numpy.array; 
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



def convert_motion_energy_to_spike_trains(tuning_prop, n_steps=100, tgt_fn_base='input_st_'):
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
    n_cells = tuning_prop.size[0]
#    n_steps = input_vecs[0, :].size
    dt = 1./n_steps

    st = []
    for cell in xrange(n_cells):
        st.append( [] )

    # TODO : use NeuroTools' SpikeList class

    for i in xrange(n_steps):
        input_vec = get_input(tuning_prop, np.float(i)/n_steps, motion='dot')
        for cell in xrange(n_cells):
            r = rnd.rand()
            if (r <= (input_vec[cell] * dt)):
                st[cell].append(i * dt)

    for cell in xrange(n_cells):
        output_fn = tgt_fn_base + str(cell) + '.dat'
        np.savetxt(output_fn, np.array(st)) # to be changed to binary numpy.save


class network_parameters(object):
    """
    This class contains the simulation parameters in a dictionary called params.
    
    TODO: use the NeuroTools.parameter class that does exactly that?
    
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

