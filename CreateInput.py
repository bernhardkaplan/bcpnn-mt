import numpy as np
import random
import os
import sys
import utils
import set_tuning_properties


class CreateInput(object):
    def __init__(self, params, visual_stim_seed=None):
        self.motion_params_created = False
        self.params = params
        self.tuning_prop_loaded = False
        if visual_stim_seed == None:
            visual_stim_seed = self.params['visual_stim_seed']
        print 'Setting seed to:', visual_stim_seed
        self.RNG = np.random.RandomState(visual_stim_seed)
        self.load_tuning_prop()

    def load_tuning_prop(self, tuning_prop_fn=None):

        if tuning_prop_fn == None:
            self.tuning_prop, self.rf_sizes = set_tuning_properties.set_tuning_properties_and_rfs_const_fovea(self.params)
            print 'Saving tuning prop to:', self.params['tuning_prop_exc_fn']
            np.savetxt(self.params['tuning_prop_exc_fn'], self.tuning_prop)
            print 'Saving receptive fields to:', self.params['receptive_fields_exc_fn']
            np.savetxt(self.params['receptive_fields_exc_fn'], self.rf_sizes)
        else:
            print 'Loading tuning prop from:', self.params['tuning_prop_exc_fn']
            print 'Loading receptive fields from:', self.params['receptive_fields_exc_fn']
            self.tuning_prop = np.loadtxt(self.params['tuning_prop_exc_fn'])
            self.rf_sizes = np.loadtxt(self.params['rf_sizes_fn'])

#    def compute_input(self, motion_params, gids, tuning_prop):
#        """
#        motion_params -- (x, y, u, v)
#        gids -- zero aligned indices of cells to get the tuning_prop
#        tuning_prop -- full np.array for receptive fields (x, y, u, v)
#        returns list of (spike_trains, spike rate envelope)
#        """


    def create_training_sequence_iteratively(self):
        """
        Training samples are drawn from the tuning properties of the cells, i.e. follow the same distribution
        Returns n_cycles of state vectors, each cycle containing a set of n_training_stim_per_cycle states.
        The set of states is shuffled for each cycle

        copied from OculomotorControl
        """
        mp_training = np.zeros((self.params['n_stim_training'], 4))
        training_states = np.zeros((self.params['n_training_stim_per_cycle'], 4))
        if self.params['n_training_stim_per_cycle'] == self.params['n_exc']:
            training_states_int = range(0, self.params['n_exc'])
        else:
            training_states_int = self.RNG.random_integers(0, self.params['n_exc'] - 1, self.params['n_training_stim_per_cycle'])
        training_states = self.tuning_prop[training_states_int, :]

        if self.params['n_stim_training'] == 1:
            x0 = self.params['initial_state'][0]
            v0 = self.params['initial_state'][2]
            mp_training[0, 0] = x0
            mp_training[0, 1] = .5
            mp_training[0, 2] = v0
        else:
            for i_cycle in xrange(self.params['n_training_cycles']):
                self.RNG.shuffle(training_states)
                i_ = i_cycle * self.params['n_training_stim_per_cycle']
                for i_stim in xrange(self.params['n_training_stim_per_cycle']):
                    plus_minus = utils.get_plus_minus(self.RNG)
                    mp_training[i_stim + i_, 0] = (training_states[i_stim][0] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_x'])) % 1.
                    mp_training[i_stim + i_, 1] = .5
                    plus_minus = utils.get_plus_minus(self.RNG)
                    mp_training[i_stim + i_, 2] =  training_states[i_stim][2] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_v'])
                    mp_training[i_stim + i_, 3] =  0.

        np.savetxt(self.params['training_stimuli_fn'], mp_training)
        return mp_training 


    def create_training_sequence_from_a_grid(self, n_stim=None):
        """
        Training samples are generated in a grid-like manner, i.e. random points from a grid on the tuning property space
        are drawn
         
        Returns n_cycles of state vectors, each cycle containing a set of n_training_stim_per_cycle states.
        The set of states is shuffled for each cycle.
        """
        if n_stim == None:
            n_stim = self.params['n_stim_training']
        mp_training = np.zeros((n_stim, 4))

        x_lim_frac = .9
        v_lim_frac = .8
        x_lim = ((1. - x_lim_frac) * (np.max(self.tuning_prop[:, 0]) - np.min(self.tuning_prop[:, 0])), x_lim_frac * np.max(self.tuning_prop[:, 0]))
        v_lim = (v_lim_frac * np.min(self.tuning_prop[:, 2]), v_lim_frac * np.max(self.tuning_prop[:, 2]))

        n_training_x = self.params['n_training_x']
        x_grid = np.linspace(x_lim[0], x_lim[1], n_training_x)
        v_grid = np.linspace(v_lim[0], v_lim[1], self.params['n_training_v'])
        training_states_x = range(0, n_training_x)
        training_states_v = range(0, self.params['n_training_v'])
        training_states = []
        for i_, x in enumerate(training_states_x):
            for j_, v in enumerate(training_states_v):
                training_states.append((x_grid[i_], v_grid[j_]))

        if self.params['n_stim_training'] == 1:
            x0 = self.params['initial_state'][0]
            v0 = self.params['initial_state'][2]
            mp_training[0, 0] = x0
            mp_training[0, 1] = .5
            mp_training[0, 2] = v0
        else:
            for i_cycle in xrange(self.params['n_training_cycles']):
                self.RNG.shuffle(training_states)
                i_ = i_cycle * self.params['n_training_stim_per_cycle']
                for i_stim in xrange(self.params['n_training_stim_per_cycle']):
                    plus_minus = utils.get_plus_minus(self.RNG)
                    mp_training[i_stim + i_, 0] = (training_states[i_stim][0] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_x'])) % 1.
                    mp_training[i_stim + i_, 1] = .5
                    plus_minus = utils.get_plus_minus(self.RNG)
                    mp_training[i_stim + i_, 2] =  training_states[i_stim][1] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_v'])
                    mp_training[i_stim + i_, 3] =  training_states[i_stim][1] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_v'])
        np.savetxt(self.params['training_stimuli_fn'], mp_training)
        return mp_training 


    def create_training_sequence_around_center(self):

        n_center = int(np.round(self.params['n_stim_training'] * self.params['frac_training_samples_center']))

        mp_center = np.zeros((n_center, 4))
        mp_center[:, 0] = self.RNG.normal(.5, self.params['center_stim_width'] + 0.01, n_center)
        mp_center[:, 2] = self.RNG.uniform(-self.params['v_max_tp'], self.params['v_max_tp'], n_center)
        return mp_center
        

    def create_test_stim_grid(self, params):
        n_stim = params['n_stim']
        vlim = (params['v_min_training'], params['v_max_training'])
        v_test = np.linspace(vlim[0], vlim[1], params['n_stim'], endpoint=True)
        mp_test = np.zeros((params['n_stim'], 4))
        x_idx_pos = (v_test > 0.).nonzero()[0]
        x_idx_neg = (v_test <= 0.).nonzero()[0]
        x_test = np.zeros(n_stim)
        for i_ in x_idx_pos:
            x_test[i_] = 0.1
        for i_ in x_idx_neg:
            x_test[i_] = 0.9

        for i_ in xrange(n_stim):
            mp_test[i_, :] = x_test, .5, v_test, .0


    def create_test_stim_1D_from_training_stim(self, test_params, training_params):
        n_stim = test_params['n_test_stim']
        # choose the first n_stim motion parameters from trainin_params
        mp_training = np.loadtxt(training_params['training_stimuli_fn'])
        stim_idx_0 = test_params['test_stim_range'][0]
        stim_idx_1 = test_params['test_stim_range'][1]
        motion_params = np.zeros((stim_idx_1 - stim_idx_0, 4))
        motion_params[stim_idx_0:stim_idx_1, :] = mp_training[stim_idx_0:stim_idx_1, :]
        self.all_starting_pos = motion_params[:, 0:2]
        self.all_speeds = motion_params[:n_stim, 2]
        self.all_thetas = np.zeros(n_stim)
        self.motion_params_created = True
        return mp_training # return all motion params, for convenience in PlotPrediction


    def create_test_stim_1D_not_trained(self, test_params):
        # create n_stim as during training
        n_stim = test_params['n_test_stim']
        stim_params = np.zeros((n_stim, 4))
        all_speeds = np.zeros(n_stim)
        all_starting_pos = np.zeros((n_stim, 2))
        visual_stim_seed = test_params['visual_stim_seed']
        self.RNG = np.random.RandomState(visual_stim_seed)

        random.seed(test_params['visual_stim_seed'] + 1)
        # create stimulus ranges as during training
        if test_params['log_scale']==1:
            speeds = np.linspace(test_params['v_min_training'], test_params['v_max_training'], num=test_params['n_training_v'], endpoint=True)
        else:
            speeds = np.logspace(np.log(test_params['v_min_training'])/np.log(test_params['log_scale']),
                            np.log(test_params['v_max_training'])/np.log(test_params['log_scale']), num=test_params['n_training_v'],
                            endpoint=True, base=test_params['log_scale'])
#            stim_cnt = 0
        for stim_cnt in xrange(n_stim):
            speed = speeds[stim_cnt]
            v0 = utils.get_plus_minus(self.RNG) * speed * self.RNG.uniform(1. - test_params['training_stim_noise_v'], 1. + test_params['training_stim_noise_v'])
            x0 = self.RNG.uniform(test_params['x_min_training'], test_params['x_max_training'])
            all_starting_pos[stim_cnt, 0] = x0
            all_speeds[stim_cnt] = v0
#            for cycle in xrange(test_params['n_training_cycles']):
#                for i_speed, speed in enumerate(speeds):
#                    for i_ in xrange(test_params['n_stim_per_direction']):
                    # add noise for the speed
#                        stim_cnt += 1
        stim_params[:, 0] = all_starting_pos[:, 0]
        stim_params[:, 1] = all_starting_pos[:, 1]
        stim_params[:, 2] = all_speeds

        # required for plotting
        self.all_starting_pos = all_starting_pos[:, 0:2]
        self.all_speeds = all_speeds
        self.all_thetas = np.zeros(n_stim)
        self.motion_params_created = True
        return stim_params



    def create_motion_sequence_1D_training(self, params, random_order):
        """Creates the motion parameters (x, y=0, u, v=0) for a sequence of one-dimensional stimuli
            
        Keyword arguments:
        random_order -- (bool) if True the sequence is shuffled
        """
        visual_stim_seed = params['visual_stim_seed']
        self.RNG = np.random.RandomState(visual_stim_seed)
        self.n_stim_per_direction = params['n_stim_per_direction']  # direction = speed in 1-D
        n_stim_total = params['n_stim_training']  # = n_training_cycles * n_training_v * n_stim_per_direction
        
        # arrays to be filled by the stimulus creation loops below
        self.all_speeds = np.zeros(n_stim_total)
        self.all_starting_pos = np.zeros((n_stim_total, 2))
        self.all_thetas = np.zeros(n_stim_total)

        import numpy.random as rnd
        rnd.seed(params['visual_stim_seed'])
        random.seed(params['visual_stim_seed'] + 1)

        # create stimulus ranges
        if params['log_scale'] == 1:
            speeds_pos = np.linspace(params['v_min_training'], params['v_max_training'], num=params['n_training_v'] / 2, endpoint=True)
        else:
            speeds_pos = np.logspace(np.log(params['v_min_training'])/np.log(params['log_scale']),
                            np.log(params['v_max_training'])/np.log(params['log_scale']), num=params['n_training_v'] / 2,
                            endpoint=True, base=params['log_scale'])


#        starting_pos = self.RNG.rand(params['n_training_stim_per_cycle'])
        starting_pos = np.linspace(0, 1., params['n_training_stim_per_cycle'])

        n_v = params['n_training_v']
        speeds = np.zeros(n_v)
        speeds[:n_v/2] = speeds_pos
        speeds[-(n_v/2):] = -speeds_pos

#        for i_speed, speed in enumerate(speeds):
#            speeds[i_speed] *= utils.get_plus_minus(self.RNG) 
#        print '\n\n'
        stim_cnt = 0
        stim_params = np.zeros((n_stim_total, 4))
        for cycle in xrange(params['n_training_cycles']):
            for i_speed, speed in enumerate(speeds):
                # add noise for the speed
                v0 = speed * self.RNG.uniform(1. - params['training_stim_noise_v'], 1. + params['training_stim_noise_v'])
                #v0 = speed * self.RNG.uniform(1. - test_params['training_noise_x'], 1. + test_params['training_noise_x'])
                x_pos = np.linspace(params['x_min_training'], params['x_max_training'], params['n_training_x'], endpoint=False)

                for i_x in xrange(params['n_training_x']):
#                    x0 = starting_pos[i_speed * params['n_training_x'] + i_x] + utils.get_plus_minus(self.RNG) * self.RNG.uniform(-params['training_stim_noise_x'], params['training_stim_noise_x'])

                    x0 = x_pos[i_x] + utils.get_plus_minus(self.RNG) * self.RNG.uniform(-params['training_stim_noise_x'], params['training_stim_noise_x'])
                    self.all_starting_pos[stim_cnt, 0] = x0 % 1.
                    self.all_speeds[stim_cnt] = v0
                    stim_cnt += 1
            stim_idx_0 = params['n_training_stim_per_cycle'] * cycle
            stim_idx_1 = params['n_training_stim_per_cycle'] * (cycle + 1)
            stim_order = range(stim_idx_0, stim_idx_1)
            if random_order:
                random.shuffle(stim_order)
            stim_params[stim_idx_0:stim_idx_1, 0] = self.all_starting_pos[stim_order, 0]
            stim_params[stim_idx_0:stim_idx_1, 1] = self.all_starting_pos[stim_order, 1]
            stim_params[stim_idx_0:stim_idx_1, 2] = self.all_speeds[stim_order]

        self.motion_params_created = True
        return stim_params
#        return pos_speed_sequence[stim_order, :]




    def create_motion_sequence_2D(self, params, random_order):
        """Creates the motion parameters (x, y, u, v) for a sequence of 2-dim stimuli

        Keyword arguments:
        random_order -- (bool) if True the sequence is shuffled
        """

        n_theta = params['n_theta_training']
        n_training_v = params['n_training_v']
        n_training_cycles = params['n_training_cycles']
        self.n_stim_per_direction = params['n_stim_per_direction']
        self.n_stim_total = n_training_v * n_theta * n_training_cycles * self.n_stim_per_direction
        random.seed(0)
        # arrays to be filled by the stimulus creation loops below
        self.all_speeds = np.zeros(self.n_stim_total)
        self.all_thetas = np.zeros(self.n_stim_total)
        self.all_starting_pos = np.zeros((self.n_stim_total, 2))

        # create stimulus ranges
        if params['log_scale']==1:
            speeds = np.linspace(params['v_min_training'], params['v_max_training'], num=params['n_v'], endpoint=True)
        else:
            speeds = np.logspace(np.log(params['v_min_training'])/np.log(params['log_scale']),
                            np.log(params['v_max_training'])/np.log(params['log_scale']), num=params['n_v'],
                            endpoint=True, base=params['log_scale'])
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        stim_cnt = 0
        for speed in xrange(n_training_v):
            v = speeds[speed]
            for cycle in xrange(n_training_cycles):
                # if random_order: for direction in random.shuffle(range(n_theta)):
                for direction in xrange(n_theta):
                    theta = thetas[direction]
                    print '\ntheta', theta, theta / (np.pi)

                    # decide where dot starts moving from
                    # 1
                    if theta == 0: # start on the left border (0, y)
                        y_0 = np.linspace(0, 1, self.n_stim_per_direction + 2)[1:-1]
                        x_0 = np.zeros(self.n_stim_per_direction)
                    elif theta == np.pi: # start on the right border (1., y)
                        y_0 = np.linspace(0, 1, self.n_stim_per_direction + 2)[1:-1]
                        x_0 = np.ones(self.n_stim_per_direction)

                    elif theta == .5 * np.pi: # start on the upper border (x, 0)
                        x_0 = np.linspace(0, 1, self.n_stim_per_direction + 2)[1:-1]
                        y_0 = np.ones(self.n_stim_per_direction)

                    elif theta == 1.5 * np.pi: # start on the lower border (x, 1)
                        x_0 = np.linspace(0, 1, self.n_stim_per_direction + 2)[1:-1]
                        y_0 = np.zeros(self.n_stim_per_direction)

                    elif theta < .5 * np.pi: # moving to lower right, start on the left or upper border
                        x_min, x_max = 0.0, .75 # improvement?: inrtoduce dependence of theta here
                        y_min, y_max = 0.25, 1.
                        up_or_left = np.array([i / int((self.n_stim_per_direction) /2) for i in range(self.n_stim_per_direction)])
                        upper_idx = up_or_left.nonzero()[0]
                        upper_x = np.linspace(x_min, x_max, upper_idx.size + 2)[1:-1]
                        x_0 = np.zeros(self.n_stim_per_direction)
                        y_0 = np.ones(self.n_stim_per_direction)
                        x_0[upper_idx] = upper_x
                        left_idx = up_or_left == 0
                        y_0[left_idx] = np.linspace(y_min, y_max, left_idx.nonzero()[0].size)

                    elif theta > .5 * np.pi and theta < np.pi: # moving to lower left, start on the right or upper border
                        x_min, x_max = 0, 1.
                        y_min, y_max = 0.5, 1.
                        upper_or_right = np.array([i / int((self.n_stim_per_direction) /2) for i in range(self.n_stim_per_direction)])
                        x_0 = np.ones(self.n_stim_per_direction)
                        upper_idx = upper_or_right.nonzero()[0]
                        right_idx = upper_or_right == 0
                        upper_x = np.linspace(x_min, x_max, upper_idx.size + 2)[1:-1]
                        x_0[upper_idx] = upper_x
                        y_0 = np.ones(self.n_stim_per_direction)
                        y_0[right_idx] = np.linspace(y_min, y_max, right_idx.nonzero()[0].size)

                    elif theta > np.pi and theta < 1.5 * np.pi: # moving to upper left, start on the right or bottom border
                        x_min, x_max = 0.25, 1.
                        y_min, y_max = 0.0, 0.5
                        bottom_or_right = np.array([i / int((self.n_stim_per_direction) /2) for i in range(self.n_stim_per_direction)])
                        x_0 = np.ones(self.n_stim_per_direction)
                        bottom_idx = bottom_or_right.nonzero()[0]
                        bottom_x = np.linspace(x_min, x_max, bottom_idx.size + 2)[1:-1]
                        x_0[bottom_idx] = bottom_x
                        y_0 = np.zeros(self.n_stim_per_direction)
                        right_idx = bottom_or_right == 0
                        y_0[right_idx] = np.linspace(y_min, y_max, right_idx.nonzero()[0].size)

                    elif theta > 1.5 * np.pi: # moving to upper right, starting at left or bottom border
                        x_min, x_max = 0., 0.75
                        y_min, y_max = 0., 0.75
                        bottom_or_left = np.array([i / int((self.n_stim_per_direction) /2) for i in range(self.n_stim_per_direction)])
                        x_0 = np.zeros(self.n_stim_per_direction)
                        bottom_idx = bottom_or_left.nonzero()[0]
                        bottom_x = np.linspace(x_min, x_max, bottom_idx.size + 2)[1:-1]
                        x_0[bottom_idx] = bottom_x
                        y_0 = np.zeros(self.n_stim_per_direction)
                        left_idx = bottom_or_left == 0
                        y_0[left_idx] = np.linspace(y_min, y_max, left_idx.nonzero()[0].size)

                    stim_order_for_one_direction = range(self.n_stim_per_direction)
                    if random_order:
                        random.shuffle(stim_order_for_one_direction)

                    for i in stim_order_for_one_direction:
        #            for i in xrange(self.n_stim_per_direction):
                        self.all_starting_pos[stim_cnt, :] = x_0[i], y_0[i]
                        self.all_speeds[stim_cnt] = v
                        rnd_rotation = params['sigma_theta_training'] * (np.random.rand() - .5)
                        self.all_thetas[stim_cnt] = theta + rnd_rotation
                        stim_cnt += 1
        self.motion_params_created = True


    def get_motion_params(self, random_order=False):

        assert (self.motion_params_created == True), '\nERROR: \n\tYou have to call one of the create_[training, test, ...] methods before calling get_motion_params'
        motion_params = np.zeros((self.all_starting_pos[:, 0].size, 5))
        n_stim_total = len(self.all_speeds)
        stim_order = range(n_stim_total)
        if random_order:
            random.shuffle(stim_order)
        motion_params[:, 0] = self.all_starting_pos[stim_order, 0]
        motion_params[:, 1] = self.all_starting_pos[stim_order, 1]
        motion_params[:, 2] = self.all_speeds[stim_order] * np.cos(self.all_thetas)
        motion_params[:, 3] = - self.all_speeds[stim_order] * np.sin(self.all_thetas)
        motion_params[:, 4] = self.all_thetas
        return motion_params




if __name__ == '__main__':
    import simulation_parameters
    PS = simulation_parameters.parameter_storage()
    params = PS.load_params()                       # params stores cell numbers, etc as a dictionary

    random_order = True

    CI = CreateInput(params)
    if params['n_grid_dimensions'] == 2:
        CI.create_motion_sequence_2D(params, random_order)
    else:
        if not params['training_run']: # TESTING
            if len(sys.argv) > 1: # test what has been trained
                training_folder = os.path.abspath(sys.argv[1]) # contains the EPTH and OB activity of simple patterns
                training_params = utils.load_params(training_folder)
                CI.create_test_stim_1D_from_training_stim(params, training_params=training_params)
            else: # test with something new
                training_folder = params['folder_name']
                training_params = utils.load_params(training_folder)
                params = training_params
                CI.create_test_stim_1D_not_trained(training_params)
        else:
            CI.create_motion_sequence_1D_training(params, random_order)

    import pylab
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    color_list = ['k', 'b', 'g', 'r', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']

    #init_rect
    ax.plot([0, 1], [0, 0], 'k--', lw=3)
    ax.plot([1, 1], [0, 1], 'k--', lw=3)
    ax.plot([1, 0], [1, 1], 'k--', lw=3)
    ax.plot([0, 0], [1, 0], 'k--', lw=3)

    x_max, y_max = 0, 0
    x_min, y_min = 0, 0
    motion_params = CI.get_motion_params(random_order)
    stim_start = 0
    if params['training_run']:
        stim_stop = params['n_stim_training']
    else:
        stim_stop = params['n_test_stim']
#    stim_start = CI.n_stim_per_direction * 8
#    stim_stop = CI.n_stim_per_direction * (9 + 1)
    n_stim_total = stim_stop - stim_start

    output_fn = params['training_stimuli_fn']
    try:
        print 'Saving motion parameters to:', output_fn
        np.savetxt(output_fn, motion_params)
    except:
        print '\nWARNING! folder %s does not exist!\nCould not save training_stimuli!' % (params['folder_name'])
        output_fn = raw_input('File name for saving training parameters\n')
        np.savetxt(output_fn, motion_params)
#        exit(1)
    plot = True
    if plot:

        for stim_id in xrange(stim_start, stim_stop):
            theta = motion_params[stim_id, 4]
            v = 5 * motion_params[stim_id, 2]
            vx, vy = v * np.cos(theta), - v * np.sin(theta)
            x0, y0 = motion_params[stim_id, 0:2]
            stim_params = [x0, y0, vx, vy]
            t_exit = utils.compute_stim_time(stim_params)
            print 'stim_params', stim_params, 't_exit', t_exit
            if params['n_grid_dimensions'] == 1: # for visibility reasons
                y0 = float(stim_id) / n_stim_total
    #            y0 = float(stim_id)
    #        print 'debug stim_id %d x0=%.3f  y0=%.3f  vx=%.3f  vy=%.3f, xpos=%.2f ypos=%.2f' % (stim_id, x0, y0, vx, vy, x_pos, y_pos)
            x_pos = x0 + vx
            y_pos = y0 + vy
            color_idx = (stim_id / params['n_stim_per_direction']) % len(color_list)
            x_max = max(x_pos, x_max)
            y_max = max(y_pos, y_max)
            x_min = min(x_pos, x_min)
            y_min = min(y_pos, y_min)

    #        ax.plot([x0, x_pos], [y0, y_pos], color=color_list[color_idx], lw=2)
    #        ax.plot([x0, x_pos], [y0, y_pos], color=color_list[color_idx], lw=2)
            scale = 1.
            ax.quiver(x0, y0, vx, vy, \
                  angles='xy', scale_units='xy', scale=scale, color=color_list[color_idx], headwidth=4, pivot='end', width=0.007)
    #        ax.plot([x0, x_pos], [y0, y_pos], color=color_list[color_idx], lw=2)

        
        ax.set_ylabel('Stimulus counter')
        ax.set_xlabel('Stimulus starting position')
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))

    #    ax.set_xlim((-0.2, 1.2))
    #    ax.set_ylim((-0.2, 1.2))

        pylab.show()
