import numpy as np
import random
import os
import sys
#matplotlib.use('Agg')


class CreateStimuli(object):
    def __init__(self):
        pass


    def create_test_stim_1D(self, test_params, training_params=None):

        n_stim = test_params['n_test_stim']
        self.n_stim_total = n_stim
        if training_params != None:
            # choose the first n_stim motion parameters from trainin_params
            mp_training = np.loadtxt(training_params['training_sequence_fn'])
            stim_idx_0 = test_params['test_stim_range'][0]
            stim_idx_1 = test_params['test_stim_range'][1]
            motion_params = np.zeros((stim_idx_1 - stim_idx_0, 4))
            motion_params[stim_idx_0:stim_idx_1, :] = mp_training[stim_idx_0:stim_idx_1, :]
            self.all_starting_pos = motion_params[:, 0:2]
            self.all_speeds = motion_params[:n_stim, 2]
            self.all_thetas = np.zeros(n_stim)

            return mp_training # return all motion params, for convenience in PlotPrediction
        else:
            # create n_stim as during training
            stim_params = np.zeros((n_stim, 4))
            all_speeds = np.zeros(n_stim)
            all_starting_pos = np.zeros((n_stim, 2))
            import numpy.random as rnd
            rnd.seed(test_params['stimuli_seed'])
            random.seed(test_params['stimuli_seed'] + 1)
            # create stimulus ranges as during training
            if test_params['log_scale']==1:
                speeds = np.linspace(test_params['v_min_training'], test_params['v_max_training'], num=test_params['n_speeds'], endpoint=True)
            else:
                speeds = np.logspace(np.log(test_params['v_min_training'])/np.log(test_params['log_scale']),
                                np.log(test_params['v_max_training'])/np.log(test_params['log_scale']), num=test_params['n_speeds'],
                                endpoint=True, base=test_params['log_scale'])
#            stim_cnt = 0
            for stim_cnt in xrange(n_stim):
                speed = speeds[stim_cnt]
                v0 = speed * rnd.uniform(1. - test_params['v_noise_training'], 1. + test_params['v_noise_training'])
                x0 = .25 * np.random.rand() # select a random start point between 0 - 0.25
                all_starting_pos[stim_cnt, 0] = x0
                all_speeds[stim_cnt] = v0
#            for cycle in xrange(test_params['n_cycles']):
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
            return stim_params




    def create_motion_sequence_1D(self, params, random_order):
        """Creates the motion parameters (x, y=0, u, v=0) for a sequence of one-dimensional stimuli
            
        Keyword arguments:
        random_order -- (bool) if True the sequence is shuffled
        """
        self.n_stim_per_direction = params['n_stim_per_direction']  # direction = speed in 1-D
        self.n_stim_total = params['n_training_stim']  # = n_cycles * n_speeds * n_stim_per_direction
        
        # arrays to be filled by the stimulus creation loops below
        self.all_speeds = np.zeros(self.n_stim_total)
        self.all_starting_pos = np.zeros((self.n_stim_total, 2))
        self.all_thetas = np.zeros(self.n_stim_total)

        import numpy.random as rnd
        rnd.seed(params['stimuli_seed'])
        random.seed(params['stimuli_seed'] + 1)

        # create stimulus ranges
        if params['log_scale'] == 1:
            speeds = np.linspace(params['v_min_training'], params['v_max_training'], num=params['n_speeds'], endpoint=True)
        else:
            speeds = np.logspace(np.log(params['v_min_training'])/np.log(params['log_scale']),
                            np.log(params['v_max_training'])/np.log(params['log_scale']), num=params['n_speeds'],
                            endpoint=True, base=params['log_scale'])

        stim_cnt = 0
        stim_params = np.zeros((self.n_stim_total, 4))
        for cycle in xrange(params['n_cycles']):
            for i_speed, speed in enumerate(speeds):
                for i_ in xrange(self.n_stim_per_direction):
                    # add noise for the speed
                    v0 = speed * rnd.uniform(1. - params['v_noise_training'], 1. + params['v_noise_training'])
                    self.all_starting_pos[stim_cnt, 0] = np.random.rand()
                    self.all_speeds[stim_cnt] = v0
                    stim_cnt += 1
            stim_idx_0 = params['n_speeds'] * cycle
            stim_idx_1 = params['n_speeds'] * (cycle + 1)
            stim_order = range(stim_idx_0, stim_idx_1)
            if random_order:
                random.shuffle(stim_order)
            stim_params[stim_idx_0:stim_idx_1, 0] = self.all_starting_pos[stim_order, 0]
            stim_params[stim_idx_0:stim_idx_1, 1] = self.all_starting_pos[stim_order, 1]
            stim_params[stim_idx_0:stim_idx_1, 2] = self.all_speeds[stim_order]

        return stim_params
#        return pos_speed_sequence[stim_order, :]




    def create_motion_sequence_2D(self, params, random_order):
        """Creates the motion parameters (x, y, u, v) for a sequence of 2-dim stimuli

        Keyword arguments:
        random_order -- (bool) if True the sequence is shuffled
        """

        n_theta = params['n_theta_training']
        n_speeds = params['n_speeds']
        n_cycles = params['n_cycles']
        self.n_stim_per_direction = params['n_stim_per_direction']
        self.n_stim_total = n_speeds * n_theta * n_cycles * self.n_stim_per_direction
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
        for speed in xrange(n_speeds):
            v = speeds[speed]
            for cycle in xrange(n_cycles):
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


    def get_motion_params(self, random_order=False):

        stim_order = range(self.n_stim_total)
        if random_order:
            random.shuffle(stim_order)
        return self.all_speeds[stim_order], self.all_starting_pos[stim_order, :], self.all_thetas[stim_order]


if __name__ == '__main__':
    import simulation_parameters
    PS = simulation_parameters.parameter_storage()
    params = PS.load_params()                       # params stores cell numbers, etc as a dictionary

    random_order = False
    CS = CreateStimuli()

    if params['n_grid_dimensions'] == 2:
        CS.create_motion_sequence_2D(params, random_order)
    else:
        if not params['training_run']:
            training_folder = os.path.abspath(sys.argv[1]) # contains the EPTH and OB activity of simple patterns
            print 'Training folder:', training_folder
            training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
            training_param_tool = simulation_parameters.parameter_storage(params_fn=training_params_fn)
            training_params = training_param_tool.params
            CS.create_test_stim_1D(params, training_params=training_params)
        else:
            CS.create_motion_sequence_1D(params, random_order)

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
    all_speeds, all_starting_pos, all_thetas = CS.get_motion_params(random_order)
    stim_start = 0
    if params['training_run']:
        stim_stop = params['n_training_stim']
    else:
        stim_stop = params['n_test_stim']
#    stim_start = CS.n_stim_per_direction * 8
#    stim_stop = CS.n_stim_per_direction * (9 + 1)
    for stim_id in xrange(stim_start, stim_stop):
        theta = all_thetas[stim_id]
        v = 5 * all_speeds[stim_id]
        vx, vy = v * np.cos(theta), - v * np.sin(theta)
        x0, y0 = all_starting_pos[stim_id, :]
        if params['n_grid_dimensions'] == 1: # for visibility reasons
            y0 = float(stim_id) / CS.n_stim_total
        print 'debug stim_id %d x0=%.3f  y0=%.3f  vx=%.3f  vy=%.3f' % (stim_id, x0, y0, vx, vy)
        x_pos = x0 + vx
        y_pos = y0 + vy
        print 'debug y_pos', y_pos
        color_idx = (stim_id / params['n_stim_per_direction']) % len(color_list)
        x_max = max(x_pos, x_max)
        y_max = max(y_pos, y_max)
        x_min = min(x_pos, x_min)
        y_min = min(y_pos, y_min)

#        ax.plot([x0, x_pos], [y0, y_pos], color=color_list[color_idx], lw=2)
#        ax.plot([x0, x_pos], [y0, y_pos], color=color_list[color_idx], lw=2)
        scale = 1.
        ax.quiver(x0, y0, vx, vy, \
              angles='xy', scale_units='xy', scale=scale, color=color_list[color_idx], headwidth=4, pivot='middle', width=0.007)
#        ax.plot([x0, x_pos], [y0, y_pos], color=color_list[color_idx], lw=2)

    
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
#    ax.set_xlim((-0.2, 1.2))
#    ax.set_ylim((-0.2, 1.2))
    pylab.show()
