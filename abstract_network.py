import os
import matplotlib
import simulation_parameters
import numpy as np

class AbstractNetwork(object):

    def __init__(self, params):

        self.params = params
        self.set_iteration(0)
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        assert (self.tuning_prop[:, 0].size == self.params['n_exc']), 'Number of cells does not match in %s and simulation_parameters!\n Wrong tuning_prop file?' % self.params['tuning_prop_means_fn']
        self.vx_tuning = self.tuning_prop[:, 2]
        self.vy_tuning = self.tuning_prop[:, 3]


    def set_iteration(self, iteration):
        self.iteration = iteration
        self.training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], iteration)


    def set_weights(self):
        weight_matrix_fn = params['folder_name'] + 'TrainingResults_%d/' % iteration + 'wij_matrix_%d.dat' % (iteration)
        if not os.path.exists(weight_matrix_fn):
            self.wij = self.get_weight_matrix()
        else:
            self.wij = np.loadtxt(weight_matrix_fn)


    def calculate_dynamics(self, output_fn_base=None):

        fn_base = self.training_input_folder + self.params['abstract_input_fn_base']
        n_cells = self.params['n_exc']

        cell = 0
        fn = fn_base + '%d.dat' % (cell)
        d_sample = np.loadtxt(fn)
        self.n_time_steps = d_sample.size
        hc_idx = np.loadtxt(params['parameters_folder'] + 'hc_list.dat')

        activity_traces = np.zeros((self.n_time_steps, n_cells))
        output_activity = np.zeros((self.n_time_steps, n_cells))
        fn_bias = params['folder_name'] + 'TrainingResults_%d/' % self.iteration + 'bias_%d.dat' % (self.iteration)
        bias = np.loadtxt(fn_bias)

        self.v_pred = np.zeros((self.n_time_steps, 3))
        self.t_axis = np.arange(self.n_time_steps)
        self.t_axis *= self.params['dt_rate']
        self.v_pred[:, 0] = self.t_axis

        for cell in xrange(n_cells):
            fn = fn_base + '%d.dat' % (cell)
            activity_traces[:, cell] = np.loadtxt(fn)

        for t in xrange(self.n_time_steps):
            output_activity[t, :] = activity_traces[t, :] + np.dot(self.wij.transpose(), activity_traces[t, :])# + bias[cell]
            # map activity in the range (0, 1)
            for cell in xrange(n_cells):
                if (output_activity[t, cell] < 0):
                    output_activity[t, cell] = 0

#             normalize activity within one HC to 1 (if larger than 1)
            for hc in xrange(hc_idx[:, 0].size):
                o_sum = output_activity[t, hc_idx[hc, 0]:hc_idx[hc, -1]].sum()
                if o_sum > 1:
                    output_activity[t, hc_idx[hc, 0]:hc_idx[hc, -1]] /= output_activity[t, hc_idx[hc, 0]:hc_idx[hc, -1]].sum()

            normed_activity = output_activity[t, :] / output_activity[t, :].sum()
            self.v_pred[t, 1] = np.dot(self.vx_tuning, normed_activity)
            self.v_pred[t, 2] = np.dot(self.vy_tuning, normed_activity)



        if output_fn_base == None:
            output_fn_activity = params['activity_folder'] + 'output_activity_%d.dat' % (self.iteration)
        print 'Saving ANN activity to:', output_fn_activity
        np.savetxt(output_fn_activity, output_activity)
        output_fn_prediction = params['activity_folder'] + 'prediction_%d.dat' % (self.iteration)
        print 'Saving ANN prediction to:', output_fn_prediction
        np.savetxt(output_fn_prediction, self.v_pred)


    def get_weight_matrix(self):
        all_wij_fn= '%sTrainingResults_%d/all_wij_%d.dat' % (self.params['folder_name'], self.iteration, self.iteration)
        print 'Getting weights from ', all_wij_fn
        d = np.loadtxt(all_wij_fn)
        n_cells = self.params['n_exc']
        wij_matrix = np.zeros((n_cells, n_cells))
        bias_matrix = np.zeros(n_cells)
        for line in xrange(d[:, 0].size):
            i, j, pij_, wij, bias_j = d[line, :]
            bias_matrix[j] = bias_j
            wij_matrix[i, j] = wij
        output_fn = params['folder_name'] + 'TrainingResults_%d/' % iteration + 'wij_matrix_%d.dat' % (iteration)
        print 'Saving to:', output_fn
        np.savetxt(output_fn, wij_matrix)
        return wij_matrix


    def eval_prediction(self):
        x0, y0, u0, v0 = np.loadtxt(self.training_input_folder + 'input_params.txt')
        v_stim = np.zeros((self.n_time_steps, 3))
        v_stim[:, 0] = self.t_axis
        v_stim[:, 1] = u0 * np.ones(self.n_time_steps) 
        v_stim[:, 2] = v0 * np.ones(self.n_time_steps) 

        vx_diff = self.v_pred[:, 1] - v_stim[:, 1]
        vy_diff = self.v_pred[:, 2] - v_stim[:, 2]
        v_diff = [np.sqrt(vx_diff[t]**2 + vy_diff[t]**2) for t in xrange(self.n_time_steps)]

        v_diff_out = np.zeros((self.n_time_steps, 4))
        v_diff_out[:, 0] = self.t_axis
        v_diff_out[:, 1] = vx_diff
        v_diff_out[:, 2] = vy_diff
        v_diff_out[:, 3] = v_diff

        output_fn_prediction_error = params['activity_folder'] + 'prediction_error_%d.dat' % (self.iteration)
        print 'Saving ANN prediction error to:', output_fn_prediction_error
        np.savetxt(output_fn_prediction_error, v_diff_out)





if __name__ == '__main__':

    PS = simulation_parameters.parameter_storage()
    params = PS.params
    n_iter = 80

    ANN = AbstractNetwork(params)

    for iteration in xrange(n_iter):
        ANN.set_iteration(iteration)
        ANN.set_weights() # load weight matrix
        ANN.calculate_dynamics() # load activity files 
        ANN.eval_prediction()



#    if plot_dynamics:
#        import matplotlib.animate as animate
        # load output files
#        for t in xrange(n_time_steps):
        # plot cell activities

