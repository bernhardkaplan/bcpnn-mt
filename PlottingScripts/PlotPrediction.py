import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import simulation_parameters
import utils
from matplotlib import cm
import re

class PlotPrediction(object):
    def __init__(self, params=None, data_fn=None):

        if params == None:
            self.network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
            self.params = self.network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        else:
            self.params = params
        self.no_spikes = False

        self.spiketimes = {} # will store all spike times with cell_type as key
        self.n_fig_x = 2
        self.n_fig_y = 2
#        self.fig_size = (11.69, 8.27) #A4
        self.fig_size = (14, 10)
        if self.params['t_blank'] == 0:
            self.show_blank = False
        else:
            self.show_blank = True

        self.spiketimes_loaded = False
        self.data_to_store = {}
        # define parameters
#        self.time_binsize = int(round(self.params['t_sim'] / 20))
        self.time_binsize = 50# [ms] 

        self.trace_length = 4 * self.time_binsize # [ms] window length for moving average 
        self.n_bins = int((self.params['t_sim'] / self.time_binsize) )
        self.time_bins = [self.time_binsize * i for i in xrange(self.n_bins)]
        self.t_axis = np.arange(0, self.n_bins * self.time_binsize, self.time_binsize)
#        self.n_vx_bins, self.n_vy_bins = 30, 30     # colormap grid dimensions for predicted direction
#        self.n_x_bins, self.n_y_bins = 50, 50       # colormap grid dimensions for predicted position
        self.n_vx_bins, self.n_vy_bins = 10, 10# colormap grid dimensions for predicted direction
        self.n_x_bins, self.n_y_bins = 20, 20       # colormap grid dimensions for predicted position
        self.t_ticks = np.linspace(0, self.params['t_sim'], 6)

        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.n_cells = self.tuning_prop[:, 0].size #self.params['n_exc']
#        assert (self.tuning_prop[:, 0].size == self.params['n_exc']), 'Number of cells does not match in %s and simulation_parameters!\n Wrong tuning_prop file?' % self.params['tuning_prop_means_fn']

        # create data structures
        self.nspikes = np.zeros(self.n_cells)                                   # summed activity
        self.nspikes_binned = np.zeros((self.n_cells, self.n_bins))             # binned activity over time
        self.nspikes_binned_normalized = np.zeros((self.n_cells, self.n_bins))  # normalized so that for each bin, the sum of the population activity = 1
        self.nspikes_normalized = np.zeros(self.n_cells)                        # activity normalized, so that sum = 1
        self.spiketrains = [[] for i in xrange(self.n_cells)]

        # sort the cells by their tuning vx, vy properties
        # vx
        self.vx_tuning = self.tuning_prop[:, 2].copy()
        self.vx_tuning.sort()
        self.sorted_indices_vx = self.tuning_prop[:, 2].argsort()
        # maximal range of vx_speeds
        self.vx_min, self.vx_max = .7 * self.tuning_prop[:, 2].min(), .7 * self.tuning_prop[:, 2].max()
#        self.vx_min, self.vx_max = np.min(self.vx_tuning), np.max(self.vx_tuning)
        self.vx_grid = np.linspace(self.vx_min, self.vx_max, self.n_vx_bins, endpoint=True)

        # vy
        self.vy_tuning = self.tuning_prop[:, 3].copy()
        self.vy_tuning.sort()
        self.sorted_indices_vy = self.tuning_prop[:, 3].argsort()
#        self.vy_min, self.vy_max = -0.5, 0.5
        self.vy_min, self.vy_max = self.tuning_prop[:, 3].min(), self.tuning_prop[:, 3].max()
#        self.vy_min, self.vy_max = np.min(self.vy_tuning), np.max(self.vy_tuning)
        self.vy_grid = np.linspace(self.vy_min, self.vy_max, self.n_vy_bins, endpoint=True)

        # x
        self.sorted_indices_x = self.tuning_prop[:, 0].argsort()
        self.x_tuning = self.tuning_prop[:, 0].copy()
        self.x_tuning.sort()
        self.x_min, self.x_max = .0, self.params['torus_width']
        self.x_grid = np.linspace(self.x_min, self.x_max, self.n_x_bins, endpoint=True)

        # y
        self.y_tuning = self.tuning_prop[:, 1].copy()
        self.y_tuning.sort()
        self.sorted_indices_y = self.tuning_prop[:, 1].argsort()
        self.y_min, self.y_max = .0, self.params['torus_height']
        self.y_grid = np.linspace(self.y_min, self.y_max, self.n_y_bins, endpoint=True)

        if self.params['training_run']:
            self.t_stim = self.params['t_training_stim']
            self.all_motion_params = np.loadtxt(self.params['training_sequence_fn'])
        else:
            self.t_stim = self.params['t_test_stim']
            self.all_motion_params = np.loadtxt(self.params['test_sequence_fn'])

        self.normalize_spiketimes(data_fn)
        if self.no_spikes:
            return
        fig_width_pt = 800.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        params = {#'backend': 'png',
                  'titel.fontsize': 16,
                  'axes.labelsize': 14,
#                  'text.fontsize': 10,
#                  'legend.fontsize': 10,
#                  'xtick.labelsize': 8,
#                  'ytick.labelsize': 8,
#                  'text.usetex': True,
                  'figure.figsize': fig_size}
        pylab.rcParams.update(params)


    def normalize_spiketimes(self, fn=None):
        """
        Fills the following arrays with data:
        self.nspikes = np.zeros(self.n_cells)                                   # summed activity
        self.nspikes_binned = np.zeros((self.n_cells, self.n_bins))             # binned activity over time
        self.nspikes_binned_normalized = np.zeros((self.n_cells, self.n_bins))  # normalized so that for each bin, the sum of the population activity = 1
        self.nspikes_normalized = np.zeros(self.n_cells)                        # activity normalized, so that sum = 1
        self.nspikes_normalized_nonlinear
        """

        print 'Loading data .... '
        if not os.path.exists(fn):
            cell_type = 'exc'
            utils.merge_and_sort_files(self.params['%s_spiketimes_fn_base' % cell_type], self.params['%s_spiketimes_fn_merged' % cell_type])

        self.nspikes, self.spiketrains = utils.get_nspikes(fn, get_spiketrains=True)
        for gid in xrange(self.n_cells):
            if (self.nspikes[gid] > 0):
                count, bins = np.histogram(self.spiketrains[gid], bins=self.n_bins, range=(0, self.params['t_sim']))
                self.nspikes_binned[gid, :] = count

        # normalization
        for i in xrange(int(self.n_bins)):
            if (self.nspikes_binned[:, i].sum() > 0):
                self.nspikes_binned_normalized[:, i] = self.nspikes_binned[:, i] / self.nspikes_binned[:,i].sum()
        self.nspikes_normalized = self.nspikes / self.nspikes.sum()

        # activity normalized, nonlinear
        nspikes_shifted = self.nspikes - self.nspikes.max()
        nspikes_exp = np.exp(nspikes_shifted)
        self.nspikes_normalized_nonlinear = nspikes_exp / nspikes_exp.sum()


    def bin_estimates(self, grid_edges, index=2):
        """
        Bring the speed estimates from the neuronal level to broader representation in a grid:
        index = index in tuning_parameters for the parameter (vx=2, vy=3)

                    ^
        vx_binned   |
                    |
                    +------>
                    time_bins

        """
        # torus dimensions
        w, h = self.params['torus_width'], self.params['torus_height']
        
        output_data = np.zeros((len(grid_edges), self.n_bins))
        for gid in xrange(self.n_cells):
            xyuv_predicted = self.tuning_prop[gid, index] # cell tuning properties
            if (index == 0):
                xyuv_predicted += self.tuning_prop[gid, 2]
#                xyuv_predicted = xyuv_predicted % w

            elif (index == 1):
                xyuv_predicted += self.tuning_prop[gid, 3]
#                xyuv_predicted = xyuv_predicted % h
            y_pos_grid = utils.get_grid_pos_1d(xyuv_predicted, grid_edges)
            output_data[y_pos_grid, :] += self.nspikes_binned_normalized[gid, :]
        return output_data, grid_edges



    def get_average_of_circular_quantity(self, confidence_vec, tuning_vec, xv='x'):
        """
        Computes the population average of a circular quantity.
        This is done by 1) mapping the quantity onto a circle
        2) weighting the single quantities on the circle
        3) getting the average by using arctan2 giving the 'directed' angle
        """

        if xv == 'x':
            range_0_1 = True
        else:
            range_0_1 = False

        n = confidence_vec.size
        sin = np.zeros(n)
        cos = np.zeros(n)

        if range_0_1:
            sin = confidence_vec * np.sin(tuning_vec * 2 * np.pi - np.pi) 
            cos = confidence_vec * np.cos(tuning_vec * 2 * np.pi - np.pi)
            avg = .5 * (np.arctan2(sin.sum(), cos.sum()) / np.pi + 1.)
        else: # range_-1_1
            sin = confidence_vec * np.sin(tuning_vec * np.pi) 
            cos = confidence_vec * np.cos(tuning_vec * np.pi)
            avg = np.arctan2(sin.sum(), cos.sum()) / np.pi

        return avg



    def compute_pos_and_v_estimates(self):
        """
        This function combines activity on the population level to estimate vx, vy

         On which time scale shall the prediction work?
         There are (at least) 3 different ways to do it:
           Very short time-scale:
           1) Compute the prediction for each time bin - based on the activitiy in the respective time bin 
           Short time-scale:
           2) Compute the prediction for each time bin based on all activity in the past
           3) Non-linear 'voting' based on 1)
           Long time-scale:
           3) Compute the prediction based on the the activity of the whole run - not time dependent
           4) Non-linear 'voting' based on 3) 
        """
        print 'Computing v estimates...'

        self.x_stim = np.zeros(self.n_bins) # stimulus positions binned
        self.y_stim = np.zeros(self.n_bins)
        # momentary result, based on the activity in one time bin
        self.x_avg = np.zeros(self.n_bins) 
        self.y_avg = np.zeros(self.n_bins)
        self.xdiff_avg = np.zeros(self.n_bins)  # stores |x_predicted(t) - x_stimulus(t)|
        self.vx_avg = np.zeros(self.n_bins) 
        self.vy_avg = np.zeros(self.n_bins)
        self.vdiff_avg = np.zeros(self.n_bins)  # stores |v_predicted(t) - v_stimulus(t)|
        # ---> gives theta_avg 

        # based on the activity in several time bins
        self.x_moving_avg = np.zeros((self.n_bins, 2))
        self.y_moving_avg = np.zeros((self.n_bins, 2))
        self.xdiff_moving_avg = np.zeros((self.n_bins, 2))
        self.vx_moving_avg = np.zeros((self.n_bins, 2))
        self.vy_moving_avg = np.zeros((self.n_bins, 2))
        self.vdiff_moving_avg = np.zeros((self.n_bins, 2))

        # non linear transformation of vx_avg
        self.x_non_linear = np.zeros(self.n_bins)
        self.y_non_linear = np.zeros(self.n_bins)
        self.xdiff_non_linear = np.zeros(self.n_bins)
        self.vx_non_linear = np.zeros(self.n_bins)
        self.vy_non_linear = np.zeros(self.n_bins)
        self.vdiff_non_linear = np.zeros(self.n_bins)

        trace_length_in_bins = int(round(self.trace_length / self.time_binsize))
        # ---> gives theta_moving_avg

        # # # # # # # # # # # # # # # # # # # # # # 
        # L O C A T I O N     P R E D I C T I O N 
        # # # # # # # # # # # # # # # # # # # # # # 
        self.x_confidence_binned = self.nspikes_binned_normalized[self.sorted_indices_x]
        self.y_confidence_binned = self.nspikes_binned_normalized[self.sorted_indices_y]
        x_prediction_trace = np.zeros((self.n_cells, self.n_bins, 2))    # _trace: prediction based on the momentary and past activity (moving average, and std) --> trace_length
        y_prediction_trace = np.zeros((self.n_cells, self.n_bins, 2))    # _trace: prediction based on the momentary and past activity (moving average, and std) --> trace_length
        # # # # # # # # # # # # # # # # # # # # # # 
        # S P E E D    P R E D I C T I O N 
        # # # # # # # # # # # # # # # # # # # # # # 
        self.vx_confidence_binned = self.nspikes_binned_normalized[self.sorted_indices_vx]
        self.vy_confidence_binned = self.nspikes_binned_normalized[self.sorted_indices_vy]
        vx_prediction_trace = np.zeros((self.n_cells, self.n_bins, 2))    # _trace: prediction based on the momentary and past activity (moving average, and std) --> trace_length
        vy_prediction_trace = np.zeros((self.n_cells, self.n_bins, 2))    # _trace: prediction based on the momentary and past activity (moving average, and std) --> trace_length


        # torus dimensions
        w, h = self.params['torus_width'], self.params['torus_height']
        for i in xrange(self.n_bins):
            t = i * self.time_binsize + .5 * self.time_binsize
            stim_idx = t / self.t_stim
            mp = self.all_motion_params[stim_idx, :]

            # 1) momentary vote
            # take the weighted average for v_prediction (weight = normalized activity)
            vx_pred = self.vx_confidence_binned[:, i] * self.vx_tuning
            vy_pred = self.vy_confidence_binned[:, i] * self.vy_tuning

            self.vx_avg[i] = self.get_average_of_circular_quantity(self.vx_confidence_binned[:, i], self.vx_tuning, xv='v')
            self.vy_avg[i] = self.get_average_of_circular_quantity(self.vy_confidence_binned[:, i], self.vy_tuning, xv='v')
#            self.vx_avg[i] = np.sum(vx_pred)
#            self.vy_avg[i] = np.sum(vy_pred)
            self.vdiff_avg[i] = np.sqrt((mp[2] - self.vx_avg[i])**2 + (mp[3] - self.vy_avg[i])**2)

            # position
            stim_pos_x = (mp[0] + mp[2] * t / self.params['t_stimulus']) % w# be sure that this works the same as utils.get_input is called!
            stim_pos_y = (mp[1] + mp[3] * t / self.params['t_stimulus']) % h # be sure that this works the same as utils.get_input is called!
            self.x_stim[i] = stim_pos_x
            self.y_stim[i] = stim_pos_y
            x_pred = self.x_confidence_binned[:, i] * self.x_tuning
            y_pred = self.y_confidence_binned[:, i] * self.y_tuning
            self.x_avg[i] = self.get_average_of_circular_quantity(self.x_confidence_binned[:, i], self.x_tuning, xv='x')
            self.y_avg[i] = self.get_average_of_circular_quantity(self.y_confidence_binned[:, i], self.y_tuning, xv='x')
#            self.x_avg[i] = np.sum(x_pred)
#            self.y_avg[i] = np.sum(y_pred)
            self.xdiff_avg[i] = np.sqrt((stim_pos_x - self.x_avg[i])**2 + (stim_pos_y - self.y_avg[i])**2)

            # 2) moving average
            past_bin = max(0, i-trace_length_in_bins)

            if i == past_bin:
                self.x_moving_avg[i, 0] = self.x_avg[i]
                self.x_moving_avg[i, 1] = 0.
                self.y_moving_avg[i, 0] = self.y_avg[i]
                self.y_moving_avg[i, 1] = 0.
                self.vx_moving_avg[i, 0] = self.vx_avg[i]
                self.vx_moving_avg[i, 1] = 0.
                self.vy_moving_avg[i, 0] = self.vy_avg[i]
                self.vy_moving_avg[i, 1] = 0.
            else:
                self.x_moving_avg[i, 0] = self.x_avg[past_bin:i].mean()
                self.x_moving_avg[i, 1] = self.x_avg[past_bin:i].std()
                self.y_moving_avg[i, 0] = self.y_avg[past_bin:i].mean()
                self.y_moving_avg[i, 1] = self.y_avg[past_bin:i].std()
                self.vx_moving_avg[i, 0] = self.vx_avg[past_bin:i].mean()
                self.vx_moving_avg[i, 1] = self.vx_avg[past_bin:i].std()
                self.vy_moving_avg[i, 0] = self.vy_avg[past_bin:i].mean()
                self.vy_moving_avg[i, 1] = self.vy_avg[past_bin:i].std()

            # x moving average
            self.xdiff_moving_avg[i, 0] = np.sqrt((stim_pos_x - self.x_moving_avg[i, 0])**2 + (stim_pos_y - self.y_moving_avg[i, 0])**2)
            x_diff = (self.x_avg[i] - stim_pos_x)
            y_diff = (self.y_avg[i] - stim_pos_x)
            self.xdiff_moving_avg[i, 1] = (2. / self.xdiff_moving_avg[i, 0]) * ( x_diff * self.x_moving_avg[i, 1] + y_diff * self.y_moving_avg[i, 1])

            # v
            self.vdiff_moving_avg[i, 0] = np.sqrt((mp[2] - self.vx_moving_avg[i, 0])**2 + (mp[3] - self.vy_moving_avg[i, 0])**2)
            # propagation of uncertainty
            vx_diff = self.vx_moving_avg[i ,0] - mp[2]
            vy_diff = self.vy_moving_avg[i ,0] - mp[3]
            self.vdiff_moving_avg[i, 1] = (2. / self.vdiff_moving_avg[i, 0]) * ( vx_diff * self.vx_moving_avg[i, 1] + vy_diff * self.vy_moving_avg[i, 1])

            # 3) soft-max
            # x
            # rescale activity to negative values
            x_shifted = self.nspikes_binned[self.sorted_indices_x, i] - self.nspikes_binned[self.sorted_indices_x, i].max()
            y_shifted = self.nspikes_binned[self.sorted_indices_y, i] - self.nspikes_binned[self.sorted_indices_y, i].max()
            # exp --> mapping to range(0, 1)
            x_exp = np.exp(x_shifted)
            y_exp = np.exp(y_shifted)
            # normalize and vote
#            x_votes = (x_exp / x_exp.sum()) * self.x_tuning
#            y_votes = (y_exp / y_exp.sum()) * self.y_tuning
#            self.x_non_linear[i] = x_votes.sum()
#            self.y_non_linear[i] = y_votes.sum()
#            self.xdiff_non_linear[i] = np.sqrt((stim_pos_x - self.x_non_linear[i])**2 + (stim_pos_y - self.y_non_linear[i])**2)
            self.x_non_linear[i] = self.get_average_of_circular_quantity(x_exp, self.x_tuning, xv='x')
            self.y_non_linear[i] = self.get_average_of_circular_quantity(y_exp, self.x_tuning, xv='x')
            self.xdiff_non_linear[i] = np.sqrt((stim_pos_x - self.x_non_linear[i])**2 + (stim_pos_y - self.y_non_linear[i])**2)

            # v
            # rescale activity to negative values
            vx_shifted = self.nspikes_binned[self.sorted_indices_vx, i] - self.nspikes_binned[self.sorted_indices_vx, i].max()
            vy_shifted = self.nspikes_binned[self.sorted_indices_vy, i] - self.nspikes_binned[self.sorted_indices_vy, i].max()
            # exp --> mapping to range(0, 1)
            vx_exp = np.exp(vx_shifted)
            vy_exp = np.exp(vy_shifted)
            # normalize and vote
#            vx_votes = (vx_exp / vx_exp.sum()) * self.vx_tuning
#            vy_votes = (vy_exp / vy_exp.sum()) * self.vy_tuning
#            self.vx_non_linear[i] = vx_votes.sum()
#            self.vy_non_linear[i] = vy_votes.sum()
#            self.vdiff_non_linear[i] = np.sqrt((mp[2] - self.vx_non_linear[i])**2 + (mp[3] - self.vy_non_linear[i])**2)

            self.vx_non_linear[i] = self.get_average_of_circular_quantity(vx_exp, self.vx_tuning, xv='v')
            self.vy_non_linear[i] = self.get_average_of_circular_quantity(vy_exp, self.vy_tuning, xv='v')
            self.vdiff_non_linear[i] = np.sqrt((mp[2]- self.vx_non_linear[i])**2 + (mp[3]- self.vy_non_linear[i])**2)

        # in the first step the trace can not have a standard deviation --> avoid NANs 
        self.x_moving_avg[0, 0] = np.sum(self.x_confidence_binned[self.sorted_indices_x, 0] * self.x_tuning)
        self.y_moving_avg[0, 0] = np.sum(self.y_confidence_binned[self.sorted_indices_y, 0] * self.y_tuning)
        self.x_moving_avg[0, 1] = 0
        self.y_moving_avg[0, 1] = 0
        self.xdiff_moving_avg[0, 1] = 0
        self.xdiff_moving_avg[0, 1] = 0

        self.vx_moving_avg[0, 0] = np.sum(self.vx_confidence_binned[self.sorted_indices_vx, 0] * self.vx_tuning)
        self.vy_moving_avg[0, 0] = np.sum(self.vy_confidence_binned[self.sorted_indices_vy, 0] * self.vy_tuning)
        self.vx_moving_avg[0, 1] = 0
        self.vy_moving_avg[0, 1] = 0
        self.vdiff_moving_avg[0, 1] = 0
        self.vdiff_moving_avg[0, 1] = 0

        # ---> time INdependent estimates: based on activity of the full run

        # compute the marginalized (over all positions) vx, vy estimates and bin them in a grid
        # is omitted for position because full run estimates for a moving stimulus do not make sense
        self.vx_marginalized_binned = np.zeros(self.n_vx_bins)
        self.vy_marginalized_binned = np.zeros(self.n_vy_bins)
        self.vx_marginalized_binned_nonlinear = np.zeros(self.n_vx_bins)
        self.vy_marginalized_binned_nonlinear = np.zeros(self.n_vy_bins)

        for gid in xrange(self.n_cells):
            vx_cell, vy_cell = self.tuning_prop[gid, 2], self.tuning_prop[gid, 3] # cell properties
            vx_grid_pos, vy_grid_pos = utils.get_grid_pos(vx_cell, vy_cell, self.vx_grid, self.vy_grid)
            self.vx_marginalized_binned[vx_grid_pos] += self.nspikes_normalized[gid]
            self.vy_marginalized_binned[vy_grid_pos] += self.nspikes_normalized[gid]
            self.vx_marginalized_binned_nonlinear[vx_grid_pos] += self.nspikes_normalized_nonlinear[gid]
            self.vy_marginalized_binned_nonlinear[vy_grid_pos] += self.nspikes_normalized_nonlinear[gid]

#        assert (np.sum(self.vx_marginalized_binned) == 1.), "Marginalization incorrect: %.10e" % (np.sum(self.vx_marginalized_binned))
#        assert (np.sum(self.vx_marginalized_binned_nonlinear) == 1.), "Marginalization incorrect: %f" % (np.sum(self.vx_marginalized_binned_nonlinear))
#        assert (np.sum(self.vy_marginalized_binned) == 1.), "Marginalization incorrect: %f" % (np.sum(self.vy_marginalized_binned))
#        assert (np.sum(self.vy_marginalized_binned_nonlinear) == 1.), "Marginalization incorrect: %f" % (np.sum(self.vy_marginalized_binned))


    def save_data(self):
        output_folder = self.params['data_folder']

        for key in self.data_to_store:
            d = self.data_to_store[key]
            data = d['data']
            fn = self.params['data_folder'] + key
            print 'Saving data to:', fn
            np.savetxt(fn, data)

    def compute_theta_estimates(self):

        # time dependent averages
        self.theta_avg = np.arctan2(self.vy_avg, self.vx_avg)
        self.theta_moving_avg = np.zeros((self.n_bins, 2))
        self.theta_moving_avg[:, 0] = np.arctan2(self.vy_moving_avg[:, 0], self.vx_moving_avg[:, 0])
        self.theta_moving_avg[:, 1] = self.theta_uncertainty(self.vx_moving_avg[:, 0], self.vx_moving_avg[:, 1], self.vy_moving_avg[:, 0], self.vy_moving_avg[:, 1])
        self.theta_non_linear = np.arctan2(self.vy_non_linear, self.vx_non_linear)

        # full run estimates
        all_thetas = np.arctan2(self.tuning_prop[:, 3], self.tuning_prop[:, 2])
        self.theta_grid = np.linspace(np.min(all_thetas), np.max(all_thetas), self.n_vx_bins, endpoint=True)
        self.theta_marginalized_binned = np.zeros(self.n_vx_bins)
        self.theta_marginalized_binned_nonlinear = np.zeros(self.n_vx_bins)
        for gid in xrange(self.n_cells):
            theta = np.arctan2(self.tuning_prop[gid, 3], self.tuning_prop[gid, 2])
            grid_pos = utils.get_grid_pos_1d(theta, self.theta_grid)
            self.theta_marginalized_binned[grid_pos] += self.nspikes_normalized[gid]
            self.theta_marginalized_binned_nonlinear[grid_pos] += self.nspikes_normalized_nonlinear[gid]

#        assert (np.sum(self.theta_marginalized_binned) == 1), "Marginalization incorrect: %.1f" % (np.sum(self.theta_marginalized_binned))
#        assert (np.sum(self.theta_marginalized_binned_nonlinear) == 1), "Marginalization incorrect: %.1f" % (np.sum(self.theta_marginalized_binned_nonlinear))


    def create_fig(self):
        print "plotting ...."
        self.fig = pylab.figure(figsize=self.fig_size)
        pylab.subplots_adjust(hspace=0.4)
        pylab.subplots_adjust(wspace=0.35)



    def load_spiketimes(self, cell_type):

        fn = self.params['%s_spiketimes_fn_merged' % cell_type]
        n_cells = self.params['n_%s' % cell_type]
        nspikes, spiketimes = utils.get_nspikes(fn, n_cells, get_spiketrains=True)
        np.savetxt(self.params['%s_nspikes_fn_merged' % cell_type] + '.dat', nspikes)
        idx = np.nonzero(nspikes)[0]
        np.savetxt(self.params['%s_nspikes_nonzero_fn' % cell_type], np.array((idx, nspikes[idx])).transpose())
        self.spiketimes[cell_type] = spiketimes
        self.spiketimes_loaded = True
        return spiketimes, nspikes


    def plot_rasterplot(self, cell_type, fig_cnt=1, show_blank=None):
        spiketimes, nspikes = self.load_spiketimes(cell_type)
        if show_blank == None:
            show_blank = self.show_blank

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        for cell in xrange(int(len(spiketimes))):
            ax.plot(spiketimes[cell], cell * np.ones(nspikes[cell]), 'o', color='k', markersize=1)
            
        ylim = ax.get_ylim()
        ax.set_ylim((0, self.params['n_%s' % cell_type]))
        ax.set_xlim(0, self.params['t_sim'])
        ax.set_title('Rasterplot of %s neurons' % cell_type)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Neuron GID')

        if show_blank:
            self.plot_blank(ax)


    def plot_raster_sorted(self, time_range=None, fig_cnt=1, title='', cell_type='exc', sort_idx=0):
        """
        sort_idx : the index in tuning properties after which the cell gids are to be sorted for  the rasterplot
        """
        if time_range == None:
            n_stim_0 = 0
            if self.params['training_run']:
                n_stim_1 = self.params['n_training_stim']
            else:
                n_stim_1 = self.params['n_test_stim']
        else:
            n_stim_0 = int(time_range[0] / self.t_stim)
            n_stim_1 = int(time_range[1] / self.t_stim)

        tp = self.tuning_prop
        tp_idx_sorted = tp[:, sort_idx].argsort()
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title(title)
        for i_, gid in enumerate(tp_idx_sorted):
#            spikes = utils.get_spiketimes(self.spike_times_merged, gid + 1)
            spikes = self.spiketrains[gid]
            nspikes = spikes.size
            y_ = np.ones(spikes.size) * tp[gid, sort_idx]
            ax.plot(spikes, y_, 'o', markersize=3, markeredgewidth=0., color='k')

        ylim = ax.get_ylim()
        for stim in xrange(n_stim_0, n_stim_1):
            t0 = stim * self.t_stim
            t1 = (stim + 1) * self.t_stim
            ax.plot((t0, t0), (ylim[0], ylim[1]), '--', c='k', lw=1)
            ax.plot((t1, t1), (ylim[0], ylim[1]), '--', c='k', lw=1)


    def plot_input_spikes_sorted(self, time_range=None, fig_cnt=1, title='', sort_idx=0):
        """
        Input spikes are stored in seperate files for each cell.
        --> filenames are Folder/InputSpikes/stim_spike_train_[GID].dat GID = cell gid
        """
        print 'Plotting input spikes ...'
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        tp = self.tuning_prop
        tp_idx_sorted = tp[:, sort_idx].argsort() # + 1 because nest indexing

        cnt_file = 0
        for fn in os.listdir(self.params['input_folder']):
            m = re.match('stim_spike_train_(\d+).dat', fn)
            if m:
                gid = int(m.groups()[0])
                if (gid < self.params['n_exc']):
                    y_pos_of_cell = tp[gid, sort_idx]
                    fn_ = self.params['input_folder'] + fn
                    d = np.loadtxt(fn_)
                    ax.plot(d, y_pos_of_cell * np.ones(d.size), 'o', markersize=3, markeredgewidth=0., alpha=.1, color='b')
                    cnt_file += 1

        if time_range == None:
            time_range = (0, self.params['t_sim'])
        ax.set_xlim((time_range[0], time_range[1]))


    def plot_network_activity(self, cell_type, fig_cnt=1):

        n_cells = self.params['n_%s' % cell_type]
        spiketimes = self.spiketimes[cell_type]

        n_bins = int(round(self.params['t_sim'] / self.time_binsize))
        binned_spiketimes = [[] for i in xrange(n_cells)]
        avg_activity = np.zeros(n_bins)
        n_active = 0
        for cell in xrange(n_cells):
            if len(spiketimes[cell]) > 0:
                binned_spiketimes[cell], bins = np.histogram(spiketimes[cell], n_bins, range=(0, self.params['t_sim']))
                n_active += 1
        for time_bin in xrange(n_bins):
            for cell in xrange(n_cells):
                if len(spiketimes[cell]) > 0:
                    avg_activity[time_bin] += binned_spiketimes[cell][time_bin]
            avg_activity[time_bin] /= n_active
        avg_activity /= (self.time_binsize / 1000.)
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_xlim((0, self.params['t_sim']))
        bins = np.linspace(0, self.params['t_sim'], n_bins, endpoint=True)
        ax.bar(bins, avg_activity, width=bins[1]-bins[0], )
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Average firing rate [Hz]')
        ax.set_title('Activity of %s cells' % cell_type)


    def plot_vx_grid_vs_time(self, fig_cnt=1, time_range=None):
        print 'plot_vx_grid_vs_time ... '
        xlabel = 'Time [ms]'
        ylabel = '$v_x$'
#        title = '$v_x$ binned vs time'
        title = ''
        vx_grid, v_edges = self.bin_estimates(self.vx_grid, index=2)
        if time_range != None:
            bin0 = int(round(time_range[0] / self.time_binsize))
            bin1 = int(round(time_range[1] / self.time_binsize))
            vx_grid = vx_grid[:, bin0:bin1]
        self.plot_grid_vs_time(vx_grid, title, xlabel, ylabel, v_edges, fig_cnt, time_range=time_range)
        self.data_to_store['vx_grid.dat'] = {'data' : vx_grid, 'edges': v_edges}



    def plot_vy_grid_vs_time(self, fig_cnt=1, time_range=None):
        print 'plot_vy_grid_vs_time ...'
        xlabel = 'Time [ms]'
        ylabel = '$v_y$'
        title = ''#$v_y$ binned vs time'
        vy_grid, v_edges = self.bin_estimates(self.vy_grid, index=3)
        if time_range != None:
            bin0 = int(round(time_range[0] / self.time_binsize))
            bin1 = int(round(time_range[1] / self.time_binsize))
            vy_grid = vy_grid[:, bin0:bin1]
        self.plot_grid_vs_time(vy_grid, title, xlabel, ylabel, v_edges, fig_cnt, time_range=time_range)
        self.data_to_store['vy_grid.dat'] = {'data' : vy_grid, 'edges': v_edges}


    def plot_x_grid_vs_time(self, fig_cnt=1, ylabel=None, time_range=None):
        print 'plot_x_grid_vs_time ...'
        xlabel = 'Time [ms]'
        if ylabel == None:
            ylabel = '$x_{predicted}$'
        title = ''#$x_{predicted}$ binned vs time'
        x_grid, x_edges = self.bin_estimates(self.x_grid, index=0)
        if time_range != None:
            bin0 = int(round(time_range[0] / self.time_binsize))
            bin1 = int(round(time_range[1] / self.time_binsize))
            x_grid = x_grid[:, bin0:bin1]
        self.plot_grid_vs_time(x_grid, title, xlabel, ylabel, x_edges, fig_cnt, time_range=time_range)
        self.data_to_store['xpos_grid.dat'] = {'data' : x_grid, 'edges': x_edges}


    def plot_y_grid_vs_time(self, fig_cnt=1, ylabel=None, time_range=None):
        print 'plot_y_grid_vs_time ...'
        xlabel = 'Time [ms]'
        if ylabel == None:
            ylabel = '$y_{predicted}$'
        title = ''#$y_{predicted}$ binned vs time'
        y_grid, y_edges = self.bin_estimates(self.y_grid, index=1)
        if time_range != None:
            bin0 = int(round(time_range[0] / self.time_binsize))
            bin1 = int(round(time_range[1] / self.time_binsize))
            y_grid = y_grid[:, bin0:bin1]
        self.plot_grid_vs_time(y_grid, title, xlabel, ylabel, y_edges, fig_cnt, time_range=time_range)
        self.data_to_store['ypos_grid.dat'] = {'data' : y_grid, 'edges': y_edges}


    def plot_grid_vs_time(self, data, title='', xlabel='', ylabel='', yticks=[], \
            fig_cnt=1, show_blank=None, time_range=None):
        """
        Plots a colormap / grid versus time
        """
        if show_blank == None:
            show_blank = self.show_blank
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title(title)
        cax = ax.pcolormesh(data)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        y_ticks = range(len(yticks))[::5]
        ax.set_yticks(y_ticks)
        y_ticklabels = ['%.2f' %i for i in yticks[::5]]
        ax.set_yticklabels(y_ticklabels)

        if time_range == None:
            n_x_bins = len(self.t_ticks)
            x_bin_labels = ['%d' % i for i in self.t_ticks]
            ax.set_xticks(np.linspace(0, self.n_bins, n_x_bins))#range(self.n_bins)[::4])
            ax.set_xticklabels(x_bin_labels)
        else:
            n_time_ticks = 5
            x_ticks = np.linspace(time_range[0] / self.time_binsize, time_range[1] / self.time_binsize, n_time_ticks)
            x_ticklabels  = ['%d' % i for i in np.linspace(time_range[0], time_range[1], n_time_ticks)]
#            x_bin_labels = ['%d' % i for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)

        ax.set_ylim((0, data[:, 0].size))
        ax.set_xlim((0, data[0, :].size))

#        max_conf = min(data.mean() + data.std(), data.max())
        max_conf = data.max()
        norm = matplotlib.mpl.colors.Normalize(vmin=0, vmax=max_conf)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)#jet)
        m.set_array(np.arange(0., max_conf, 0.01))
        cb = pylab.colorbar(m, orientation='horizontal', aspect=40, anchor=(.5, .0))
        ticklabels = cb.ax.get_xticklabels()
#        print 'ticklabels', ticklabels, type(ticklabels)
#        ticklabels = list(ticklabels)
        ticklabel_texts = []
        for text in ticklabels:
            ticklabel_texts.append('%s' % text.get_text())
        ticklabel_texts[-1] = '$\geq %.1f$' % max_conf
        for i_, text in enumerate(ticklabels):
            text.set_text(ticklabel_texts[i_])
        cb.ax.set_xticklabels(ticklabel_texts)
        cb.set_label('Prediction confidence')
        if show_blank:
            self.plot_blank_on_cmap(cax, txt='blank')


    def plot_xdiff(self, fig_cnt=1, show_blank=None, time_range=None):
        if show_blank == None:
            show_blank = self.show_blank
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('Position prediction error: \n $|\\vec{x}_{diff}(t)| = |\\vec{x}_{stim}(t) - \\vec{x}_{predicted}(t)|$')#, fontsize=self.plot_params['title_fs'])
        ax.plot(self.t_axis, self.xdiff_avg, ls='-', lw=2, label='linear')
#        ax.plot(self.t_axis, self.xdiff_moving_avg[:, 0], ls='--', lw=2, label='moving avg')
#        ax.errorbar(self.t_axis, self.xdiff_moving_avg[:, 0], yerr=self.xdiff_moving_avg[:, 1], ls='--', lw=2, label='moving avg')
#        ax.plot(self.t_axis, self.xdiff_non_linear, ls=':', lw=2, label='soft-max')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$|\\vec{x}_{diff}|$')
        ax.legend()#loc='upper right')
        ny = self.t_axis.size
        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
#        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in self.t_ticks]
        ax.set_xticks(self.t_ticks)
        ax.set_xticklabels(t_labels)
        if time_range == None:
            time_range = (0, self.params['t_sim'])
        ax.set_xlim((time_range[0], time_range[1]))
        if show_blank:
            self.plot_blank(ax)

        print 'xdiff_avg.sum:', self.xdiff_avg.sum()
        output_data = np.zeros((self.t_axis.size, 4))
        output_data[:, 0] = self.t_axis
        output_data[:, 1] = self.xdiff_avg
        output_data[:, 2] = self.xdiff_moving_avg[:, 0]
        output_data[:, 3] = self.xdiff_non_linear
        output_fn = self.params['xdiff_vs_time_fn']
        self.data_to_store[output_fn] = {'data' : output_data}
    

    def plot_vdiff(self, fig_cnt=1, show_blank=None, time_range=None):
        if show_blank == None:
            show_blank = self.show_blank
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('Velocity prediction error: \n $|\\vec{v}_{diff}(t)| = |\\vec{v}_{stim}-\\vec{v}_{predicted}(t)|$')#, fontsize=self.plot_params['title_fs'])
        ax.plot(self.t_axis, self.vdiff_avg, ls='-', lw=2, label='linear')
#        ax.plot(self.t_axis, self.vdiff_moving_avg[:, 0], ls='--', lw=2, label='moving avg')
#        ax.errorbar(self.t_axis, self.vdiff_moving_avg[:, 0], yerr=self.vdiff_moving_avg[:, 1], ls='--', lw=2, label='moving avg')
#        ax.plot(self.t_axis, self.vdiff_non_linear, ls=':', lw=2, label='soft-max')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$|\\vec{v}_{diff}|$')
        ax.legend()#loc='upper right')
        ny = self.t_axis.size
#        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
#        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in self.t_ticks]
        ax.set_xticks(self.t_ticks)
        ax.set_xticklabels(t_labels)
        if time_range == None:
            time_range = (0, self.params['t_sim'])
        ax.set_xlim((time_range[0], time_range[1]))
        if show_blank:
            self.plot_blank(ax)

        print 'vdiff_avg.sum:', self.vdiff_avg.sum()
        output_data = np.zeros((self.t_axis.size, 4))
        output_data[:, 0] = self.t_axis
        output_data[:, 1] = self.vdiff_avg
        output_data[:, 2] = self.vdiff_moving_avg[:, 0]
        output_data[:, 3] = self.vdiff_non_linear
        output_fn = self.params['vdiff_vs_time_fn']
        self.data_to_store[output_fn] = {'data' : output_data}
    

    def plot_nspikes_binned(self):

        ax = self.fig.add_subplot(421)
        ax.set_title('Spiking activity over time')
        self.cax = ax1.pcolormesh(self.nspikes_binned)
        ax.set_ylim((0, self.nspikes_binned[:, 0].size))
        ax.set_xlim((0, self.nspikes_binned[0, :].size))
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('GID')
        ax.set_xticks(range(self.n_bins)[::2])
        ax.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        pylab.colorbar(self.cax)

    def plot_nspikes_binned_normalized(self):

        ax = self.fig.add_subplot(422)
        ax.set_title('Normalized activity over time')
        self.cax = ax2.pcolormesh(self.nspikes_binned_normalized)
        ax.set_ylim((0, self.nspikes_binned_normalized[:, 0].size))
        ax.set_xlim((0, self.nspikes_binned_normalized[0, :].size))
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('GID')
        ax.set_xticks(range(self.n_bins)[::2])
        ax.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        pylab.colorbar(self.cax)


    def plot_vx_confidence_binned(self):
        ax = self.fig.add_subplot(423)
        ax.set_title('Vx confidence over time')
        self.cax = ax.pcolormesh(self.vx_confidence_binned)
        ax.set_ylim((0, self.vx_confidence_binned[:, 0].size))
#        ax.set_xlim((0, self.vx_confidence_binned[0, :].size))
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$u$')
        ax.set_xticks(range(self.n_bins)[::2])
        ax.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        ny = self.vx_tuning.size
        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
        yticks = [self.vx_tuning[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        ylabels = ['%.1e' % i for i in yticks]
        ax.set_yticks([int(i * ny/n_ticks) for i in xrange(n_ticks)])
        ax.set_yticklabels(ylabels)
        ax.set_xticks(range(self.n_bins)[::2])
        ax.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        ax.set_xlim(0, self.params['t_sim'])
        pylab.colorbar(self.cax)



    def plot_vy_confidence_binned(self):
        ax = self.fig.add_subplot(424)
        ax.set_title('vy confidence over time')
        self.cax = ax.pcolormesh(self.vy_confidence_binned)
        ax.set_ylim((0, self.vy_confidence_binned[:, 0].size))
#        ax.set_xlim((0, self.vy_confidence_binned[0, :].size))
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$\vecv$')
        ax.set_xticks(range(self.n_bins)[::2])
        ax.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        ny = self.vy_tuning.size
        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
        yticks = [self.vy_tuning[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        ylabels = ['%.1e' % i for i in yticks]
        ax.set_yticks([int(i * ny/n_ticks) for i in xrange(n_ticks)])
        ax.set_yticklabels(ylabels)
        ax.set_xticks(range(self.n_bins)[::2])
        ax.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        ax.set_xlim(0, self.params['t_sim'])
        pylab.colorbar(self.cax)

    def plot_x_estimates(self, fig_cnt=1, show_blank=None, time_range=None):

        if show_blank == None:
            show_blank = self.show_blank
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('$x$-predictions')#: avg, moving_avg, nonlinear')
        ax.plot(self.t_axis, self.x_avg, ls='-', lw=2, label='linear')
#        ax.plot(self.t_axis, self.x_moving_avg[:, 0], ls='--', lw=2, label='moving avg')
#        ax.errorbar(self.t_axis, self.x_moving_avg[:, 0], yerr=self.x_moving_avg[:, 1], ls='--', lw=2, label='moving avg')
#        ax.plot(self.t_axis, self.x_non_linear, ls=':', lw=2, label='soft-max')
        ax.plot(self.t_axis, self.x_stim, ls='-', c='k', lw=2, label='$x_{stim}$')
        ax.legend()#loc='upper left')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$x$ position [a.u.]')
        ny = self.t_axis.size
        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
#        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in self.t_ticks]
        ax.set_xticks(self.t_ticks)
        ax.set_xticklabels(t_labels)

        if time_range == None:
            time_range = (0, self.params['t_sim'])
        ax.set_xlim((time_range[0], time_range[1]))
        if show_blank:
            self.plot_blank(ax)


    def plot_y_estimates(self, fig_cnt=1, show_blank=None, time_range=None):
        if show_blank == None:
            show_blank = self.show_blank
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('$y$-predictions')#: avg, moving_avg, nonlinear')
        ax.plot(self.t_axis, self.y_avg, ls='-', lw=2, label='linear')
#        ax.plot(self.t_axis, self.y_moving_avg[:, 0], ls='--', lw=2, label='moving avg')
#        ax.errorbar(self.t_axis, self.y_moving_avg[:, 0], yerr=self.y_moving_avg[:, 1], ls='--', lw=2, label='moving avg')
#        ax.plot(self.t_axis, self.y_non_linear, ls=':', lw=2, label='soft-max')
        ax.plot(self.t_axis, self.y_stim, ls='-', c='k', lw=2, label='$y_{stim}$')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$y$ position [a.u.]')
#        ax.legend()
        ax.legend()#loc='lower right')
        ny = self.t_axis.size
        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
#        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in self.t_ticks]
        ax.set_xticks(self.t_ticks)
        ax.set_xticklabels(t_labels)
        if time_range == None:
            time_range = (0, self.params['t_sim'])
        ax.set_xlim((time_range[0], time_range[1]))
        if show_blank:
            self.plot_blank(ax)



    def plot_vx_estimates(self, fig_cnt=1, show_blank=None, time_range=None):
        if show_blank == None:
            show_blank = self.show_blank
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('$v_{x}$-predictions')#: avg, moving_avg, nonlinear')
        ax.plot(self.t_axis, self.vx_avg, ls='-', lw=2, label='linear')
#        ax.plot(self.t_axis, self.vx_moving_avg[:, 0], ls='--', lw=2, label='moving avg')
#        ax.errorbar(self.t_axis, self.vx_moving_avg[:, 0], yerr=self.vx_moving_avg[:, 1], ls='--', lw=2, label='moving avg')
#        ax.plot(self.t_axis, self.vx_non_linear, ls=':', lw=2, label='soft-max')
        vx = self.params['motion_params'][2] * np.ones(self.t_axis.size)
        ax.plot(self.t_axis, vx, ls='-', c='k', lw=2, label='$v_{y, stim}$')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$v_x$')
        ax.legend()#loc='lower right')
        ny = self.t_axis.size
        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
#        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in self.t_ticks]
        ax.set_xticks(self.t_ticks)
        ax.set_xticklabels(t_labels)
        if time_range == None:
            time_range = (0, self.params['t_sim'])
        ax.set_xlim((time_range[0], time_range[1]))
        if show_blank:
            self.plot_blank(ax)

        output_data = np.array((self.t_axis, self.vx_avg))
        self.data_to_store['vx_linear_vs_time.dat'] = {'data' : output_data}


    def plot_vy_estimates(self, fig_cnt=1, show_blank=None, time_range=None):
        if show_blank == None:
            show_blank = self.show_blank
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.plot(self.t_axis, self.vy_avg, ls='-', lw=2, label='linear')
#        ax.plot(self.t_axis, self.vy_moving_avg[:, 0], lw=2, ls='--', label='moving avg')
#        ax.errorbar(self.t_axis, self.vy_moving_avg[:, 0], yerr=self.vy_moving_avg[:, 1], lw=2, ls='--', label='moving avg')
#        ax.plot(self.t_axis, self.vy_non_linear, ls=':', lw=2, label='soft-max')
        vy = self.params['motion_params'][3] * np.ones(self.t_axis.size)
        ax.plot(self.t_axis, vy, ls='-', c='k', lw=2, label='$v_{y, stim}$')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$v_y$')
        ax.set_title('$v_{y}$-predictions')#: avg, moving_avg, nonlinear')
        ax.legend()#loc='lower right')
        ny = self.t_axis.size
        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
#        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in self.t_ticks]
        ax.set_xticks(self.t_ticks)
        ax.set_xticklabels(t_labels)
        if time_range == None:
            time_range = (0, self.params['t_sim'])
        ax.set_xlim((time_range[0], time_range[1]))
        if show_blank:
            self.plot_blank(ax)

        output_data = np.array((self.t_axis, self.vy_avg))
        self.data_to_store['vy_linear_vs_time.dat'] = {'data' : output_data}

    def plot_theta_estimates(self, fig_cnt=1, show_blank=None, time_range=None):
        if show_blank == None:
            show_blank = self.show_blank
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('$\Theta$-predictions: avg, moving_avg, nonlinear')
        ax.plot(self.t_axis, self.theta_avg, ls='-')
#        ax.plot(self.t_axis, self.theta_moving_avg[:, 0], ls='--')
#        ax.errorbar(self.t_axis, self.theta_moving_avg[:, 0], yerr=self.theta_moving_avg[:, 1], ls='--')
#        ax.plot(self.t_axis, self.theta_non_linear, ls=':')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('$\Theta$')
        ny = self.t_axis.size
        n_ticks = min(11, int(round(self.params['t_sim'] / 100.)))
#        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in self.t_ticks]
        ax.set_xticks(self.t_ticks)
        ax.set_xticklabels(t_labels)
        if time_range == None:
            time_range = (0, self.params['t_sim'])
        ax.set_xlim((time_range[0], time_range[1]))
        if show_blank:
            self.plot_blank(ax)

        output_data = np.array((self.t_axis, self.theta_avg))
        self.data_to_store['theta_linear_vs_time.dat'] = {'data' : output_data}


    def plot_fullrun_estimates_vx(self, fig_cnt=1):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        bin_width = .5 * (self.vx_grid[1] - self.vx_grid[0])
        vx_linear = (np.sum(self.vx_grid * self.vx_marginalized_binned), self.get_uncertainty(self.vx_marginalized_binned, self.vx_grid))
        vx_nonlinear = (np.sum(self.vx_grid * self.vx_marginalized_binned_nonlinear), self.get_uncertainty(self.vx_marginalized_binned_nonlinear, self.vx_grid))
        ax.bar(self.vx_grid, self.vx_marginalized_binned, width=bin_width, label='Linear votes: $v_x=%.2f \pm %.2f$' % (vx_linear[0], vx_linear[1]))
        ax.bar(self.vx_grid+bin_width, self.vx_marginalized_binned_nonlinear, width=bin_width, facecolor='g', label='Non-linear votes: $v_x=%.2f \pm %.2f$' % (vx_nonlinear[0], vx_nonlinear[1]))
        ax.set_title('Estimates based on full run activity with %s connectivity\nblue: linear marginalization over all positions, green: non-linear voting' % self.params['connectivity_code'])
        ax.set_xlabel('$v_x$')
        ax.set_ylabel('Confidence')
        ax.legend()


    def get_uncertainty(self, p, v):
        """
        p, v are vectors storing the confidence of the voters in p, and the values they vote for in v.
        The uncertainty is estimated as:
        sum_i p_i * (1. - p_i) * v_i
        Idea behind it:
        (1. - p_i) * v_i gives the uncertainty for each vote of v_i
        multiplying it with p_i takes into account how much weight this uncertainty should have in the overall vote
        """
        uncertainties = (np.ones(len(p)) - p) * v
        weighted_uncertainties = p * uncertainties
        return np.sum(weighted_uncertainties)

    def plot_fullrun_estimates_vy(self, fig_cnt=1):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        bin_width = .5 * (self.vy_grid[1] - self.vy_grid[0])
        vy_linear = (np.sum(self.vy_grid * self.vy_marginalized_binned), self.get_uncertainty(self.vy_marginalized_binned, self.vy_grid))
        vy_nonlinear = (np.sum(self.vy_grid * self.vy_marginalized_binned_nonlinear), self.get_uncertainty(self.vy_marginalized_binned_nonlinear, self.vy_grid))
        ax.bar(self.vy_grid, self.vy_marginalized_binned, width=bin_width, label='Linear votes: $v_y=%.2f \pm %.2f$' % (vy_linear[0], vy_linear[1]))
        ax.bar(self.vy_grid+bin_width, self.vy_marginalized_binned_nonlinear, width=bin_width, facecolor='g', label='Non-linear votes: $v_y=%.2f \pm %.2f$' % (vy_nonlinear[0], vy_nonlinear[1]))
        ax.set_xlabel('$v_y$')
        ax.set_ylabel('Confidence')
        ax.legend()

    def plot_fullrun_estimates_theta(self, fig_cnt=1):

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        bin_width = .5 * (self.theta_grid[-1] - self.theta_grid[-2])
        theta_linear = (np.sum(self.theta_grid * self.theta_marginalized_binned), self.get_uncertainty(self.theta_marginalized_binned, self.theta_grid))
        theta_nonlinear = (np.sum(self.theta_grid * self.theta_marginalized_binned_nonlinear), self.get_uncertainty(self.theta_marginalized_binned_nonlinear, self.theta_grid))
        ax.bar(self.theta_grid, self.theta_marginalized_binned, width=bin_width, label='Linear votes: $\Theta=%.2f \pm %.2f$' % (theta_linear[0], theta_linear[1]))
        ax.bar(self.theta_grid+bin_width, self.theta_marginalized_binned_nonlinear, width=bin_width, facecolor='g', label='Non-linear votes: $\Theta=%.2f \pm %.2f$' % (theta_nonlinear[0], theta_nonlinear[1]))
        ax.bar(self.theta_grid, self.theta_marginalized_binned, width=bin_width)
        ax.bar(self.theta_grid+bin_width, self.theta_marginalized_binned_nonlinear, width=bin_width, facecolor='g')
        ax.set_xlim((-np.pi, np.pi))
        ax.legend()


#        n_bins = 50
#        count, theta_bins = np.histogram(self.theta_tuning, n_bins)
#        pred_avg, x = np.histogram(self.theta_avg_fullrun, n_bins)
#        pred_nonlinear, x = np.histogram(self.theta_nonlinear_fullrun, n_bins)
#        bin_width = theta_bins[1]-theta_bins[0]
#        ax.bar(theta_bins[:-1], pred_avg, width=bin_width*.5)
#        ax.bar(theta_bins[:-1]-.5*bin_width, pred_nonlinear, width=bin_width*.5, facecolor='g')
#        ax.set_xlim((self.theta_tuning.min() - bin_width, self.theta_tuning.max()))
        ax.set_xlabel('$\Theta$')
        ax.set_ylabel('Confidence')

    def plot_nspike_histogram(self, fig_cnt=1):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        mean_nspikes = self.nspikes.mean()* 1000./self.params['t_sim'] 
        std_nspikes = self.nspikes.std() * 1000./self.params['t_sim']
        ax.bar(range(self.n_cells), self.nspikes* 1000./self.params['t_sim'], label='$f_{mean} = (%.1f \pm %.1f)$ Hz' % (mean_nspikes, std_nspikes))
        ax.set_xlabel('Cell gids')
        ax.set_ylabel('Output rate $f_{out}$')
        ax.legend()

    def theta_uncertainty(self, vx, dvx, vy, dvy):
        """
        theta = arctan(vy / vx)
        Please check with http://en.wikipedia.org/wiki/Propagation_of_uncertainty
        """
        return vx / (vx**2 + vy**2) * dvy - vy / (vx**2 + vx**2) * dvx


    def quiver_plot(self, weights, title='', fig_cnt=1):
        """
        Cells are binned according to their spatial position (tuning prop) and for each spatial bin, the resulting predicted vector is computed:
         - v_prediction[x, y] = sum of all predicted directions by cells positioned at x, y
         weights = confidence for each cell's vote
        """
        # place cells in spatial grid --> look-up table: gid | (x_pos, y_pos)
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)

        n_bins_x, n_bins_y = 20, 20
        x_edges = np.linspace(self.tuning_prop[:, 0].min(), self.tuning_prop[:, 0].max(), n_bins_x)
        y_edges = np.linspace(self.tuning_prop[:, 1].min(), self.tuning_prop[:, 1].max(), n_bins_y)
        x_edge_width = x_edges[-1] - x_edges[-2]
        y_edge_width = y_edges[-1] - y_edges[-2]
        vx_in_grid = np.zeros((n_bins_x, n_bins_y))
        vy_in_grid = np.zeros((n_bins_x, n_bins_y))
        v_norm = np.zeros((n_bins_x, n_bins_y))

        for gid in xrange(self.n_cells):
            x_pos_cell, y_pos_cell = self.tuning_prop[gid, 0], self.tuning_prop[gid, 1] # cell properties
            x, y = utils.get_grid_pos(x_pos_cell, y_pos_cell, x_edges, y_edges) # cell's position in the grid
            vx_in_grid[x, y] += weights[gid] * self.tuning_prop[gid, 2]
            vy_in_grid[x, y] += weights[gid] * self.tuning_prop[gid, 3]

        scale = .1
        ax.quiver(x_edges, y_edges, vx_in_grid, vy_in_grid, angles='xy', scale_units='xy', scale=scale)
        x_key, y_key, u_key, v_key = self.params['motion_params'][0], self.params['motion_params'][1], self.params['motion_params'][2], self.params['motion_params'][3]
        key_scale = 1#.05
        key_label ='Stimulus'
        ax.quiver(x_key, y_key, u_key, v_key, color='y', angles='xy', scale_units='xy', scale=key_scale)
        ax.annotate(key_label, (x_key, y_key-0.1), fontsize=12)
        l,r,b,t = pylab.axis()
        dx, dy = r-l, t-b
        ax.set_title(title)
#        pylab.axis([l-0.1*dx, r+0.1*dx, b-0.1*dy, t+0.1*dy])
#        pylab.show()

    def plot_blank(self, ax, c='k'):
        ylim = ax.get_ylim()
        ax.plot((self.params['t_before_blank'], self.params['t_before_blank']), (ylim[0], ylim[1]), ls='--', c=c, lw=2)
        ax.plot((self.params['t_before_blank'] + self.params['t_blank'], self.params['t_before_blank'] + self.params['t_blank']), (ylim[0], ylim[1]), ls='--', c=c, lw=2)
        ax.set_ylim(ylim)

    def plot_blank_on_cmap(self, cax, c='w', txt=''):
        ax = cax.axes

        # plot lines for blank
        ax.axvline(self.params['t_before_blank'] / self.time_binsize, ls='--', color=c, lw=3)
        ax.axvline((self.params['t_before_blank'] + self.params['t_blank']) / self.time_binsize, ls='--', color=c, lw=3)

        # plot lines before stimulus starts
        ax.axvline(0, ls='--', color=c, lw=3)
        ax.axvline(self.params['t_start'] / self.time_binsize, ls='--', color=c, lw=3)

        if txt != '':
            txt_pos_x = (self.params['t_before_blank'] + .15 * self.params['t_blank']) / self.time_binsize
            ylim = ax.get_ylim()
            txt_pos_y = .15 * ylim[1]
            ax.annotate(txt, (txt_pos_x, txt_pos_y), fontsize=14, color='w')

            txt_pos_x = (.15 * self.params['t_start']) / self.time_binsize
            ylim = ax.get_ylim()
            txt_pos_y = .15 * ylim[1]
            ax.annotate(txt, (txt_pos_x, txt_pos_y), fontsize=14, color='w')


def plot_prediction(stim_range=None, params=None, data_fn=None, inh_spikes = None):
    """
    Make use of the PlotPrediction class
    """

    if params== None:
        network_params = simulation_parameters.parameter_storage()  
        params = network_params.params

    if data_fn == None:
        data_fn = params['exc_spiketimes_fn_merged']

    plotter = PlotPrediction(params, data_fn)
    pylab.rcParams['axes.labelsize'] = 14
    pylab.rcParams['axes.titlesize'] = 16
    if plotter.no_spikes:
        return

    plotter.compute_pos_and_v_estimates()

    # fig 1
    # neuronal level
    output_fn_base = '%sprediction' % (params['figures_folder'])

    if params['training_run']:
        t_stim = params['t_training_stim']
    else:
        t_stim = params['t_test_stim']
    if stim_range == None:
        if params['training_run']:
            stim_range = (0, params['n_training_stim'])
        else:
            stim_range = (0, params['n_test_stim'])

    for stim in xrange(stim_range[0], stim_range[1]):
        time_range = (stim * t_stim, (stim + 1) * t_stim)
#        plotter.create_fig()  # create an empty figure
#        pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.93, wspace=0.3, hspace=.2)
#        plotter.n_fig_x = 2
#        plotter.n_fig_y = 2
#        plotter.plot_x_grid_vs_time(1, time_range=time_range)
#        plotter.plot_raster_sorted(time_range, fig_cnt=2, title='Exc cells sorted by x-position', sort_idx=0)
#        plotter.plot_input_spikes_sorted(time_range, fig_cnt=2, sort_idx=0)
#        plotter.plot_vx_grid_vs_time(3, time_range=time_range)              # 3 
#        plotter.plot_raster_sorted(time_range, fig_cnt=4, title='Exc cells sorted by $v_x$', sort_idx=2)
#        plotter.plot_input_spikes_sorted(time_range, fig_cnt=4, sort_idx=2)
#        output_fn = output_fn_base + '_stim%d.png' % stim
#        print 'Saving figure to:', output_fn
#        pylab.savefig(output_fn, dpi=200)
#        plotter.save_data()

        # poplation level, short time-scale
        plotter.n_fig_x = 2
        plotter.n_fig_y = 2
        plotter.create_fig()
        pylab.rcParams['legend.fontsize'] = 12
        pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.93, wspace=0.3, hspace=.3)
        plotter.plot_x_estimates(1, time_range=time_range)
        plotter.plot_xdiff(2, time_range=time_range)
    #        plotter.plot_vy_estimates(2)
        plotter.plot_vx_estimates(3, time_range=time_range)
        plotter.plot_vdiff(4, time_range=time_range)
    #    plotter.plot_y_estimates(5) 
        output_fn = output_fn_base + '_population_readout_%d.png' % stim
        print 'Saving figure to:', output_fn
        pylab.savefig(output_fn, dpi=200)

    #    plotter.plot_nspikes_binned()               # 1
    #    plotter.plot_nspikes_binned_normalized()    # 2
    #    plotter.plot_vx_confidence_binned()         # 3
    #    plotter.plot_vy_confidence_binned()         # 4

        # fig 4
    #    plotter.n_fig_x = 1
    #    plotter.n_fig_y = 2
    #    plotter.create_fig()  # create an empty figure
    #    plotter.plot_network_activity('exc', 1)
    #    plotter.plot_network_activity('inh_spec', 2)
    #    output_fn = output_fn_base + '_4.png'
    #    print 'Saving figure to:', output_fn
    #    pylab.savefig(output_fn, dpi=200)

    #    plotter.n_fig_x = 1
    #    plotter.n_fig_y = 1
    #    plotter.create_fig()
    #    weights = [plotter.nspikes_binned_normalized[i, :].sum() / plotter.n_bins for i in xrange(plotter.n_cells)]
    #    plotter.quiver_plot(weights, fig_cnt=1)
    #    output_fn = output_fn_base + '_quiver.png'
    #    print 'Saving figure to:', output_fn
    #    pylab.savefig(output_fn, dpi=200)

    #    plotter.make_infotextbox()



if __name__ == '__main__':


    stim_range = (0, 10)
    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'

        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)
        plot_prediction(stim_range, params=params)

    else:
        print '\nPlotting the default parameters give in simulation_parameters.py\n'
        plot_prediction(stim_range)

#    pylab.show()
