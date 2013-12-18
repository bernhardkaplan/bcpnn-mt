# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import pylab
import json
import utils
import numpy as np
import json


class PlotAnticipation(object):

    def __init__(self, params):
        self.params = params
        self.n_fig_x = 1
        self.n_fig_y = 1
#        self.fig_size = (11.69, 8.27) #A4
        self.fig_size = (12, 8)
        self.colorlist = ['k', 'r', 'b', 'g', 'm', 'y', 'c']

        rcParams = { 'axes.labelsize' : 20,
                    'axes.titlesize'  : 20,
                    'label.fontsize': 20,
                    'xtick.labelsize' : 18, 
                    'ytick.labelsize' : 18, 
                    'legend.fontsize': 16, 
                    'lines.markeredgewidth' : 0}
        pylab.rcParams.update(rcParams)
        self.tp = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.tau_filter = 100.
        self.dt_filter = 1.
        self.normalized_traces = None

    def filter_and_normalize_spikes(self):
        """
        Filter all spike trains with exponentials and for each time step divide by the sum.
        """
        print 'Filtering and normalizing all spike traces ... '
        spike_fn = self.params['exc_spiketimes_fn_merged'] + '.ras'
        all_spikes = np.loadtxt(spike_fn)
        gids = [range(self.params['n_exc'])]
        all_traces, mean_trace, std_trace = self.get_exponentially_filtered_spiketrain_traces(gids, dt_filter=self.dt_filter, tau_filter=self.tau_filter)
        self.normalized_traces = np.zeros(all_traces.shape)
        n_data = np.int(self.params['t_sim'] / self.dt_filter)
        for t_ in xrange(n_data):
            s = all_traces[t_, :].sum()
            if s > 0:
                self.normalized_traces[t_, :] = all_traces[t_, :] / s


    def create_fig(self):
        print "plotting ...."
        self.fig = pylab.figure(figsize=self.fig_size)


    def plot_selected_voltages(self, fig_cnt=1):
        """
        This function plots the membrane potentials of selected subpopulations.
        Data file:
            volt_fn = self.params['exc_volt_fn_base'] + '_pop%d.v' % (i_)
        During the simulation membrane potentials from different subparts
        have been recorded.
        The information for subpopulation i_ is found in file:
            pop_info_fn = self.params['parameters_folder'] + 'pop_%d.info' % i_
        """
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)

        n_cells_per_pop = self.params['n_cells_to_record_per_location']
        n_pop = len(self.params['locations_to_record'])
        n_data = self.params['t_sim'] / self.params['dt_sim'] 
        all_data = np.zeros((n_data, n_pop * n_cells_per_pop + 1)) # + 1 for time_axis
        mean_volt = np.zeros((n_data, n_pop))
        std_volt = np.zeros((n_data, n_pop))

        # all volt_fn contain the same data ... --> PyNN-0.7.5 bug
        volt_fn = self.params['exc_volt_fn_base'] + '_pop0.v'
        volt_data = np.loadtxt(volt_fn)
        for i_ in xrange(n_pop):
            pop_info_fn = self.params['parameters_folder'] + 'pop_%d.info' % i_
            f = file(pop_info_fn, 'r')
            info = json.load(f)
            gids = info['gids']
            print 'Population %d gids:' % i_, gids
            for j_, gid in enumerate(gids):
                time_axis, d = utils.extract_trace(volt_data, gid)
                all_data[:, i_ * n_cells_per_pop + j_] = d
                ax.plot(time_axis, d, c=self.colorlist[i_], alpha=0.2)
            print 'Computing mean voltages for population %d / %d' % (i_, n_pop)
            idx_0 = i_ * n_cells_per_pop
            idx_1 = (i_ + 1) * n_cells_per_pop
            for t_ in xrange(int(n_data)):
                mean_volt[t_, i_] = all_data[t_, idx_0:idx_1].mean()
                std_volt[t_, i_] = all_data[t_, idx_0:idx_1].std()
            ax.plot(time_axis, mean_volt[:, i_], c=self.colorlist[i_])
            ax.errorbar(time_axis, mean_volt[:, i_], yerr=std_volt[:, i_], c=self.colorlist[i_], lw=1, alpha=0.2)
        all_data[:, -1] = time_axis
            

    def plot_exponential_spiketrains(self, fig_cnt=1, gids=None, normalize=False):

        if gids == None:
            gids = self.load_selected_cells()
        if normalize == True:
            assert (self.normalized_traces != None), 'Call filter_and_normalize_spikes before!'
        n_pop = len(gids)
        n_cells_per_pop = len(gids[0])
        spike_fn = self.params['exc_spiketimes_fn_merged'] + '.ras'
        all_spikes = np.loadtxt(spike_fn)
        all_traces, mean_trace, std_trace = self.get_exponentially_filtered_spiketrain_traces(gids, dt_filter=self.dt_filter, tau_filter=self.tau_filter)
        confidence_trace = np.zeros(mean_trace.shape)
        n_data = all_traces[:, 0].size
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)

        if normalize:
            for i_ in xrange(n_pop):
                print 'Computing mean trace for population %d / %d' % (i_, n_pop)
                for t_ in xrange(int(n_data)):
                    confidence_trace[t_, i_] = self.normalized_traces[t_, gids[i_]].sum()
                ax.plot(all_traces[:, -1], confidence_trace[:, i_], c=self.colorlist[i_], lw=3)
        else:
            for i_ in xrange(n_pop):
                ax.plot(all_traces[:, -1], mean_trace[:, i_], c=self.colorlist[i_], lw=3)

        confidence_trace[:, -1] = all_traces[:, -1]
        data_fn = self.params['data_folder'] + 'not_aligned_mean_trace.dat'
        print 'Saving data to:', data_fn
        np.savetxt(data_fn, mean_trace)

        data_fn = self.params['data_folder'] + 'not_aligned_confidence_trace.dat'
        print 'Saving data to:', data_fn
        np.savetxt(data_fn, confidence_trace)
#        ax.set_xlabel('Time [ms]')
        ax.set_title('Not aligned to stimulus')
        ax.set_ylabel('Mean filtered spiketrain\naveraged over %d cells' % n_cells_per_pop)


    def get_exponentially_filtered_spiketrain_traces(self, gids, dt_filter=1., tau_filter=30.):
        spike_fn = self.params['exc_spiketimes_fn_merged'] + '.ras'
        all_spikes = np.loadtxt(spike_fn)
        n_pop = len(gids)
        n_cells_per_pop = len(gids[0])
        n_data = self.params['t_sim'] / dt_filter
        all_traces = np.zeros((n_data, n_pop * n_cells_per_pop + 1)) # + 1 for time_axis
        mean_trace = np.zeros((n_data, n_pop + 1)) # + 1 for time_axis
        std_trace = np.zeros((n_data, n_pop))
        idx = 0
        print 'DEBUG Gids:', gids
        for i_ in xrange(n_pop):
            print 'Population %d gids:' % i_, gids[i_]
            for j_, gid in enumerate(gids[i_]):
                st = utils.get_spiketimes(all_spikes, gid, gid_idx=1, time_idx=0)
                t_vec, filter_spike_train = utils.filter_spike_train(st, dt=dt_filter, tau=tau_filter, t_max=self.params['t_sim'])
                all_traces[:, idx] = filter_spike_train
                idx += 1
            print 'Computing mean trace for population %d / %d' % (i_, n_pop)
            idx_0 = i_ * n_cells_per_pop
            idx_1 = (i_ + 1) * n_cells_per_pop
            for t_ in xrange(int(n_data)):
                mean_trace[t_, i_] = all_traces[t_, idx_0:idx_1].mean()
                std_trace[t_, i_] = all_traces[t_, idx_0:idx_1].std()
        all_traces[:, -1] = t_vec
        mean_trace[:, -1] = t_vec

        self.all_traces = all_traces
        self.mean_trace = mean_trace
        self.std_trace = std_trace
        return self.all_traces, self.mean_trace, self.std_trace


    def plot_aligned_exponential_spiketrains(self, fig_cnt=1, gids=None, normalize=False):

        if gids == None:
            gids = self.load_selected_cells()
        all_traces, mean_trace, std_trace = self.get_exponentially_filtered_spiketrain_traces(gids, dt_filter=self.dt_filter, tau_filter=self.tau_filter)
        confidence_trace = np.zeros(mean_trace.shape)
        aligned_traces = np.zeros(mean_trace.shape)
        n_data = all_traces[:, 0].size
        n_pop = len(gids)
        n_cells_per_pop = len(gids[0])
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        for i_ in xrange(n_pop):
#            print 'Population %d gids:' % i_, gids[i_]
            # compute the mean x-position of all cells in this subpopulation
            x_mean = self.tp[gids[i_], 0].mean()
            x_std = self.tp[gids[i_], 0].std()
            print 'Population %d x_mean = %.3f +- %.3f' % (i_, x_mean, x_std)

            # compute time when stimulus is at x_mean
            t_arrive = utils.torus_distance(x_mean, self.params['motion_params'][0]) / self.params['motion_params'][2]
            shift_ = int(t_arrive * 1000./ self.dt_filter) + n_data * .5
            print 't_arrive:', t_arrive, shift_, self.colorlist[i_]
            shifted_trace = np.r_[mean_trace[shift_:, i_], mean_trace[:shift_, i_]]
            aligned_traces[:, i_] = shifted_trace
            if normalize:
                print 't_arrive:', t_arrive, shift_, self.colorlist[i_]
                for t_ in xrange(int(n_data)):
                    confidence_trace[t_, i_] = self.normalized_traces[t_, gids[i_]].sum()
                confidence_trace[:, i_] = np.r_[confidence_trace[shift_:, i_], confidence_trace[:shift_, i_]]
                ax.plot(all_traces[:, -1], confidence_trace[:, i_], c=self.colorlist[i_], lw=3, label='$\\bar{x}_%d=%.2f$' % (i_, x_mean))
            else:
                ax.plot(all_traces[:, -1], shifted_trace, c=self.colorlist[i_], lw=3, label='$\\bar{x}_%d=%.2f$' % (i_, x_mean))

        aligned_traces[:, -1] = all_traces[:, -1]
        confidence_trace[:, -1] = all_traces[:, -1]

        data_fn = self.params['data_folder'] + 'aligned_mean_trace.dat'
        print 'Saving data to:', data_fn
        np.savetxt(data_fn, aligned_traces)

        data_fn = self.params['data_folder'] + 'aligned_confidence_trace.dat'
        print 'Saving data to:', data_fn
        np.savetxt(data_fn, confidence_trace)
#        data_fn = self.params['data_folder'] + 'aligned_std_trace.dat'
#        np.savetxt(data_fn, std_trace)

        pylab.legend()

        xticks = np.arange(-self.params['t_sim'] * .5, self.params['t_sim'] * .5, 200)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Time [ms]')
        ax.set_title('Response aligned to stimulus arrival at $\\bar{x}_i$')
        ax.set_ylabel('Mean filtered spiketrain\naveraged over %d cells' % n_cells_per_pop)

    def load_selected_cells(self):
        """
        Returns a list of gids
        """
        n_cells_per_pop = self.params['n_cells_to_record_per_location']
        n_pop = len(self.params['locations_to_record'])
        gids = [[] for i in xrange(n_pop)]
        for i_ in xrange(n_pop):
            pop_info_fn = self.params['parameters_folder'] + 'pop_%d.info' % i_
            f = file(pop_info_fn, 'r')
            info = json.load(f)
            gids_subpop = info['gids']
            print 'Population %d gids:' % i_, gids
            gids[i_] = gids_subpop
        return gids


    def plot_selected_cells_in_tuning_space(self, gids=None, fig_cnt=1, plot_all_cells=False):
        """
        gids -- list of cell populations (list of lists)
        """
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('Tuning properties of selected cells\n or: Electrode position in tuning space')
        ax.set_xlabel('Receptive field position $x$')
        ax.set_ylabel('Preferred speed $v_x$')
        if gids == None:
            gids = self.load_selected_cells()
        if plot_all_cells:
            ax.plot(self.tp[:, 0], self.tp[:, 2], 'o', c='k', alpha=.2, markersize=3)

        x_min, x_max = 10.0, 0.
        vx_min, vx_max = 10.0, 0.
        for i_ in xrange(len(gids)):
            cms_x = 0.
            cms_vx = 0.
            for j_, gid in enumerate(gids[i_]):
                x_min = min(x_min, self.tp[gid, 0])
                x_max = max(x_max, self.tp[gid, 0])
                vx_min = min(vx_min, self.tp[gid, 2])
                vx_max = max(vx_max, self.tp[gid, 2])
                ax.plot(self.tp[gid, 0], self.tp[gid, 2], 'o', c=self.colorlist[i_], markersize=10)
                cms_x += self.tp[gid, 0]
                cms_vx += self.tp[gid, 2]
            cms_x /= len(gids[i_])
            cms_vx /= len(gids[i_])
            ax.plot(cms_x, cms_vx, '*', c=self.colorlist[i_], markersize=20)
        

        ax.set_xlim((x_min * .75, x_max * 1.25))
        ax.set_ylim((vx_min * .75, vx_max * 1.25))
        xlim = ax.get_xlim()
        ax.plot((xlim[0], xlim[1]), (self.params['motion_params'][2], self.params['motion_params'][2]), ls='--', c='k')


    def select_cells(self, x, vx, n_cells, w_pos=1.):
        """
        Select cells to be plotted around situated around (x, vx) in tuning space
        with relative weight w_pos.
        If w_pos < 1. speed is more important in the selection of cells.
        """
        x_diff = utils.torus_distance_array(self.tp[:, 0], x) * w_pos + np.abs(self.tp[:, 2] - vx)
        idx_sorted = np.argsort(x_diff)
        gids = idx_sorted[:n_cells]
        return gids



if __name__ == '__main__':

    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)
    else:
        print '\nPlotting the default parameters give in simulation_parameters.py\n'
        import simulation_parameters
        network_params = simulation_parameters.parameter_storage()
        params = network_params.params

    output_fn_base = params['figures_folder'] + 'snn_anticipation_wsx%.2e_wsv%.2e_' % (params['w_sigma_x'], params['w_sigma_v'])

    P = PlotAnticipation(params)

#    locations_to_record = [  .05 + params['motion_params'][0], \
#                             .15 + params['motion_params'][0], \
#                             .25 + params['motion_params'][0], \
#                             .35 + params['motion_params'][0]]

#    locations_to_record = [  .10 + params['motion_params'][0], \
#                             .20 + params['motion_params'][0], \
#                             .30 + params['motion_params'][0], \
#                             .40 + params['motion_params'][0]]
    locations_to_record = [  .15 + params['motion_params'][0], \
                             .25 + params['motion_params'][0], \
                             .35 + params['motion_params'][0], \
                             .45 + params['motion_params'][0]]


    w_pos = .5
    n_pop = len(locations_to_record)
    n_cells_per_pop = 10
    vx_record = .4 # params['motion_params'][2]
    gids = [[] for i in xrange(n_pop)]
    for i_ in xrange(n_pop):
        gids[i_] = P.select_cells(locations_to_record[i_], vx_record, n_cells_per_pop, w_pos=w_pos)
        gids[i_].sort()
        print 'Pop %d:' % i_, gids[i_]

    P.create_fig()
    P.plot_selected_cells_in_tuning_space(fig_cnt=1, gids=gids, plot_all_cells=True)
#    P.plot_selected_cells_in_tuning_space(fig_cnt=1)

    output_fn = output_fn_base + 'tuning_prop.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=300)

    normalize = False # if True: plot the 'confidence' based on the normalized filtered spike rate
    P.filter_and_normalize_spikes()
    P.n_fig_x = 1
    P.n_fig_y = 2
    pylab.subplots_adjust(hspace=0.25)
    P.create_fig()
    P.plot_exponential_spiketrains(fig_cnt=1, gids=gids, normalize=normalize)
    P.plot_aligned_exponential_spiketrains(fig_cnt=2, gids=gids, normalize=normalize)
    output_fn = output_fn_base + 'mean_exp_traces.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=300)

#    pylab.show()

