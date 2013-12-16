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
        self.exp_traces_computed = False

        rcParams = { 'axes.labelsize' : 20,
                    'axes.titlesize'  : 20,
                    'label.fontsize': 20,
                    'xtick.labelsize' : 18, 
                    'ytick.labelsize' : 18, 
                    'legend.fontsize': 16, 
                    'lines.markeredgewidth' : 0}
        pylab.rcParams.update(rcParams)

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
            

    def plot_raster_grouped(self, fig_cnt=1):

        dt_filter = 1.
        tau_filter = 30.
        n_cells_per_pop = self.params['n_cells_to_record_per_location']
        n_pop = len(self.params['locations_to_record'])
        spike_fn = self.params['exc_spiketimes_fn_merged'] + '.ras'
        all_spikes = np.loadtxt(spike_fn)
        all_traces, mean_trace, std_trace = self.get_exponentially_filtered_spiketrain_traces(dt_filter=dt_filter, tau_filter=tau_filter)
        n_data = all_traces[:, 0].size
#        fig = pylab.figure()
#        ax1 = fig.add_subplot(131)
#        ax2 = fig.add_subplot(132)
#        ax3 = fig.add_subplot(133)
#        fig = pylab.figure()
#        ax = fig.add_subplot(111)
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)

        idx = 0
        max_filter = 0.
        for i_ in xrange(n_pop):
            pop_info_fn = self.params['parameters_folder'] + 'pop_%d.info' % i_
            f = file(pop_info_fn, 'r')
            info = json.load(f)
            gids = info['gids']
            print 'Population %d gids:' % i_, gids
            for j_, gid in enumerate(gids):
                st = utils.get_spiketimes(all_spikes, gid, gid_idx=1, time_idx=0)
#                ax1.plot(st, np.ones(st.size) * idx, 'o', markeredgewidth=0, markersize=3, c=self.colorlist[i_])
            
                max_filter = max(max_filter, max(all_traces[:, idx]))
#                ax2.plot(all_traces[:, -1], all_traces[:, idx], c=self.colorlist[i_])
#                ax3.plot(all_traces[:, -1], all_traces[:, idx]/ 15. + idx, c=self.colorlist[i_])
                idx += 1

            print 'Computing mean voltages for population %d / %d' % (i_, n_pop)
            idx_0 = i_ * n_cells_per_pop
            idx_1 = (i_ + 1) * n_cells_per_pop
            for t_ in xrange(int(n_data)):
                mean_trace[t_, i_] = all_traces[t_, idx_0:idx_1].mean()
                std_trace[t_, i_] = all_traces[t_, idx_0:idx_1].std()
            ax.plot(all_traces[:, -1], mean_trace[:, i_], c=self.colorlist[i_], lw=3)
#            ax.errorbar(all_traces[:, -1], mean_trace[:, i_], yerr=std_trace[:, i_], c=self.colorlist[i_], lw=1, alpha=0.2)

#        ax.set_xlabel('Time [ms]')
        ax.set_title('Not aligned to stimulus')
#        ax.set_ylabel('Mean spiketrain\nexponentially filtered')
        ax.set_ylabel('Mean filtered spiketrain\naveraged over %d cells' % n_cells_per_pop)
        print 'Maximum filtered spike trains:', max_filter
#        ax1.set_ylim((-.5, idx + .5))
#        ax3.set_ylim((-.5, idx + .5))


    def get_exponentially_filtered_spiketrain_traces(self, dt_filter=1., tau_filter=30.):
        if not self.exp_traces_computed:
            spike_fn = self.params['exc_spiketimes_fn_merged'] + '.ras'
            all_spikes = np.loadtxt(spike_fn)
            n_cells_per_pop = self.params['n_cells_to_record_per_location']
            n_pop = len(self.params['locations_to_record'])
            n_data = self.params['t_sim'] / dt_filter
            all_traces = np.zeros((n_data, n_pop * n_cells_per_pop + 1)) # + 1 for time_axis
            mean_trace = np.zeros((n_data, n_pop))
            std_trace = np.zeros((n_data, n_pop))
            idx = 0
            for i_ in xrange(n_pop):
                pop_info_fn = self.params['parameters_folder'] + 'pop_%d.info' % i_
                f = file(pop_info_fn, 'r')
                info = json.load(f)
                gids = info['gids']
                print 'Population %d gids:' % i_, gids
                for j_, gid in enumerate(gids):
                    st = utils.get_spiketimes(all_spikes, gid, gid_idx=1, time_idx=0)
                    t_vec, filter_spike_train = utils.filter_spike_train(st, dt=dt_filter, tau=tau_filter, t_max=self.params['t_sim'])
                    all_traces[:, idx] = filter_spike_train
                    idx += 1
                print 'Computing mean voltages for population %d / %d' % (i_, n_pop)
                idx_0 = i_ * n_cells_per_pop
                idx_1 = (i_ + 1) * n_cells_per_pop
                for t_ in xrange(int(n_data)):
                    mean_trace[t_, i_] = all_traces[t_, idx_0:idx_1].mean()
                    std_trace[t_, i_] = all_traces[t_, idx_0:idx_1].std()
            all_traces[:, -1] = t_vec

            self.all_traces = all_traces
            self.mean_trace = mean_trace
            self.std_trace = std_trace
            self.exp_traces_computed = True
            return self.all_traces, self.mean_trace, self.std_trace

        else:
            return self.all_traces, self.mean_trace, self.std_trace


    def plot_aligned_exponential_spiketrains(self, fig_cnt=1):

        dt_filter = 1.
        tau_filter = 30.
        all_traces, mean_trace, std_trace = self.get_exponentially_filtered_spiketrain_traces(dt_filter=dt_filter, tau_filter=tau_filter)
        n_data = all_traces[:, 0].size
        tp = np.loadtxt(self.params['tuning_prop_means_fn'])
        n_cells_per_pop = self.params['n_cells_to_record_per_location']
        n_pop = len(self.params['locations_to_record'])

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)

        idx = 0
        for i_ in xrange(n_pop):
            pop_info_fn = self.params['parameters_folder'] + 'pop_%d.info' % i_
            f = file(pop_info_fn, 'r')
            info = json.load(f)
            gids = info['gids']
            print 'Population %d gids:' % i_, gids

            # compute the mean x-position of all cells in this subpopulation
            x_mean = tp[gids, 0].mean()
            x_std = tp[gids, 0].std()
            print 'Population %d x_mean = %.3f +- %.3f' % (i_, x_mean, x_std)

            # compute time when stimulus is at x_mean
            t_arrive = utils.torus_distance(x_mean, self.params['motion_params'][0]) / self.params['motion_params'][2]
            shift_ = int(t_arrive * 1000./ dt_filter) + n_data * .5
            print 't_arrive:', t_arrive, shift_, self.colorlist[i_]
            shifted_trace = np.r_[mean_trace[shift_:, i_], mean_trace[:shift_, i_]]
            ax.plot(all_traces[:, -1], shifted_trace, c=self.colorlist[i_], lw=3, label='$\\bar{x}_%d=%.2f$' % (i_, x_mean))

        pylab.legend()

        xticks = np.arange(-self.params['t_sim'] * .5, self.params['t_sim'] * .5, 200)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Time [ms]')
        ax.set_title('Response aligned to stimulus arrival at $\\bar{x}_i$')
        ax.set_ylabel('Mean filtered spiketrain\naveraged over %d cells' % n_cells_per_pop)



    def plot_selected_cells_in_tuning_space(self, fig_cnt=1):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('Tuning properties of selected position\n or: Electrode position in tuning space')
        ax.set_xlabel('Receptive field position $x$')
        ax.set_ylabel('Preferred speed $v_x$')

        n_cells_per_pop = self.params['n_cells_to_record_per_location']
        n_pop = len(self.params['locations_to_record'])
        tp = np.loadtxt(self.params['tuning_prop_means_fn'])
        idx = 0
        for i_ in xrange(n_pop):
            pop_info_fn = self.params['parameters_folder'] + 'pop_%d.info' % i_
            f = file(pop_info_fn, 'r')
            info = json.load(f)
            gids = info['gids']
            print 'Population %d gids:' % i_, gids
            for j_, gid in enumerate(gids):
                ax.plot(tp[gid, 0], tp[gid, 2], 'o', c=self.colorlist[i_], markersize=10)

        xlim = ax.get_xlim()
        ax.plot((xlim[0], xlim[1]), (.5, .5), ls='--', c='k')



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
#    P.create_fig()
#    P.plot_selected_voltages()


#    P.create_fig()
    P.create_fig()
    P.plot_selected_cells_in_tuning_space(fig_cnt=1)
    output_fn = output_fn_base + 'tuning_prop.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=300)

    P.n_fig_x = 1
    P.n_fig_y = 2
    pylab.subplots_adjust(hspace=0.25)
    P.create_fig()
    P.plot_raster_grouped(fig_cnt=1)
    P.plot_aligned_exponential_spiketrains(fig_cnt=2)

    output_fn = output_fn_base + 'mean_exp_traces.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=300)

    pylab.show()

