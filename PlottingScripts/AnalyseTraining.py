import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
import pylab
import matplotlib
import pylab
import numpy as np
import sys
import os
import utils
import re
import Bcpnn

gid_axis = 0
time_axis = 1

class TrainingAnalyser(object):

    def __init__(self, params, it_max=None):
        self.params = params

        if it_max == None:
            self.it_max = self.params['n_training_stim']
        else:
            self.it_max = it_max

        self.spike_times_loaded = False



    def load_spike_data(self, cell_type):
        """
        cell_type -- 'exc', 'inh_spec', 'inh_unspec'
        """
        if cell_type == 'exc':
            fn = self.params['exc_spiketimes_fn_merged']
        elif cell_type == 'inh_spec':
            fn = self.params['inh_spec_spiketimes_fn_merged']
        elif cell_type == 'inh_unspec':
            fn = self.params['inh_unspec_spiketimes_fn_merged']
        else:
            print 'Invalid cell type provided to TrainingAnalyser.load_spike_data: %s' % cell_type
        if not (os.path.exists(fn)):
            utils.merge_and_sort_files(params['%s_spiketimes_fn_base' % cell_type], params['%s_spiketimes_fn_merged' % cell_type])
        print 'Loading:', fn
        d = np.loadtxt(fn)
#        d = np.zeros(d_.shape)
#        print 'debug', d_.shape, d.shape
#        d[:, 0] = np.array(d_[:, 0], dtype=np.int)
#        d[:, 1] = d_[:, 1]

        return d


    def load_tuning_prop(self):
        print 'TrainingAnalyser.load_tuning_prop ...'
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
#        self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])
#        self.x_grid = np.linspace(0, 1, self.n_bins_x, endpoint=False)
#        self.gid_to_posgrid_mapping = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 0], self.x_grid)


    def get_coactivated_cell_gids(self, stim, spike_data):

        """
        Gets all cell GIDs activated in the training stimulation number stim
        and return their GIDs as a list
        """
        t0 = stim * self.params['t_training_stim']
        t1 = (stim + 1) * self.params['t_training_stim']
        spikes, gids = utils.get_spikes_within_interval(spike_data, t0, t1, time_axis=1, gid_axis=0)
        print 'stim %d (%d - %d [ms]) gids:' % (stim, t0, t1), np.unique(gids)
        mean_rate = np.zeros(np.unique(gids).size)

        for i_, gid in enumerate(np.unique(gids)):
            nspikes = (np.array(gids) == gid).nonzero()[0].size
            f_out = nspikes / (self.params['t_training_stim'] / 1000.)
            mean_rate[i_] = f_out
            print 'GID %d fired %d spikes (= %.2f [Hz])' % (gid, nspikes, f_out)
        print 'During stim %d mean output rate was %.2f +- %.2f' % (stim, mean_rate.mean(), mean_rate.std())


    def plot_traces(self, pre_trace, post_trace, zi, zj, ei, ej, eij, pi, pj, pij, wij, bias, output_fn=None):
        dt = .1
        t_axis = dt * np.arange(zi.size)

        plot_params = {'backend': 'png',
                      'axes.labelsize': 16,
                      'axes.titlesize': 16,
                      'text.fontsize': 16,
                      'xtick.labelsize': 16,
                      'ytick.labelsize': 16,
                      'legend.pad': 0.2,     # empty space around the legend box
                      'legend.fontsize': 14,
                       'lines.markersize': 0,
                       'lines.linewidth': 3,
                      'font.size': 12,
                      'path.simplify': False,
                      'figure.subplot.hspace':.40,
                      'figure.subplot.wspace':.10,
                      'figure.subplot.left':.10,
                      'figure.subplot.bottom':.07, 
                      'figure.subplot.right':.95,
                      'figure.subplot.top':.95}
        #              'figure.figsize': get_fig_size(800)}

        pylab.rcParams.update(plot_params)
        fig = pylab.figure(figsize=utils.get_figsize(1200, portrait=False))
#        ax1 = fig.add_subplot(321)
#        ax2 = fig.add_subplot(322)
#        ax3 = fig.add_subplot(323)
#        ax4 = fig.add_subplot(324)
#        ax5 = fig.add_subplot(325)
#        ax6 = fig.add_subplot(326)

        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512)
        ax3 = fig.add_subplot(513)
        ax4 = fig.add_subplot(514)
        ax5 = fig.add_subplot(515)
        self.title_fontsize = 18

        plots = []
        ax1.set_title('$\\tau_{z_i} = %d$ ms, $\\tau_{z_j} = %d$ ms' % \
                (self.params['bcpnn_params']['tau_i'], self.params['bcpnn_params']['tau_j']), fontsize=self.title_fontsize)
        ax1.plot(t_axis, pre_trace, c='b', alpha=.2, lw=2)
        ax1.plot(t_axis, post_trace + 1, c='g', alpha=.2, lw=2)
        p1, = ax1.plot(t_axis, zi, c='b', label='$z_i$', lw=2)
        p2, = ax1.plot(t_axis, zj, c='g', label='$z_j$', lw=2)
        plots += [p1, p2]
        labels_z = ['$z_i$', '$z_j$']
        ax1.legend(plots, labels_z)
#        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('z-traces')
        ax1.set_xlim((0, t_axis[-1]))

        plots = []
        p1, = ax2.plot(t_axis, ei, c='b', lw=2)
        p2, = ax2.plot(t_axis, ej, c='g', lw=2)
        p3, = ax2.plot(t_axis, eij, c='r', lw=2)
        plots += [p1, p2, p3]
        labels_e = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax2.set_title('$\\tau_{e} = %d$ ms' % \
                (self.params['bcpnn_params']['tau_e']), fontsize=self.title_fontsize)
        ax2.legend(plots, labels_e)
#        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('e-traces')
        ax2.set_xlim((0, t_axis[-1]))

        plots = []
        p1, = ax3.plot(t_axis, pi, c='b', lw=2)
        p2, = ax3.plot(t_axis, pj, c='g', lw=2)
        p3, = ax3.plot(t_axis, pij, c='r', lw=2)
        plots += [p1, p2, p3]
        labels_p = ['$p_i$', '$p_j$', '$p_{ij}$']
        ax3.set_title('$\\tau_{p} = %d$ ms' % \
                (self.params['bcpnn_params']['tau_p']), fontsize=self.title_fontsize)
        ax3.legend(plots, labels_p)
#        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('p-traces')
        ax3.set_xlim((0, t_axis[-1]))

        plots = []
        p1, = ax4.plot(t_axis, wij, c='b', lw=2)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w)
#        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')
        ax4.set_xlim((0, t_axis[-1]))

        plots = []
        p1, = ax5.plot(t_axis, bias, c='b', lw=2)
        plots += [p1]
        labels_ = ['bias']
        ax5.legend(plots, labels_)
        ax5.set_xlabel('Time [ms]')
        ax5.set_ylabel('Bias')
        ax5.set_xlim((0, t_axis[-1]))

#        ax5.set_yticks([])
#        ax5.set_xticks([])

#        ax5.annotate('$v_{stim} = %.2f, v_{0}=%.2f, v_{1}=%.2f$\ndx: %.2f\
#                \nWeight max: %.3e\nWeight end: %.3e\nWeight avg: %.3e\nt(w_max): %.1f [ms]' % \
#                (self.v_stim, self.tp_s[0][2], self.tp_s[1][2], self.dx, self.w_max, self.w_end, self.w_avg, \
#                self.t_max * dt), (.1, .1), fontsize=20)

#        ax5.set_xticks([])
#        output_fn = self.params['figures_folder'] + 'traces_tauzi_%04d_tauzj%04d_taue%d_taup%d_dx%.2e_dv%.2e_vstim%.1e.png' % \
#                (self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j'], self.bcpnn_params['tau_e'], self.bcpnn_params['tau_p'], self.dx, self.dv, self.v_stim)
#        output_fn = self.params['figures_folder'] + 'traces_dx%.2e_dv%.2e_vstim%.1e_tauzi_%04d_tauzj%04d_taue%d_taup%d.png' % \
#                (dx, dv, v_stim, params['bcpnn_params']['tau_i'], self.params['bcpnn_params']['tau_j'], self.params['bcpnn_params']['tau_e'], self.params['bcpnn_params']['tau_p'])
        if output_fn != None:
            print 'Saving traces to:', output_fn
            pylab.savefig(output_fn)


    def plot_bcpnn_traces(self, spike_data, pre_gid, post_gid, syn_params=None, t_max=None):

        spiketimes_pre = utils.get_spiketimes(spike_data, pre_gid, gid_idx=0, time_idx=1)
        spiketimes_post = utils.get_spiketimes(spike_data, post_gid, gid_idx=0, time_idx=1)
        if t_max == None:
            t_max = self.params['t_sim'] + 1
        pre_trace = utils.convert_spiketrain_to_trace(spiketimes_pre, t_max, dt=.1, spike_width=10) # + 1 is to handle spikes in the last time step
        post_trace = utils.convert_spiketrain_to_trace(spiketimes_post, t_max, dt=.1, spike_width=10) # + 1 is to handle spikes in the last time step

        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace, self.params['bcpnn_params'])
        print 'Weight after training:', wij[-1], ' maximum in trace:', np.max(wij)
        output_fn = self.params['figures_folder'] + 'traces_gid%d-%d.png' % (pre_gid, post_gid)
        self.plot_traces(pre_trace, post_trace, zi, zj, ei, ej, eij, pi, pj, pij, wij, bias, output_fn)





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
        import simulation_parameters
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params


    Plotter = TrainingAnalyser(params)#, it_max=1)

    Plotter.load_tuning_prop()
    exc_spike_data = Plotter.load_spike_data('exc')
    inh_spec_spike_data = Plotter.load_spike_data('inh_spec')
    inh_unspec_spike_data = Plotter.load_spike_data('inh_unspec')

    for stim in xrange(params['n_training_stim']):
        Plotter.get_coactivated_cell_gids(stim, exc_spike_data)

    pre_gid = 1
    post_gid = 72
    initial_value = None#1e-6
    Plotter.plot_bcpnn_traces(exc_spike_data, pre_gid, post_gid, syn_params=params['bcpnn_params'])
    pylab.show()
