import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import utils
import re
import numpy as np
import pylab
import BCPNN
import simulation_parameters
import FigureCreator
import json

class TracePlotter(object):

    def __init__(self, params, bcpnn_params=None):
        self.params = params
        self.spikes_loaded = False
        self.weights_loaded = False
        self.tuning_prop_loaded = False
        self.load_spikes()
        if bcpnn_params == None:
            bcpnn_params = self.params['bcpnn_params']
        self.bcpnn_params = bcpnn_params
        self.bcpnn_traces = [] # stores all computed traces as list [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]


    def get_tuning_prop(self, gid):
        if not self.tuning_prop_loaded:
            self.load_tuning_prop()
        # assume GIDS are NEST gids, i.e. 1 - aligned
        idx = gid - 1 
        tp = self.tuning_prop[idx, :]
        return tp


    def load_spikes(self):
        fn_pre = self.params['exc_spiketimes_fn_merged']
        if not os.path.exists(fn_pre):
            utils.merge_and_sort_files(self.params['exc_spiketimes_fn_base'], self.params['exc_spiketimes_fn_merged'])
        print 'TracePlotter loads:', fn_pre
        self.spike_data = np.loadtxt(fn_pre)
        self.post_spikes = self.spike_data 
        self.spikes_loaded = True

    
    def load_weights(self):
        fn = self.params['merged_conn_list_ee']
        if not os.path.exists(fn):
            utils.merge_connection_files(self.params, conn_type='ee')
        self.weights = np.loadtxt(fn)
        self.weights_loaded = True


    def load_tuning_prop(self):
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_exc_fn'])
        self.tuning_prop_loaded = True

#    def get_cells_that_spiked(self, n_thresh=1):
#        """
#        Return cell gids that spiked more than once
#        """
#        assert (self.spikes_loaded), 'Please call self.load_spikes() before calling get_cells_that_spiked()'
#        gids = []
#        nspikes = []
#        for gid in xrange(self.params['n_exc']):
#            ns = np.nonzero(gid + 1 == self.spike_data[:, 0])[0].size
#            if ns > n_thresh:
#                gids.append(gid + 1)
#                nspikes.append(ns)
#        print 'gids:', gids
#        print 'nspikes:', nspikes
#        return gids, nspikes


    def get_spike_data_for_stimulus(self, stim_id):
        """
        Return a dictionary:
         [gid] = np.array([spike_times ... ])
        """
        assert (self.spikes_loaded), 'Please call self.load_spikes() before calling get_cells_that_spiked()'
        self.stim_duration = np.loadtxt(self.params['stim_durations_fn'])
        if self.params['n_stim'] > 1:
            t_range = (self.stim_duration[:stim_id].sum(), self.stim_duration[:stim_id+1].sum())
        else:
            t_range = (0, self.stim_duration)
        (spikes, gids) = utils.get_spikes_within_interval(self.spike_data, t_range[0], t_range[1], time_axis=1, gid_axis=0)
#        print 'debug spikes', spikes
        gids_uni = np.unique(gids)
        spike_data = { gid : [] for gid in np.unique(gids_uni)}
        for gid in gids_uni:
            idx = np.nonzero(gid == gids)[0]
            spike_data[gid] = spikes[idx]
        return spike_data



    def compute_traces(self, gid_pre, gid_post, spike_data_pre, spike_data_post, t_range):

        st_pre = spike_data_pre[gid_pre]
        st_post = spike_data_post[gid_post]
        print 'DEBUG compute_traces nspikes in st_pre:', len(st_pre)
        print 'DEBUG compute_traces nspikes in st_post:', len(st_post)
        s_pre = BCPNN.convert_spiketrain_to_trace(st_pre, t_range[1], t_min=t_range[0])
        s_post = BCPNN.convert_spiketrain_to_trace(st_post, t_range[1], t_min=t_range[0])
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, self.bcpnn_params)
        self.bcpnn_traces.append([wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post])
        return [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]


    def plot_trace_with_spikes(self, bcpnn_traces, bcpnn_params, dt, t_offset=0., output_fn=None, fig=None, \
            color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None, \
            extra_txt=None, w_title=None):
        # unpack the bcpnn_traces
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, pre_trace, post_trace = bcpnn_traces
        t_axis = dt * np.arange(zi.size) + t_offset
        plots = []
        pylab.rcParams.update({'figure.subplot.hspace': 0.50})
        if fig == None:
            fig = pylab.figure(figsize=FigureCreator.get_fig_size(1200, portrait=False))
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)
        else:
            ax1, ax2, ax3, ax4, ax5, ax6 = fig.get_axes()
        linewidth = 1
        self.title_fontsize = 24
        ax1.set_title('$\\tau_{z_i} = %d$ ms, $\\tau_{z_j} = %d$ ms' % \
                (bcpnn_params['tau_i'], bcpnn_params['tau_j']), fontsize=self.title_fontsize)
        ax1.plot(t_axis, pre_trace, c=color_pre, lw=linewidth, ls=':')
        ax1.plot(t_axis, post_trace, c=color_post, lw=linewidth, ls=':')
        p1, = ax1.plot(t_axis, zi, c=color_pre, label='$z_i$', lw=linewidth)
        p2, = ax1.plot(t_axis, zj, c=color_post, label='$z_j$', lw=linewidth)
        plots += [p1, p2]
        labels_z = ['$z_i$', '$z_j$']
        ax1.legend(plots, labels_z, loc='upper left')
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('z-traces')

        plots = []
        p1, = ax5.plot(t_axis, pi, c=color_pre, lw=linewidth)
        p2, = ax5.plot(t_axis, pj, c=color_post, lw=linewidth)
        p3, = ax5.plot(t_axis, pij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$p_i$', '$p_j$', '$p_{ij}$']
        ax5.set_title('$\\tau_{p} = %d$ ms' % \
                (bcpnn_params['tau_p']), fontsize=self.title_fontsize)
        ax5.legend(plots, labels_p, loc='upper left')
        ax5.set_xlabel('Time [ms]')
        ax5.set_ylabel('p-traces')

        plots = []
        p1, = ax3.plot(t_axis, ei, c=color_pre, lw=linewidth)
        p2, = ax3.plot(t_axis, ej, c=color_post, lw=linewidth)
        p3, = ax3.plot(t_axis, eij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax3.set_title('$\\tau_{e} = %d$ ms' % \
                (bcpnn_params['tau_e']), fontsize=self.title_fontsize)
        ax3.legend(plots, labels_p, loc='upper left')
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')

        plots = []
        p1, = ax4.plot(t_axis, wij, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w, loc='upper left')
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')
        if w_title != None:
            ax4.set_title(w_title)

        plots = []
        p1, = ax6.plot(t_axis, bias, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_, loc='upper left')
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Bias')

        if K_vec != None:
#            print 'debug t_axis K_vec sizes', len(t_axis), len(K_vec)
            p1, = ax2.plot(t_axis, K_vec, c='k', lw=linewidth)
            ax2.set_ylabel('Reward')
            ylim = (np.min(K_vec), np.max(K_vec))
#            ylim = ax2.get_ylim()
            y0 = ylim[0] - (ylim[1] - ylim[0]) * 0.05
            y1 = ylim[1] + (ylim[1] - ylim[0]) * 0.05
            ax2.set_ylim((y0, y1))
        if extra_txt != None:
            ax2.set_title(extra_txt)

        if output_fn != None:
            print 'Saving traces to:', output_fn
            pylab.savefig(output_fn)

        return fig


    def get_nest_weight(self, gid_pre, gid_post):

        if not self.weights_loaded:
            self.load_weights()
    
        # idx  == column in weights 
        w = utils.extract_weight_from_connection_list(self.weights, gid_pre, gid_post, idx=2)
        return w




    def find_cell_spiking_within_range(self, t_range):
        assert (self.spikes_loaded), 'Please call self.load_spikes() before calling get_cells_that_spiked()'
        (spikes, gids) = utils.get_spikes_within_interval(self.spike_data, t_range[0], t_range[1], time_axis=1, gid_axis=0)

        gids_uni = np.unique(gids)
        spike_data = { gid : [] for gid in np.unique(gids_uni)}
        for gid in gids_uni:
            idx = np.nonzero(gid == gids)[0]
            spike_data[gid] = spikes[idx]
        return spike_data


#    def get_spike_data_for_gids(self, spike_data_dict, gid):


    def get_spikes_for_gid(self, gid, t_range=None):
        assert (self.spikes_loaded), 'Please call self.load_spikes() before calling get_cells_that_spiked()'
        if t_range == None:
            idx = np.nonzero(gid == self.spike_data[:, 0])[0]
            return self.spike_data[idx, 1]
        else: 
            (spikes, gids) = utils.get_spikes_within_interval(self.spike_data, t_range[0], t_range[1], time_axis=1, gid_axis=0)
            idx = np.nonzero(gid == gids)[0]
            return spikes[idx]



    def get_stimulus_time(self, mp, params):
        training_stim = np.loadtxt(params['training_stimuli_fn'])
        gids, dist = utils.get_gids_near_stim(mp, training_stim)
        stim_durations = np.loadtxt(params['stim_durations_fn'])
#        print 'training_stim durations:', stim_durations[gids]
#        print 'time:', stim_durations[:gids[0]].sum(), stim_durations[gids[0]]
        time_range = (stim_durations[:gids[0]].sum(), stim_durations[gids[0]] + stim_durations[:gids[0]].sum())
        return time_range


if __name__ == '__main__':
    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params

    bcpnn_params = params['bcpnn_params']
#    bcpnn_params['tau_p'] = 500.
    bcpnn_params['gain'] = 1.
    TP = TracePlotter(params)
    dt = 0.1

    # SELECT CELLS BY TUNING PROPERTIES
    TP.load_tuning_prop()
    tp_pre = [0.1, 0.5, 1.0, .0]
    tp_post = [0.3, 0.5, -0.1, .0]
    t_range_trace_computation = (0, params['t_sim'])
#    t_range_trace_computation = TP.get_stimulus_time(tp_pre, params)
    print 'Time range:', t_range_trace_computation, ' is: ', t_range_trace_computation[1] - t_range_trace_computation[0], ' ms'

#    exit(1)
    gids_pre, dist = utils.get_gids_near_stim(tp_pre, TP.tuning_prop, n=1)
    gids_post, dist = utils.get_gids_near_stim(tp_post, TP.tuning_prop, n=1)
    print 'gids_pre:', gids_pre
    print 'gids_post:', gids_post
    spike_data = {}
#    for gid_pre in gid_pres
    gid_pre = gids_pre[0]
    gid_post = gids_post[0]
    spike_data = { gid_pre : [], 
            gid_post :[] }
    spike_data[gid_pre] = TP.get_spikes_for_gid(gid_pre, t_range_trace_computation)
    spike_data[gid_post] = TP.get_spikes_for_gid(gid_post, t_range_trace_computation)
    print 'spike_data', spike_data
    bcpnn_traces = TP.compute_traces(gid_pre, gid_post, spike_data, spike_data, t_range_trace_computation)

    # SELECT CELLS BY GID
#    gid_pre = 250 + 1
#    gid_post = 648 + 1
#    spike_data = { gid_pre : [], 
#            gid_post :[] }
#    spike_data[gid_pre] = TP.get_spikes_for_gid(gid_pre, t_range_trace_computation)
#    spike_data[gid_post] = TP.get_spikes_for_gid(gid_post, t_range_trace_computation)
#    bcpnn_traces = TP.compute_traces(gid_pre, gid_post, spike_data, spike_data, t_range_trace_computation)


    # SELECT CELLS BASED ON TIME INTERVAL
#    spike_data = TP.find_cell_spiking_within_range((500, 700))
#    gid_pre = spike_data.keys()[0]
#    gid_post = spike_data.keys()[1]
#    print 'Tuning prop pre %d:' % gid_pre, TP.get_tuning_prop(gid_pre)
#    print 'Tuning prop post %d:' % gid_post, TP.get_tuning_prop(gid_post)
#    bcpnn_traces = TP.compute_traces(gid_pre, gid_post, spike_data, spike_data, t_range_trace_computation)

    # SELECT CELLS THAT SPIKED
#    gids, nspikes = TP.get_cells_that_spiked()
#    gid_pre = gids[1]
#    gid_post = gids[1]

    # SELECT CELLS BASED ON STIMULUS ID 
#    stim_id_pre = 0
#    stim_id_post = 0
#    spike_data_pre = TP.get_spike_data_for_stimulus(stim_id_pre)
#    spike_data_post = TP.get_spike_data_for_stimulus(stim_id_post)
#    print 'spike_data_pre:', spike_data_pre
#    print 'spike_data_post:', spike_data_post
#    gids_pre = spike_data_pre.keys()
#    gids_post = spike_data_post.keys()
#    gid_pre = gids_pre[0]
#    gid_post = gids_post[2]
#    print 'Plotting traces for gid pre %d\t gid post: %d' % (gid_pre, gid_post)
#    bcpnn_traces = TP.compute_traces(gid_pre, gid_post, spike_data_pre, spike_data_post, t_range_trace_computation)



    TP.plot_trace_with_spikes(bcpnn_traces, bcpnn_params, dt, t_offset=0., output_fn=None, fig=None, \
            color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None, \
            extra_txt=None, w_title=None)

    if not params['debug']:
        w_nest = TP.get_nest_weight(gid_pre, gid_post)
        print 'w_nest:', w_nest

    output_fn = params['figures_folder'] + 'bcpnn_trace_%d_%d_tauzi%03d_tauzj%03d_taue%03d_taup%05d.png' % (gid_pre, gid_post, \
            bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'])
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    pylab.show()

