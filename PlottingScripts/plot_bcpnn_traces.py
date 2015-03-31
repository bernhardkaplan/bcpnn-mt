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
import itertools


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

    def get_cells_that_spiked(self, n_thresh=1):
        """
        Return cell gids that spiked more than once
        """
        assert (self.spikes_loaded), 'Please call self.load_spikes() before calling get_cells_that_spiked()'
        gids = []
        nspikes = []
        for gid in xrange(self.params['n_exc']):
            ns = np.nonzero(gid + 1 == self.spike_data[:, 0])[0].size
            if ns > n_thresh:
                gids.append(gid + 1)
                nspikes.append(ns)
        print 'gids:', gids
        print 'nspikes:', nspikes
        self.load_tuning_prop()
        for i_, gid in enumerate(gids):
            print 'tp[%d, :]' % (gid-1), self.tuning_prop[gid-1, :], 'spiked %d times' % (nspikes[i_])
        return gids, nspikes


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



    def compute_traces(self, gid_pre, gid_post, spike_data_pre, spike_data_post, t_range, dt=0.1):

        st_pre = spike_data_pre[gid_pre]
        st_post = spike_data_post[gid_post]
        print 'DEBUG compute_traces nspikes in st_pre:', len(st_pre)
        print 'DEBUG compute_traces nspikes in st_post:', len(st_post)
        s_pre = BCPNN.convert_spiketrain_to_trace(st_pre, t_range[1], t_min=t_range[0], dt=dt)
        s_post = BCPNN.convert_spiketrain_to_trace(st_post, t_range[1], t_min=t_range[0], dt=dt)
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, self.bcpnn_params, dt=dt)
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


    def get_spikes_for_gid(self, gid, t_range=None):
        assert (self.spikes_loaded), 'Please call self.load_spikes() before calling get_cells_that_spiked()'
        if t_range == None:
            idx = np.nonzero(gid == self.spike_data[:, 0])[0]
            return self.spike_data[idx, 1]
        else: 
            (spikes, gids) = utils.get_spikes_within_interval(self.spike_data, t_range[0], t_range[1], time_axis=1, gid_axis=0)
            idx = np.nonzero(gid == gids)[0]
            return spikes[idx]



if __name__ == '__main__':
    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params

    bcpnn_params = params['bcpnn_params']

    bcpnn_params['tau_p'] = params['t_sim'] * params['ratio_tsim_taup']
#    bcpnn_params['tau_p'] = 500.
    #bcpnn_params['p_i'] = 1e-3
    bcpnn_params['gain'] = 1.
    bcpnn_params['K'] = 1.
    print 'bcpnn_params:', bcpnn_params
    TP = TracePlotter(params)
    dt = 2.
    bcpnn_params['tau_e'] = dt

    assert dt <= bcpnn_params['tau_i'], 'Not allowed due to numerical instability'
    assert dt <= bcpnn_params['tau_j'], 'Not allowed due to numerical instability'
    assert dt <= bcpnn_params['tau_e'], 'Not allowed due to numerical instability'
    assert dt <= bcpnn_params['tau_p'], 'Not allowed due to numerical instability'

    # SELECT CELLS BY TUNING PROPERTIES
    TP.load_tuning_prop()
#    t_range_trace_computation = (0, 50000)
    t_range_trace_computation = (0, params['t_sim']) 

    tp_pre = [0.65, 0.5, 0.0, .0, 0.0]
    tp_post = [0.8, 0.5, 0.0, .0, 0.0]
#    tp_pre = [0.4, 0.5, 0.8, .0]
#    tp_post = [0.6, 0.5, 0.8, .0]
    n_cells_pre = 3
    n_cells_post = 3
    print 'Time range:', t_range_trace_computation, ' is: ', t_range_trace_computation[1] - t_range_trace_computation[0], ' ms'
    gids_pre, dist = utils.get_gids_near_stim_nest(tp_pre, TP.tuning_prop, n=n_cells_pre)
    gids_post, dist = utils.get_gids_near_stim_nest(tp_post, TP.tuning_prop, n=n_cells_post)
#    gid_pre = gids_pre[0]
#    gid_post = gids_post[0]

#    gids_pre = [311]
#    gids_post = [211]

    spike_data = {}
    print 'Tuning properties:'
    for gid_pre in gids_pre:
        print 'GID pre: ', gid_pre, TP.tuning_prop[gid_pre - 1, :]
        spike_data[gid_pre] = []
        spike_data[gid_pre] = TP.get_spikes_for_gid(gid_pre, t_range_trace_computation)
    for gid_post in gids_post:
        print 'GID post: ', gid_post, TP.tuning_prop[gid_post- 1, :]
        spike_data[gid_post] = []
        spike_data[gid_post] = TP.get_spikes_for_gid(gid_post, t_range_trace_computation)

    assert len(spike_data.keys()) > 1, 'ERROR: only one gid found, check gids returned by get_gids_near_stim_nest (tp too close?)'
#    exit(1)

    # SELECT CELLS THAT SPIKED
#    gids, nspikes = TP.get_cells_that_spiked()
#    gid_pre = gids[0]
#    gid_post = gids[1]
#    print 'gid_pre:', gid_pre
#    print 'gid_post:', gid_post

#    spike_data = { gid_pre : [], 
#            gid_post :[] }
#    spike_data[gid_pre] = TP.get_spikes_for_gid(gid_pre, t_range_trace_computation)
#    spike_data[gid_post] = TP.get_spikes_for_gid(gid_post, t_range_trace_computation)
#    print 'spike_data', spike_data


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


    plot_mean_traces = False
    fig = None
    traces = {}
    combinations = itertools.product(gids_pre, gids_post)
#    for gid_pre, gid_post in zip(gids_pre, gids_post):
    n = int((t_range_trace_computation[1] - t_range_trace_computation[0]) / dt)
    n_traces = len(gids_pre) * len(gids_post)
    print 'debug n:', n
    if plot_mean_traces:
        wij_all = np.zeros((n+1, n_traces))
        bias_all = np.zeros((n+1, n_traces))
        pi_all = np.zeros((n+1, n_traces))
        pj_all = np.zeros((n+1, n_traces))
        pij_all = np.zeros((n+1, n_traces))
        ei_all = np.zeros((n+1, n_traces))
        ej_all = np.zeros((n+1, n_traces))
        eij_all = np.zeros((n+1, n_traces))
        zi_all = np.zeros((n+1, n_traces))
        zj_all = np.zeros((n+1, n_traces))
        spre_all = np.zeros((n+1, n_traces))
        spost_all = np.zeros((n+1, n_traces))
    for i_, (gid_pre, gid_post) in enumerate(combinations):
        print 'Trace pair: %d / %d' % (i_ + 1, n_traces)
        bcpnn_traces = TP.compute_traces(gid_pre, gid_post, spike_data, spike_data, t_range_trace_computation, dt)

        if plot_mean_traces:
            [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post] = bcpnn_traces
            wij_all[:, i_] = wij
            bias_all[:, i_] = bias
            pi_all[:, i_] = pi
            pj_all[:, i_] = pj
            pij_all[:, i_] = pij
            eij_all[:, i_] = eij
            ei_all[:, i_] = ei
            ej_all[:, i_] = ej
            zi_all[:, i_] = zi
            zj_all[:, i_] = zj
            spre_all[:, i_] = s_pre
            spost_all[:, i_] = s_post
            traces[(gid_pre, gid_post)] = bcpnn_traces
        fig = TP.plot_trace_with_spikes(bcpnn_traces, bcpnn_params, dt, t_offset=0., output_fn=None, fig=fig, \
                color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None, \
                extra_txt=None, w_title=None)
    output_fn = params['figures_folder'] + 'bcpnn_trace_%d_%d_tauzi%03d_tauzj%03d_taue%03d_taup%05d.png' % (gid_pre, gid_post, \
            bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'])
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)


    if plot_mean_traces:
        print 'Computing mean traces...'
        wij_mean = np.zeros((n+1, 2))
        bias_mean = np.zeros((n+1, 2))
        pi_mean = np.zeros((n+1, 2))
        pj_mean = np.zeros((n+1, 2))
        pij_mean = np.zeros((n+1, 2))
        ei_mean = np.zeros((n+1, 2))
        ej_mean = np.zeros((n+1, 2))
        eij_mean = np.zeros((n+1, 2))
        zi_mean = np.zeros((n+1, 2))
        zj_mean = np.zeros((n+1, 2))
        spre_mean = np.zeros((n+1, 2))
        spost_mean = np.zeros((n+1, 2))
        for i_ in xrange(n + 1):
            if i_ % 1000 == 0:
                print 't:', i_ * dt
            wij_mean[i_, 0] = wij_all[i_, :].mean()
            wij_mean[i_, 1] = wij_all[i_, :].std()
            bias_mean[i_, 0] = bias_all[i_, :].mean()
            bias_mean[i_, 1] = bias_all[i_, :].std()
            pi_mean[i_, 0] = pi_all[i_, :].mean()
            pi_mean[i_, 1] = pi_all[i_, :].std()
            pj_mean[i_, 0] = pj_all[i_, :].mean()
            pj_mean[i_, 1] = pj_all[i_, :].std()
            pij_mean[i_, 0] = pij_all[i_, :].mean()
            pij_mean[i_, 1] = pij_all[i_, :].std()
            ei_mean[i_, 0] = ei_all[i_, :].mean()
            ei_mean[i_, 1] = ei_all[i_, :].std()
            ej_mean[i_, 0] = ej_all[i_, :].mean()
            ej_mean[i_, 1] = ej_all[i_, :].std()
            eij_mean[i_, 0] = eij_all[i_, :].mean()
            eij_mean[i_, 1] = eij_all[i_, :].std()
            zi_mean[i_, 0] = zi_all[i_, :].mean()
            zi_mean[i_, 1] = zi_all[i_, :].std()
            zj_mean[i_, 0] = zj_all[i_, :].mean()
            zj_mean[i_, 1] = zj_all[i_, :].std()
            spre_mean[i_, 0] = spre_all[i_, :].mean()
            spre_mean[i_, 1] = spre_all[i_, :].std()
            spost_mean[i_, 0] = spost_all[i_, :].mean()
            spost_mean[i_, 1] = spost_all[i_, :].std()
        print 'done'
        if not params['debug']:
            w_nest = TP.get_nest_weight(gid_pre, gid_post)
            print 'w_nest:', w_nest
        mean_traces  = [wij_mean[:, 0], bias_mean[:, 0], pi_mean[:, 0], pj_mean[:, 0], pij_mean[:, 0], ei_mean[:, 0], ej_mean[:, 0], eij_mean[:, 0], \
                zi_mean[:, 0], zj_mean[:, 0], spre_mean[:, 0], spost_mean[:, 0]]
        title = 'Mean weight'
        extra_txt = 'Average traces over %d cell pairs' % (n_traces)
        TP.plot_trace_with_spikes(mean_traces, bcpnn_params, dt, t_offset=0., output_fn=None, fig=None, \
                color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None, \
                extra_txt=extra_txt, w_title=title)
        output_fn = params['figures_folder'] + 'bcpnn_trace_mean_xpre%.2f_xpost%.2f_tauzi%03d_tauzj%03d_taue%03d_taup%05d.png' % ( tp_pre[0], tp_post[0], \
                bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'])
        print 'Saving figure to:', output_fn
        pylab.savefig(output_fn, dpi=200)

    pylab.show()

