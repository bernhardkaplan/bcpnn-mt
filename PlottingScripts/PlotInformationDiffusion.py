import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import pylab
import json
import utils
import numpy as np
import simulation_parameters
import plot_rasterplots as plot_rp

class PlotInformationDiffusion(object):
    """
    Plots the (average) connectivity for a number of selected cells
    and combined with their activity the information diffusion as seen from these cells.
    """

    def __init__(self, params):
        self.params = params
        self.n_fig_x = 1
        self.n_fig_y = 1
#        self.fig_size = (11.69, 8.27) #A4
#        self.fig_size = (10, 8)
        self.fig_size = utils.get_figsize(1200, portrait=False)
        self.colorlist = ['k', 'r', 'b', 'g', 'm', 'y', 'c']
        self.markerlist = ['.', 'D', '+', 's', '^', '>', 'd']
        rcParams = { 'axes.labelsize' : 20,
                    'axes.titlesize'  : 22,
                    'label.fontsize': 20,
                    'xtick.labelsize' : 18, 
                    'ytick.labelsize' : 18, 
                    'legend.fontsize': 18, 
                    'figure.subplot.left':.15,
                    'figure.subplot.bottom':.15,
                    'figure.subplot.right':.90,
                    'figure.subplot.top':.90, 
                    'figure.subplot.hspace':.40, 
                    'figure.subplot.wspace':.30, 
                    'lines.markeredgewidth' : 0}
        pylab.rcParams.update(rcParams)
        self.tp = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.conn_lists = {}

        # for plotting within the same ranges
        self.x_min, self.x_max = 10.0, 0.
        self.vx_min, self.vx_max = 10.0, 0.
        self.n_bins = 40
        self.x_range = (-.30, .30)

    def load_spike_data(self):

        self.spiketimes = {}
        self.nspikes = {}
        for cell_type in ['exc', 'inh']:
            self.spiketimes[cell_type], self.nspikes[cell_type] = utils.load_spiketimes(params, cell_type)

    def load_connection_data(self, conn_type='ee'):
        utils.merge_connlists(self.params, verbose=False)
#        for conn_type in ['ee', 'ei', 'ie', 'ii']:
        conn_fn = self.params['merged_conn_list_%s' % conn_type]
        print 'Loading connection data from ', conn_fn
        self.conn_lists[conn_type] = np.loadtxt(conn_fn)


    def create_fig(self):
        print "plotting ...."
        self.fig = pylab.figure(figsize=self.fig_size)



    def plot_vertical_line(self, ax, x_pos, color, ymax=None):
        ls = '--'
        ylim = ax.get_ylim()
        ymin = ylim[0]
        if ymax == None:
            ymax = ylim[1]
        ax.plot((x_pos, x_pos), (ymin, ymax), ls=ls, c=color, lw=3)


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
#            print 'Population %d gids:' % i_, gids
            gids[i_] = gids_subpop
        return gids


    def plot_selected_cells_in_tuning_space(self, gids, ax=None, fig_cnt=1, plot_cms=True, plot_all_cells=False, \
            marker='o', color='k', alpha=1.):
        """
        gids -- list of cell populations (list of lists)
        """
        if ax == None:
            ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title('Tuning properties of selected cells')
        ax.set_xlabel('Receptive field position $x$')
        ax.set_ylabel('Preferred speed $v_x$')
#        if gids == None:
#            gids = self.load_selected_cells()
        if plot_all_cells:
            ax.plot(self.tp[:, 0], self.tp[:, 2], 'o', c='k', alpha=.2, markersize=3)

#        for i_ in xrange(len(gids)):
        cms_x = 0.
        cms_vx = 0.
        for i_, gid in enumerate(gids):
            self.x_min = min(self.x_min, self.tp[gid, 0])
            self.x_max = max(self.x_max, self.tp[gid, 0])
            self.vx_min = min(self.vx_min, self.tp[gid, 2])
            self.vx_max = max(self.vx_max, self.tp[gid, 2])
            ax.plot(self.tp[gid, 0], self.tp[gid, 2], marker, c=color, markersize=10, alpha=alpha)
            cms_x += self.tp[gid, 0]
            cms_vx += self.tp[gid, 2]
        if plot_cms:
            cms_x /= len(gids)
            cms_vx /= len(gids)
            ax.plot(cms_x, cms_vx, '*', c=color, markersize=20, alpha=alpha)
        
        return ax
#        ax.set_xlim((x_min * .75, x_max * 1.25))
#        ax.set_ylim((vx_min * .75, vx_max * 1.25))
#        xlim = ax.get_xlim()
#        ax.plot((xlim[0], xlim[1]), (self.params['motion_params'][2], self.params['motion_params'][2]), ls='--', c='k')
#        


    def select_cells(self, x, vx, n_cells=1, w_pos=1.):
        """
        Select cells to be plotted around situated around (x, vx) in tuning space
        with relative weight w_pos.
        If w_pos < 1. speed is more important in the selection of cells.
        """
        x_diff = utils.torus_distance_array(self.tp[:, 0], x) * w_pos + np.abs(self.tp[:, 2] - vx)
        idx_sorted = np.argsort(x_diff)
        gids = idx_sorted[:n_cells]
        return gids



    def plot_connections(self, gids):
        """
        Plot the tuning properties of cells with gids,
         + their outgoing and incoming connections
         
        """
        tgt_gids = {}
        w_out ={}
        src_gids = {}
        w_in = {}

        fig = pylab.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111)

        for i_, gid in enumerate(gids):
            color = self.colorlist[i_ % len(self.colorlist)]
            print 'Debug gid:', gid, self.tp[gid, :], 'color:', color
            marker = self.markerlist[i_ % len(self.markerlist)]
            PID.plot_selected_cells_in_tuning_space(gids=[gid], ax=ax, plot_cms=False, marker='o', color=color)

            conn_data_tgt = utils.get_targets(self.conn_lists['ee'], gid)
            tgt_gids[gid] = conn_data_tgt[:, 1].astype(np.int)
            w_out[gid] = conn_data_tgt[:, 2]
            print 'debug tgts', tgt_gids[gid], self.tp[tgt_gids[gid], :]
            PID.plot_selected_cells_in_tuning_space(gids=tgt_gids[gid], ax=ax, plot_cms=True, marker=marker, color=color)

            conn_data_src = utils.get_sources(self.conn_lists['ee'], gid)
            src_gids[gid] = conn_data_src[:, 0].astype(np.int)
            w_in[gid] = conn_data_src[:, 2]
            print 'debug srcs', self.tp[src_gids[gid], :]
            PID.plot_selected_cells_in_tuning_space(gids=src_gids[gid], ax=ax, plot_cms=True, marker=marker, color=color, alpha=.3)


    def get_w_in(self, tgt_gid, src_type, conn_list_type='ee'):
        """
        tgt_gid -- the receiving neuron gid
        src_type -- origin of connections; either 'exc', or 'inh' 
        """
        conn_data_src = utils.get_sources(self.conn_lists[conn_list_type], tgt_gid)
        src_gids = conn_data_src[:, 0].astype(np.int)
        weights_in = conn_data_src[:, 2]
        nspikes_in = self.nspikes['exc'][src_gids]

        weights_in_binned_x, xbins = np.histogram(self.tp[src_gids, 0] - self.tp[tgt_gid, 0], bins=self.n_bins, \
                range=self.x_range, weights=weights_in)
        cond_in_binned_x, xbins = np.histogram(self.tp[src_gids, 0] - self.tp[tgt_gid, 0], bins=self.n_bins, \
                range=self.x_range, weights=weights_in * nspikes_in)

        if src_type == 'inh':
            weights_in_binned_x *= -1
            cond_in_binned_x *= -1

        # debug output
#        info = '\ntgt_gid = %d ' % (tgt_gid)
#        info += '\nsrcs = %s' % (str(src_gids))
#        info += '\nweights_in= %s' % (str(weights_in))
#        info += '\nnspikes_in= %s' % (str(nspikes_in))
#        info += '\nx_src = %s' % (str(self.tp[src_gids, 0]))
#        info += '\nweights_in_binned_x %s' % (str(weights_in_binned_x))
#        info += '\ncond_in_binned_x %s' % (str(cond_in_binned_x))
#        print 'info', info
        return weights_in_binned_x, cond_in_binned_x, xbins


    def get_w_out(self, src_gid, tgt_type, conn_list_type='ee'):
        """
        src_gid -- the receiving neuron gid
        tgt_type -- origin of connections; either 'exc', or 'inh' 
        """
        conn_data_tgt = utils.get_targets(self.conn_lists[conn_list_type], src_gid)
        tgt_gids = conn_data_tgt[:, 1].astype(np.int)
        weights_out = conn_data_tgt[:, 2]
        nspikes_out = self.nspikes['exc'][tgt_gids]
        weights_out_binned_x, xbins = np.histogram(self.tp[tgt_gids, 0] - self.tp[src_gid, 0], bins=self.n_bins, \
                range=self.x_range, weights=weights_out)
        cond_out_binned_x, xbins = np.histogram(self.tp[tgt_gids, 0] - self.tp[src_gid, 0], bins=self.n_bins, \
                range=self.x_range, weights=weights_out * nspikes_out)
        if tgt_type == 'inh':
            weights_out_binned_x *= -1
            cond_out_binned_x *= -1
        return weights_out_binned_x, cond_out_binned_x, xbins


    def plot_conductance_flow_histograms(self, gids):

        n_cells = len(gids)

        w_in_exc_hist = np.zeros((n_cells, self.n_bins))
        w_out_exc_hist = np.zeros((n_cells, self.n_bins))
        w_in_inh_hist = np.zeros((n_cells, self.n_bins))
        w_out_inh_hist = np.zeros((n_cells, self.n_bins))
        g_in_exc_hist = np.zeros((n_cells, self.n_bins))
        g_out_exc_hist = np.zeros((n_cells, self.n_bins))
        g_in_inh_hist = np.zeros((n_cells, self.n_bins))
        g_out_inh_hist = np.zeros((n_cells, self.n_bins))

        # mean and std container
        g_in_exc_hist_mean = np.zeros((self.n_bins, 2))
        g_out_exc_hist_mean = np.zeros((self.n_bins, 2))
        g_in_inh_hist_mean = np.zeros((self.n_bins, 2))
        g_out_inh_hist_mean = np.zeros((self.n_bins, 2))

        w_in_exc_hist_mean = np.zeros((self.n_bins, 2))
        w_out_exc_hist_mean = np.zeros((self.n_bins, 2))
        w_in_inh_hist_mean = np.zeros((self.n_bins, 2))
        w_out_inh_hist_mean = np.zeros((self.n_bins, 2))


        # plot connectivity histograms for the gids
        fig2 = pylab.figure(figsize=self.fig_size)
        title = '$\\tau_{pred}=%.3f$ n_exc=%d, p_ee=%.1e, w_ee=%.1e, w_ei=%.1e, w_ie=%.1e' % \
                (self.params['tau_prediction'], self.params['n_exc'], self.params['p_ee'], \
                self.params['w_tgt_in_per_cell_ee'], self.params['w_tgt_in_per_cell_ei'], \
                self.params['w_tgt_in_per_cell_ie'])
        output_filename = 'CondAnisotropy_%.1e_%d_%.1e_%.1e_%.1e_%.1e.png' % \
                (self.params['tau_prediction'], self.params['n_exc'], self.params['p_ee'], \
                self.params['w_tgt_in_per_cell_ee'], self.params['w_tgt_in_per_cell_ei'], \
                self.params['w_tgt_in_per_cell_ie'])
#        pylab.subplots_adjust(hspace=0.4)
#        pylab.subplots_adjust(wspace=0.4)
        ax1 = fig2.add_subplot(231)
        ax2 = fig2.add_subplot(232)
        ax3 = fig2.add_subplot(233)
        ax4 = fig2.add_subplot(234)
        ax5 = fig2.add_subplot(235)
        ax6 = fig2.add_subplot(236)

        # gather the in and out going data for all gids
        for i_, gid in enumerate(gids):
            # w, g in from exc
            weights_in_binned_x, cond_in_binned_x, xbins = PID.get_w_in(gid, 'exc')
            w_in_exc_hist[i_, :] = weights_in_binned_x
            g_in_exc_hist[i_, :] = cond_in_binned_x

            # w, g out from / to exc
            weights_out_binned_x, cond_out_binned_x, xbins = PID.get_w_out(gid, 'exc')
            w_out_exc_hist[i_, :] = weights_out_binned_x
            g_out_exc_hist[i_, :] = cond_out_binned_x

            # w, g in from inh
            weights_in_binned_x, cond_in_binned_x, xbins = PID.get_w_in(gid, 'inh', conn_list_type='ie')
            w_in_inh_hist[i_, :] = weights_in_binned_x
            g_in_inh_hist[i_, :] = cond_in_binned_x

            # w, g out from / to inh
#            weights_out_binned_x, cond_out_binned_x, xbins = PID.get_w_out(gid, 'inh')
#            w_out_inh_hist[i_, :] = weights_out_binned_x
#            g_out_inh_hist[i_, :] = cond_out_binned_x


        for i_bin in xrange(self.n_bins):
            w_in_exc_hist_mean[i_bin, 0] = w_in_exc_hist[:, i_bin].mean()
            w_in_exc_hist_mean[i_bin ,1] = w_in_exc_hist[:, i_bin].std()
            w_out_exc_hist_mean[i_bin, 0] = w_out_exc_hist[:, i_bin].mean()
            w_out_exc_hist_mean[i_bin ,1] = w_out_exc_hist[:, i_bin].std()

            w_in_inh_hist_mean[i_bin, 0] = w_in_inh_hist[:, i_bin].mean()
            w_in_inh_hist_mean[i_bin ,1] = w_in_inh_hist[:, i_bin].std()
#            w_out_inh_hist_mean[i_bin, 0] = w_out_inh_hist[:, i_bin].mean()
#            w_out_inh_hist_mean[i_bin ,1] = w_out_inh_hist[:, i_bin].std()

            g_in_exc_hist_mean[i_bin, 0] = g_in_exc_hist[:, i_bin].mean()
            g_in_exc_hist_mean[i_bin ,1] = g_in_exc_hist[:, i_bin].std()
            g_out_exc_hist_mean[i_bin, 0] = g_out_exc_hist[:, i_bin].mean()
            g_out_exc_hist_mean[i_bin ,1] = g_out_exc_hist[:, i_bin].std()

            g_in_inh_hist_mean[i_bin, 0] = g_in_inh_hist[:, i_bin].mean()
            g_in_inh_hist_mean[i_bin ,1] = g_in_inh_hist[:, i_bin].std()
            g_out_inh_hist_mean[i_bin, 0] = g_out_inh_hist[:, i_bin].mean()
            g_out_inh_hist_mean[i_bin ,1] = g_out_inh_hist[:, i_bin].std()

        # difference exc - inh conductance
        g_in_diff = g_in_exc_hist_mean[:, 0] + g_in_inh_hist_mean[:, 0]
        g_out_diff = g_out_exc_hist_mean[:, 0] + g_out_inh_hist_mean[:, 0]

        # compute anisotropy of conductance flux
        negative_idx = (xbins < 0).nonzero()[0].tolist()
        positive_idx = (xbins > 0).nonzero()[0].tolist()
        negative_idx.reverse()
        positive_idx.pop() # remove the last element because x_bins has one element more than the cond values
        negative_idx.pop() # remove one to have the same length

        print 'debug size negative positive idx:', len(negative_idx), len(positive_idx), negative_idx, positive_idx
#        max_idx = min(negative_idx.size, positive_idx.size)
        print 'debug xbins[neg idx]', xbins[negative_idx]
        print 'debug xbins[pos idx]', xbins[positive_idx]
        g_anisotropic = g_out_diff[positive_idx] - g_out_diff[negative_idx]
        print 'g_anisotropic:', g_anisotropic

        bin_width = (xbins[1] - xbins[0]) / 2.
        # plot 1 
        p1 = ax1.bar(xbins[:-1], w_in_exc_hist_mean[:, 0], yerr=w_in_exc_hist_mean[:, 1], width=bin_width, color='b')
        p2 = ax1.bar(xbins[:-1] + bin_width, w_out_exc_hist_mean[:, 0], yerr=w_out_exc_hist_mean[:, 1], width=bin_width, color='r')
        ax1_info_p1 = '$w_{in}^{exc}$'
        ax1_info_p2 = '$w_{out}^{exc}$'
        ax1.legend([p1[0], p2[0]], [ax1_info_p1, ax1_info_p2], loc='upper right')
        ax1.set_ylabel('Exc weights')
        ax1.set_ylim((0, ax1.get_ylim()[1]))

        # plot 2: exc conductances in / out
        p1 = ax2.bar(xbins[:-1], g_in_exc_hist_mean[:, 0], yerr=g_in_exc_hist_mean[:, 1], width=bin_width, color='b')
        p2 = ax2.bar(xbins[:-1] + bin_width, g_out_exc_hist_mean[:, 0], yerr=g_out_exc_hist_mean[:, 1], width=bin_width, color='r')
        ax2_info_p1 = '$g_{in}^{exc}$'
        ax2_info_p2 = '$g_{out}^{exc}$'
        ax2.legend([p1[0], p2[0]], [ax2_info_p1, ax2_info_p2], loc='upper right')
        ax2.set_ylabel('Exc conductance')
        ax2.set_ylim((0, ax2.get_ylim()[1]))

        # plot 4: exc - inh weights incoming
        p1 = ax4.bar(xbins[:-1], w_in_exc_hist_mean[:, 0] + w_in_inh_hist_mean[:, 0], width=bin_width, color='b')
        p2 = ax4.bar(xbins[:-1] + bin_width, w_out_exc_hist_mean[:, 0] + w_out_inh_hist_mean[:, 0], width=bin_width, color='r')
        ax4_info_p1 = '$w_{in}^{exc} + w_{in}^{inh}$'
        ax4_info_p2 = '$w_{out}^{exc} + w_{out}^{inh}$'
        ax4.legend([p1[0], p2[0]], [ax4_info_p1, ax4_info_p2], loc='upper right')
        ax4.set_ylabel('Exc+Inh weights')
#        ax4_info = '$w_{in}^{exc} - w_{in}^{inh}$'
#        ax4.set_title(ax4_info)

        # plot 5: exc - inh conductances incoming
#        g_in_diff = g_in_exc_hist_mean[:, 0] + g_in_inh_hist_mean[:, 0]
#        p1 = ax5.bar(xbins[:-1], g_in_exc_hist_mean[:, 0] + g_in_inh_hist_mean[:, 0], width=bin_width, color='b')
#        p2 = ax5.bar(xbins[:-1] + bin_width, g_out_exc_hist_mean[:, 0] + g_out_inh_hist_mean[:, 0], width=bin_width, color='r')
        p1 = ax5.bar(xbins[:-1], g_in_diff, width=bin_width, color='b')
        p2 = ax5.bar(xbins[:-1] + bin_width,  g_out_diff, width=bin_width, color='r')
#        ax5_info = '$g_{in}^{exc} - g_{in}^{inh}$'
#        ax5.set_title(ax5_info)
        ax5_info_p1 = '$g_{in}^{exc} + g_{in}^{inh}$'
        ax5_info_p2 = '$g_{out}^{exc} + g_{out}^{inh}$'
        ax5.legend([p1[0], p2[0]], [ax5_info_p1, ax5_info_p2], loc='upper right')
        ax5.set_ylabel('Exc+Inh conductance')
        

        # plot 6 : anisotropy of conductance
        bin_width = (xbins[1] - xbins[0])
        p1 = ax6.bar(xbins[positive_idx], g_anisotropic, width=bin_width, color='g')
        ax6.set_title('Conductance difference anisotropy')
        ax6.set_ylabel('Conductance')
        ax6.set_xlabel('Forward direction')


        # plot 3: rasterplot with blank
        plot_rp.plot_input_spikes_sorted_in_space(self.params, self.tp, ax3, c='b', sort_idx=0, ms=3)
        plot_rp.plot_output_spikes_sorted_in_space(self.params, self.tp, ax3, 'exc', c='k', sort_idx=0, ms=3)
        plot_rp.plot_vertical_blank(params, ax3)
        plot_rp.plot_start_stop(params, ax3)
        xticks = [0, 500, 1000, 1500]
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(['%d' % i for i in xticks])
        ax3.set_yticklabels(['', '.2', '.4', '.6', '.8', '1.0'])
        ax3.set_xlabel('Time [ms]')
        ax3.set_xlim((0, params['t_sim']))

        ax1.set_xlim(self.x_range)
        ax2.set_xlim(self.x_range)
        ax4.set_xlim(self.x_range)
        ax5.set_xlim(self.x_range)
        print 'Saving to:', self.params['figures_folder'] + output_filename
        ax2.set_title(title)
        pylab.savefig(output_filename, dpi=300)
        pylab.savefig(self.params['figures_folder']+output_filename, dpi=300)
#            ax2.bar(bins[:-1] + i_ * bin_width, x_cnt_tgt, width=bin_width, color=self.colorlist[i_ % len(self.colorlist)])

#        bin_width = (bins[1] - bins[0]) / 2.
#        ax1.bar(bins[:-1], x_cnt_tgt, width=bin_width, color='b')
#        ax1.bar(bins[:-1]+bin_width, x_cnt_src, width=bin_width, color='r')



if __name__ == '__main__':

    if len(sys.argv) == 2:
        folder_name = sys.argv[1]
        params = utils.load_params(folder_name)
    else:
        print '\nPlotting the default parameters give in simulation_parameters.py\n'
        network_params = simulation_parameters.parameter_storage()
        params = network_params.params

    PID = PlotInformationDiffusion(params)

    # select cells 
    n_cells = 30
    x_tgt = params['motion_params'][0] + .6 * params['motion_params'][2] * params['t_before_blank'] / 1000.

#    x_tgt = .3 + params['motion_params'][2] * params['t_before_blank'] / 1000.
    gids = PID.select_cells(x_tgt, params['motion_params'][2], n_cells=n_cells, w_pos=.5)
    print 'x_tgt:', x_tgt
#    print 'gids', gids
    print 'x_pos', PID.tp[gids, 0]
    print 'vx', PID.tp[gids, 2]

    PID.load_spike_data()
#    PID.create_fig()
#    PID.plot_selected_cells_in_tuning_space(gids=[gids])

    PID.load_connection_data(conn_type='ee')
    PID.load_connection_data(conn_type='ie')
#    PID.plot_connections(gids)
    PID.plot_conductance_flow_histograms(gids)
#    PID.get_w_in(gids[0], 'exc')

#    pylab.show()

