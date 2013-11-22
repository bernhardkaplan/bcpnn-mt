import matplotlib
matplotlib.use('Agg')
import numpy as np
import utils
import pylab
import sys
import os
import simulation_parameters
from matplotlib import cm
import json

# --------------------------------------------------------------------------


class ConnectionPlotter(object):

    def __init__(self, params):
        self.params = params

        self.tp_exc = np.loadtxt(params['tuning_prop_means_fn'])
        self.tp_inh = np.loadtxt(params['tuning_prop_inh_fn'])
        self.connection_matrices = {}
        self.connection_lists = {}
        self.delays = {}

#        self.lw_max = 10 # maximum line width for connection strengths
#        self.ax.set_xlim((0.1, 0.75))
#        self.ax.set_ylim((0.25, 0.75))
        self.legends = []
        self.quivers = {}
        self.directions = {'src' : {}, 'tgt':{}} # use two different dictionaries for source or target cells
#            (x, y, u, v, c, shaft_width) = self.quivers[key]
        self.conn_list_loaded = [False, False, False, False]
        self.conn_mat_loaded = [False, False, False, False]
        self.delay_colorbar_set = False
        self.x_min, self.x_max = 1.0, .0
        self.y_min, self.y_max = 1.0, .0
        self.quiver_scale = 1.

    def create_fig(self, n_plots_x, n_plots_y):
        self.n_plots_x, self.n_plots_y = n_plots_x, n_plots_y
        self.markersize_cell = 12
        self.markersize_min = 5
        self.markersize_max = 15
        self.shaft_width = 0.01
#        self.shaft_width = 0.005
        pylab.rcParams['axes.labelsize'] = 28
        pylab.rcParams['axes.titlesize'] = 32
        pylab.rcParams['xtick.labelsize'] = 24
        pylab.rcParams['ytick.labelsize'] = 24
        self.fig = pylab.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(n_plots_y, n_plots_x, 1, aspect='equal')

        # set title according to connectivity configuration
        if self.params['conn_conf'] == 'motion-based':
            title = 'Motion-based anisotropic'
        elif self.params['conn_conf'] == 'direction-based':
            title = 'Direction-based anisotropic'
        elif self.params['conn_conf'] == None:
            title = 'Isotropic' 
        else:
            print 'Unknown connectivity profile:', self.params['conn_conf']
        self.ax.set_title(title)
        self.ax.set_xlabel('$x$-position')
        self.ax.set_ylabel('$y$-position')


    def plot_cell(self, cell_id, exc=True, color='g', marker='D', annotate=False):
        """
        markers = {0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 'D': 'diamond', 
                    6: 'caretup', 7: 'caretdown', 's': 'square', '|': 'vline', '': 'nothing', 'None': 'nothing', 'x': 'x', 
                   5: 'caretright', '_': 'hline', '^': 'triangle_up', ' ': 'nothing', 'd': 'thin_diamond', None: 'nothing', 
                   'h': 'hexagon1', '+': 'plus', '*': 'star', ',': 'pixel', 'o': 'circle', '.': 'point', '1': 'tri_down', 
                   'p': 'pentagon', '3': 'tri_left', '2': 'tri_up', '4': 'tri_right', 'H': 'hexagon2', 'v': 'triangle_down', 
                   '8': 'octagon', '<': 'triangle_left', '>': 'triangle_right'}
        """
        if exc:
#            color = 'r'
            tp = self.tp_exc
        else:
#            color = 'b'
            tp = self.tp_inh

        # torus dimensions
        w, h = self.params['torus_width'], self.params['torus_height']
        x0, y0, u0, v0 = tp[cell_id, 0] % w, tp[cell_id, 1] % h, tp[cell_id, 2], tp[cell_id, 3]
#        x0, y0, u0, v0 = tp[cell_id, 0], tp[cell_id, 1], tp[cell_id, 2], tp[cell_id, 3]
        self.ax.plot(x0, y0, marker, c=color, markersize=self.markersize_cell, zorder=100000)

        if exc:
            color = 'y'
        else:
            color = 'b'

        print 'cell tuning:', x0, y0, u0, v0
        print 'cell target ', x0 + u0, y0 + v0
        self.ax.quiver(x0, y0, u0, v0, angles='xy', scale_units='xy', scale=self.quiver_scale, color=color, headwidth=3, width=self.shaft_width * 2, linewidths=(1,), edgecolors=('k'), zorder=100000)
#        self.quivers[cell_id] = (x0, y0, u0, v0, 'y', self.shaft_width*3, 'k')
        if annotate:
            self.ax.annotate('%d' % cell_id, (x0 + 0.01, y0 + 0.01), fontsize=12)


    def plot_connections(self, tgt_ids, tgt_tp, weights, marker, color, with_directions=False, annotate=False, is_target=True): 
        """
        """
        markersizes = utils.linear_transformation(weights, self.markersize_min, self.markersize_max)

        if is_target:
            quiver_style = '-'
            direction_color = (.4, .4, .4)
            direction_dict = self.directions['tgt']
        else:
#            quiver_style = ':'
            quiver_style = '-'
            direction_color = (.4, .4, .4)
#            direction_color = (.0, .0, .0)
            direction_dict = self.directions['src']

        for i_, tgt in enumerate(tgt_ids):
            x_tgt = tgt_tp[tgt, 0] % self.params['torus_width']#% 1
            y_tgt = tgt_tp[tgt, 1] % self.params['torus_height']#% 1
            self.x_min = min(x_tgt, self.x_min)
            self.y_min = min(y_tgt, self.y_min)
            self.x_max = max(x_tgt, self.x_max)
            self.y_max = max(y_tgt, self.y_max)
#            print 'debug', tgt, x_tgt, y_tgt
            w = weights[i_]
#            if is_target:
#                print 'x, y', x_tgt, y_tgt, w
            plot = self.ax.plot(x_tgt, y_tgt, marker, c=color, markersize=markersizes[i_], zorder=1000)
            if with_directions:
                direction_dict[tgt] = (x_tgt, y_tgt, tgt_tp[tgt, 2], tgt_tp[tgt, 3], direction_color, self.shaft_width, quiver_style)
            if annotate:
                self.ax.annotate('%d' % tgt, (x_tgt + 0.01, y_tgt + 0.01), fontsize=12)

        return plot

    
    def plot_connection_histogram(self, gid, conn_type):

        self.load_connection_list(conn_type)
        targets = utils.get_targets(self.connection_lists[conn_type], gid)
        tgt_ids, tgt_weights, tgt_delays = targets[:, 1], targets[:, 2], targets[:, 3]

        sources = utils.get_sources(self.connection_lists[conn_type], gid)
        src_ids, src_weights, src_delays = sources[:, 1], sources[:, 2], sources[:, 3]
        
        fig = pylab.figure(figsize=(14, 10))
#        ax1 = fig.add_subplot(1, 1, 1) # set blank? the set title
        pylab.subplots_adjust(hspace=.35, wspace=.25)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        

        ax1.set_title('$\sigma^w_{X} = %.2f \sigma^w_{V}=%.2f$' % (self.params['w_sigma_x'], self.params['w_sigma_v']))
        tgt_weights_sorted = tgt_weights.copy()
        tgt_weights_sorted.sort()
        ax1.bar(range(len(tgt_ids)), tgt_weights_sorted, width=1)
        ax1.set_ylabel('Outgoing weights [uS]')
        ax1.set_xlabel('sorted targets')
        ax1.set_xlim((0, len(tgt_ids)))

        n_weight_bins = 20
        count, bins = np.histogram(tgt_weights, bins=n_weight_bins)
        ax2.bar(bins[:-1], count, width=bins[1] - bins[0])
        ax2.set_xlabel('Outgoing weight [uS]')
        ax2.set_ylabel('#')


        src_weights_sorted = src_weights.copy()
        src_weights_sorted.sort()
        ax3.bar(range(len(src_ids)), src_weights_sorted, width=1)
        ax3.set_ylabel('Incoming weights [uS]')
        ax3.set_xlabel('sorted sources')
        ax3.set_xlim((0, len(src_ids)))

        n_weight_bins = 20
        count, bins = np.histogram(src_weights, bins=n_weight_bins)
        ax4.bar(bins[:-1], count, width=bins[1] - bins[0])
        ax4.set_xlabel('Incoming weight [uS]')
        ax4.set_ylabel('#')

        output_fn = self.params['figures_folder'] + 'connection_histogram_wsigmaxv_%.2f_%.2f_%d.png' % (self.params['w_sigma_x'], self.params['w_sigma_v'], gid)
        print 'Saving fig to:', output_fn
        pylab.savefig(output_fn, dpi=200)



    def plot_connection_type(self, src_gid, conn_type, marker, color, outgoing_conns=True, with_directions=False, plot_delays=False, annotate=False, with_histogram=False):
        self.load_connection_list(conn_type)
        if outgoing_conns:
            src_tgts = utils.get_targets(self.connection_lists[conn_type], src_gid)
            tgt_ids, weights, delays = src_tgts[:, 1], src_tgts[:, 2], src_tgts[:, 3]
            print 'Cell %d connects to:' % src_gid, tgt_ids
        else:
            src_tgts = utils.get_sources(self.connection_lists[conn_type], src_gid)
            tgt_ids, weights, delays = src_tgts[:, 0], src_tgts[:, 2], src_tgts[:, 3]
            print 'Cell %d receives input from:' % src_gid, tgt_ids
        if conn_type == 'ee':
            src_tp = self.tp_exc
            tgt_tp = self.tp_exc
            legend_txt = 'exc src gid: %d --> exc tgts, n=%d' % (src_gid, len(tgt_ids))
        elif conn_type == 'ei':
            src_tp = self.tp_exc
            tgt_tp = self.tp_inh
            legend_txt = 'exc src gid: %d --> inh tgts, n=%d' % (src_gid, len(tgt_ids))
        elif conn_type == 'ie':
            src_tp = self.tp_inh
            tgt_tp = self.tp_exc
            legend_txt = 'inh src gid: %d --> exc tgts, n=%d' % (src_gid, len(tgt_ids))
        elif conn_type == 'ii':
            src_tp = self.tp_inh
            tgt_tp = self.tp_inh
            legend_txt = 'inh src gid: %d --> inh tgts, n=%d' % (src_gid, len(tgt_ids))

        if len(tgt_ids) > 0:
            plot = self.plot_connections(tgt_ids, tgt_tp, weights, marker, color, with_directions, annotate, outgoing_conns)
            if outgoing_conns:
                print 'Average weight for outgoing connections: %.2e +- %.2e ' % (weights.mean(), weights.std())
            if outgoing_conns:
                print 'Average weight for outgoing connections: %.2e +- %.2e ' % (weights.mean(), weights.std())
        else:
            return []

        if plot_delays:
            delay_min, delay_max = 0, 1500
#            delay_min, delay_max = delays.min(), delays.max()
#            delay_min, delay_max = self.params['delay_range'][0], self.params['delay_range'][1]
            norm = matplotlib.mpl.colors.Normalize(vmin=delay_min, vmax=delay_max)
            m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)#spring)
            m.set_array(np.arange(delay_min, delay_max, 0.01))
#            if not self.delay_colorbar_set:
#                cb = self.fig.colorbar(m)
#                cb.set_label('Connection delays [ms]', fontsize=28)
#                self.delay_colorbar_set = True

            x_src, y_src = src_tp[src_gid, 0], src_tp[src_gid, 1]
            for i_, tgt_gid in enumerate(tgt_ids):
                x_tgt, y_tgt = tgt_tp[tgt_gid, 0] % self.params['torus_width'], tgt_tp[tgt_gid, 1] % self.params['torus_height']
                c = m.to_rgba(delays[i_])
                self.ax.plot((x_src, x_tgt), (y_src, y_tgt), c=c, lw=2, alpha=.5)

#            s = 1. # saturation
#            for 
#                if activity[frame, tgt_gid] < 0:
#                    l = 1. - 0.5 * activity[frame, tgt_gid] / activity_min
#                    h = 0.
#                else:
#                    l = 1. - 0.5 * activity[frame, tgt_gid] / activity_max
#                    h = 240.
#                assert (0 <= h and h < 360)
#                assert (0 <= l and l <= 1)
#                assert (0 <= s and s <= 1)
#                (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
#                colors[frame, tgt_gid, :] = [r, g, b]
#                

        if with_histogram:
            print '\nPlotting weight and delay histogram'
            self.plot_weight_and_delay_histogram(weights, delays)

        self.legends.append((plot[0], legend_txt))
        print 'src_gid %d has %d outgoing %s->%s connection' % (src_gid, len(weights), conn_type[0].capitalize(), conn_type[1].capitalize())
        return tgt_ids

    
    def plot_weight_and_delay_histogram(self, weights, delays):
        n_bins = 20
        w_mean, w_std = weights.mean(), weights.std()
        self.ax2 = self.fig.add_subplot(self.n_plots_x + 1, self.n_plots_y, 3)
        self.ax3 = self.fig.add_subplot(self.n_plots_x + 1, self.n_plots_y, 4)
        n_w, bins_w = np.histogram(weights, bins=n_bins, normed=False)
        bin_width = bins_w[1] - bins_w[0]
        self.ax2.bar(bins_w[:-1]-.5*bin_width, n_w, width=bin_width, label='$w_{mean} = %.2e \pm %.2e$' % (w_mean, w_std))
        self.ax2.set_xlabel('Weights')
        self.ax2.set_ylabel('Count')
        self.ax2.set_title('Histogram of outgoing weights')

        n_d, bins_d = np.histogram(delays, bins=n_bins, normed=False)
        bin_didth = bins_d[1] - bins_d[0]
        self.ax3.bar(bins_d[:-1]-.5*bin_didth, n_d, width=bin_didth, label='$w_{mean} = %.2e \pm %.2e$' % (w_mean, w_std))
        self.ax3.set_title('Histogram of connection delays')
        self.ax3.set_xlabel('Delays')
        self.ax3.set_ylabel('Count')




    def plot_cells_as_dots(self, gids, tp):
        marker = 'o'
        ms = 1
        color = 'k'
        for i in xrange(len(gids)):
            gid = gids[i]
            x, y = tp[gid, 0], tp[gid, 1]
            self.ax.plot(x, y, marker, markersize=ms, c=color)


    def make_legend(self):

        plots = []
        labels = []
        for i in xrange(len(self.legends)):
            plots.append(self.legends[i][0])
            labels.append(self.legends[i][1])
        self.ax.legend(plots, labels, loc='upper left')


    def plot_directions(self):

        alpha = .4
        data_tgt = np.zeros((len(self.directions['tgt'].keys()), 4))
        for i_, key in enumerate(self.directions['tgt'].keys()):
            (x, y, u, v, c, shaft_width, ls) = self.directions['tgt'][key]
            data_tgt[i_, :] = np.array([x, y, u, v])
            a = self.ax.quiver(data_tgt[i_, 0], data_tgt[i_, 1], data_tgt[i_, 2], data_tgt[i_, 3], angles='xy', scale_units='xy', scale=self.quiver_scale, linewidth=0, headwidth=5, width=shaft_width, alpha=alpha, linestyles=ls)#, zorder=1)

        data_src= np.zeros((len(self.directions['src'].keys()), 4))
        for i_, key in enumerate(self.directions['src'].keys()):
            (x, y, u, v, c, shaft_width, ls) = self.directions['src'][key]
            data_src[i_, :] = np.array([x, y, u, v])
            a = self.ax.quiver(data_src[i_, 0], data_src[i_, 1], data_src[i_, 2], data_src[i_, 3], angles='xy', scale_units='xy', scale=self.quiver_scale, linewidth=0, headwidth=5, width=shaft_width, alpha=alpha, linestyles=ls)#, zorder=1)
#            a = self.ax.quiver(data_src[i_, 0], data_src[i_, 1], data_src[i_, 2], data_src[i_, 3], angles='xy', scale_units='xy', scale=self.quiver_scale, facecolor='none', linewidth=2, headwidth=3, width=shaft_width, alpha=alpha, linestyles=ls)#, zorder=1)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim((xlim[0] - 0.02, xlim[1] + 0.25))
        self.ax.set_ylim((ylim[0] - 0.05, ylim[1] + 0.05))
#        self.ax.set_xlim((-.1, 1.1))
#        self.ax.set_ylim((-.1, 1.1))
#        print 'x_min, x_max', self.x_min, self.x_max
#        print 'y_min, y_max', self.y_min, self.y_max
#        self.ax.set_xlim((self.x_min - 0.05, self.x_max + 0.05))
#        self.ax.set_ylim((self.y_min - 0.05, self.y_max + 0.05))


    def load_connection_list(self, conn_type):

        if conn_type == 'ee':
            loaded = self.conn_list_loaded[0]
        elif conn_type == 'ei':
            loaded = self.conn_list_loaded[1]
        elif conn_type == 'ie':
            loaded = self.conn_list_loaded[2]
        elif conn_type == 'ii':
            loaded = self.conn_list_loaded[3]

        if loaded:
            return

        conn_list_fn = self.params['merged_conn_list_%s' % conn_type]
        print 'Trying to load', conn_list_fn
        if not os.path.exists(conn_list_fn):
            print '\n%s NOT FOUND:' % conn_list_fn
            print '\n Calling python merge_connlists.py\n'
            os.system('python merge_connlists.py %s' % self.params['folder_name']) 
        self.connection_lists[conn_type] = np.loadtxt(conn_list_fn)
            
        if conn_type == 'ee':
            self.conn_list_loaded[0] = True
        elif conn_type == 'ei':
            self.conn_list_loaded[1] = True
        elif conn_type == 'ie':
            self.conn_list_loaded[2] = True
        elif conn_type == 'ii':
            self.conn_list_loaded[3] = True



    def load_connection_matrices(self, conn_type):
        """
        deprecated - should not be used because of unnecessary memory consumption
        use load_connection_list instead to get sources / targets
        """

        if conn_type == 'ee':
            n_src, n_tgt = self.params['n_exc'], self.params['n_exc']
            loaded = self.conn_mat_loaded[0]
        elif conn_type == 'ei':
            n_src, n_tgt = self.params['n_exc'], self.params['n_inh']
            loaded = self.conn_mat_loaded[1]
        elif conn_type == 'ie':
            n_src, n_tgt = self.params['n_inh'], self.params['n_exc']
            loaded = self.conn_mat_loaded[2]
        elif conn_type == 'ii':
            n_src, n_tgt = self.params['n_inh'], self.params['n_inh']
            loaded = self.conn_mat_loaded[3]

        if loaded:
            return

        conn_mat_fn = self.params['conn_mat_fn_base'] + '%s.dat' % (conn_type)
        delay_mat_fn = self.params['delay_mat_fn_base'] + '%s.dat' % (conn_type)
        if os.path.exists(conn_mat_fn):
            print 'Loading', conn_mat_fn
            self.connection_matrices[conn_type] = np.loadtxt(conn_mat_fn)
        #    delays_ee = np.loadtxt(delay_mat_ee_fn)
        else:
            self.connection_matrices[conn_type], self.delays[conn_type] = utils.convert_connlist_to_matrix(params['merged_conn_list_%s' % conn_type], n_src, n_tgt)
            np.savetxt(conn_mat_fn, self.connection_matrices[conn_type])
#            np.savetxt(delay_mat_fn, self.delays[conn_type])
            
        if conn_type == 'ee':
            self.conn_mat_loaded[0] = True
        elif conn_type == 'ei':
            self.conn_mat_loaded[1] = True
        elif conn_type == 'ie':
            self.conn_mat_loaded[2] = True
        elif conn_type == 'ii':
            self.conn_mat_loaded[3] = True


    def find_exc_gid_to_plot(self):

        if os.path.exists(self.params['gids_to_record_fn']):
            good_gids = np.loadtxt(self.params['gids_to_record_fn'], dtype='int')
            idx = self.tp_exc[good_gids, 0].argsort()
            gids_to_check = good_gids[idx]
#            print 'debug x_pos', self.tp_exc[idx, 0]

            conn_list_ei = np.loadtxt(self.params['merged_conn_list_ei'])

            for gid in gids_to_check:
                inh_targets = utils.get_targets(conn_list_ei, gid)
                print 'gid %d at xpos %.2f has %d inh targets' % (gid, self.tp_exc[gid, 0], len(inh_targets))
                if len(inh_targets) > 0:
                    return gid
                 

        else: # choose any cell as source 
            gid = int(.5 * self.params['n_exc'])

        return gid


    def find_cell_closest_to_vector(self, v, direction=None):
        """
        v : target vector
        This function searches the  exc tuning properties and 
        returns the gid of the cell being closest to the target vecort
        """

        x_diff = (self.tp_exc[:, 0] - v[0])**2
        y_diff = (self.tp_exc[:, 1] - v[1])**2
        dist = x_diff + y_diff
        idx = dist.argsort()
        gid = idx[0]

        n = int(round(.10 * self.tp_exc[:, 0].size))
        if direction != None:
            assert (len(direction) == 2), 'Two dimensional vector required'
            # take the n cells closest to v and find the vector best aligned with direction
            gids = idx[0:n]
            cell_directions = np.array((self.tp_exc[gids, 2], self.tp_exc[gids, 3]))
            u_diff = (self.tp_exc[gids, 2] - direction[0])**2
            v_diff = (self.tp_exc[gids, 3] - direction[1])**2
            diff = u_diff + v_diff
            idx_ = diff.argsort()
            gid = gids[idx_[0]]

        print 'find_cell_closest_to_vector', v, direction
        print 'is ', gid, self.tp_exc[gid, :]
        return gid#, self.tp_exc[gid, :]
    


if __name__ == '__main__':


#    print 'Running merge_connlists.py...'
#    os.system('python merge_connlists.py')


    with_directions = True
    with_delays = True
    with_histogram = False
    if with_histogram:
        n_plots_x, n_plots_y = 1, 2
    else:
        n_plots_x, n_plots_y = 1, 1

    np.random.seed(0)
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            gid = int(sys.argv[1])
            param_fn = sys.argv[2]
            if os.path.isdir(param_fn):
                param_fn += '/Parameters/simulation_parameters.json'
            import json
            f = file(param_fn, 'r')
            print 'Loading parameters from', param_fn
            params = json.load(f)
        else:
            param_fn = sys.argv[1]
            if os.path.isdir(param_fn):
                param_fn += '/Parameters/simulation_parameters.json'
            import json
            f = file(param_fn, 'r')
            print 'Loading parameters from', param_fn
            params = json.load(f)
            gid = np.int(np.loadtxt(params['gids_to_record_fn'])[0])
    else:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        gid = np.int(np.loadtxt(params['gids_to_record_fn'])[0])

    P = ConnectionPlotter(params)

#    gid = 5339

    # here you can choose where the cell to plot should be sitting and what the preferred direction should be 
#    target_vector = (.3, .5)
#    direction = (.5, 0.)
#    gid = P.find_cell_closest_to_vector(target_vector, direction)

    print 'plotting gid', gid
    P.plot_connection_histogram(gid, 'ee')

    P.create_fig(n_plots_x, n_plots_y)
#    exc_color = (.5, .5, .5)
    outgoing_conns = True
    ee_targets = P.plot_connection_type(gid, 'ee', 'o', 'k', outgoing_conns, with_directions, plot_delays=with_delays, with_histogram=with_histogram)
    print 'ee_targets:', ee_targets
    print 'len(ee_targets):', len(ee_targets)
    outgoing_conns = False
    ee_sources = P.plot_connection_type(gid, 'ee', '^', 'r', outgoing_conns, with_directions, plot_delays=with_delays, with_histogram=with_histogram)
#    ei_targets = P.plot_connection_type(gid, 'ei', 'x', 'r', with_directions, plot_delays=with_delays)#, annotate=True)
    P.plot_cell(gid, exc=True, color='y')


    # search for an adequate inhibitory target cell
#    distances_between_exc_and_inh = np.zeros(len(ei_targets))
#    exc_x_pos, exc_y_pos = P.tp_exc[gid, 0], P.tp_exc[gid, 1]
#    for i_, inh_gid in enumerate(ei_targets):
#        inh_x_pos = P.tp_inh[inh_gid, 0]
#        inh_y_pos = P.tp_inh[inh_gid, 1]
#        distances_between_exc_and_inh[i_] = (inh_x_pos - exc_x_pos)**2 + (inh_y_pos - exc_y_pos)**2
#    idx = distances_between_exc_and_inh.argsort()
#    inh_gid = ei_targets[idx[int(.2 * len(ei_targets))]]

#    inh_color = (.5, .5, .5)
#    with_directions = False
#    with_delays = False
    inh_color = 'b'
#    inh_gid = ei_targets[1]
#    print 'inh gid', inh_gid
#    P.plot_cell(inh_gid, exc=False, color='b')
#    ie_targets = P.plot_connection_type(inh_gid, 'ie', 'o', inh_color, with_directions, with_delays)
#    ii_targets = P.plot_connection_type(inh_gid, 'ii', 'x', inh_color, with_directions, with_delays)

#    P.plot_ee(gid)
#    tgts = P.plot_ei(gid)
#    gid = tgts[0]
#    P.plot_cell(gid, exc=False, color='b')
#    P.plot_connection_type(gid, 'ee', 'x', 'r', with_directions)

    if with_directions:
        P.plot_directions()

#    P.plot_cells_as_dots(range(params['n_exc']), P.tp_exc)
#    P.plot_cells_as_dots(range(params['n_exc']), P.tp_inh)


#    P.make_legend()

    output_fig = params['figures_folder'] + 'connectivity_profile_%d_wsx%.2f_wsv%.2f_%s_varying_xylim.png' % (gid, params['w_sigma_x'], params['w_sigma_v'], str(params['conn_conf']))
    print 'Saving figure to', output_fig
    pylab.savefig(output_fig)

#    pylab.show()

