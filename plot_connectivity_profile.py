import numpy as np
import utils
import pylab
import sys
import os
import simulation_parameters
import matplotlib
from matplotlib import cm

class ConnectionPlotter(object):

    def __init__(self, params):
        self.params = params

        self.tp_exc = np.loadtxt(params['tuning_prop_means_fn'])
        self.tp_inh = np.loadtxt(params['tuning_prop_inh_fn'])
        self.connection_matrices = {}
        self.connection_lists = {}
        self.delays = {}

#        self.lw_max = 10 # maximum line width for connection strengths
        self.markersize_cell = 10
        self.markersize_min = 3
        self.markersize_max = 12
        self.shaft_width = 0.003
        self.fig = pylab.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('x position')
        self.ax.set_ylabel('y position')
        self.ax.set_xlim((-0.1, 1.1))
        self.ax.set_ylim((-0.1, 1.1))
        self.legends = []
        self.quivers = {}
        self.conn_list_loaded = [False, False, False, False]
        self.conn_mat_loaded = [False, False, False, False]
        self.delay_colorbar_set = False



    def plot_cell(self, cell_id, exc=True, color='r', annotate=True):
        """
        markers = {0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 'D': 'diamond', 
                    6: 'caretup', 7: 'caretdown', 's': 'square', '|': 'vline', '': 'nothing', 'None': 'nothing', 'x': 'x', 
                   5: 'caretright', '_': 'hline', '^': 'triangle_up', ' ': 'nothing', 'd': 'thin_diamond', None: 'nothing', 
                   'h': 'hexagon1', '+': 'plus', '*': 'star', ',': 'pixel', 'o': 'circle', '.': 'point', '1': 'tri_down', 
                   'p': 'pentagon', '3': 'tri_left', '2': 'tri_up', '4': 'tri_right', 'H': 'hexagon2', 'v': 'triangle_down', 
                   '8': 'octagon', '<': 'triangle_left', '>': 'triangle_right'}
        """
        marker = '^'
        if exc:
#            color = 'r'
            tp = self.tp_exc
        else:
#            color = 'b'
            tp = self.tp_inh

        x0, y0, u0, v0 = tp[cell_id, 0], tp[cell_id, 1], tp[cell_id, 2], tp[cell_id, 3]
        self.ax.plot(x0, y0, marker, c=color, markersize=self.markersize_cell)
        self.quivers[cell_id] = (x0, y0, u0, v0, 'y', self.shaft_width*3)
        if annotate:
            self.ax.annotate('%d' % cell_id, (x0 + 0.01, y0 + 0.01), fontsize=12)


    def plot_connections(self, tgt_ids, tgt_tp, weights, marker, color, quiver=False, annotate=False):
        """
        """
        markersizes = utils.linear_transformation(weights, self.markersize_min, self.markersize_max)
        for i_, tgt in enumerate(tgt_ids):
            x_tgt = tgt_tp[tgt, 0] 
            y_tgt = tgt_tp[tgt, 1] 
            w = weights[i_]
            plot = self.ax.plot(x_tgt, y_tgt, marker, c=color, markersize=markersizes[i_])
            if quiver:
                self.quivers[tgt] = (x_tgt, y_tgt, tgt_tp[tgt, 2], tgt_tp[tgt, 3], color, self.shaft_width)
            if annotate:
                self.ax.annotate('%d' % tgt, (x_tgt + 0.01, y_tgt + 0.01), fontsize=12)
        return plot

    def plot_connection_type(self, src_gid, conn_type, marker, color, quiver=False, plot_delays=False, annotate=False):
        self.load_connection_list(conn_type)
        targets = utils.get_targets(self.connection_lists[conn_type], src_gid)
        tgt_ids, weights, delays = targets[:, 1], targets[:, 2], targets[:, 3]
        if conn_type == 'ee':
            src_tp = self.tp_exc
            tgt_tp = self.tp_exc
            legend_txt = 'exc src %d --> exc tgts' % src_gid
        elif conn_type == 'ei':
            src_tp = self.tp_exc
            tgt_tp = self.tp_inh
            legend_txt = 'exc src %d --> inh tgts' % src_gid
        elif conn_type == 'ie':
            src_tp = self.tp_inh
            tgt_tp = self.tp_exc
            legend_txt = 'inh src %d --> exc tgts' % src_gid
        elif conn_type == 'ii':
            src_tp = self.tp_inh
            tgt_tp = self.tp_inh
            legend_txt = 'inh src %d --> inh tgts' % src_gid
        plot = self.plot_connections(tgt_ids, tgt_tp, weights, marker, color, quiver, annotate)

        if plot_delays:
#            delay_max = self.connection_lists[conn_type][:, 3].max()
#            delay_min = self.connection_lists[conn_type][:, 3].min()
            delay_min, delay_max = self.params['delay_range'][0], self.params['delay_range'][1]
            norm = matplotlib.mpl.colors.Normalize(vmin=delay_min, vmax=delay_max)
            m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)#spring)
            m.set_array(np.arange(delay_min, delay_max, 0.01))

            x_src, y_src = src_tp[src_gid, 0], src_tp[src_gid, 1]
            for i_, tgt_gid in enumerate(tgt_ids):
                x_tgt, y_tgt = tgt_tp[tgt_gid, 0], tgt_tp[tgt_gid, 1]
                c = m.to_rgba(delays[i_])
                self.ax.plot((x_src, x_tgt), (y_src, y_tgt), c=c, lw=2)

            if not self.delay_colorbar_set:
                self.fig.colorbar(m)
                self.delay_colorbar_set = True
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

        self.legends.append((plot[0], legend_txt))
        print 'src_gid %d has %d outgoing %s->%s connection' % (src_gid, len(weights), conn_type[0].capitalize(), conn_type[1].capitalize())
        return tgt_ids


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
        self.ax.legend(plots, labels)


    def plot_quivers(self):

        data = np.zeros((len(self.quivers.keys()), 4))
        for i_, key in enumerate(self.quivers.keys()):
            (x, y, u, v, c, shaft_width) = self.quivers[key]
            data[i_, :] = np.array([x, y, u, v])
            self.ax.quiver(data[i_, 0], data[i_, 1], data[i_, 2], data[i_, 3], angles='xy', scale_units='xy', scale=1, color=c, headwidth=3, width=shaft_width)
#        self.ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], angles='xy', scale_units='xy', scale=1, color=c, headwidth=3)


    def load_connection_list(self, conn_type):

        if conn_type == 'ee':
            n_src, n_tgt = self.params['n_exc'], self.params['n_exc']
            loaded = self.conn_list_loaded[0]
        elif conn_type == 'ei':
            n_src, n_tgt = self.params['n_exc'], self.params['n_inh']
            loaded = self.conn_list_loaded[1]
        elif conn_type == 'ie':
            n_src, n_tgt = self.params['n_inh'], self.params['n_exc']
            loaded = self.conn_list_loaded[2]
        elif conn_type == 'ii':
            n_src, n_tgt = self.params['n_inh'], self.params['n_inh']
            loaded = self.conn_list_loaded[3]

        if loaded:
            return

        conn_list_fn = self.params['merged_conn_list_%s' % conn_type]
        if not os.path.exists(conn_list_fn):
            print '\n%s NOT FOUND\n Calling python merge_connlists.py\n'
            os.system('python merge_connlists.py')
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


if __name__ == '__main__':
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    P = ConnectionPlotter(params)

    np.random.seed(0)
    try:
        gid = int(sys.argv[1])
    except:
        # choose a cell as source 
        gid = int(.5 * params['n_exc'])
#        gid = np.random.randint(0, params['n_exc'], 1)[0]
        print 'plotting GID', gid
        
    with_directions = False
    with_delays = True

    exc_color = (.5, .5, .5)
    P.plot_cell(gid, exc=True, color='r')
    ee_targets = P.plot_connection_type(gid, 'ee', 'o', 'r', with_directions, with_delays)
    ei_targets = P.plot_connection_type(gid, 'ei', 'x', 'r', with_directions, with_delays)#, annotate=True)

#    inh_color = (.5, .5, .5)

    with_directions = False
    with_delays = False
    inh_color = 'b'
    inh_gid = ei_targets[0]
    P.plot_cell(inh_gid, exc=False, color='b')
    ie_targets = P.plot_connection_type(inh_gid, 'ie', 'o', inh_color, with_directions, with_delays)
    ii_targets = P.plot_connection_type(inh_gid, 'ii', 'x', inh_color, with_directions, with_delays)

#    P.plot_ee(gid)
#    tgts = P.plot_ei(gid)
#    gid = tgts[0]
#    P.plot_cell(gid, exc=False, color='b')
#    P.plot_connection_type(gid, 'ee', 'x', 'r', with_directions)

#    if with_directions:
    P.plot_quivers()

#    P.plot_cells_as_dots(range(params['n_exc']), P.tp_exc)
#    P.plot_cells_as_dots(range(params['n_exc']), P.tp_inh)


    # debug #find tgt inh cells which have
#    import CreateConnections as CC 
#    tp_src = np.loadtxt(params['tuning_prop_means_fn'])
#    tp_tgt = np.loadtxt(params['tuning_prop_inh_fn'])
#    src = gid
#    n_tgt = params['n_inh']
#    p, latency = np.zeros(n_tgt), np.zeros(n_tgt)
#    for tgt in xrange(n_tgt):
#        p[tgt], latency[tgt] = CC.get_p_conn(tp_src[src, :], tp_tgt[tgt, :], params['w_sigma_x'], params['w_sigma_v'])
#    sorted_indices = np.argsort(p)
#    n_tgt_cells_per_neuron = int(round(params['p_ei'] * n_tgt))
#    targets = sorted_indices[-n_tgt_cells_per_neuron:] 
#    for i in xrange(len(targets)):
#        tgt = targets[i]
#        print 'gid, tp_tgt, p', tgt, tp_tgt[tgt, :], p[tgt]
#    print 'tp_src', tp_src[src, :]
#    gid = targets[0]
    P.make_legend()

    output_fig = params['figures_folder'] + 'connectivity_profile_%d.png' % gid
    pylab.savefig(output_fig)

    pylab.show()

