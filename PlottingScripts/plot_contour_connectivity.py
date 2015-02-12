import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np 
import utils
import json


class ConnectivityPlotter(object):

    def __init__(self, params):
        self.params = params
        self.conn_lists = {}
        self.conn_lists_loaded = {
                'ee': False, \
                'ei_spec': False, \
                'ei_unspec': False, \
                'ie_spec': False, \
                'ie_unspec': False
                }
        plot_params = {'backend': 'png',
                      'axes.labelsize': 20,
                      'axes.titlesize': 20,
                      'text.fontsize': 20,
                      'xtick.labelsize': 16,
                      'ytick.labelsize': 16,
                      'legend.pad': 0.2,     # empty space around the legend box
                      'legend.fontsize': 14,
                       'lines.markersize': 1,
                       'lines.markeredgewidth': 0.,
                       'lines.linewidth': 1,
                      'font.size': 12,
                      'path.simplify': False,
                      'figure.subplot.left':.15,
                      'figure.subplot.bottom':.15,
                      'figure.subplot.right':.90,
                      'figure.subplot.top':.88,
                      'figure.subplot.hspace':.05, 
                      'figure.subplot.wspace':.30, 
                      'figure.figsize': utils.get_figsize(1000)}
        pylab.rcParams.update(plot_params)


    def plot_outgoing_connections_exc(self, tp_params=None):
        """
        tp_params -- select cells near these parameters in the tuning property space
        """
        markersize_cell = 30
        markersize_min = 1
        markersize_max = 15

        conn_type = 'ee'
        self.load_conn_list(conn_type)
        d = self.conn_lists[conn_type]
        tp = np.loadtxt(self.params['tuning_prop_exc_fn'])
        src_gids = d[:, 0]
        tgt_gids = d[:, 0]
        gids, dist = utils.get_gids_near_stim(tp_params, tp, n=1)
        src_gid = gids[0]
        
        # get the targets for this cell

        tgts_with_weights = utils.get_targets(d, src_gid)
        target_gids = np.array(tgts_with_weights[:, 1], dtype=int) - 1
        print 'Source cell %d (+ 1) projects to:' % (src_gid), target_gids 
        weights = tgts_with_weights[:, 2]

        fig = pylab.figure()
        ax = fig.add_subplot(111)

        x_tgts = tp[target_gids, 0]
        vx_tgts = tp[target_gids, 2]


        abs_max = max(abs(weights.min()), weights.max())
#        markersizes = utils.transform_linear(weights, (0., abs_max))
        print 'weights:', weights, abs(weights), np.min(weights), np.max(weights), weights.size
        if weights.min() == weights.max():
            markersizes = np.ones(weights.size)
            print 'WARNING all weights are equal!'
        else:
            markersizes = utils.transform_linear(abs(weights), (markersize_min, markersize_max))
        norm = matplotlib.colors.Normalize(vmin=weights.min(), vmax=weights.max())

        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.bwr) # large weights -- black, small weights -- white
        m.set_array(weights)
        rgba_colors = m.to_rgba(weights)

#        ax.plot(tp[:, 0], tp[:, 2], 'o', c='k', markersize='1', ls='')
        ax.plot(tp[:, 0], tp[:, 2], 'o', markeredgewidth=0, c='k', markersize=1, ls='')

        for i_, tgt in enumerate(target_gids):
            ax.plot(x_tgts[i_], vx_tgts[i_], 'o', markeredgewidth=0, c=rgba_colors[i_], markersize=markersizes[i_])
#            print 'debug i_ %d tgt_gid %d weight %.2f ms %.1f color' % (i_, tgt, weights[i_], markersizes[i_]), rgba_colors[i_]


        annotate = False
        if annotate:
            sort_idx = np.argsort(weights)
            n_ = 3
            idx = np.array(np.round(np.linspace(0, weights.size, n_, endpoint=False)), dtype=int)
            for i_ in xrange(n_):
                gid_ = sort_idx[idx[i_]]
                x_, y_ = tp[gid_, 0], tp[gid_, 2]
                ax.text(x_, y_, '%d, w=%.2f' % (gid_, weights[idx[i_]]))
                print 'weight:', weights[idx[i_]]

#            pos_idx = np.nonzero(weights > 0)[0] 
#            for i_ in xrange(n_):
#                gid_ = sort_idx[pos_idx[i_]]
#                x_, y_ = tp[gid_, 0], tp[gid_, 2]
#                ax.text(x_, y_, '%d, w=%.2f' % (gid_, weights[pos_idx[i_]]))
#                print 'weight:', weights[pos_idx[i_]]
           
        sort_idx = np.argsort(weights)
        print 'min and max weights:'
        print 'GID weight'
        for i_ in xrange(5):
            print '%d\t%.2f' % (target_gids[sort_idx[-(i_+1)]], weights[sort_idx[-(i_+1)]])
        for i_ in xrange(5):
            print '%d\t%.2f' % (target_gids[sort_idx[i_]], weights[sort_idx[i_]])

        ax.plot(tp[src_gid, 0], tp[src_gid, 2], '*', markersize=markersize_cell, c='y', markeredgewidth=1, label='source')#, zorder=source_gids.size + 10)
        output_fn = self.params['figures_folder'] + 'contour_taui%d_src%d.png' % (self.params['taui_bcpnn'], src_gid)
        print 'Saving fig to:', output_fn
        pylab.savefig(output_fn, dpi=200)


    def load_conn_list(self, conn_type):
        conn_list_fn = params['merged_conn_list_%s' % conn_type]
        if not os.path.exists(conn_list_fn):
            print 'Merging connection files...'
            utils.merge_connection_files(params, conn_type, iteration=None)
        print 'Loading ', conn_list_fn
        try:
            self.conn_lists[conn_type] = np.loadtxt(conn_list_fn)
        except:
            print '\nERROR: Could not find conn_list_fn:', conn_list_fn
            print '\tCheck if the simulation finished!\n\tWill now quit'
            exit(1)
        self.conn_lists_loaded[conn_type] = True


if __name__ == '__main__':

    tp_params = (0.5, 0.5, 0.5, 0.)
    if len(sys.argv) == 1:
        print 'Case 1: default parameters'
        import simulation_parameters
        GP = simulation_parameters.parameter_storage()
        params = GP.params
        P = ConnectivityPlotter(params)
        P.plot_outgoing_connections_exc(tp_params)
    elif len(sys.argv) == 2:
        print 'Case 2'
        if sys.argv[1].endswith('.json') or os.path.isdir(sys.argv[1]):
            params = utils.load_params(sys.argv[1])
            P = ConnectivityPlotter(params)
            P.plot_outgoing_connections_exc(tp_params)
        else:          
            print 'Please provide the folder / simulation_parameters.json file and not the conn_list.dat file!'
            exit(1)
    elif len(sys.argv) > 2:
        print 'Case 3'
        for fn in fns:
            params = utils.load_params(fn)
            P = ConnectivityPlotter(params)
            P.plot_outgoing_connections_exc(tp_params)
            del P 
    pylab.show()

