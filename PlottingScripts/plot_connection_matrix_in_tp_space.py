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
import simulation_parameters
import functions


plot_params = {'backend': 'png',
              'axes.labelsize': 24,
              'axes.titlesize': 24,
              'text.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 14,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 1,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.15,
              'figure.subplot.bottom':.14,
              'figure.subplot.right':.90,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}

pylab.rcParams.update(plot_params)

def load_conn_list(params, conn_type='ee'):
    conn_list_fn = params['merged_conn_list_%s' % conn_type]
    if not os.path.exists(conn_list_fn):
        print 'Merging connection files...'
        utils.merge_connection_files(params, conn_type, iteration=None)
    print 'Loading ', conn_list_fn





def plot_connections_incoming(params, feature_dimension=2, ax=None):
    """
    Target perspective, trying to estimate the different amounts of 
    excitation arriving at a target cell.
    Maybe a speed (v_i/j) specific gain for bcpnn-weights can be derived 
    in order to find good regimes for motion extrapolation and: finding a 
    ratio for ampa/nmda weights that is in agreement with experiments
    Ref. Watt, van Rossum et al 2000 "Activity coregulates quantal AMPA and NMDA currents at neocortical synapses"
    """

    if feature_dimension==4:
        v_range = (.0, 180.)
        v_tolerance = 10.
    else:
        v_tolerance = .1
        v_range = (-1.0, 1.)
    tau_i = params['bcpnn_params']['tau_i']
    conn_fn = params['conn_matrix_mc_fn']
    if not os.path.exists(conn_fn):
        utils.merge_connection_files(params, 'ee', iteration=None)
#        print 'ERROR! Could not find:', conn_fn
#        conn_fn = raw_input('\n Please enter connection matrix (mc-mc) filename!\n')
    print 'Loading:', conn_fn, 
    W = np.loadtxt(conn_fn)
    print 'done'

    W_in_sum = np.zeros(params['n_mc'])
    W_in_exc = np.zeros((params['n_mc'], 3))
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    # get the average tp for a mc
    avg_tp = utils.get_avg_tp(params, tp)

    clim = (v_range[0], v_range[1])
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(avg_tp[:, feature_dimension])
    colorlist= m.to_rgba(avg_tp[:, feature_dimension])

    if ax == None:
        fig = pylab.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False)

    for mc_tgt in xrange(params['n_mc']):
        v_tgt = avg_tp[mc_tgt, feature_dimension]
#        print 'v_tgt:', v_tgt
        x_tgt = avg_tp[mc_tgt, 0]
        v_src = avg_tp[:, feature_dimension]
        x_src = avg_tp[:, 0]
        w_in = W[:, mc_tgt]
        exc_idx = np.where(w_in > 0.)[0]
        W_in_exc[mc_tgt, 0] = x_tgt
        W_in_exc[mc_tgt, 1] = v_tgt
        W_in_exc[mc_tgt, 2] = w_in[exc_idx].sum()
        W_in_sum[mc_tgt] = w_in.sum()
        valid_mc_idx = np.where(np.abs((v_src - v_tgt) / v_tgt) < v_tolerance)[0]
        if (v_tgt > v_range[0]) and (v_tgt < v_range[1]):
            ax.plot(x_src[valid_mc_idx] - x_tgt, w_in[valid_mc_idx], '-o', ms=3, c=colorlist[mc_tgt], lw=1)#, label='$x_{tgt}=%.2f\ v_{tgt}=%.2f$' % (x_tgt, v_tgt))

        ax.scatter(x_src - x_tgt, w_in, c='k', linewidths=0)

#        ax.scatter(x_tgt - x_src, w_in, c=m.to_rgba(v_src), linewidths=0)

#        ax.scatter(x_tgt - x_src, w_in, c=m.to_rgba(v_src), linewidths=0)
#            ax.plot(x_tgt - x_src[valid_mc_idx], w_in[valid_mc_idx], '-o', ms=3, c=colorlist[mc_tgt], lw=1)#, label='$x_{tgt}=%.2f\ v_{tgt}=%.2f$' % (x_tgt, v_tgt))

    output_fn = params['data_folder'] + 'w_in_exc_taui%d.dat' % (params['bcpnn_params']['tau_i'])
    print 'Saving incoming excitation to:', output_fn
    np.savetxt(output_fn, W_in_exc)

    ax.set_ylim((W.min(), W.max()))
    ax.set_xlim((-1., 1.))
    xlim = ax.get_xlim()
    ax.plot((xlim[0], xlim[1]), (0., 0.), '--', c='k', lw=2)
    ylim = ax.get_ylim()
    ax.plot((0., 0.), (ylim[0], ylim[1]), '--', c='k', lw=2)
    title = 'Incoming weights depending on target position'
    title += '\n $\\tau_{i}=%d\ v_{i}=%.1f$' % (params['bcpnn_params']['tau_i'], params['v_min_tp'])
    ax.set_title(title)
    ax.set_ylabel('$w_{in}$')
    ax.set_xlabel('Distance to target')
    cb = pylab.colorbar(m)
    if feature_dimension == 4:
        cblabel = '$\\theta_{tgt}$'
    else:
        cblabel = '$v_{tgt}$'
    cb.set_label(cblabel)
#    pylab.legend()
    output_fn = params['figures_folder'] + 'ingoing_bcpnn_weights_vs_pos_taui%04d_v%.1f.png' % (tau_i, params['v_min_tp'])
#    output_fn = 'ingoing_bcpnn_weights_vs_pos_taui%04d_v%.1f.png' % (tau_i, params['v_min_tp'])
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    return ax



def plot_connections_out(params, feature_dimension=2, ax=None):
    if feature_dimension==4:
        v_range = (.0, 180.)
        v_tolerance = 10.
    else:
        v_tolerance = .1
        v_range = (-1.0, 1.)
#    v_range = (0.5, 2.)
    tau_i = params['bcpnn_params']['tau_i']
#    conn_fn = sys.argv[2]
    conn_fn = params['conn_matrix_mc_fn']
    if not os.path.exists(conn_fn):
        utils.merge_connection_files(params, 'ee', iteration=None)
#        print 'ERROR! Could not find:', conn_fn
#        conn_fn = raw_input('\n Please enter connection matrix (mc-mc) filename!\n')
    print 'Loading:', conn_fn, 
    W = np.loadtxt(conn_fn)
    print 'done'

    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    # get the average tp for a mc
    avg_tp = utils.get_avg_tp(params, tp)
    print 'avg_tp:', avg_tp

    clim = (v_range[0], v_range[1])
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(avg_tp[:, feature_dimension])
    colorlist= m.to_rgba(avg_tp[:, feature_dimension])

    if ax == None:
        fig = pylab.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False)

    linestyles = ['-', '--', ':', '-.']

    for mc_src in xrange(params['n_mc']):
        v_src = avg_tp[mc_src, feature_dimension]
        print 'v_src:', v_src
        w_out = W[mc_src, :]
        x_src = avg_tp[mc_src, 0]
        v_tgt = avg_tp[:, feature_dimension]
        x_tgt = avg_tp[:, 0]
        valid_mc_idx = np.where(np.abs((v_tgt - v_src) / v_src) < v_tolerance)[0]
        if (v_src > v_range[0]) and (v_src < v_range[1]):
            ax.plot(x_tgt[valid_mc_idx] - x_src, w_out[valid_mc_idx], '-o', ms=3, c=colorlist[mc_src], lw=1)#, label='$x_{src}=%.2f\ v_{src}=%.2f$' % (x_src, v_src))

        ax.scatter(x_tgt - x_src, w_out, c=m.to_rgba(v_src), linewidths=0)
#        ax.scatter(d[:, 0], w_out, 1], c=colors, linewidths=0)
#            ax.plot(x_src - x_tgt[valid_mc_idx], w_out[valid_mc_idx], '-o', ms=3, c=colorlist[mc_src], lw=1)#, label='$x_{src}=%.2f\ v_{src}=%.2f$' % (x_src, v_src))


#    x = np.arange(-1., 1., 0.01)
#    mu = 0.
#    sigma = 0.2
#    alpha = 6.0
#    A = 1.5
#    offset = -2
#    skew_pos = A * functions.skew_normal(x, mu, sigma, alpha) + offset
#    ax.plot(x, skew_pos, label='Skew normal distribution $\sigma=%.1f\ \\alpha=%.1f$' % (sigma, alpha), c='k', lw=4)
      
#    ax.plot(x, x * skew_pos, label='Skew normal * x', c='g', lw=4)

    xlim = ax.get_xlim()
    ax.plot((xlim[0], xlim[1]), (0., 0.), '--', c='k', lw=2)
    ylim = ax.get_ylim()
    ax.plot((0., 0.), (ylim[0], ylim[1]), '--', c='k', lw=2)
    title = 'Outgoing weights depending on target position'
    title += '\n $\\tau_{i}=%d\ v_{i}=%.1f$' % (params['bcpnn_params']['tau_i'], params['v_min_tp'])
    ax.set_title(title)
    ax.set_ylabel('$w_{out}$')
    ax.set_xlabel('Distance to source')
    cb = pylab.colorbar(m)
    if feature_dimension == 4:
        cblabel = '$\\theta_{tgt}$'
    else:
        cblabel = '$v_{tgt}$'
    cb.set_label(cblabel)
#    pylab.legend()
    output_fn = 'outgoing_bcpnn_weights_vs_pos_taui%04d_v%.1f.png' % (tau_i, params['v_min_tp'])
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    return ax




if __name__ == '__main__':

    ax = None
    in_out = 'incoming'
#    in_out = 'outgoing'
    if len(sys.argv) == 1:
        print 'Case 1: default parameters'
        GP = simulation_parameters.parameter_storage()
        params = GP.params
        if params['with_orientation']:
            feature_dimension = 4
        if in_out == 'incoming':
            plot_connections_incoming(params, feature_dimension=feature_dimension)
        else:
            plot_connections_out(params, feature_dimension=feature_dimension)
        show = True
    elif len(sys.argv) == 2:
        params = utils.load_params(sys.argv[1])
        if params['with_orientation']:
            feature_dimension = 4
        if in_out == 'incoming':
            plot_connections_incoming(params, feature_dimension=feature_dimension)
        else:
            plot_connections_out(params, feature_dimension=feature_dimension)
        show = True
    else:
        for folder in sys.argv[1:]:
            params = utils.load_params(folder)
            if params['with_orientation']:
                feature_dimension = 4
            if in_out == 'incoming':
                plot_connections_incoming(params, feature_dimension=feature_dimension)
            else:
                ax = plot_connections_out(params, feature_dimension=feature_dimension, ax=ax)
            show = False
    if show:
        pylab.show()



