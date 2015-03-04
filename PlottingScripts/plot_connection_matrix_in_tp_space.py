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





def plot_connections_out(params, ax=None):
    v_tolerance = .1
    v_range = (-0.0, 1.)
#    v_range = (0.5, 2.)
    tau_i = params['bcpnn_params']['tau_i']
#    conn_fn = sys.argv[2]
    conn_fn = params['conn_matrix_mc_fn']
    if not os.path.exists(conn_fn):
        print 'ERROR! Could not find:', conn_fn
        conn_fn = raw_input('\n Please enter connection matrix (mc-mc) filename!\n')
    print 'Loading:', conn_fn, 
    W = np.loadtxt(conn_fn)
    print 'done'


    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    # get the average tp for a mc
    avg_tp = get_avg_tp(params, tp)

    clim = (v_range[0], v_range[1])
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(avg_tp[:, 2])
    colorlist= m.to_rgba(avg_tp[:, 2])

    if ax == None:
        fig = pylab.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False)

    linestyles = ['-', '--', ':', '-.']

    for mc_src in xrange(params['n_mc']):
        v_src = avg_tp[mc_src, 1]
        if (v_src > v_range[0]) and (v_src < v_range[1]):
            x_src = avg_tp[mc_src, 0]
            v_tgt = avg_tp[:, 2]
            x_tgt = avg_tp[:, 0]
            w_out = W[mc_src, :]
            valid_mc_idx = np.where(np.abs((v_tgt - v_src) / v_src) < v_tolerance)[0]
            ax.plot(x_tgt[valid_mc_idx] - x_src, w_out[valid_mc_idx], '-o', ms=3, c=colorlist[mc_src], lw=1)#, label='$x_{src}=%.2f\ v_{src}=%.2f$' % (x_src, v_src))
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
    cb.set_label('$v_{src}$')
#    pylab.legend()
    output_fn = 'outgoing_bcpnn_weights_vs_pos_taui%04d_v%.1f.png' % (tau_i, params['v_min_tp'])
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    return ax




if __name__ == '__main__':

    ax = None
    if len(sys.argv) == 1:
        print 'Case 1: default parameters'
        GP = simulation_parameters.parameter_storage()
        params = GP.params
        plot_connections_out(params)
        show = True
    elif len(sys.argv) == 2:
        params = utils.load_params(sys.argv[1])
        plot_connections_out(params)
        show = True
    else:
        for folder in sys.argv[1:]:
            params = utils.load_params(folder)
            ax = plot_connections_out(params, ax)
            show = False
    if show:
        pylab.show()



