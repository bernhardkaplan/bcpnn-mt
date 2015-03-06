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


def estimate_connectivity_target_perspective(params, ax=None):
    """
    Estimate the number (and weights) of nmda / ampa connections based on the provided weight matrices.
    """

    tau_i = params['bcpnn_params']['tau_i']

    print 'Loading: weight matrices ...'
    W_ampa = np.loadtxt(params['conn_matrix_ampa_fn'])
    W_nmda = np.loadtxt(params['conn_matrix_nmda_fn'])
    print 'done'

    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    rfs = np.loadtxt(params['receptive_fields_exc_fn'])

    # get the average tp for a mc
    avg_tp = utils.get_avg_tp(params, tp)
    avg_rfs = utils.get_avg_tp(params, rfs)
#    print 'avg_tp:', avg_tp
    print 'avg_rfs:', avg_rfs 

    v_coactivation = avg_rfs[:, 2].mean()

    dt_coactivation_ampa = 2 * params['tau_syn']['ampa']

#    clim = (v_range[0], v_range[1])
#    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
#    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
#    m.set_array(avg_tp[:, 2])
#    colorlist= m.to_rgba(avg_tp[:, 2])
#    if ax == None:
#        fig = pylab.figure(figsize=(12, 12))
#        ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False)

    for mc_tgt in xrange(params['n_mc']):
        v_tgt = avg_tp[mc_tgt, 2]
        x_tgt = avg_tp[mc_tgt, 0]
        # V - coactivation
        # estimate which MC will be activated by the same stimulus as the target
        coactivated_mcs_v = np.where(np.abs(v_tgt - avg_tp[:, 2]) < v_coactivation)[0]
        coactivated_mcs_v_set = set(coactivated_mcs_v)

        # estimate which MC will be activated roughly at the same time as the target
        # coactivation radius:
        dx_stim = np.abs(dt_coactivation_ampa * v_tgt / params['t_stimulus']) + avg_rfs[:, 0].mean()
        print 'v_tgt:', v_tgt, 'x_tgt:', x_tgt, 'dx_stim:', dx_stim
        coactivated_mcs_x = np.where(np.abs(x_tgt - avg_tp[:, 0]) < dx_stim)[0]
        print 'debug condition:', np.abs(x_tgt - avg_tp[:, 0]), ' < ', dx_stim
        coactivated_mcs_x_set = set(coactivated_mcs_x)
        coactivated_mcs = np.array(list(coactivated_mcs_v_set.intersection(coactivated_mcs_x_set)))
        
#        print 'coactivated_mcs_x:', coactivated_mcs_x
#        print 'coactivated_mcs:', coactivated_mcs
        if coactivated_mcs.size > 0:
            print 'Coactivated to tgt %d' % mc_tgt, coactivated_mcs, 'tp coactivated_mcs_x:', avg_tp[coactivated_mcs, 0], avg_tp[coactivated_mcs, 2]
        else:
            print 'No coactivation found\n\n'

#        print 'coactivated_mcs_v ', coactivated_mcs_v 
#        print 'src_ampa:', src_ampa.sum()

#        src_nmda = W_nmda[:, mc_tgt]
#        print 'src_nmda:', src_nmda.sum()


#        v_src = avg_tp[mc_src, 2]
#        print 'v_src:', v_src
#            ax.plot(x_tgt[valid_mc_idx] - x_src, w_out[valid_mc_idx], '-o', ms=3, c=colorlist[mc_src], lw=1)#, label='$x_{src}=%.2f\ v_{src}=%.2f$' % (x_src, v_src))
#    xlim = ax.get_xlim()
#    ax.plot((xlim[0], xlim[1]), (0., 0.), '--', c='k', lw=2)
#    ylim = ax.get_ylim()
#    ax.plot((0., 0.), (ylim[0], ylim[1]), '--', c='k', lw=2)
#    title = 'Outgoing weights depending on target position'
#    title += '\n $\\tau_{i}=%d\ v_{i}=%.1f$' % (params['bcpnn_params']['tau_i'], params['v_min_tp'])
#    ax.set_title(title)
#    ax.set_ylabel('$w_{out}$')
#    ax.set_xlabel('Distance to source')
#    cb = pylab.colorbar(m)
#    cb.set_label('$v_{src}$')
#    output_fn = 'outgoing_bcpnn_weights_vs_pos_taui%04d_v%.1f.png' % (tau_i, params['v_min_tp'])
#    print 'Saving fig to:', output_fn
#    pylab.savefig(output_fn, dpi=200)
#    return ax




if __name__ == '__main__':

    ax = None
    if len(sys.argv) == 1:
        print 'Case 1: default parameters'
        GP = simulation_parameters.parameter_storage()
        params = GP.params
        estimate_connectivity_target_perspective(params)
        show = True
    elif len(sys.argv) == 2:
        params = utils.load_params(sys.argv[1])
        estimate_connectivity_target_perspective(params)
        show = True
    else:
        for folder in sys.argv[1:]:
            params = utils.load_params(folder)
            ax = estimate_connectivity_target_perspective(params, ax)
            show = False
    if show:
        pylab.show()



