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


def get_gids_to_mc(params, pyr_gid):
    """
    Return the HC, MC within the HC in which the cell with pyr_gid is in
    and the min and max gid of pyr cells belonging to the same MC.
    """
    mc_idx = pyr_gid / params['n_exc_per_mc']
    hc_idx = mc_idx / params['n_mc_per_hc']
    gid_min = mc_idx * params['n_exc_per_mc']
    gid_max = (mc_idx + 1) * params['n_exc_per_mc']  # here no +1 because it's used for randrange and +1 would include a false cell
    return (hc_idx, mc_idx, gid_min, gid_max)


def get_avg_tp(params, tp):

    avg_tp = np.zeros((params['n_mc'], 2))
    cnt_cells= np.zeros(params['n_mc'])
    for i_ in xrange(tp[:, 0].size):
        (hc_idx, mc_idx, gid_min, gid_max) = get_gids_to_mc(params, i_)
        avg_tp[mc_idx, 0] += tp[i_, 0] # position
        avg_tp[mc_idx, 1] += tp[i_, 2] # speed
        cnt_cells[mc_idx] += 1

    for i_mc in xrange(params['n_mc']):

        # check if gid - mc mapping was correctly
        assert cnt_cells[i_mc] == params['n_exc_per_mc']
        avg_tp[i_mc, :] /= cnt_cells[i_mc]

    return avg_tp


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'Case 1: default parameters'
        GP = simulation_parameters.parameter_storage()
        params = GP.params
    else:
        params = utils.load_params(sys.argv[1])

    v_tolerance = .1
#    v_range = (-0.0, .2)
    v_range = (0.5, 2.)
    tau_i = 150
#    conn_fn = sys.argv[2]
    conn_fn = 'connection_matrix_20x16_taui%d_trained_with_AMPA_input_only.dat' % (tau_i)
    output_fn = 'outgoing_bcpnn_weights_vs_pos_taui%d.png' % (tau_i)
    print 'Loading:', conn_fn, 
    print 'done'
    W = np.loadtxt(conn_fn)

    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    # get the average tp for a mc
    avg_tp = get_avg_tp(params, tp)

#    colorlist = utils.get_colorlist(params['n_mc'])

    clim = (v_range[0], v_range[1])
#    clim = (.0, 1.) # tuning speed limit for colorcode
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(avg_tp[:, 1])
    colorlist= m.to_rgba(avg_tp[:, 1])
#    colorlist= m.to_rgba(np.abs(avg_tp[:, 1]))

    fig = pylab.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False)

    linestyles = ['-', '--', ':', '-.']

    for mc_src in xrange(params['n_mc']):
#    for mc_src in xrange(10):
        v_src = avg_tp[mc_src, 1]
        if (v_src > v_range[0]) and (v_src < v_range[1]):
            x_src = avg_tp[mc_src, 0]
            v_tgt = avg_tp[:, 1]
            x_tgt = avg_tp[:, 0]
            w_out = W[mc_src, :]
            valid_mc_idx = np.where(np.abs((v_tgt - v_src) / v_src) < v_tolerance)[0]
#            if x_tgt 
#            ls = 
            ax.plot(x_tgt[valid_mc_idx] - x_src, w_out[valid_mc_idx], '-o', ms=3, c=colorlist[mc_src], lw=1)#, label='$x_{src}=%.2f\ v_{src}=%.2f$' % (x_src, v_src))
            ax.plot(x_tgt[valid_mc_idx] - x_src, w_out[valid_mc_idx], '-o', ms=3, c=colorlist[mc_src], lw=1)#, label='$x_{src}=%.2f\ v_{src}=%.2f$' % (x_src, v_src))
#            ax.plot(x_src - x_tgt[valid_mc_idx], w_out[valid_mc_idx], '-o', ms=3, c=colorlist[mc_src], lw=1)#, label='$x_{src}=%.2f\ v_{src}=%.2f$' % (x_src, v_src))


    x = np.arange(-1., 1., 0.01)
    mu = 0.
    sigma = 0.2
    alpha = 6.0
    A = 1.5
    offset = -2
    skew_pos = A * functions.skew_normal(x, mu, sigma, alpha) + offset
    ax.plot(x, skew_pos, label='Skew normal distribution $\sigma=%.1f\ \\alpha=%.1f$' % (sigma, alpha), c='k', lw=4)

    xlim = ax.get_xlim()
    ax.plot((xlim[0], xlim[1]), (0., 0.), '--', c='k', lw=2)
    ylim = ax.get_ylim()
    ax.plot((0., 0.), (ylim[0], ylim[1]), '--', c='k', lw=2)
    ax.set_title('Outgoing weights depending on target position')
    ax.set_ylabel('$w_{out}$')
    ax.set_xlabel('Distance to source')
    cb = pylab.colorbar(m)
    cb.set_label('$v_{src}$')
#    pylab.legend()
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    pylab.show()

