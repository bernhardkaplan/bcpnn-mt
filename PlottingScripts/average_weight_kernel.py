import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import numpy as np
import utils
import pylab
import matplotlib
from PlottingScripts.plot_spikes_sorted import plot_spikes_sorted
import json
from PlottingScripts.get_average_weights_vs_distance import get_average_weights

plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 12,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 10,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 3,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.15,
              'figure.subplot.bottom':.15,
              'figure.subplot.right':.95,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}


def plot_average_weights(params, fig=None, force_rerun=False):
                
    pylab.rcParams.update(plot_params)

    new_fig = True
    if fig == None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]
        new_fig = False

    clim = (5., 200) # = tau_i range
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(np.linspace(clim[0], clim[1], 1000))

    data_fn = params['data_folder'] + 'average_weights_vs_distance.dat'
#    if os.path.exists(data_fn) or force_rerun:
    if not os.path.exists(data_fn) or force_rerun:
        print 'Running get_average_weights...'
        get_average_weights(params)

    d = np.loadtxt(data_fn)
    xbins = d[:, 0]
    avg_weight = d[:, 1:]
    ax.plot(xbins, avg_weight[:, 0], c=m.to_rgba(params['bcpnn_params']['tau_i']))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot((xlim[0], xlim[1]), (0, 0), ls='--', lw=1, c='k')
    ax.plot((0, 0), (ylim[0], ylim[1]), ls='--', lw=1, c='k')
#    ax.errorbar(xbins, avg_weight[:, 0], yerr=avg_weight[:, 1])

    xlabel = 'Distance between source and target cell'
    ylabel = 'Weight'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, m
#    output_fig = params['figures_folder'] + 'average_weight_vs_distance_taui%d_vtrain%.2f.png' % (params['bcpnn_params']['tau_i'], params['v_min_training'])



if __name__ == '__main__':

#    folder_name = 'TrainingSim_Cluster__50x2x1_0-100_taui50_nHC20_nMC2_vtrain1.00-1.0'
    fig = None
    force_rerun = False
    for folder_name in sys.argv[1:]:
        params = utils.load_params(folder_name)
        fig, m = plot_average_weights(params, fig, force_rerun)
    cb = pylab.colorbar(m)
    cb.set_label('Window of correlation $\\tau_i$')

    output_fig = 'average_weight_vs_distance.png'
    print 'Saving figure to:', output_fig
    fig.savefig(output_fig, dpi=200)

    pylab.show()

