import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import numpy as np
import utils
import pylab
import matplotlib
import json
from PlottingScripts.plot_spikes_sorted import plot_spikes_sorted_simple


"""
Anticipation data is written by PlottingScripts/PlotAnticipation plot_anticipation_cmap(params)

This script just loads the json files from params['data_folder'] + 'anticipation_data.json'
"""

plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 18,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 12,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 2,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.15,
              'figure.subplot.bottom':.15,
              'figure.subplot.right':.92,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.50,
              'figure.subplot.wspace':.30}


if __name__ == '__main__':


    all_data = []
    d = np.zeros((len(sys.argv[1:]), 6)) 

    # filter the data
    taui_nmda = 200

    #   0                   1           2       3       4                   5
    # t_anticipation    taui_ampa   taui_nmda   gain    w_input_exc     ampa_nmda_ratio
    for i_, folder_name in enumerate(sys.argv[1:]):
        params = utils.load_params(folder_name)
        fn = params['data_folder'] + 'anticipation_data.json'
        f = file(fn, 'r')
        da = json.load(f)
        all_data.append(da)
        if da['taui_nmda'] == taui_nmda: 
            d[i_, 0] = np.abs(da['t_anticipation'])
            d[i_, 1] = da['taui_ampa']
            d[i_, 2] = da['taui_nmda']
            d[i_, 3] = da['bcpnn_gain']
            d[i_, 4] = da['w_input_exc']
            d[i_, 5] = da['ampa_nmda_ratio']

    ratios = np.unique(d[:, 5])

    t_anticipation_baseline = 23 # from a run with gain=0
    d[:, 0] -= t_anticipation_baseline

    markers = ['o', 'v', 'D', '*', 's', '^', '<', 'x', '>', '+', '1']
    linestyles = ['-', '--', ':']

    pylab.rcParams.update(plot_params)
    fig = pylab.figure()

    ax = fig.add_subplot(111)
    ax.set_xlabel('gain')
    ax.set_ylabel('t_anticipation [ms]')
#    ax.set_title('Increase of t_anticipation 

#    value_range = [5, 200]

    # GAIN
    value_range = [0.5, 2.5] 
    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(np.arange(value_range[0], value_range[1], 0.1))


#    for i_ in xrange(d[:, 0].size):
#        ax.plot(d[:, 3], d[:, 0], 'o'
#    ax.scatter(d[:, 3], d[:, 0], s=20, linewidths=0, c=m.to_rgba(d[:, 2]))

    plots = []
    labels = []
    for i_, ratio in enumerate(ratios):
        idx = np.where(d[:, 5] == ratio)[0]
#        ax.scatter(d[idx, 3], d[idx, 0], marker=markers[i_], s=20, c=m.to_rgba(d[:, 5]), linewidths=0)
        ax.scatter(d[idx, 3], d[idx, 0], marker=markers[i_], s=20, linewidths=0)
#        ax.scatter(d[idx, 3], d[idx, 0], marker=markers[i_], s=20, c=m.to_rgba(d[:, 5]), linewidths=0)

#        p, = ax.scatter(d[idx, 5], d[idx, 0], ls='None', marker=markers[i_], s=5 )
#        plots.append(p)
#        labels.append('Ratio = %.2f' % ratio)

    print 'Ratios:', ratios
    ax.legend(plots, labels)
#    cbar_label = 'Tau z_i for NMDA'
    cbar_label = 'Ratio AMPA/NDMA'
    cbar1 = pylab.colorbar(m, ax=ax)
    cbar1.set_label(cbar_label)
    pylab.show()

