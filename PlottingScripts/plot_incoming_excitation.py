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
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
This script can only be run after having run PlottingScripts.plot_connection_matrix_in_tp_space.py on a training folder.

--> preferably several folders with differet tau_i

This script shows the amount of excitation stored in the files:
   column 0 - x_tgt
   column 1 - v_tgt
   column 2 - sum w_in_exc for that tau_i

e.g. TrainingSim_ClusternoSTP__1x200x1_0-200_taui200_nHC20_nMC4_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40/Data/w_in_exc_taui200.dat
"""

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


def plot_sum_w_in_exc(folder_names):

    markers = Line2D.filled_markers
    assert (len(markers) >= len(folder_names)), 'Not enough different marker symbols provided!'


    w_in = {}
    fig = pylab.figure(figsize=(12,12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    labels = []
    plots = []
    plots2 = []
    tauis = []
    for i_, folder in enumerate(folder_names):
        params = utils.load_params(folder)
        taui = params['bcpnn_params']['tau_i']
        tauis.append(taui)
        print 'taui:', taui
        fn = params['data_folder'] + 'w_in_exc_taui%d.dat' % (taui)
        assert os.path.exists(fn), '\nERROR!\n\tFile does not exist %s\n\tPlease run PlottingScripts.plot_connection_matrix_in_tp_space.py on that folder before!\n'
        print 'Loading incoming excitation from:', fn
        d = np.loadtxt(fn)
        w_in[taui] = d
        tp = np.loadtxt(params['tuning_prop_exc_fn'])
        avg_tp = utils.get_avg_tp(params, tp)
        v_range = (-1.0, 1.)
        clim = (v_range[0], v_range[1])
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
        m_v = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
        m_v.set_array(avg_tp[:, 2])
        colors_v = m_v.to_rgba(d[:, 1])
        x_range = (.0, 1.)
        clim = (x_range[0], x_range[1])
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
        m_x = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
        m_x.set_array(avg_tp[:, 0])
        colors_x = m_x.to_rgba(d[:, 0])

#        p1 = ax1.scatter(d[:, 0], d[:, 2], s=100, marker=markers[i_], linewidth=0, c=m_v.to_rgba(d[:, 1]))
#        p2 = ax2.scatter(d[:, 1], d[:, 2], s=100, marker=markers[i_], linewidth=0, c=m_x.to_rgba(d[:, 0]))
        for j_ in xrange(d[:, 0].size):
            p1, = ax1.plot(d[j_, 0], d[j_, 2], ms=10, marker=markers[i_], ls='', markeredgewidth=0, c=colors_v[j_])
            p2, = ax2.plot(d[j_, 1], d[j_, 2], ms=10, marker=markers[i_], ls='', markeredgewidth=0, c=colors_x[j_])

        labels.append('$\\tau_i=%d$ ms' % taui)
        plots.append(p1)
        plots2.append(p2)

    ax1.legend(plots, labels, scatterpoints=1, loc='upper left', ncol=1, numpoints=1)
    ax2.legend(plots2, labels, scatterpoints=1, loc='upper left', ncol=1, numpoints=1)
    # for scatter
#    ax1.legend(plots, labels, scatterpoints=1, loc='upper left', ncol=1)
#    ax2.legend(plots2, labels, scatterpoints=1, loc='upper left', ncol=1)

    ax1.xaxis.set_major_locator(MultipleLocator(.2))
    divider2 = make_axes_locatable(ax1)
    cax1 = divider2.append_axes("right", size="5%", pad=0.15)

    ax2.xaxis.set_major_locator(MultipleLocator(.2))
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.15)

    cb_v = pylab.colorbar(m_v, cax=cax1)
    cb_x = pylab.colorbar(m_x, cax=cax2)
    cb_x.set_label('$x_{tgt}$')
    cb_v.set_label('$v_{tgt}$')

    ax1.set_ylabel('Sum of incoming excitation')
    ax2.set_ylabel('Sum of incoming excitation')
    ax1.set_xlabel('Target MC position $x_j$')
    ax2.set_xlabel('Target MC speed $v_j$')

    #### new figure for ratio
    fig = pylab.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    d1 = w_in[tauis[0]]
    d2 = w_in[tauis[1]]

    ratio_win= d2[:, 2] / d1[:, 2]
#    ratio_win= d1[:, 2] / d2[:, 2]
    avg_x = .5 * (d1[:, 0] + d2[:, 0])
    avg_v = .5 * (d1[:, 1] + d2[:, 1])

    for j_ in xrange(d[:, 0].size):
        ax1.plot(avg_x[j_], ratio_win[j_], 'o', ls='', ms=8, c=colors_v[j_])
        ax2.plot(avg_v[j_], ratio_win[j_], 'o', ls='', ms=8, c=colors_x[j_])

    # average the ratio of incoming AMPA/NMDA excitation for different speeds
    speeds = [-.8, -.4, .4, .8]
    x0_idx_offset = 3 # start position for averaging, discard hypercolumns left / right of that idx depending on speed
    dv = 0.05

    mean_ratios = np.zeros(len(speeds))
    for i_, v_ in enumerate(speeds):
        valid_idx = np.where(np.abs(avg_v - v_) < dv)[0]
        v_mean = avg_v[valid_idx].mean()
        if v_ > 0.:
            ratio_mean = ratio_win[valid_idx][x0_idx_offset :].mean()
        else:
            ratio_mean = ratio_win[valid_idx][:-x0_idx_offset].mean()
        mean_ratios[i_] = ratio_mean
        print 'Cells with speed: %.1f +- %.2f have an average ratio of %.1f for incoming AMPA/NMDA weights' % (v_mean, dv, ratio_mean) 
        ax1.plot((0, 1), (ratio_mean, ratio_mean), '--', lw=3, c=m_v.to_rgba(v_mean))
    print 'Total mean ratio:', mean_ratios.mean(), ' +- ', mean_ratios.std()
    p, = ax1.plot((0, 1), (mean_ratios.mean(), mean_ratios.mean()), '-', lw=5, c='k')
    ax1.plot((0, 1), (mean_ratios.mean() + mean_ratios.std(), mean_ratios.mean() + mean_ratios.std()), '--', lw=2, c='k')
    ax1.plot((0, 1), (mean_ratios.mean() - mean_ratios.std(), mean_ratios.mean() - mean_ratios.std()), '--', lw=2, c='k')
    label='mean ratio=%.2f +- %.2f' % (mean_ratios.mean(), mean_ratios.std())
    ax1.legend([p], [label])

    avg_ratio = ratio_win.mean()
    std_ratio = ratio_win.mean()

    title = 'Ratio $\\tau_i = %d / %d $' % (tauis[1], tauis[0])
    ax1.set_title(title)
    ax1.set_ylabel('Ratio of incoming excitation')
    ax2.set_ylabel('Ratio of incoming excitation')
    ax1.set_xlabel('Target MC position $x_j$')
    ax2.set_xlabel('Target MC speed $v_j$')



if __name__ == '__main__':

#    folder_names = ['TrainingSim_ClusternoSTP__1x200x1_0-200_taui5_nHC20_nMC4_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40', \
#            'TrainingSim_ClusternoSTP__1x200x1_0-200_taui200_nHC20_nMC4_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40']

    folder_names = ['TrainingSim_ClusternoSTP__1x200x1_0-200_taui5_nHC20_nMC4_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40', \
#            'TrainingSim_Clusterv04_tuning_1x200x1_0-200_taui20_nHC20_nMC4_vmin0.4_vmax0.8', \
#            'TrainingSim_Clusterv04_tuning_1x200x1_0-200_taui50_nHC20_nMC4_vmin0.4_vmax0.8', \
#            'TrainingSim_Clusterv04_tuning_1x200x1_0-200_taui100_nHC20_nMC4_vmin0.4_vmax0.8', \
#            'TrainingSim_Clusterv04_tuning_1x200x1_0-200_taui150_nHC20_nMC4_vmin0.4_vmax0.8'
            'TrainingSim_Clusterv04_tuning_1x200x1_0-200_taui200_nHC20_nMC4_vmin0.4_vmax0.8'
            ]
#            'TrainingSim_ClusternoSTP__1x200x1_0-200_taui200_nHC20_nMC4_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40']



    plot_sum_w_in_exc(folder_names)

#    markers = ['o', '*', '^']

    pylab.show()



#    if len(sys.argv) == 1:
#        print 'Case 1: default parameters'
#        GP = simulation_parameters.parameter_storage()
#        params = GP.params
#        plot_sum_w_in_exc(params)
#        show = True
#    elif len(sys.argv) == 2:
#        params = utils.load_params(sys.argv[1])
#        plot_sum_w_in_exc(params)
#        show = True
#    else:
#        for folder in sys.argv[1:]:
#            params = utils.load_params(folder)
#            plot_sum_w_in_exc(params)
#            show = False
#    if show:
#        pylab.show()



