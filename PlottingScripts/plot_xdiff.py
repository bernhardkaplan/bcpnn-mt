import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import simulation_parameters
import utils
from matplotlib import cm
import re
import json


def plot_xdiff(params, stim_idx=0):
    fn = params['data_folder'] + 'x-estimates_stim%d.dat' % stim_idx
    f = file(fn, 'r')
    d_ = json.load(f)
    d = np.array(d_['data'])

    time_axis = d[:, 0]
    time_bin = time_axis[1] - time_axis[0]
    x_estimate = d[:, 1]
    x_stim = d[:, 2]
    x_diff = x_stim - x_estimate
    idx_start_blank = int(params['t_start_blank'] / time_bin)
    idx_stop_blank = int((params['t_start_blank'] + params['t_blank']) / time_bin)
    xdiff_avg_before_blank = x_diff[1:idx_start_blank].mean()
    xdiff_avg_during_blank = x_diff[idx_start_blank:idx_stop_blank].mean()
    print 'average xdiff before blank:', xdiff_avg_before_blank 
    fig = pylab.figure(figsize=utils.get_figsize_A4())
    ax = fig.add_subplot(212)
    ax.plot(time_axis[1:-1], x_diff[1:-1], 'o-', ms=8, label='$x_{stim}(t) - x_{estimate} (t)$')
    ax.plot(time_axis[1:-1], xdiff_avg_before_blank * np.ones(time_axis[1:-1].size), '--', c='g', label='$|\overline{x_{diff}}|$ before blank')
    ax.plot(time_axis[1:-1], xdiff_avg_during_blank * np.ones(time_axis[1:-1].size), '-', c='r', label='$|\overline{x_{diff}}|$ during blank')
    utils.plot_blank(params, ax)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('x-prediction error')
    pylab.legend(loc='upper left')

    # find out when the perception / prediction breaks down
#    fail_tolerance = 1. * np.abs(xdiff_avg_during_blank)
    fail_tolerance = 0.10
    idx_fail = np.where(np.abs(x_diff) > fail_tolerance)[0]

    print 'idx_fail:', idx_fail, x_diff.size, time_axis.size
    print 'time idx_fail:', time_axis[idx_fail]
    for failed_idx in idx_fail:
        ax.plot(time_axis[failed_idx], x_diff[failed_idx], marker='D', c='k', markersize=15)

    # debug show 'raw' data
#    fig = pylab.figure()
    ax2 = fig.add_subplot(211)
    ax2.plot(time_axis, x_estimate, label='x-estimate', c='b')
    ax2.plot(time_axis, x_stim, label='$x_{stim}(t)$', c='k')
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('$x(t)$')

    for failed_idx in idx_fail:
        ax2.plot(time_axis[failed_idx], x_estimate[failed_idx], marker='D', c='k', markersize=15)
    utils.plot_blank(params, ax2)
    pylab.legend(loc='upper left')

    output_fn = params['figures_folder'] + 'x-prediction_error_stim%d.png' % stim_idx
    print 'Saving fig to:', output_fn
    pylab.savefig(output_fn, dpi=200)

    
#    fn = params['data_folder'] + 'xpos_grid_stim%d.dat' % stim_idx
#    f = file(fn, 'r')
#    d_ = json.load(f)
#    d = np.array(d_['data'])
#    edges = np.array(d_['edges'])
#    print 'xpos grid edges:', edges
#    for i_time in xrange(len(d)):
#        print 'xpos grid data max confidence:', i_time, np.max(d[i_time])





    
plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 12,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 14,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 3,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.18,
              'figure.subplot.bottom':.15,
              'figure.subplot.right':.90,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}


if __name__ == '__main__':
    pylab.rcParams.update(plot_params)
    if len(sys.argv) == 1:
        show = True
        print 'Case 0, loading standard parameters from simulation_parameters.py'
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        plot_xdiff(params)
    elif len(sys.argv) == 2:
        show = True
        print 'Case 1, loading parameters from', sys.argv[1]
        params = utils.load_params(sys.argv[1])
        plot_xdiff(params)
    else:
        show = False
        for folder in sys.argv[1]:
            params = utils.load_params(folder)
            plot_xdiff(params)
            
    if show:
        pylab.show()
