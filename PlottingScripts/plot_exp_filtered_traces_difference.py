import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import pylab
import json
import utils
import numpy as np
import json



def plot_diff_data(params1, params2, normalize=False):

    colorlist = ['k', 'r', 'b', 'g', 'm', 'y', 'c']
    # load data
    if normalize:
        data_fn1 = params1['data_folder'] + 'not_aligned_confidence_trace.dat'
        data_fn2 = params2['data_folder'] + 'not_aligned_confidence_trace.dat'
    else:
        data_fn1 = params1['data_folder'] + 'not_aligned_mean_trace.dat'
        data_fn2 = params2['data_folder'] + 'not_aligned_mean_trace.dat'
    print 'Loading data from:', data_fn1
    mean_trace1 = np.loadtxt(data_fn1)
    print 'Loading data from:', data_fn2
    mean_trace2 = np.loadtxt(data_fn2)
    # diff
    diff_data = mean_trace1 - mean_trace2
    t_axis = mean_trace1[:, -1]

    # plot not aligned data
    rcParams = { 'axes.labelsize' : 20,
                'axes.titlesize'  : 20,
                'label.fontsize': 20,
                'xtick.labelsize' : 18, 
                'ytick.labelsize' : 18, 
                'legend.fontsize': 16, 
                'figure.subplot.left':.15,
                'figure.subplot.bottom':.10,
                'figure.subplot.right':.95,
                'figure.subplot.top':.90, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)
    fig = pylab.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    n_pop = mean_trace1[0, :].size - 1
    print 'debug', mean_trace1.shape, mean_trace1[0, :] - 1
    for i_ in xrange(n_pop):
        ax1.plot(t_axis, diff_data[:, i_], c=colorlist[i_], lw=3)

    ##############################
    # plot the aligned data
    ##############################
    if normalize: 
        data_fn1 = params1['data_folder'] + 'aligned_confidence_trace.dat'
        data_fn2 = params2['data_folder'] + 'aligned_confidence_trace.dat'
    else:
        data_fn1 = params1['data_folder'] + 'aligned_aligned_mean_trace.dat'
        data_fn2 = params2['data_folder'] + 'aligned_aligned_mean_trace.dat'
    print 'Loading data from:', data_fn1
    aligned_mean_trace1 = np.loadtxt(data_fn1)
    print 'Loading data from:', data_fn2
    aligned_mean_trace2 = np.loadtxt(data_fn2)

    diff_data = aligned_mean_trace1 - aligned_mean_trace2
    for i_ in xrange(n_pop):
        ax2.plot(t_axis, diff_data[:, i_], c=colorlist[i_], lw=3)

    xticks = np.arange(-params1['t_sim'] * .5, params1['t_sim'] * .5, 200)
    ax2.set_xticklabels(xticks.astype(int))
    ax2.set_xlabel('Time [ms]')
    ax2.set_title('Response difference aligned to stimulus arrival at $\\bar{x}_i$')
    ax2.set_ylabel('Difference between \nmean filtered spiketrains')

    # plot both data files in one plot
    fig2 = pylab.figure()
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)
    for i_ in xrange(n_pop):
        ax1.plot(t_axis, mean_trace1[:, i_], c=colorlist[i_], lw=2, alpha=.4)
        ax1.plot(t_axis, mean_trace2[:, i_], '--', c=colorlist[i_], lw=2)

        ax2.plot(t_axis, aligned_mean_trace1[:, i_], c=colorlist[i_], lw=2, alpha=.4)
        ax2.plot(t_axis, aligned_mean_trace2[:, i_], '--', c=colorlist[i_], lw=2)
    ax2.set_xticklabels(xticks)


if __name__ == '__main__':
    param_pre_fn = sys.argv[1]
    if os.path.isdir(param_pre_fn):
        param_pre_fn += '/Parameters/simulation_parameters.json'
    import json
    f = file(param_pre_fn, 'r')
    print 'Loading param_preeters from', param_pre_fn
    params_pre = json.load(f)

    param_post_fn = sys.argv[2]
    if os.path.isdir(param_post_fn):
        param_post_fn += '/Parameters/simulation_parameters.json'
    import json
    f = file(param_post_fn, 'r')
    print 'Loading param_posteters from', param_post_fn
    params_post = json.load(f)

    replot_anticipation = True
    if replot_anticipation:
        cmd = 'python PlottingScripts/PlotAnticipation.py %s' % (param_pre_fn)
        print cmd
        os.system(cmd)

        print cmd
        cmd = 'python PlottingScripts/PlotAnticipation.py %s' % (param_post_fn)
        os.system(cmd)

    plot_diff_data(params_pre, params_post, normalize=True)
    pylab.show()
