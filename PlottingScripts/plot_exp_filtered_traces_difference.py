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

    # get the locations from where the membrane potentials have been recorded from
    fn = params1['data_folder'] + 'locations_recorded_from.json'
    f = file(fn, 'r')
    locations = json.load(f)

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
                'legend.fontsize': 14, 
                'figure.subplot.left':.15,
                'figure.subplot.bottom':.10,
                'figure.subplot.right':.95,
                'figure.subplot.top':.85, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)
    fig = pylab.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    n_pop = mean_trace1[0, :].size - 1
    print 'debug', mean_trace1.shape, mean_trace1[0, :] - 1
    for i_ in xrange(n_pop):
        ax1.plot(t_axis, diff_data[:, i_], c=colorlist[i_], lw=3)
    ax1.set_ylabel('Not aligned difference')
    
    wsx_1, wsv_1 = params1['w_sigma_x'], params1['w_sigma_v']
    wsx_2, wsv_2 = params2['w_sigma_x'], params2['w_sigma_v']
    diff_title = 'Difference plot between '
    if normalize:
        diff_title += 'normalized traces\n'
    else:
        diff_title += 'not-normalized traces\n'
    diff_title += '$\sigma_{X}^1 = %d$' % (wsx_1)
    diff_title += '   $\sigma_{V}^1 = %d$;' % (wsv_1)
    diff_title += '\t $\sigma_{X}^2 = %d$' % (wsx_2)
    diff_title += '   $\sigma_{V}^2 = %d$' % (wsv_2)
#    diff_title += '$\sigma_{X}^1 = %.2e$' % (wsx_1)
#    diff_title += '\t $\sigma_{V}^1 = %.2e$' % (wsv_1)
#    diff_title += '\n $\sigma_{X}^2 = %.2e$' % (wsx_2)
#    diff_title += '\t $\sigma_{V}^2 = %.2e$' % (wsv_2)


    ax1.set_title(diff_title)

    ##############################
    # plot the aligned data
    ##############################
    if normalize: 
        data_fn1 = params1['data_folder'] + 'aligned_confidence_trace.dat'
        data_fn2 = params2['data_folder'] + 'aligned_confidence_trace.dat'
    else:
        data_fn1 = params1['data_folder'] + 'aligned_mean_trace.dat'
        data_fn2 = params2['data_folder'] + 'aligned_mean_trace.dat'
    print 'Loading data from:', data_fn1
    aligned_mean_trace1 = np.loadtxt(data_fn1)
    print 'Loading data from:', data_fn2
    aligned_mean_trace2 = np.loadtxt(data_fn2)

    diff_data = aligned_mean_trace1 - aligned_mean_trace2
    for i_ in xrange(n_pop):
        ax2.plot(t_axis, diff_data[:, i_], c=colorlist[i_], lw=3)

        # plot vertical line for stimulus arrival time
        t_arrive = 1000. * utils.torus_distance(locations[i_], params1['motion_params'][0]) / params1['motion_params'][2]
        plot_vertical_line(ax1, t_arrive, colorlist[i_])
#        plot_vertical_line(ax2, 0, 'grey')

#    xticks = np.arange(-params1['t_sim'] * .5, params1['t_sim'] * .5, 200)
#    ax2.set_xticklabels(xticks.astype(int))

    old_xticks = ax2.get_xticks()
    xticks = np.linspace(old_xticks[0] - .5 * old_xticks[-1], old_xticks[-1] - .5 * old_xticks[-1], old_xticks.size)
    xticks = np.array(xticks, dtype=np.int)
    ax2.set_xticklabels(xticks)
    # plot vertical bar indicating the estimated arrival time of the stimulus
#    ax2.plot((t_axis

#    ax2.set_xlabel('Time [ms]')
    ax2.set_xlabel('Time [ms] with respect to arrival at $\\bar{x}_i$')
    ax2.set_title('Response difference aligned to stimulus arrival at $\\bar{x}_i$')
    ax2.set_ylabel('Difference between \nmean filtered spiketrains')

    output_fn = 'difference_plot_wsx1_%.2e_wsv1_%.2e_wsx2_%.2e_wsv2_%.2e.png' % (wsx_1, wsv_1, wsx_2, wsv_2)
    print 'Saving difference plot to:', output_fn
    fig.savefig(output_fn, dpi=300)

    # plot both data files in one plot
    # get the information about the different traces
    assert (len(locations) == n_pop), 'Something wrong with the locations written by PlotAnticipation and the data loaded here ... '

    fig2 = pylab.figure(figsize=(14, 10))
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)
    plots, legend_txt = [], []
    for i_ in xrange(n_pop):
        p, = ax1.plot(t_axis, mean_trace1[:, i_], c=colorlist[i_], lw=2, alpha=.4)
        plots.append(p)
        p, = ax1.plot(t_axis, mean_trace2[:, i_], '--', c=colorlist[i_], lw=2)
        plots.append(p)

        p, = ax2.plot(t_axis, aligned_mean_trace1[:, i_], c=colorlist[i_], lw=2, alpha=.4)
        p, = ax2.plot(t_axis, aligned_mean_trace2[:, i_], ':', c=colorlist[i_], lw=2)

        # create the legend text / plot labels
        label_1 = '$\sigma_{X} = %d$' % (wsx_1)
        label_1 += '\t $\sigma_{V} = %d$' % (wsv_1)
        label_1 += '\t $\\bar{x} = %.2f$' % (locations[i_])
        legend_txt.append(label_1)
        label_2 = '$\sigma_{X} = %d$' % (wsx_2)
        label_2 += '\t $\sigma_{V} = %d$' % (wsv_2)
        label_2 += '\t $\\bar{x} = %.2f$' % (locations[i_])
        legend_txt.append(label_2)

        # plot vertical line for stimulus arrival time
        t_arrive = 1000. * utils.torus_distance(locations[i_], params1['motion_params'][0]) / params1['motion_params'][2]
        plot_vertical_line(ax1, t_arrive, colorlist[i_])
        plot_vertical_line(ax2, .5 * t_axis[-1], 'grey')


    ax2.set_xticklabels(xticks)
    ax1.legend(plots, legend_txt, loc='upper right')


def plot_vertical_line(ax, x_pos, color):
    ls = '--'
    y_lim = ax.get_ylim()
    ax.plot((x_pos, x_pos), (y_lim[0], y_lim[1]), ls=ls, c=color, lw=3)


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

    normalize = False
    replot_anticipation = True
    if replot_anticipation:
        cmd = 'python PlottingScripts/PlotAnticipation.py %s' % (param_pre_fn)
        print cmd
        os.system(cmd)

        print cmd
        cmd = 'python PlottingScripts/PlotAnticipation.py %s' % (param_post_fn)
        os.system(cmd)

    plot_diff_data(params_pre, params_post, normalize=normalize)
    pylab.show()
