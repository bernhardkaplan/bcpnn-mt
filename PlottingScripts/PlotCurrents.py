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
               'lines.linewidth': 1,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.15,
              'figure.subplot.bottom':.08,
              'figure.subplot.right':.80,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}

def filter_tuning_prop(tp, v_range, axis=0):
    idx_low = np.where(tp[:, axis] >= v_range[0])[0]
    idx_high = np.where(tp[:, axis] <= v_range[1])[0]
    idx_joint = set(idx_low).intersection(set(idx_high))
    if len(idx_joint) == 0:
        print 'WARNING filter_tuning_prop: No values found between %.3e and %.3e' % (v_range[0], v_range[1])
    return np.array(list(idx_joint), dtype=int)



def get_averages(params, trace_data):
    idx_start_blank = int(params['t_start_blank'] / params['dt_volt'])
    idx_stop_blank = int((params['t_start_blank'] + params['t_blank']) / params['dt_volt'])
    avg_before_blank = trace_data[:idx_start_blank].mean()
    std_before_blank = trace_data[:idx_start_blank].std()
    avg_during_blank = trace_data[idx_start_blank:idx_stop_blank].mean()
    std_during_blank = trace_data[idx_start_blank:idx_stop_blank].std()
    return (avg_before_blank, std_before_blank, avg_during_blank, std_during_blank)


def run_plot_currents(params):

    pylab.rcParams.update(plot_params)
    # LOAD MEASURABLES
    d = {}
    measurables = ['AMPA', 'NMDA', 'GABA', 'AMPA_NEG', 'NMDA_NEG']
#    measurables = ['AMPA', 'NMDA']
    tau_syn = [params['tau_syn']['ampa'], params['tau_syn']['nmda'], params['tau_syn']['gaba'], params['tau_syn']['ampa'], params['tau_syn']['nmda']]
    for m in measurables:
        fn = params['volt_folder'] + 'exc_I_%s.dat' % m
        print 'Loading fn:', fn
        d[m] = np.loadtxt(fn)
    d['V_m'] = np.loadtxt(params['volt_folder'] + 'exc_V_m.dat')
    gids = np.array(np.unique(d['V_m'][:, 0]), dtype=int)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])

    # get the blank position 
    mp = np.loadtxt(params['test_sequence_fn'])
    mp = mp[params['stim_range'][0]:params['stim_range'][1], :]
#    if params['n_stim'] == 1:
#        mp = mp.reshape((1, 4))
    stim_idx = 0
    x_start_blank = params['t_start_blank'] * mp[stim_idx, 2] / params['t_stimulus'] + mp[stim_idx, 0]
    print 'x_start_blank:', x_start_blank
    dx = 0.10
#    pos_range = (x_start_blank - 2 * dx, x_start_blank + dx)
    pos_range = (0., 1.)
    # extract the gids that are 1) recorded (=gids) and 2) between x_start_blank and +dx
    idx_in_gids_in_range = filter_tuning_prop(tp[gids - 1, :], (pos_range[0], pos_range[1]))
    gids_in_range_nest = gids[idx_in_gids_in_range]
    gids_in_range = gids[idx_in_gids_in_range] - 1
    print 'gids_in_range:', gids_in_range
    print 'tp:', tp[gids_in_range, 0]

    n_plots = len(measurables)
    fig2 = pylab.figure(figsize=utils.get_figsize_A4(portrait=False))
    ax2 = fig2.add_subplot(111)
    fig = pylab.figure(figsize=utils.get_figsize_A4(portrait=False))

    #colorlist= utils.get_colorlist_sorted((pos_range[0], pos_range[1]), tp[gids_in_range - 1, 0])
    mapped_axis = tp[gids_in_range - 1, 0]
    value_range = pos_range
    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(mapped_axis)
    colorlist= m.to_rgba(mapped_axis)
    #measurable = 'V_m'
    axes = []

    # get the typical trace length of one example cell
    idx = np.where(d[measurables[0]][:, 0] == gids_in_range_nest[0])[0]
    trace_example = d[measurables[0]][idx, 1]
    n_steps = trace_example.size
    net_currents = np.zeros((n_steps, len(gids_in_range_nest)))

    for i_plot, measurable in enumerate(measurables):
        ax = fig.add_subplot(n_plots, 1, i_plot + 1)
        axes.append(ax)
        for i_, gid in enumerate(gids_in_range_nest):
            idx = np.where(d[measurable][:, 0] == gid)[0]
            t_axis = np.arange(idx.size) * params['dt_volt']
            trace_data = d[measurable][idx, 1]
            (avg_before_blank, std_before_blank, avg_during_blank, std_during_blank) = get_averages(params, trace_data)
            label = '$\\overline{I}_{stim}=%.1e \ \\overline{I}_{blank}=%.1e$' % (avg_before_blank, avg_during_blank)
            x_pos_cell = tp[gid - 1, 0]
#            print 'gid: %d x_pos: %.2f  avg(%s) before blank = %.2e  during blank = %.2e' % (gid, x_pos_cell, measurable, avg_before_blank, avg_during_blank)
            ax.plot(t_axis, d[measurable][idx, 1], c=colorlist[i_], label=label)
            ylim = ax.get_ylim()
            net_currents[:, i_] += trace_data
#            ax.text(params['t_start_blank'], ylim[0] + (ylim[1] - ylim[0]) * .8, '$\\overline{%s}=%.1e$' % (measurable, avg_before_blank))
#            ax.text(params['t_start_blank'] + 1.1 * params['t_blank'], ylim[0] + (ylim[1] - ylim[0]) * .8, '$\\overline{%s}=%.1e$' % (measurable, avg_during_blank))
        if i_plot == len(measurables) - 1:
            ax.set_xlabel('Time [ms]')
        ylabel = '%s\n$\\tau_{%s}=%d$' % (measurable, measurable, tau_syn[i_plot])
        ax.set_ylabel(ylabel)
        ax.set_xlim((t_axis[0], t_axis[-1]))
        utils.plot_blank(params, ax)
#        pylab.legend()

    for i_, gid in enumerate(gids_in_range_nest):
        ax2.plot(net_currents[:, i_], c=colorlist[i_])

    ax2.set_xlim((t_axis[0], t_axis[-1]))
    utils.plot_blank(params, ax2)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Net current')

    if params['with_stp']:
        title = 'With STP $gain_{BCPNN}=%.1f $ \n $R\\frac{AMPA}{NMDA}=%.1e$' % (params['bcpnn_gain'], params['ampa_nmda_ratio'])
    else:
        title = 'No STP $gain_{BCPNN}=%.1f $ \n $R\\frac{AMPA}{NMDA}=%.1e$' % (params['bcpnn_gain'], params['ampa_nmda_ratio'])

    axes[0].set_title(title)
    
    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.78, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(m, cax=cbar_ax)
    cbar.set_label('Cell position')

    fig2.subplots_adjust(right=0.75)
    cbar_ax2 = fig2.add_axes([0.78, 0.15, 0.05, 0.7])
    cbar2 = fig2.colorbar(m, cax=cbar_ax2)
    cbar2.set_label('Cell position')


    output_fn = params['figures_folder'] + 'currents.png'
    print 'Saving fig to:', output_fn
    fig.savefig(output_fn, dpi=200)

    output_fn = params['figures_folder'] + 'net_currents.png'
    print 'Saving fig to:', output_fn
    fig2.savefig(output_fn, dpi=200)


#    spikes = np.loadtxt(params['exc_spiketimes_fn_merged'])
#    plot_spikes_sorted(params)


#    utils.merge_connection_files(params)
#    conn_fn = params['merged_conn_list_ee']
#    conn_data = np.loadtxt(conn_fn)
    #print conn_fn
#    spid = spikes[:, 0]
#    utils.get_indices_for_gid(params, spid)

    

if __name__ == '__main__':
    if len(sys.argv) == 1:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        show = True
        run_plot_currents(params)
    elif len(sys.argv) == 2: 
        folder_name = sys.argv[1]
        params = utils.load_params(folder_name)
        show = False
        run_plot_currents(params)
    else:
        for folder_name in sys.argv[1:]:
            params = utils.load_params(folder_name)
            show = False
            run_plot_currents(params)
    if show:
        pylab.show()
