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
    sum_before_blank = trace_data[:idx_start_blank].sum()
    avg_before_blank = trace_data[:idx_start_blank].mean()
    std_before_blank = trace_data[:idx_start_blank].std()
    sum_during_blank = trace_data[idx_start_blank:idx_stop_blank].sum()
    avg_during_blank = trace_data[idx_start_blank:idx_stop_blank].mean()
    std_during_blank = trace_data[idx_start_blank:idx_stop_blank].std()
    return (sum_before_blank, avg_before_blank, std_before_blank, sum_during_blank, avg_during_blank, std_during_blank)


def run_plot_currents(params, cell_type='free_vmem'):

    pylab.rcParams.update(plot_params)
    # LOAD MEASURABLES
    d = {}
    currents = ['AMPA', 'NMDA', 'GABA', 'AMPA_NEG', 'NMDA_NEG']
#    measurables = ['AMPA', 'NMDA', 'GABA', 'NMDA_NEG']
    measurables = currents + ['V_m']
#    currents = ['AMPA', 'NMDA']
    tau_syn = [params['tau_syn']['ampa'], params['tau_syn']['nmda'], params['tau_syn']['gaba'], params['tau_syn']['ampa'], params['tau_syn']['nmda']]
    for m in currents:
        fn = params['volt_folder'] + '%s_I_%s.dat' % (cell_type, m)
        print 'Loading fn:', fn
        d[m] = np.loadtxt(fn)
    d['V_m'] = np.loadtxt(params['volt_folder'] + '%s_V_m.dat' % cell_type)
    tp = np.loadtxt(params['tuning_prop_recorder_neurons_fn'])
#    rec_gids = np.array(np.unique(d['V_m'][:, 0]), dtype=int)
    rec_gids = tp[:, 5]

    feature_dim = 4
    mapped_axis = tp[:, feature_dim]
    value_range = (0., 90.)
#    value_range = (params['x_min_recorder_neurons'], params['x_max_recorder_neurons'])
    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(mapped_axis)
    colorlist= m.to_rgba(mapped_axis)

    n_plots = len(measurables)
    fig2 = pylab.figure(figsize=utils.get_figsize_A4(portrait=False))
    ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212)
    fig = pylab.figure(figsize=utils.get_figsize_A4(portrait=False))

    axes = []

    # get the typical trace length of one example cell
    idx = np.where(d[measurables[0]][:, 0] == rec_gids[0])[0]
    trace_example = d[measurables[0]][idx, 2]
    n_steps = trace_example.size
    net_currents = np.zeros((n_steps, len(rec_gids)))
    ratio_nmda_ampa_currents = np.zeros((n_steps, len(rec_gids)))

    output_data = {}
    for i_plot, measurable in enumerate(measurables):
        ax = fig.add_subplot(n_plots, 1, i_plot + 1)
        axes.append(ax)
        output_data[measurable] = {}
        output_data[measurable]['sum_before_blank'] = {}
        output_data[measurable]['avg_before_blank'] = {}
        output_data[measurable]['std_before_blank'] = {}
        output_data[measurable]['sum_during_blank'] = {}
        output_data[measurable]['avg_during_blank'] = {}
        output_data[measurable]['std_during_blank'] = {}
        output_data[measurable]['full_integral'] = {}
        for i_, gid in enumerate(rec_gids):
            idx = np.where(d[measurable][:, 0] == gid)[0]
            t_axis = np.arange(idx.size) * params['dt_volt']
            trace_data = d[measurable][idx, 2]
            (sum_before_blank, avg_before_blank, std_before_blank, sum_during_blank, avg_during_blank, std_during_blank) = get_averages(params, trace_data)
            output_data[measurable]['sum_during_blank'][gid] = sum_during_blank
            output_data[measurable]['avg_during_blank'][gid] = avg_during_blank
            output_data[measurable]['std_during_blank'][gid] = std_during_blank
            output_data[measurable]['sum_before_blank'][gid] = sum_before_blank
            output_data[measurable]['avg_before_blank'][gid] = avg_before_blank
            output_data[measurable]['std_before_blank'][gid] = std_before_blank
            output_data[measurable]['full_integral'][gid] = trace_data.sum()
            label = '$\\overline{I}_{stim}=%.1e \ \\overline{I}_{blank}=%.1e$' % (avg_before_blank, avg_during_blank)
            x_pos_cell = tp[i_, 0]
#            print 'gid: %d x_pos: %.2f  avg(%s) before blank = %.2e  during blank = %.2e' % (gid, x_pos_cell, measurable, avg_before_blank, avg_during_blank)
            ax.plot(t_axis, trace_data, c=m.to_rgba(tp[i_, feature_dim]), label=label)
#            ax.plot(t_axis, trace_data, c=colorlist[i_], label=label)
            ylim = ax.get_ylim()
            if measurable in currents:
                net_currents[:, i_] += trace_data
#            ax.text(params['t_start_blank'], ylim[0] + (ylim[1] - ylim[0]) * .8, '$\\overline{%s}=%.1e$' % (measurable, avg_before_blank))
#            ax.text(params['t_start_blank'] + 1.1 * params['t_blank'], ylim[0] + (ylim[1] - ylim[0]) * .8, '$\\overline{%s}=%.1e$' % (measurable, avg_during_blank))
        if i_plot == len(measurables) - 1:
            ax.set_xlabel('Time [ms]')

        if measurable in currents:
            ylabel = '%s\n$\\tau_{%s}=%d$' % (measurable, measurable, tau_syn[i_plot])
        else:
            ylabel = '%s' % (measurable)
        ax.set_ylabel(ylabel)
        ax.set_xlim((t_axis[0], t_axis[-1]))
#        utils.plot_blank(params, ax)
#        pylab.legend()


    n_stop_plot = int(0.7 * n_steps)
    average_ratios = np.zeros((len(rec_gids), 2))
    for i_, gid in enumerate(rec_gids):
        ax2.plot(t_axis, net_currents[:, i_], c=colorlist[i_])
        idx = np.where(d['AMPA'][:, 0] == gid)[0]
        ampa_trace = d['AMPA'][idx, 2]
        nmda_trace = d['NMDA'][idx, 2]
        nonzero_idx = np.nonzero(ampa_trace)[0]
        ratio_nmda_ampa_currents[nonzero_idx, i_] = nmda_trace[nonzero_idx] / ampa_trace[nonzero_idx]
        average_ratios[i_, 0] = ratio_nmda_ampa_currents[:n_stop_plot].mean()
        average_ratios[i_, 1] = ratio_nmda_ampa_currents[:n_stop_plot].std()
#        print 'Average ratio %d: ' % gid, average_ratios[i_, 0], '+-', average_ratios[i_, 1]
        ax3.plot(t_axis[:n_stop_plot], ratio_nmda_ampa_currents[:n_stop_plot, i_], c=colorlist[i_])

        ax3.plot((t_axis[0], t_axis[n_stop_plot]), (average_ratios[i_, 0], average_ratios[i_, 0]), '-', lw=3, c=colorlist[i_])

#    ax3.set_ylim((0, 10))
    # output data
    output_fn = params['data_folder'] + 'input_currents.json'
    print 'Writing current data to:', output_fn
    f = file(output_fn, 'w')
    json.dump(output_data, f, indent=2)

    ax2.set_xlim((t_axis[0], t_axis[-1]))
#    utils.plot_blank(params, ax2)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Net current')

    ax3.set_xlabel('Time [ms]')
    ax3.set_ylabel('Ratio NMDA/AMPA')

#    if params['with_stp']:
#        title = 'With STP $gain_{BCPNN}=%.1f $ \n $R\\frac{AMPA}{NMDA}=%.1e$' % (params['bcpnn_gain'], params['ampa_nmda_ratio'])
#    else:
#        title = 'No STP $gain_{BCPNN}=%.1f $ \n $R\\frac{AMPA}{NMDA}=%.1e$' % (params['bcpnn_gain'], params['ampa_nmda_ratio'])

    title = '$gain=%.2f\ R(\\frac{AMPA}{NMDA})=%.1e\ n_{exc}^{per MC}=%d\ p_{ee}=%.2f$ \n $w_{ei}=%.1f\ w_{ie}=%.1f\ w_{ii}=%.2f\ w_{exc}^{input}=%.1f$' % ( \
            params['bcpnn_gain'], params['ampa_nmda_ratio'], params['n_exc_per_mc'], params['p_ee_global'], params['w_ei_unspec'], params['w_ie_unspec'], params['w_ii_unspec'], params['w_input_exc'])
    axes[0].set_title(title)
    ax2.set_title(title)
    
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
        show = True
        run_plot_currents(params)
    else:
        for folder_name in sys.argv[1:]:
            params = utils.load_params(folder_name)
            show = False
            run_plot_currents(params)
    if show:
        pylab.show()