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


def run_plot_currents(params, cell_type='exc'):

    pylab.rcParams.update(plot_params)
    # LOAD MEASURABLES
    d = {}
    measurables = ['AMPA', 'NMDA', 'GABA', 'AMPA_NEG', 'NMDA_NEG']
    #measurables = ['AMPA', 'NMDA', 'GABA', 'NMDA_NEG']
    #measurables = ['AMPA', 'NMDA']
    tau_syn = [params['tau_syn']['ampa'], params['tau_syn']['nmda'], params['tau_syn']['gaba'], params['tau_syn']['ampa'], params['tau_syn']['nmda']]
    for m in measurables:
        fn = params['volt_folder'] + '%s_I_%s.dat' % (cell_type, m)
        print 'Loading fn:', fn
        d[m] = np.loadtxt(fn)
    d['V_m'] = np.loadtxt(params['volt_folder'] + '%s_V_m.dat' % cell_type)
    gids = np.array(np.unique(d['V_m'][:, 0]), dtype=int)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])

    # get the blank position 
    mp = np.loadtxt(params['test_sequence_fn'])
#    stim_params = np.loadtxt(params['test_sequence_fn'])
    if params['n_stim'] == 1 and params['stim_range'][0] != 0:
        mp = mp[params['stim_range'][0]:params['stim_range'][1], :]
    elif params['n_stim'] == 1 and params['stim_range'][0] == 0:
        mp = mp.reshape((1, 5))
#        mp = mp[0, :]

#    mp = mp[params['stim_range'][0]:params['stim_range'][1], :]
#    if params['n_stim'] == 1:
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

    measurables.append('V_m')
    n_plots = len(measurables)
    fig2 = pylab.figure(figsize=utils.get_figsize_A4(portrait=False))
    ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212)
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
    data_column_idx = 2
    trace_example = d[measurables[0]][idx, data_column_idx]
    n_steps = trace_example.size
    net_currents = np.zeros((n_steps, len(gids_in_range_nest)))
    ratio_nmda_ampa_currents = np.zeros((n_steps, len(gids_in_range_nest)))

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
        for i_, gid in enumerate(gids_in_range_nest):
            idx = np.where(d[measurable][:, 0] == gid)[0]
            t_axis = np.arange(idx.size) * params['dt_volt']
            trace_data = d[measurable][idx, data_column_idx]
            if params['t_blank'] > 0:
                (sum_before_blank, avg_before_blank, std_before_blank, sum_during_blank, avg_during_blank, std_during_blank) = get_averages(params, trace_data)
                output_data[measurable]['sum_during_blank'][gid] = sum_during_blank
                output_data[measurable]['avg_during_blank'][gid] = avg_during_blank
                output_data[measurable]['std_during_blank'][gid] = std_during_blank
                output_data[measurable]['sum_before_blank'][gid] = sum_before_blank
                output_data[measurable]['avg_before_blank'][gid] = avg_before_blank
                output_data[measurable]['std_before_blank'][gid] = std_before_blank
                output_data[measurable]['full_integral'][gid] = trace_data.sum()
                label = '$\\overline{I}_{stim}=%.1e \ \\overline{I}_{blank}=%.1e$' % (avg_before_blank, avg_during_blank)
                ax.plot(t_axis, d[measurable][idx, data_column_idx], c=colorlist[i_], label=label)
            else:
                ax.plot(t_axis, d[measurable][idx, data_column_idx], c=colorlist[i_])
#            x_pos_cell = tp[gid - 1, 0]
#            print 'gid: %d x_pos: %.2f  avg(%s) before blank = %.2e  during blank = %.2e' % (gid, x_pos_cell, measurable, avg_before_blank, avg_during_blank)
            ylim = ax.get_ylim()
            net_currents[:, i_] += trace_data
#            ax.text(params['t_start_blank'], ylim[0] + (ylim[1] - ylim[0]) * .8, '$\\overline{%s}=%.1e$' % (measurable, avg_before_blank))
#            ax.text(params['t_start_blank'] + 1.1 * params['t_blank'], ylim[0] + (ylim[1] - ylim[0]) * .8, '$\\overline{%s}=%.1e$' % (measurable, avg_during_blank))
        if i_plot == len(measurables) - 1:
            ax.set_xlabel('Time [ms]')
        ylabel = '%s\n$\\tau_{%s}=%d$' % (measurable, measurable, tau_syn[i_plot])
        ax.set_ylabel(ylabel)
        ax.set_xlim((t_axis[0], t_axis[-1]))
#        utils.plot_blank(params, ax)
#        pylab.legend()


    n_stop_plot = int(0.7 * n_steps)
    average_ratios = np.zeros((len(gids_in_range_nest), 2))
    sum_ampa = np.zeros(len(gids_in_range_nest))
    sum_nmda = np.zeros(len(gids_in_range_nest))
    mean_volt = np.zeros(len(gids_in_range_nest))
    I_noise =  np.zeros(len(gids_in_range_nest))
    AMPA_E_rev = 0. # since not defined otherwise in simulation_parameters it's the default of 0 mV
    # expected AMPA input from noise is calculated from V_m and g_noise (per time step)
    g_noise = params['f_noise_exc'] * params['tau_syn']['ampa'] * params['w_noise_exc'] / 1000. # / 1000. because tau_syn is in ms
    for i_, gid in enumerate(gids_in_range_nest):
        ax2.plot(net_currents[:, i_], c=colorlist[i_])
        idx = np.where(d['AMPA'][:, 0] == gid)[0]
        ampa_trace = d['AMPA'][idx, data_column_idx]
        nmda_trace = d['NMDA'][idx, data_column_idx]
        nonzero_idx = np.nonzero(ampa_trace)[0]
        ratio_nmda_ampa_currents[nonzero_idx, i_] = nmda_trace[nonzero_idx] / ampa_trace[nonzero_idx]
        average_ratios[i_, 0] = ratio_nmda_ampa_currents[:n_stop_plot].mean()
        average_ratios[i_, 1] = ratio_nmda_ampa_currents[:n_stop_plot].std()
        mean_volt[i_] = d['V_m'][idx, data_column_idx].mean()
        I_noise[i_] = g_noise * (AMPA_E_rev - mean_volt[i_])
#        I_noise[i_] = g_noise * (AMPA_E_rev - mean_volt[i_])
        sum_ampa[i_] = (ampa_trace - I_noise[i_]).sum()
        sum_nmda[i_] = nmda_trace.sum()
#        print 'Average ratio %d: ' % gid, average_ratios[i_, 0], '+-', average_ratios[i_, 1]
        ax3.plot(t_axis[:n_stop_plot], ratio_nmda_ampa_currents[:n_stop_plot, i_], c=colorlist[i_])
        ax3.plot((t_axis[0], t_axis[n_stop_plot]), (average_ratios[i_, 0], average_ratios[i_, 0]), '-', lw=3, c=colorlist[i_])

    sum_ampa /= sum_ampa.size
    sum_nmda /= sum_nmda.size
    for i_, gid in enumerate(gids_in_range_nest):
        print 'gid %d\tAMPA in sum: %.3e\tNMDA %.3e\tI_noise=%.3e\tV_mean=%.1f' % (gid, sum_ampa[i_].sum(), sum_nmda[i_].sum(), I_noise[i_], mean_volt[i_])

    I_noise_mean = I_noise.mean()
    print 'Average I_noise: %.1f pA' % I_noise_mean
    fig_bar_sum = pylab.figure()
    ax_ampa_bar = fig_bar_sum.add_subplot(311)
    ax_nmda_bar = fig_bar_sum.add_subplot(312)
    ax_ratio_nmda_ampa = fig_bar_sum.add_subplot(313)
    ax_ampa_bar.bar(range(len(gids_in_range_nest)), sum_ampa, width=1)
    ax_nmda_bar.bar(range(len(gids_in_range_nest)), sum_nmda, width=1)
    ax_ampa_bar.set_ylabel('Incoming AMPA currents')
    ax_nmda_bar.set_ylabel('Incoming NMDA currents')
#    ax_ampa_bar.set_xlabel('Cells')
    ax_ampa_bar.plot((0, len(gids_in_range_nest)), (sum_ampa.mean(), sum_ampa.mean()), lw=2, ls='--', c='k', label='Average sum of incoming AMPA without noise')
    ax_nmda_bar.plot((0, len(gids_in_range_nest)), (sum_nmda.mean(), sum_nmda.mean()), lw=2, ls='--', c='k', label='Average sum of incoming NMDA')

    non_zero_idx = np.nonzero(sum_ampa)[0]
    ax_ratio_nmda_ampa.bar(range(len(gids_in_range_nest)), sum_nmda[non_zero_idx] / sum_ampa[non_zero_idx], width=1)
    ax_ratio_nmda_ampa.set_title('Ratio $\\frac{I_{network}^{NMDA}}{I^{AMPA}_{input} + I_{network}^{AMPA}}$')
    ax_ratio_nmda_ampa.set_xlabel('Cells')
    fig_bar_sum.subplots_adjust(left=0.20, hspace=0.6, bottom=0.15)

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
