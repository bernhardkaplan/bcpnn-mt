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


plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 12,
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
              'figure.subplot.right':.90,
              'figure.subplot.top':.85,
              'figure.subplot.hspace':.50,
              'figure.subplot.wspace':.30}

#def filter_tuning_prop(tp, v_range, axis=0):
#    idx_low = np.where(tp[:, axis] >= v_range[0])[0]
#    idx_high = np.where(tp[:, axis] <= v_range[1])[0]
#    idx_joint = set(idx_low).intersection(set(idx_high))
#    if len(idx_joint) == 0:
#        print 'WARNING filter_tuning_prop: No values found between %.3e and %.3e' % (v_range[0], v_range[1])
#    return np.array(list(idx_joint), dtype=int)


def recompute_input(params, tp, gids):
    """
    Returns the envelope of the poisson process that is fed into the cells with gids
    Keyword arguments:
    params -- parameter dictionary
    tp -- tuning property array of all cells
    gids -- array or list of cell gids
    """
    dt = params['dt_rate'] * 10# [ms] time step for the non-homogenous Poisson process
    time = np.arange(0, params['t_sim'], dt)
    n_cells = len(gids)
    L_input = np.zeros((n_cells, time.shape[0]))
    L_avg = np.zeros(time.shape[0])
    L_std = np.zeros(time.shape[0])
    for i_time, time_ in enumerate(time):
        L_input[:, i_time] = utils.get_input(tp[gids, :], params, time_/params['t_stimulus'], motion=params['motion_type'])
        L_avg[i_time] = L_input[:, i_time].mean()
        L_std[i_time] = L_input[:, i_time].std()
    return L_avg, L_std, time


def plot_voltages_near_mp(params, mp, n_cells=1, ax=None):
    if ax == None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)

    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    all_volt_data = np.loadtxt(params['volt_folder'] + 'exc_V_m.dat')
    recorded_gids = np.array(np.unique(all_volt_data[:, 0]), dtype=np.int)
    assert recorded_gids.size > 0
    
    tp_cells = tp[recorded_gids-1, :]
    idx_in_gids = utils.get_gids_near_stim_nest(mp, tp_cells, n=n_cells, ndim=1)[0] - 1
    gids_near_mp = recorded_gids[idx_in_gids]

    coloraxis = 0
    if coloraxis == 0:
        value_range = (0, 1.)
        cbar_label = 'Position'
    elif coloraxis == 4:
        value_range = (0, 180.)
        cbar_label = 'Orientation'
    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(tp[:, coloraxis])
    colorlist = m.to_rgba(tp[:, coloraxis])
    
    for i_, gid in enumerate(gids_near_mp):
        t_axis, volt = utils.extract_trace(all_volt_data, gid)
        ax.plot(t_axis, volt, c=colorlist[gid-1])

#    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Volt [mV]')
    cbar = pylab.colorbar(m,ax=ax)
    cbar.set_label(cbar_label)
    return gids_near_mp

def plot_free_vmem(params, ax=None):
    if ax == None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
    fn = params['volt_folder'] + params['free_vmem_fn_base'] + '_V_m.dat'
    volt_data = np.loadtxt(fn)
    tp = np.loadtxt(params['tuning_prop_recorder_neurons_fn'])
    recorder_gids = np.unique(volt_data[:, 0])

    coloraxis = 0
    if coloraxis == 0:
        value_range = (0, 1.)
        cbar_label = 'Position'
    elif coloraxis == 4:
        value_range = (0, 180.)
        cbar_label = 'Orientation'
    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(np.linspace(value_range[0], value_range[1], 0.01))

#    print 'debug recorder_gids:', recorder_gids
    for i_, gid in enumerate(recorder_gids):
        idx = np.where(tp[:, 5] == gid)[0][0]
        assert idx.size == 1
        tp_cell = tp[idx, :]
        print 'Plotting free mem with tp:', tp_cell[0], tp_cell[4]
        t_axis, volt = utils.extract_trace(volt_data, gid)
        ax.plot(t_axis, volt, c=m.to_rgba(tp_cell[coloraxis]))

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Volt [mV]')
    cbar = pylab.colorbar(m,ax=ax)
    cbar.set_label(cbar_label)

 
def shift_trace_to_stimulus_arrival(params, trace, tp_cell, mp, n, dt=1.):
    """
    Align the trace so that the stimulus arrival is in the middle of the trace
    and fill the trace with 0s accordingly (before and/or after)
    trace -- original (unaligned) trace
    n -- target size of the trace to be returned
    tp_cell -- tuning properties of the cell corresponding to the trace as 5-dim tuple 
    mp -- motion params (5-dim tuple)
    """
    x_cells = tp_cell[0]
    stim_arrival_time = (tp_cell[0] - mp[0]) / params['v_min_test'] * 1000.
    idx_max = int(stim_arrival_time / dt) # expected maximum response
    # cut the original trace, compute border indices in original trace
    idx0 = max(0, idx_max - n / 2)
    idx1 = min(n, idx_max + n / 2)
    # Now, align the time axis for each cell according to stim_arrival_time
    # indices of the cut-out trace in the new shifted trace
    idx_start = n - idx1
    idx_stop = n - idx0 
#    debug_info = 'debug x=%.2f t_arrival=%.1f\t idx_max %d \t idx0 %d idx_max-n/2 %d\tidx1 %d idx_max+n/2 %d\tidx_start %d\tidx_stop %d' % (tp_cell[0], stim_arrival_time, idx_max, idx0, idx_max - n/2, idx1, idx_max + n/2, idx_start, idx_stop)
#    print debug_info
    shifted_trace = np.zeros(n)
    shifted_trace[idx_start:idx_stop] = trace[idx0:idx1]
    return shifted_trace


def select_cells_for_filtering(tp, mp, n_cells):
    x_start = 0.2
    x_stop = 0.8
    pos = np.linspace(x_start, x_stop, n_cells)
    cell_gids = []
    for i_ in xrange(n_cells):
        mp = [pos[i_], mp[1], mp[2], mp[3], mp[4]]
        cell_gids.append(utils.get_gids_near_stim_nest(mp, tp, n=1)[0][0])
    return np.array(cell_gids)



def plot_anticipation(params, show=False, n_cell=5):
    stim_idx = 0
    orientation = 0.
#    n_cells = params['n_mc']  # number of cells to be analyzed
    n_cells = 10  # number of cells to be analyzed with spiketrain filtering

    pylab.rcParams.update(plot_params)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    all_spikes = np.loadtxt(params['exc_spiketimes_fn_merged'])
    motion_params_test = np.loadtxt(params['test_sequence_fn'])
    if motion_params_test.size == 5:
        motion_params_test = motion_params_test.reshape((1, 5))
    nspikes, spiketrains = utils.get_nspikes(all_spikes, n_cells=params['n_exc'], cell_offset=0, get_spiketrains=True, pynest=True)

    # select cells based on tuning properties for spike train filtering
    x_electrode = params['v_min_test'] * params['t_start_blank'] / 1000. #+ 0.05
#    mp = [params['target_crf_pos'], .55, .0, .0, orientation]
    mp = [x_electrode, .50, .0, .0, orientation]

    normal_gids_nest = utils.get_gids_near_stim_nest(mp, tp, n=n_cells)[0]
    print 'normal_gids_nest ', normal_gids_nest 
    print 'tp (normal_gids_nest): \n', tp[normal_gids_nest-1, :]

    figsize = utils.get_figsize(1200, portrait=True)
    fig = pylab.figure(figsize=figsize)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    plot_spikes_sorted_simple(params, sort_idx=0, ax=ax1, color_idx=4, set_xlabel=False)
    gids_rec_v = plot_voltages_near_mp(params, mp, n_cells=n_cells, ax=ax2)
    plot_free_vmem(params, ax=ax3)

    coloraxis = 0
    if coloraxis == 0:
        value_range = (0, 1.)
        cbar_label = 'Position'
    elif coloraxis == 4:
        value_range = (0, 180.)
        cbar_label = 'Orientation'
    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(tp[:, coloraxis])
    colorlist = m.to_rgba(tp[:, coloraxis])

    title1 = 'Protocol:%s $\\tau_i^{AMPA}=%d\ \\tau_i^{NMDA}=%d$\nSpike activity' % (params['test_protocols'][0], params['taui_ampa'], params['taui_nmda'])
    ax1.set_title(title1)

    # set xlim
    xlim = (params['t_start'], params['t_sim'] - params['t_stim_pause'])
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax3.set_xlim(xlim)

    # set ylim
    ax2.set_ylim((-65., -45.))
#    ylim = ax1.get_ylim()
#    ylim = (ylim[0], mp[0] + .1)
#    ax1.set_ylim(ylim)

    output_fig = params['figures_folder'] + 'spikes_and_voltages_tauiAMPA_%d_tauiNMDA_%d.png' % (params['taui_ampa'], params['taui_nmda'])
    print 'Saving figure to:', output_fig
    fig.savefig(output_fig, dpi=200)
    if show:
        pylab.show()

    return output_fig


def plot_anticipation_cmap(params):

    tau_filter = 20. # for filtering spike trains
    threshold = 0.1  # determines t_anticipation when the average filtered spike trains cross this value (min + thresh * (max-min)) --> t_anticipation
    stim_idx = 0 
    pylab.rcParams.update(plot_params)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    all_spikes = np.loadtxt(params['exc_spiketimes_fn_merged'])
    motion_params_test = np.loadtxt(params['test_sequence_fn'])
    if motion_params_test.size == 5:
        motion_params_test = motion_params_test.reshape((1, 5))
    nspikes, spiketrains = utils.get_nspikes(all_spikes, n_cells=params['n_exc'], cell_offset=0, get_spiketrains=True, pynest=True)
    
    figsize = utils.get_figsize(1200, portrait=False)
    fig = pylab.figure(figsize=figsize)
    ax0 = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)

    # colors for x-y traces
    coloraxis = 0
    if coloraxis == 0:
        value_range = (0, 1.)
        cbar_label = 'Position'
    elif coloraxis == 4:
        value_range = (0, 180.)
        cbar_label = 'Orientation'

    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(tp[:, coloraxis])
#    colorlist = m.to_rgba(tp[:, coloraxis])

    # select spike trains for the selected gids
    # filter spike trains
    dt = 2. # time resolution for filtered spike trains
    n_trace_data = int(params['t_sim'] * .6 / dt)
    print 'n_trace_data:', n_trace_data
    t_vec_trace = dt * np.arange(0, n_trace_data) - n_trace_data / 2 * dt 

    # filter cells along a trajectory (start, stop defined in select_cells_for_filtering)
    mp = [.0, .5, .0, .0, params['test_stim_orientation']]
    n_cells = 50
    filter_cells = select_cells_for_filtering(tp, mp, n_cells)
    print 'filter_cells:', filter_cells

    normalized_traces = np.zeros((n_cells, int(params['t_sim'] / dt)))
    filtered_traces = np.zeros((n_cells, int(params['t_sim'] / dt)))
    filtered_aligned_spiketrains = np.zeros((n_cells, n_trace_data))
    mean_trace = np.zeros((n_trace_data, 2))

    # 1) get the filtered traces
#    for i_ in xrange(filter_cells): #params['n_exc']):
    for i_, gid in enumerate(filter_cells): #params['n_exc']):
        t_vec, y = utils.filter_spike_train(spiketrains[gid-1], dt=dt, tau=tau_filter, t_max=params['t_sim'])
        normalized_traces[i_, :] = y
        filtered_traces[i_, :] = y

#    print 'Normalizing spike trains ...'
    # 2) normalize the traces
#    for t_ in xrange(n_trace_data):
    for t_ in xrange(int(params['t_sim'] / dt)):
        summed_filtered_activity = normalized_traces[:, t_].sum()
#        print 'debug t_ summed_filtered_activity:', t_, summed_filtered_activity
        if summed_filtered_activity > 0:
            normalized_traces[:, t_] /= summed_filtered_activity

    # 3) align the filtered and normalized responses
#    for i_ in xrange(params['n_exc']):
    for i_, gid in enumerate(filter_cells): #params['n_exc']):
        shifted_trace = shift_trace_to_stimulus_arrival(params, normalized_traces[i_, :], tp[gid-1, :], motion_params_test[stim_idx, :], n_trace_data, dt=dt)
        filtered_aligned_spiketrains[i_, :] = shifted_trace
        ax1.plot(t_vec_trace, shifted_trace, c=m.to_rgba(tp[gid-1, 0]), lw=1)
        shifted_trace_not_normalized = shift_trace_to_stimulus_arrival(params, filtered_traces[i_, :], tp[gid-1, :], motion_params_test[stim_idx, :], n_trace_data, dt=dt)
        ax0.plot(t_vec_trace, shifted_trace_not_normalized, c=m.to_rgba(tp[gid-1, 0]), lw=1)

    for t_ in xrange(n_trace_data):
        mean_trace[t_, 0] = filtered_aligned_spiketrains[:, t_].mean()
        mean_trace[t_, 1] = filtered_aligned_spiketrains[:, t_].std()
        mean_trace[t_, 1] /= np.sqrt(params['n_exc'])


    prediction_threshold = mean_trace[:, 0].min() + (mean_trace[:, 0].max() - mean_trace[:, 0].min()) * threshold
    idx_above_thresh = np.where(mean_trace[:, 0] > prediction_threshold)[0]
    t_prediction = idx_above_thresh[0] * dt - n_trace_data / 2 * dt 
#    print 'idx_above thresh:', idx_above_thresh
    print 't_prediction', t_prediction

    ax1.errorbar(t_vec_trace, mean_trace[:, 0], yerr=mean_trace[:, 1], c='k', lw=3)

    ylim = ax1.get_ylim()
    plots = []
    label1 = 'Stimulus arrival'
    p1, = ax1.plot((t_vec_trace[n_trace_data/2], t_vec_trace[n_trace_data/2]), (ylim[0], ylim[1]), '--', c='k', lw=3, label=label1)
    ax1.text(t_vec_trace[n_trace_data/2] + 5, ylim[0] + 0.8 * (ylim[1]-ylim[0]), 'Stimulus arrival')

    # plot t_prediction
    label2 = 'Mean anticipation signal'
    p2, = ax1.plot((t_prediction, t_prediction), (ylim[0], ylim[1]), ':', c='k', lw=3, label=label2)
    plots = [p1, p2]
    labels = [label1, label2]
    ax1.text(t_prediction + 5, ylim[0] + 0.9 * (ylim[1]-ylim[0]), 't_anticipation = %.1f ms' % t_prediction)
    ax1.legend(plots, labels, loc='upper right')

    ax1.set_xlabel('Time [ms]')
    title_cmap = 'Filtered spike trains $\\tau_i^{AMPA}=%d\ \\tau_i^{NMDA}=%d$ [ms]' % (params['taui_ampa'], params['taui_nmda'])
    ax1.set_title(title_cmap)
    cbar1 = pylab.colorbar(m, ax=ax1)
    cbar1.set_label(cbar_label)
    ax1.set_xlim((-250, 250))
    ax1.set_ylim((0., 1.))
    ax0.set_xlim((-250, 250))

    cbar0 = pylab.colorbar(m, ax=ax0)
    cbar0.set_label(cbar_label)
    ax0.set_title('Aligned response of %d cells\nalong stimulus trajectory' % n_cells)
    ax0.set_ylabel('Filtered spike activity')
    ax1.set_ylabel('Normalized filtered spike activity')

    order_of_gids = np.argsort(tp[filter_cells-1, 0])
    cax = ax2.pcolormesh(filtered_aligned_spiketrains[order_of_gids, :])
    ax2.set_xlim((0, n_trace_data))
    ax2.set_ylim((0, n_cells))
    ylim2 = ax2.get_ylim()
    ax2.plot((n_trace_data / 2, n_trace_data / 2), (ylim2[0], ylim2[1]), c='w', ls='--', lw=3)
    cbar = pylab.colorbar(cax)
    cbar.set_label('Normalized filtered\nactivity')
    output_fig = params['figures_folder'] + 'anticipation_cmap_tauiAMPA_%d_tauiNMDA_%d.png' % (params['taui_ampa'], params['taui_nmda'])
    print 'Saving figure to:', output_fig
    fig.savefig(output_fig, dpi=200)
    return output_fig


if __name__ == '__main__':

    if len(sys.argv) == 1:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        show = True
        plot_anticipation_cmap(params)
#        plot_anticipation(params)
    elif len(sys.argv) == 2: 
        folder_name = sys.argv[1]
        params = utils.load_params(folder_name)
        show = True
        plot_anticipation_cmap(params)
#        plot_anticipation(params)
    else:
        fig_fns = []
        for folder_name in sys.argv[1:]:
            params = utils.load_params(folder_name)
            show = False
            fig_fn = plot_anticipation_cmap(params)
#            fig_fn = plot_anticipation(params)
            fig_fns.append(fig_fn)
        print 'Figures:\n'
        for fn in fig_fns:
            print fn
    if show:
        pylab.show()
