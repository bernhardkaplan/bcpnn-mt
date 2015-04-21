import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import numpy as np
import utils
import matplotlib
#matplotlib.use('Agg')
import pylab
import json
from PlottingScripts.plot_spikes_sorted import plot_spikes_sorted_simple


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
              'figure.subplot.left':.12,
              'figure.subplot.bottom':.10,
              'figure.subplot.right':.92,
              'figure.subplot.top':.95,
              'figure.subplot.hspace':.30,
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

 
def shift_trace_to_stimulus_arrival(params, trace, tp_cell, mp, n, dt=1., default_value=0):
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
#    idx0 = min(n, max(0, idx_max - n / 2))
    idx0 = max(0, idx_max - n / 2)
    idx1 = idx_max + n / 2
#    idx1 = min(n, idx_max + n / 2)

    # Now, align the time axis for each cell according to stim_arrival_time
    # indices of the cut-out trace in the new shifted trace

    #new
#    idx_start = min(n, max(0, n - idx1))
#    idx_stop = max(0, min(n, n - idx0))
#    idx_start =  n - idx1
#    idx_stop = n - idx0
    idx_start =  n / 2 - (idx_max - idx0)
    idx_stop = n / 2 + (idx1 - idx_max)


#    debug_info = 'debug x=%.2f t_arrival=%.1f\t idx_max %d \t idx0 %d\tidx_max-n/2 %d\tidx1 %d\tidx_max+n/2 %d\tidx_start %d\tidx_stop %d' % (tp_cell[0], stim_arrival_time, idx_max, idx0, idx_max - n/2, idx1, idx_max + n/2, idx_start, idx_stop)
#    print debug_info
    shifted_trace = default_value * np.ones(n)
    shifted_trace[idx_start:idx_stop] = trace[idx0:idx1]
    return shifted_trace


#def select_cells_for_filtering_new(tp, mp, n_cells):
#    """
#    Return gids of n_cells within x_start and x_stop near mp.
#    """
#    return utils.get_gids_near_stim_nest(mp, tp, n=n_cells, ndim=1)[0]

#    v_tolerance = 30
#    x_width = 0.4

#    gids_1 = np.where(np.abs(tp[:, 0] - 0.5) < x_width)[0]
#    gids_2 = np.where(utils.torus_distance_array(tp[:, 4], mp[4], w=180.) < v_tolerance)[0]
#    gids = np.unique(np.r_[gids_1, gids_2])
#    idx = np.argsort(tp[gids, 4] - mp[4])[:n_cells]
#    return gids[idx] + 1

#    assert gids.size == n_cells

#    idx_sorted, distances = utils.sort_gids_by_distance_to_stimulus(tp, mp, params) # cells in indices should have the highest response to the stimulus
#    idx_sorted = np.argsort(tp[:, 4] - mp[4])[:n_cells]
#    print 'debug tp[idx_sorted, :]', tp[idx_sorted, :]
#    return idx_sorted[:n_cells]


def select_cells_for_filtering(tp, mp, n_cells):
    x_start = 0.1
    x_stop = 0.9
#    pos = np.linspace(x_start, x_stop, n_cells)
    pos = np.linspace(x_start, x_stop, 10000)
    cell_gids = []
    f = 0.1
    rnd_rotation = np.random.uniform(-10, 10, n_cells)
    for i_ in xrange(n_cells):
#        mp = [pos[i_], mp[1], mp[2], mp[3], mp[4]]
        mp = [pos[np.random.randint(0, 10000)], mp[1], mp[2], mp[3], mp[4] + rnd_rotation[i_]]
        cell_gids.append(utils.get_gids_near_stim_nest(mp, tp, n=1)[0][0])
    return np.array(cell_gids)


def select_cells_for_filtering_new(tp, mp, n_cells):
    f = 0.
    dx = 0.3
    x_start = 0.5 - dx
    x_stop = 0.5 + dx
    cell_gids = []
    cnt = 0
    n_ = 0
#    rnd_rotation = np.random.uniform(-10, 10, n_cells)
    np.random.seed(0)
    while cnt != n_cells:
#        mp = [np.random.uniform(.5 - dx, .5 + dx), mp[1], mp[2], mp[3], mp[4] + np.random.uniform(-10, 10)]
        mp = [np.random.uniform(x_start, x_stop), mp[1], mp[2], mp[3], mp[4] + np.random.uniform(-5, 5)]
        gid = utils.get_gids_near_stim_nest(mp, tp, n=1)[0][0]
        if gid not in cell_gids:
            cell_gids.append(gid)
            cnt += 1
        else:
            f += 0.001
#            dx += 0.001
        n_ += 1
        if (n_ > 10 * n_cells): #or (f > 0.3):
            cell_gids = utils.get_gids_near_stim_nest(mp, tp, n=n_cells)[0]
            print 'Not enough unique cells found, decrease n_cells or change dx!'
            break
#        print 'DEBUG f:', f, 'n:', n_
    print 'DEBUG f:', f, 'n:', n_, 'dx', dx, 'cnt', cnt
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

def filter_and_normalize_all_spiketrains(params, spiketrains, tau_filter, dt):
    normalized_traces = np.zeros((params['n_exc'], int(params['t_sim'] / dt)))
    filtered_traces = np.zeros((params['n_exc'], int(params['t_sim'] / dt)))
    # 1) get the filtered traces
    for i_ in xrange(params['n_exc']):
        t_vec, y = utils.filter_spike_train(spiketrains[i_], dt=dt, tau=tau_filter, t_max=params['t_sim'])
        normalized_traces[i_, :] = y
        filtered_traces[i_, :] = y

    # 2) normalize the traces
    for t_ in xrange(int(params['t_sim'] / dt)):
        summed_filtered_activity = normalized_traces[:, t_].sum()
        if summed_filtered_activity > 0:
            normalized_traces[:, t_] /= summed_filtered_activity

    return filtered_traces, normalized_traces


def plot_vmem_aligned(params):

    plot_params = {'backend': 'png',
                  'axes.labelsize': 20,
                  'axes.titlesize': 20,
                  'text.fontsize': 18,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'legend.pad': 0.2,     # empty space around the legend box
                  'legend.fontsize': 14,
                   'lines.markersize': 1,
                   'lines.markeredgewidth': 0.,
                   'lines.linewidth': 2,
                  'font.size': 12,
                  'path.simplify': False,
                  'figure.figsize': utils.get_figsize(800, portrait=False), 
                  'figure.subplot.left':.12,
                  'figure.subplot.bottom':.10,
                  'figure.subplot.right':.92,
                  'figure.subplot.top':.95,
                  'figure.subplot.hspace':.30,
                  'figure.subplot.wspace':.30}

    pylab.rcParams.update(plot_params)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    mp = [.0, .5, .0, .0, params['test_stim_orientation']]
    dt = params['dt_volt']
    threshold = 0.25
    v_tolerance = 10.
    coloraxis = 4
    view_range = (-300, 300)
    print 'debug view_range', view_range
    min_avg_window = .1 * (view_range[1] - view_range[0])
#    n_trace_data = int(params['t_sim'] * .8 / dt)
#    print 'n_trace_data', n_trace_data
    n_trace_data = int((view_range[1] - view_range[0]) / dt)
    print 'n_trace_data', n_trace_data
    t_vec_trace = dt * np.arange(0, n_trace_data) - n_trace_data / 2 * dt 
    motion_params_test = np.loadtxt(params['test_sequence_fn'])
    if motion_params_test.size == 5:
        motion_params_test = motion_params_test.reshape((1, 5))
    stim_idx = 0 

    # load voltage file
    all_volt_data = np.loadtxt(params['volt_folder'] + 'exc_V_m.dat')
    recorded_gids = np.array(np.unique(all_volt_data[:, 0]), dtype=np.int)
    n_cells = recorded_gids.size
    assert (recorded_gids.size > 0), 'No cells for recording V_m found, was record_v == True?'
    tp_cells = tp[recorded_gids-1, :]
    

    # for all recorded cells, select only those that have their preferred orientation within a v_tolerance
    # and calculate the average of those 
    responding_idx = np.where(np.abs(tp_cells[:, 4] - mp[4]) < v_tolerance)[0] # should respond the stimulus
    n_responding_cells = responding_idx.size
    aligned_vmem_orientation_filtered = np.zeros((responding_idx.size, n_trace_data))
    print 'Averaging over %d responding cells' % (n_responding_cells)
    print 'with tp_cells[responding_idx, :]', tp_cells[responding_idx, :]
    responding_idx = list(responding_idx) # cast to list to remove elements

    # color the voltage traces according to their preferred orientation
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

    fig = pylab.figure()
    ax0 = fig.add_subplot(111)
#    ax1 = fig.add_subplot(312)
    cnt = 0
    cnt_n = 0
    # default_trace_value  is required to fill the overlap / missing pieces of the aligned trace with a more meaningful value than 0.
    # as the time of stimulus arrival varies a lot for the recorded cells, there is some missing/overhanging piece of data that
    # disturbs the mean (if np.zeros is the default)
    # hence, simply for visual appearance the default value is calculated from the average responses of non responsive cells
    default_trace_value = -64. # set by visual inspection by looking at the mean membrane potential in the absence of any stimulus
    for i_, gid in enumerate(recorded_gids): 
        t_axis, volt = utils.extract_trace(all_volt_data, gid)
        shifted_trace = shift_trace_to_stimulus_arrival(params, volt, tp[gid-1, :], motion_params_test[stim_idx, :], n_trace_data, dt=dt, default_value=default_trace_value)
        ax0.plot(t_vec_trace, shifted_trace, c=m.to_rgba(tp[gid-1, coloraxis]), lw=1)
        if i_ in responding_idx:
            aligned_vmem_orientation_filtered[cnt, :] = shifted_trace
            responding_idx.remove(i_)
            cnt += 1

    # calculate the average Vmem response from the responding cells
    mean_vmem_response = np.ones((n_trace_data, 2))
    for t_ in xrange(n_trace_data):
        mean_vmem_response[t_, 0] = aligned_vmem_orientation_filtered[:, t_].mean()
        mean_vmem_response[t_, 1] = aligned_vmem_orientation_filtered[:, t_].std()
        mean_vmem_response[t_, 1] /= np.sqrt(n_responding_cells)

    p0 = ax0.errorbar(t_vec_trace, mean_vmem_response[:, 0], yerr=mean_vmem_response[:, 1], lw=3, c='k')
    label0 = '$\overline{V}(t)$ aligned'

    print 'debug min_avg_window:', min_avg_window
    t_anticipation, v_min, v_max = get_t_anticipation(mean_vmem_response, threshold, dt, min_avg_window)
    print 't_anticipation in vmem', t_anticipation
    label1 = 'Mean signal > %d %%' % (threshold * 100)
    ylim0 = ax0.get_ylim()
    p1, = ax0.plot((t_anticipation, t_anticipation), (ylim0[0], ylim0[1]), ':', c='k', lw=3, label=label1)
    ax0.text(t_anticipation - 100, ylim0[0] + 0.8 * (ylim0[1]-ylim0[0]), 't_anticipation = %.1f ms' % t_anticipation, fontsize=18)
    xlim = ax0.get_xlim()
    ax0.plot((xlim[0], xlim[1]), (v_min, v_min), '-', c='k', lw=1)
    ax0.plot((xlim[0], xlim[1]), (v_max, v_max), '-', c='k', lw=1)

    label2 = 'Stimulus arrival'
    p2, = ax0.plot((t_vec_trace[n_trace_data/2], t_vec_trace[n_trace_data/2]), (ylim0[0], ylim0[1]), '--', c='k', lw=3, label=label2)

    plots = [p0, p1, p2]
    labels = [label0, label1, label2]
    ax0.legend(plots, labels, loc='upper right')

    ax0.set_xlabel('Time [ms]')
    ax0.set_ylabel('Volt [mV]')
    ax0.set_xlim(view_range)
    cbar = pylab.colorbar(m,ax=ax0)
    cbar.set_label(cbar_label)
    output_fn = params['figures_folder'] + 'vmem_aligned.png'
    print 'Output fig:', output_fn
    fig.savefig(output_fn, dpi=200)

    # output data
    d = {}
    d['w_input_exc'] = params['w_input_exc']
    d['folder_name'] = params['folder_name']
    d['taui_ampa'] = params['taui_ampa']
    d['taui_nmda'] = params['taui_nmda']
    d['bcpnn_gain'] = params['bcpnn_gain']
    d['ampa_nmda_ratio'] = params['ampa_nmda_ratio']
    d['t_anticipation_vmem'] = t_anticipation
    d['n_cells'] = n_cells
    d['dt'] = dt
    d['threshold'] = threshold
    output_fn = params['data_folder'] + 'anticipation_volt_data.json'
    print 'Saving data to:', output_fn
    output_file = file(output_fn, 'w')
    json.dump(d, output_file, indent=2)


def get_t_anticipation(mean_trace, threshold, dt, min_avg_window=100):
    """
    determines t_anticipation in mean_trace
    t_anticipation is the point in time when mean_trace crosses threshold
    as measured between a minimum value (v_min) and the max within mean_trace
    v_min is determined by the average of the first min_avg_window steps
    """
    n_ = mean_trace[:, 0].size
    v_min = mean_trace[:min_avg_window, 0].mean()
    print 'debug mean_trace min:', mean_trace[:, 0].min(), 'min calculate:', v_min
    prediction_threshold = v_min + (mean_trace[:, 0].max() - v_min) * threshold
    assert (prediction_threshold > v_min), 'Can not determine a t_anticipation, because data has no maximum in the given range. Check the traces or modify min_avg_window!'
#    prediction_threshold = mean_trace[:, 0].min() + (mean_trace[:, 0].max() - mean_trace[:, 0].min()) * threshold
    idx_above_thresh = np.where(mean_trace[:, 0] > prediction_threshold)[0]
    t_anticipation = idx_above_thresh[0] * dt - n_/ 2 * dt 
    return t_anticipation, v_min, mean_trace[:, 0].max()
#    print 'idx_above thresh:', idx_above_thresh



def plot_anticipation_cmap(params):
    """
    Analyse the (filtered) spike response and export an estimate of the 
    anticipatory response to file
    """

    tau_filter = 25. # for filtering spike trains
#    tau_filter = params['bcpnn_params']['tau_p']
    threshold = 0.25  # determines t_anticipation when the average filtered spike trains cross this value (min + thresh * (max-min)) --> t_anticipation
    stim_idx = 0 
    mp = [.0, .5, .0, .0, params['test_stim_orientation']]
    dt = 1.                 # time resolution for filtered spike trains
    view_range = (-200, 200)
    min_avg_window = .1 * (view_range[1] - view_range[0])
    n_trace_data = int((view_range[1] - view_range[0]) / dt)
#    n_trace_data = int(params['t_sim'] * .6 / dt)
    t_vec_trace = dt * np.arange(0, n_trace_data) - n_trace_data / 2 * dt 

    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    # select and filter spike trains for the selected gids
    # filter cells along a trajectory (start, stop defined in the function select_cells_for_filtering)
    n_cells = 200
    filter_cells = np.unique(select_cells_for_filtering_new(tp, mp, n_cells))
    n_cells = filter_cells.size
#    filter_cells = select_cells_for_filtering(tp, mp, n_cells)
#    assert len(np.unique(filter_cells)) == n_cells, 'n unique filter_cells found: %d' % (len(np.unique(filter_cells)))

    pylab.rcParams.update(plot_params)
    all_spikes = np.loadtxt(params['exc_spiketimes_fn_merged'])
    motion_params_test = np.loadtxt(params['test_sequence_fn'])
    if motion_params_test.size == 5:
        motion_params_test = motion_params_test.reshape((1, 5))
    nspikes, spiketrains = utils.get_nspikes(all_spikes, n_cells=params['n_exc'], cell_offset=0, get_spiketrains=True, pynest=True)
    
    figsize = utils.get_figsize(1000, portrait=False)
    fig = pylab.figure(figsize=figsize)
    ax0 = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)

    ax0.set_title('Aligned response of %d cells along stimulus trajectory' % n_cells)
    ax0.set_ylabel('Filtered spiketrains')
    ax1.set_ylabel('Normalized and filtered\nspike activity')
    ax2.set_xlabel('Time to stimulus arrival [ms]')

    # colors for x-y traces
    coloraxis = 0
    if coloraxis == 0:
        value_range = (0, 1.)
        cbar_label = 'Position'
        cmap = matplotlib.cm.jet
    elif coloraxis == 4:
        value_range = (0, 180.)
        cbar_label = 'Orientation'
        cmap = matplotlib.cm.hsv

    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap) # large weights -- black, small weights -- white
    m.set_array(np.linspace(np.min(tp[:, coloraxis]), np.max(tp[:, coloraxis]), 0.01))
#    m.set_array(tp[:, coloraxis])

    filtered_aligned_spiketrains = np.zeros((n_cells, n_trace_data))
    mean_trace = np.zeros((n_trace_data, 2)) 
    filtered_traces, normalized_traces = filter_and_normalize_all_spiketrains(params, spiketrains, tau_filter, dt)

    # 3) align the filtered and normalized responses
#    fig_dbg = pylab.figure()
#    ax_dbg = fig_dbg.add_subplot(111)
    for i_, gid in enumerate(filter_cells): 
#        print 'debug spiketrains[gid].size', gid, len(spiketrains[gid]), 'tp:', tp[gid-1, :]
        shifted_trace = shift_trace_to_stimulus_arrival(params, normalized_traces[gid-1, :], tp[gid-1, :], motion_params_test[stim_idx, :], n_trace_data, dt=dt)
        filtered_aligned_spiketrains[i_, :] = shifted_trace
#        ax_dbg.plot(tp[gid-1, 0], len(spiketrains[gid]), 'o', c=m.to_rgba(tp[gid-1, 0]), ms=5)
        ax1.plot(t_vec_trace, shifted_trace, c=m.to_rgba(tp[gid-1, 0]), lw=1, alpha=0.8)
        shifted_trace_not_normalized = shift_trace_to_stimulus_arrival(params, filtered_traces[gid-1, :], tp[gid-1, :], motion_params_test[stim_idx, :], n_trace_data, dt=dt)
        ax0.plot(t_vec_trace, shifted_trace_not_normalized, c=m.to_rgba(tp[gid-1, 0]), lw=1, alpha=0.8)

    for t_ in xrange(n_trace_data):
        mean_trace[t_, 0] = filtered_aligned_spiketrains[:, t_].mean()
        mean_trace[t_, 1] = filtered_aligned_spiketrains[:, t_].std()
        mean_trace[t_, 1] /= np.sqrt(n_cells) #params['n_exc'])

#    cbar_dbg = pylab.colorbar(m, ax=ax_dbg)

    prediction_threshold = mean_trace[:, 0].min() + (mean_trace[:, 0].max() - mean_trace[:, 0].min()) * threshold
    idx_above_thresh = np.where(mean_trace[:, 0] > prediction_threshold)[0]
    t_anticipation = idx_above_thresh[0] * dt - n_trace_data / 2 * dt 
    t_anticipation, v_min, v_max = get_t_anticipation(mean_trace, threshold, dt, min_avg_window)
#    print 'idx_above thresh:', idx_above_thresh
    print 't_anticipation', t_anticipation

    p0 = ax1.errorbar(t_vec_trace, mean_trace[:, 0], yerr=mean_trace[:, 1], c='k', ls='-', lw=3)[0]
    label0 ='Mean normalized trace'

    ylim1 = ax1.get_ylim()
    plots = []
    label1 = 'Stimulus arrival'
    p1, = ax1.plot((t_vec_trace[n_trace_data/2], t_vec_trace[n_trace_data/2]), (ylim1[0], ylim1[1]), '--', c='k', lw=3, label=label1)
#    ax1.text(t_vec_trace[n_trace_data/2] + 5, ylim1[0] + 0.75 * (ylim1[1]-ylim1[0]), 'Stimulus arrival', fontsize=18)

    # plot t_anticipation
    label2 = 'Mean signal > %d %%' % (threshold * 100)
    p2, = ax1.plot((t_anticipation, t_anticipation), (ylim1[0], ylim1[1]), ':', c='k', lw=3, label=label2)
    plots = [p0, p1, p2]
    labels = [label0, label1, label2]
    ax1.text(t_anticipation - 50, ylim1[0] + 0.8 * (ylim1[1]-ylim1[0]), 't_anticipation = %.1f ms' % t_anticipation, fontsize=18)
    ax1.legend(plots, labels, loc='upper right')

    title2 = 'gain = %.2f $w^{input}_{exc}=%.1f\  R(AMPA/NMDA) = %.2f$' % (params['bcpnn_gain'], params['w_input_exc'], params['ampa_nmda_ratio'])
    title1 = 'Filtered spike trains $\\tau_i^{AMPA}=%d\ \\tau_i^{NMDA}=%d$ [ms]' % (params['taui_ampa'], params['taui_nmda'])
    ax1.set_title(title1)
    ax2.set_title(title2)
    cbar1 = pylab.colorbar(m, ax=ax1)
    cbar1.set_label(cbar_label)
    ax1.set_xlim(view_range)
    ax1.set_ylim((0., ylim1[1]))
    ax0.set_xlim(view_range)

    cbar0 = pylab.colorbar(m, ax=ax0)
    cbar0.set_label(cbar_label)

    order_of_gids = np.argsort(tp[filter_cells-1, 0])
    cax = ax2.pcolormesh(filtered_aligned_spiketrains[order_of_gids, :])
    ax2.set_xlim((n_trace_data / 2 + view_range[0] * dt, n_trace_data / 2 + view_range[1] * dt))
    ax2.set_ylim((0, n_cells))
    xticks2 = [n_trace_data / 2 + view_range[0] + i_ * 50 for i_ in xrange(6)]
    ax2.set_xticks(xticks2)
    xticks_cmap = ['%d' % (int(v) - n_trace_data/2) for v in ax2.get_xticks()]
    ax2.set_xticklabels(xticks_cmap)
    ylim2 = ax2.get_ylim()
    ax2.plot((n_trace_data / 2, n_trace_data / 2), (ylim2[0], ylim2[1]), c='k', ls='-', lw=3)
    cbar = pylab.colorbar(cax)#ax=ax2)
    cbar.set_label('Normalized filtered\nactivity')
    output_fig = params['figures_folder'] + 'anticipation_cmap_tauiAMPA_%d_tauiNMDA_%d.png' % (params['taui_ampa'], params['taui_nmda'])
    print 'Saving figure to:', output_fig
    fig.savefig(output_fig, dpi=200)
    
    # output data
    d = {}
    d['w_input_exc'] = params['w_input_exc']
    d['folder_name'] = params['folder_name']
    d['taui_ampa'] = params['taui_ampa']
    d['taui_nmda'] = params['taui_nmda']
    d['bcpnn_gain'] = params['bcpnn_gain']
    d['tau_filter'] = tau_filter
    d['ampa_nmda_ratio'] = params['ampa_nmda_ratio']
    d['t_anticipation_spikes_filtered'] = t_anticipation
    d['n_cells'] = n_cells
    d['dt'] = dt
    d['threshold'] = threshold
    output_fn = params['data_folder'] + 'anticipation_spike_data.json'
    print 'Saving data to:', output_fn
    output_file = file(output_fn, 'w')
    json.dump(d, output_file, indent=2)

    return output_fig


if __name__ == '__main__':

    if len(sys.argv) == 1:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        show = True
        plot_anticipation_cmap(params)
#        plot_vmem_aligned(params)
#        plot_anticipation(params)
    elif len(sys.argv) == 2: 
        folder_name = sys.argv[1]
        params = utils.load_params(folder_name)
        show = True
        plot_anticipation_cmap(params)
#        plot_vmem_aligned(params)
#        plot_anticipation(params)
    else:
        fig_fns = []
        for folder_name in sys.argv[1:]:
            params = utils.load_params(folder_name)
            show = False
            fig_fn = plot_anticipation_cmap(params)
            fig_fns.append(fig_fn)
#            plot_vmem_aligned(params)
#            fig_fn = plot_anticipation(params)
            fig_fns.append(fig_fn)
        print 'Figures:\n'
        for fn in fig_fns:
            print fn
    print 'Show:', show
    if show:
        pylab.show()
