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
              'legend.fontsize': 10,
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


def plot_free_vmem(params, ax=None):
    if ax == None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
    fn = params['volt_folder'] + params['free_vmem_fn_base'] + '_V_m.dat'
    print 'Debug loading volt data from:', fn
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

 

def plot_anticipation(params, show=False):
    orientation = 0.
    n_cells = 5  # number of cells to be analyzed

    pylab.rcParams.update(plot_params)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    all_spikes = np.loadtxt(params['exc_spiketimes_fn_merged'])
    nspikes, spiketrains = utils.get_nspikes(all_spikes, n_cells=params['n_exc'], cell_offset=0, get_spiketrains=True, pynest=True)

    x_electrode = params['v_min_test'] * params['t_start_blank'] / 1000.
#    mp = [params['target_crf_pos'], .55, .0, .0, orientation]
    mp = [x_electrode, .50, .0, .0, orientation]

    normal_gids_nest = utils.get_gids_near_stim_nest(mp, tp, n=n_cells)[0]
#    print 'normal_gids_nest ', normal_gids_nest 
#    print 'tp (normal_gids_nest)', tp[normal_gids_nest-1, :]
    # select spike trains for the selected gids

    figsize = utils.get_figsize(1200, portrait=True)
    fig = pylab.figure(figsize=figsize)
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    plot_spikes_sorted_simple(params, sort_idx=0, ax=ax1, color_idx=4)

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

    # filter spike trains
    filtered_spiketrains = [None for i_ in xrange(n_cells)]
    for i_, gid in enumerate(normal_gids_nest):
        t_vec, y = utils.filter_spike_train(spiketrains[gid-1], dt=1., tau=30., t_max=params['t_sim'])
        filtered_spiketrains[i_] = y
        ax2.plot(t_vec, y, c=colorlist[gid-1])

    plot_free_vmem(params, ax=ax4)
    plot_voltages_near_mp(params, mp, n_cells=n_cells, ax=ax3)

    title1 = 'Protocol:%s $\\tau_i=%d$\nSpike activity' % (params['test_protocols'][0], params['bcpnn_params']['tau_i'])
    title2 = 'Filtered spike trains of selected cells $x=%.2f$ $\\theta=%.1f$' % (mp[0], mp[4])
    ax1.set_title(title1)
    ax2.set_title(title2)
#    ax2.set_xlabel('Time [ms]')

#    xlim = (0., params['t_sim'])
    xlim = (params['t_start'], params['t_sim'] - params['t_stim_pause'])
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax3.set_xlim(xlim)
    ax4.set_xlim(xlim)

#    ylim = ax1.get_ylim()
#    ylim = (ylim[0], mp[0] + .1)
#    ax1.set_ylim(ylim)

#    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.get_axes()[1]
    cbar = pylab.colorbar(m,ax=ax2)
    cbar.set_label(cbar_label)

    output_fig = params['figures_folder'] + 'filter_spiketrains.png'
    print 'Saving figure to:', output_fig
    fig.savefig(output_fig, dpi=200)
    if show:
        pylab.show()
    return output_fig


    

if __name__ == '__main__':

    if len(sys.argv) == 1:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        show = True
        plot_anticipation(params)
    elif len(sys.argv) == 2: 
        folder_name = sys.argv[1]
        params = utils.load_params(folder_name)
        show = True
        plot_anticipation(params)
    else:
        fig_fns = []
        for folder_name in sys.argv[1:]:
            params = utils.load_params(folder_name)
            show = False
            fig_fn = plot_anticipation(params)
            fig_fns.append(fig_fn)
        print 'Figures:\n'
        for fn in fig_fns:
            print fn
    if show:
        pylab.show()
