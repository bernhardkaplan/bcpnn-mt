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

plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
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
              'figure.subplot.bottom':.13,
              'figure.subplot.right':.90,
              'figure.subplot.top':.88,
              'figure.subplot.hspace':.36,
              'figure.subplot.wspace':.30}


def plot_rates(params):
    time_binsize = 100. # [ms]

    pylab.rcParams.update(plot_params)
    data_fn = params['exc_spiketimes_fn_merged']
    if os.path.getsize(data_fn) == 0:
        print '\nWARNING!\nNO SPIKES WERE FOUND!'
    tuning_prop = np.loadtxt(params['tuning_prop_exc_fn'])
    mp = np.loadtxt(params['training_stimuli_fn'])
    sorted_indices_x = tuning_prop[:, 0].argsort()
    if os.path.exists(params['stim_durations_fn']):
        stim_duration = np.loadtxt(params['stim_durations_fn'])
        params['t_sim'] = stim_duration.sum() # otherwise the default t_sim is taken, which is likely wrong! 
    n_bins = int((params['t_sim'] / time_binsize) )
    nspikes_binned = np.zeros((params['n_exc'], n_bins))             # binned activity over time
    nspikes, spiketrains = utils.get_nspikes(data_fn, get_spiketrains=True, n_cells=params['n_exc'])
    nspikes_mc = np.zeros((params['n_mc'], n_bins))
    avg_tp = utils.get_avg_tp(params, tuning_prop)
    t_axis = np.arange(0, n_bins * time_binsize, time_binsize)
    plot_rates = False
    if plot_rates:
        rate_factor = 1000. / time_binsize 
    else:
        rate_factor = 1.

    clim = (-params['v_max_tp'], params['v_max_tp'])
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.seismic) # large weights -- black, small weights -- white
    m.set_array(np.arange(tuning_prop[:, 2].min(), tuning_prop[:, 2].max(), 0.01))
    colorlist = m.to_rgba(tuning_prop[:, 2])
    colorlist_mc = m.to_rgba(avg_tp[:, 2])

    fig = pylab.figure(figsize=utils.get_figsize_A4(portrait=False))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    for gid in xrange(params['n_exc']):
        if (nspikes[gid] > 0):
            count, bins = np.histogram(spiketrains[gid], bins=n_bins, range=(0, params['t_sim']))
            nspikes_binned[gid, :] = count
            ax1.plot(t_axis, rate_factor * count, 'o-', c=colorlist[gid])
            mc_idx = utils.get_mc_index_for_gid(params, gid + 1)
            nspikes_mc[mc_idx, :] += count
    cb = fig.colorbar(m)
    cb.set_label('$v_{src}$')

    for mc_idx in xrange(params['n_mc']):
        ax2.plot(t_axis, nspikes_mc[mc_idx, :] * rate_factor / params['n_exc_per_mc'], 'o-', c=colorlist_mc[mc_idx])
        print 'MC: %d v=%.2f fires in total: %d spikes' % (mc_idx, avg_tp[mc_idx, 2], nspikes_mc[mc_idx, :].sum())

#    ax1.set_xlabel('Time [ms]')
    ax2.set_xlabel('Time [ms]')
    if plot_rates:
        ax1.set_title('Rate[Hz] vs. Time [ms]')
    else:
        ax1.set_title('Number of spikes vs. Time [ms]')
    ax1.set_ylabel('Cell activity')
    ax2.set_ylabel('MC avg activity')

#    mc_idx_sorted_v = np.argsort(avg_tp[:, 2])
#    fig2 = pylab.figure()
    ax = fig.add_subplot(122)
    colors = m.to_rgba(tuning_prop[:, 2])
    ax.scatter(tuning_prop[:, 2], nspikes, c=colors, linewidths=0)
    ax.set_xlabel('$v_i$')
    ax.set_ylabel('Number of spikes fired')
    ax.set_title('Cell activity during training \n(integrated over all training)')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        print 'Case 1'
        params = utils.load_params(sys.argv[1])
    else:
        network_params = simulation_parameters.parameter_storage()  
        params = network_params.params

    plot_rates(params)
    pylab.show()
