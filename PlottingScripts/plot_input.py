import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import utils
import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import sys
import os
import json

rcP= { 'axes.labelsize' : 18,
            'label.fontsize': 18,
            'xtick.labelsize' : 18, 
            'ytick.labelsize' : 18, 
            'axes.titlesize'  : 20,
            'figure.subplot.left':.15,
            'figure.subplot.bottom':.15,
            'figure.subplot.right':.90,
            'figure.subplot.top':.90,
            'figure.subplot.hspace':.15,
            'figure.subplot.wspace':.15, 
            'legend.fontsize': 9}

pylab.rcParams.update(rcP)

def plot_input_old(params):
    rate_fn = params['input_rate_fn_base'] + '%d_%d.dat' % (gid, stim_idx)
    spike_fn = params['input_st_fn_base']  + '%d_%d.dat' % (gid, stim_idx)
    print 'Loading data from', rate_fn, '\n', spike_fn
    print 'debug', params['figures_folder']

    #else:
    #    info = "\n\tuse:\n \
    #    \t\tpython plot_input.py   [RATE_ENVELOPE_FILE]  [SPIKE_INPUT_FILE]\n \
    #    \tor: \n\
    #    \t\tpython plot_input.py [gid of the cell to plot]" 
    #    print info


    rate = np.loadtxt(rate_fn)
    #rate /= np.max(rate)
    y_min = rate.min()
    y_max = rate.max()

    spikes = np.loadtxt(spike_fn) # spikedata

    #spikes *= 10. # because rate(t) = L(t) was created with a stepsize of .1 ms

    binsize = 50
    n_bins = int(round(params['t_sim'] / binsize))
    n, bins = np.histogram(spikes, bins=n_bins, range=(0, params['t_sim']))
    print 'n, bins', n, 'total', np.sum(n), 'binsize:', binsize

    fig = pylab.figure()
    pylab.subplots_adjust(bottom=.10, left=.12, hspace=.15, top=0.94)#55)
    ax = fig.add_subplot(211)

    nspikes = spikes.size
    w_input_exc = 2e-3
    cond_in = w_input_exc * 1000. * nspikes
    print 'Cond_in: %.3e [nS] nspikes: %d' % (cond_in, nspikes)
    ax.set_title('Input spike train and L(t)')
    for s in spikes:
        ax.plot((s, s), (0.2 * (y_max-y_min) + y_min, 0.8 * (y_max-y_min) + y_min), c='k')
    #rate_half = .5 * (np.max(rate) - np.min(rate))
    #ax.plot(spikes, rate_half * np.ones(spikes.size), '|', markersize=1)
    #ax.plot(spikes, 0.5 * np.ones(spikes.size), '|', markersize=1)
    print 'rate', rate

    n_steps = int(round(1. / params['dt_rate']))
    rate = rate[::n_steps] # ::10 because dt for rate creation was 0.1 ms
    ax.plot(np.arange(rate.size), rate, label='Cond_in = %.3e nS' % cond_in, lw=2, c='b')
    #ax.set_xlabel('Time [ms]')
    ax.set_xticks([])

    ax.set_ylabel('Input rate (t) [kHz]')
    def set_yticks(ax, n_ticks=5, endpoint=False):
        ylim = ax.get_ylim()
        ticks = np.linspace(ylim[0], ylim[1], n_ticks, endpoint=endpoint)
        ax.set_yticks(ticks)
        ax.set_yticklabels(['%d' % i for i in ticks])

    set_yticks(ax, 5)
    #ax.set_yticklabels(['', '.7', '1.4', '2.1', '2.8'])

    #ax.legend()
    ax = fig.add_subplot(212)
    ax.bar(bins[:-1], n, width= bins[1] - bins[0])
    #ax.set_title('Binned input spike train, binsize=%.1f ms' % binsize)

    ax.set_xlim((0, params['t_sim']))
    ax.set_ylabel('Num input spikes')
    ax.set_xlabel('Time [ms]')
    #ylabels = ax.get_yticklabels()
    set_yticks(ax, 4)

    output_fn = params['figures_folder'] + 'input_%d.png' % (gid)
    print 'Saving to', output_fn
    pylab.savefig(output_fn, dpi=200)

    #output_fn = 'delme.dat'
    #np.savetxt(output_fn, data)
    pylab.show()


def get_cell_gid(params, tp_params, stim_range, n_cells=1):
    tp = np.loadtxt(params['tuning_prop_exc_fn'])

    gids_found = False
    gids, dist = utils.get_gids_near_stim_nest(tp_params, tp, n=n_cells)
    print 'debug gids dist', gids, dist, 'tp_params:', tp_params
    print 'tp[gids]', tp[gids, :]
    cnt_ = np.zeros(n_cells)
    for i_, gid in enumerate(gids):
        for stim_idx in stim_range:
            rate_fn = params['input_rate_fn_base'] + '%d_%d.dat' % (gid, stim_idx)
            if os.path.exists(rate_fn):
                cnt_[i_] += 1
    return gids[np.argsort(cnt_)[-n_cells:]]


def get_cell_gids_with_input_near_tp(params, tp_params, stim_range, n_cells=1):

    tp = np.loadtxt(params['tuning_prop_exc_fn'])

    gids, dist = utils.get_gids_near_stim_nest(tp_params, tp, n=params['n_exc'])

    idx = np.argsort(dist)
    print 'gids', gids[idx]
    print 'dist', dist[idx]
    gids_to_return = []
    for gid in gids[idx]:
        n_input_files_for_cell = 0
        for stim_idx in stim_range:
            rate_fn = params['input_rate_fn_base'] + '%d_%d.dat' % (gid, stim_idx)
            if os.path.exists(rate_fn):
                gids_to_return.append(gid)
                break

    print 'gids_to_return:', gids_to_return[:n_cells]
    return gids_to_return[:n_cells]



if __name__ == '__main__':

    if len(sys.argv) == 2:
        params = utils.load_params(sys.argv[1])
    else:
        print 'Using standard params'
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
    input_folder_exists = os.path.exists(params['input_folder'])
    if not input_folder_exists:
        print 'ERROR. input folder does not exist. plot_input quit'

    tp_params = (0.1, 0.5, 1.0, 0.)
    n_cells = 10
    ms = 4

    if not os.path.exists(params['exc_spiketimes_fn_merged']):
        print 'Merging spike files...', 
        utils.merge_spike_files_exc(params)
        print 'done'
    print 'Loading spike data...', 
    spike_data = np.loadtxt(params['exc_spiketimes_fn_merged'])
    print 'done'
    if params['training_run']:
        mp = np.loadtxt(params['training_stimuli_fn'])
    else:
        mp = np.loadtxt(params['test_sequence_fn'])
    if params['n_stim'] == 1:
        mp = np.array([mp])
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    stim_durations = np.loadtxt(params['stim_durations_fn'])
#    stim_range = params['stim_range']
    stim_range = [0, 2]
    gids = get_cell_gid(params, tp_params, stim_range, n_cells)
    print 'Plotting GID:', gids
    n_stim = stim_range[1] - stim_range[0]
    nspikes_in = np.zeros((n_stim, n_cells))
    nspikes_out = np.zeros((n_stim, n_cells))
    colorlist = utils.get_colorlist(n_cells)

    fig = pylab.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax1.set_title('$\\beta_X = %.2f\  \\beta_V=%.2f$' % (params['blur_X'], params['blur_V']))
    
    for i_, gid in enumerate(gids):
        
        for stim_idx in range(stim_range[0], stim_range[1]):
            rate_fn = params['input_rate_fn_base'] + '%d_%d.dat' % (gid, stim_idx)
            spike_fn = params['input_st_fn_base']  + '%d_%d.dat' % (gid, stim_idx)
            try:
                input_spikes = np.loadtxt(spike_fn)
            except:
                input_spikes = np.array([])
#            print 'debug input_spikes', input_spikes, type(input_spikes), spike_fn
            print 'Input for cell %d stim %d nspikes %d, tp' % (gid, stim_idx, input_spikes.size), tp[gid-1, :], 'mp:', mp[stim_idx, 0], mp[stim_idx, 2]

            if params['n_stim'] > 1:
                t_offset = stim_durations[:stim_idx].sum()# + params['stim_durations_pause'] * stim_idx 
                t_range = (t_offset, t_offset + stim_durations[stim_idx])
                t_axis = np.arange(t_offset, t_offset + stim_durations[stim_idx] - params['dt_rate'], params['dt_rate'])
            else:
                t_offset = 0 
                t_range = (0, stim_durations)
                t_axis = np.arange(0, stim_durations - params['dt_rate'], params['dt_rate'])
            
            spikes = utils.get_spikes_for_gid(spike_data, gid, t_range=t_range)
#            ax1.plot(spikes, gid * np.ones(spikes.size), 'o', c='k', markersize=ms)
#            ax1.plot(input_spikes, gid * np.ones(len(input_spikes)), 'o', c='b', markersize=ms, alpha=0.15)
            ax1.plot(spikes, gid * np.ones(spikes.size), 'o', c=colorlist[i_], markersize=ms, markeredgewidth=0)
            ax1.plot(input_spikes, gid * np.ones(input_spikes.size), 'o', c=colorlist[i_], markersize=ms, markeredgewidth=0, alpha=0.1)
            if os.path.exists(rate_fn):
                rate = np.loadtxt(rate_fn)
                if rate.size != t_axis.size:
                    rate = rate[:-1]

                y_min = rate.min()
                y_max = rate.max()

                ax2.plot(t_axis, rate, c=colorlist[i_])
#                ax2.plot(t_axis, rate, c='b')
            else:
                ax2.plot(t_axis, np.zeros(t_axis.size), c=colorlist[i_])


            nspikes_in[stim_idx, i_] = input_spikes.size
            nspikes_out[stim_idx, i_] = len(spikes)
            ax3.plot(nspikes_in[stim_idx, i_], nspikes_out[stim_idx, i_], 'o', c=colorlist[i_], ms=5)

    ax3.set_xlabel('N spikes in')
    ax3.set_ylabel('N spikes out')

    if params['n_stim'] > 1:
        t0 = stim_durations[:stim_range[0]].sum()
        t1 = stim_durations[:stim_range[1]].sum()
    else:
        t0 = 0
        t1 = stim_durations
    ax1.set_xlim((t0, t1))


    pylab.show()
#    spike_fn = params['input_st_fn_base']  + '%d_%d.dat' % (gid, stim_idx)
