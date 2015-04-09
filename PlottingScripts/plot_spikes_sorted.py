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
import PlottingScripts.PlotPrediction as PP




def plot_spikes_sorted_simple(params, sort_idx=0, color_idx=None, ax=None):
    data_fn = params['exc_spiketimes_fn_merged']
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    if ax == None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
    tp_idx_sorted = tp[:, sort_idx].argsort()

    nspikes, spiketrains = utils.get_nspikes(data_fn, n_cells=params['n_exc'], cell_offset=0, get_spiketrains=True, pynest=True)

    if color_idx == 4:
        clim = (0., 180.)
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
        m.set_array(tp[:, color_idx])
        colorlist= m.to_rgba(tp[:, color_idx])
        cbar_label = 'Orientation'

    elif color_idx == 0:
        clim = (0., 1.)
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
        m.set_array(tp[:, color_idx])
        colorlist = m.to_rgba(tp[:, 0])
        cbar_label = 'Position'

    for i_, gid in enumerate(tp_idx_sorted):
        spikes = spiketrains[gid]
        nspikes = len(spikes)
        y_ = np.ones(nspikes) * tp[gid, sort_idx]
        if color_idx != None:
            ax.scatter(spikes, y_, c=m.to_rgba(tp[gid-1, color_idx]), linewidths=0, s=3)
        else:
            ax.plot(spikes, y_, 'o', markersize=3, markeredgewidth=0., color='k')

    ax.set_xlabel('Time [ms]')
    if sort_idx == 0:
        ax.set_ylabel('RF-position')
    elif sort_idx == 2 or sort_idx == 3:
        ax.set_ylabel('Preferred speed')
    elif sort_idx == 4:
        ax.set_ylabel('Preferred orientation')

    if color_idx != None:
        cbar = pylab.colorbar(m,ax=ax)
        cbar.set_label(cbar_label)

def plot_spikes_sorted(params):
    data_fn = params['exc_spiketimes_fn_merged']
    plotter = PP.PlotPrediction(params, data_fn)
    if plotter.no_spikes:
        print '\nWARNING!\nNO SPIKES WERE FOUND!'

    stim_range = params['stim_range']
    plotter.load_motion_params()
    stim_duration = np.loadtxt(params['stim_durations_fn'])
    if params['training_run']:
        mp = np.loadtxt(params['training_stimuli_fn'])
    else:
        mp = np.loadtxt(params['test_sequence_fn'])
    input_folder_exists = os.path.exists(params['input_folder'])

    for i_stim, stim in enumerate(range(stim_range[0], stim_range[1])):
        plotter.compute_pos_and_v_estimates(stim)
        print 'Stim:', stim
        if params['n_stim'] > 1:
            t0 = stim_duration[:i_stim].sum()
            t1 = stim_duration[:i_stim+1].sum()
        else:
            t0 = 0
            t1 = stim_duration

        time_range = (t0, t1)
        stim_range = (stim, stim + 1)
        plotter.n_fig_x = 1
        plotter.n_fig_y = 1
        plotter.create_fig()  # create an empty figure
        title = 'Exc cells sorted by x-position, Stim %d $x_{stim}=%.2f\ v_{stim}=%.2f$' % (stim, mp[stim, 0], mp[stim, 2])
        if input_folder_exists:
            plotter.plot_input_spikes_sorted(time_range, fig_cnt=1, sort_idx=0, plot_blank=False)
        plotter.plot_raster_sorted(stim_range, fig_cnt=1, title=title, sort_idx=0, plot_blank=False)

        plotter.create_fig()  # create an empty figure
        if input_folder_exists:
            plotter.plot_input_spikes_sorted(time_range, fig_cnt=1, sort_idx=4, plot_blank=False)
        plotter.plot_raster_sorted(stim_range, fig_cnt=1, title=title, sort_idx=4, plot_blank=False)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        print 'Case 1'
        params = utils.load_params(sys.argv[1])
    else:
        network_params = simulation_parameters.parameter_storage()  
        params = network_params.params

    plot_spikes_sorted(params)
    pylab.show()
