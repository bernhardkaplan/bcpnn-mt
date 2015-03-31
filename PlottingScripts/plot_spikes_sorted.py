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


def plot_spikes_sorted(params):
    data_fn = params['exc_spiketimes_fn_merged']
    plotter = PP.PlotPrediction(params, data_fn)
    if plotter.no_spikes:
        print '\nWARNING!\nNO SPIKES WERE FOUND!'

    stim_range = params['stim_range']
    plotter.load_motion_params()
    stim_duration = np.loadtxt(params['stim_durations_fn'])
    mp = np.loadtxt(params['training_stimuli_fn'])
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
            plotter.plot_input_spikes_sorted(time_range, fig_cnt=1, sort_idx=0)
        plotter.plot_raster_sorted(stim_range, fig_cnt=1, title=title, sort_idx=0)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        print 'Case 1'
        params = utils.load_params(sys.argv[1])
    else:
        network_params = simulation_parameters.parameter_storage()  
        params = network_params.params

    plot_spikes_sorted(params)
    pylab.show()
