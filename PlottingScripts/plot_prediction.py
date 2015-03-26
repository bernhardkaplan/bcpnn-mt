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
from matplotlib import cm
import re
from PlotPrediction import PlotPrediction

def plot_prediction(params, stim_range):

    data_fn = params['exc_spiketimes_fn_merged']
    plotter = PlotPrediction(params, data_fn)

    plotter.load_motion_params()
    output_fn_base = '%sprediction' % (params['figures_folder'])
    stim_duration = np.loadtxt(params['stim_durations_fn'])
#    print 'trajectories:', plotter.trajectories_x

#    for i_stim, stim in enumerate(range(stim_range[0], stim_range[1])):
#        plotter.compute_pos_and_v_estimates(stim)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        network_params = simulation_parameters.parameter_storage()  
        params = network_params.params
        plot_prediction(params, stim_range)
    elif len(sys.argv) == 2:
        print 'Case 1'
        params = utils.load_params(sys.argv[1])
        stim_range = params['stim_range']
        plot_prediction(params, stim_range)
    elif len(sys.argv) == 4:
        print 'Case 2'
        print '\nPlotting the default parameters give in simulation_parameters.py\n'
        params = utils.load_params(sys.argv[1])
        stim_range = (int(sys.argv[2]), int(sys.argv[3]))
        plot_prediction(params, stim_range)
