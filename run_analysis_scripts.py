import sys
import os
import utils
import numpy as np
import json
import plot_prediction as pp


def run_analysis(params):
    if params['n_grid_dimensions'] == 2:
        pp.plot_prediction_2D(params)
    else:
        pp.plot_prediction_1D(params)
    os.system('python plot_rasterplots.py %s' % params['folder_name'])
    os.system('python plot_weight_and_delay_histogram.py %s' % params['folder_name'])
    os.system('python plot_connectivity_profile.py %s' % params['folder_name'])
    os.system('python PlottingScripts/PlotAnticipation.py %s' % params['folder_name'])
    os.system('python PlottingScripts/plot_contour_connectivity.py %s' % params['folder_name'])
#    os.system('ristretto %s' % (params['figures_folder']))

if len(sys.argv) == 2:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.json'
    f = file(param_fn, 'r')
    print 'Loading parameters from', param_fn
    params = json.load(f)
    run_analysis(params)
else:
    for i_, param_fn in enumerate(sys.argv[1:]):
        print '\n\n=========================\nAnalysis %d / %d begins\n==================================\n\n' % (i_ + 1, len(sys.argv[1:]))
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)
        run_analysis(params)
        print '\n\n=========================\nAnalysis %d / %d ends \n==================================\n\n' % (i_ + 1, len(sys.argv[1:]))
