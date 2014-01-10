import sys
import os
import utils
import numpy as np
import json


param_fn = sys.argv[1]
if os.path.isdir(param_fn):
    param_fn += '/Parameters/simulation_parameters.json'
f = file(param_fn, 'r')
print 'Loading parameters from', param_fn
params = json.load(f)

#script_names = ['plot_rasterplots.py', 'plot_prediction.py', 'plot_connectivity_profile.py', 'plot_weight_and_delay_histogram.py ee',\
#        'plot_weight_and_delay_histogram.py ei', 'plot_weight_and_delay_histogram.py ie', 'plot_weight_and_delay_histogram.py ii']
#for sn in script_names:
#    os.system('python %s %s' % (sn, params['folder_name']))

import plot_prediction as pp

if params['n_grid_dimensions'] == 2:
    pp.plot_prediction_2D(params)
else:
    pp.plot_prediction_1D(params)
os.system('python plot_rasterplots.py %s' % params['folder_name'])
os.system('python plot_weight_and_delay_histogram.py %s' % params['folder_name'])
os.system('python plot_connectivity_profile.py %s' % params['folder_name'])
os.system('python PlottingScripts/PlotAnticipation.py %s' % params['folder_name'])
os.system('python PlottingScripts/plot_contour_connectivity.py %s' % params['folder_name'])
os.system('ristretto %s' % (params['figures_folder']))
