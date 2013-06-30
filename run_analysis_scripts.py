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

script_names = ['plot_rasterplots.py', 'plot_prediction.py', 'plot_connectivity_profile.py', 'plot_weight_and_delay_histogram.py ee',\
        'plot_weight_and_delay_histogram.py ei', 'plot_weight_and_delay_histogram.py ie', 'plot_weight_and_delay_histogram.py ii']


for sn in script_names:
    os.system('python %s %s' % (sn, params['folder_name']))

