import sys
import os
import utils
import numpy as np
import json


#if len(sys.argv) > 1:

param_fn = sys.argv[1]
if os.path.isdir(param_fn):
    param_fn += '/Parameters/simulation_parameters.json'
f = file(param_fn, 'r')
print 'Loading parameters from', param_fn
params = json.load(f)

#    import NeuroTools.parameters as NTP
#    fn_as_url = utils.convert_to_url(param_fn)
#    params = NTP.ParameterSet(fn_as_url)
#else:
#    print '\nPlotting the default parameters given in simulation_parameters.py\n'
#    import simulation_parameters
#    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
#    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

script_names = ['plot_rasterplots.py', 'plot_prediction.py', 'plot_connectivity_profile.py', 'plot_weight_and_delay_histogram.py ee',\
        'plot_weight_and_delay_histogram.py ei', 'plot_weight_and_delay_histogram.py ie', 'plot_weight_and_delay_histogram.py ii']
        #'analyse_simple.py']

#script_names = ['merge_connlists.py', 'analyse_simple.py', 'merge_connlists.py', \
#'get_conductance_matrix.py', 'plot_spike_histogram.py exc', 'plot_spike_histogram.py inh', 'plot_connectivity_profile.py']

for sn in script_names:
    os.system('python %s %s' % (sn, params['folder_name']))

