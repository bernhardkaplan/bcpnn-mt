# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:40:36 2013

@author: aliakbari-.m
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import simulation_parameters
import NeuroTools.parameters as NTP
import ResultsCollector
import re
import pylab
import json
import sys

"""
This script requires that plot_prediction.py has been called for all folders that 
are to be processed here (that appear in dir_names).

--> you can use run_plot_prediction.py to automatically do this
"""

data_fn = '/Data/something_new.dat'
# Define the pattern of folder names where to look for the data_fn

blur_v = 0.05
tbb = 400
to_match = '^Sweep_bx(.*)'
print 'Folder to_match:', to_match
final_sweep_output_fn = 'dummy.dat'
print 'final_sweep_output_fn', final_sweep_output_fn

dir_names = []
if len(sys.argv) == 1:
    for thing in os.listdir('.'):
        if os.path.isdir(thing):
            m = re.search('%s' % to_match, thing)
            if m:
                dir_names.append(thing)
else:
    for thing in sys.argv[1:]:
        if os.path.isdir(thing):
            m = re.search('%s' % to_match, thing)
            if m:
                dir_names.append(thing)


missing_dirs = []
for name in dir_names:
    if not os.path.exists(name + data_fn):
        missing_dirs.append(name)

print 'dirnames', dir_names
if len(missing_dirs) > 0:
    fn_out = 'missing_data_dirs.json'
    output_file = file(fn_out, 'w')
    d = json.dump(missing_dirs, output_file)
    
    print '\nData files missing in the following dirs:\n', missing_dirs
    print 'len(missing_dirs):', len(missing_dirs)
    print 'please run:\n python run_plot_prediction.py %s' % fn_out
    exit(1)


network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.params

RC = ResultsCollector.ResultsCollector(params)

RC.set_dirs_to_process(dir_names)
RC.get_parameter(param_name)

#print "RC.dirs_to_process", RC.dirs_to_process
RC.get_xvdiff_integral()#t_range=t_range)
param_name = 'blur_X'

#RC.get_parameter('w_sigma_x')
#RC.get_parameter('w_sigma_v')
print 'RC param_space', RC.param_space
RC.n_fig_x = 1
RC.n_fig_y = 2
RC.create_fig()
RC.plot_param_vs_xvdiff_integral(param_name, xv='x', fig_cnt=1)#, t_integral=t_range)
RC.plot_param_vs_xvdiff_integral(param_name, xv='v', fig_cnt=2)#, t_integral=t_range)
RC.save_output_data(final_sweep_output_fn)
#pylab.show()
#RC.get_cgxv()
#RC.plot_cgxv_vs_xvdiff()
