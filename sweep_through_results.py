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

try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size

except:
    USE_MPI = False
    comm = None
    pc_id, n_proc = 0, 1

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.params

RC = ResultsCollector.ResultsCollector(params)

#w_ee = 0.030
#t_blank = 200
conn_code = 'AIII'
# the parameter to sweep for
#param_name = 'blur_X'
param_name = 'w_tgt_in_per_cell_ee'
#param_name = 't_blank'
#t_range=(0, 1000)
#param_name = 'delay_scale'
t_range=(0, 1600)
fmaxstim = 1000.

blur_v = 0.05
tbb = 400
to_match = '^LS_(.*)'
#to_match = '^LS_xpred_(.*)'
#to_match = '^LargeScaleModel_AIII_pee1(.*)delay250_scaleLatency0.50'
#to_match = '^LargeScaleModel_%s_fmaxstim1\.50e\+03_scaleLatency0\.15_tbb400_(.*)' % (conn_code)
#to_match = '^LargeScaleModel_%s_fmaxstim(.*)' % (conn_code)
#to_match = '^LargeScaleModel_%s_fmaxstim%.2e(.*)' % (conn_code,fmaxstim)
print 'to_match', to_match
#to_match = '^LargeScaleModel_%s_fmaxstim%.2e_(.*)' % (conn_code, fmaxstim)
#to_match = '^LargeScaleModel_%s_fmaxstim(.*)_tbb%d$' % (conn_code, tbb)
#output_fn = 'xvdiff_%s_tbb%d_fmaxstim%.1e_weeSweep_t%d-%d.dat' % (conn_code, tbb, fmaxstim, t_range[0], t_range[1])
output_fn = 'dummy.dat'
print 'output_fn', output_fn

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
    if not os.path.exists(name + '/Data/vx_grid.dat'):
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


RC.set_dirs_to_process(dir_names)
#print "RC.dirs_to_process", RC.dirs_to_process
RC.get_xvdiff_integral()#t_range=t_range)
RC.get_parameter(param_name)
#RC.get_parameter('w_sigma_x')
#RC.get_parameter('w_sigma_v')
print 'RC param_space', RC.param_space
RC.n_fig_x = 1
RC.n_fig_y = 2
RC.create_fig()
RC.plot_param_vs_xvdiff_integral(param_name, xv='x', fig_cnt=1)#, t_integral=t_range)
RC.plot_param_vs_xvdiff_integral(param_name, xv='v', fig_cnt=2)#, t_integral=t_range)
RC.save_output_data(output_fn)
#pylab.show()
#RC.get_cgxv()
#RC.plot_cgxv_vs_xvdiff()
