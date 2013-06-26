# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:40:36 2013

@author: aliakbari-.m
"""

import matplotlib
#matplotlib.use('Agg')
import numpy as np
import os
import simulation_parameters as sp
import ResultsCollector
import re
import pylab
import json
import sys


ps = sp.parameter_storage()
params = ps.load_params()
protocols = ps.allowed_protocols



def browse_folder_for_data(to_match):
    data_fn = '/Spikes/exc_nspikes.dat'
    param_fn = '/Parameters/simulation_parameters.json'
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
    data_paths = []
    param_file_paths = []
    for name in dir_names:
        if not os.path.exists(name + data_fn):
            missing_dirs.append(name)
        else:
            data_paths.append(name + data_fn)
            param_file_paths.append(name + param_fn)
    print 'missing data:', missing_dirs
    return data_paths, param_file_paths

# Define the pattern of folder names where to look for the data_fn

#protocol = 'crf_only'
n_files = 5
orientation_axis  = np.zeros((len(protocols), n_files))
response_axis = np.zeros((len(protocols), n_files))
    
for j , protocol in enumerate(protocols):
    print protocol
    to_match = '^Debug_(.*)odIIII-bar-%s' % (protocol)
#    print 'Folder to_match:', to_match
    final_sweep_output_fn = 'orientation_tuning_curve.dat'
#    print 'final_sweep_output_fn', final_sweep_output_fn
     
    data_paths, param_file_paths = browse_folder_for_data(to_match)
    n_files = len(data_paths)
    print 'n_files', n_files
    response_axis_unsrt = np.zeros(n_files)
    orientation_axis_unsrt  = np.zeros( n_files)


    
    for i_ in xrange(n_files):
        print i_
        pf = file(param_file_paths[i_], 'r')
        p = json.load(pf)
        mp = p['motion_params']
        orientation_axis_unsrt[i_] = mp[4]
        spike_data = np.loadtxt(data_paths[i_])
        gids = np.loadtxt(p['gids_to_record_fn'], dtype=np.int)
        selected_nspikes = spike_data[gids, 1]
        response_axis_unsrt[i_] = selected_nspikes.mean()
        print response_axis_unsrt[i_]
#        print 'debug gids', gids
#        print 'n_spikes', spike_data[gids, 1]
    print 'response_axis_unsrt',response_axis_unsrt
    
    sorted_idx = np.argsort(orientation_axis_unsrt)
    orientation_axis = orientation_axis_unsrt[sorted_idx]
    print 'shape of response_axis_unsrt[sorted_idx]',np.shape(response_axis_unsrt), sorted_idx.size
    print 'shape of response_axis',np.shape(response_axis)

    response_axis[j,:] = response_axis_unsrt[sorted_idx]
    
print 'orientation_axis', orientation_axis


fig = pylab.figure()
ax = fig.add_subplot(111)
for j , protocol in enumerate(protocols):
    print protocol
    print 'response_axis', response_axis[j,:]

    ax.plot(orientation_axis, response_axis[j,:], 'o-', label = protocol)
ax.set_xlabel('Orientation')
ax.set_ylabel('NSpikes averaged over %d cells' % len(gids))
pylab.legend()
pylab.savefig('orientation tuning for all motion protocols')
pylab.show()

