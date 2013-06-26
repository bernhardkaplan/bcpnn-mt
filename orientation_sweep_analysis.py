# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:40:36 2013

@author: aliakbari-.m
"""

import matplotlib
#matplotlib.use('Agg')
import numpy as np
import os
import simulation_parameters
import ResultsCollector
import re
import pylab
import json
import sys

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
    print 'missing data:', missing data
    return data_paths, param_file_paths

# Define the pattern of folder names where to look for the data_fn

protocol = 'random_predictor'
to_match = '^OrientationTuning_(.*)odAIII-bar-%s' % (protocol)
print 'Folder to_match:', to_match
final_sweep_output_fn = 'orientation_tuning_curve.dat'
print 'final_sweep_output_fn', final_sweep_output_fn


data_paths, param_file_paths = browse_folder_for_data(to_match)
n_files = len(data_paths)

orientation_axis_unsrt  = np.zeros(n_files)
response_axis_unsrt = np.zeros(n_files)
#orientation_axis  = np.zeros(n_files)
#response_axis = np.zeros(n_files)

for i_ in xrange(n_files):
    pf = file(param_file_paths[i_], 'r')
    p = json.load(pf)
    mp = p['motion_params']
    orientation_axis_unsrt[i_] = mp[4]
    spike_data = np.loadtxt(data_paths[i_])
    gids = np.loadtxt(p['gids_to_record_fn'], dtype=np.int)
    selected_nspikes = spike_data[gids, 1]
    response_axis_unsrt[i_] = selected_nspikes.mean()
    print 'debug gids', gids
    print 'n_spikes', spike_data[gids, 1]



sorted_idx = np.argsort(orientation_axis_unsrt)
orientation_axis = orientation_axis_unsrt[sorted_idx]
response_axis = response_axis_unsrt[sorted_idx]
print 'debug results:'
print 'response_axis', response_axis
print 'orientation_axis', orientation_axis


fig = pylab.figure()
ax = fig.add_subplot(111)

ax.plot(orientation_axis, response_axis, 'o-')
ax.set_xlabel('Orientation')
ax.set_ylabel('NSpikes averaged over %d cells' % len(gids))
pylab.show()

