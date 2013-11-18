# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

import pylab
import json
import utils
import numpy as np
import os
import utils


fig = pylab.figure()

def load_data_for_connectivity(cc, protocol):
    folder_name = 'Debug_5.24e-01od%s-bar-' % (cc) + protocol + '/'
    param_fn = folder_name + 'Parameters/simulation_parameters.json'
    if not os.path.exists(param_fn):
        print '\n File does not exist:\n\t', param_fn
        print 'Will not print:', cc, protocol
        return [], []
    else:
        pf = file(param_fn, 'r')
        params = json.load(pf)
    volt_fn = params['exc_volt_anticipation']
    if not os.path.exists(volt_fn):
        print '\n File does not exist:\n\t', volt_fn
        print 'Will not print:', cc, protocol
        return [], []
    print 'Loading:', volt_fn
    data = np.loadtxt(volt_fn)
    recorded_gids = np.loadtxt(params['gids_to_record_fn'])
    return data, recorded_gids


def average_voltage_traces(volt_data, recorded_gids):

    # get a sample trace to create empty containers
    time_axis, volt_ = utils.extract_trace(volt_data, recorded_gids[0])
    avg_trace = np.zeros(time_axis.size)
    for j_, gid in enumerate(recorded_gids):
        time_axis, volt = utils.extract_trace(volt_data, gid)
        avg_trace += volt
    avg_trace /= len(recorded_gids)    
    return avg_trace, time_axis


things_to_plot = [ ('crf_only', 'AIII'), ('crf_only', 'IIII'), \
        ('congruent', 'AIII'), ('incongruent', 'AIII'), ('random_predictor', 'AIII'), ('missing_crf', 'AIII'), \
        ('congruent', 'IIII'), ('incongruent', 'IIII'), ('random_predictor', 'IIII'), ('missing_crf', 'IIII')]


n_rows = 5
n_cols = 1
# decide which protocol to plot where in advance
map_protocol_to_plot = {'congruent' : 1, 'incongruent' : 2, 'missing_crf' : 3, 'random_predictor' : 4, 'crf_only' : 5}
map_conn_to_linestyle = {'AIII' : '-', 'IIII' : '--'}

subplots = []
for i in xrange(n_rows):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    subplots.append(ax)

for i_, thing in enumerate(things_to_plot):
    protocol = thing[0]
    conn_code = thing[1]
    print 'protocol conn_code', protocol, conn_code
    volt_data, recorded_gids = load_data_for_connectivity(conn_code, protocol)
    if not len(volt_data) == 0: # if files have not been found

        avg_trace, time_axis = average_voltage_traces(volt_data, recorded_gids)

        ls = map_conn_to_linestyle[conn_code]
        plot_pos = map_protocol_to_plot[protocol]
        ax = subplots[plot_pos - 1]
        ax.plot(time_axis, avg_trace, ls=ls, label='%s-%s' % (conn_code, protocol))

        ax.legend(loc='upper left')
        ax.set_ylabel('Mean voltage [mV]')


subplots[-1].set_xlabel('Time [ms]')
subplots[0].set_title('Comparison of mean v_mem from %d cells\nfor different protocols and connectivities' % (len(recorded_gids)))


pylab.show()
