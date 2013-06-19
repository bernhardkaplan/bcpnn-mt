import os
import sys
import numpy as np
import utils
import pylab

if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.json'
    import json
    f = file(param_fn, 'r')
    print 'Loading parameters from', param_fn
    params = json.load(f)
else:
    import simulation_parameters as sp
    network_params = sp.parameter_storage()
    params = network_params.load_params()

tp = np.loadtxt(params['tuning_prop_means_fn'])
fn = params['exc_volt_anticipation']
print 'Loading ', fn
d = np.loadtxt(fn)

n_pop = 4
selected_gids, pops = utils.select_well_tuned_cells(tp, params, params['n_gids_to_record'], n_pop)
print 'pops', pops

time_axis, volt = utils.extract_trace(d, pops[0][0])

avg_volts = np.zeros((time_axis.size, len(pops) + 1))

avg_volts[:, 0] = time_axis
#selected_gids = utils.all_anticipatory_gids(params)
print 'selected_gids', len(selected_gids)
for j_, pop in enumerate(pops): 
    print 'debug', pop, len(pop)
    time_axis, volt = utils.extract_trace(d, pop[0])
    volt_sum = np.zeros(time_axis.size)
    x_group, y_group, u_group, v_group = np.zeros(len(pop)),np.zeros(len(pop)),np.zeros(len(pop)),np.zeros(len(pop))
    for i_, gid in enumerate(pop):
        x, y, u, v = tp[gid, :]
        x_group[i_] = x
        y_group[i_] = y
        u_group[i_] = u
        v_group[i_] = v
        time_axis, volt = utils.extract_trace(d,gid)
        volt_sum += volt
    print 'debug, population info', j_
    print 'x_avg:', x_group.mean(), x_group.std()
    print 'y_avg:', y_group.mean(), y_group.std()
    print 'u_avg:', u_group.mean(), u_group.std()
    print 'v_avg:', v_group.mean(), v_group.std()
    avg_volt = volt_sum / len(pop)
    avg_volts[:, j_ + 1] = avg_volt
    
   
data_fn = params['population_voltages_fn']
print 'Saving output to:', data_fn
np.savetxt(data_fn, avg_volts)

fig = pylab.figure()
ax = fig.add_subplot(111)
for i in xrange(len(pops)):
    ax.plot(time_axis, avg_volts[:, i+1], label='pop %d' % i, lw=3)

pylab.legend()
output_fn = params['figures_folder'] + 'anticipatory_avg_volt.png'
print 'Saving to', output_fn
pylab.savefig(output_fn)

pylab.show()
