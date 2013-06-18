# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

import simulation_parameters as sp
import utils as u
network_params = sp.parameter_storage()
params = network_params.load_params()

d = np.loadtxt(params['exc_volt_anticipation'])
pops = u.pop_anticipatory_gids(params)


avg_volts = []
selected_gids = u.all_anticipatory_gids(params)
for pop in pops: 
    print 'debug', pop
    time_axis, volt = u.extract_trace(d, selected_gids[0])
    volt_sum = np.zeros(time_axis.size)
    for gid in pop:
        time_axis, volt = u.extract_trace(d,gid)
        volt_sum += volt
    avg_volt = volt_sum / len(selected_gids)
    avg_volts.append(avg_volt)
    
   


fig = pylab.figure()
ax = fig.add_subplot(111)
for i in xrange(len(pops)):
    ax.plot(time_axis, avg_volts[i], label='pop %d' % i)
pylab.legend()

