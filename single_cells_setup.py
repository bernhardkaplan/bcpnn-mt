"""
A single IF neuron with exponential, conductance-based synapses, fed by two
spike sources.

Run as:

$ python IF_cond_exp.py <simulator>

where <simulator> is 'neuron', 'nest', etc

Andrew Davison, UNIC, CNRS
May 2006

$Id: IF_cond_exp.py 917 2011-01-31 15:23:34Z apdavison $
"""

import sys
import numpy as np
from pyNN.nest import *
import CreateConnections as CC
import utils
import simulation_parameters
ps = simulation_parameters.parameter_storage()
params = ps.params

tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])

(delay_min, delay_max) = params['delay_range']
setup(timestep=0.1, min_delay=delay_min, max_delay=delay_max)

gids = [130, 177]
n_exc = len(gids)
exc_pop = Population(n_exc, IF_cond_exp, params['cell_params_exc'], label='exc_cells')

# connect stimulus --> cells
for tgt, tgt_gid in enumerate(gids):
    fn = params['input_st_fn_base'] + str(tgt_gid) + '.npy'
    spike_times = np.load(fn)
    ssa = create(SpikeSourceArray, {'spike_times': spike_times})
    connect(ssa, exc_pop[tgt], params['w_input_exc'], synapse_type='excitatory')

    # connect cells
#    for src, src_gid in enumerate(gids):
#        if src_gid != tgt_gid:
#            p, latency = CC.get_p_conn(tuning_prop[src_gid, :], tuning_prop[tgt_gid, :], params['w_sigma_x'], params['w_sigma_v']) 
#            print 'p', p
#            p = np.array([1e-6, p])
#            w = utils.linear_transformation(p, params['w_min'], params['w_max'])
#            w = w[1]
#            print "src tgt w", src_gid, tgt_gid, w
#            delay = min(max(latency * params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
#            connect(exc_pop[src], exc_pop[tgt], w, delay=delay, synapse_type='excitatory')

#delay = 30
p, latency = CC.get_p_conn(tuning_prop[gids[1], :], tuning_prop[gids[0], :], params['w_sigma_x'], params['w_sigma_v']) 
delay = latency * params['t_stimulus'] * 0.4
#delay = min(max(latency * params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
print 'latency %.3e\tdelay %.2e' % (latency, delay)

delay = 1
w = 0.5e-2
connect(exc_pop[1], exc_pop[0], w, delay=delay, synapse_type='excitatory')
    
exc_pop.record()
exc_pop.record_v()

run(params['t_sim'])

folder_name = 'BarcelonaData/'
exc_pop.printSpikes(folder_name + 'spikes.ras')
exc_pop.print_v(folder_name + 'voltages_%.1e.v' % w, compatible_output=False)
#exc_pop.print_v(folder_name + 'voltages_%.1e_delay.v' % (w), compatible_output=False)
#exc_pop.print_v(folder_name + 'voltages_no_rec.v', compatible_output=False)

#g_exc = f_exc * w_exc * tau_syn_exc
#g_inh = f_inh * w_inh * tau_syn_inh

#from pyNN.errors import RecordingError
#try:
    #record_gsyn(ifcell, "Results/IF_cond_exp_%s.gsyn" % simulator_name)
#except (NotImplementedError, RecordingError):
    #pass

end()

