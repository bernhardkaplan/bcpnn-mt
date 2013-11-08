import numpy as np
import nest

t_sim = 1000.
f_in = 100.

if (not 'bcpnn_synapse' in nest.Models('synapses')):
	nest.Install('pt_module')
initial_weight = np.log(nest.GetDefaults('bcpnn_synapse')['p_ij']/(nest.GetDefaults('bcpnn_synapse')['p_i']*nest.GetDefaults('bcpnn_synapse')['p_j']))
initial_bias = np.log(nest.GetDefaults('bcpnn_synapse')['p_j'])
syn_param = {"weight": initial_weight, "bias": initial_bias,"K":1.0,"delay":1.0,"tau_i":10.0,"tau_j":10.0,"tau_e":100.0,"tau_p":1000.0}

neuron1_spike_gen = nest.Create('poisson_generator', params={'rate': f_in})
neuron2_spike_gen = nest.Create('poisson_generator', params={'rate': f_in})


neuron1 = nest.Create("iaf_neuron", 3)
neuron2 = nest.Create("iaf_neuron", 3)

#neuron1 = nest.Create("iaf_neuron")
#neuron2 = nest.Create("iaf_neuron")
#nest.Connect(neuron1_spike_gen, neuron1, params={'weight':  1000.0})
#nest.Connect(neuron2_spike_gen, neuron2, params={'weight':  1000.0})
#nest.Connect(neuron1,neuron2,params=syn_param, model="bcpnn_synapse")

nest.DivergentConnect(neuron1, neuron2, model="bcpnn_synapse")
params = nest.GetStatus(nest.GetConnections(neuron1))

for i, c in enumerate(params):
    print 'weight %d:' % i, c['weight']

conns = nest.GetConnections(neuron1, neuron2, synapse_model='bcpnn_synapse')
#print 'neuron1:', neuron1
#print 'neuron2:', neuron2
print 'default params:', params
nest.SetStatus(nest.GetConnections(neuron1), {'weight': 1999.6661} )
params = nest.GetStatus(nest.GetConnections(neuron1))

print 'modified params:', params
#fconns = nest.FindConnections([neuron1[0]], neuron2)

#print 'n1 - n2 conns:', conns
#print 'n1 - n2 find conns:', fconns
