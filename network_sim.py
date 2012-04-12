"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons
"""

import numpy
import time
from pyNN.utility import get_script_args

# # # # # # # # # # # # # # # # # # # # #
#     Simulation parameters             #
# # # # # # # # # # # # # # # # # # # # #

n_exc = 12
n_inh = 3
n_neurons = n_exc + n_inh

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)

tstop = 1000.0
rate = 100.0

setup(timestep=0.1, min_delay=0.2, max_delay=1.0)

cell_params = {'tau_refrac':2.0,'v_thresh':-50.0,'tau_syn_E':2.0, 'tau_syn_I':2.0}
exc_pop = Population(n_exc, IF_cond_exp, cell_params, label="exc")
inh_pop = Population(n_inh, IF_cond_exp, cell_params, label="inh")

# # # # # # # # # # # # 
#     I N P U  T      # 
# # # # # # # # # # # #
# TODO (meduz) : make a proper MT input
n_in = n_exc
number = int(n_in*tstop*rate/1000.0)
numpy.random.seed(26278342)
spike_times = numpy.add.accumulate(numpy.random.exponential(1000.0/rate, size=number))
assert spike_times.max() > tstop
input_population  = Population(n_exc, SpikeSourceArray, {'spike_times': spike_times}, label="input")

input_projection = Projection(input_population, exc_pop, AllToAllConnector())
input_projection.setWeights(1.0)

input_population.record()
exc_pop.record()
exc_pop.record_v()

t1 = time.time()
run(tstop)
t2 = time.time()
print "Simulation time: %d sec or %d min" % (t2-t1, (t2-t1)/60.)

exc_pop.printSpikes("Results/exc_output_%s.ras" % simulator_name)
input_population.printSpikes("Results/simpleNetwork_input_%s.ras" % simulator_name)
exc_pop.print_v("Results/exc_%s.v" % simulator_name)

end()
