"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons
"""

import numpy
import numpy.random as rnd
import time
from pyNN.utility import get_script_args
simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)

# # # # # # # # # # # # # # # # # # # # #
#     Simulation parameters             #
# # # # # # # # # # # # # # # # # # # # #
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

setup(timestep=0.1, min_delay=0.2, max_delay=1.0)

exc_pop = []
inh_pop = []
for i in xrange(params['n_mc']):
    exc_pop.append(Population(params['n_exc_per_mc'], IF_cond_exp, params['cell_params'], label="mc%d" % i))
    inh_pop.append(Population(params['n_inh_per_mc'], IF_cond_exp, params['cell_params'], label="inh%d" % i))
    # v_init
    exc_pop[i].randomInit(rnd.normal(params['v_init'], params['v_init_sigma'], params['n_exc_per_mc']))
    inh_pop[i].randomInit(rnd.normal(params['v_init'], params['v_init_sigma'], params['n_inh_per_mc']))


# # # # # # # # # # # # 
#     I N P U  T      # 
# # # # # # # # # # # #
# TODO (meduz) : make a proper MT input
input_pop = []
for column in xrange(params['n_mc']):
    fn = params['input_st_fn_base'] + str(column) + '.npy'
    spike_times = numpy.load(fn)
    input_pop.append(Population(1, SpikeSourceArray, {'spike_times': spike_times}, label="input%d" % column))


# # # # # # # # # # # # 
#     C O N N E C T   #
# # # # # # # # # # # #
# CONNECT INPUT TO exc_pop
input_prj = []
for column in xrange(params['n_mc']):
    n_conns = len(exc_pop[column]) * len(input_pop[column])
    connector = AllToAllConnector(weights=rnd.normal(params['w_exc_input'], params['w_exc_input'] * params['w_exc_input_sigma'], n_conns))
    input_prj.append(Projection(input_pop[column], exc_pop[column], connector))
# EXC - EXC
# EXC - INH
# INH - EXC
# INH - INH


# # # # # # # # # # # # # # # # 
#     N O I S E   I N P U T 
# # # # # # # # # # # # # # # # 
noise_prj_exc = []
noise_pop_exc = []
for column in xrange(params['n_mc']):
    noise_pop_exc.append(Population(params['n_exc_per_mc'], SpikeSourcePoisson, {'rate': params['f_exc_noise']}, "expoisson%d" % column))
    connector = OneToOneConnector(weights=params['w_exc_noise'] * numpy.ones(params['n_exc_per_mc']))
    noise_prj_exc.append(Projection(noise_pop_exc[-1], exc_pop[column], connector))

#TODO:
# record
for column in xrange(params['n_mc']):
    exc_pop[column].record()
    exc_pop[column].record_v()
#    input_pop[column].record()
#    inh_pop[column].record()
#    inh_pop[column].record_v()

t1 = time.time()
print "Running simulation ... "
run(params['t_sim'])
t2 = time.time()
print "Simulation time: %d sec or %.1f min for %d cells" % (t2-t1, (t2-t1)/60., params['n_cells'])

# # # # # # # # # # # # # # # # #
#     P R I N T    R E S U L T S 
# # # # # # # # # # # # # # # # #
for column in xrange(params['n_mc']):
    exc_pop[column].printSpikes("%s%d.ras" % (params['exc_spiketimes_fn_base'], column))
    exc_pop[column].print_v("%s%d.v" % (params['exc_volt_fn_base'], column))
#    inh_pop[column].printSpikes("%s%d" % (params['inh_spiketimes_fn_base'], column))
#    inh_pop[column].print_v("%s%d" % (params['exc_volt_fn_base'], column))
#    input_pop[column].printSpikes("%sinput_spikes_%s.ras" % (params['spiketimes_folder'], column))


end()
