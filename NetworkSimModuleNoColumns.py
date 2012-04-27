"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons
"""

import time
import numpy as np
import numpy.random as rnd
from pyNN.nest import *

def run_sim(params, sim_cnt):
    simulator_name = 'nest'

    # # # # # # # # # # # # 
    #     S E T U P       #
    # # # # # # # # # # # #
    setup()# timestep=0.1, min_delay=0.1, max_delay=1.0)
    rng_v = NumpyRNG(seed = sim_cnt*3147 + params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes
    rng_conn = NumpyRNG(seed = params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes

    v_init_dist = RandomDistribution('normal', 
            (params['v_init'], params['v_init_sigma']), 
            rng=rng_v, 
            constrain='redraw', 
            boundaries=(-80, -60))

    w_ei_dist = RandomDistribution('normal', 
            (params['w_ei_mean'], params['w_ei_sigma']), 
            rng=rng_conn, 
            constrain='redraw', 
            boundaries=(0, 0.01))

    w_ie_dist = RandomDistribution('normal', 
            (params['w_ie_mean'], params['w_ie_sigma']), 
            rng=rng_conn, 
            constrain='redraw', 
            boundaries=(0, 0.01))

    w_ii_dist = RandomDistribution('normal', 
            (params['w_ii_mean'], params['w_ii_sigma']), 
            rng=rng_conn, 
            constrain='redraw', 
            boundaries=(0, 0.01))

    w_input_exc_dist = RandomDistribution('normal', 
            (params['w_input_exc'], params['w_input_exc_sigma']), 
            rng=rng_conn, 
            constrain='redraw', 
            boundaries=(0, 0.01))

    delay_dist = RandomDistribution('normal', 
            (1, 0.01), 
            rng=rng_conn, 
            constrain='redraw', 
            boundaries=(0, 1000))


    # # # # # # # # # # # # 
    #     C R E A T E     #
    # # # # # # # # # # # #
    # Excitatory populations
    exc_pop = Population(params['n_exc'], IF_cond_exp, params['cell_params_exc'], label='exc_cells')
    exc_pop.initialize('v', v_init_dist)

    # Inhibitory population
    inh_pop = Population(params['n_inh'], IF_cond_exp, params['cell_params_inh'], label="inh_pop")
    inh_pop.initialize('v', v_init_dist)

    # Input spike trains
    input_pop = []
    for column in xrange(params['n_mc']):
        fn = params['input_st_fn_base'] + str(column) + '.npy'
        spike_times = np.load(fn)
        input_pop.append(SpikeSourceArray({'spike_times': spike_times}))
#        input_pop.append(Population(1, SpikeSourceArray, {'spike_times': spike_times}, label="input%d" % column))

    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    E X C - E X C  #
    # # # # # # # # # # # # # # # # # # # #
    # during the learning process load the updated connection matrix
    # plastic connections are retrieved from the file
    connector_ee = FromFileConnector(params['conn_list_ee_fn_base'] + str(sim_cnt) + '.dat')
    prj_ee = Projection(exc_pop, exc_pop, connector_ee, target='excitatory')


    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    E X C - I N H  #
    # # # # # # # # # # # # # # # # # # # #
#    connector_ei = FastFixedProbabilityConnector(params['p_exc_inh_global'], weights=params['w_exc_inh_global'], delays=delay_dist)
#    connector_ei = FromFileConnector(params['conn_list_ei_fn'])
    connector_ei = FastFixedProbabilityConnector(params['p_ei'], weights=w_ei_dist, delays=delay_dist)
    exc_inh_prj = Projection(exc_pop, inh_pop, connector_ei, target='excitatory')

    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    I N H - E X C  #
    # # # # # # # # # # # # # # # # # # # #
#    connector_ie = FastFixedProbabilityConnector(params['p_inh_exc_global'], weights=params['w_inh_exc_global'], delays=delay_dist)
#    connector_ie = FromFileConnector(params['conn_list_ie_fn'])
    connector_ie = FastFixedProbabilityConnector(params['p_ie'], weights=w_ie_dist, delays=delay_dist)
    inh_exc_prj = Projection(inh_pop, exc_pop, connector_ie)

    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    I N H - I N H  #
    # # # # # # # # # # # # # # # # # # # #
#    connector_ii = FromFileConnector(params['conn_list_ii_fn'])
    connector_ii = FastFixedProbabilityConnector(params['p_ii'], weights=w_ii_dist, delays=delay_dist)
    inh_inh_prj = Projection(exc_pop, inh_pop, connector_ii)

    # # # # # # # # # # # # # # # # # # # # # # 
    #     C O N N E C T    I N P U T - E X C  #
    # # # # # # # # # # # # # # # # # # # # # # 
    for i in xrange(params['n_exc']):
        print "debug", i, len(input_pop)

        connect(input_pop[i], exc_pop[i], params['w_input_exc'])
#        connector = AllToAllConnector(weights=params['w_input_exc'])    # weights are drawn from the given random distibution
#        inh_exc_prj = Projection(input_pop[i], exc_pop, connector)


    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    B I A S   T O   C E L L S  # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # 
    bias_currents = []
    bias_values = np.loadtxt(params['bias_values_fn_base'] + str(sim_cnt) + '.npy')
    for cell in xrange(params['n_exc']):
        i_amp = bias_values[cell]
        bias_currents.append(DCSource({'amplitude':i_amp, 'start':0, 'stop':params['t_sim']}))
        bias_currents[-1].inject_into([exc_pop[cell]])


    # # # # # # # # # # # # # # # # 
    #     N O I S E   I N P U T   #
    # # # # # # # # # # # # # # # # 
    noise_prj_exc = []
    noise_pop_exc = []
    for column in xrange(params['n_mc']):
        if (simulator_name == 'nest'): # for nest one can use the optimized Poisson generator
            noise_pop_exc.append(Population(params['n_exc_per_mc'], native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']}))
        else:
            noise_pop_exc.append(Population(params['n_exc_per_mc'], SpikeSourcePoisson, {'rate': params['f_exc_noise']}, "expoisson%d" % column))
        connector = OneToOneConnector(weights=params['w_exc_noise'])
        noise_prj_exc.append(Projection(noise_pop_exc[-1], exc_pop[column], connector))


    # # # # # # # # # # # #
    #     R E C O R D     #
    # # # # # # # # # # # #
    for column in xrange(params['n_mc']):
        exc_pop[column].record()
        exc_pop[column].record_v()
    #    input_pop[column].record()

    inh_pop.record()
    inh_pop.record_v()

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
        exc_pop[column].print_v("%s%d.v" % (params['exc_volt_fn_base'], column), compatible_output=False)
    #    input_pop[column].printSpikes("%sinput_spikes_%s.ras" % (params['spiketimes_folder'], column))

    inh_pop.printSpikes(params['inh_spiketimes_fn_base'])
    inh_pop.print_v(params['inh_volt_fn_base'], compatible_output=False)

    end()
