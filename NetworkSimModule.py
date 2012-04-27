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
    rng = NumpyRNG(seed = params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes

    v_init_distr = RandomDistribution('normal', 
            (params['v_init'], params['v_init_sigma']), 
            rng=rng, 
            constrain='redraw', 
            boundaries=(-80, -60))

    w_input_exc_distr = RandomDistribution('normal', 
            (params['w_input_exc'], params['w_input_exc_sigma']), 
            rng=rng, 
            constrain='redraw', 
            boundaries=(0, 0.01))

    delay_distr = RandomDistribution('normal', 
            (1, 0.01), 
            rng=rng, 
            constrain='redraw', 
            boundaries=(0, 1000))

    #= RandomDistribution(distribution='normal',
    #            parameters=[87e-3, 8.7e-3],
    #            rng=script_rng,
    #            boundaries=(0, 1000*87e-3),
    #            constrain='redraw')

    #delay_distr_exc_exc = RandomDistribution(distribution='normal',
    #            parameters=[1.5, 0.75],
    #            rng=rng,
    #            boundaries=(0.1,1000*1.5),
    #            constrain='redraw')



    # # # # # # # # # # # # 
    #     C R E A T E     #
    # # # # # # # # # # # #
    # Excitatory populations
    exc_pop = []

    for i in xrange(params['n_mc']):
        exc_pop.append(Population(params['n_exc_per_mc'], IF_cond_exp, params['cell_params_exc'], label="mc%d" % i))
        exc_pop[i].initialize('v', v_init_distr)

    # Inhibitory population
    inh_pop = Population(params['n_inh'], IF_cond_exp, params['cell_params_inh'], label="inh_pop")
    inh_pop.initialize('v', v_init_distr)

    # Input spike trains
    input_pop = []
    for column in xrange(params['n_mc']):
        fn = params['input_st_fn_base'] + str(column) + '.npy'
        spike_times = np.load(fn)
        input_pop.append(Population(1, SpikeSourceArray, {'spike_times': spike_times}, label="input%d" % column))


    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    E X C - E X C  #
    # # # # # # # # # # # # # # # # # # # #
    # during the learning process load the updated connection matrix
    #pop_conn_mat = np.load(params['conn_mat_ee_fn_base']) + str(sim_cnt) + '.npy'
    pop_conn_mat = np.load(params['conn_mat_mc_mc'] + str(sim)
    exc_exc_prj = []
    for src in xrange(pop_conn_mat[:, 0].size):
        for tgt in xrange(pop_conn_mat[:, 0].size):
            if pop_conn_mat[src, tgt] != 0:
                connector = FastFixedProbabilityConnector(p_connect=params['p_exc_exc_global'], weights=params['w_exc_exc_global'], delays=delay_distr)
                exc_exc_prj.append(Projection(exc_pop[src], exc_pop[tgt], connector, target='excitatory'))

    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    E X C - I N H  #
    # # # # # # # # # # # # # # # # # # # #
    exc_inh_prj = []
    for src in xrange(params['n_mc']):
        connector = FastFixedProbabilityConnector(params['p_exc_inh_global'], weights=params['w_exc_inh_global'], delays=delay_distr)
        exc_inh_prj.append(Projection(exc_pop[src], inh_pop, connector, target='excitatory'))

    inh_exc_prj = []
    for src in xrange(params['n_mc']):
        connector = FastFixedProbabilityConnector(params['p_inh_exc_global'], weights=params['w_inh_exc_global'], delays=delay_distr)
        inh_exc_prj.append(Projection(exc_pop[src], inh_pop, connector))


    # # # # # # # # # # # # # # # # # # # # # # 
    #     C O N N E C T    I N P U T - E X C  #
    # # # # # # # # # # # # # # # # # # # # # # 
    input_prj = []
    for column in xrange(params['n_mc']):
        connector = AllToAllConnector(weights=w_input_exc_distr)    # weights are drawn from the given random distribution
        input_prj.append(Projection(input_pop[column], exc_pop[column], connector))


    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    B I A S   T O   C E L L S  # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # 
    bias_currents = []
    bias_values = np.loadtxt(params['bias_values_fn_base'] + str(sim_cnt) + '.npy')
    for cell in xrange(params['n_exc']):
        i_amp = bias_values[cell]
        bias_currents.append(DCSource({'amplitude':i_amp, 'start':0, 'stop':params['t_sim']}))


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
