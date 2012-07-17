"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons
"""

import time
import numpy as np
import numpy.random as rnd
import sys
import NeuroTools.parameters as ntp
import os


def run_sim(params, sim_cnt, initial_connectivity='precomputed', connect_exc_exc=True):

    simulator_name = params['simulator']
    if simulator_name == 'nest':
        from pyNN.nest import *
    elif simulator_name=='brian':
        from pyNN.brian import *
    else:
        print 'using ', params['simulator'] , ' is no imeplemented :-( '  
    t1 = time.time()
#    sim_cnt = int(sys.argv[1])

    # # # # # # # # # # # # 
    #     S E T U P       #
    # # # # # # # # # # # #
    # this function expects a parameter dictionary
    (delay_min, delay_max) = params['delay_range']
    setup(timestep=0.1, min_delay=delay_min, max_delay=delay_max)
    rng_v = NumpyRNG(seed = sim_cnt*3147 + params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes
    rng_conn = NumpyRNG(seed = params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes

    # # # # # # # # # # # # # # # # # # # # # # # # #
    #     R A N D O M    D I S T R I B U T I O N S  #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    v_init_dist = RandomDistribution('normal',
            (params['v_init'], params['v_init_sigma']),
            rng=rng_v,
            constrain='redraw',
            boundaries=(-80, -60))

    if initial_connectivity == 'random':
        w_range=(0, 5e-3)
        w_lambda = params['w_distribution_fit_wlambda']
        w_ee_dist = RandomDistribution('exponential',
                (w_lambda,),
                rng=rng_conn,
                constrain='clip',
#                constrain='redraw',
                boundaries=w_range)

        delay_dist = RandomDistribution('uniform',
                params['delay_range'],
                rng=rng_conn,
                constrain='redraw',
                boundaries=params['delay_range'])

    w_ei_dist = RandomDistribution('normal',
            (params['w_ei_mean'], params['w_ei_sigma']),
            rng=rng_conn,
            constrain='redraw',
            boundaries=(0, params['w_ei_mean'] * 10.))

    w_ie_dist = RandomDistribution('normal',
            (params['w_ie_mean'], params['w_ie_sigma']),
            rng=rng_conn,
            constrain='redraw',
            boundaries=(0, params['w_ie_mean'] * 10.))

    w_ii_dist = RandomDistribution('normal',
            (params['w_ii_mean'], params['w_ii_sigma']),
            rng=rng_conn,
            constrain='redraw',
            boundaries=(0, params['w_ii_mean'] * 10.))

#    w_input_exc_dist = RandomDistribution('normal',
#            (params['w_input_exc'], params['w_input_exc_sigma']),
#            rng=rng_conn,
#            constrain='redraw',
#            boundaries=(0, 0.01))

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


    print "Loading input spiketrains..."
    # Input spike trains
    input_pop = []
    for cell in xrange(params['n_exc']):
        try:
            fn = params['input_st_fn_base'] + str(cell) + '.npy'
            spike_times = np.load(fn)
        except:
            print "Missing file: ", fn
            spike_times = []
            # this cell does not get any input
        ssa = create(SpikeSourceArray, {'spike_times': spike_times})
        input_pop.append(ssa)
    #        input_pop.append(Population(1, SpikeSourceArray, {'spike_times': spike_times}, label="input%d" % cell))
    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    E X C - E X C  #
    # # # # # # # # # # # # # # # # # # # #
    # during the learning process load the updated connection matrix
    # plastic connections are retrieved from the file

    if connect_exc_exc:
        if initial_connectivity == 'precomputed':
            conn_list_fn = params['conn_list_ee_fn_base'] + str(sim_cnt) + '.dat'

        else: # random
            conn_list_fn = params['random_weight_list_fn'] + str(sim_cnt) + '.dat'

        print "Connecting exc - exc from file", conn_list_fn
        connector_ee = FromFileConnector(conn_list_fn)
        prj_ee = Projection(exc_pop, exc_pop, connector_ee, target='excitatory')


    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    E X C - I N H  #
    # # # # # # # # # # # # # # # # # # # #
    #    connector_ei = FastFixedProbabilityConnector(params['p_exc_inh_global'], weights=params['w_exc_inh_global'], delays=delay_dist)
    #    connector_ei = FromFileConnector(params['conn_list_ei_fn'])
    print "Connecting exc - inh ..."
    connector_ei = FastFixedProbabilityConnector(params['p_ei'], weights=w_ei_dist, delays=delay_dist)
    exc_inh_prj = Projection(exc_pop, inh_pop, connector_ei, target='excitatory')

    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    I N H - E X C  #
    # # # # # # # # # # # # # # # # # # # #
    #    connector_ie = FastFixedProbabilityConnector(params['p_inh_exc_global'], weights=params['w_inh_exc_global'], delays=delay_dist)
    #    connector_ie = FromFileConnector(params['conn_list_ie_fn'])
    print "Connecting inh - exc ..."
    connector_ie = FastFixedProbabilityConnector(params['p_ie'], weights=w_ie_dist, delays=delay_dist)
    inh_exc_prj = Projection(inh_pop, exc_pop, connector_ie, target='inhibitory')

    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    I N H - I N H  #
    # # # # # # # # # # # # # # # # # # # #
    #    connector_ii = FromFileConnector(params['conn_list_ii_fn'])
    print "Connecting inh - inh ..."
    connector_ii = FastFixedProbabilityConnector(params['p_ii'], weights=w_ii_dist, delays=delay_dist)
    inh_inh_prj = Projection(exc_pop, inh_pop, connector_ii, target='inhibitory')

    # # # # # # # # # # # # # # # # # # # # # # 
    #     C O N N E C T    I N P U T - E X C  #
    # # # # # # # # # # # # # # # # # # # # # # 
    print "Connecting input - exc ..."
    input_conns = []
    #    spike_sourceE = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(5,105,10)]})
    #    spike_sourceI = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(155,255,10)]})
    for i in xrange(params['n_exc']):
    #        connE = connect(spike_sourceE, exc_pop[i], weight=0.006, synapse_type='excitatory',delay=2.0)

        input_conns.append(connect(input_pop[i], exc_pop[i], params['w_input_exc'], synapse_type='excitatory'))
    #        connE = connect(spike_sourceE, ifcell, weight=0.006, synapse_type='excitatory',delay=2.0)
    #        connector = AllToAllConnector(weights=params['w_input_exc'])    # weights are drawn from the given random distibution
    #        inh_exc_prj = Projection(input_pop[i], exc_pop, connector)

    # # # # # # # # # # # # # # # # # # # #
    #     P R I N T    W E I G H T S      # 
    # # # # # # # # # # # # # # # # # # # #
    print 'Printing weights to :\n  %s\n  %s\n  %s' % (params['conn_list_ei_fn'], params['conn_list_ie_fn'], params['conn_list_ii_fn'])
    exc_inh_prj.saveConnections(params['conn_list_ei_fn'])
    inh_exc_prj.saveConnections(params['conn_list_ie_fn'])
    inh_inh_prj.saveConnections(params['conn_list_ii_fn'])


    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    B I A S   T O   C E L L S  # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # 
#    print "Setting bias currents ... "
#    bias_currents = []
#    bias_values = np.loadtxt(params['bias_values_fn_base'] + str(sim_cnt) + '.dat')
#    for cell in xrange(params['n_exc']):
#        i_amp = bias_values[cell]
#        dcsource = DCSource({'amplitude' : i_amp})
#        dcsource.inject_into([exc_pop[cell]])
#        bias_currents.append(dcsource)



    #        bias_currents.append(DCSource({'amplitude':i_amp, 'start':0, 'stop':params['t_sim']}))
    #        bias_currents[-1].inject_into([exc_pop[cell]])


    # # # # # # # # # # # # # # # # 
    #     N O I S E   I N P U T   #
    # # # # # # # # # # # # # # # # 
    print "Connecting noise - exc ... "
    noise_pop_exc = []
    noise_pop_inh = []
    for cell in xrange(params['n_exc']):
        if (simulator_name == 'nest'): # for nest one can use the optimized Poisson generator
            noise_pop_exc.append(create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']}))
            noise_pop_inh.append(create(native_cell_type('poisson_generator'), {'rate' : params['f_inh_noise']}))
        else:
            noise_pop_exc.append(create(SpikeSourcePoisson, {'rate' : params['f_exc_noise']}))
            noise_pop_inh.append(create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']}))
        connect(noise_pop_exc[-1], exc_pop[cell], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)
        connect(noise_pop_inh[-1], exc_pop[cell], weight=params['w_inh_noise'], synapse_type='inhibitory', delay=1.)

    print "Connecting noise - inh ... "
    for cell in xrange(params['n_inh']):
        if (simulator_name == 'nest'): # for nest one can use the optimized Poisson generator
            noise_pop_exc.append(create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']}))
            noise_pop_inh.append(create(native_cell_type('poisson_generator'), {'rate' : params['f_inh_noise']}))
        else:
            noise_pop_exc.append(create(SpikeSourcePoisson, {'rate' : params['f_exc_noise']}))
            noise_pop_inh.append(create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']}))
        connect(noise_pop_exc[-1], inh_pop[cell], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)
        connect(noise_pop_inh[-1], inh_pop[cell], weight=params['w_inh_noise'], synapse_type='inhibitory', delay=1.)
    #        noise_prj_exc.append(Projection(noise_pop_exc[-1], exc_pop[cell], connector))


    # # # # # # # # # # # #
    #     R E C O R D     #
    # # # # # # # # # # # #
    print "Recording spikes to file: %s" % (params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
    for cell in xrange(params['n_exc']):
        record(exc_pop[cell], params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)

    n_cells_to_record = params['n_exc'] * 0.02
    record_exc = False
    if os.path.exists(params['gids_to_record_fn']):
        gids_to_record = np.loadtxt(params['gids_to_record_fn'], dtype='int')
        record_exc = True
        gids_to_record = gids_to_record
    else:
        gids_to_record = np.random.randint(0, params['n_exc'], n_cells_to_record)


    exc_pop_view = PopulationView(exc_pop, gids_to_record, label='good_exc_neurons')
    exc_pop_view.record_v()

    inh_pop.record()
    inh_pop.record_v()

    print "Running simulation ... "
    t_pre_run = time.time()
    run(params['t_sim'])
    t_post_run = time.time()
    t_diff = t_post_run - t_pre_run
    print "Simulation time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (t_diff, (t_diff)/60., params['n_cells'], params['n_exc'], params['n_inh'])

    # # # # # # # # # # # # # # # # #
    #     P R I N T    R E S U L T S 
    # # # # # # # # # # # # # # # # #
#    if record_exc:
    print 'print_v to file: %s.v' % (params['exc_volt_fn_base'])
    exc_pop_view.print_v("%s.v" % (params['exc_volt_fn_base']), compatible_output=False)
#        for gid in gids_to_record:
#            exc_pop_view[gid].print_v("%s%d.v" % (params['exc_volt_fn_base'], gid), compatible_output=False)
#            print_v(exc_pop[cell], "%s%d.v" % (params['exc_volt_fn_base'], cell), compatible_output=False)

#            exc_pop[cell].print_v("%s%d.v" % (params['exc_volt_fn_base'], cell), compatible_output=False)
#            exc_pop[cell].printSpikes("%s%d.ras" % (params['exc_spiketimes_fn_base'], cell))
    #    input_pop[cell].printSpikes("%sinput_spikes_%s.ras" % (params['spiketimes_folder'], cell))

    print "Printing inhibitory spikes"
    inh_pop.printSpikes(params['inh_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
    print "Printing inhibitory membrane potentials"
    inh_pop.print_v("%s.v" % (params['inh_volt_fn_base']), compatible_output=False)

    print "calling pyNN.end() ...."
    end()
    t_end = time.time()
    t_diff = t_end - t1
    print "Full pyNN run time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (t_diff, (t_diff)/60., params['n_cells'], params['n_exc'], params['n_inh'])


