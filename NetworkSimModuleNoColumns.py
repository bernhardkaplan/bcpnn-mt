"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons
"""

import time
import numpy as np
import random as rnd
import sys
import NeuroTools.parameters as ntp
import os
import CreateConnections as CC
import utils

def get_local_indices(pop, offset=0):
    """
    Returns the list of indices (not IDs) local to the MPI node 
    of a population
    """
    list_of_locals = []
    for tgt_id in pop.all():
        tgt = int(tgt_id) - offset - 1 # IDs are 1 aligned
        if pop.is_local(tgt_id) and (tgt < pop.size):
            list_of_locals.append(tgt)
    return list_of_locals

def run_sim(params, sim_cnt, initial_connectivity='precomputed', connect_exc_exc=True): # this function expects a parameter dictionary
    simulator_name = params['simulator']
    exec("from pyNN.%s import *" % simulator_name)
    import pyNN
    print 'pyNN.version: ', pyNN.__version__
    try:
        from mpi4py import MPI
        USE_MPI = True
        comm = MPI.COMM_WORLD
        pc_id, n_proc = comm.rank, comm.size
        print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
    except:
        USE_MPI = False
        pc_id, n_proc, comm = 0, 1, None
        print "MPI not used"

    from pyNN.utility import Timer
    timer = Timer()
    timer.start()
    times = {}
    # # # # # # # # # # # # 
    #     S E T U P       #
    # # # # # # # # # # # #
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

    delay_dist = RandomDistribution('normal',
            (1, 0.01),
            rng=rng_conn,
            constrain='redraw',
            boundaries=(0, 1000))

    times['t_setup'] = timer.diff()

    # # # # # # # # # # # # 
    #     C R E A T E     #
    # # # # # # # # # # # #
    # Excitatory populations
    exc_pop = Population(params['n_exc'], IF_cond_exp, params['cell_params_exc'], label='exc_cells')
    local_idx_exc = get_local_indices(exc_pop, offset=0)
    print 'Debug, pc_id %d has local exc indices:' % pc_id, local_idx_exc
    exc_pop.initialize('v', v_init_dist)

    # Inhibitory population
    inh_pop = Population(params['n_inh'], IF_cond_exp, params['cell_params_inh'], label="inh_pop")
    local_idx_inh = get_local_indices(inh_pop, offset=params['n_exc'])
    print 'Debug, pc_id %d has local inh indices:' % pc_id, local_idx_inh
    inh_pop.initialize('v', v_init_dist)

    times['t_create'] = timer.diff()

    # # # # # # # # # # # # # # # # # # # # # # 
    #     C O N N E C T    I N P U T - E X C  #
    # # # # # # # # # # # # # # # # # # # # # # 
    print "Loading and connecting input spiketrains..."
    for tgt in local_idx_exc:
        try:
            fn = params['input_st_fn_base'] + str(tgt) + '.npy'
            spike_times = np.load(fn)
        except: # this cell does not get any input
            print "Missing file: ", fn
            spike_times = []
        ssa = create(SpikeSourceArray, {'spike_times': spike_times})
        connect(ssa, exc_pop[tgt], params['w_input_exc'], synapse_type='excitatory')

    # # # # # # # # # # # # # # # # # # # #
    #     C O N N E C T    E X C - E X C  #
    # # # # # # # # # # # # # # # # # # # #
    print 'Connecting cells exc - exc ...'
    debug_connectivity = True
    if debug_connectivity:
#        debug_fn = 'debug_output_' + '%d.dat' % (pc_id)
#        debug_file = open(debug_fn, 'w')
#        debug_output = ''
        conn_list_fn = params['conn_list_ee_fn_base'] + '%d.dat' % (pc_id)
        conn_file = open(conn_list_fn, 'w')
        output = ''
    if connect_exc_exc:
        tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])
        sigma_x, sigma_v = params['w_sigma_x'], params['w_sigma_v']
        cnt = 0
        if initial_connectivity == 'precomputed':
            print 'Computing connections ...'
            p_max_local, p_min_local = 0., 0.
            local_weights = [np.zeros(params['n_src_cells_per_neuron']) for i in xrange(params['n_exc'])]
            for i_, tgt in enumerate(local_idx_exc):
                p = np.zeros(params['n_exc'])
                latency = np.zeros(params['n_exc'])
                for src in xrange(params['n_exc']):
                    if (src != tgt):
                        p[src], latency[src] = CC.get_p_conn(tuning_prop[src, :], tuning_prop[tgt, :], sigma_x, sigma_v) #                            print 'debug pc_id src tgt ', pc_id, src, tgt#, int(ID) < params['n_exc']
                sorted_indices = np.argsort(p)
                sources = sorted_indices[-params['n_src_cells_per_neuron']:] 
#                debug_output += 'tgt: %d\t' + str(p[sources]) + '\tsources' + str(sources) + '\n' 
                p_max_local = max(p_max_local, max(p[sources]))
#                p_min_local = min(p_min_local, min(p[sources]))
                local_weights[i_] = p[sources]

            if comm != None:
                # communicate the local p to all other nodes
                # in order to make a proper scaling
                rcvbuf = None # not used
                all_p_max = comm.allgather(p_max_local, rcvbuf)
#                all_p_min = comm.allgather(p_min_local, rcvbuf)
                p_max_global = max(all_p_max)
#                p_min_global = min(all_p_min)
#                print 'debug, pc_id %d p_min_local %f, p_min_global %f' % (pc_id, p_min_local, p_min_global), all_p_min
                w_max = params['w_max'] * p_max_local / p_max_global
                w_min = params['w_min']
#                w_min = params['w_min'] * p_min_local / p_min_global


            for i_, tgt in enumerate(local_idx_exc):
#                w = utils.linear_transformation(local_weights[i_], w_min, w_max)
                w = params['w_tgt_in'] / p[sources].sum() * p[sources]
                for i in xrange(len(sources)):
#                        w[i] = max(params['w_min'], min(w[i], params['w_max']))
                    delay = min(max(latency[sources[i]] * params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
                    connect(exc_pop[sources[i]], exc_pop[tgt], w[i], delay=delay, synapse_type='excitatory')
                    if debug_connectivity:
                        output += '%d\t%d\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], delay) #                    output += '%d\t%d\t%.2e\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], latency[sources[i]], p[sources[i]])
                    cnt += 1

        else: # random TODO
            print 'Drawing random connections'
            for tgt in local_idx_exc:
                p = np.zeros(params['n_exc'], dtype='float32')
                latency = np.zeros(params['n_exc'], dtype='float32')
                for src in xrange(params['n_exc']):
                    if (src != tgt):
                        p[src], latency[src] = CC.get_p_conn(tuning_prop[src, :], tuning_prop[tgt, :], sigma_x, sigma_v) #                            print 'debug pc_id src tgt ', pc_id, src, tgt#, int(ID) < params['n_exc']
                sources = rnd.sample(xrange(params['n_exc']), int(params['n_src_cells_per_neuron']))
                idx = p[sources] > 0
                non_zero_idx = np.nonzero(idx)[0]
                p_ = p[sources][non_zero_idx]
                l_ = latency[sources][non_zero_idx]

#                all_p_sources = p[sources]
#                all_idx = p[sources] > 0
#                non_zero_idx = np.nonzero(all_idx)[0]
#                idx = non_zero_idx
#                info = 'all_p[sources]' + str(all_p_sources)
#                info += 'min p[all_idx] = %d, argmin p[all_idx] = %d, p[argmin] = %.10e, p[argmin] > 0? %d\n' % (np.min(p[all_idx]), np.argmin(p[all_idx]),p[np.argmin(p[all_idx])], p[np.argmin(p[all_idx])] > 0)
#                info += 'min p[non_zero_idx] = %d, argmin p[non_zero_id] = %d, p[argmin] = %.10e, p[argmin] > 0? %d\n' % (np.min(p[non_zero_idx]), np.argmin(p[non_zero_idx]),p[np.argmin(p[non_zero_idx])], p[np.argmin(p[non_zero_idx])] > 0)
#                info += 'p[all_idx]\n' + str(p[all_idx])
#                info += 'p[non_zero_idx]\n' + str(p[non_zero_idx])
#                print info
#                sources = np.unique(sources)
#                non_zero_idx = p[sources] > 1e-8
#                sources = np.nonzero(non_zero_idx)[0]
#                sources = np.nonzero(p[sources] > 1e-2])[0]
#                w = params['w_tgt_in'] / p[sources].sum() * p[sources]
#                a = np.max(p[idx]) / np.min(p[idx])
#                if a == np.inf:
#                    print 'ERRRRRRRRROR p[sources]', p[sources], idx, p[idx], a, np.min(p[idx]), np.max(p[idx])
#                    print 'inf', pc_id
#                    exit(1)

                w = utils.linear_transformation(p_, params['w_min'], params['w_max'])
#                if w = nan:
#                print 'debug pid tgt w', pc_id, tgt, w, '\nnonzeros idx', idx, p[idx]
                for i in xrange(len(p_)):
#                        w[i] = max(params['w_min'], min(w[i], params['w_max']))
                    delay = min(max(l_[i] * params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
                    connect(exc_pop[non_zero_idx[i]], exc_pop[tgt], w[i], delay=delay, synapse_type='excitatory')
                    if debug_connectivity:
                        output += '%d\t%d\t%.2e\t%.2e\n' % (non_zero_idx[i], tgt, w[i], delay) #                    output += '%d\t%d\t%.2e\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], latency[sources[i]], p[sources[i]])
                    cnt += 1
            """
            Different possibilities to do it:
            1) Calculate the weights as above and sample sources randomly
            2) Load a file --> From FileConnector
            3) Create a random distribution
            w_ee_dist = RandomDistribution('normal',
                    (params['w_ee_mean'], params['w_ee_sigma']),
                    rng=rng_conn,
                    constrain='redraw',
                    boundaries=(0, params['w_ee_mean'] * 10.))

            connector_ee = FastFixedProbabilityConnector(params['p_ee'], weights=w_ee_dist, delays=delay_dist)
            prj_ee = Projection(exc_pop, exc_pop, connector_ee, target='excitatory')

            conn_list_fn = params['random_weight_list_fn'] + str(sim_cnt) + '.dat'
            print "Connecting exc - exc from file", conn_list_fn
            connector_ee = FromFileConnector(conn_list_fn)
            prj_ee = Projection(exc_pop, exc_pop, connector_ee, target='excitatory')
            """

    times['t_calc_conns'] = timer.diff()

    if debug_connectivity:
        print 'DEBUG writing to file:', conn_list_fn
        conn_file.write(output)
        conn_file.close()
#        debug_file.write(debug_output)
#        debug_file.close()


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

    times['t_connect'] = timer.diff()

    # # # # # # # # # # # # # # # # # # # #
    #     P R I N T    W E I G H T S      # 
    # # # # # # # # # # # # # # # # # # # #
    print 'Printing weights to :\n  %s\n  %s\n  %s' % (params['conn_list_ei_fn'], params['conn_list_ie_fn'], params['conn_list_ii_fn'])
    exc_inh_prj.saveConnections(params['conn_list_ei_fn'])
    inh_exc_prj.saveConnections(params['conn_list_ie_fn'])
    inh_inh_prj.saveConnections(params['conn_list_ii_fn'])
    times['t_save_conns'] = timer.diff()


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
#    for cell in xrange(params['n_exc']):
    for tgt in local_idx_exc:
        # old
#        if (simulator_name == 'nest'): # for nest one can use the optimized Poisson generator
#            noise_pop_exc.append(create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']}))
#            noise_pop_inh.append(create(native_cell_type('poisson_generator'), {'rate' : params['f_inh_noise']}))
#        else:
#            noise_pop_exc.append(create(SpikeSourcePoisson, {'rate' : params['f_exc_noise']}))
#            noise_pop_inh.append(create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']}))
#        connect(noise_pop_exc[-1], exc_pop[tgt], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)
#        connect(noise_pop_inh[-1], exc_pop[tgt], weight=params['w_inh_noise'], synapse_type='inhibitory', delay=1.)
        #new
        if (simulator_name == 'nest'): # for nest one can use the optimized Poisson generator
            noise_exc = create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']})
            noise_inh = create(native_cell_type('poisson_generator'), {'rate' : params['f_inh_noise']})
        else:
            noise_exc = create(SpikeSourcePoisson, {'rate' : params['f_exc_noise']})
            noise_inh = create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']})
        connect(noise_exc, exc_pop[tgt], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)
        connect(noise_inh, exc_pop[tgt], weight=params['w_inh_noise'], synapse_type='inhibitory', delay=1.)

    print "Connecting noise - inh ... "
#    for cell in xrange(params['n_inh']):
    for tgt in local_idx_inh:
        # old
#        if (simulator_name == 'nest'): # for nest one can use the optimized Poisson generator
#            noise_pop_exc.append(create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']}))
#            noise_pop_inh.append(create(native_cell_type('poisson_generator'), {'rate' : params['f_inh_noise']}))
#        else:
#            noise_pop_exc.append(create(SpikeSourcePoisson, {'rate' : params['f_exc_noise']}))
#            noise_pop_inh.append(create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']}))
#        connect(noise_pop_exc[-1], inh_pop[tgt], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)
#        connect(noise_pop_inh[-1], inh_pop[tgt], weight=params['w_inh_noise'], synapse_type='inhibitory', delay=1.)
        if (simulator_name == 'nest'): # for nest one can use the optimized Poisson generator
            noise_exc = create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']})
            noise_inh = create(native_cell_type('poisson_generator'), {'rate' : params['f_inh_noise']})
        else:
            noise_exc = create(SpikeSourcePoisson, {'rate' : params['f_exc_noise']})
            noise_inh = create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']})
        connect(noise_exc, inh_pop[tgt], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)
        connect(noise_inh, inh_pop[tgt], weight=params['w_inh_noise'], synapse_type='inhibitory', delay=1.)


    # # # # # # # # # # # #
    #     R E C O R D     #
    # # # # # # # # # # # #
#    print "Recording spikes to file: %s" % (params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
#    for cell in xrange(params['n_exc']):
#        record(exc_pop[cell], params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)

    record_exc = True
    if os.path.exists(params['gids_to_record_fn']):
        gids_to_record = np.loadtxt(params['gids_to_record_fn'], dtype='int')[:params['n_gids_to_record']]
        record_exc = True
    else:
        n_cells_to_record = 5# params['n_exc'] * 0.02
        gids_to_record = np.random.randint(0, params['n_exc'], n_cells_to_record)

    exc_pop_view = PopulationView(exc_pop, gids_to_record, label='good_exc_neurons')
    exc_pop_view.record_v()
    inh_pop_view = PopulationView(inh_pop, np.random.randint(0, params['n_inh'], params['n_gids_to_record']), label='random_inh_neurons')
    inh_pop_view.record_v()

    inh_pop.record()
    exc_pop.record()
    times['t_record'] = timer.diff()

    print "Running simulation ... "
    run(params['t_sim'])
    times['t_sim'] = timer.diff()

    # # # # # # # # # # # # # # # # #
    #     P R I N T    R E S U L T S 
    # # # # # # # # # # # # # # # # #
    print 'print_v to file: %s.v' % (params['exc_volt_fn_base'])
    exc_pop_view.print_v("%s.v" % (params['exc_volt_fn_base']), compatible_output=False)
    print "Printing excitatory spikes"
    exc_pop.printSpikes(params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
    print "Printing inhibitory spikes"
    inh_pop.printSpikes(params['inh_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
    print "Printing inhibitory membrane potentials"
    inh_pop_view.print_v("%s.v" % (params['inh_volt_fn_base']), compatible_output=False)

    times['t_print'] = timer.diff()
    print "calling pyNN.end() ...."
    end()
    times['t_end'] = timer.diff()

    if pc_id == 0:
        times['t_all'] = 0.
        for k in times.keys():
            times['t_all'] += times[k]
        times['n_exc'] = params['n_exc']
        times['n_inh'] = params['n_inh']
        times['n_cells'] = params['n_cells']
        times['n_proc'] = n_proc
        times = ntp.ParameterSet(times)
        print "Proc %d Simulation time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (pc_id, times['t_sim'], (times['t_sim'])/60., params['n_cells'], params['n_exc'], params['n_inh'])
        print "Proc %d Full pyNN run time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (pc_id, times['t_all'], (times['t_all'])/60., params['n_cells'], params['n_exc'], params['n_inh'])
        times.save('times_dict_np%d.py' % n_proc)


if __name__ == '__main__':

    import simulation_parameters
    ps = simulation_parameters.parameter_storage()
    ps.create_folders()
    ps.write_parameters_to_file()
    params = ps.params
    sim_cnt = 0
    run_sim(params, sim_cnt, params['initial_connectivity'], params['connect_exc_exc'])

