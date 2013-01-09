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


class NetworkModel(object):

    def __init__(self, params):

        self.params = params
        self.debug_connectivity = True

    def setup(self):
#        try:
        from mpi4py import MPI
        USE_MPI = True
        self.comm = MPI.COMM_WORLD
        self.pc_id, self.n_proc = self.comm.rank, self.comm.size
        print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', self.pc_id, self.n_proc
#        except:
#            USE_MPI = False
#            self.pc_id, self.n_proc, self.comm = 0, 1, None
#            print "MPI not used"

        from pyNN.utility import Timer
        self.timer = Timer()
        self.timer.start()
        self.times = {}
        # # # # # # # # # # # # 
        #     S E T U P       #
        # # # # # # # # # # # #
        (delay_min, delay_max) = self.params['delay_range']
        setup(timestep=0.1, min_delay=delay_min, max_delay=delay_max, rng_seeds_seed=self.params['seed'])
        rng_v = NumpyRNG(seed = sim_cnt*3147 + self.params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes
        rng_conn = NumpyRNG(seed = self.params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes

        # # # # # # # # # # # # # # # # # # # # # # # # #
        #     R A N D O M    D I S T R I B U T I O N S  #
        # # # # # # # # # # # # # # # # # # # # # # # # #
        self.v_init_dist = RandomDistribution('normal',
                (self.params['v_init'], self.params['v_init_sigma']),
                rng=rng_v,
                constrain='redraw',
                boundaries=(-80, -60))

        self.w_ei_dist = RandomDistribution('normal',
                (self.params['w_ei_mean'], self.params['w_ei_sigma']),
                rng=rng_conn,
                constrain='redraw',
                boundaries=(0, self.params['w_ei_mean'] * 10.))

        self.w_ie_dist = RandomDistribution('normal',
                (self.params['w_ie_mean'], self.params['w_ie_sigma']),
                rng=rng_conn,
                constrain='redraw',
                boundaries=(0, self.params['w_ie_mean'] * 10.))

        self.w_ii_dist = RandomDistribution('normal',
                (self.params['w_ii_mean'], self.params['w_ii_sigma']),
                rng=rng_conn,
                constrain='redraw',
                boundaries=(0, self.params['w_ii_mean'] * 10.))

        self.delay_dist = RandomDistribution('normal',
                (1, 0.01),
                rng=rng_conn,
                constrain='redraw',
                boundaries=(0, 1000))

        self.times['t_setup'] = self.timer.diff()

    def create(self):
        """
            # # # # # # # # # # # # 
            #     C R E A T E     #
            # # # # # # # # # # # #
        """
        # Excitatory populations
        self.exc_pop = Population(self.params['n_exc'], IF_cond_exp, self.params['cell_params_exc'], label='exc_cells')
        self.local_idx_exc = get_local_indices(self.exc_pop, offset=0)
        print 'Debug, pc_id %d has local exc indices:' % self.pc_id, self.local_idx_exc
        self.exc_pop.initialize('v', self.v_init_dist)

        # Inhibitory population
        self.inh_pop = Population(self.params['n_inh'], IF_cond_exp, self.params['cell_params_inh'], label="inh_pop")
        self.local_idx_inh = get_local_indices(self.inh_pop, offset=self.params['n_exc'])
        print 'Debug, pc_id %d has local inh indices:' % self.pc_id, self.local_idx_inh
        self.inh_pop.initialize('v', self.v_init_dist)

        self.times['t_create'] = self.timer.diff()

    def connect_input_to_exc(self):
        """
            # # # # # # # # # # # # # # # # # # # # # # 
            #     C O N N E C T    I N P U T - E X C  #
            # # # # # # # # # # # # # # # # # # # # # # 
        """

        if self.pc_id == 0:
            print "Loading and connecting input spiketrains..."
        for tgt in self.local_idx_exc:
            try:
                fn = self.params['input_st_fn_base'] + str(tgt) + '.npy'
                spike_times = np.load(fn)
            except: # this cell does not get any input
                print "Missing file: ", fn
                spike_times = []
            ssa = create(SpikeSourceArray, {'spike_times': spike_times})
            connect(ssa, self.exc_pop[tgt], self.params['w_input_exc'], synapse_type='excitatory')


    def connect_ee_linear_transform(self):
        """
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    E X C - E X C    L I N E A R    W E I G H T   T R A N S F O R M A T I O N    #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        """
        sigma_x, sigma_v = self.params['w_sigma_x'], self.params['w_sigma_v']
        (delay_min, delay_max) = self.params['delay_range']
        cnt = 0
        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_ee_fn_base'] + '%d.dat' % (self.pc_id)
            conn_file = open(conn_list_fn, 'w')
            output = ''
        if self.pc_id == 0:
            print 'Computing connections ...'
        p_max_local, p_min_local = 0., 0.
        local_weights = [np.zeros(self.params['n_src_cells_per_neuron']) for i in xrange(len(self.local_idx_exc))]
        local_sources = [[] for i in xrange(len(self.local_idx_exc))]
        for i_, tgt in enumerate(self.local_idx_exc):
            p = np.zeros(self.params['n_exc'])
            latency = np.zeros(self.params['n_exc'])
            for src in xrange(self.params['n_exc']):
                if (src != tgt):
                    p[src], latency[src] = CC.get_p_conn(self.tuning_prop[src, :], self.tuning_prop[tgt, :], sigma_x, sigma_v) #                            print 'debug pc_id src tgt ', self.pc_id, src, tgt#, int(ID) < self.params['n_exc']
            sorted_indices = np.argsort(p)
            sources = sorted_indices[-self.params['n_src_cells_per_neuron']:] 
            p_max_local = max(p_max_local, max(p[sources]))
#                p_min_local = min(p_min_local, min(p[sources]))
            local_weights[i_] = p[sources]
            local_sources[i_] = sources

        if self.comm != None:
            # self.communicate the local p to all other nodes
            # in order to make a proper scaling
            rcvbuf = None # not used
            all_p_max = self.comm.allgather(p_max_local, rcvbuf)
#                all_p_min = self.comm.allgather(p_min_local, rcvbuf)
            p_max_global = max(all_p_max)
#                p_min_global = min(all_p_min)
            w_max = self.params['w_max'] * p_max_local / p_max_global
            w_min = self.params['w_min']
#                w_min = self.params['w_min'] * p_min_local / p_min_global
        else:
            w_max = self.params['w_max']
            w_min = self.params['w_min']

        for i_, tgt in enumerate(self.local_idx_exc):
            w = utils.linear_transformation(local_weights[i_], w_min, w_max)
            sources = local_sources[i_]
            for i in xrange(len(sources)):
#                        w[i] = max(self.params['w_min'], min(w[i], self.params['w_max']))
                delay = min(max(latency[sources[i]] * self.params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
                connect(self.exc_pop[sources[i]], self.exc_pop[tgt], w[i], delay=delay, synapse_type='excitatory')
                if self.debug_connectivity:
                    output += '%d\t%d\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], delay) 
                cnt += 1

        if self.debug_connectivity:
            if self.pc_id == 0:
                print 'DEBUG writing to file:', conn_list_fn
            conn_file.write(output)
            conn_file.close()


    def connect_ee_convergence_constrained(self):
        """
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    E X C - E X C    L I N E A R    C O N V E R G E N C E   C O N S T R A I N E D    #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        """
        sigma_x, sigma_v = self.params['w_sigma_x'], self.params['w_sigma_v']
        (delay_min, delay_max) = self.params['delay_range']
        cnt = 0
        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_ee_fn_base'] + '%d.dat' % (self.pc_id)
            conn_file = open(conn_list_fn, 'w')
            output = ''
        for i_, tgt in enumerate(self.local_idx_exc):
            p = np.zeros(self.params['n_exc'])
            latency = np.zeros(self.params['n_exc'])
            for src in xrange(self.params['n_exc']):
                if (src != tgt):
                    p[src], latency[src] = CC.get_p_conn(self.tuning_prop[src, :], self.tuning_prop[tgt, :], sigma_x, sigma_v) #                            print 'debug pc_id src tgt ', self.pc_id, src, tgt#, int(ID) < self.params['n_exc']
            sorted_indices = np.argsort(p)
            sources = sorted_indices[-self.params['n_src_cells_per_neuron']:] 
            w = self.params['w_tgt_in'] / p[sources].sum() * p[sources]
            for i in xrange(len(sources)):
#                        w[i] = max(self.params['w_min'], min(w[i], self.params['w_max']))
                delay = min(max(latency[sources[i]] * self.params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
                connect(self.exc_pop[sources[i]], self.exc_pop[tgt], w[i], delay=delay, synapse_type='excitatory')
                if self.debug_connectivity:
                    output += '%d\t%d\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], delay) #                    output += '%d\t%d\t%.2e\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], latency[sources[i]], p[sources[i]])
                cnt += 1
        if self.debug_connectivity:
            if self.pc_id == 0:
                print 'DEBUG writing to file:', conn_list_fn
            conn_file.write(output)
            conn_file.close()



    def connect_ee_random(self):
        """
            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    E X C - E X C    R A N D O M   #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        """

        if self.pc_id == 0:
            print 'Drawing random connections'
        sigma_x, sigma_v = self.params['w_sigma_x'], self.params['w_sigma_v']
        (delay_min, delay_max) = self.params['delay_range']
        cnt = 0
        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_ee_fn_base'] + '%d.dat' % (self.pc_id)
            conn_file = open(conn_list_fn, 'w')
            output = ''
        for tgt in self.local_idx_exc:
            p = np.zeros(self.params['n_exc'], dtype='float32')
            latency = np.zeros(self.params['n_exc'], dtype='float32')
            for src in xrange(self.params['n_exc']):
                if (src != tgt):
                    p[src], latency[src] = CC.get_p_conn(self.tuning_prop[src, :], self.tuning_prop[tgt, :], sigma_x, sigma_v) #                            print 'debug pc_id src tgt ', self.pc_id, src, tgt#, int(ID) < self.params['n_exc']
            sources = rnd.sample(xrange(self.params['n_exc']), int(self.params['n_src_cells_per_neuron']))
            idx = p[sources] > 0
            non_zero_idx = np.nonzero(idx)[0]
            p_ = p[sources][non_zero_idx]
            l_ = latency[sources][non_zero_idx]

            w = utils.linear_transformation(p_, self.params['w_min'], self.params['w_max'])
#                if w = nan:
#                print 'debug pid tgt w', self.pc_id, tgt, w, '\nnonzeros idx', idx, p[idx]
            for i in xrange(len(p_)):
#                        w[i] = max(self.params['w_min'], min(w[i], self.params['w_max']))
                delay = min(max(l_[i] * self.params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
                connect(self.exc_pop[non_zero_idx[i]], self.exc_pop[tgt], w[i], delay=delay, synapse_type='excitatory')
                if self.debug_connectivity:
                    output += '%d\t%d\t%.2e\t%.2e\n' % (non_zero_idx[i], tgt, w[i], delay) #                    output += '%d\t%d\t%.2e\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], latency[sources[i]], p[sources[i]])
                cnt += 1

        if self.debug_connectivity:
            if self.pc_id == 0:
                print 'DEBUG writing to file:', conn_list_fn
            conn_file.write(output)
            conn_file.close()

        """
        Different possibilities to draw random connections:
        1) Calculate the weights as above and sample sources randomly
        2) Load a file --> From FileConnector
        3) Create a random distribution with similar parameters as the non-random connectivition distribution
        w_ee_dist = RandomDistribution('normal',
                (self.params['w_ee_mean'], self.params['w_ee_sigma']),
                rng=rng_conn,
                constrain='redraw',
                boundaries=(0, self.params['w_ee_mean'] * 10.))

        connector_ee = FastFixedProbabilityConnector(self.params['p_ee'], weights=w_ee_dist, delays=self.delay_dist)
        prj_ee = Projection(self.exc_pop, self.exc_pop, connector_ee, target='excitatory')

        conn_list_fn = self.params['random_weight_list_fn'] + str(sim_cnt) + '.dat'
        print "Connecting exc - exc from file", conn_list_fn
        connector_ee = FromFileConnector(conn_list_fn)
        prj_ee = Projection(self.exc_pop, self.exc_pop, connector_ee, target='excitatory')
        """


    def connect_ee(self):
        """
            # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    E X C - E X C  #
            # # # # # # # # # # # # # # # # # # # #
        """
        if self.pc_id == 0:
            print 'Connecting cells exc - exc ...'

        if params['connect_exc_exc']:
            self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
            if self.params['initial_connectivity'] == 'precomputed_linear_transform':
                self.connect_ee_linear_transform()

            elif self.params['initial_connectivity'] == 'precomputed_convergence_constrained':
                self.connect_ee_convergence_constrained()
            else:
                self.connect_ee_random()
        self.times['t_calc_conns'] = self.timer.diff()

    def connect_ei(self):
        """
            # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    E X C - I N H  #
            # # # # # # # # # # # # # # # # # # # #
        #    connector_ei = FastFixedProbabilityConnector(self.params['p_exc_inh_global'], weights=self.params['w_exc_inh_global'], delays=self.delay_dist)
        #    connector_ei = FromFileConnector(self.params['conn_list_ei_fn'])
        """
        if self.params['selective_inhibition']:
            conn_list_fn = self.params['conn_list_ei_fn_base'] + '%d.dat' % (self.pc_id)
            conn_file = open(conn_list_fn, 'w')
            output = ''

            if self.pc_id == 0:
                print "Connecting exc - inh with selective inhibition" 
            exc_inh_adj = np.loadtxt(self.params['exc_inh_adjacency_list_fn'])
            for inh in self.local_idx_inh:
                exc_srcs = exc_inh_adj[inh, :]
                for exc in exc_srcs:
                    connect(self.exc_pop[int(exc)], self.inh_pop[int(inh)], self.params['w_ei_mean'], delay=2, synapse_type='excitatory')
                    output += '%d\t%d\t%.2e\t%.2e\n' % (exc, inh, self.params['w_ei_mean'], 2) 
            if self.pc_id == 0:
                print 'Writing E -> I connections to file:', conn_list_fn
            conn_file.write(output)
            conn_file.close()
        else:
            if self.pc_id == 0:
                print "Connecting exc - inh non-selective inhibition" 
            connector_ei = FastFixedProbabilityConnector(self.params['p_ei'], weights=self.w_ei_dist, delays=self.delay_dist)
            exc_inh_prj = Projection(self.exc_pop, self.inh_pop, connector_ei, target='excitatory')
            exc_inh_prj.saveConnections(self.params['merged_conn_list_ei'])

    def connect_ie(self):
        """
            # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    I N H - E X C  #
            # # # # # # # # # # # # # # # # # # # #
        #    connector_ie = FastFixedProbabilityConnector(self.params['p_inh_exc_global'], weights=self.params['w_inh_exc_global'], delays=self.delay_dist)
        #    connector_ie = FromFileConnector(self.params['conn_list_ie_fn'])
        """
        if self.params['selective_inhibition']:
            conn_list_fn = self.params['conn_list_ie_fn_base'] + '%d.dat' % (self.pc_id)
            conn_file = open(conn_list_fn, 'w')
            output = ''
            if self.pc_id == 0:
                print "Connecting inh - exc with selective inhibition"
            inh_pos = np.loadtxt(self.params['inh_cell_pos_fn'])
            self.tuning_prop = np.loadtxt(self.params['self.tuning_prop_means_fn'])
            n_ie = int(round(self.params['p_ie'] * self.params['n_inh']))
            for exc in self.local_idx_exc:
                x_e, y_e = self.tuning_prop[exc, 0], self.tuning_prop[exc, 1]
                dist_ie = np.zeros(self.params['n_inh'])
                for inh in xrange(self.params['n_inh']):
                    x_i, y_i = inh_pos[inh, 0], inh_pos[inh, 1]
                    dx, dy = utils.torus_distance(x_e, x_i), utils.torus_distance(y_e, y_i)
                    dist_ie[inh] = np.sqrt(dx**2 + dy**2)
                idx = np.argsort(dist_ie)
                inh_src = idx[:n_ie]
                for i in xrange(n_ie):
                    connect(self.inh_pop[int(inh_src[i])], self.exc_pop[int(exc)], self.params['w_ie_mean'], delay=2, synapse_type='inhibitory')
                    output += '%d\t%d\t%.2e\t%.2e\n' % (inh_src[i], exc, self.params['w_ie_mean'], 2) 
            if self.pc_id == 0:
                print 'Writing I -> E connections to file:', conn_list_fn
            conn_file.write(output)
            conn_file.close()

        else:
            if self.pc_id == 0:
                print "Connecting inh - exc ..."
            connector_ie = FastFixedProbabilityConnector(self.params['p_ie'], weights=self.w_ie_dist, delays=self.delay_dist)
            inh_exc_prj = Projection(self.inh_pop, self.exc_pop, connector_ie, target='inhibitory')
            inh_exc_prj.saveConnections(self.params['merged_conn_list_ie'])


    def connect_ii(self):
        """
            # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    I N H - I N H  #
            # # # # # # # # # # # # # # # # # # # #
        """
        #    connector_ii = FromFileConnector(self.params['conn_list_ii_fn'])
        if self.pc_id == 0:
            print "Connecting inh - inh ..."
        connector_ii = FastFixedProbabilityConnector(self.params['p_ii'], weights=self.w_ii_dist, delays=self.delay_dist)
        inh_inh_prj = Projection(self.inh_pop, self.inh_pop, connector_ii, target='inhibitory')
        inh_inh_prj.saveConnections(self.params['merged_conn_list_ii'])



    def connect_noise(self):
        """
            # # # # # # # # # # # # # # # # 
            #     N O I S E   I N P U T   #
            # # # # # # # # # # # # # # # # 
        """
        if self.pc_id == 0:
            print "Connecting noise - exc ... "
        noise_pop_exc = []
        noise_pop_inh = []
        for tgt in self.local_idx_exc:
            #new
            if (self.params['simulator'] == 'nest'): # for nest one can use the optimized Poisson generator
                noise_exc = create(native_cell_type('poisson_generator'), {'rate' : self.params['f_exc_noise']})
                noise_inh = create(native_cell_type('poisson_generator'), {'rate' : self.params['f_inh_noise']})
            else:
                noise_exc = create(SpikeSourcePoisson, {'rate' : self.params['f_exc_noise']})
                noise_inh = create(SpikeSourcePoisson, {'rate' : self.params['f_inh_noise']})
            connect(noise_exc, self.exc_pop[tgt], weight=self.params['w_exc_noise'], synapse_type='excitatory', delay=1.)
            connect(noise_inh, self.exc_pop[tgt], weight=self.params['w_inh_noise'], synapse_type='inhibitory', delay=1.)

        if self.pc_id == 0:
            print "Connecting noise - inh ... "
        for tgt in self.local_idx_inh:
            if (self.params['simulator'] == 'nest'): # for nest one can use the optimized Poisson generator
                noise_exc = create(native_cell_type('poisson_generator'), {'rate' : self.params['f_exc_noise']})
                noise_inh = create(native_cell_type('poisson_generator'), {'rate' : self.params['f_inh_noise']})
            else:
                noise_exc = create(SpikeSourcePoisson, {'rate' : self.params['f_exc_noise']})
                noise_inh = create(SpikeSourcePoisson, {'rate' : self.params['f_inh_noise']})
            connect(noise_exc, self.inh_pop[tgt], weight=self.params['w_exc_noise'], synapse_type='excitatory', delay=1.)
            connect(noise_inh, self.inh_pop[tgt], weight=self.params['w_inh_noise'], synapse_type='inhibitory', delay=1.)



    def connect(self):
        self.connect_input_to_exc()
        self.connect_ee()
        self.connect_ei()
        self.connect_ie()
        self.connect_ii()
        self.connect_noise()
        self.times['t_connect'] = self.timer.diff()
        # # # # # # # # # # # # # # # # # # # # # # # # # #
        #     C O N N E C T    B I A S   T O   C E L L S  # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #    print "Setting bias currents ... "
        #    bias_currents = []
        #    bias_values = np.loadtxt(self.params['bias_values_fn_base'] + str(sim_cnt) + '.dat')
        #    for cell in xrange(self.params['n_exc']):
        #        i_amp = bias_values[cell]
        #        dcsource = DCSource({'amplitude' : i_amp})
        #        dcsource.inject_into([self.exc_pop[cell]])
        #        bias_currents.append(dcsource)
        #        bias_currents.append(DCSource({'amplitude':i_amp, 'start':0, 'stop':self.params['t_sim']}))
        #        bias_currents[-1].inject_into([self.exc_pop[cell]])



    def run_sim(self, sim_cnt): # this function expects a parameter dictionary
        # # # # # # # # # # # # # # # # # # # #
        #     P R I N T    W E I G H T S      # 
        # # # # # # # # # # # # # # # # # # # #
    #    print 'Printing weights to :\n  %s\n  %s\n  %s' % (self.params['conn_list_ei_fn'], self.params['conn_list_ie_fn'], self.params['conn_list_ii_fn'])
    #    exc_inh_prj.saveConnections(self.params['conn_list_ei_fn'])
    #    inh_exc_prj.saveConnections(self.params['conn_list_ie_fn'])
    #    inh_inh_prj.saveConnections(self.params['conn_list_ii_fn'])
    #    self.times['t_save_conns'] = self.timer.diff()


        # # # # # # # # # # # #
        #     R E C O R D     #
        # # # # # # # # # # # #
    #    print "Recording spikes to file: %s" % (self.params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
    #    for cell in xrange(self.params['n_exc']):
    #        record(self.exc_pop[cell], self.params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
        record_exc = True
        if os.path.exists(self.params['gids_to_record_fn']):
            gids_to_record = np.loadtxt(self.params['gids_to_record_fn'], dtype='int')[:self.params['n_gids_to_record']]
            record_exc = True
        else:
            n_cells_to_record = 5# self.params['n_exc'] * 0.02
            gids_to_record = np.random.randint(0, self.params['n_exc'], n_cells_to_record)

        self.exc_pop_view = PopulationView(self.exc_pop, gids_to_record, label='good_exc_neurons')
        self.exc_pop_view.record_v()
        self.inh_pop_view = PopulationView(self.inh_pop, np.random.randint(0, self.params['n_inh'], self.params['n_gids_to_record']), label='random_inh_neurons')
        self.inh_pop_view.record_v()

        self.inh_pop.record()
        self.exc_pop.record()
        self.times['t_record'] = self.timer.diff()

        if self.pc_id == 0:
            print "Running simulation ... "
        run(self.params['t_sim'])
        self.times['t_sim'] = self.timer.diff()

    def print_results(self):
        """
            # # # # # # # # # # # # # # # # #
            #     P R I N T    R E S U L T S 
            # # # # # # # # # # # # # # # # #
        """
        if self.pc_id == 0:
            print 'print_v to file: %s.v' % (self.params['exc_volt_fn_base'])
        self.exc_pop_view.print_v("%s.v" % (self.params['exc_volt_fn_base']), compatible_output=False)
        if self.pc_id == 0:
            print "Printing excitatory spikes"
        self.exc_pop.printSpikes(self.params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
    #    nspikes = self.exc_pop.get_spike_counts(gather=False)
    #    print '%d get spike counts:', nspikes

        if self.pc_id == 0:
            print "Printing inhibitory spikes"
        self.inh_pop.printSpikes(self.params['inh_spiketimes_fn_merged'] + '%d.ras' % sim_cnt)
        if self.pc_id == 0:
            print "Printing inhibitory membrane potentials"
        self.inh_pop_view.print_v("%s.v" % (self.params['inh_volt_fn_base']), compatible_output=False)

        self.times['t_print'] = self.timer.diff()
        if self.pc_id == 0:
            print "calling pyNN.end() ...."
        end()
        self.times['t_end'] = self.timer.diff()

        if self.pc_id == 0:
            self.times['t_all'] = 0.
            for k in self.times.keys():
                self.times['t_all'] += self.times[k]
            self.times['n_exc'] = self.params['n_exc']
            self.times['n_inh'] = self.params['n_inh']
            self.times['n_cells'] = self.params['n_cells']
            self.times['n_proc'] = self.n_proc
            self.times = ntp.ParameterSet(self.times)
            print "Proc %d Simulation time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (self.pc_id, self.times['t_sim'], (self.times['t_sim'])/60., self.params['n_cells'], self.params['n_exc'], self.params['n_inh'])
            print "Proc %d Full pyNN run time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (self.pc_id, self.times['t_all'], (self.times['t_all'])/60., self.params['n_cells'], self.params['n_exc'], self.params['n_inh'])
            self.times.save(params['folder_name'] + 'times_dict_np%d.py' % self.n_proc)


if __name__ == '__main__':

    import simulation_parameters
    ps = simulation_parameters.parameter_storage()
    ps.create_folders()
    ps.write_parameters_to_file()
    params = ps.params
    sim_cnt = 0

    exec("from pyNN.%s import *" % params['simulator'])
    import pyNN
    print 'pyNN.version: ', pyNN.__version__

    NM = NetworkModel(params)
    NM.setup()
    NM.create()
    NM.connect()
    NM.run_sim(sim_cnt)
    NM.print_results()
