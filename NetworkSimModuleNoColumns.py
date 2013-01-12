"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons
"""

import time
import numpy as np
import numpy.random as nprnd
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

        self.tuning_prop_exc = utils.set_tuning_prop(params, mode='hexgrid', cell_type='exc')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
        self.tuning_prop_inh= utils.set_tuning_prop(params, mode='hexgrid', cell_type='inh')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
#        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
#        self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])

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

    def connect(self):
        self.connect_input_to_exc()
        self.connect_ee()
        self.connect_ei()
        self.connect_ie()
        self.connect_ii()
        self.connect_noise()
        self.times['t_connect'] = self.timer.diff()



    def connect_input_to_exc(self, load_files=False):
        """
            # # # # # # # # # # # # # # # # # # # # # # 
            #     C O N N E C T    I N P U T - E X C  #
            # # # # # # # # # # # # # # # # # # # # # # 
        """

        if self.pc_id == 0:
            print "Loading and connecting input spiketrains..."
        if load_files:
            for tgt in self.local_idx_exc:
                try:
                    fn = self.params['input_st_fn_base'] + str(tgt) + '.npy'
                    spike_times = np.load(fn)
                except: # this cell does not get any input
                    print "Missing file: ", fn
                    spike_times = []
                ssa = create(SpikeSourceArray, {'spike_times': spike_times})
                connect(ssa, self.exc_pop[tgt], self.params['w_input_exc'], synapse_type='excitatory')

        else:
            nprnd.seed(self.params['input_spikes_seed'])
            dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
            time = np.arange(0, self.params['t_sim'], dt)
            blank_idx = np.arange(1./dt * self.params['t_stimulus'], 1. / dt * (self.params['t_stimulus'] + self.params['t_blank']))

            my_units = self.local_idx_exc
            n_cells = len(my_units)
            L_input = np.zeros((n_cells, time.shape[0]))
            for i_time, time_ in enumerate(time):
                if (i_time % 100 == 0):
                    print "t:", time_
                L_input[:, i_time] = utils.get_input(self.tuning_prop_exc[my_units, :], self.params, time_/self.params['t_stimulus'])
                L_input[:, i_time] *= self.params['f_max_stim']

            for i_time in blank_idx:
                L_input[:, i_time] = 0.


            for i_, unit in enumerate(my_units):
                rate_of_t = np.array(L_input[i_, :]) 
#                output_fn = self.params['input_rate_fn_base'] + str(unit) + '.npy'
#                np.save(output_fn, rate_of_t)
                # each cell will get its own spike train stored in the following file + cell gid
                n_steps = rate_of_t.size
                spike_times= []
                for i in xrange(n_steps):
                    r = nprnd.rand()
                    if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                        spike_times.append(i * dt) 
                ssa = create(SpikeSourceArray, {'spike_times': spike_times})
                connect(ssa, self.exc_pop[unit], self.params['w_input_exc'], synapse_type='excitatory')


    def connect_anisotropic(self, conn_type):
        """
        """
        if self.pc_id == 0:
            print 'Connect anisotropic %s' % conn_type
        (n_src, n_tgt, src_pop, tgt_pop, tp_src, tp_tgt, tgt_cells, syn_type) = self.resolve_src_tgt(conn_type)

        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_%s_fn_base' % conn_type] + '%d.dat' % (self.pc_id)
            conn_file = open(conn_list_fn, 'w')
            output = ''

        n_src_cells_per_neuron = int(round(self.params['p_%s' % conn_type] * n_src))
        (delay_min, delay_max) = self.params['delay_range']
        for tgt in tgt_cells:
            p = np.zeros(n_src)
            latency = np.zeros(n_src)
            for src in xrange(n_src):
                if conn_type[0] == conn_type[1]: # no self-connection
                    if (src != tgt):
                        p[src], latency[src] = CC.get_p_conn(tp_src[src, :], tp_tgt[tgt, :], params['w_sigma_x'], params['w_sigma_v']) #                            print 'debug pc_id src tgt ', self.pc_id, src, tgt#, int(ID) < self.params['n_exc']
                else: # different populations --> same indices mean different cells, no check for src != tgt
                    p[src], latency[src] = CC.get_p_conn(tp_src[src, :], tp_tgt[tgt, :], params['w_sigma_x'], params['w_sigma_v']) #                            print 'debug pc_id src tgt ', self.pc_id, src, tgt#, int(ID) < self.params['n_exc']

            sorted_indices = np.argsort(p)
            if conn_type[0] == 'e':
                sources = sorted_indices[-n_src_cells_per_neuron:] 
            else:
                sources = sorted_indices[:n_src_cells_per_neuron] 
            w = (self.params['w_tgt_in_per_cell_%s' % conn_type] / p[sources].sum()) * p[sources]
            for i in xrange(len(sources)):
#                        w[i] = max(self.params['w_min'], min(w[i], self.params['w_max']))
                delay = min(max(latency[sources[i]] * self.params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
                connect(src_pop[sources[i]], tgt_pop[tgt], w[i], delay=delay, synapse_type=syn_type)
                if self.debug_connectivity:
                    output += '%d\t%d\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], delay) #                    output += '%d\t%d\t%.2e\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], latency[sources[i]], p[sources[i]])


        if self.debug_connectivity:
            if self.pc_id == 0:
                print 'DEBUG writing to file:', conn_list_fn
            conn_file.write(output)
            conn_file.close()



    def resolve_src_tgt(self, conn_type):
        """
        Deliver the correct source and target parameters based on conn_type
        """

        if conn_type == 'ee':
            n_src, n_tgt = self.params['n_exc'], self.params['n_exc']
            src_pop, tgt_pop = self.exc_pop, self.exc_pop
            tgt_cells = self.local_idx_exc
            tp_src = self.tuning_prop_exc
            tp_tgt = self.tuning_prop_exc
            syn_type = 'excitatory'

        elif conn_type == 'ei':
            n_src, n_tgt = self.params['n_exc'], self.params['n_inh']
            src_pop, tgt_pop = self.exc_pop, self.inh_pop
            tgt_cells = self.local_idx_inh
            tp_src = self.tuning_prop_exc
            tp_tgt = self.tuning_prop_inh
            syn_type = 'excitatory'

        elif conn_type == 'ie':
            n_src, n_tgt = self.params['n_inh'], self.params['n_exc']
            src_pop, tgt_pop = self.inh_pop, self.exc_pop
            tgt_cells = self.local_idx_exc
            tp_src = self.tuning_prop_inh
            tp_tgt = self.tuning_prop_exc
            syn_type = 'inhibitory'

        elif conn_type == 'ii':
            n_src, n_tgt = self.params['n_inh'], self.params['n_inh']
            src_pop, tgt_pop = self.inh_pop, self.inh_pop
            tgt_cells = self.local_idx_inh
            tp_src = self.tuning_prop_inh
            tp_tgt = self.tuning_prop_inh
            syn_type = 'inhibitory'

        return (n_src, n_tgt, src_pop, tgt_pop, tp_src, tp_tgt, tgt_cells, syn_type)





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
        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_ee_fn_base'] + '%d.dat' % (self.pc_id)
            conn_file = open(conn_list_fn, 'w')
            output = ''
        for tgt in self.local_idx_exc:
            p = np.zeros(self.params['n_exc'], dtype='float32')
            latency = np.zeros(self.params['n_exc'], dtype='float32')
            for src in xrange(self.params['n_exc']):
                if (src != tgt):
                    p[src], latency[src] = CC.get_p_conn(self.tuning_prop_exc[src, :], self.tuning_prop_exc[tgt, :], sigma_x, sigma_v) #                            print 'debug pc_id src tgt ', self.pc_id, src, tgt#, int(ID) < self.params['n_exc']
            sources = random.sample(xrange(self.params['n_exc']), int(self.params['n_src_cells_per_neuron']))
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

    def connect_isotropic(self, conn_type='ee'):
        """
        conn_type must be 'ee', 'ei', 'ie' or 'ii'
        Connect cells in a distant dependent manner:
            p_ij = exp(- d_ij / (2 * w_sigma_x**2))

        This will give a 'convergence constrained' connectivity, i.e. each cell will have the same sum of incoming weights 
        ---> could be problematic for outlier cells
        """
        if self.pc_id == 0:
            print 'Drawing isotropic connections'

        (n_src, n_tgt, src_pop, tgt_pop, tp_src, tp_tgt, tgt_cells, syn_type) = self.resolve_src_tgt(conn_type)
        if conn_type == 'ee':
#            p_max = self.params['p_ee']
            p_max = self.params['p_ee_local']
            w_= self.params['w_max']
            w_tgt_in = params['w_tgt_in']

        elif conn_type == 'ei':
            p_max = self.params['p_ee_local']
#            p_max = self.params['p_ei']
            w_= self.params['w_ie']
            w_tgt_in = params['w_tgt_in']

        elif conn_type == 'ie':
            p_max = self.params['p_ee_local']
#            p_max = self.params['p_ie']
            w_= self.params['w_ie']
            w_tgt_in = params['w_tgt_in']

        elif conn_type == 'ii':
            p_max = self.params['p_ee_local']
#            p_max = self.params['p_ii']
            w_= self.params['w_ii']
            w_tgt_in = params['w_tgt_in']

        sigma_x, sigma_v = self.params['w_sigma_x'], self.params['w_sigma_v']
        sigma_x, sigma_v = self.params['w_sigma_x'], self.params['w_sigma_v']
        (delay_min, delay_max) = self.params['delay_range']

        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_%s_fn_base' % conn_type] + '%d.dat' % (self.pc_id)
            conn_file = open(conn_list_fn, 'w')
            output = ''

        for tgt in tgt_cells:
            w = np.zeros(n_src, dtype='float32') 
            for src in xrange(n_src):
                if (src != tgt):
#                    d_ij = np.sqrt((tp_src[src, 0] - tp_tgt[tgt, 0])**2 + (tp_src[src, 1] - tp_tgt[tgt, 1])**2)
                    d_ij = utils.torus_distance2D(tp_src[src, 0], tp_tgt[tgt, 0], tp_src[src, 1], tp_tgt[tgt, 1])
                    p_ij = p_max * np.exp(-d_ij / (2 * params['w_sigma_x']**2))
#                    print 'p_ij', p_ij, np.exp(-d_ij / (2 * params['w_sigma_x']**2))
                    if np.random.rand() <= p_ij:
                        w[src] = w_
            w *= w_tgt_in / w.sum()
            srcs = w.nonzero()[0]
            weights = w[srcs]
            for src in srcs:
                connect(src_pop[int(src)], tgt_pop[int(tgt)], w[src], delay=params['standard_delay'], synapse_type=syn_type)
                output += '%d\t%d\t%.2e\t%.2e\n' % (src, tgt, w[src], params['standard_delay']) 
                    
        if self.debug_connectivity:
            if self.pc_id == 0:
                print 'DEBUG writing to file:', conn_list_fn
            conn_file.write(output)
            conn_file.close()

#   isotropic nearest neighbour code:
#        for tgt in tgt_cells:
#            n_src_to_choose = int(round(p_max * n_src)) # guarantee that all cells have same number of connections
#            dist = np.zeros(n_src, dtype='float32')
#            for src in xrange(n_src):
#                if (src != tgt):
#                    dist[src] = np.sqrt((tp_src[src, 0] - tp_tgt[tgt, 0])**2 + (tp_src[src, 1] - tp_tgt[tgt, 1])**2)
#            src_idx = dist.argsort()[:n_src_to_choose] # choose cells closest to the target
#            for src in src_idx:
#                connect(src_pop[int(src)], tgt_pop[int(tgt)], w_, delay=params['standard_delay'], synapse_type='excitatory')
#                output += '%d\t%d\t%.2e\t%.2e\n' % (src, tgt, w_, params['standard_delay']) 


    def connect_ee(self):
        """
            # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    E X C - E X C  #
            # # # # # # # # # # # # # # # # # # # #
        """
        if self.pc_id == 0:
            print 'Connecting cells exc - exc ...'

        if params['connect_exc_exc']:
            if self.params['connectivity'] == 'anisotropic':
                self.connect_anisotropic('ee')
#                self.connect_ee_convergence_constrained()

            elif self.params['connectivity'] == 'isotropic':
                self.connect_isotropic(conn_type='ee')

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
        if self.params['connectivity'] == 'anisotropic':
            self.connect_anisotropic('ei')

        elif self.params['connectivity'] == 'isotropic':
            self.connect_isotropic(conn_type='ei')
        else:
            self.connect_ei_random()

#        if self.params['selective_inhibition']:
#            conn_list_fn = self.params['conn_list_ei_fn_base'] + '%d.dat' % (self.pc_id)
#            conn_file = open(conn_list_fn, 'w')
#            output = ''
#            if self.pc_id == 0:
#                print "Connecting exc - inh with selective inhibition" 
#            exc_inh_adj = np.loadtxt(self.params['exc_inh_adjacency_list_fn'])
#            for inh in self.local_idx_inh:
#                exc_srcs = exc_inh_adj[inh, :]
#                for exc in exc_srcs:
#                    connect(self.exc_pop[int(exc)], self.inh_pop[int(inh)], self.params['w_ei_mean'], delay=params['standard_delay'], synapse_type='excitatory')
#                    output += '%d\t%d\t%.2e\t%.2e\n' % (exc, inh, self.params['w_ei_mean'], params['standard_delay']) 
#            if self.pc_id == 0:
#                print 'Writing E -> I connections to file:', conn_list_fn
#            conn_file.write(output)
#            conn_file.close()
#        else:
#            if self.pc_id == 0:
#                print "Connecting exc - inh non-selective inhibition" 
#            connector_ei = FastFixedProbabilityConnector(self.params['p_ei'], weights=self.w_ei_dist, delays=self.delay_dist)
#            exc_inh_prj = Projection(self.exc_pop, self.inh_pop, connector_ei, target='excitatory')
#            exc_inh_prj.saveConnections(self.params['merged_conn_list_ei'])

    def connect_ie(self):
        """
            # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    I N H - E X C  #
            # # # # # # # # # # # # # # # # # # # #
        #    connector_ie = FastFixedProbabilityConnector(self.params['p_inh_exc_global'], weights=self.params['w_inh_exc_global'], delays=self.delay_dist)
        #    connector_ie = FromFileConnector(self.params['conn_list_ie_fn'])
        """
        if self.params['connectivity'] == 'anisotropic':
            self.connect_anisotropic('ie')
        elif self.params['connectivity'] == 'isotropic':
            self.connect_isotropic(conn_type='ie')
        else:
            self.connect_ie_random()

#        if self.params['selective_inhibition']:
#            conn_list_fn = self.params['conn_list_ie_fn_base'] + '%d.dat' % (self.pc_id)
#            conn_file = open(conn_list_fn, 'w')
#            output = ''
#            if self.pc_id == 0:
#                print "Connecting inh - exc with selective inhibition"
#            inh_pos = np.loadtxt(self.params['inh_cell_pos_fn'])
#            n_ie = int(round(self.params['p_ie'] * self.params['n_inh']))
#            for exc in self.local_idx_exc:
#                x_e, y_e = self.tuning_prop_exc[exc, 0], self.tuning_prop_exc[exc, 1]
#                dist_ie = np.zeros(self.params['n_inh'])
#                for inh in xrange(self.params['n_inh']):
#                    x_i, y_i = inh_pos[inh, 0], inh_pos[inh, 1]
#                    d_ij = utils.torus_distance2D(x_e, x_i, y_e, y_i)
#                    dist_ie[inh] = d_ij
#                idx = np.argsort(dist_ie)
#                inh_src = idx[:n_ie]
#                for i in xrange(n_ie):
#                    connect(self.inh_pop[int(inh_src[i])], self.exc_pop[int(exc)], self.params['w_ie_mean'], delay=params['standard_delay'], synapse_type='inhibitory')
#                    output += '%d\t%d\t%.2e\t%.2e\n' % (inh_src[i], exc, self.params['w_ie_mean'], params['standard_delay']) 
#            if self.pc_id == 0:
#                print 'Writing I -> E connections to file:', conn_list_fn
#            conn_file.write(output)
#            conn_file.close()

#        else:
#            if self.pc_id == 0:
#                print "Connecting inh - exc ..."
#            connector_ie = FastFixedProbabilityConnector(self.params['p_ie'], weights=self.w_ie_dist, delays=self.delay_dist)
#            inh_exc_prj = Projection(self.inh_pop, self.exc_pop, connector_ie, target='inhibitory')
#            inh_exc_prj.saveConnections(self.params['merged_conn_list_ie'])


    def connect_ii(self):
        """
            # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    I N H - E X C  #
            # # # # # # # # # # # # # # # # # # # #
        #    connector_ii = FastFixedProbabilityConnector(self.params['p_inh_exc_global'], weights=self.params['w_inh_exc_global'], delays=self.delay_dist)
        #    connector_ii = FromFileConnector(self.params['conn_list_ii_fn'])
        """
        if self.params['connectivity'] == 'anisotropic':
            self.connect_anisotropic('ii')
        elif self.params['connectivity'] == 'isotropic':
            self.connect_isotropic(conn_type='ii')
        else:
            self.connect_ii_random()


    def connect_ii_random(self):
        """
            # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    I N H - I N H  #
            # # # # # # # # # # # # # # # # # # # #
        #    connector_ii = FromFileConnector(self.params['conn_list_ii_fn'])
        """
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
