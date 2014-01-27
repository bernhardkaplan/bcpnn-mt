"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons

on the cluster:
    frioul_batch -M "[['w_tgt_in_per_cell_ee', 'w_tgt_in_per_cell_ee', 'w_tgt_in_per_cell_ee'],[0.4, 0.8, 1.2]]" 'python NetworkSimModuleNoColumns.py'


"""
import time
times = {}
t0 = time.time()
import numpy as np
import numpy.random as nprnd
import sys
import os
import json
import CreateConnections as CC
import utils
import simulation_parameters

import pyNN
import pyNN.space as space
from pyNN.utility import Timer # for measuring the times to connect etc.
print 'pyNN.version: ', pyNN.__version__
try:
#    I_fail_because_I_do_not_want_to_use_MPI
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"
times['time_to_import'] = time.time() - t0


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

    def __init__(self, params, comm):
        """
        params: the container that stores all the parameters
        comm: MPI communicator
        """

        self.params = params
        self.debug_connectivity = True # should be true if you want to plot connectivity profiles etc
        self.comm = comm
        if self.comm != None:
            self.pc_id, self.n_proc = self.comm.rank, self.comm.size
            print "USE_MPI: yes", '\tpc_id, n_proc:', self.pc_id, self.n_proc
        else:
            self.pc_id, self.n_proc = 0, 1
            print "MPI not used"

        np.random.seed(params['np_random_seed'] + self.pc_id)
        self.anticipatory_record = False



    def setup(self, load_tuning_prop=False, times={}):

        self.projections = {}
        self.projections['ee'] = []
        self.projections['ei'] = []
        self.projections['ie'] = []
        self.projections['ii'] = []
        if not load_tuning_prop:
            self.tuning_prop_exc = utils.set_tuning_prop(self.params, mode='hexgrid', cell_type='exc')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
            self.tuning_prop_inh = utils.set_tuning_prop(self.params, mode='hexgrid', cell_type='inh')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
        else:
            self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
            self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])

        indices, distances = utils.sort_gids_by_distance_to_stimulus(self.tuning_prop_exc, self.params['motion_params'], self.params) # cells in indices should have the highest response to the stimulus
        if self.pc_id == 0:
            print "Saving tuning_prop to file:", self.params['tuning_prop_means_fn']
            np.savetxt(self.params['tuning_prop_means_fn'], self.tuning_prop_exc)
            print "Saving tuning_prop to file:", self.params['tuning_prop_inh_fn']
            np.savetxt(self.params['tuning_prop_inh_fn'], self.tuning_prop_inh)
            print 'Saving gids to record to: ', self.params['gids_to_record_fn']
            np.savetxt(self.params['gids_to_record_fn'], indices[:self.params['n_gids_to_record']], fmt='%d')

        if self.comm != None:
            self.comm.Barrier()
        self.timer = Timer()
        self.timer.start()
        self.times = times
        self.times['t_all'] = 0
        # # # # # # # # # # # #
        #     S E T U P       #
        # # # # # # # # # # # #
        (delay_min, delay_max) = self.params['delay_range']
        setup(timestep=0.1, min_delay=delay_min, max_delay=delay_max, rng_seeds_seed=self.params['seed'])
        rng_v = NumpyRNG(seed = sim_cnt*3147 + self.params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes
        self.rng_conn = NumpyRNG(seed = self.params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes

        # # # # # # # # # # # # # # # # # # # # # # # # #
        #     R A N D O M    D I S T R I B U T I O N S  #
        # # # # # # # # # # # # # # # # # # # # # # # # #
        self.v_init_dist = RandomDistribution('normal',
                (self.params['v_init'], self.params['v_init_sigma']),
                rng=rng_v,
                constrain='redraw',
                boundaries=(-80, -60))

        self.times['t_setup'] = self.timer.diff()
        self.times['t_calc_conns'] = 0
        if self.comm != None:
            self.comm.Barrier()
        self.torus = space.Space(axes='xy', periodic_boundaries=((0., self.params['torus_width']), (0., self.params['torus_height'])))


    def create(self, input_created=False):
        """
            # # # # # # # # # # # #
            #     C R E A T E     #
            # # # # # # # # # # # #
            create the Populations and initialize the membrane potentials
        """
        # choose the neuron model
        if self.params['neuron_model'] == 'IF_cond_exp':
            self.exc_pop = Population(self.params['n_exc'], IF_cond_exp, self.params['cell_params_exc'], label='exc_cells')
            self.inh_pop = Population(self.params['n_inh'], IF_cond_exp, self.params['cell_params_inh'], label="inh_pop")
        elif self.params['neuron_model'] == 'IF_cond_alpha':
            self.exc_pop = Population(self.params['n_exc'], IF_cond_alpha, self.params['cell_params_exc'], label='exc_cells')
            self.inh_pop = Population(self.params['n_inh'], IF_cond_alpha, self.params['cell_params_inh'], label="inh_pop")
        elif self.params['neuron_model'] == 'EIF_cond_exp_isfa_ista':
            self.exc_pop = Population(self.params['n_exc'], EIF_cond_exp_isfa_ista, self.params['cell_params_exc'], label='exc_cells')
            self.inh_pop = Population(self.params['n_inh'], EIF_cond_exp_isfa_ista, self.params['cell_params_inh'], label="inh_pop")
        else:
            print '\n\nUnknown neuron model:\n\t', self.params['neuron_model']
        self.local_idx_exc = get_local_indices(self.exc_pop, offset=0)
        self.local_idx_inh = get_local_indices(self.inh_pop, offset=self.params['n_exc'])
        print 'Debug, pc_id %d has local %d exc indices:' % (self.pc_id, len(self.local_idx_exc)), self.local_idx_exc
        print 'Debug, pc_id %d has local %d inh indices:' % (self.pc_id, len(self.local_idx_inh)), self.local_idx_inh

        cell_pos_exc = np.zeros((3, self.params['n_exc']))
        cell_pos_exc[0, :] = self.tuning_prop_exc[:, 0]
        cell_pos_exc[1, :] = self.tuning_prop_exc[:, 1]
        self.exc_pop.positions = cell_pos_exc

        cell_pos_inh = np.zeros((3, self.params['n_inh']))
        cell_pos_inh[0, :] = self.tuning_prop_inh[:, 0]
        cell_pos_inh[1, :] = self.tuning_prop_inh[:, 1]
        self.inh_pop.positions = cell_pos_inh

        if not input_created:
            self.spike_times_container = [ [] for i in xrange(len(self.local_idx_exc))]

        self.exc_pop.initialize('v', self.v_init_dist)
        self.inh_pop.initialize('v', self.v_init_dist)
        self.times['t_create'] = self.timer.diff()


    def connect(self):
        if self.params['n_exc'] > 5000:
            save_output = False
        else:
            save_output = True

        self.connect_input_to_exc()
        self.connect_populations('ee')
        self.connect_populations('ei')
        self.connect_populations('ie')
        self.connect_populations('ii')
        self.connect_noise()
        self.times['t_calc_conns'] = self.timer.diff()
        if self.comm != None:
            self.comm.Barrier()


    def create_input(self, load_files=False, save_output=False):


        if load_files:
            if self.pc_id == 0:
                print "Loading input spiketrains..."
            for i_, tgt in enumerate(self.local_idx_exc):
                try:
                    fn = self.params['input_st_fn_base'] + str(tgt) + '.npy'
                    spike_times = np.load(fn)
                except: # this cell does not get any input
                    print "Missing file: ", fn
                    spike_times = []
                self.spike_times_container[i_] = spike_times
        else:
            if self.pc_id == 0:
                print "Computing input spiketrains..."
            nprnd.seed(self.params['input_spikes_seed'])
            dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process
            time = np.arange(0, self.params['t_sim'], dt)
            blank_idx = np.arange(1./dt * self.params['t_before_blank'], 1. / dt * (self.params['t_before_blank'] + self.params['t_blank']))
            before_stim_idx = np.arange(0, self.params['t_start'] * 1./dt)
            blank_idx = np.concatenate((blank_idx, before_stim_idx))


            my_units = self.local_idx_exc
            n_cells = len(my_units)
            L_input = np.zeros((n_cells, time.shape[0]))

            # get the input signal
            print 'Calculating input signal'
            for i_time, time_ in enumerate(time):
#                L_input[:, i_time] = utils.get_input(self.tuning_prop_exc[my_units, :], self.params, time_/self.params['t_stimulus'])
                L_input[:, i_time] = utils.get_input_delay(self.tuning_prop_exc[my_units, :], self.params, time_/self.params['t_stimulus'])
                L_input[:, i_time] *= self.params['f_max_stim']
                if (i_time % 500 == 0):
                    print "t:", time_
#                    print 'L_input[:, %d].max()', L_input[:, i_time].max()
            # blanking
            for i_time in blank_idx:
#                L_input[:, i_time] = 0.
                L_input[:, i_time] = np.random.permutation(L_input[:, i_time])

            # create the spike trains
            print 'Creating input spiketrains for unit'
            for i_, unit in enumerate(my_units):
                print unit,
                rate_of_t = np.array(L_input[i_, :])
                # each cell will get its own spike train stored in the following file + cell gid
                n_steps = rate_of_t.size
                spike_times = []
                for i in xrange(n_steps):
                    r = nprnd.rand()
                    if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                        spike_times.append(i * dt)
                self.spike_times_container[i_] = spike_times
                if save_output:
                    output_fn = self.params['input_rate_fn_base'] + str(unit) + '.npy'
                    np.save(output_fn, rate_of_t)
                    output_fn = self.params['input_st_fn_base'] + str(unit) + '.npy'
                    np.save(output_fn, np.array(spike_times))

        self.times['create_input'] = self.timer.diff()
        return self.spike_times_container



    def connect_input_to_exc(self):
        """
            # # # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    I N P U T - E X C  #
            # # # # # # # # # # # # # # # # # # # # # #
        """
        if self.pc_id == 0:
            print "Connecting input spiketrains..."
        for i_, unit in enumerate(self.local_idx_exc):
            spike_times = self.spike_times_container[i_]
            ssa = create(SpikeSourceArray, {'spike_times': spike_times})
            connect(ssa, self.exc_pop[unit], self.params['w_input_exc'], synapse_type='excitatory')
        self.times['connect_input'] = self.timer.diff()


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


    def connect_anisotropic(self, conn_type):
        """
        conn_type = ['ee', 'ei', 'ie', 'ii']
        """
        if self.pc_id == 0:
            print 'Connect anisotropic %s - %s' % (conn_type[0].capitalize(), conn_type[1].capitalize())

        (n_src, n_tgt, src_pop, tgt_pop, tp_src, tp_tgt, tgt_cells, syn_type) = self.resolve_src_tgt(conn_type)

        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_%s_fn_base' % conn_type] + '%d.dat' % (self.pc_id)

        n_src_cells_per_neuron = int(round(self.params['p_%s' % conn_type] * n_src))
        (delay_min, delay_max) = self.params['delay_range']
        local_connlist = np.zeros((n_src_cells_per_neuron * len(tgt_cells), 4))
        # we connect the source population (tp_src) to each target neuron in the target population (tgt_cells)
        for i_, tgt in enumerate(tgt_cells):
            if self.params['conn_conf'] == 'direction-based':
                p, latency = CC.get_p_conn_direction_based(params, tp_src, tp_tgt[tgt, :])
            elif self.params['conn_conf'] == 'motion-based':
                p, latency = CC.get_p_conn_motion_based(params, tp_src, tp_tgt[tgt, :])
            elif self.params['conn_conf'] == 'orientation-direction':
                p, latency = CC.get_p_conn_direction_and_orientation_based(params, tp_src, tp_tgt[tgt, :])
            else:
                print '\n\nERROR! Wrong connection configuration conn_conf parameter provided\nShould be direction-based, motion-based or orientation-direction\n'
                exit(1)

            # avoid autapses:
            if conn_type[0] == conn_type[1]:
                p[tgt], latency[tgt] = 0., 0.

            # random delays? --> np.permutate(latency) or latency[sources] * self.params['delay_scale'] * np.rand
            #print latency, latency.mean()#, np.nonzero(self.params['delay_range'][0] >  np.nonzero(self.params['delay_range'][0] > latency * self.params['delay_scale'] >  self.params['delay_range'][1] + latency * self.params['delay_scale'] > self.params['delay_range'][1])
            invalid_idx = np.nonzero(latency * self.params['delay_scale'] < self.params['delay_range'][0])[0]
            print 'DEBUG: # of neurons with a too short latency = ', invalid_idx.size, 'total number of source = ', tp_src[:, 0].size
            p[invalid_idx], latency[invalid_idx] = 0., 0.
            invalid_idx = np.nonzero(latency * self.params['delay_scale'] > self.params['delay_range'][1])[0]
            print 'DEBUG: # of neurons with a too long latency = ', invalid_idx.size, 'total number of source = ', tp_src[:, 0].size
            p[invalid_idx], latency[invalid_idx] = 0., 0.

            sorted_indices = np.argsort(p) # from smallest to biggest probability
            if conn_type[0] == 'e':
                sources = sorted_indices[-n_src_cells_per_neuron:]
            else: # source = inhibitory
                # laurent: I am not sure this is the correct way to connect inhibitory - but we use isotropic, right?
                if conn_type[0] == conn_type[1]:
                    sources = sorted_indices[1:n_src_cells_per_neuron+1]  # shift indices to avoid self-connection, because p_ii = .0
                else:
                    sources = sorted_indices[:n_src_cells_per_neuron]

#            print 'debug sources', sources.size
            assert (sources.size > 0)
            eta = 1e-12
            # equal weights:
            if self.params['equal_weights']:
                p_to_w = np.zeros(n_src)
                p_to_w[sources] = 1.
                w = (self.params['w_tgt_in_per_cell_%s' % conn_type] / (p_to_w.sum() + eta)) * p_to_w[sources]
            else:
                # unequal weights:
                w = (self.params['w_tgt_in_per_cell_%s' % conn_type] / (p[sources].sum() + eta)) * p[sources]

#            print 'debug p', i_, tgt, p[sources]
#            print 'debug sources', i_, tgt, sources
#            print 'debug w', i_, tgt, w

            delays = np.minimum(np.maximum(latency[sources] * self.params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
            #delays = latency[sources] * self.params['delay_scale']
            #delays[p[sources] == 0.] = 0.

#            delays = (self.params['delay_range'][1] - self.params['delay_range'][0]) * np.random.rand(sources.size) + self.params['delay_range'][0]

#            w = np.minimum(np.maximum(w, self.params['w_min']), self.params['w_max'])  # map the weights into a certain range
            conn_list = np.array((sources, tgt * np.ones(n_src_cells_per_neuron), w, delays))
            local_connlist[i_ * n_src_cells_per_neuron : (i_ + 1) * n_src_cells_per_neuron, :] = conn_list.transpose()
            connector = FromListConnector(conn_list.transpose())
            # laurent: here there is something I do not get: you are still connecting tg_src to tgt, but you issue a connector to the whole population... isn't there an indentation issue?
            prj = Projection(src_pop, tgt_pop, connector)
            self.projections[conn_type].append(prj)

        if self.debug_connectivity:
            if self.pc_id == 0:
                print 'DEBUG writing to file:', conn_list_fn
            np.savetxt(conn_list_fn, local_connlist, fmt='%d\t%d\t%.4e\t%.4e')


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
                    p[src], latency[src] = CC.get_p_conn(self.tuning_prop_exc[src, :], self.tuning_prop_exc[tgt, :], sigma_x, sigma_v, params['connectivity_radius']) 
            sources = random.sample(xrange(self.params['n_exc']), int(self.params['n_src_cells_per_neuron']))
            idx = p[sources] > 0
            non_zero_idx = np.nonzero(idx)[0]
            p_ = p[sources][non_zero_idx]
            l_ = latency[sources][non_zero_idx] * self.params['delay_scale']

            w = utils.linear_transformation(p_, self.params['w_min'], self.params['w_max'])
            for i in xrange(len(p_)):
#                        w[i] = max(self.params['w_min'], min(w[i], self.params['w_max']))
                delay = min(max(l_[i], delay_min), delay_max)  # map the delay into the valid range
                connect(self.exc_pop[non_zero_idx[i]], self.exc_pop[tgt], w[i], delay=delay, synapse_type='excitatory')
                if self.debug_connectivity:
                    output += '%d\t%d\t%.2e\t%.2e\n' % (non_zero_idx[i], tgt, w[i], delay) #                    output += '%d\t%d\t%.2e\t%.2e\t%.2e\n' % (sources[i], tgt, w[i], latency[sources[i]], p[sources[i]])

        if self.debug_connectivity:
            if self.pc_id == 0:
                print 'DEBUG writing to file:', conn_list_fn
            conn_file.write(output)
            conn_file.close()

    def connect_isotropic(self, conn_type='ee'):
        """
        conn_type must be 'ee', 'ei', 'ie' or 'ii'
        Connect cells in a distant dependent manner:
            p_ij = exp(- d_ij / (2 * w_sigma_x**2))

        This will give a 'convergence constrained' connectivity, i.e. each cell will have the same sum of incoming weights
        ---> could be problematic for outlier cells
        """
        if self.pc_id == 0:
            print 'Connect isotropic %s - %s' % (conn_type[0].capitalize(), conn_type[1].capitalize())

        (n_src, n_tgt, src_pop, tgt_pop, tp_src, tp_tgt, tgt_cells, syn_type) = self.resolve_src_tgt(conn_type)
        if conn_type == 'ee':
            w_ = self.params['w_max']
            w_tgt_in = params['w_tgt_in_per_cell_%s' % conn_type]
            n_max_conn = n_src * n_tgt - n_tgt

        elif conn_type == 'ei':
            w_ = self.params['w_ei_mean']
            w_tgt_in = params['w_tgt_in_per_cell_%s' % conn_type]
            n_max_conn = n_src * n_tgt

        elif conn_type == 'ie':
            w_ = self.params['w_ie_mean']
            w_tgt_in = params['w_tgt_in_per_cell_%s' % conn_type]
            n_max_conn = n_src * n_tgt

        elif conn_type == 'ii':
            w_ = self.params['w_ii_mean']
            w_tgt_in = params['w_tgt_in_per_cell_%s' % conn_type]
            n_max_conn = n_src * n_tgt - n_tgt

        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_%s_fn_base' % conn_type] + '%d.dat' % (self.pc_id)

        w_mean = w_tgt_in / (self.params['p_%s' % conn_type] * n_max_conn / n_tgt)
        w_sigma = self.params['w_sigma_distribution'] * w_mean

        w_dist = RandomDistribution('normal',
                (w_mean, w_sigma),
                rng=self.rng_conn,
                constrain='redraw',
                boundaries=(0, w_mean * 10.))
        delay_dist = RandomDistribution('normal',
                (self.params['standard_delay'], self.params['standard_delay_sigma']),
                rng=self.rng_conn,
                constrain='redraw',
                boundaries=(self.params['delay_range'][0], self.params['delay_range'][1]))

        p_max = utils.get_pmax(self.params['p_%s' % conn_type], self.params['w_sigma_isotropic'], conn_type)
        connector = DistanceDependentProbabilityConnector('%f * exp(-d**2/(2*%f**2))' % (p_max, params['w_sigma_isotropic']), allow_self_connections=False, \
                weights=w_dist, delays=delay_dist, space=self.torus)#, n_connections=n_conn_ee)
        print 'p_max for %s' % conn_type, p_max
        prj = Projection(src_pop, tgt_pop, connector, target=syn_type)
        self.projections[conn_type].append(prj)
        if self.debug_connectivity:
            prj.saveConnections(self.params['conn_list_%s_fn_base' % conn_type] + '.dat', gather=True)


    def connect_random(self, conn_type):
        """
        There exist different possibilities to draw random connections:
        1) Calculate the weights as for the anisotropic case and sample sources randomly
        2) Load a file which stores some random connectivity --> # connector = FromFileConnector(self.params['conn_list_.... ']
        3) Create a random distribution with similar parameters as the non-random connectivition distribution

        connector_ee = FastFixedProbabilityConnector(self.params['p_ee'], weights=w_ee_dist, delays=self.delay_dist)
        prj_ee = Projection(self.exc_pop, self.exc_pop, connector_ee, target='excitatory')

        conn_list_fn = self.params['random_weight_list_fn'] + str(sim_cnt) + '.dat'
        print "Connecting exc - exc from file", conn_list_fn
        connector_ee = FromFileConnector(conn_list_fn)
        prj_ee = Projection(self.exc_pop, self.exc_pop, connector_ee, target='excitatory')
        """
        if self.pc_id == 0:
            print 'Connect random connections %s - %s' % (conn_type[0].capitalize(), conn_type[1].capitalize())
        (n_src, n_tgt, src_pop, tgt_pop, tp_src, tp_tgt, tgt_cells, syn_type) = self.resolve_src_tgt(conn_type)
        w_mean = self.params['w_tgt_in_per_cell_%s' % conn_type] / (n_src * self.params['p_%s' % conn_type])
        w_sigma = self.params['w_sigma_distribution'] * w_sigma

        weight_distr = RandomDistribution('normal',
                (w_mean, w_sigma),
                rng=self.rng_conn,
                constrain='redraw',
                boundaries=(0, w_mean * 10.))

        delay_dist = RandomDistribution('normal',
                (self.params['standard_delay'], self.params['standard_delay_sigma']),
                rng=self.rng_conn,
                constrain='redraw',
                boundaries=(self.params['delay_range'][0], self.params['delay_range'][1]))

        connector= FastFixedProbabilityConnector(self.params['p_%s' % conn_type], weights=weight_distr, delays=delay_dist)
        prj = Projection(src_pop, tgt_pop, connector, target=syn_type)

        conn_list_fn = self.params['conn_list_%s_fn_base' % conn_type] + '%d.dat' % (self.pc_id)
        print 'Saving random %s connections to %s' % (conn_type, conn_list_fn)
        prj.saveConnections(conn_list_fn, gather=False)



    def connect_populations(self, conn_type):
        """
            # # # # # # # # # # # #
            #     C O N N E C T   #
            # # # # # # # # # # # #
            Calls the right function according to the connectivity pattern set for the src-tgt populations in simultation_parameters.py
        """
        if self.params['connectivity_%s' % conn_type] == 'anisotropic':
            self.connect_anisotropic(conn_type)
        elif self.params['connectivity_%s' % conn_type] == 'isotropic':
            self.connect_isotropic(conn_type)
        elif self.params['connectivity_%s' % conn_type] == 'random':
            self.connect_random(conn_type)
        else: # populations do not get connected
            pass


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
        self.times['connect_noise'] = self.timer.diff()



    def place_electrode(self):
        # TODO: create a list or dictionary of population views, depending on the number of locations to record from
        # from each location n_cells_to_record_per_location will be recorded
        self.selected_pop_views = []
        for i_ in xrange(len(self.params['locations_to_record'])):
            mp_to_record = [self.params['motion_params'][0] + self.params['locations_to_record'][i_], \
                    self.params['motion_params'][1], self.params['motion_params'][2], self.params['motion_params'][3]]
            gids_ = utils.select_well_tuned_cells_1D(self.tuning_prop_exc, mp_to_record, self.params['n_cells_to_record_per_location'])
            self.selected_pop_views.append(PopulationView(self.exc_pop, gids_, label='anticipation_%d' % i_))
            self.selected_pop_views[i_].record_v()
#            self.selected_pop_views[i_].record_gsyn()

            pop_info_fn = self.params['parameters_folder'] + 'pop_%d.info' % i_
            f = file(pop_info_fn, 'w')
            d = {'gids' : gids_.tolist(), 'mp' : mp_to_record}
            json.dump(d, f, sort_keys=True, indent=4)

        self.anticipatory_record = True


    def run_sim(self, sim_cnt, record_v=True):

        # # # # # # # # # # # #
        #     R E C O R D     #
        # # # # # # # # # # # #
        record_exc = True
#        if os.path.exists(self.params['gids_to_record_fn']):
#            gids_to_record = np.loadtxt(self.params['gids_to_record_fn'], dtype='int')[:self.params['n_gids_to_record']]
#            record_exc = True
#            n_rnd_cells_to_record = 2
#        else:

        if record_v:
            self.exc_pop_view = PopulationView(self.exc_pop, gids_to_record, label='good_exc_neurons')
            self.exc_pop_view.record_v()
            self.inh_pop_view = PopulationView(self.inh_pop, np.random.randint(0, self.params['n_inh'], self.params['n_gids_to_record']), label='random_inh_neurons')
            if self.params['anticipatory_mode']:
                self.place_electrode()
            else:
                n_cells_to_record = self.params['n_gids_to_record']
                gids_to_record = np.random.randint(0, self.params['n_exc'], n_cells_to_record)
                self.inh_pop_view.record_v()

        self.inh_pop.record()
        self.exc_pop.record()
        self.times['t_record'] = self.timer.diff()

        # # # # # # # # # # # # # #
        #     R U N N N I N G     #
        # # # # # # # # # # # # # #
        if self.pc_id == 0:
            print "Running simulation ... "
        run(self.params['t_sim'])
        self.times['t_sim'] = self.timer.diff()

    def print_results(self, print_v=True):
        """
            # # # # # # # # # # # # # # # # #
            #   P R I N T    R E S U L T S  #
            # # # # # # # # # # # # # # # # #
        """
        if print_v:
            if self.pc_id == 0:
                print 'print_v to file: %s.v' % (self.params['exc_volt_fn_base'])
            self.exc_pop_view.print_v("%s.v" % (self.params['exc_volt_fn_base']), compatible_output=False)
            if self.pc_id == 0:
                print "Printing inhibitory membrane potentials"
            self.inh_pop_view.print_v("%s.v" % (self.params['inh_volt_fn_base']), compatible_output=False)

        if self.anticipatory_record == True:
            for i_ in xrange(len(self.params['locations_to_record'])):
                output_fn = self.params['exc_volt_fn_base'] + '_pop%d.v' % (i_)
                print 'Printing to:', output_fn
                self.selected_pop_views[i_].print_v(output_fn, compatible_output=False)
#                output_fn = self.params['exc_gsyn_fn_base'] + '_pop%d.gsyn' % (i_)
#                print 'Printing to:', output_fn
#                self.selected_pop_views[i_].print_gsyn(output_fn, compatible_output=False)

        if self.pc_id == 0:
            print "Printing excitatory spikes"
        self.exc_pop.printSpikes(self.params['exc_spiketimes_fn_merged'] + '.ras')
        if self.pc_id == 0:
            print "Printing inhibitory spikes"
        self.inh_pop.printSpikes(self.params['inh_spiketimes_fn_merged'] + '.ras')

        self.times['t_print'] = self.timer.diff()
        if self.pc_id == 0:
            print "calling pyNN.end() ...."
        end()
        self.times['t_end'] = self.timer.diff()

        if self.pc_id == 0:
            self.times['t_all'] = 0.
            for k in self.times.keys():
                self.times['t_all'] += self.times[k]

            self.n_cells = {}
            self.n_cells['n_exc'] = self.params['n_exc']
            self.n_cells['n_inh'] = self.params['n_inh']
            self.n_cells['n_cells'] = self.params['n_cells']
            self.n_cells['n_proc'] = self.n_proc
            output = {'times' : self.times, 'n_cells_proc' : self.n_cells}
            print "Proc %d Simulation time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (self.pc_id, self.times['t_sim'], (self.times['t_sim'])/60., self.params['n_cells'], self.params['n_exc'], self.params['n_inh'])
            print "Proc %d Full pyNN run time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (self.pc_id, self.times['t_all'], (self.times['t_all'])/60., self.params['n_cells'], self.params['n_exc'], self.params['n_inh'])
            fn = utils.convert_to_url(params['folder_name'] + 'times_dict_np%d.py' % self.n_proc)

            output_file = file(self.params['params_fn_json'], 'w')
            d = json.dump(self.params, output_file, sort_keys=True, indent=4)


def clean_up_results_directory(params):
    filenames = [params['exc_nspikes_fn_merged'], \
                params['exc_spiketimes_fn_merged'], \
                params['inh_spiketimes_fn_merged'], \
                params['inh_nspikes_fn_merged'], \
                params['exc_spiketimes_fn_base'], \
                params['inh_spiketimes_fn_base'], \
                params['merged_conn_list_ee'], \
                params['merged_conn_list_ei'], \
                params['merged_conn_list_ie'], \
                params['merged_conn_list_ii']]

    for fn in filenames:
        cmd = 'rm %s*' % (fn)
        print 'Removing %s' % (cmd)
        os.system(cmd)




if __name__ == '__main__':

    input_created = False

    """
    If you want to do a parameter sweep:
        get the sys.argv:
        e.g. 
             w_sigma_x = float(sys.argv[1])
             w_sigma_v = float(sys.argv[2])
        and pass it to params
             params['w_sigma_x'] = w_sigma_x
             params['w_sigma_v'] = w_sigma_v
        IMPORTANT:
        results only get stored in different folders, when the sweep parameter (w_sigma_x/v) is contained in 
        the main folder name params['folder_name']

        set the filenames with the new parameter:
        ps.set_filenames()
    Alternatively, you can create the main folder name here:
        ps.params['folder_name'] = None # sys.argv[1] + '=' + sys.argv[2] + '/' #'Sweep_%.3e' % self.params['w_tgt_in_per_cell_ee']
        ps.set_filenames(folder_name=ps.params['folder_name'])
    """

#    fn = str(sys.argv[1])

    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
#        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        ps = simulation_parameters.parameter_storage(param_fn)
        params = ps.params
    else:
        ps = simulation_parameters.parameter_storage()#fn)
        params = ps.params

    if pc_id == 0:
        clean_up_results_directory(params)
    if comm != None:
        comm.Barrier()
    ps.set_filenames()

    if pc_id == 0:
        ps.create_folders()
        ps.write_parameters_to_file()
    if comm != None:
        comm.Barrier()
    sim_cnt = 0

    max_neurons_to_record = 10000
    if params['n_cells'] > max_neurons_to_record:
        load_files = False
        record_v = False
        save_input_files = False
    else: # choose yourself
        load_files = True
        record_v = False
        save_input_files = not load_files

    save_input_files = True

    NM = NetworkModel(ps.params, comm)
    exec("from pyNN.%s import *" % NM.params['simulator'])
    NM.setup(times=times)
    NM.create(input_created)
    if not input_created:
        spike_times_container = NM.create_input(load_files=load_files, save_output=save_input_files)
        input_created = True # this can be set True ONLY if the parameter does not affect the input i.e. set this to false when sweeping f_max_stim, or blur_X/V!
    else:
        NM.spike_times_container = spike_times_container

    NM.connect()

    NM.run_sim(sim_cnt, record_v=record_v)
    NM.print_results(print_v=record_v)

    """
    The following scripts should only be called, if you are not running on a cluster...
    if pc_id == 0 and params['n_cells'] < max_neurons_to_record:
        import plot_prediction as pp
        pp.plot_prediction(params)

        os.system('python plot_rasterplots.py %s' % ps.params['folder_name'])
        os.system('python plot_connectivity_profile.py %s' % ps.params['folder_name'])

    if pc_id == 1 or not(USE_MPI):
        os.system('python plot_connectivity_profile.py %s' % ps.params['folder_name'])
        for conn_type in ['ee', 'ei', 'ie', 'ii']:
            os.system('python plot_weight_and_delay_histogram.py %s %s' % (conn_type, ps.params['folder_name']))

    if pc_id == 1 or not(USE_MPI):
        os.system('python analyse_connectivity.py %s' % ps.params['folder_name'])
    """
    if comm != None:
        comm.Barrier()

    if pc_id == 0 and params['n_cells'] < max_neurons_to_record:
        import plot_prediction as pp

        if params['n_grid_dimensions'] == 2:
            pp.plot_prediction_2D(params)
        else:
            pp.plot_prediction_1D(params)

        os.system('python plot_rasterplots.py %s' % ps.params['folder_name'])
        os.system('python plot_weight_and_delay_histogram.py %s' % ps.params['folder_name'])
        os.system('python plot_connectivity_profile.py %s' % ps.params['folder_name'])
        os.system('python PlottingScripts/PlotAnticipation.py %s' % ps.params['folder_name'])
        os.system('python PlottingScripts/plot_contour_connectivity.py %s' % ps.params['folder_name'])
        os.system('ristretto %s' % (ps.params['figures_folder']))
