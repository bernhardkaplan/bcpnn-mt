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
#import NeuroTools.parameters as ntp
import os
import utils
import simulation_parameters
ps = simulation_parameters.parameter_storage()
params = ps.params
import nest
import CreateStimuli
import json
times['time_to_import'] = time.time() - t0


class NetworkModel(object):

    def __init__(self, params, iteration=0):

        self.params = params
        self.debug_connectivity = True
        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.iteration = 0  # the learning iteration (cycle)


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
        if self.pc_id == 0:
            print "Saving tuning_prop to file:", self.params['tuning_prop_means_fn']
            np.savetxt(self.params['tuning_prop_means_fn'], self.tuning_prop_exc)
            print "Saving tuning_prop to file:", self.params['tuning_prop_inh_fn']
            np.savetxt(self.params['tuning_prop_inh_fn'], self.tuning_prop_inh)

#        from pyNN.utility import Timer
#        self.timer = Timer()
#        self.timer.start()
#        self.times = times
#        self.times['t_all'] = 0
        # # # # # # # # # # # #
        #     S E T U P       #
        # # # # # # # # # # # #
        (delay_min, delay_max) = self.params['delay_range']
        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder'], 'resolution': .1})
#        nest.SetKernelStatus({'tics_per_ms':self.params['dt_sim'], 'min_delay':delay_min, 'max_delay':delay_max})

        # # # # # # # # # # # # # # # # # # # # # # # # #
        #     R A N D O M    D I S T R I B U T I O N S  #
        # # # # # # # # # # # # # # # # # # # # # # # # #
        self.pyrngs = [np.random.RandomState(s) for s in range(self.params['seed'], self.params['seed'] + self.n_proc)]
        nest.SetKernelStatus({'grng_seed' : self.params['seed'] + self.n_proc})
        nest.SetKernelStatus({'rng_seeds' : range(self.params['seed'] + self.n_proc + 1, \
                self.params['seed'] + 2 * self.n_proc + 1)})

        self.setup_synapse_types()

        # clean spiketimes_folder:
        cmd = 'rm %s* ' % self.params['spiketimes_folder']
        os.system(cmd)


    def setup_synapse_types(self):

        nest.CopyModel('static_synapse', 'input_exc_0', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
        nest.CopyModel('static_synapse', 'input_exc_1', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 1})

        if (not 'bcpnn_synapse' in nest.Models('synapses')):
            nest.Install('pt_module')


    def get_local_indices(self, pop):
        local_nodes = []
        node_info   = nest.GetStatus(pop)
        for i_, d in enumerate(node_info):
            if d['local']:
                local_nodes.append((d['global_id'], d['vp']))
        return local_nodes
        

    def initialize_vmem(self, nodes):
        for gid, vp in nodes:
            nest.SetStatus([gid], {'V_m': self.pyrngs[vp].normal(self.params['v_init'], self.params['v_init_sigma'])})


    def create(self, input_created=False):
        """
            # # # # # # # # # # # #
            #     C R E A T E     #
            # # # # # # # # # # # #
        # TODO:

        the cell_params dict entries tau_syn should be set according to the cells' tuning properties
        --> instead of populations, create single cells?
        """

        self.list_of_populations = [ [] for i in xrange(self.params['n_hc'])]
        self.local_idx_exc = []
        self.local_idx_inh = []
        cell_idx = 0
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                cell_params = self.params['cell_params_exc'].copy()
                # TODO:
                # get the tuning properties for cells in that MC and 
                # calculate the time_constants based on the preferred velocity
                pop = nest.Create(self.params['neuron_model'], self.params['n_exc_per_mc'], params=cell_params)
                # TODO:
                # write tau_syn parameters to file
                self.local_idx_exc += self.get_local_indices(pop)
                self.list_of_populations[hc].append(pop)

#        print 'DEBUG ', len(self.list_of_populations)
        # set the cell parameters
        self.inh_pop = nest.Create(self.params['neuron_model'], self.params['n_inh'], params=self.params['cell_params_inh'])
        self.local_idx_inh = self.get_local_indices(self.inh_pop)

        # v_init
#        self.initialize_vmem(self.local_idx_exc)
#        self.initialize_vmem(self.local_idx_inh)

        if not input_created:
            self.spike_times_container = [ [] for i in xrange(len(self.local_idx_exc))]

#        self.times['t_create'] = self.timer.diff()




    def create_training_input(self, load_files=False, save_output=False, with_blank=False):


        if load_files:
            if self.pc_id == 0:
                print "Loading input spiketrains..."
            for i_, tgt in enumerate(self.local_idx_exc):
                try:
                    fn = self.params['input_st_fn_base'] + str(tgt[0] - 1) + '.npy'
                    spike_times = np.load(fn)
                except: # this cell does not get any input
                    print "Missing file: ", fn
                    spike_times = []
                self.spike_times_container[i_] = spike_times
        else:
            if self.pc_id == 0:
                print "Computing input spiketrains..."



            my_units = np.array(self.local_idx_exc)[:, 0] - 1
#            print 'debug  my_units', my_units
            n_cells = len(my_units)
            dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process
            time = np.arange(0, self.params['t_sim'], dt)
            L_input = np.zeros((n_cells, time.shape[0]))  # the envelope of the Poisson process

            # get the order of training stimuli
            CS = CreateStimuli.CreateStimuli()
            random_order = self.params['random_training_order']
            motion_params = CS.create_motion_sequence_1D(self.params, random_order)
            np.savetxt(self.params['training_sequence_fn'], motion_params)
            n_stim_total = self.params['n_training_stim']
            for i_stim in xrange(n_stim_total):
                print 'Calculating input signal for training stim %d / %d (%.1f percent)' % (i_stim, n_stim_total, float(i_stim) / n_stim_total * 100.)
                x0, v0 = motion_params[i_stim, 0], motion_params[i_stim, 1]


                # get the input signal
                idx_t_start = np.int(i_stim * self.params['t_training_stim'] / dt)
                idx_t_stop = np.int((i_stim + 1) * self.params['t_training_stim'] / dt)
                idx_within_stim = 0
                for i_time in xrange(idx_t_start, idx_t_stop):
                    time_ = (idx_within_stim * dt) / self.params['t_stimulus']
                    x_stim = x0 + time_ * v0
                    L_input[:, i_time] = utils.get_input(self.tuning_prop_exc[my_units, :], self.params, (x_stim, 0, v0, 0, 0))
                    L_input[:, i_time] *= self.params['f_max_stim']
                    if (i_time % 1000 == 0):
                        print "t: %.2f [ms]" % (time_ * self.params['t_stimulus'])
                    idx_within_stim += 1

            
                if with_blank:
                    blank_idx = np.arange(1./dt * self.params['t_before_blank'] + idx_t_start, 1. / dt * (self.params['t_before_blank'] + self.params['t_blank']) + idx_t_start)
                    before_stim_idx = np.arange(0, self.params['t_start'] * 1./dt + idx_t_start)
                    blank_idx = np.concatenate((blank_idx, before_stim_idx))
                    # blanking
                    for i_time in blank_idx:
                        L_input[:, i_time] = np.random.permutation(L_input[:, i_time])


            nprnd.seed(self.params['input_spikes_seed'])
            # create the spike trains
            print 'Creating input spiketrains for unit'
            for i_, unit in enumerate(my_units):
                print 'Creating input spiketrain for unit %d / %d (%.1f percent)' % (i_, len(my_units), float(i_) / len(my_units)* 100.)
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

#        self.times['create_input'] = self.timer.diff()
        return self.spike_times_container
    





    def connect_input_to_exc(self):
        """
            # # # # # # # # # # # # # # # # # # # # # #
            #     C O N N E C T    I N P U T - E X C  #
            # # # # # # # # # # # # # # # # # # # # # #
        """
        if self.pc_id == 0:
            print "Connecting input spiketrains..."
        

        print 'Proc %d will connect %d spike trains' % (self.pc_id, len(self.local_idx_exc))

        self.stimulus = nest.Create('spike_generator', len(self.local_idx_exc))

        n_per_hc = self.params['n_mc_per_hc'] * self.params['n_exc_per_mc']
        for i_, (unit, vp) in enumerate(self.local_idx_exc):
            spike_times = self.spike_times_container[i_]
            nest.SetStatus([self.stimulus[i_]], {'spike_times' : spike_times})
            # get the cell from the list of populations
            mc_idx = (unit - 1) / self.params['n_exc_per_mc']
            hc_idx = (unit - 1) / n_per_hc
            mc_idx_in_hc = mc_idx - hc_idx * self.params['n_mc_per_hc']
            idx_in_pop = (unit - 1) - mc_idx * self.params['n_exc_per_mc']

#            print 'debug unit %d\ti_=%d\thc_idx = %d\tmc_idx_in_hc = %d\tidx_in_pop = %d' % (unit, i_, hc_idx, mc_idx_in_hc, idx_in_pop)
            nest.Connect([self.stimulus[i_]], [self.list_of_populations[hc_idx][mc_idx_in_hc][idx_in_pop]], model='input_exc_0')





#        self.stimulus = Population(len(self.local_idx_exc), SpikeSourceArray)
#            self.exc_pop = Population(n_exc, IF_cond_exp, self.params['cell_params_exc'], label='exc_cells')
#                prj = Projection(src_pop, tgt_pop, connector, target=syn_type)
#            self.projections[conn_type].append(prj)

#        self.projections['stim'] = []
#        self.stimuli = []
#        self.pop_views = [] 
#        conn = OneToOneConnector(weights=self.params['w_input_exc'])
#        for i_, unit in enumerate(self.local_idx_exc):
#            spike_times = self.spike_times_container[i_]

#            ssa = create(SpikeSourceArray, {'spike_times': spike_times})
#            ssa = Population(1, SpikeSourceArray, {'spike_times': spike_times})
#            ssa.set({'spike_times' : spike_times})
#            self.stimuli.append(ssa)

#            if self.params['with_short_term_depression']:

#                connect(ssa, self.exc_pop[unit], self.params['w_input_exc'], synapse_type='excitatory', synapse_dynamics=self.short_term_depression)
#                selector = np.zeros(self.params['n_exc'], dtype=np.bool)
#                selector[unit] = True
#                print 'debug unit', unit, type(unit)
#                w[i_] = 1.#self.params['w_input_exc']
#                tgt = PopulationView(self.exc_pop, np.array([unit]))
#                self.pop_views.append(tgt)
#                prj = Projection(ssa, tgt, conn, target='excitatory', synapse_dynamics=self.short_term_depression)
#                prj = Projection(self.stimuli[-1], self.pop_views[-1], conn, target='excitatory', synapse_dynamics=self.short_term_depression)
#                self.projections['stim'].append(prj)
#            else:
#            connect(ssa, self.exc_pop[unit], self.params['w_input_exc'], synapse_type='excitatory')
#        self.times['connect_input'] = self.timer.diff()


    def connect(self):
        if self.params['n_exc'] > 5000:
            save_output = False
        else:
            save_output = True

        self.connect_input_to_exc()

        self.connect_ee(load_weights=False)
#        self.connect_populations('ee')
#        self.connect_populations('ei')
#        self.connect_populations('ie')
#        self.connect_populations('ii')
#        self.connect_noise()
#        self.times['t_calc_conns'] = self.timer.diff()



    def connect_ee(self, load_weights=False):

#        initial_weight = np.log(nest.GetDefaults('bcpnn_synapse')['p_ij']/(nest.GetDefaults('bcpnn_synapse')['p_i']*nest.GetDefaults('bcpnn_synapse')['p_j']))
        initial_weight = 0.
        initial_bias = np.log(nest.GetDefaults('bcpnn_synapse')['p_j'])
        syn_param = {'weight': initial_weight, 'bias': initial_bias,'gain': 0.0, 'delay': 1.0,\
                'tau_i': 10.0, 'tau_j': 10.0, 'tau_e': 100.0, 'tau_p': 1000.0}

        if load_weights:
            conn_mat_ee = np.load(self.params['conn_mat_fn_base'] + 'ee_' + str(self.iteration) + '.npy')
        else:
            conn_mat_ee = initial_weight * np.ones((self.params['n_mc'], self.params['n_mc']))
        for src_hc in xrange(self.params['n_hc']):
            for src_mc in xrange(self.params['n_mc_per_hc']):
                src_pop = self.list_of_populations[src_hc][src_mc]
                src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                for tgt_hc in xrange(self.params['n_hc']):
                    for tgt_mc in xrange(self.params['n_mc_per_hc']):
                        tgt_pop = self.list_of_populations[tgt_hc][tgt_mc]
                        nest.ConvergentConnect(src_pop, tgt_pop, model='bcpnn_synapse')
                        tgt_pop_idx = tgt_hc * self.params['n_mc_per_hc'] + tgt_mc

                        # modify the parameters
                        nest.SetStatus(nest.GetConnections(src_pop, tgt_pop), {'weight': conn_mat_ee[src_pop_idx, tgt_pop_idx]})

#                        print 'modified params:', params


    def get_weights_after_learning_cycle(self):

        print 'NetworkModel.get_weights_after_learning_cycle ...'
        n_my_conns = 0
        my_units = np.array(self.local_idx_exc)[:, 0] # !GIDs are 1-aligned!
        my_adj_list = {}
        for nrn in my_units:
            my_adj_list[nrn] = []
        for src_hc in xrange(self.params['n_hc']):
            print 'Proc %d src_hc' % (self.pc_id, src_hc)
            for src_mc in xrange(self.params['n_mc_per_hc']):
                src_pop = self.list_of_populations[src_hc][src_mc]
                src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                for tgt_hc in xrange(self.params['n_hc']):
                    for tgt_mc in xrange(self.params['n_mc_per_hc']):
                        tgt_pop = self.list_of_populations[tgt_hc][tgt_mc]
                        tgt_pop_idx = tgt_hc * self.params['n_mc_per_hc'] + tgt_mc

                        # get the list of connections stored on the current MPI node
                        conns = nest.GetConnections(src_pop, tgt_pop)
                        for c in conns:
                            cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                            w = cp[0]['weight'] 
                            if w != 0:
                                my_adj_list[c[1]].append((c[0], cp[0]['weight']))
                        n_my_conns += len(conns)



#        print 'Proc %d holds connections' % self.pc_id, my_adj_list
        print 'Proc %d holds %d connections' % (self.pc_id, n_my_conns)
        output_fn = self.params['conn_mat_fn_base'] + 'AS_%d_%d.json' % (self.iteration, self.pc_id)
        print 'Saving connection list to: ', output_fn
        f = file(output_fn, 'w')
        json.dump(my_adj_list, f)



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
                    p[src], latency[src] = CC.get_p_conn(self.tuning_prop_exc[src, :], self.tuning_prop_exc[tgt, :], sigma_x, sigma_v, params['connectivity_radius']) #                            print 'debug pc_id src tgt ', self.pc_id, src, tgt#, int(ID) < self.params['n_exc']
            sources = random.sample(xrange(self.params['n_exc']), int(self.params['n_src_cells_per_neuron']))
            idx = p[sources] > 0
            non_zero_idx = np.nonzero(idx)[0]
            p_ = p[sources][non_zero_idx]
            l_ = latency[sources][non_zero_idx] * self.params['delay_scale']

            w = utils.linear_transformation(p_, self.params['w_thresh_min'], self.params['w_thresh_max'])
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
            w_tgt_in = params['w_tgt_in_per_cell_%s' % conn_type]
            n_max_conn = n_src * n_tgt - n_tgt

        elif conn_type == 'ei':
            w_tgt_in = params['w_tgt_in_per_cell_%s' % conn_type]
            n_max_conn = n_src * n_tgt

        elif conn_type == 'ie':
            w_tgt_in = params['w_tgt_in_per_cell_%s' % conn_type]
            n_max_conn = n_src * n_tgt

        elif conn_type == 'ii':
            w_tgt_in = params['w_tgt_in_per_cell_%s' % conn_type]
            n_max_conn = n_src * n_tgt - n_tgt

        if self.debug_connectivity:
            conn_list_fn = self.params['conn_list_%s_fn_base' % conn_type] + '%d.dat' % (self.pc_id)
#            conn_file = open(conn_list_fn, 'w')
#            output = ''
#            output_dist = ''

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
        connector = DistanceDependentProbabilityConnector('%f * exp(-d/(2*%f**2))' % (p_max, params['w_sigma_isotropic']), allow_self_connections=False, \
                weights=w_dist, delays=delay_dist, space=self.torus)#, n_connections=n_conn_ee)
        if self.params['with_short_term_depression']:
            prj = Projection(src_pop, tgt_pop, connector, target=syn_type, synapse_dynamics=self.short_term_depression)
        else:
            prj = Projection(src_pop, tgt_pop, connector, target=syn_type)#, synapse_dynamics=self.STD)
        self.projections[conn_type].append(prj)
        if self.debug_connectivity:
#                if self.pc_id == 0:
#                    print 'DEBUG writing to file:', conn_list_fn
            prj.saveConnections(self.params['conn_list_%s_fn_base' % conn_type] + '.dat', gather=True)
#            prj.saveConnections(self.params['conn_list_%s_fn_base' % conn_type] + 'gid%d.dat' % tgt, gather=False)
#                conn_file.close()


#            w = np.zeros(n_src, dtype='float32')
#            delays = np.zeros(n_src, dtype='float32')
#            for src in xrange(n_src):
#                if conn_type[0] == conn_type[1]:
#                    if (src != tgt): # no self-connections / autapses
#                        d_ij = utils.torus_distance2D(tp_src[src, 0], tp_tgt[tgt, 0], tp_src[src, 1], tp_tgt[tgt, 1])
#                        p_ij = p_max * np.exp(-d_ij**2 / (2 * params['w_sigma_isotropic']**2))
#                        if np.random.rand() <= p_ij:
#                            w[src] = w_
#                            delays[src] = d_ij * params['delay_scale']
#                else:
#                    d_ij = utils.torus_distance2D(tp_src[src, 0], tp_tgt[tgt, 0], tp_src[src, 1], tp_tgt[tgt, 1])
#                    p_ij = p_max * np.exp(-d_ij**2 / (2 * params['w_sigma_isotropic']**2))
#                    if np.random.rand() <= p_ij:
#                        w[src] = w_
#                        delays[src] = d_ij * params['delay_scale']
#            w *= w_tgt_in / w.sum()
#            srcs = w.nonzero()[0]
#            weights = w[srcs]
#            for src in srcs:
#                if w[src] > self.params['w_thresh_connection']:
#                delay = min(max(delays[src], self.params['delay_range'][0]), self.params['delay_range'][1])  # map the delay into the valid range
#                connect(src_pop[int(src)], tgt_pop[int(tgt)], w[src], delay=delay, synapse_type=syn_type)
#                output += '%d\t%d\t%.2e\t%.2e\n' % (src, tgt, w[src], delay)

#        if self.debug_connectivity:
#            if self.pc_id == 0:
#                print 'DEBUG writing to file:', conn_list_fn
#            conn_file.write(output)
#            conn_file.close()


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
#            conn_file = open(conn_list_fn, 'w')
#            output = ''
#            output_dist = ''

        w_mean = w_tgt_in / (self.params['p_%s' % conn_type] * n_max_conn / n_tgt)
        w_sigma = self.params['w_sigma_distribution'] * w_mean

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
        if self.params['with_short_term_depression']:
            prj = Projection(src_pop, tgt_pop, connector, target=syn_type, synapse_dynamics=self.short_term_depression)
        else:
            prj = Projection(src_pop, tgt_pop, connector, target=syn_type)

        conn_list_fn = self.params['conn_list_%s_fn_base' % conn_type] + '%d.dat' % (self.pc_id)
        print 'Saving random %s connections to %s' % (conn_type, conn_list_fn)
        prj.saveConnections(conn_list_fn, gather=False)




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
#        self.times['connect_noise'] = self.timer.diff()



    def get_indices_for_gid(self, gid):
        """Returns the HC, MC, and within MC index for the gid
        """

        n_per_hc = self.params['n_mc_per_hc'] * self.params['n_exc_per_mc']
        mc_idx = (gid - 1) / self.params['n_exc_per_mc']
        hc_idx = (gid - 1) / n_per_hc
        mc_idx_in_hc = mc_idx - hc_idx * self.params['n_mc_per_hc']
        idx_in_mc = (gid - 1) - mc_idx * self.params['n_exc_per_mc']

        return hc_idx, mc_idx_in_hc, idx_in_mc


    def run_sim(self, sim_cnt, record_v=False):
        # # # # # # # # # # # # # # # # # # # #
        #     P R I N T    W E I G H T S      #
        # # # # # # # # # # # # # # # # # # # #
    #    print 'Printing weights to :\n  %s\n  %s\n  %s' % (self.params['conn_list_ei_fn'], self.params['conn_list_ie_fn'], self.params['conn_list_ii_fn'])
    #    exc_inh_prj.saveConnections(self.params['conn_list_ei_fn'])
    #    inh_exc_prj.saveConnections(self.params['conn_list_ie_fn'])
    #    inh_inh_prj.saveConnections(self.params['conn_list_ii_fn'])
    #    self.times['t_save_conns'] = self.timer.diff()

              
        if record_v:
            mp_r = np.array(self.params['motion_params'])
            selected_gids_r, pops = utils.select_well_tuned_cells_trajectory(self.tuning_prop_exc, \
                    mp_r, params, self.params['n_gids_to_record'] / 2, 1)
            mp_l = mp_r.copy()
            mp_l[2] *= (-1.)
            selected_gids_l, pops = utils.select_well_tuned_cells_trajectory(self.tuning_prop_exc, \
                    mp_l, params, self.params['n_gids_to_record'] / 2, 1)
            print 'Recording cells close to mp_l', mp_l, '\nGIDS:', selected_gids_l
            print 'Recording cells close to mp_r', mp_r, '\nGIDS:', selected_gids_r
             
            gids_to_record = np.r_[selected_gids_r, selected_gids_l]
            np.savetxt(self.params['gids_to_record_fn'], gids_to_record, fmt='%d')
            voltmeter = nest.Create('voltmeter')
            nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'volt'}])
            
        exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                nest.ConvergentConnect(self.list_of_populations[hc][mc], exc_spike_recorder)

        if record_v: 
            for gid in gids_to_record:
#            for i_, (unit, vp) in enumerate(self.local_idx_exc):
                hc_idx, mc_idx_in_hc, idx_in_mc = self.get_indices_for_gid(gid)
                nest.ConvergentConnect(voltmeter, [self.list_of_populations[hc_idx][mc_idx_in_hc][idx_in_mc]])
    

        # # # # # # # # # # # # # #
        #     R U N N N I N G     #
        # # # # # # # # # # # # # #
        if self.pc_id == 0:
            print "Running simulation ... "
        nest.Simulate(self.params['t_sim'])
#        self.times['t_sim'] = self.timer.diff()


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

        print 'DEBUG printing anticipatory cells', self.anticipatory_record
        if self.anticipatory_record == True:   
            print 'print_v to file: %s' % (self.params['exc_volt_anticipation'])
            self.exc_pop_view_anticipation.print_v("%s" % (self.params['exc_volt_anticipation']), compatible_output=False)
            print 'print_gsyn to file: %s' % (self.params['exc_gsyn_anticipation'])
            self.exc_pop_view_anticipation.print_gsyn("%s" % (self.params['exc_gsyn_anticipation']), compatible_output=False)


        if self.pc_id == 0:
            print "Printing excitatory spikes"
        self.exc_pop.printSpikes(self.params['exc_spiketimes_fn_merged'] + '.ras')
        if self.pc_id == 0:
            print "Printing inhibitory spikes"
        self.inh_pop.printSpikes(self.params['inh_spiketimes_fn_merged'] + '.ras')


#        self.times['t_print'] = self.timer.diff()
#        self.times['t_end'] = self.timer.diff()

#        if self.pc_id == 0:
#            self.times['t_all'] = 0.
#            for k in self.times.keys():
#                self.times['t_all'] += self.times[k]

#            self.n_cells = {}
#            self.n_cells['n_exc'] = self.params['n_exc']
#            self.n_cells['n_inh'] = self.params['n_inh']
#            self.n_cells['n_cells'] = self.params['n_cells']
#            self.n_cells['n_proc'] = self.n_proc
#            output = {'times' : self.times, 'n_cells_proc' : self.n_cells}
#            print "Proc %d Simulation time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (self.pc_id, self.times['t_sim'], (self.times['t_sim'])/60., self.params['n_cells'], self.params['n_exc'], self.params['n_inh'])
#            print "Proc %d Full pyNN run time: %d sec or %.1f min for %d cells (%d exc %d inh)" % (self.pc_id, self.times['t_all'], (self.times['t_all'])/60., self.params['n_cells'], self.params['n_exc'], self.params['n_inh'])
#            fn = utils.convert_to_url(params['folder_name'] + 'times_dict_np%d.py' % self.n_proc)

#            output = ntp.ParameterSet(output)
#            output.save(fn)


if __name__ == '__main__':

    input_created = False

#    orientation = float(sys.argv[1])
#    protocol = str(sys.argv[2])
#    ps.params['motion_params'][4] = orientation
#    ps.params['motion_protocol'] = protocol

    # always call set_filenames to update the folder name and all depending filenames!
    ps.set_filenames()

    ps.create_folders()
    ps.write_parameters_to_file()

    sim_cnt = 0
    max_neurons_to_record = 15800
    if params['n_cells'] > max_neurons_to_record:
        load_files = False
        record = False
        save_input_files = False
    else: # choose yourself
        load_files = True
        record = True
        save_input_files = not load_files

    NM = NetworkModel(ps.params, iteration=0)

    NM.setup(times=times)

    NM.create(input_created)
    if not input_created:
        spike_times_container = NM.create_training_input(load_files=load_files, save_output=save_input_files, with_blank=False)
        input_created = True # this can be set True ONLY if the parameter does not affect the input i.e. set this to false when sweeping f_max_stim, or blur_X/V!
    else:
        NM.spike_times_container = spike_times_container
    NM.connect()
    NM.run_sim(sim_cnt, record_v=record)
    NM.get_weights_after_learning_cycle()
#    NM.print_results(print_v=record)

#    if comm != None:
#        comm.Barrier()

#    if pc_id == 0 and params['n_cells'] < max_neurons_to_record:
#        os.system('python plot_rasterplots.py %s' % ps.params['folder_name'])
#        import plot_prediction as pp
#        pp.plot_prediction(params)
#    if comm != None:
#        comm.Barrier()

