"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons

on the cluster:
    frioul_batch -M "[['w_tgt_in_per_cell_ee', 'w_tgt_in_per_cell_ee', 'w_tgt_in_per_cell_ee'],[0.4, 0.8, 1.2]]" 'python NetworkSimModuleNoColumns.py'


"""
import time
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


class NetworkModel(object):

    def __init__(self, params, iteration=0):

        self.params = params
        self.debug_connectivity = True
        self.iteration = 0  # the learning iteration (cycle)
        self.times = {}
        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()

    def setup(self, load_tuning_prop=False):

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

        # # # # # # # # # # # #
        #     S E T U P       #
        # # # # # # # # # # # #
        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder'], 'resolution': .1})
        (delay_min, delay_max) = self.params['delay_range']
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
        if self.pc_id == 0:
            cmd = 'rm %s* ' % self.params['spiketimes_folder']
            os.system(cmd)



    def setup_synapse_types(self):

        nest.CopyModel('static_synapse', 'input_exc_0', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
        nest.CopyModel('static_synapse', 'input_exc_1', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 1})

        nest.CopyModel('static_synapse', 'exc_exc_local', \
                {'weight': self.params['w_ee_local'], 'receptor_type': 0})

        nest.CopyModel('static_synapse', 'exc_inh_unspec', \
                {'weight': self.params['w_ei_unspec'], 'receptor_type': 0})

        if (not 'bcpnn_synapse' in nest.Models('synapses')):
            nest.Install('pt_module')


    def get_local_indices(self, pop):
        local_nodes = []
        node_info   = nest.GetStatus(pop)
        for i_, d in enumerate(node_info):
            if d['local']:
                local_nodes.append(d['global_id'])
#                local_nodes.append((d['global_id'], d['vp']))
        return local_nodes
        

    def initialize_vmem(self, nodes):
        for gid, vp in nodes:
            nest.SetStatus([gid], {'V_m': self.pyrngs[vp].normal(self.params['v_init'], self.params['v_init_sigma'])})


    def create(self, input_created=False):
        """
            # # # # # # # # # # # #
            #     C R E A T E     #
            # # # # # # # # # # # #
        TODO:

        the cell_params dict entries tau_syn should be set according to the cells' tuning properties
        --> instead of populations, create single cells?
        """

        self.list_of_exc_pop = [ [] for i in xrange(self.params['n_hc'])]
        self.list_of_unspecific_inh_pop = [ [] for i in xrange(self.params['n_hc'])]
        self.list_of_specific_inh_pop = [ [] for i in xrange(self.params['n_hc'])]
        self.local_idx_exc = []
        self.local_idx_inh_unspec = []
        self.local_idx_inh_spec = []
        ##### EXC CELLS
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                cell_params = self.params['cell_params_exc'].copy()
                pop = nest.Create(self.params['neuron_model'], self.params['n_exc_per_mc'], params=cell_params)
                self.local_idx_exc += self.get_local_indices(pop)
                self.list_of_exc_pop[hc].append(pop)

        ##### UNSPECIFIC INHIBITORY CELLS
        for hc in xrange(self.params['n_hc']):
            pop = nest.Create(self.params['neuron_model'], self.params['n_inh_unspec_per_hc'], params=cell_params)
            self.list_of_unspecific_inh_pop.append(pop)
            self.local_idx_inh_unspec += self.get_local_indices(pop)

        ##### SPECIFIC INHIBITORY CELLS
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                pop = nest.Create(self.params['neuron_model'], self.params['n_inh_per_mc'], params=cell_params)
                self.list_of_specific_inh_pop.append(pop)
                self.local_idx_inh_spec += self.get_local_indices(pop)

#        print 'DEBUG ', len(self.list_of_exc_pop)
        # set the cell parameters
        self.spike_times_container = [ [] for i in xrange(len(self.local_idx_exc))]

        # v_init
#        self.initialize_vmem(self.local_idx_exc)
#        self.initialize_vmem(self.local_idx_inh)


#        self.times['t_create'] = self.timer.diff()




    def create_training_input(self, load_files=False, save_output=False, with_blank=False):


        if load_files:
            self.load_input()
        else:
            if self.pc_id == 0:
                print "Computing input spiketrains..."

            my_units = np.array(self.local_idx_exc) - 1
#            my_units = np.array(self.local_idx_exc)[:, 0] - 1
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
                x0, v0 = motion_params[i_stim, 0], motion_params[i_stim, 2]

                # get the input signal
                idx_t_start = np.int(i_stim * self.params['t_training_stim'] / dt)
                idx_t_stop = np.int((i_stim + 1) * self.params['t_training_stim'] / dt) 
                idx_within_stim = 0
                for i_time in xrange(idx_t_start, idx_t_stop):
                    time_ = (idx_within_stim * dt) / self.params['t_stimulus']
                    x_stim = x0 + time_ * v0
                    L_input[:, i_time] = utils.get_input(self.tuning_prop_exc[my_units, :], self.params, (x_stim, 0, v0, 0, 0))
                    L_input[:, i_time] *= self.params['f_max_stim']
                    if (i_time % 5000 == 0):
                        print "t: %.2f [ms]" % (time_ * self.params['t_stimulus'])
                    idx_within_stim += 1

            
                if with_blank:
                    start_blank = idx_t_start + 1. / dt * self.params['t_before_blank']
                    stop_blank = idx_t_start + 1. / dt * (self.params['t_before_blank'] + self.params['t_blank'])
                    blank_idx = np.arange(start_blank, stop_blank)
                    before_stim_idx = np.arange(idx_t_start, self.params['t_start'] * 1./dt + idx_t_start)
                    print 'debug stim before_stim_idx', i_stim, before_stim_idx, before_stim_idx.size
                    print 'debug stim blank_idx', i_stim, blank_idx, blank_idx.size
                    blank_idx = np.concatenate((blank_idx, before_stim_idx))
                    # blanking
                    for i_time in blank_idx:
                        L_input[:, i_time] = np.random.permutation(L_input[:, i_time])


            nprnd.seed(self.params['input_spikes_seed'])
            # create the spike trains
            print 'Creating input spiketrains for unit'
            for i_, unit in enumerate(my_units):
                print 'Creating input spiketrain for unit %d (%d / %d) (%.1f percent)' % (unit, i_, len(my_units), float(i_) / len(my_units) * 100.)
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
                    output_fn = self.params['input_rate_fn_base'] + str(unit) + '.dat'
                    np.savetxt(output_fn, rate_of_t)
                    output_fn = self.params['input_st_fn_base'] + str(unit) + '.dat'
                    np.savetxt(output_fn, np.array(spike_times))

#        self.times['create_input'] = self.timer.diff()
        return self.spike_times_container
    

    def load_input(self):

        if self.pc_id == 0:
            print "Loading input spiketrains..."
        for i_, tgt in enumerate(self.local_idx_exc):
            try:
#                fn = self.params['input_st_fn_base'] + str(tgt[0] - 1) + '.npy'
                fn = self.params['input_st_fn_base'] + str(tgt - 1) + '.dat'
                spike_times = np.loadtxt(fn)
            except: # this cell does not get any input
                print "Missing file: ", fn
                spike_times = []

            if type(spike_times) == type(1.0): # it's just one spike
                self.spike_times_container[i_] = np.array([spike_times])
            else:
                self.spike_times_container[i_] = spike_times



    def connect(self):
        nest.SetDefaults('bcpnn_synapse', params=self.params['bcpnn_params'])

        self.connect_input_to_exc()

        if self.params['training_run']:
            self.connect_ee(load_weights=False)
            self.connect_ei_unspecific()
        else: # load the weight matrix
            self.connect_ee(load_weights=True)
            self.connect_ei_unspecific()
            self.connect_ei_specific(load_weights=True)
            self.connect_ie() # connect unspecific and specific inhibition to excitatory cells
            self.connect_ii() # connect unspecific and specific inhibition to excitatory cells

#        self.times['t_calc_conns'] = self.timer.diff()


    def connect_ei_unspecific(self):
        pass   

#        for 
#        for src_hc in xrange(self.params['n_hc']):
#            tgt_hc = src_hc
#            for src_mc in xrange(self.params['n_mc_per_hc']):
#                src_pop = self.list_of_exc_pop[src_hc][src_mc]
#                tgt_pop = self.list_of_unspecific_inh_pop[tgt_hc]
#                print 'src_pop %d %d' % (src_hc, src_mc), src_pop





    def connect_ee(self, load_weights=False):

        initial_weight = 0.

        if load_weights:
            fn = self.params['conn_mat_fn_base'] + 'ee_' + str(self.iteration) + '.dat'
            assert (os.path.exists(fn)), 'ERROR: Weight matrix not found %s\n\tCorrect training_run flag set?\n\t\tHave you run Network_PyNEST_weight_analysis_after_training.py?\n' % fn
            conn_mat_ee = np.loadtxt(self.params['conn_mat_fn_base'] + 'ee_' + str(self.iteration) + '.dat')
        else:
            conn_mat_ee = initial_weight * np.ones((self.params['n_mc'], self.params['n_mc']))

        if load_weights:
            for src_hc in xrange(self.params['n_hc']):
                for src_mc in xrange(self.params['n_mc_per_hc']):
                    src_pop = self.list_of_exc_pop[src_hc][src_mc]
                    src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                    # TODO:
                    # get the tuning properties for cells in that MC and 
                    # calculate the time_constants based on the preferred velocity
                    print 'Debug TODO Need to get the tuning prop of this source pop:', src_pop
                    tp_src = self.tp_src[src_pop, 2] # get the v_x tuning for these cells
                    for tgt_hc in xrange(self.params['n_hc']):
                        for tgt_mc in xrange(self.params['n_mc_per_hc']):
                            # TODO:
                            # write tau_syn parameters to file
                            tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
                            nest.ConvergentConnect(src_pop, tgt_pop, model='bcpnn_synapse')
                            tgt_pop_idx = tgt_hc * self.params['n_mc_per_hc'] + tgt_mc

                            # modify the parameters
                            nest.SetStatus(nest.GetConnections(src_pop, tgt_pop), {'weight': conn_mat_ee[src_pop_idx, tgt_pop_idx]})
        else:
            for tgt_gid in self.local_idx_exc:
                hc_idx, mc_idx, gid_min, gid_max = self.get_gids_to_mc(tgt_gid)
                n_rnd_src = int(round(self.params['n_exc_per_mc'] * self.params['p_ee_local']))
                source_gids = np.array([])
                while source_gids.size != n_rnd_src:
                    source_gids = np.random.randint(gid_min, gid_max, n_rnd_src)
                nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local')


#            for src_hc in xrange(self.params['n_hc']):
#                for src_mc in xrange(self.params['n_mc_per_hc']):
#                    src_pop = self.list_of_exc_pop[src_hc][src_mc]
#                     for each neuron in src_pop choose p_ee_local * n_exc_per_mc sources
#                    for i_, src_gid in enumerate(src_pop):





    

    def connect_ei(self):
        for src_hc in xrange(self.params['n_hc']):
            tgt_pop = self.list_of_unspecific_inh_pop[src_hc]
            for src_mc in xrange(self.params['n_mc_per_hc']):
                src_pop = self.list_of_exc_pop[src_hc][src_mc]
                nest.ConvergentConnect(src_pop, tgt_pop, model='')



    def get_gids_to_mc(self, pyr_gid):
        """
        Return the HC, MC within the HC in which the cell with pyr_gid is in
        and the min and max gid of pyr cells belonging to the same MC.
        """
        mc_idx = (pyr_gid - 1) / self.params['n_exc_per_mc']
        hc_idx = mc_idx / self.params['n_mc_per_hc']
        gid_min = mc_idx * self.params['n_exc_per_mc'] + 1 # + 1 because of NEST's 'smart' indexing
        gid_max = (mc_idx + 1) * self.params['n_exc_per_mc']  # here no +1 because it's used for randrange and +1 would include a false cell
        return (hc_idx, mc_idx, gid_min, gid_max)




    
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
#        for i_, (unit, vp) in enumerate(self.local_idx_exc):
        for i_, unit in enumerate(self.local_idx_exc):
            spike_times = self.spike_times_container[i_]
#            print 'debug', type(spike_times), spike_times, unit
            if spike_times.size > 1:
                nest.SetStatus([self.stimulus[i_]], {'spike_times' : spike_times})
                # get the cell from the list of populations
                mc_idx = (unit - 1) / self.params['n_exc_per_mc']
                hc_idx = (unit - 1) / n_per_hc
                mc_idx_in_hc = mc_idx - hc_idx * self.params['n_mc_per_hc']
                idx_in_pop = (unit - 1) - mc_idx * self.params['n_exc_per_mc']
    #            print 'debug unit %d\ti_=%d\thc_idx = %d\tmc_idx_in_hc = %d\tidx_in_pop = %d' % (unit, i_, hc_idx, mc_idx_in_hc, idx_in_pop)
                nest.Connect([self.stimulus[i_]], [self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_pop]], model='input_exc_0')


    def get_weights_after_learning_cycle(self):
        """
        Saves the weights as adjacency lists (TARGET = Key) as dictionaries to file.
        """

        t_start = time.time()
        print 'NetworkModel.get_weights_after_learning_cycle ...'
        n_my_conns = 0
#        my_units = np.array(self.local_idx_exc)[:, 0] # !GIDs are 1-aligned!
        my_units = self.local_idx_exc # !GIDs are 1-aligned!
        my_adj_list = {}
        for nrn in my_units:
            my_adj_list[nrn] = []
        for src_hc in xrange(self.params['n_hc']):
            print 'get_weights_after_learning_cycle: Proc %d src_hc %d' % (self.pc_id, src_hc)
            for src_mc in xrange(self.params['n_mc_per_hc']):
                src_pop = self.list_of_exc_pop[src_hc][src_mc]
                src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                for tgt_hc in xrange(self.params['n_hc']):
                    for tgt_mc in xrange(self.params['n_mc_per_hc']):
                        tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
                        tgt_pop_idx = tgt_hc * self.params['n_mc_per_hc'] + tgt_mc

                        # get the list of connections stored on the current MPI node
                        conns = nest.GetConnections(src_pop, tgt_pop)
                        for c in conns:
                            cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                            w = cp[0]['weight'] 
                            if w != 0:
                                my_adj_list[c[1]].append((c[0], cp[0]['weight']))
                        n_my_conns += len(conns)

        print 'Proc %d holds %d connections' % (self.pc_id, n_my_conns)
        output_fn = self.params['adj_list_tgt_fn_base'] + 'AS_%d_%d.json' % (self.iteration, self.pc_id)
        print 'Saving connection list to: ', output_fn
        f = file(output_fn, 'w')
        json.dump(my_adj_list, f, indent=0, ensure_ascii=False)

        t_stop = time.time()
        self.times['t_get_weights'] = t_stop - t_start



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

        t_start = time.time()
              
        if record_v:
            mp_r = np.array(self.params['motion_params'])
            # cells well tuned to the normal stimulus
            selected_gids_r, pops = utils.select_well_tuned_cells_trajectory(self.tuning_prop_exc, \
                    mp_r, params, self.params['n_gids_to_record'] / 2, 1)
            mp_l = mp_r.copy()
            # opposite direction
            mp_l[2] *= (-1.)
            # cells well tuned to the normal stimulus
            selected_gids_l, pops = utils.select_well_tuned_cells_trajectory(self.tuning_prop_exc, \
                    mp_l, params, self.params['n_gids_to_record'] / 2, 1)
            print 'Recording cells close to mp_l', mp_l, '\nGIDS:', selected_gids_l
            print 'Recording cells close to mp_r', mp_r, '\nGIDS:', selected_gids_r
             
            gids_to_record = np.r_[selected_gids_r, selected_gids_l]
            np.savetxt(self.params['gids_to_record_fn'], gids_to_record, fmt='%d')
            voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.5})
            nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'volt'}])
            
        exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                nest.ConvergentConnect(self.list_of_exc_pop[hc][mc], exc_spike_recorder)

        # TODO: why is there a conflict between record_v and recording spikes?
        if record_v: 
            for gid in gids_to_record:
#            for i_, (unit, vp) in enumerate(self.local_idx_exc):
                if gid in self.local_idx_exc:
                    hc_idx, mc_idx_in_hc, idx_in_mc = self.get_indices_for_gid(gid)
                    nest.ConvergentConnect(voltmeter, [self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_mc]])
    

        # # # # # # # # # # # # # #
        #     R U N N N I N G     #
        # # # # # # # # # # # # # #
        if self.pc_id == 0:
            print "Running simulation ... "
        nest.Simulate(self.params['t_sim'])
        t_stop = time.time()
        self.times['t_sim'] = t_stop - t_start


    def print_v(self, fn, multimeter):
        """
        This function is copied from Henrik's code,
        it's NOT TESTED, but maybe useful to save the voltages into one 
        file from the beginning
        """
        ids = nest.GetStatus(multimeter, 'events')[0]['senders']
        sender_ids = np.unique(ids)
        volt = nest.GetStatus(multimeter, 'events')[0]['V_m']
        time = nest.GetStatus(multimeter, 'events')[0]['times']
        senders = nest.GetStatus(multimeter, 'events')[0]['senders']
        print 'debug\n', nest.GetStatus(multimeter, 'events')[0]
        print 'volt size', volt.size
        print 'time size', time.size
        print 'senders size', senders.size
        n_col = volt.size / len(sender_ids)
        d = np.zeros((n_col, len(sender_ids) + 1))

        for sender in sender_ids:
            idx = (sender == senders).nonzero()[0]
            print sender, time[idx]
            d[:, sender] = volt[idx]
            d[:, 0] = time[idx]
        np.savetxt(fn, d)

    def print_times(self):
        fn_out = self.params['tmp_folder'] + 'times.json'
        f = file(fn_out, 'w')
        json.dump(self.times, f, indent=0, ensure_ascii=True)

         


if __name__ == '__main__':

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

    t_0 = time.time()

#    if pc_id == 0:
#        ok = raw_input('\nNew folder: %s\nContinue to create this folder structure? Parameters therein will be overwritten\n\ty / Y / blank = OK; anything else --> exit\n' % ps.params['folder_name'])
#        if not ((ok == '') or (ok.capitalize() == 'Y')):
#            print 'quit'
#            exit(1)
#    if comm != None:
#        comm.Barrier()

    # always call set_filenames to update the folder name and all depending filenames (if params are modified and folder names change due to that)!
    ps.set_filenames() 
    ps.create_folders()
    ps.write_parameters_to_file()

    sim_cnt = 0
    load_files = True
    record = True
    save_input_files = not load_files

    NM = NetworkModel(ps.params, iteration=0)

    NM.setup()

    NM.create(load_files)

    if not load_files:
        spike_times_container = NM.create_training_input(load_files=load_files, save_output=save_input_files, with_blank=(not params['training_run']))
    else:
        NM.load_input()
    NM.connect()

#    NM.run_sim(sim_cnt, record_v=record)
#    NM.get_weights_after_learning_cycle()
#    NM.print_times()

#    t_end = time.time()
#    t_diff = t_end - t_0
#    print "Simulating %d cells for %d ms took %.3f seconds or %.2f minutes" % (params['n_cells'], params["t_sim"], t_diff, t_diff / 60.)

