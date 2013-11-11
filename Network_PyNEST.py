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
import nest
import CreateStimuli
import json


class NetworkModel(object):

    def __init__(self, param_tool, iteration=0):

        self.param_tool = param_tool
        self.params = param_tool.params
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
#            self.tuning_prop_inh = utils.set_tuning_prop(self.params, mode='hexgrid', cell_type='inh')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
        else:
            self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
#            self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])

        # update 
        self.param_tool.set_vx_tau_transformation_params(self.tuning_prop_exc[:, 2].min(), self.tuning_prop_exc[:, 2].max())
        self.params = self.param_tool.params

        if self.pc_id == 0:
            print "Saving tuning_prop to file:", self.params['tuning_prop_means_fn']
            np.savetxt(self.params['tuning_prop_means_fn'], self.tuning_prop_exc)
#            print "Saving tuning_prop to file:", self.params['tuning_prop_inh_fn']
#            np.savetxt(self.params['tuning_prop_inh_fn'], self.tuning_prop_inh)

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

        nest.SetKernelStatus({'overwrite_files' : True})
        self.setup_synapse_types()

        # clean spiketimes_folder:
        if self.pc_id == 0:
            cmd = 'rm %s* ' % self.params['spiketimes_folder']
            os.system(cmd)
            cmd = 'rm %s* ' % self.params['volt_folder']
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
        nest.CopyModel('static_synapse', 'inh_exc_unspec', \
                {'weight': self.params['w_ie_unspec'], 'receptor_type': 2})

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
        self.list_of_unspecific_inh_pop = []
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

        # set the cell parameters
        self.spike_times_container = [ np.array([]) for i in xrange(len(self.local_idx_exc))]

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
#                    print 'debug stim before_stim_idx', i_stim, before_stim_idx, before_stim_idx.size
#                    print 'debug stim blank_idx', i_stim, blank_idx, blank_idx.size
                    blank_idx = np.concatenate((blank_idx, before_stim_idx))
                    # blanking
                    for i_time in blank_idx:
                        L_input[:, i_time] = np.random.permutation(L_input[:, i_time])


            nprnd.seed(self.params['input_spikes_seed'])
            # create the spike trains
            print 'Creating input spiketrains...'
            for i_, unit in enumerate(my_units):
                if not (i_ % 10):
                    print 'Creating input spiketrain for unit %d (%d / %d) (%.1f percent)' % (unit, i_, len(my_units), float(i_) / len(my_units) * 100.)
                rate_of_t = np.array(L_input[i_, :])
                # each cell will get its own spike train stored in the following file + cell gid
                n_steps = rate_of_t.size
                spike_times = []
                for i in xrange(n_steps):
                    r = nprnd.rand()
                    if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                        spike_times.append(i * dt)
                self.spike_times_container[i_] = np.array(spike_times)
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
            print 'Connecting exc - exc'
            self.connect_ee()
            print 'Connecting exc - inh unspecific'
            self.connect_ei_unspecific()
            print 'Connecting inh - exc unspecific'
            self.connect_ie_unspecific() # normalizing inhibition
        else: # load the weight matrix
            print 'Connecting exc - exc '
            self.connect_ee()
            self.connect_ee_testing()
            print 'Connecting exc - inh unspecific'
            self.connect_ei_unspecific()
            print 'Connecting exc - inh specific'
            self.connect_ei_specific(load_weights=True)
            print 'Connecting inh - exc unspecific'
            self.connect_ie_unspecific() # normalizing inhibition
            print 'Connecting inh - exc specific'
            self.connect_ie_specific()
            self.connect_ii() # connect unspecific and specific inhibition to excitatory cells

#        self.times['t_connect'] = self.timer.diff()






    def connect_ee(self):

        initial_weight = 0.

        # connect cells within one MC
        for tgt_gid in self.local_idx_exc:
            hc_idx, mc_idx, gid_min, gid_max = self.get_gids_to_mc(tgt_gid)
            n_src = int(round(self.params['n_exc_per_mc'] * self.params['p_ee_local']))
            source_gids = np.array([])
            while source_gids.size != n_src:
                source_gids = np.random.randint(gid_min, gid_max, n_src)
                source_gids = np.unique(source_gids)
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local')


        # setup an all-to-all connectivity
#        for src_hc in xrange(self.params['n_hc']):
#            for src_mc in xrange(self.params['n_mc_per_hc']):
#                src_pop = self.list_of_exc_pop[src_hc][src_mc]
#                src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                # TODO:
                # get the tuning properties for cells in that MC and 
                # calculate the time_constants based on the preferred velocity
#                print 'Debug TODO Need to get the tuning prop of this source pop:', src_pop
#                tp_src = self.tuning_prop_exc[src_pop_idx, 2] # get the v_x tuning for these cells
#                for tgt_hc in xrange(self.params['n_hc']):
#                    for tgt_mc in xrange(self.params['n_mc_per_hc']):
#                        tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
#                        nest.ConvergentConnect(src_pop, tgt_pop, model='bcpnn_synapse')
#                        tgt_pop_idx = tgt_hc * self.params['n_mc_per_hc'] + tgt_mc
#                        nest.SetStatus(nest.GetConnections(src_pop, tgt_pop), {'weight': initial_weight})

        eps = 1e-12
        self.debug_tau_zi = np.zeros((self.params['n_exc'], 3))
        for i_, tgt_gid in enumerate(self.local_idx_exc):
            tgt_hc, tgt_mc, gid_min, gid_max = self.get_gids_to_mc(tgt_gid)
            if not (i_ % 20):
                print 'Connecting exc to tgt %d (%d / %d) (%.1f percent)' % (tgt_gid, i_, len(self.local_idx_exc), float(i_) / len(self.local_idx_exc) * 100.)
            # setup an all-to-all connectivity
#            tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
            for src_hc in xrange(self.params['n_hc']):
                for src_mc in xrange(self.params['n_mc_per_hc']):
                    src_pop = self.list_of_exc_pop[src_hc][src_mc]
                    nest.ConvergentConnect(src_pop, [tgt_gid], model='bcpnn_synapse')#, params={'tau_i':tau_zi[)

#                    if np.any(tau_zi) < 0:
#                        print 'Wrong transformation'
#                        exit(1)
#                    for j_, src_gid in enumerate(src_pop):
#                        vx_src = self.tuning_prop_exc[np.array(src_gid) - 1, 2]
#                        tau_zi = utils.transform_tauzi_from_vx(vx_src, self.params)
#                        self.debug_tau_zi[np.array(src_gid) - 1, 0] = np.array(src_gid) - 1
#                        self.debug_tau_zi[np.array(src_gid) - 1, 1] = vx_src
#                        self.debug_tau_zi[np.array(src_gid) - 1, 2] = tau_zi
#                        nest.Connect([src_gid], [tgt_gid], model='bcpnn_synapse', params={'tau_i':tau_zi, 'weight':initial_weight})

#                        conns = nest.GetConnections([src_gid], [tgt_gid]) # get the list of connections stored on the current MPI node
#                        print 'conns;', conns
#                        for c in conns:
#                            cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
#                            print 'cp', cp
#                            if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
#                                nest.SetStatus(nest.GetConnections([src_gid], [tgt_gid]), {'weight': initial_weight, 'tau_i':tau_zi})
#                                nest.SetStatus(c, {'weight': initial_weight, 'tau_i':tau_zi})

#                        print 'debugtauzi', tgt_gid, src_gid, tau_zi[j_]
#                    nest.ConvergentConnect(src_pop, [tgt_gid], model='bcpnn_synapse', params={'tau_i':tau_zi[)

        fn_out = self.params['parameters_folder'] + 'tau_zi_%d.dat' % (self.pc_id)
        np.savetxt(fn_out, self.debug_tau_zi)


    def connect_ee_testing(self):

        # connect cells within one MC
        for tgt_gid in self.local_idx_exc:
            hc_idx, mc_idx, gid_min, gid_max = self.get_gids_to_mc(tgt_gid)
            n_src = int(round(self.params['n_exc_per_mc'] * self.params['p_ee_local']))
            source_gids = np.array([])
            while source_gids.size != n_src:
                source_gids = np.random.randint(gid_min, gid_max, n_src)
                source_gids = np.unique(source_gids)
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local')

        # setup long-range connectivity based on trained connection matrix
        fn = self.params['conn_mat_fn_base'] + 'ee_' + str(self.iteration) + '.dat'
        assert (os.path.exists(fn)), 'ERROR: Weight matrix not found %s\n\tCorrect training_run flag set?\n\t\tHave you run Network_PyNEST_weight_analysis_after_training.py?\n' % fn
        conn_mat_ee = np.loadtxt(self.params['conn_mat_fn_base'] + 'ee_' + str(self.iteration) + '.dat')
        for src_hc in xrange(self.params['n_hc']):
            for src_mc in xrange(self.params['n_mc_per_hc']):
                src_pop = self.list_of_exc_pop[src_hc][src_mc]
                src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                # TODO:
                # get the tuning properties for cells in that MC and 
                # calculate the time_constants based on the preferred velocity
                print 'Debug TODO Need to get the tuning prop of this source pop:', src_pop
                tp_src = self.tp_src[src_pop_idx, 2] # get the v_x tuning for these cells
                for tgt_hc in xrange(self.params['n_hc']):
                    for tgt_mc in xrange(self.params['n_mc_per_hc']):
                        # TODO:
                        # write tau_syn parameters to file
                        tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
                        nest.ConvergentConnect(src_pop, tgt_pop, model='bcpnn_synapse')
                        tgt_pop_idx = tgt_hc * self.params['n_mc_per_hc'] + tgt_mc
    

    def connect_ei_unspecific(self):
        """
        Iterates over target neurons and takes p_ei_unspec * n_exc_per_hc neurons as sources
        """

        
        for tgt_gid in self.local_idx_inh_unspec:
            n_src = int(round(self.params['p_ei_unspec'] * self.params['n_exc_per_hc']))
            hc_idx = (tgt_gid - self.params['n_exc'] - 1) / self.params['n_inh_unspec_per_hc']
            src_gid_range = (hc_idx * self.params['n_exc_per_hc'] + 1, (hc_idx + 1) * self.params['n_exc_per_hc'] + 1)
            nest.RandomConvergentConnect(range(src_gid_range[0], src_gid_range[1]), [tgt_gid], \
                    n_src, weight=self.params['w_ei_unspec'], delay=1., model='exc_inh_unspec', options={'allow_multapses':False})


    def connect_ie_unspecific(self):
        """
        Iterate over all local exc indices and connect cells with GIDs in the range for the unspecific inh neurons
        with a probability p_ie_unspec to the target neuron.
        """

        
        for tgt_gid in self.local_idx_exc:
            n_src = int(round(self.params['p_ie_unspec'] * self.params['n_inh_unspec_per_hc']))
            hc_idx = (tgt_gid - 1) / self.params['n_exc_per_hc']
            src_gid_range = (hc_idx * self.params['n_inh_unspec_per_hc'] + self.params['n_exc'] + 1, (hc_idx + 1) * self.params['n_inh_unspec_per_hc'] + self.params['n_exc'] + 1)
            nest.RandomConvergentConnect(range(src_gid_range[0], src_gid_range[1]), [tgt_gid], \
                    n_src, weight=self.params['w_ie_unspec'], delay=1., model='exc_inh_unspec', options={'allow_multapses':False})


#            while (source_gids.size != n_src):
#                source_gids = np.random.randint(src_gid_range[0], src_gid_range[1], n_src)
#                source_gids = np.unique(source_gids)
#            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_inh_unspec')

#        for src_hc in xrange(self.params['n_hc']):
#            tgt_pop = self.list_of_unspecific_inh_pop[src_hc]
#            for src_mc in xrange(self.params['n_mc_per_hc']):
#                src_pop = self.list_of_exc_pop[src_hc][src_mc]
#                nest.ConvergentConnect(src_pop, tgt_pop, model='')



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
            if not (i_ % 20):
                print 'Connecting input spiketrain for unit %d (%d / %d) (%.1f percent)' % (unit, i_, len(self.local_idx_exc), float(i_) / len(self.local_idx_exc) * 100.)
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

        # too slow
#        for i_, tgt_gid in enumerate(self.local_idx_exc):
#            if not (i_ % 20):
#                print 'Retrieving weights to exc to tgt %d (%d / %d) (%.1f percent)' % (tgt_gid, i_, len(self.local_idx_exc), float(i_) / len(self.local_idx_exc) * 100.) #            tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
#            for src_hc in xrange(self.params['n_hc']):
#                for src_mc in xrange(self.params['n_mc_per_hc']):
#                    src_pop = self.list_of_exc_pop[src_hc][src_mc]
#                    conns = nest.GetConnections(src_pop, [tgt_gid]) #get the list of connections stored on the current MPI node
#                    for c in conns:
#                        cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
#                        w = cp[0]['weight'] 
#                        if ((cp[0]['synapse_model'] == 'bcpnn_synapse') and (w != 0)):
#                            my_adj_list[c[1]].append((c[0], cp[0]['weight']))
#                    n_my_conns += len(conns)


        for src_hc in xrange(self.params['n_hc']):
            print 'get_weights_after_learning_cycle: Proc %d src_hc %d' % (self.pc_id, src_hc)
            for src_mc in xrange(self.params['n_mc_per_hc']):
                src_pop = self.list_of_exc_pop[src_hc][src_mc]
                for tgt_hc in xrange(self.params['n_hc']):
                    for tgt_mc in xrange(self.params['n_mc_per_hc']):
                        tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
                        conns = nest.GetConnections(src_pop, tgt_pop) # get the list of connections stored on the current MPI node
                        for c in conns:
                            cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                            w = cp[0]['weight'] 
#                            print 'debugw', cp[0]['synapse_model'], w, cp
                            if ((cp[0]['synapse_model'] == 'bcpnn_synapse') and (w != 0)):
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




    def get_indices_for_gid(self, gid):
        """Returns the HC, MC, and within MC index for the gid
        """

        n_per_hc = self.params['n_mc_per_hc'] * self.params['n_exc_per_mc']
        mc_idx = (gid - 1) / self.params['n_exc_per_mc']
        hc_idx = (gid - 1) / n_per_hc
        mc_idx_in_hc = mc_idx - hc_idx * self.params['n_mc_per_hc']
        idx_in_mc = (gid - 1) - mc_idx * self.params['n_exc_per_mc']

        return hc_idx, mc_idx_in_hc, idx_in_mc


    def run_sim(self, sim_cnt):
        t_start = time.time()
              
        # Record spikes 
        exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
        inh_spec_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'inh_spec_spikes'})
        inh_unspec_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'inh_unspec_spikes'})
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                nest.ConvergentConnect(self.list_of_exc_pop[hc][mc], exc_spike_recorder)
                if not self.params['training_run']:
                    nest.ConvergentConnect(self.list_of_specific_inh_pop[hc][mc], inh_spec_spike_recorder)
        for hc in xrange(self.params['n_hc']):
            nest.ConvergentConnect(self.list_of_unspecific_inh_pop[hc], inh_unspec_spike_recorder)
    

        # # # # # # # # # # # # # #
        #     R U N N N I N G     #
        # # # # # # # # # # # # # #
        if self.pc_id == 0:
            print "Running simulation ... "
        nest.Simulate(self.params['t_sim'])
        print "Simulation finished"
        t_stop = time.time()
        self.times['t_sim'] = t_stop - t_start



    def record_v_exc(self):
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
        nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'exc_volt'}])
#        nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'exc_volt'}])

        # TODO: why is there a conflict between record_v and recording spikes?
        for gid in gids_to_record:
#            for i_, (unit, vp) in enumerate(self.local_idx_exc):
            if gid in self.local_idx_exc:
                hc_idx, mc_idx_in_hc, idx_in_mc = self.get_indices_for_gid(gid)
                nest.ConvergentConnect(voltmeter, [self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_mc]])


    def record_v_inh_unspec(self):
        voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.5})
        nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'inh_unspec_volt'}])
        for hc in xrange(self.params['n_hc']):
            nest.DivergentConnect(voltmeter, self.list_of_unspecific_inh_pop[hc])


    def print_v(self, fn, multimeter):
        """
        This function is copied from Henrik's code,
        it's NOT TESTED, but maybe useful to save the voltages into one 
        file from the beginning
        """
        print 'print_v'
        ids = nest.GetStatus(multimeter, 'events')[0]['senders']
        sender_ids = np.unique(ids)
        volt = nest.GetStatus(multimeter, 'events')[0]['V_m']
        time = nest.GetStatus(multimeter, 'events')[0]['times']
        senders = nest.GetStatus(multimeter, 'events')[0]['senders']
#        print 'debug\n', nest.GetStatus(multimeter, 'events')[0]
#        print 'volt size', volt.size
#        print 'time size', time.size
#        print 'senders size', senders.size
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

#    try: 
#        from mpi4py import MPI
#        USE_MPI = True
#        comm = MPI.COMM_WORLD
#        pc_id, n_proc = comm.rank, comm.size
#        print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
#    except:
#        USE_MPI = False
#        pc_id, n_proc, comm = 0, 1, None
#        print "MPI not used"

    t_0 = time.time()
    import simulation_parameters
    ps = simulation_parameters.parameter_storage()
    params = ps.params

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
    record = False
    save_input_files = not load_files

    NM = NetworkModel(ps, iteration=0)

    NM.setup()

    NM.create(load_files)

    if not load_files:
        spike_times_container = NM.create_training_input(load_files=load_files, save_output=save_input_files, with_blank=(not params['training_run']))
    else:
        NM.load_input()
    NM.connect()

    if record:
        NM.record_v_exc()
        NM.record_v_inh_unspec()
    NM.run_sim(sim_cnt)
    NM.get_weights_after_learning_cycle()
#    NM.print_times()

    t_end = time.time()
    t_diff = t_end - t_0
    print "Simulating %d cells for %d ms took %.3f seconds or %.2f minutes" % (params['n_cells'], params["t_sim"], t_diff, t_diff / 60.)

