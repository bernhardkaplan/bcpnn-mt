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
        self.training_params = None # only relevant for testing 


    def setup(self, load_tuning_prop=False):

        print 'NetworkModel.setup'
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

        exit(1)
        self.rf_sizes = self.set_receptive_fields('exc')
        np.savetxt(self.params['receptive_fields_exc_fn'], self.rf_sizes)

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
        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder'], 'resolution': .1, 'overwrite_files' : True})
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
            cmd = 'rm %s* ' % self.params['volt_folder']
            os.system(cmd)


    def set_receptive_fields(self, cell_type):
        """
        Can be called only after set_tuning_prop.
        Receptive field sizes increase linearly depending on their relative position.
        TODO: receptive field sizes for inhibitory neurons
        """
        n_cells = self.params['n_exc']
        rfs = np.zeros((n_cells, 4))
        rfs[:, 0] = self.params['rf_size_x_gradient'] * np.abs(self.tuning_prop_exc[:, 0] - .5) + self.params['rf_size_x_min']
        rfs[:, 1] = self.params['rf_size_y_gradient'] * np.abs(self.tuning_prop_exc[:, 1] - .5) + self.params['rf_size_y_min']
        rfs[:, 2] = self.params['rf_size_vx_gradient'] * np.abs(self.tuning_prop_exc[:, 2]) + self.params['rf_size_vx_min']
        rfs[:, 3] = self.params['rf_size_vy_gradient'] * np.abs(self.tuning_prop_exc[:, 3]) + self.params['rf_size_vy_min']

        return rfs




    def setup_synapse_types(self):

        # STATIC SYNAPSES 
        # input -> exc: AMPA
        nest.CopyModel('static_synapse', 'input_exc_fast', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
        # input -> exc: NMDA
        nest.CopyModel('static_synapse', 'input_exc_slow', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 1})
        # exc - exc local (within a minicolumn) AMPA
        nest.CopyModel('static_synapse', 'exc_exc_local_fast', \
                {'weight': self.params['w_ee_local'], 'receptor_type': 0})
        # exc - exc local (within a minicolumn) NMDA
        nest.CopyModel('static_synapse', 'exc_exc_local_slow', \
                {'weight': self.params['w_ee_local'], 'receptor_type': 1})
        # exc - inh unspecific (within one hypercolumn) AMPA
        nest.CopyModel('static_synapse', 'exc_inh_unspec_fast', \
                {'weight': self.params['w_ei_unspec'], 'receptor_type': 0})
        # exc - inh unspecific (within one hypercolumn) NMDA
        nest.CopyModel('static_synapse', 'exc_inh_unspec_slow', \
                {'weight': self.params['w_ei_unspec'], 'receptor_type': 1})
        # inh - exc unspecific (within one hypercolumn) GABA_A
        nest.CopyModel('static_synapse', 'inh_exc_unspec_fast', \
                {'weight': self.params['w_ie_unspec'], 'receptor_type': 2})
        # inh - exc unspecific (within one hypercolumn) GABA_B
#        nest.CopyModel('static_synapse', 'inh_exc_unspec_slow', \
#                {'weight': self.params['w_ie_unspec'], 'receptor_type': 3})
        # inh - inh unspecific (within one hypercolumn) GABA_A
        nest.CopyModel('static_synapse', 'inh_inh_unspec_fast', \
                {'weight': self.params['w_ii_unspec'], 'receptor_type': 2})
        # inh - inh unspecific (within one hypercolumn) GABA_B
#        nest.CopyModel('static_synapse', 'inh_inh_unspec_slow', \
#                {'weight': self.params['w_ii_unspec'], 'receptor_type': 3})
        # inh - exc global specific (between hypercolumns): GABA_A
        nest.CopyModel('static_synapse', 'inh_exc_specific_fast', \
                {'weight': self.params['w_ie_spec'], 'receptor_type': 2})
        # inh - exc global specific (between hypercolumns): GABA_B
#        nest.CopyModel('static_synapse', 'inh_exc_specific_slow', \
#                {'weight': self.params['w_ie_spec'], 'receptor_type': 3})


        # TRAINED SYNAPSES, weights are set according to the connection matrix
        # exc - exc global (between hypercolumns): AMPA
        nest.CopyModel('static_synapse', 'exc_exc_global_fast', \
                {'receptor_type': 0})
        # exc - exc global (between hypercolumns): AMPA
        nest.CopyModel('static_synapse', 'exc_exc_global_slow', \
                {'receptor_type': 1})
        # exc - inh global specific (between hypercolumns): AMPA
        nest.CopyModel('static_synapse', 'exc_inh_specific_fast', \
                {'receptor_type': 0})
        # exc - inh global specific (between hypercolumns): NMDA
        nest.CopyModel('static_synapse', 'exc_inh_specific_slow', \
                {'receptor_type': 1})

        if (not 'bcpnn_synapse' in nest.Models('synapses')):
            if self.params['Cluster']:
                nest.sr('(/cfs/milner/scratch/b/bkaplan/BCPNN-Module/share/nest/sli) addpath')
                nest.Install('/cfs/milner/scratch/b/bkaplan/BCPNN-Module/lib/nest/pt_module')
            else:
                try:
                    nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
                    nest.Install('pt_module')
                except:
                    nest.Install('pt_module')


    def get_local_indices(self, pop):
        local_nodes = []
        local_nodes_vp = []
        node_info   = nest.GetStatus(pop)
        for i_, d in enumerate(node_info):
            if d['local']:
                local_nodes.append(d['global_id'])
                local_nodes_vp.append((d['global_id'], d['vp']))
        return local_nodes
        

    def initialize_vmem(self, gids):
        for gid in gids:
            nest.SetStatus([gid], {'V_m': self.pyrngs[self.pc_id].normal(self.params['v_init'], self.params['v_init_sigma'])})
            nest.SetStatus([gid], {'C_m': self.pyrngs[self.pc_id].normal(self.params['C_m_mean'], self.params['C_m_sigma'])})


    def create(self):
        """
            # # # # # # # # # # # #
            #     C R E A T E     #
            # # # # # # # # # # # #
        TODO:

        the cell_params dict entries tau_syn should be set according to the cells' tuning properties
        --> instead of populations, create single cells?
        """
        print 'NetworkModel.create'

        self.list_of_exc_pop = [ [] for i in xrange(self.params['n_hc'])]
        self.list_of_unspecific_inh_pop = []
        self.list_of_specific_inh_pop = [ [] for i in xrange(self.params['n_hc'])]
        self.local_idx_exc = []
        self.local_idx_inh_unspec = []
        self.local_idx_inh_spec = []
        self.exc_gids = []
        self.inh_unspec_gids = []
        self.inh_spec_gids = []

        ##### EXC CELLS
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                pop = nest.Create(self.params['neuron_model'], self.params['n_exc_per_mc'], params=self.params['cell_params_exc'])
                self.local_idx_exc += self.get_local_indices(pop)
                self.list_of_exc_pop[hc].append(pop)
                self.exc_gids += pop
        local_gids_dict = {}
        local_gids_dict.update({gid : self.pc_id for gid in self.local_idx_exc})
        f = file(self.params['local_gids_fn_base'] + '%d.json' % (self.pc_id), 'w')
        json.dump(local_gids_dict, f)

        ##### UNSPECIFIC INHIBITORY CELLS
        for hc in xrange(self.params['n_hc']):
            pop = nest.Create(self.params['neuron_model'], self.params['n_inh_unspec_per_hc'], params=self.params['cell_params_inh'])
            self.list_of_unspecific_inh_pop.append(pop)
            self.local_idx_inh_unspec += self.get_local_indices(pop)
            self.inh_unspec_gids += pop

        ##### SPECIFIC INHIBITORY CELLS
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                pop = nest.Create(self.params['neuron_model'], self.params['n_inh_per_mc'], params=self.params['cell_params_inh'])
                self.list_of_specific_inh_pop[hc].append(pop)
                self.local_idx_inh_spec += self.get_local_indices(pop)
                self.inh_spec_gids += pop

        ##### RECORDER NEURONS
        self.recorder_neurons = nest.Create(self.params['neuron_model'], self.params['n_recorder_neurons'], params=self.params['cell_params_recorder_neurons'])

        # set the cell parameters
        self.spike_times_container = [ np.array([]) for i in xrange(len(self.local_idx_exc))]

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
    
        # v_init
        self.initialize_vmem(self.local_idx_exc)
        self.initialize_vmem(self.local_idx_inh_spec)
        self.initialize_vmem(self.local_idx_inh_unspec)

        self.recorder_free_vmem = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.5})
        nest.SetStatus(self.recorder_free_vmem, [{"to_file": True, "withtime": True, 'label' : self.params['free_vmem_fn_base']}])
        nest.DivergentConnect(self.recorder_free_vmem, self.recorder_neurons)



    def create_test_input(self, load_files=False, save_output=False, with_blank=False, training_params=None):
        """
        Keyword arguments:
        load_files -- Load the input spike trains for each cell from the folder params['input_folder']
        save_output -- saves the input rate and spike trains
        with_blank -- for each test stimulus, set a certain time to blank / noise
        training_params-- if not None, it has to be the parameter dictionary used for training 
                        ==> load the same motion parameters as used during training to generate test stimuli
        """
        print 'NetworkModel.create_test_input(load_files=%s, save_output=%s, with_blank=%s, training_params=%s)' % (\
                load_files, save_output, with_blank, training_params!=None)
        if load_files:
            self.load_input() # load previously generated files
            return True

        my_units = np.array(self.local_idx_exc) - 1
        n_cells = len(my_units)
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process
        time = np.arange(0, self.params['t_sim'], dt)
        L_input = np.zeros((n_cells, time.shape[0]))  # the envelope of the Poisson process
        CS = CreateStimuli.CreateStimuli()
        motion_params = CS.create_test_stim_1D(self.params, training_params=training_params)
        np.savetxt(self.params['test_sequence_fn'], motion_params)
        n_stim_total = self.params['n_test_stim']

        stimuli = range(self.params['test_stim_range'][0], self.params['test_stim_range'][1])
        for i_stim, stim_idx in enumerate(stimuli):
            print 'Calculating input signal for training stim %d / %d (%.1f percent)' % (i_stim, n_stim_total, float(i_stim) / n_stim_total * 100.)
            x0, v0 = motion_params[stim_idx, 0], motion_params[stim_idx, 2]

            # get the input signal
            idx_t_start = np.int(i_stim * self.params['t_test_stim'] / dt)
            idx_t_stop = np.int((i_stim + 1) * self.params['t_test_stim'] / dt) 
            idx_within_stim = 0
            for i_time in xrange(idx_t_start, idx_t_stop):
                time_ = (idx_within_stim * dt) / self.params['t_stimulus']
#                x_stim = x0 + time_ * v0
                x_stim = (x0 + time_ * v0) % self.params['torus_width']
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
                blank_idx = np.concatenate((blank_idx, before_stim_idx))
                # blanking
                for i_time in blank_idx:
                    L_input[:, i_time] = np.random.permutation(L_input[:, i_time])

            # make a pause between the test stimuli
            idx_t_start_pause = np.int(((i_stim + 1) * self.params['t_test_stim']  - self.params['t_training_pause'])/ dt) 
            idx_t_stop_pause = np.int((i_stim + 1) * self.params['t_test_stim'] / dt) 
            print 'Debug idx_t_start_pause', idx_t_start_pause
            print 'Debug idx_t_stop_pause', idx_t_stop_pause
            L_input[:, idx_t_start_pause:idx_t_stop_pause] = 0.

        nprnd.seed(self.params['input_spikes_seed'])
        # create the spike trains
        print 'Creating input spiketrains...'
        for i_, unit in enumerate(my_units):
            if not (i_ % 10):
                print 'Creating input spiketrain for unit %d (%d / %d) (%.1f percent)' % (unit, i_, len(my_units), float(i_) / len(my_units) * 100.)
            rate_of_t = np.array(L_input[i_, :])
            # each cell will get its own spike train stored in the following file + cell gid
            n_steps = rate_of_t.size
#            spike_times = []
            r = nprnd.rand(n_steps)
            spike_idx = (r <= (rate_of_t/1000.) * dt).nonzero()[0]
            spike_times = spike_idx * dt
            print 'Debug unit %d receives in total: %d spikes' % (gid, spike_times)

#            for i in xrange(n_steps):
#                r = nprnd.rand()
#                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
#                    spike_times.append(i * dt)
            self.spike_times_container[i_] = np.array(spike_times)
            if (save_output and self.spike_times_container[i_].size > 1):
                #output_fn = self.params['input_rate_fn_base'] + str(unit) + '_stim%d-%d.dat' % (self.params['test_stim_range'][0], self.params['test_stim_range'][1])
                #np.savetxt(output_fn, rate_of_t)
                output_fn = self.params['input_st_fn_base'] + str(unit) + '_stim%d-%d.dat' % (self.params['test_stim_range'][0], self.params['test_stim_range'][1])
                np.savetxt(output_fn, np.array(spike_times))
        return True



    def create_training_input(self, load_files=False, save_output=False, with_blank=False):

        if load_files:
            self.load_input()
            return True

        #else:
        if self.pc_id == 0:
            print "Computing input spiketrains..."

        my_units = np.array(self.local_idx_exc) - 1
#            my_units = np.array(self.local_idx_exc)[:, 0] - 1
        n_cells = len(my_units)
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process
        time = np.arange(0, self.params['t_sim'], dt)
        print 'L_input shape:', n_cells, time.shape[0]
        L_input = np.zeros((n_cells, time.shape[0]))  # the envelope of the Poisson process

        # get the order of training stimuli
        CS = CreateStimuli.CreateStimuli()
        random_order = self.params['random_training_order']
        motion_params = CS.create_motion_sequence_1D(self.params, random_order)
        np.savetxt(self.params['training_sequence_fn'], motion_params)
#        print 'quit'
#        exit(1)
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
                x_stim = (x0 + time_ * v0) % self.params['torus_width']
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

            idx_t_start_pause = np.int(((i_stim + 1) * self.params['t_training_stim']  - self.params['t_training_pause'])/ dt) 
            idx_t_stop_pause = np.int((i_stim + 1) * self.params['t_training_stim'] / dt) 
#            print 'Debug idx_t_start_pause', idx_t_start_pause
#            print 'Debug idx_t_stop_pause', idx_t_stop_pause
            L_input[:, idx_t_start_pause:idx_t_stop_pause] = 0.
        

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
                #output_fn = self.params['input_rate_fn_base'] + str(unit) + '.dat'
                #np.savetxt(output_fn, rate_of_t)
                output_fn = self.params['input_st_fn_base'] + str(unit) + '.dat'
                np.savetxt(output_fn, np.array(spike_times))
        return True
    


    def load_input(self):

        if self.pc_id == 0:
            print "Loading input spiketrains..."
        for i_, tgt_gid_nest in enumerate(self.local_idx_exc):
            print 'Loading input for cell %d / %d (%.1f percent)' % (i_, len(self.local_idx_exc), float(i_) / len(self.local_idx_exc) * 100.)
            if self.params['training_run']:
                gid = tgt_gid_nest - 1
                fn = self.params['input_st_fn_base'] + str(gid) + '.dat'
                if os.path.exists(fn):
                    spike_times = np.around(np.loadtxt(fn), decimals=1)
                else: # this cell does not get any input
                    print "Missing file: ", fn
                    spike_times = np.array([])
            else:
                gid = tgt_gid_nest - 1
                fn = self.params['input_rate_fn_base'] + str(gid) + '_stim%d-%d.dat' % (self.params['test_stim_range'][0], self.params['test_stim_range'][1])
                if os.path.exists(fn):
                    spike_times = np.around(np.loadtxt(fn), decimals=1)
                    print 'Loaded %d spikes for cell %d' % (spike_times.size, gid)
                else:
                    print "Missing file: ", fn
                    spike_times = np.array([])

            if type(spike_times) == type(1.0): # it's just one spike
                self.spike_times_container[i_] = np.array([spike_times])
            else:
                self.spike_times_container[i_] = spike_times



    def connect(self):
        nest.SetDefaults('bcpnn_synapse', params=self.params['bcpnn_params'])

        self.connect_input_to_exc()

        if self.params['training_run']:
            print 'Connecting exc - exc'
            self.connect_ee() # within MCs and bcpnn-all-to-all connections
            print 'Connecting exc - inh unspecific'
            self.connect_ei_unspecific()
            print 'Connecting inh - exc unspecific'
            self.connect_ie_unspecific() # normalizing inhibition
            self.connect_input_to_recorder_neurons()
        else: # load the weight matrix
            # setup long-range connectivity based on trained connection matrix
            self.load_training_weights()
            print 'Connecting exc - exc '
            self.connect_ee()
            self.connect_ee_testing()
            self.connect_input_to_recorder_neurons()
            print 'Connecting exc - inh unspecific'
            self.connect_ei_unspecific()
            print 'Connecting exc - inh specific'
#            self.connect_ei_specific()
            print 'Connecting inh - exc unspecific'
            self.connect_ie_unspecific() # normalizing inhibition
            print 'Connecting inh - exc specific'
#            self.connect_ie_specific()
            self.connect_ii() # connect unspecific and specific inhibition to excitatory cells
            self.connect_recorder_neurons()

#        self.times['t_connect'] = self.timer.diff()


    def connect_ie_specific():
        """
        Connect the specific inhibitory neurons (local to one minicolumn) to the excitatory cells in that minicolumn.
        RSNP -> PYR
        """
        # connect cells within one MC
        for tgt_gid in self.local_idx_exc:
            hc_idx, mc_idx, gid_min, gid_max = self.get_gids_to_mc(tgt_gid)
            gid_min += self.params['inh_spec_offset']
            gid_max += self.params['inh_spec_offset']
            n_src = int(round(self.params['n_inh_per_mc'] * self.params['p_ie_spec']))
            source_gids = np.random.randint(gid_min, gid_max, n_src)
            source_gids = np.unique(source_gids)
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='inh_exc_specific_fast')



    def connect_ii(self):

        for tgt_gid in self.local_idx_inh_unspec:
            n_src = int(round(self.params['p_ii_unspec'] * self.params['n_inh_unspec_per_hc']))
            hc_idx = (tgt_gid - self.params['n_inh_unspec_per_hc'] - 1) / self.params['n_inh_unspec_per_hc']
            src_gid_range = (hc_idx * self.params['n_inh_unspec_per_hc'] + 1, (hc_idx + 1) * self.params['n_inh_unspec_per_hc'] + 1)
            nest.RandomConvergentConnect(range(src_gid_range[0], src_gid_range[1]), [tgt_gid], \
                    n_src, weight=self.params['w_ii_unspec'], delay=1., model='inh_inh_unspec_fast', options={'allow_multapses':False})


    def connect_input_to_recorder_neurons(self):

        sorted_gids = np.argsort(self.tuning_prop_exc[:, 0])
        rnd_ = np.linspace(0, self.params['n_exc'], self.params['n_recorder_neurons'], endpoint=False)
        rnd_gids = np.array(rnd_, dtype=int)
        gids = sorted_gids[rnd_gids]
        # gids is the neurons that are to 'copied' for recorder neurons
        self.recorder_neuron_gid_mapping = {}

        print 'DEBUG recorder gids:', gids
        # 1) connect the same input to recorder neurons as to the normal neurons
        if self.params['training_run']:
            spike_times_container = self.create_training_input_for_cells(gids, with_blank=False)
        else:
            spike_times_container = self.create_training_input_for_cells(gids, with_blank=True)
        self.recorder_stimulus = nest.Create('spike_generator', len(gids))

        for i_, unit in enumerate(gids):
            spike_times = spike_times_container[i_]
            nest.SetStatus([self.recorder_stimulus[i_]], {'spike_times' : np.sort(spike_times)})
            nest.Connect([self.recorder_stimulus[i_]], [self.recorder_neurons[i_]], model='input_exc_fast')
            nest.Connect([self.recorder_stimulus[i_]], [self.recorder_neurons[i_]], model='input_exc_slow')
            self.recorder_neuron_gid_mapping[self.recorder_neurons[i_]] = unit

        f = file(self.params['recorder_neurons_gid_mapping'], 'w')
        json.dump(self.recorder_neuron_gid_mapping, f, indent=2)



    def create_training_input_for_cells(self, gids, with_blank):

        n_cells = len(gids)
        spike_times_container = [ np.array([]) for i in xrange(len(gids))]
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process
        time = np.arange(0, self.params['t_sim'], dt)
        L_input = np.zeros((n_cells, time.shape[0]))  # the envelope of the Poisson process

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
                x_stim = (x0 + time_ * v0) % self.params['torus_width']
                L_input[:, i_time] = utils.get_input(self.tuning_prop_exc[gids, :], self.params, (x_stim, 0, v0, 0, 0))
                L_input[:, i_time] *= self.params['f_max_stim']
                if (i_time % 5000 == 0):
                    print "t: %.2f [ms]" % (time_ * self.params['t_stimulus'])
                idx_within_stim += 1
        
            if with_blank:
                start_blank = idx_t_start + 1. / dt * self.params['t_before_blank']
                stop_blank = idx_t_start + 1. / dt * (self.params['t_before_blank'] + self.params['t_blank'])
                blank_idx = np.arange(start_blank, stop_blank)
                before_stim_idx = np.arange(idx_t_start, self.params['t_start'] * 1./dt + idx_t_start)
                blank_idx = np.concatenate((blank_idx, before_stim_idx))
                # blanking
                for i_time in blank_idx:
                    L_input[:, i_time] = np.random.permutation(L_input[:, i_time])

        nprnd.seed(self.params['input_spikes_seed'])
        # create the spike trains
        print 'Creating input spiketrains...'
        for i_, unit in enumerate(gids):
            if not (i_ % 10):
                print 'Creating input spiketrain for unit %d (%d / %d) (%.1f percent)' % (unit, i_, len(gids), float(i_) / len(gids) * 100.)
            rate_of_t = np.array(L_input[i_, :])
            # each cell will get its own spike train stored in the following file + cell gid
            n_steps = rate_of_t.size
            spike_times = []
            for i in xrange(n_steps):
                r = nprnd.rand()
                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    spike_times.append(i * dt)
            spike_times_container[i_] = np.array(spike_times)
            output_fn = self.params['input_rate_fn_base'] + str(unit) + '.dat'
            np.savetxt(output_fn, rate_of_t)
            output_fn = self.params['input_st_fn_base'] + str(unit) + '.dat'
            np.savetxt(output_fn, np.array(spike_times))
        return spike_times_container


    def connect_ee(self):

        initial_weight = 0.

        # connect cells within one MC
        for tgt_gid in self.local_idx_exc:
            hc_idx, mc_idx, gid_min, gid_max = self.get_gids_to_mc(tgt_gid)
            n_src = int(round(self.params['n_exc_per_mc'] * self.params['p_ee_local']))
            source_gids = np.random.randint(gid_min, gid_max, n_src)
            source_gids = np.unique(source_gids)
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local_fast')
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local_slow')

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

#        self.debug_tau_zi = np.zeros((self.params['n_exc'], 3))
        if self.params['training_run']:
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

#        fn_out = self.params['parameters_folder'] + 'tau_zi_%d.dat' % (self.pc_id)
#        np.savetxt(fn_out, self.debug_tau_zi)


    def connect_ee_testing(self):

        # connect cells within one MC
        for tgt_gid in self.local_idx_exc:
            hc_idx, mc_idx, gid_min, gid_max = self.get_gids_to_mc(tgt_gid)
            n_src = int(round(self.params['n_exc_per_mc'] * self.params['p_ee_local']))
            source_gids = np.random.randint(gid_min, gid_max, n_src)
            source_gids = np.unique(source_gids)
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local_fast')
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local_slow')

        w_debug = np.zeros(self.conn_mat_training.shape)
        # connect the minicolumns according to their weight after training
        for src_hc in xrange(self.params['n_hc']):
            for src_mc in xrange(self.params['n_mc_per_hc']):
                src_pop = self.list_of_exc_pop[src_hc][src_mc]
                src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                # TODO:
                # get the tuning properties for cells in that MC and 
                # calculate the time_constants based on the preferred velocity
#                print 'Debug TODO Need to get the tuning prop of this source pop:', src_pop
                tp_src = self.tuning_prop_exc[src_pop_idx, 2] # get the v_x tuning for these cells
                for tgt_hc in xrange(self.params['n_hc']):
                    for tgt_mc in xrange(self.params['n_mc_per_hc']):
                        # TODO:
                        # write tau_syn parameters to file
                        tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
                        tgt_pop_idx = tgt_hc * self.params['n_mc_per_hc'] + tgt_mc
                        w = self.conn_mat_training[src_pop_idx, tgt_pop_idx]
                        if w != 0:
                            w_ = self.transform_weight(w)
                            w_debug[src_pop_idx, tgt_pop_idx] = w_
#                            print 'debug', src_pop, tgt_pop, w_, self.params['delay_ee_global']
                            nest.ConvergentConnect(src_pop, tgt_pop, weight=[w_], delay=[self.params['delay_ee_global']], \
                                    model='exc_exc_global_fast')
                            nest.ConvergentConnect(src_pop, tgt_pop, weight=[w_], delay=[self.params['delay_ee_global']], \
                                    model='exc_exc_global_slow')
        debug_fn = self.params['connections_folder'] + 'w_conn_ee_debug.txt'
        print 'Saving debug connection matrix to:', debug_fn
        np.savetxt(debug_fn, w_debug)
    
    def load_training_weights(self):
        fn = self.training_params['conn_mat_fn_base'] + 'ee_' + str(self.iteration) + '.dat'
        assert (os.path.exists(fn)), 'ERROR: Weight matrix not found %s\n\tCorrect training_run flag set?\n\t\tHave you run Network_PyNEST_weight_analysis_after_training.py?\n' % fn
        print 'Loading connection matrix from training:', fn
        self.conn_mat_training = np.loadtxt(fn)
        self.w_bcpnn_max = np.max(self.conn_mat_training)
        self.w_bcpnn_min = np.min(self.conn_mat_training)


    def transform_weight(self, w):
        if w > 0:
            w_ = w * self.params['w_ee_global_max'] / self.w_bcpnn_max
        elif w < 0:
            w_ = -1. * w * self.params['w_ei_global_max'] / self.w_bcpnn_min
        return w_



    def connect_ei_unspecific(self):
        """
        Iterates over target neurons and takes p_ei_unspec * n_exc_per_hc neurons as sources
        """
        for tgt_gid in self.local_idx_inh_unspec:
            n_src = int(round(self.params['p_ei_unspec'] * self.params['n_exc_per_hc']))
            hc_idx = (tgt_gid - self.params['n_exc'] - 1) / self.params['n_inh_unspec_per_hc']
            src_gid_range = (hc_idx * self.params['n_exc_per_hc'] + 1, (hc_idx + 1) * self.params['n_exc_per_hc'] + 1)
            source_gids = np.random.randint(src_gid_range[0], src_gid_range[1], n_src)
            source_gids = np.unique(source_gids)
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_inh_unspec_fast')
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_inh_unspec_slow')


    def connect_ie_unspecific(self):
        """
        Iterate over all local exc indices and connect cells with GIDs in the range for the unspecific inh neurons
        with a probability p_ie_unspec to the target neuron.
        """
        for tgt_gid in self.local_idx_exc:
            n_src = int(round(self.params['p_ie_unspec'] * self.params['n_inh_unspec_per_hc']))
            hc_idx = (tgt_gid - 1) / self.params['n_exc_per_hc']
            src_gid_range = (hc_idx * self.params['n_inh_unspec_per_hc'] + self.params['n_exc'] + 1, (hc_idx + 1) * self.params['n_inh_unspec_per_hc'] + self.params['n_exc'] + 1)
            source_gids = np.random.randint(src_gid_range[0], src_gid_range[1], n_src)
            source_gids = np.unique(source_gids)
            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='inh_exc_unspec_fast')
#            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='inh_exc_unspec_slow')


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
        for i_, unit in enumerate(self.local_idx_exc):
            if not (i_ % 20):
                print 'Connecting input spiketrain for unit %d (%d / %d) (%.1f percent)' % (unit, i_, len(self.local_idx_exc), float(i_) / len(self.local_idx_exc) * 100.)
            spike_times = self.spike_times_container[i_]
            if spike_times.size > 1:
                nest.SetStatus([self.stimulus[i_]], {'spike_times' : np.sort(spike_times)})
                # get the cell from the list of populations
                mc_idx = (unit - 1) / self.params['n_exc_per_mc']
                hc_idx = (unit - 1) / n_per_hc
                mc_idx_in_hc = mc_idx - hc_idx * self.params['n_mc_per_hc']
                idx_in_pop = (unit - 1) - mc_idx * self.params['n_exc_per_mc']
                nest.Connect([self.stimulus[i_]], [self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_pop]], model='input_exc_fast')
                nest.Connect([self.stimulus[i_]], [self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_pop]], model='input_exc_slow')


    def get_p_values(self, neurons):
        conns = nest.GetConnections([neurons[0]], [neurons[1]])
#        print 'debug', conns
        if len(conns) > 0:
            cp = nest.GetStatus([conns[0]])
            pi = cp[0]['p_i']
            pj = cp[0]['p_j']
            pij = cp[0]['p_ij']
            wij = cp[0]['weight']
#            print 'debug tracking', conns[0], cp
            return (pi, pj, pij, wij)
        return False


    def check_if_conn_on_node(self, pre_gid, post_gid):
        hc_idx, mc_idx_in_hc, idx_in_mc = self.get_indices_for_gid(pre_gid)
        pre_neuron = self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_mc]
        hc_idx, mc_idx_in_hc, idx_in_mc = self.get_indices_for_gid(post_gid)
        post_neuron = self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_mc]
        on_node = self.get_p_values([pre_neuron, post_neuron])
        if on_node == False:
            return False
        else:
            return True

    def get_weights_after_learning_cycle(self):
        """
        Saves the weights as adjacency lists (TARGET = Key) as dictionaries to file.
        """

        t_start = time.time()
        print 'NetworkModel.get_weights_after_learning_cycle ...'
        n_my_conns = 0
#        my_units = np.array(self.local_idx_exc)[:, 0] # !GIDs are 1-aligned!
        my_units = self.local_idx_exc # !GIDs are 1-aligned!
        my_adj_list_tgt = {}
#        my_adj_list_src = {}
        for nrn in my_units:
            my_adj_list_tgt[nrn] = []

        my_pi_values = {}
        my_pj_values = {}
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
                            if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                                pi = cp[0]['p_i']
                                my_pi_values[c[0]] = cp[0]['p_i']
                                my_pj_values[c[0]] = cp[0]['p_j']
                                pj = cp[0]['p_j']
                                pij = cp[0]['p_ij']
                                w = np.log(pij / (pi * pj))
                                # this does not work for some reason: 
                                # w = cp[0]['weight'] 
                                if (w != 0):
                                    my_adj_list_tgt[c[1]].append([c[0], w])
                                # update the source-indexed adjacency list
#                                if c[0] not in my_adj_list_src.keys():
#                                    my_adj_list_src[c[0]] = []
#                                else:
#                                    my_adj_list_src[c[0]].append([c[1], w])
                                 
                        n_my_conns += len(conns)

        print 'Proc %d holds %d connections' % (self.pc_id, n_my_conns)
        output_fn = self.params['adj_list_tgt_fn_base'] + 'AS_%d_%d.json' % (self.iteration, self.pc_id)
        print 'Saving connection list to: ', output_fn
        f = file(output_fn, 'w')
        json.dump(my_adj_list_tgt, f, indent=2)
        f.flush()

#        output_fn = self.params['adj_list_src_fn_base'] + 'AS_%d_%d.json' % (self.iteration, self.pc_id)
#        print 'Saving connection list to: ', output_fn
#        f = file(output_fn, 'w')
#        json.dump(my_adj_list_src, f, indent=2)
#        f.flush()

        t_stop = time.time()
        self.times['t_get_weights'] = t_stop - t_start
#        self.get_weights_to_recorder_neurons()



    def connect_recorder_neurons(self):
        """
        Must be called after the normal connect
        """

        my_adj_list = {}
        f = file(self.params['recorder_neurons_gid_mapping'], 'r')
        gid_mapping = json.load(f)
        for rec_nrn in self.recorder_neurons:
            mirror_gid = gid_mapping[rec_nrn]
            for src_hc in xrange(self.params['n_hc']):
                for src_mc in xrange(self.params['n_mc_per_hc']):
                    src_pop = self.list_of_exc_pop[src_hc][src_mc]
                    conns = nest.GetConnections(src_pop, [mirror_gid]) # get the list of connections stored on the current MPI node
                    print 'Debug conns', conns
        exit(1)
#                for c in conns:
#                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
#                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
#                        pi = cp[0]['p_i']
#                        pj = cp[0]['p_j']
#                        pij = cp[0]['p_ij']
#                        w = np.log(pij / (pi * pj))
#                        if (w != 0):
#                            my_adj_list[c[1]].append((c[0], w))
#        output_fn = self.params['adj_list_tgt_fn_base'] + 'recorderNrns_%d_%d.json' % (self.iteration, self.pc_id)
#        print 'Saving connection list to: ', output_fn
#        f = file(output_fn, 'w')
#        json.dump(my_adj_list, f, indent=0, ensure_ascii=False)




    def get_indices_for_gid(self, gid):
        """Returns the HC, MC, and within MC index for the gid
        """

        n_per_hc = self.params['n_mc_per_hc'] * self.params['n_exc_per_mc']
        mc_idx = (gid - 1) / self.params['n_exc_per_mc']
        hc_idx = (gid - 1) / n_per_hc
        mc_idx_in_hc = mc_idx - hc_idx * self.params['n_mc_per_hc']
        idx_in_mc = (gid - 1) - mc_idx * self.params['n_exc_per_mc']

        return hc_idx, mc_idx_in_hc, idx_in_mc


              
    def run_sim(self, sim_time):
        t_start = time.time()
        # # # # # # # # # # # # # #
        #     R U N N N I N G     #
        # # # # # # # # # # # # # #
        if self.pc_id == 0:
            print "Running simulation for %d milliseconds" % (sim_time)
        nest.Simulate(sim_time)
        t_stop = time.time()
        t_diff = t_stop - t_start
        print "Simulation finished: %d [sec]" % (t_diff)


    def record_v_exc(self):
        mp_r = np.array(self.params['motion_params'])
        # cells well tuned to the normal stimulus
        selected_gids_r, pops = utils.select_well_tuned_cells_trajectory(self.tuning_prop_exc, \
                mp_r, self.params, self.params['n_gids_to_record'] / 2, 1)
        mp_l = mp_r.copy()
        # opposite direction
        mp_l[2] *= (-1.)
        # cells well tuned to the normal stimulus
        selected_gids_l, pops = utils.select_well_tuned_cells_trajectory(self.tuning_prop_exc, \
                mp_l, self.params, self.params['n_gids_to_record'] / 2, 1)
        print 'Recording cells close to mp_l', mp_l, '\nGIDS:', selected_gids_l
        print 'Recording cells close to mp_r', mp_r, '\nGIDS:', selected_gids_r
         
        if len(self.params['gids_to_record']) > 0:
            gids_to_record = np.r_[self.params['gids_to_record'], selected_gids_r, selected_gids_l]
        else:
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


    def merge_local_gid_files(self):
        print 'Updating the local_gid file ...'
        local_gids_dict = {}
        for pid in xrange(self.n_proc):
            f = file(self.params['local_gids_fn_base'] + '%d.json' % (pid), 'r')
            d = json.load(f)
            local_gids_dict.update(d)

        f = file(self.params['local_gids_merged_fn'], 'w')
        json.dump(local_gids_dict, f, indent=1)
        f.flush()



