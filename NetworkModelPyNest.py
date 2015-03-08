import time
t0 = time.time()
import numpy as np
import numpy.random as nprnd
import sys
#import NeuroTools.parameters as ntp
import os
import utils
import nest
import CreateInput
import json
import set_tuning_properties
from copy import deepcopy
import WeightAnalyser
import create_training_stimuli as CS

class NetworkModel(object):

    def __init__(self, params, iteration=0, comm=None):

        self.params = params
        self.debug_connectivity = True
        self.iteration = 0  # the learning iteration (cycle)
        self.times = {}
        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.comm = comm # mpi communicator needed to broadcast nspikes between processes
        if comm != None:
            assert (comm.rank == self.pc_id), 'mpi4py and NEST tell me different PIDs!'
            assert (comm.size == self.n_proc), 'mpi4py and NEST tell me different PIDs!'
            self.comm.Barrier()


    def setup(self, training_stimuli=None):
#        if training_params != None:
#            self.training_params = training_params
        if self.params['regular_tuning_prop']:
            self.tuning_prop_exc, self.rf_sizes = set_tuning_properties.set_tuning_properties_regular(self.params)
        else:
            self.tuning_prop_exc, self.rf_sizes = set_tuning_properties.set_tuning_prop_1D_with_const_fovea_and_const_velocity(self.params)

        if self.pc_id == 0:
            print "Saving tuning_prop to file:", self.params['tuning_prop_exc_fn']
            np.savetxt(self.params['tuning_prop_exc_fn'], self.tuning_prop_exc)
            np.savetxt(self.params['receptive_fields_exc_fn'], self.rf_sizes)

        if self.comm != None:
            self.comm.Barrier()
#        exit(1)
            
        if self.params['training_run']:
            if training_stimuli == None:
                if self.params['regular_tuning_prop']:
                    training_stimuli = CS.create_regular_training_stimuli(self.params, self.tuning_prop_exc)
                else:
                    training_stimuli = CS.create_training_stimuli_based_on_tuning_prop(self.params)
            self.motion_params = training_stimuli
            np.savetxt(self.params['training_stimuli_fn'], self.motion_params)
        else:
            self.CI = CreateInput.CreateInput(self.params, self.tuning_prop_exc, self.rf_sizes)
#            self.motion_params = self.CI.create_test_stim_1D_from_training_stim(self.params, self.training_params)
            self.motion_params = self.CI.create_test_stim_grid(self.params)
            np.savetxt(self.params['test_sequence_fn'], self.motion_params)

        if self.comm != None:
            self.comm.Barrier()

        self.stim_durations = np.zeros(self.params['n_stim'])
        print 'DEBUG stim_durations:'
        for i_, stim_idx in enumerate(range(self.params['stim_range'][0], self.params['stim_range'][1])):
            stim_params = self.motion_params[i_, :]
            t_exit = utils.compute_stim_time(stim_params) + self.params['t_stim_pause']
            self.stim_durations[i_] = t_exit
            print 'stim_idx %d (%d) t_exit: %d stim_duration: %d' % (i_, stim_idx, t_exit, self.stim_durations[i_])

        t_sim = self.stim_durations.sum()
        self.params['t_sim'] = t_sim
        self.update_bcpnn_params()
        np.savetxt(self.params['stim_durations_fn'], self.stim_durations)
        print 'NetworkModel.setup preparing for %.1f [ms] simulation' % (self.params['t_sim'])
        self.projections = {}
        self.projections['ee'] = []
        self.projections['ei'] = []
        self.projections['ie'] = []
        self.projections['ii'] = []

        # update 
        utils.set_vx_tau_transformation_params(self.params, self.tuning_prop_exc[:, 2].min(), self.tuning_prop_exc[:, 2].max())

#        exit(1)
        # # # # # # # # # # # #
        #     S E T U P       #
        # # # # # # # # # # # #
        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder'], 'resolution': self.params['dt_sim'], 'overwrite_files' : True})
        (delay_min, delay_max) = self.params['delay_range']
#        nest.SetKernelStatus({'tics_per_ms':self.params['dt_sim'], 'min_delay':delay_min, 'max_delay':delay_max})

        # # # # # # # # # # # # # # # # # # # # # # # # #
        #     R A N D O M    D I S T R I B U T I O N S  #
        # # # # # # # # # # # # # # # # # # # # # # # # #
        self.RNG = np.random.RandomState(self.params['visual_stim_seed'] + self.pc_id)
        nprnd.seed(self.params['input_spikes_seed'] + self.pc_id)
#        self.RNGs = [np.random.RandomState(s) for s in range(self.params['visual_stim_seed'], self.params['visual_stim_seed'] + self.n_proc)]
        nest.SetKernelStatus({'grng_seed' : self.params['seed'] + self.n_proc})
        nest.SetKernelStatus({'rng_seeds' : range(self.params['seed'] + self.n_proc + 1, \
                self.params['seed'] + 2 * self.n_proc + 1)})

        self.setup_synapse_types()

#    def set_receptive_fields(self, cell_type):
#        """
#        Can be called only after set_tuning_prop.
#        Receptive field sizes increase linearly depending on their relative position.
#        TODO: receptive field sizes for inhibitory neurons
#        """
#        n_cells = self.params['n_exc']
#        rfs = np.zeros((n_cells, 4))
#        rfs[:, 0] = self.params['rf_size_x_gradient'] * np.abs(self.tuning_prop_exc[:, 0] - .5) + self.params['rf_size_x_min']
#        rfs[:, 1] = self.params['rf_size_y_gradient'] * np.abs(self.tuning_prop_exc[:, 1] - .5) + self.params['rf_size_y_min']
#        rfs[:, 2] = self.params['rf_size_vx_gradient'] * np.abs(self.tuning_prop_exc[:, 2]) + self.params['rf_size_vx_min']
#        rfs[:, 3] = self.params['rf_size_vy_gradient'] * np.abs(self.tuning_prop_exc[:, 3]) + self.params['rf_size_vy_min']

#        return rfs




    def setup_synapse_types(self):
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

        # STATIC SYNAPSES 
        # input -> exc: AMPA
        nest.CopyModel('static_synapse', 'input_exc_fast', \
                {'weight': self.params['w_input_exc'], 'delay': 0.1, 'receptor_type': self.params['syn_ports']['ampa']})  # numbers must be consistent with cell_params_exc
        # input -> exc: NMDA
        #nest.CopyModel('static_synapse', 'input_exc_slow', \
                #{'weight': self.params['w_input_exc'], 'delay': 0.1, 'receptor_type': self.params['syn_ports']['nmda']})
        # trigger -> exc: AMPA
        nest.CopyModel('static_synapse', 'trigger_synapse', \
                {'weight': self.params['w_trigger'], 'delay': 0.1, 'receptor_type': self.params['syn_ports']['ampa']})  # numbers must be consistent with cell_params_exc

        # noise synapses
        nest.CopyModel('static_synapse', 'noise_syn_exc', \
                {'weight': self.params['w_noise_exc'], 'delay': 0.1, 'receptor_type': self.params['syn_ports']['ampa']})  # numbers must be consistent with cell_params_exc
#        nest.CopyModel('static_synapse', 'noise_syn_exc', \
#                {'weight': self.params['w_noise_exc'], 'delay': 0.1, 'receptor_type': self.params['syn_ports']['nmda']})  # numbers must be consistent with cell_params_exc
        nest.CopyModel('static_synapse', 'noise_syn_inh', \
                {'weight': self.params['w_noise_inh'], 'delay': 0.1, 'receptor_type': self.params['syn_ports']['gaba']})  # numbers must be consistent with cell_params_exc


        # exc - inh unspecific (within one hypercolumn) AMPA
        nest.CopyModel('static_synapse', 'exc_inh_unspec_fast', \
                {'weight': self.params['w_ei_unspec'], 'delay': self.params['delay_ei_unspec'], 'receptor_type': self.params['syn_ports']['ampa']})
        # exc - inh unspecific (within one hypercolumn) NMDA
        nest.CopyModel('static_synapse', 'exc_inh_unspec_slow', \
                {'weight': self.params['w_ei_unspec'], 'delay': self.params['delay_ei_unspec'], 'receptor_type': self.params['syn_ports']['nmda']})
        assert (self.params['w_ei_unspec'] > 0), 'Excitatory weights need to be positive!'
        # inh - exc unspecific (within one hypercolumn) GABA_A
        nest.CopyModel('static_synapse', 'inh_exc_unspec', \
                {'weight': self.params['w_ie_unspec'], 'delay': self.params['delay_ie_unspec'], 'receptor_type': self.params['syn_ports']['gaba']})
        assert (self.params['w_ie_unspec'] < 0), 'Inhibitory weights need to be negative!'
        # inh - exc unspecific (within one hypercolumn) GABA_B
#        nest.CopyModel('static_synapse', 'inh_exc_unspec_slow', \
#                {'weight': self.params['w_ie_unspec'], 'receptor_type': self.params['syn_ports']['gaba']})
        # inh - inh unspecific (within one hypercolumn) GABA_A
        nest.CopyModel('static_synapse', 'inh_inh_unspec_fast', \
                {'weight': self.params['w_ii_unspec'], 'delay': self.params['delay_ii_unspec'], 'receptor_type': self.params['syn_ports']['gaba']})
        assert (self.params['w_ii_unspec'] < 0), 'Inhibitory weights need to be negative!'
        # inh - inh unspecific (within one hypercolumn) GABA_B
#        nest.CopyModel('static_synapse', 'inh_inh_unspec_slow', \
#                {'weight': self.params['w_ii_unspec'], 'receptor_type': self.params['syn_ports']['gaba']})
        # inh - exc global specific (between hypercolumns): GABA_A
        nest.CopyModel('static_synapse', 'inh_exc_specific_fast', \
                {'weight': self.params['w_ie_spec'], 'delay': self.params['delay_ie_spec'], 'receptor_type': self.params['syn_ports']['gaba']})
        assert (self.params['w_ie_spec'] < 0), 'Inhibitory weights need to be negative!'



        # inh - exc global specific (between hypercolumns): GABA_B
#        nest.CopyModel('static_synapse', 'inh_exc_specific_slow', \
#                {'weight': self.params['w_ie_spec'], 'receptor_type': self.params['syn_ports']['gaba']})

        # exc - exc local (within a minicolumn) AMPA
#        nest.CopyModel('static_synapse', 'exc_exc_local_fast', \
#                {'weight': self.params['w_ee_local'], 'delay': self.params['delay_ee_local'], 'receptor_type': self.params['syn_ports']['ampa']})
        # exc - exc local (within a minicolumn) NMDA
#        nest.CopyModel('static_synapse', 'exc_exc_local_slow', \
#                {'weight': self.params['w_ee_local'], 'delay': self.params['delay_ee_local'], 'receptor_type': self.params['syn_ports']['nmda']})

        # exc - exc global (between hypercolumns): AMPA
        if self.params['training_run']:
#            syn_params = deepcopy(self.params['bcpnn_params'])
            nest.SetDefaults('bcpnn_synapse', params=self.params['bcpnn_params'])
            nest.CopyModel('bcpnn_synapse', 'exc_exc_local_training', self.params['bcpnn_params'])
            nest.CopyModel('bcpnn_synapse', 'exc_exc_global_training', self.params['bcpnn_params'])
        else:
            # exc - exc fast: AMPA

            if self.params['with_stp']:
                nest.CopyModel('tsodyks_synapse', 'exc_exc_global_fast', \
                        {'delay': self.params['delay_ee_local'], 'receptor_type': self.params['syn_ports']['ampa'], 'tau_psc': self.params['tau_syn']['ampa']})
            else:
                nest.CopyModel('static_synapse', 'exc_exc_global_fast', \
                        {'delay': self.params['delay_ee_local'], 'receptor_type': self.params['syn_ports']['ampa']})
#                self.params['bcpnn_params']['gain'] = self.params['bcpnn_gain']
#                self.params['bcpnn_params']['K'] = 0.
#                nest.CopyModel('bcpnn_synapse', 'exc_exc_global_fast', \
#                        {'delay': self.params['delay_ee_local'], 'receptor_type': self.params['syn_ports']['ampa']})

            # exc - exc slow: AMPA
            if self.params['with_stp']:
                nest.CopyModel('tsodyks_synapse', 'exc_exc_global_slow', \
                        {'delay': self.params['delay_ee_local'], 'receptor_type': self.params['syn_ports']['nmda'], 'tau_psc': self.params['tau_syn']['nmda']})
            else:
                nest.CopyModel('static_synapse', 'exc_exc_global_slow', \
                        {'delay': self.params['delay_ee_local'], 'receptor_type': self.params['syn_ports']['nmda']})

            # exc - inh global specific (between hypercolumns): AMPA
            nest.CopyModel('static_synapse', 'exc_inh_specific_fast', \
                    {'delay': self.params['delay_ei_spec'], 'receptor_type': self.params['syn_ports']['ampa']})
            # exc - inh global specific (between hypercolumns): NMDA
            nest.CopyModel('static_synapse', 'exc_inh_specific_slow', \
                    {'delay': self.params['delay_ei_spec'], 'receptor_type': self.params['syn_ports']['nmda']})

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
            nest.SetStatus([gid], {'V_m': self.RNG.normal(self.params['v_init'], self.params['v_init_sigma'])})
            nest.SetStatus([gid], {'C_m': self.RNG.normal(self.params['C_m_mean'], self.params['C_m_sigma'])})


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
        column_to_gid = {'exc':{}, 'inh_unspec':{}, 'inh_spec': {}}
        gid_to_column = {'exc':{}, 'inh_unspec':{}, 'inh_spec': {}}
        gid_to_type = {}

        ##### EXC CELLS
        for hc in xrange(self.params['n_hc']):
            column_to_gid['exc'][hc] = {}
            for mc in xrange(self.params['n_mc_per_hc']):
                pop = nest.Create(self.params['neuron_model'], self.params['n_exc_per_mc'], params=self.params['cell_params_exc'])
                column_to_gid['exc'][hc][mc] = pop
                self.local_idx_exc += self.get_local_indices(pop)
                self.list_of_exc_pop[hc].append(pop)
                self.exc_gids += pop
                for i_, gid in enumerate(pop):
                    gid_to_column['exc'][gid] = (hc, mc)
                    gid_to_type[gid] = 'exc'

        local_gids_dict = {}
        local_gids_dict.update({gid : self.pc_id for gid in self.local_idx_exc})
        f = file(self.params['local_gids_fn_base'] + '%d.json' % (self.pc_id), 'w')
        json.dump(local_gids_dict, f, indent=2)

        if self.params['with_inhibitory_neurons']:
        ##### UNSPECIFIC INHIBITORY CELLS
            for hc in xrange(self.params['n_hc']):
                column_to_gid['inh_unspec'][hc] = {}
                pop = nest.Create(self.params['neuron_model'], self.params['n_inh_unspec_per_hc'], params=self.params['cell_params_inh'])
                self.list_of_unspecific_inh_pop.append(pop)
                self.local_idx_inh_unspec += self.get_local_indices(pop)
                self.inh_unspec_gids += pop
                column_to_gid['inh_unspec'][hc] = pop
                for i_, gid in enumerate(pop):
                    gid_to_column['inh_unspec'][gid] = (hc, 1) # mc index for all unspec inhibitory cells == 1
                    gid_to_type[gid] = 'inh_unspec'

        ##### SPECIFIC INHIBITORY CELLS
#        for hc in xrange(self.params['n_hc']):
#            column_to_gid['inh_spec'][hc] = {}
#            for mc in xrange(self.params['n_mc_per_hc']):
#                pop = nest.Create(self.params['neuron_model'], self.params['n_inh_per_mc'], params=self.params['cell_params_inh'])
#                column_to_gid['inh_spec'][hc][mc] = pop
#                self.list_of_specific_inh_pop[hc].append(pop)
#                self.local_idx_inh_spec += self.get_local_indices(pop)
#                self.inh_spec_gids += pop
#                for i_, gid in enumerate(pop):
#                    gid_to_column['inh_spec'][gid] = (hc, mc)
#                    gid_to_type[gid] = 'inh_spec'

        gids_dict = {}
        gids_dict['column_to_gid'] = column_to_gid
        gids_dict['gid_to_column'] = gid_to_column
        gids_dict['gid_to_type'] = gid_to_type
        if self.pc_id == 0:
            f_gids = file(self.params['gid_fn'], 'w')
            json.dump(gids_dict, f_gids, indent=2)
            f_gids.flush()
            f_gids.close()

        ####### Trigger spikes
        self.trigger_spike_source = nest.Create('spike_generator', 1)
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                nest.DivergentConnect(self.trigger_spike_source, self.list_of_exc_pop[hc][mc], model='trigger_synapse')

        self.background_noise_exc = nest.Create('poisson_generator', 1)
        self.background_noise_inh = nest.Create('poisson_generator', 1)
        nest.SetStatus(self.background_noise_exc, {'rate': self.params['f_noise_exc']})
        nest.SetStatus(self.background_noise_inh, {'rate': self.params['f_noise_inh']})
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                nest.DivergentConnect(self.background_noise_exc, self.list_of_exc_pop[hc][mc], model='noise_syn_exc')
                nest.DivergentConnect(self.background_noise_inh, self.list_of_exc_pop[hc][mc], model='noise_syn_inh')

        if self.comm != None:
            self.comm.Barrier()


        ##### RECORDER NEURONS
        self.setup_recorder_neurons()

        # set the cell parameters
        self.spike_times_container = [ np.array([]) for i in xrange(len(self.local_idx_exc))]

        # Record spikes 
        self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':False, 'label':'exc_spikes'})
        self.inh_spec_spike_recorder = nest.Create('spike_detector', params={'to_file':False, 'label':'inh_spec_spikes'})
        self.inh_unspec_spike_recorder = nest.Create('spike_detector', params={'to_file':False, 'label':'inh_unspec_spikes'})
#        self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
#        self.inh_spec_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'inh_spec_spikes'})
#        self.inh_unspec_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'inh_unspec_spikes'})
        self.spike_recorders = {}
        self.params['cell_types'] = ['exc', 'inh_spec', 'inh_unspec']
        self.spike_recorders['exc'] = self.exc_spike_recorder
        self.spike_recorders['inh_spec'] = self.inh_spec_spike_recorder
        self.spike_recorders['inh_unspec'] = self.inh_unspec_spike_recorder
        #self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
        #self.inh_spec_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'inh_spec_spikes'})
        #self.inh_unspec_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'inh_unspec_spikes'})
        for hc in xrange(self.params['n_hc']):
            for mc in xrange(self.params['n_mc_per_hc']):
                nest.ConvergentConnect(self.list_of_exc_pop[hc][mc], self.exc_spike_recorder)
                if not self.params['training_run'] and self.params['with_rsnp_cells']:
                    nest.ConvergentConnect(self.list_of_specific_inh_pop[hc][mc], self.inh_spec_spike_recorder)

        if self.params['with_inhibitory_neurons']:
            for hc in xrange(self.params['n_hc']):
                nest.ConvergentConnect(self.list_of_unspecific_inh_pop[hc], self.inh_unspec_spike_recorder)
    
        # v_init
        self.initialize_vmem(self.local_idx_exc)
        self.initialize_vmem(self.local_idx_inh_spec)
        self.initialize_vmem(self.local_idx_inh_unspec)

        if self.params['record_v']:
            self.record_v_exc()

        self.record_from = ['V_m', 'I_AMPA', 'I_NMDA', 'I_NMDA_NEG', 'I_AMPA_NEG', 'I_GABA']
#        self.record_from = ['V_m', 'g_AMPA', 'g_NMDA', 'g_NMDA_NEG', 'g_AMPA_NEG', 'g_GABA']
#        self.record_from = ['V_m']#, 'I_AMPA', 'I_NMDA', 'I_NMDA_NEG', 'I_AMPA_NEG', 'I_GABA']
        self.recorder_free_vmem = nest.Create('multimeter', params={'record_from': self.record_from, 'interval': self.params['dt_volt']})
        #self.recorder_free_vmem = nest.Create('multimeter', params={'record_from': ['V_m', 'I_AMPA', 'I_NMDA', 'I_NMDA_NEG', 'I_AMPA_NEG', 'I_GABA'], 'interval': self.params['dt_volt']})
        nest.SetStatus(self.recorder_free_vmem, [{"to_file": False, "withtime": True}])
        nest.DivergentConnect(self.recorder_free_vmem, self.recorder_neurons)

#        if self.params['training_run']:
#            self.create_training_input



    def create_input_for_recorder_neurons(self, stim_idx, mp_idx, with_blank=False, save_output=True):

        mapped_gids = np.array(self.recorder_neuron_gid_mapping.values()) - 1
        tp = self.tuning_prop_exc[mapped_gids, :]
        x0, v0 = self.motion_params[mp_idx, 0], self.motion_params[mp_idx, 2]
        print 'Computing input for stim_idx=%d' % stim_idx, 'mp:', x0, v0
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process
        idx_t_stop = np.int(self.stim_durations[stim_idx] / dt)
        L_input = np.zeros((self.params['n_recorder_neurons'], idx_t_stop))
        # compute the trajectory
        for i_time in xrange(idx_t_stop):
            time_ = (i_time * dt) / self.params['t_stimulus']
            x_stim = (x0 + time_ * v0)
            L_input[:, i_time] = self.params['f_max_stim'] * utils.get_input(self.tuning_prop_exc[mapped_gids, :], \
                    self.rf_sizes[mapped_gids, :], self.params, (x_stim, 0, v0, 0, 0))
        t_offset = self.stim_durations[:stim_idx].sum() #+ stim_idx * self.params['t_stim_pause']
#        print 't_offset:', t_offset
#        print 'self.stim_durations', self.stim_durations
#        print 'self.stim_durations[:%d]' % stim_idx, self.stim_durations[:stim_idx]

        if with_blank:
            start_blank = 1. / dt * (self.params['t_start_blank'])
            stop_blank = 1. / dt * (self.params['t_start_blank'] + self.params['t_blank'])
            blank_idx = np.arange(start_blank, stop_blank)
            before_stim_idx = np.arange(self.params['t_start'] * 1. / dt)
            blank_idx = np.concatenate((before_stim_idx, blank_idx))
            # blanking
            for i_time in blank_idx:
                L_input[:, i_time] = np.random.permutation(L_input[:, i_time])
#                L_input[:, i_time] = 0.

        L_input[:, -np.int(self.params['t_stim_pause'] / dt):] = 0
        print 'Proc %d creates input for stim %d' % (self.pc_id, stim_idx)
        for i_, tgt_gid_nest in enumerate(self.recorder_neuron_gid_mapping.keys()):
            rate_of_t = np.array(L_input[i_, :])
            # each cell will get its own spike train stored in the following file + cell gid
            n_steps = rate_of_t.size - int(self.params['t_stim_pause'] / dt)
            spike_times = []
            for i in xrange(n_steps):
                r = nprnd.rand()
                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    spike_times.append(i * dt + t_offset)
#                    print 'debug', i*dt + t_offset, t_offset

            if len(spike_times) > 0:
#                print 'DEBUGINPUT nspikes into cell %d (%d): %d' % (tgt_gid_nest, i_, len(spike_times)), ' tp : ', self.tuning_prop_exc[tgt_gid_nest-1, :], self.motion_params[mp_idx, :]
                nest.SetStatus([self.recorder_stimulus[i_]], {'spike_times' : np.around(np.sort(spike_times), decimals=1)})
                if save_output and self.pc_id == 0:
                    output_fn = self.params['recorder_neuron_input_rate_fn_base'] + '%d_%d.dat' % (tgt_gid_nest, stim_idx)
                    np.savetxt(output_fn, rate_of_t)
                    output_fn = self.params['recorder_neuron_input_fn_base'] + '%d_%d.dat' % (tgt_gid_nest, stim_idx)
                    print 'Saving output to:', output_fn
                    np.savetxt(output_fn, np.array(spike_times))
        if self.comm != None:
            self.comm.barrier()



    def create_input_for_stim(self, stim_idx, mp_idx, save_output=False, with_blank=False, my_units=None):

        if my_units == None:
            my_units = np.array(self.local_idx_exc) - 1
        x0, v0 = self.motion_params[mp_idx, 0], self.motion_params[mp_idx, 2]
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process
        idx_t_stop = np.int(self.stim_durations[stim_idx] / dt)
        L_input = np.zeros((len(self.local_idx_exc), idx_t_stop))

        # compute the trajectory
        for i_time in xrange(idx_t_stop):
            time_ = (i_time * dt) / self.params['t_stimulus']
            x_stim = (x0 + time_ * v0)
            L_input[:, i_time] = self.params['f_max_stim'] * utils.get_input(self.tuning_prop_exc[my_units, :], \
                    self.rf_sizes[my_units, :], self.params, (x_stim, 0, v0, 0, 0))
        t_offset = self.stim_durations[:stim_idx].sum() #+ stim_idx * self.params['t_stim_pause']
        print 't_offset:', t_offset
        print 'self.stim_durations', self.stim_durations
        print 'self.stim_durations[:%d]' % stim_idx, self.stim_durations[:stim_idx]

        if with_blank:
            start_blank = 1. / dt * self.params['t_start_blank']
            stop_blank = 1. / dt * (self.params['t_start_blank'] + self.params['t_blank'])
            blank_idx = np.arange(start_blank, stop_blank)
            before_stim_idx = np.arange(self.params['t_start'] * 1. / dt)
            blank_idx = np.concatenate((before_stim_idx, blank_idx))
            # blanking
            for i_time in blank_idx:
                L_input[:, i_time] = np.random.permutation(L_input[:, i_time])
                #L_input[:, i_time] = 0.

        L_input[:, -np.int(self.params['t_stim_pause'] / dt):] = 0
        print 'Proc %d creates input for stim %d' % (self.pc_id, stim_idx)
        for i_, tgt_gid_nest in enumerate(self.local_idx_exc):
            rate_of_t = np.array(L_input[i_, :])
            # each cell will get its own spike train stored in the following file + cell gid
            n_steps = rate_of_t.size - int(self.params['t_stim_pause'] / dt)
            spike_times = []
            for i in xrange(n_steps):
                r = nprnd.rand()
                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    spike_times.append(i * dt + t_offset)
#                    print 'debug', i*dt + t_offset, t_offset

            self.spike_times_container[i_] = np.array(spike_times)
            if len(spike_times) > 0:
#                print 'DEBUGINPUT nspikes into cell %d (%d): %d' % (tgt_gid_nest, i_, len(spike_times)), ' tp : ', self.tuning_prop_exc[tgt_gid_nest-1, :], self.motion_params[stim_idx, :]
                nest.SetStatus([self.stimulus[i_]], {'spike_times' : np.around(spike_times, decimals=1)})
                if save_output:
                    output_fn = self.params['input_rate_fn_base'] + '%d_%d.dat' % (tgt_gid_nest, stim_idx)
                    np.savetxt(output_fn, rate_of_t)
                    output_fn = self.params['input_st_fn_base'] + '%d_%d.dat' % (tgt_gid_nest, stim_idx)
                    print 'Saving output to:', output_fn
                    np.savetxt(output_fn, np.array(spike_times))


    def create_training_input(self, my_units=None, load_files=False, save_output=False, with_blank=False):

        if load_files:
            self.load_input()
            return True

        #else:
        if self.pc_id == 0:
            print "Computing input spiketrains..."

        if my_units == None:
            my_units = np.array(self.local_idx_exc) - 1
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process

        time = np.arange(0, self.params['t_sim'], dt)
        n_cells = len(my_units)
        L_input = np.zeros((n_cells, time.shape[0]))  # the envelope of the Poisson process
        n_stim_total = self.params['n_stim_training']

        self.trained_stimuli = []
        for i_stim, stim_idx in enumerate(range(self.params['stim_range'][0], self.params['stim_range'][1])):
            print 'Calculating OLD training input signal for %d cells in training stim %d / %d (%.1f percent)' % (n_cells, i_stim, n_stim_total, float(i_stim) / n_stim_total * 100.)
            x0, v0 = self.motion_params[stim_idx, 0], self.motion_params[stim_idx, 2]
            self.trained_stimuli.append((x0, .5, v0, .0, 0.))

#            self.compute_input(my_units)

            # get the input signal
            idx_t_start = np.int(self.stim_durations[:i_stim].sum() / dt)
            idx_t_stop = np.int(self.stim_durations[:i_stim+1].sum() / dt)
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
                start_blank = idx_t_start + 1. / dt * self.params['t_start_blank']
                stop_blank = idx_t_start + 1. / dt * (self.params['t_start_blank'] + self.params['t_blank'])
                blank_idx = np.arange(start_blank, stop_blank)
                before_stim_idx = np.arange(idx_t_start, self.params['t_start'] * 1./dt + idx_t_start)
#                    print 'debug stim before_stim_idx', i_stim, before_stim_idx, before_stim_idx.size
#                    print 'debug stim blank_idx', i_stim, blank_idx, blank_idx.size
                blank_idx = np.concatenate((blank_idx, before_stim_idx))
                # blanking
                for i_time in blank_idx:
                    L_input[:, i_time] = np.random.permutation(L_input[:, i_time])

            idx_t_start_pause = np.int((self.stim_durations[:i_stim+1].sum() - self.params['t_stim_pause'])/ dt)
            idx_t_stop_pause = np.int(self.stim_durations[:i_stim+1].sum() / dt)
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
                if len(spike_times) > 0:
                    output_fn = self.params['input_rate_fn_base'] + str(unit) + '.dat'
                    np.savetxt(output_fn, rate_of_t)
                    output_fn = self.params['input_st_fn_base'] + str(unit) + '.dat'
                    print 'Saving output to:', output_fn
                    np.savetxt(output_fn, np.array(spike_times))
        self.params['trained_stimuli'] = self.trained_stimuli
        np.savetxt(self.params['presented_stim_fn'], np.array(self.trained_stimuli))
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

        self.connect_input_to_exc()
        if self.params['training_run'] and not self.params['debug']:
            print 'Connecting exc - exc'
            self.connect_ee_sparse() # within MCs and bcpnn-all-to-all connections
            self.setup_recorder_neurons() # DOES NOT WORK
            if self.params['with_inhibitory_neurons']:
                print 'Connecting exc - inh unspecific'
                self.connect_ei_unspecific()
                print 'Connecting inh - exc unspecific'
                self.connect_ie_unspecific() # normalizing inhibition
        elif not self.params['debug']: # load the weight matrix
            print 'Connecting exc - inh unspecific'
            if self.params['with_inhibitory_neurons']:
                self.connect_ei_unspecific()
                print 'Connecting inh - exc unspecific'
                self.connect_ie_unspecific() # normalizing inhibition
                self.connect_ii() # connect unspecific and specific inhibition to excitatory cells
            if self.params['with_rsnp_cells']:
                print 'Connecting inh - exc specific'
                self.connect_ie_specific()
                print 'Connecting exc - inh specific'
                self.connect_ei_specific()

            # setup long-range connectivity based on trained connection matrix
            print 'Connecting recorder neurons'
            self.connect_recorder_neurons()
            print 'Connecting exc - exc '
            self.connect_ee_testing()



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

        for i_hc in xrange(self.params['n_hc']):
            nest.RandomConvergentConnect(self.list_of_unspecific_inh_pop[i_hc], self.list_of_unspecific_inh_pop[i_hc], \
                    self.params['n_conn_ii_per_hc'], weight=self.params['w_ii_unspec'], delay=1., model='inh_inh_unspec_fast', \
                    options={'allow_multapses':False, 'allow_autapses': False})


    def setup_recorder_neurons(self):
        self.recorder_neurons = nest.Create(self.params['neuron_model'], self.params['n_recorder_neurons'], params=self.params['cell_params_recorder_neurons'])
        self.recorder_stimulus = nest.Create('spike_generator', self.params['n_recorder_neurons'])
        for i_ in xrange(self.params['n_recorder_neurons']):
            nest.Connect([self.recorder_stimulus[i_]], [self.recorder_neurons[i_]], model='input_exc_fast')

        self.recorder_neuron_gid_mapping = {} # {recorder_neuron_gid : original_neuron_gid}
        self.recorder_neuron_mc_mapping = {} # {recorder_neuron_gid : original_neuron_gid}
        i_rec_nrn = 0
        for speed in self.params['record_tuning_prop_v']:
            pos = np.linspace(self.params['x_min_tp'], self.params['x_max_tp'], self.params['n_recorder_neurons_per_speed'], endpoint=True)
            for x_, i_ in enumerate(pos):
                gid = utils.get_gids_near_stim_nest([x_, .5, speed, .0], self.tuning_prop_exc)

                self.recorder_neuron_gid_mapping[self.recorder_neurons[i_rec_nrn]] = gid[0][0]
#                print 'DEBUG setup_recorder_neurons ', self.recorder_neurons[i_rec_nrn], self.recorder_neuron_gid_mapping[self.recorder_neurons[i_rec_nrn]], 'xpos v:', x_, speed
                i_rec_nrn += 1
                    


    def connect_input_to_recorder_neurons(self):
        """
        The recorder stimulus is inserted into recorder neurons which are not supposed to spike (high v_thresh).
        """
        self.recorder_stimulus = nest.Create('spike_generator', self.params['n_recorder_neurons'])
        for i_ in xrange(self.params['n_recorder_neurons']):
            nest.SetStatus([self.recorder_stimulus[i_]], {'spike_times' : np.sort(spike_times)})
            #nest.Connect([self.recorder_stimulus[i_]], [self.recorder_neurons[i_]], model='input_exc_slow')


        """
        # pick recorder neurons along the spatial axis --> sort the neurons according to their x-pos
        sorted_gids = np.argsort(self.tuning_prop_exc[:, 0])
        # pick evenly spaced indices
        rnd_ = np.linspace(0, self.params['n_exc'], self.params['n_recorder_neurons'], endpoint=False)
        rnd_gids = np.array(rnd_, dtype=int)
        # get the indices for the corresponding 'normal' neurons
        gids_normal = sorted_gids[rnd_gids]


        for i_, gid in enumerate(self.recorder_neurons):
            self.recorder_neuron_gid_mapping[self.recorder_neurons[i_]] = self.exc_gids[gids_normal[i_]] + 1 
#            print 'i_, gid recorder neuron, gid normal, exc_gids', i_, gid, gids_normal[i_], self.exc_gids[gids_normal[i_]]

        print 'DEBUG recorder_neuron_gid_mapping:', self.recorder_neuron_gid_mapping

        # 1) connect the same input to recorder neurons as to the normal neurons
        if self.params['training_run']:
            input_spikes_for_recorder_neurons = self.create_training_input_for_cells(gids_normal, with_blank=False)
#            self.compute_input(gids_normal, self.motion_params, save_output=self.save_output, with_blank=False)
        else:
            input_spikes_for_recorder_neurons = self.create_training_input_for_cells(gids_normal, with_blank=True)
#            input_spikes_for_recorder_neurons = self.compute_input(gids_normal, self.motion_params, save_output=self.save_output, with_blank=True)
        
            np.savetxt(self.params['recorder_neuron_input_fn_base'] + '%d.dat' % (self.recorder_neurons[i_]), spike_times)

            spike_times = input_spikes_for_recorder_neurons[i_]

        """


    def create_training_input_for_cells(self, stim_idx, gids, with_blank):
        n_cells = len(gids)
        spike_times_container = [ np.array([]) for i in xrange(len(gids))]
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process
        time = np.arange(0, self.params['t_sim'], dt)
        L_input = np.zeros((n_cells, time.shape[0]))  # the envelope of the Poisson process

        np.savetxt(self.params['training_stimuli_fn'], self.motion_params)
        n_stim_total = self.params['n_stim_training']
        for i_stim in xrange(n_stim_total):
            print 'Re-Calculating input signal for training stim %d / %d (%.1f percent)' % (i_stim, n_stim_total, float(i_stim) / n_stim_total * 100.)
            x0, v0 = self.motion_params[i_stim, 0], self.motion_params[i_stim, 2]

            # get the input signal
            idx_t_start = np.int(self.stim_durations[:i_stim].sum() / dt)
            idx_t_stop = np.int(self.stim_durations[:i_stim+1].sum() / dt)
            idx_within_stim = 0
            for i_time in xrange(idx_t_start, idx_t_stop):
                time_ = (idx_within_stim * dt) / self.params['t_stimulus']
                x_stim = (x0 + time_ * v0) % self.params['torus_width']
                L_input[:, i_time] = utils.get_input(self.tuning_prop_exc[gids, :], self.rf_sizes[gids, :], self.params, (x_stim, 0, v0, 0, 0))
#                L_input[:, i_time] = utils.get_input(tp, self.params, (x_stim, 0, v0, 0, 0))
                L_input[:, i_time] *= self.params['f_max_stim']
                if (i_time % 5000 == 0):
                    print "t: %.2f [ms]" % (time_ * self.params['t_stimulus'])
                idx_within_stim += 1
        
            if with_blank:
                start_blank = idx_t_start + 1. / dt * self.params['t_start_blank']
                stop_blank = idx_t_start + 1. / dt * (self.params['t_start_blank'] + self.params['t_blank'])
                blank_idx = np.arange(start_blank, stop_blank)
                before_stim_idx = np.arange(idx_t_start, self.params['t_start'] * 1./dt + idx_t_start)
                blank_idx = np.concatenate((blank_idx, before_stim_idx))
                # blanking
                for i_time in blank_idx:
                    L_input[:, i_time] = np.random.permutation(L_input[:, i_time])

            idx_t_start_pause = np.int(((i_stim + 1) * self.params['t_test_stim']  - self.params['t_stim_pause'])/ dt) 
            idx_t_stop_pause = np.int((i_stim + 1) * self.params['t_test_stim'] / dt) 
#            print 'Debug idx_t_start_pause', idx_t_start_pause
#            print 'Debug idx_t_stop_pause', idx_t_stop_pause
            L_input[:, idx_t_start_pause:idx_t_stop_pause] = 0.

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
                r = self.RNG.rand()
                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    spike_times.append(i * dt)
            spike_times_container[i_] = np.array(spike_times)
            output_fn = self.params['input_rate_fn_base'] + str(unit) + '.dat'
            np.savetxt(output_fn, rate_of_t)
            output_fn = self.params['input_st_fn_base'] + str(unit) + '.dat'
            np.savetxt(output_fn, np.array(spike_times))
        return spike_times_container



    def connect_ee_sparse(self):
        """
        Connect each minicolumn to all other minicolumns but only sparsely
        """
#        self.connect_ee_within_one_MC()
        if self.params['training_run']:
            self.connect_ee_training()


    def connect_ee_within_one_MC(self):
        # connect cells within one MC
#        for tgt_gid in self.local_idx_exc:
#            hc_idx, mc_idx, gid_min, gid_max = self.get_gids_to_mc(tgt_gid)
#            n_src = int(round(self.params['n_exc_per_mc'] * self.params['p_ee_local']))
#            source_gids = np.random.randint(gid_min, gid_max, n_src)
#            source_gids = np.unique(source_gids)
#            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local_fast')
#            nest.ConvergentConnect(source_gids.tolist(), [tgt_gid], model='exc_exc_local_slow')

        if self.params['training_run']:
            for hc in xrange(self.params['n_hc']):
                for mc in xrange(self.params['n_mc_per_hc']):
                    nest.RandomDivergentConnect(self.list_of_exc_pop[hc][mc], self.list_of_exc_pop[hc][mc], self.params['n_conn_ee_local_out_per_pyr'], \
                            options={'allow_autapses': False, 'allow_multapses': False}, \
                            model='exc_exc_local_training')
        else:
            raise NotImplementedError 

#                for i_ in xrange(self.params['n_conn_ee_local']):
#                    nest.RandomDivergentConnect([self.list_of_exc_pop[hc][mc][i_]], [self.list_of_exc_pop[hc][mc][(i_ + 1) % self.params['n_exc_per_mc']]], model='exc_exc_local_fast')
#                    nest.Connect([self.list_of_exc_pop[hc][mc][i_]], [self.list_of_exc_pop[hc][mc][(i_ + 1) % self.params['n_exc_per_mc']]], model='exc_exc_local_fast')
#                    nest.Connect([self.list_of_exc_pop[hc][mc][i_]], [self.list_of_exc_pop[hc][mc][(i_ + 1) % self.params['n_exc_per_mc']]], model='exc_exc_local_slow')


    def connect_ee_training(self):
        # setup a sparse all-to-all connectivity

        print 'Connect E-E for training...'
        for i_hc_src in xrange(self.params['n_hc']):
            print 'i_hc_src:', i_hc_src, 'pc_id:', self.pc_id
            for i_mc_src in xrange(self.params['n_mc_per_hc']):
                mc_idx_src  = i_hc_src * self.params['n_mc_per_hc'] + i_mc_src
                for i_hc_tgt in xrange(self.params['n_hc']):
                    for i_mc_tgt in xrange(self.params['n_mc_per_hc']):
                        nest.DivergentConnect(self.list_of_exc_pop[i_hc_src][i_mc_src], self.list_of_exc_pop[i_hc_tgt][i_mc_tgt], 
                                model='exc_exc_global_training')

#                        mc_idx_tgt  = i_hc_tgt * self.params['n_mc_per_hc'] + i_mc_tgt
#                        if mc_idx_src != mc_idx_tgt:
#                            nest.RandomDivergentConnect(self.list_of_exc_pop[i_hc_src][i_mc_src], self.list_of_exc_pop[i_hc_tgt][i_mc_tgt], 
#                                    self.params['n_conn_ee_global_out_per_pyr'], model='exc_exc_global_training', \
#                                    options={'allow_multapses': False})





    def connect_ee(self):

        initial_weight = 0.
        self.connect_ee_within_one_MC()
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


    def set_connection_matrices(self, conn_fn_ampa, conn_fn_nmda):

        print 'DEBUG, loading ampa weight matrix from:', conn_fn_ampa
        self.W_ampa = np.loadtxt(conn_fn_ampa)
        assert (self.W_ampa.shape[0] == self.params['n_mc'] and self.W_ampa.shape[1] == self.params['n_mc']), 'ERROR: provided ampa weight matrix has wrong dimension. Check simulation parameters!'
        print 'DEBUG, saving ampa weight matrix to:', self.params['conn_matrix_ampa_fn']
        np.savetxt(self.params['conn_matrix_ampa_fn'], self.W_ampa)


        print 'DEBUG, loading nmda weight matrix from:', conn_fn_nmda
        self.W_nmda = np.loadtxt(conn_fn_nmda)
        assert (self.W_nmda.shape[0] == self.params['n_mc'] and self.W_nmda.shape[1] == self.params['n_mc']), 'ERROR: provided nmda weight matrix has wrong dimension. Check simulation parameters!'
        print 'DEBUG, saving nmda weight matrix to:', self.params['conn_matrix_nmda_fn']
        np.savetxt(self.params['conn_matrix_nmda_fn'], self.W_nmda)

        if self.comm != None:
            self.comm.Barrier()


    def connect_ee_testing(self):

        if self.comm != None:
            self.comm.Barrier()

        w_ampa_max = np.max(self.W_ampa)
        w_ampa_min = np.min(self.W_ampa)
        w_nmda_max = np.max(self.W_nmda)
        w_nmda_min = np.min(self.W_nmda)

        if self.params['with_stp']:
            U_ampa = self.params['stp_params']['ampa']['U']
            U_nmda = self.params['stp_params']['nmda']['U']
        else:
            U_ampa = 1.
            U_nmda = 1.

        for src_hc in xrange(self.params['n_hc']):
            for src_mc in xrange(self.params['n_mc_per_hc']):
                print 'DEBUG connect_ee_testing: src_hc _mc', src_hc, src_mc
                src_pop = self.list_of_exc_pop[src_hc][src_mc]
                src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                for tgt_hc in xrange(self.params['n_hc']):
                    for tgt_mc in xrange(self.params['n_mc_per_hc']):
                        tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
                        tgt_pop_idx = tgt_hc * self.params['n_mc_per_hc'] + tgt_mc
                        w_ampa = self.W_ampa[src_pop_idx, tgt_pop_idx]
                        w_nmda = self.W_nmda[src_pop_idx, tgt_pop_idx]

                        if w_ampa < 0:
                            w_gaba_ = w_ampa * self.params['bcpnn_gain'] / self.params['tau_syn']['gaba']
                            nest.RandomConvergentConnect(src_pop, tgt_pop, n=self.params['n_conn_ee_global_out_per_pyr'],\
                                    weight=[w_gaba_], delay=[self.params['delay_ee_global']], \
                                    model='inh_exc_specific_fast', options={'allow_autapses': False, 'allow_multapses': False})
                        else:
                            w_ampa_ = w_ampa * self.params['bcpnn_gain'] / self.params['tau_syn']['ampa'] / U_ampa
                            nest.RandomConvergentConnect(src_pop, tgt_pop, n=self.params['n_conn_ee_global_out_per_pyr'],\
                                    weight=[w_ampa_], delay=[self.params['delay_ee_global']], \
                                    model='exc_exc_global_fast', options={'allow_autapses': False, 'allow_multapses': False})

                        w_nmda_ = w_nmda * self.params['bcpnn_gain'] / (self.params['ampa_nmda_ratio'] * self.params['tau_syn']['nmda'] * U_nmda)
                        nest.RandomConvergentConnect(src_pop, tgt_pop, n=self.params['n_conn_ee_global_out_per_pyr'],\
                                weight=[w_nmda_], delay=[self.params['delay_ee_global']], \
                                model='exc_exc_global_slow', options={'allow_autapses': False, 'allow_multapses': False})


    
    def load_training_weights(self):

        fn = self.training_params['merged_conn_list_ee']
        if not os.path.exists(fn):
            print 'Merging connection files...'
            utils.merge_connection_files(self.training_params, 'ee', iteration=None)

        # copy the merged connection file to the test folder
        os.system('cp %s %s' % (fn, self.params['connections_folder']))


    def transform_weight(self, w, w_bcpnn_min, w_bcpnn_max):
        if w > 0:
            w_ = w * self.params['w_ee_global_max'] / w_bcpnn_max
        elif w < 0:
            # CAUTION: if using di-synaptic inhibition via RSNP cells --> use:
            if self.params['with_rsnp_cells']:
                w_ = -1. * w * self.params['w_ei_spec'] / w_bcpnn_min
            else:
                w_ = w * self.params['w_ei_spec'] / w_bcpnn_min
        return w_



    def connect_ei_unspecific(self):
        for i_hc in xrange(self.params['n_hc']):
            for i_mc in xrange(self.params['n_mc_per_hc']):
                nest.RandomDivergentConnect(self.list_of_exc_pop[i_hc][i_mc], self.list_of_unspecific_inh_pop[i_hc], \
                        self.params['n_conn_ei_unspec_per_mc'], \
                        options={'allow_autapses': False, 'allow_multapses': False}, \
                        model='exc_inh_unspec_fast')


    def connect_ie_unspecific(self):
        for i_hc in xrange(self.params['n_hc']):
            for i_mc in xrange(self.params['n_mc_per_hc']):
                nest.RandomConvergentConnect(self.list_of_unspecific_inh_pop[i_hc], self.list_of_exc_pop[i_hc][i_mc], \
                        self.params['n_conn_ie_unspec_per_mc'], \
                        options={'allow_autapses': False, 'allow_multapses': False}, \
                        model='inh_exc_unspec')


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
            # get the cell from the list of populations
            mc_idx = (unit - 1) / self.params['n_exc_per_mc']
            hc_idx = (unit - 1) / n_per_hc
            mc_idx_in_hc = mc_idx - hc_idx * self.params['n_mc_per_hc']
            idx_in_pop = (unit - 1) - mc_idx * self.params['n_exc_per_mc']
            nest.Connect([self.stimulus[i_]], [self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_pop]], model='input_exc_fast')
            #nest.Connect([self.stimulus[i_]], [self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_pop]], model='input_exc_slow')


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


    def get_weights_static(self):

        print 'NetworkModel.get_weights_static ...'
        conn_txt_ee = ''
        conn_txt_ei = ''
        conn_txt_ie = ''
        conn_txt_ii = ''
        n_conns_ee = 0
        n_conns_ei = 0
        n_conns_ie = 0
        n_conns_ii = 0

        for i_hc in xrange(self.params['n_hc']):
            print 'DEBUG get_weights_static hc:', i_hc
            conns_ii = nest.GetConnections(self.list_of_unspecific_inh_pop[i_hc], self.list_of_unspecific_inh_pop[i_hc])
            if conns_ii != None:
                for i_, c in enumerate(conns_ii):
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    conn_txt_ii += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])
                    n_conns_ii += 1
            for i_mc in xrange(self.params['n_mc_per_hc']):
                print 'DEBUG get_weights_static hc:', i_hc
                conns_ei = nest.GetConnections(self.list_of_exc_pop[i_hc][i_mc], self.list_of_unspecific_inh_pop[i_hc])
                conns_ie = nest.GetConnections(self.list_of_unspecific_inh_pop[i_hc], self.list_of_exc_pop[i_hc][i_mc])
                if conns_ei != None:
                    for i_, c in enumerate(conns_ei):
                        cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                        conn_txt_ei += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])
                        n_conns_ei += 1
                if conns_ie != None:
                    for i_, c in enumerate(conns_ie):
                        cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                        conn_txt_ie += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])
                        n_conns_ie += 1

        if not self.params['training_run']:
            for i_hc_src in xrange(self.params['n_hc']):
                print 'DEBUG get_weights_static EE hc_src:', i_hc_src
                for i_mc_src in xrange(self.params['n_mc_per_hc']):
                    print 'DEBUG get_weights_static EE hc_src, mc_src:', i_hc_src, i_mc_src
                    for i_hc_tgt in xrange(self.params['n_hc']):
                        for i_mc_tgt in xrange(self.params['n_mc_per_hc']):
                            conns_ee = nest.GetConnections(self.list_of_exc_pop[i_hc_src][i_mc_src], self.list_of_exc_pop[i_hc_tgt][i_mc_tgt])
                            if conns_ee != None:
                                for i_, c in enumerate(conns_ee):
                                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                                    conn_txt_ee += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])
                                    n_conns_ee += 1

            print 'Proc %d holds %d E->E connections' % (self.pc_id, n_conns_ee)
            fn_out_ee = self.params['conn_list_ee_fn_base'] + 'debug_%d.txt' % (self.pc_id)
            print 'Writing E-E connections to:', fn_out_ee
            conn_f_ee = file(fn_out_ee, 'w')
            conn_f_ee.write(conn_txt_ee)
            conn_f_ee.flush()
            conn_f_ee.close()


        print 'Proc %d holds %d E->I connections' % (self.pc_id, n_conns_ei)
        fn_out_ei = self.params['conn_list_ei_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing E-I connections to:', fn_out_ei
        conn_f_ei = file(fn_out_ei, 'w')
        conn_f_ei.write(conn_txt_ei)
        conn_f_ei.flush()
        conn_f_ei.close()

        print 'Proc %d holds %d I->E connections' % (self.pc_id, n_conns_ie)
        fn_out_ie = self.params['conn_list_ie_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing I-E connections to:', fn_out_ie
        conn_f_ie = file(fn_out_ie, 'w')
        conn_f_ie.write(conn_txt_ie)
        conn_f_ie.flush()
        conn_f_ie.close()

        print 'Proc %d holds %d I->I connections' % (self.pc_id, n_conns_ii)
        fn_out_ii = self.params['conn_list_ii_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing E-I connections to:', fn_out_ii
        conn_f_ii = file(fn_out_ii, 'w')
        conn_f_ii.write(conn_txt_ii)
        conn_f_ii.flush()
        conn_f_ii.close()



    def get_weights_after_learning_cycle(self, iteration=None):
        """
        Saves the weights as adjacency lists (TARGET = Key) as dictionaries to file.
        """

        t_start = time.time()
        print 'NetworkModel.get_weights_after_learning_cycle ...'
        conn_txt = ''
        bias = {}

        my_units = self.local_idx_exc # !GIDs are 1-aligned!
        n_my_conns = 0

        for i_hc_src in xrange(self.params['n_hc']):
            for i_mc_src in xrange(self.params['n_mc_per_hc']):
                for i_hc_tgt in xrange(self.params['n_hc']):
                    for i_mc_tgt in xrange(self.params['n_mc_per_hc']):
                        conns = nest.GetConnections(self.list_of_exc_pop[i_hc_src][i_mc_src], self.list_of_exc_pop[i_hc_tgt][i_mc_tgt])
                        if conns != None:
                            for i_, c in enumerate(conns):
                                cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                                pi = cp[0]['p_i']
                                pj = cp[0]['p_j']
                                pij = cp[0]['p_ij']
                                w = np.log(pij / (pi * pj))
                                conn_txt += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij)
                                bias[cp[0]['target']] = cp[0]['bias']
                                n_my_conns += 1


#        for i_pre, gid_pre in enumerate(my_units):
#            for tgt_hc in xrange(self.params['n_hc']):
#                for tgt_mc in xrange(self.params['n_mc_per_hc']):
#                    tgt_pop = self.list_of_exc_pop[tgt_hc][tgt_mc]
#                    conns = nest.GetConnections([gid_pre], tgt_pop) # get the list of connections stored on the current MPI node
#                    if conns != None:
#                        for i_, c in enumerate(conns):
#                            cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
#                            pi = cp[0]['p_i']
#                            pj = cp[0]['p_j']
#                            pij = cp[0]['p_ij']
#                            w = np.log(pij / (pi * pj))
#                            conn_txt += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij)
#                            bias[cp[0]['target']] = cp[0]['bias']
#                            n_my_conns += 1

        print 'Proc %d holds %d connections' % (self.pc_id, n_my_conns)
        if iteration == None:
            fn_out = self.params['conn_list_ee_fn_base'] + '%d.txt' % (self.pc_id)
            fn_out_bias = self.params['bias_ee_fn_base'] + '%d.json' % (self.pc_id)
        else:
            fn_out = self.params['conn_list_ee_fn_base'] + 'it%04d_%d.txt' % (iteration, self.pc_id)
            fn_out_bias = self.params['bias_ee_fn_base'] + 'it%04d_%d.json' % (iteration, self.pc_id)
        print 'Saving connection list to: ', fn_out

        bias_f = file(fn_out_bias, 'w')
        json.dump(bias, bias_f, indent=0)
        print 'Writing E - E connections (%d) to:' % (n_my_conns) , fn_out
        conn_f = file(fn_out, 'w')
        conn_f.write(conn_txt)
        conn_f.flush()
        conn_f.close()



        """
        n_my_conns = 0
#        my_units = np.array(self.local_idx_exc)[:, 0] # !GIDs are 1-aligned!
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

        """



    def connect_recorder_neurons(self):
        """
        Must be called after the normal connect
        """

        if self.params['with_stp']:
            U_ampa = self.params['stp_params']['ampa']['U']
            U_nmda = self.params['stp_params']['nmda']['U']
        else:
            U_ampa = 1.
            U_nmda = 1.

        for rec_gid in self.recorder_neurons:
            hc_idx, mc_idx_in_hc, idx_in_mc = self.get_indices_for_gid(self.recorder_neuron_gid_mapping[rec_gid])
            tgt_pop_idx = hc_idx * self.params['n_mc_per_hc'] + mc_idx_in_hc
            for src_hc in xrange(self.params['n_hc']):
                for src_mc in xrange(self.params['n_mc_per_hc']):
                    src_pop_idx = src_hc * self.params['n_mc_per_hc'] + src_mc
                    w_ampa = self.W_ampa[src_pop_idx, tgt_pop_idx]
                    w_nmda = self.W_nmda[src_pop_idx, tgt_pop_idx]
                    src_pop = self.list_of_exc_pop[src_hc][src_mc]
                    if w_ampa < 0:
                        w_gaba_ = w_ampa * self.params['bcpnn_gain'] / self.params['tau_syn']['gaba']
                        nest.RandomConvergentConnect(src_pop, [rec_gid], n=self.params['n_conn_ee_global_out_per_pyr'],\
                                weight=[w_gaba_], delay=[self.params['delay_ee_global']], \
                                model='inh_exc_specific_fast', options={'allow_autapses': False, 'allow_multapses': False})
                    else:
                        w_ampa_ = w_ampa * self.params['bcpnn_gain'] / self.params['tau_syn']['ampa'] / U_ampa
                        nest.RandomConvergentConnect(src_pop, [rec_gid], n=self.params['n_conn_ee_global_out_per_pyr'],\
                                weight=[w_ampa_], delay=[self.params['delay_ee_global']], \
                                model='exc_exc_global_fast', options={'allow_autapses': False, 'allow_multapses': False})

                    w_nmda_ = w_nmda * self.params['bcpnn_gain'] / (self.params['ampa_nmda_ratio'] * self.params['tau_syn']['nmda'] * U_nmda)
                    nest.RandomConvergentConnect(src_pop, [rec_gid], n=self.params['n_conn_ee_global_out_per_pyr'],\
                            weight=[w_nmda_], delay=[self.params['delay_ee_global']], \
                            model='exc_exc_global_slow', options={'allow_autapses': False, 'allow_multapses': False})

        f = file(self.params['recorder_neurons_gid_mapping'], 'w')
        json.dump(self.recorder_neuron_gid_mapping, f, indent=2)



#        my_adj_list = {}
#        f = file(self.params['recorder_neurons_gid_mapping'], 'r')
#        gid_mapping = json.load(f)
#        for rec_nrn in self.recorder_neurons:
#            mirror_gid = gid_mapping[rec_nrn]
#            for src_hc in xrange(self.params['n_hc']):
#                for src_mc in xrange(self.params['n_mc_per_hc']):
#                    src_pop = self.list_of_exc_pop[src_hc][src_mc]
#                    conns = nest.GetConnections(src_pop, [mirror_gid]) # get the list of connections stored on the current MPI node
#                    print 'Debug conns', conns

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


              
    def run_sim(self):
        t_start = time.time()
        # # # # # # # # # # # # # #
        #     R U N N N I N G     #
        # # # # # # # # # # # # # #

        n_stim_total = self.params['n_stim']
        print 'Run sim for %d stim' % (n_stim_total)
        mp = []
        for i_stim, stim_idx in enumerate(range(self.params['stim_range'][0], self.params['stim_range'][1])):
            if self.pc_id == 0:
                print 'Calculating input signal for %d cells in training stim %d / %d (%.1f percent) mp:' % (len(self.local_idx_exc), i_stim, n_stim_total, float(i_stim) / n_stim_total * 100.), self.motion_params[stim_idx, :]
            #self.create_input_for_stim(i_stim, save_output=self.params['save_input'], with_blank=not self.params['training_run'])
            self.create_input_for_stim(i_stim, stim_idx, save_output=self.params['save_input'], with_blank=not self.params['training_run'])
            #self.create_input_for_recorder_neurons(i_stim, with_blank=not self.params['training_run'], save_output=self.params['save_input'])
            self.create_input_for_recorder_neurons(i_stim, stim_idx, with_blank=not self.params['training_run'], save_output=self.params['save_input'])
            sim_time = self.stim_durations[i_stim]
            if self.pc_id == 0:
                print "Running stimulus %d with tau_i=%d for %d milliseconds, t_sim_total = %d, mp:" % (i_stim, self.params['taui_bcpnn'], sim_time, self.params['t_sim']), self.motion_params[stim_idx, :]
            if self.comm != None:
                self.comm.Barrier()
            nest.Simulate(sim_time)
            if self.comm != None:
                self.comm.Barrier()
            mp.append(self.motion_params[stim_idx, :])

            if self.params['training_run'] and self.params['weight_tracking']:
                self.get_weights_after_learning_cycle(iteration=stim_idx)

        if not self.params['training_run']:
#            np.savetxt(self.params['test_sequence_fn'], np.array(mp))
            np.savetxt(self.params['test_sequence_fn'], self.motion_params)
        t_stop = time.time()
        t_diff = t_stop - t_start
        print "Simulation finished on proc %d after: %d [sec]" % (self.pc_id, t_diff)


    def record_v_exc(self):
        self.voltmeter_exc = nest.Create('multimeter', params={'record_from': self.record_from, 'interval':  self.params['dt_volt']})
        nest.SetStatus(self.voltmeter_exc,[{"to_file": False, "withtime": True, 'label' : 'exc_volt'}])

        for i_hc in xrange(self.params['n_hc']):
            for i_mc in xrange(self.params['n_mc_per_hc']):
                nest.ConvergentConnect(self.voltmeter_exc, [self.list_of_exc_pop[i_hc][i_mc][0]])


#        mp_r = np.array(self.params['motion_params']) #cells well tuned to the normal stimulus
#        selected_gids_r, pops = utils.select_well_tuned_cells_trajectory(self.tuning_prop_exc, \
#                mp_r, self.params, self.params['n_gids_to_record'] / 2, 1)
#        mp_l = mp_r.copy() #opposite direction
#        mp_l[2] *= (-1.) # cells well tuned to the normal stimulus
#        selected_gids_l, pops = utils.select_well_tuned_cells_trajectory(self.tuning_prop_exc, \
#                mp_l, self.params, self.params['n_gids_to_record'] / 2, 1)
#        print 'Recording cells close to mp_l', mp_l, '\nGIDS:', selected_gids_l
#        print 'Recording cells close to mp_r', mp_r, '\nGIDS:', selected_gids_r
#        if len(self.params['gids_to_record']) > 0:
#            gids_to_record = np.r_[self.params['gids_to_record'], selected_gids_r, selected_gids_l]
#        else:
#            gids_to_record = np.r_[selected_gids_r, selected_gids_l]
#        np.savetxt(self.params['gids_to_record_fn'], gids_to_record, fmt='%d')
#                        conns = nest.GetConnections(self.list_of_exc_pop[i_hc_src][i_mc_src], self.list_of_exc_pop[i_hc_tgt][i_mc_tgt])
        # TODO: why is there a conflict between record_v and recording spikes?
#        for gid in gids_to_record:
#            if gid in self.local_idx_exc:
#                hc_idx, mc_idx_in_hc, idx_in_mc = self.get_indices_for_gid(gid)
#                nest.ConvergentConnect(voltmeter, [self.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_mc]])


    def record_v_inh_unspec(self):
        voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' : self.params['dt_volt']})
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



    def trigger_spikes(self):
        """
        At the end of the training call this function to trigger spikes in all cells 
        in order to initiate a weight update
        """
        t_pause = 50.
        n_spikes = 4
        t_spikes = [np.around(self.params['t_sim'] + t_pause + i_ * self.params['dt_sim'], decimals=1) for i_ in xrange(n_spikes)]
        nest.SetStatus(self.trigger_spike_source, {'spike_times' : t_spikes})
        nest.Simulate(t_pause + 10.)


    def collect_spikes(self):
        for cell_type in self.params['cell_types']:
            if self.pc_id == 0:
                spike_recorder = self.spike_recorders[cell_type]
                sptimes = nest.GetStatus(spike_recorder)[0]['events']['times']
                gids = nest.GetStatus(spike_recorder)[0]['events']['senders']
                if self.comm != None:
                    for i_proc in xrange(1, self.n_proc):
                        sptimes = np.r_[sptimes, self.comm.recv(source=i_proc, tag=0)]
                        gids = np.r_[gids, self.comm.recv(source=i_proc, tag=1)]
                    fn = self.params['%s_spiketimes_fn_merged' % cell_type]
                    output = np.array((gids, sptimes))
                    np.savetxt(fn, output.transpose())
            else:
                spike_recorder = self.spike_recorders[cell_type]
                sptimes = nest.GetStatus(spike_recorder)[0]['events']['times']
                gids = nest.GetStatus(spike_recorder)[0]['events']['senders']
                self.comm.send(nest.GetStatus(spike_recorder)[0]['events']['times'],dest=0, tag=0)
                self.comm.send(nest.GetStatus(spike_recorder)[0]['events']['senders'],dest=0, tag=1)
            self.comm.barrier()
            

    def collect_vmem_data(self):
        print 'collect_vmem_data...', 
        output_fn_base = [self.params['free_vmem_fn_base'], 'exc']
        for i_, recorder in enumerate([self.recorder_free_vmem, self.voltmeter_exc]):
            for observable in self.record_from:
                if self.pc_id == 0:
                    output_vec = nest.GetStatus(recorder)[0]['events'][observable]
                    for i_proc in xrange(1, self.n_proc):
                        output_vec = np.r_[output_vec, self.comm.recv(source=i_proc)]
                else:
                    self.comm.send(nest.GetStatus(recorder)[0]['events'][observable],dest=0)
                if self.comm != None:
                    self.comm.barrier()

                if self.pc_id == 0:
                    gids = nest.GetStatus(recorder)[0]['events']['senders']
                    for i_proc in xrange(1, self.n_proc):
                        gids = np.r_[gids, self.comm.recv(source=i_proc)]
                    fn = self.params['volt_folder'] + output_fn_base[i_] + '_%s.dat' % (observable)
                    output = np.array((gids, output_vec))
                    np.savetxt(fn, output.transpose())
#                    print 'debug output_vec', output_vec
#                    print 'debug get status:', nest.GetStatus(recorder)
#                    print 'debug observable:', observable
                else:
                    self.comm.send(nest.GetStatus(recorder)[0]['events']['senders'],dest=0)

        print 'done'


    def update_bcpnn_params(self):
        self.params['taup_bcpnn'] = self.params['t_sim'] * self.params['ratio_tsim_taup']
        self.params['bcpnn_params']['tau_p'] = self.params['taup_bcpnn']
        epsilon = 1 / (self.params['fmax_bcpnn'] * self.params['taup_bcpnn'])
        self.params['bcpnn_params']['epsilon'] = epsilon

