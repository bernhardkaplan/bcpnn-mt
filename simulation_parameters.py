import json
import numpy as np
import numpy.random as rnd
import os
import utils

class parameter_storage(object):
    """
    This class contains the simulation parameters in a dictionary called params.
    """

    def __init__(self, params_fn=None):

        if params_fn == None:
            self.params = {}
            self.set_default_params()
            self.set_filenames()
        else:
            self.params = self.load_params_from_file(params_fn)


    def set_default_params(self):
        self.params['simulator'] = 'nest' 

        self.params['training_run'] = True# if false, it's a test run and you should run main_test.py
        self.params['Cluster'] = True
        self.params['sim_id'] = ''

        # ###################
        # HEXGRID PARAMETERS
        # ###################
        self.params['n_grid_dimensions'] = 1     # decide on the spatial layout of the network

        self.params['n_rf'] = 20
#        self.params['n_rf'] = 8
        if self.params['n_grid_dimensions'] == 2:
            self.params['n_rf_x'] = np.int(np.sqrt(self.params['n_rf'] * np.sqrt(3)))
            self.params['n_rf_y'] = np.int(np.sqrt(self.params['n_rf'])) 
            # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of n_rfdots?"
            self.params['n_theta'] = 1# resolution in velocity norm and direction
        else:
            self.params['n_rf_x'] = self.params['n_rf']
            self.params['n_rf_y'] = 1
            self.params['n_theta'] = 1

        self.params['frac_rf_x_fovea'] = 0.5 # this fraction of all n_rf_x cells will have constant (minimum) RF size
        self.params['n_rf_x_fovea'] = np.int(np.round(self.params['frac_rf_x_fovea'] * self.params['n_rf_x']))
        if self.params['n_rf_x_fovea'] % 2:
            self.params['n_rf_x_fovea'] += 1
        self.params['n_rf_x_log'] = self.params['n_rf_x'] - self.params['n_rf_x_fovea']
        assert (self.params['n_rf_x_log'] % 2 == 0), 'ERROR: please make sure that n_rf_x_log is an even number (as n_rf_x_fovea), so please change n_hc (=n_rf_x) or frac_rf_x_fovea'

        self.params['n_v'] = 10
        assert (self.params['n_v'] % 2 == 0), 'n_v must be an even number (for equal number of negative and positive speeds)'
        self.params['n_hc'] = self.params['n_rf_x'] * self.params['n_rf_y']
        self.params['n_mc_per_hc'] = self.params['n_v'] * self.params['n_theta']
        self.params['n_mc'] = self.params['n_hc'] * self.params['n_mc_per_hc']  # total number of minicolumns
        self.params['n_exc_per_mc'] = 8# must be an integer multiple of 4
        self.params['n_exc_per_hc'] = self.params['n_mc_per_hc'] * self.params['n_exc_per_mc']
        self.params['n_exc'] = self.params['n_mc'] * self.params['n_exc_per_mc']
        self.params['n_recorder_neurons'] = 40  # number of dummy neurons with v_thresh --> inf that act as 'electrodes'

        self.params['log_scale'] = 2.0 # base of the logarithmic tiling of particle_grid; linear if equal to one
        self.params['n_orientation'] = 1 # number of preferred orientations

        self.params['x_max_tp'] = 0.45 # [a.u.] minimal distance to the center  
        self.params['x_min_tp'] = 0.1  # [a.u.] all cells with abs(rf_x - .5) < x_min_tp are considered to be in the center and will have constant, minimum RF size (--> see n_rf_x_fovea)
        self.params['v_max_tp'] = 1.0   # [Hz] maximal velocity in visual space for tuning proprties (for each component), 1. means the whole visual field is traversed within 1 second
        self.params['v_min_tp'] = 0.02  # [a.u.] minimal velocity in visual space for tuning property distribution


        # receptive field size parameters
        # receptive field sizes are determined by their relative position (for x/y relative to .5, for u/v relative to 0.)
        # rf_size = rf_size_gradient * |relative_rf_pos| + min_rf_size
        # check for reference: Dow 1981 "Magnification Factor and Receptive Field Size in Foveal Striate Cortex of the Monkey"
        self.params['regular_tuning_prop'] = False
        self.params['rf_x_center_distance'] = 0.0001   
        self.params['xpos_hc_0'] = 0.05 # the position of the first HC index (distance from the 'border')

        self.params['rf_x_distribution_steepness'] = 0.4 # 'free' parameter determining the steep-ness of the exponential distribution for x-pos
        if self.params['regular_tuning_prop']:
            self.params['sigma_rf_pos'] = .001 # some variability in the position of RFs
            self.params['sigma_rf_speed'] = .001 # some variability in the speed of RFs
            self.params['sigma_rf_direction'] = .25 * 2 * np.pi # some variability in the direction of RFs
            self.params['sigma_rf_orientation'] = .1 * np.pi # some variability in the direction of RFs
            # regular tuning prop
            self.params['rf_size_x_gradient'] = .0  # receptive field size for x-pos increases with distance to .5
            self.params['rf_size_y_gradient'] = .0  # receptive field size for y-pos increases with distance to .5
            self.params['rf_size_x_min'] = 1. / self.params['n_rf_x']
            self.params['rf_size_y_min'] = 1. / self.params['n_rf_y']
            self.params['rf_size_vx_gradient'] = .0 # receptive field size for vx-pos increases with distance to 0.0
            self.params['rf_size_vy_gradient'] = .0 #
#            self.params['rf_size_vx_min'] = self.params['v_max_tp'] / self.params['n_v']
#            self.params['rf_size_vy_min'] = self.params['v_max_tp'] / self.params['n_v']
            # when negative speeds are allowed:
            self.params['rf_size_vx_min'] = 2 * self.params['v_max_tp'] / self.params['n_v']
            self.params['rf_size_vy_min'] = 2 * self.params['v_max_tp'] / self.params['n_v']
        else:
            self.params['sigma_rf_pos'] = 0.05 #.02 # some variability in the position of RFs
            self.params['sigma_rf_speed'] = 0.10 #.03 # some variability in the speed of RFs
            self.params['sigma_rf_direction'] = .25 * 2 * np.pi # some variability in the direction of RFs
            self.params['sigma_rf_orientation'] = .1 * np.pi # some variability in the direction of RFs
    #        self.params['rf_size_x_gradient'] = .2  # receptive field size for x-pos increases with distance to .5
    #        self.params['rf_size_y_gradient'] = .2  # receptive field size for y-pos increases with distance to .5
    #        self.params['rf_size_x_min'] = .01      # cells situated at .5 have this receptive field size
    #        self.params['rf_size_y_min'] = .01      # cells situated at .5 have this receptive field size
            self.params['rf_size_vx_gradient'] = .5 # receptive field size for vx-pos increases with distance to 0.0
            self.params['rf_size_vy_gradient'] = .1 #
            self.params['rf_size_vx_min'] = .01
            self.params['rf_size_vy_min'] = .05 
            # regular tuning prop
            self.params['rf_size_x_gradient'] = .4  # receptive field size for x-pos increases with distance to .5
            self.params['rf_size_y_gradient'] = .0  # receptive field size for y-pos increases with distance to .5
            self.params['rf_size_x_min'] = 0.01 #1. / self.params['n_rf_x']
            self.params['rf_size_y_min'] = 1. / self.params['n_rf_y']
    #        self.params['rf_size_vx_gradient'] = .0 # receptive field size for vx-pos increases with distance to 0.0
    #        self.params['rf_size_vy_gradient'] = .0 #
    #        self.params['rf_size_vx_min'] = 2 * self.params['v_max_tp'] / self.params['n_v']
    #        self.params['rf_size_vy_min'] = 2 * self.params['v_max_tp'] / self.params['n_v']



        # ###################
        # NETWORK PARAMETERS
        # ###################
        self.params['fraction_inh_cells'] = 0.25 # fraction of inhibitory cells in the network, only approximately!
        # neuron numbers: based on n_mc
        self.params['n_inh_unspec'] = int(round(self.params['fraction_inh_cells'] * self.params['n_exc'])) # normalizing inhibition on HC level, based on the assumption that n_inh_unspec = n_inh_spec
        self.params['n_inh_unspec_per_hc'] = int(round(self.params['n_inh_unspec'] / self.params['n_hc']))
        self.params['n_inh_spec'] =  self.params['n_inh_unspec'] # local inhibition
        self.params['n_inh_per_mc'] = int(round(self.params['n_inh_spec'] / float(self.params['n_mc']))) # specific local inhibition
        self.params['n_inh' ] = self.params['n_inh_unspec'] + self.params['n_inh_spec']
        self.params['n_theta_inh'] = self.params['n_theta']
        self.params['n_v_inh'] = self.params['n_v']
        self.params['n_rf_inh'] = int(round(self.params['fraction_inh_cells'] * self.params['n_rf']))
        self.params['n_rf_x_inh'] = np.int(np.sqrt(self.params['n_rf_inh'] * np.sqrt(3))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of n_rf dots?"

        # cell gid offsets --> take care that cells are created in that order!
        self.params['exc_offset'] = 0
        self.params['inh_unspec_offset'] = self.params['n_exc']
        self.params['inh_spec_offset'] = self.params['n_exc'] + self.params['n_inh_unspec']

        self.params['n_cells'] = self.params['n_exc'] + self.params['n_inh']
        print '\nModular network structure:'
        print '\tn_hc: %d\tn_mc_per_hc: %d\tn_mc: %d\tn_exc_per_mc: %d' % (self.params['n_hc'], self.params['n_mc_per_hc'], self.params['n_mc'], self.params['n_exc_per_mc'])
        print '\tn_exc_per_mc: %d\tn_inh_per_mc: %d' % (self.params['n_exc_per_mc'], self.params['n_inh_per_mc'])
        print '\tn_exc_per_hc: %d\tn_inh_per_mc specific: %d\tn_inh_unspec_per_hc %d' % \
                (self.params['n_exc_per_mc'] * self.params['n_mc_per_hc'], self.params['n_inh_per_mc'], self.params['n_inh_unspec_per_hc'])
        print 'n_cells: %d\tn_exc: %d\tn_inh: %d\nn_inh / n_exc = %.3f\tn_inh / n_cells = %.3f' \
                % (self.params['n_cells'], self.params['n_exc'], self.params['n_inh'], \
                self.params['n_inh'] / float(self.params['n_exc']), self.params['n_inh'] / float(self.params['n_cells']))
        self.params['cell_types'] = ['exc', 'inh_spec', 'inh_unspec']

        # ###################
        # CELL PARAMETERS   #
        # ###################
        self.params['use_pynest'] = True
        # receptor types: 0 -- AMPA (3 ms), 1 -- NMDA (100 ms), 2 -- GABA_A (5 ms), 3 -- GABA_B (50 ms)
        if self.params['use_pynest']:
            self.params['neuron_model'] = 'aeif_cond_exp_multisynapse'
#            self.params['neuron_model'] = 'iaf_psc_exp_multisynapse'
#            self.params['neuron_model'] = 'iaf_psc_alpha_multisynapse'
            self.params['g_leak'] = 25. # before it was 16.66667
            self.params['cell_params_exc'] = {'C_m': 250.0, 'E_L': -70.0, 'I_e': 0.0, 'V_m': -70.0, \
                    'V_reset': -70.0, 'V_th': -55.0, 't_ref': 2.0, \
                    'a': 0., 'b': 0., \
                    'g_L': self.params['g_leak']}

            self.params['cell_params_inh'] = self.params['cell_params_exc']

#                    'n_synapses': 3, 'tau_syn': [3., 100., 15.], 'receptor_types': [0, 1, 2]}
#            self.params['cell_params_inh'] = {'C_m': 250.0, 'E_L': -70.0, 'I_e': 0.0, 'V_m': -70.0, \
#                    'V_reset': -70.0, 'V_th': -55.0, 't_ref': 2.0, 'tau_m': 10.0, \
#                    'tau_minus': 20.0, 'tau_minus_triplet': 110.0, \
#                    'g_L': self.params['g_leak'],
#                    'n_synapses': 3, 'tau_syn': [3., 100., 15.], 'receptor_types': [0, 1, 2]}

            self.params['cell_params_recorder_neurons'] = self.params['cell_params_exc'].copy()
            self.params['cell_params_recorder_neurons']['V_th'] = 1000. # these neurons should not spike, but only record the 'free membrane potential'
            self.params['cell_params_recorder_neurons']['V_peak'] = 1001. # these neurons should not spike, but only record the 'free membrane potential'

            self.params['v_init'] = self.params['cell_params_exc']['V_m'] + .5 * (self.params['cell_params_exc']['V_th'] - self.params['cell_params_exc']['V_m'])
            self.params['v_init_sigma'] = .2 * (self.params['cell_params_exc']['V_th'] - self.params['cell_params_exc']['V_m'])
            self.params['C_m_mean'] = self.params['cell_params_exc']['C_m']
            self.params['C_m_sigma'] = .1 * self.params['C_m_mean']

        else:
            self.params['tau_syn_exc'] = 5.0 # 10.
            self.params['tau_syn_inh'] = 10.0 # 20.
            if self.params['neuron_model'] == 'IF_cond_exp':
                self.params['cell_params_exc'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E': self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70}
                self.params['cell_params_inh'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E': self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70}
            elif self.params['neuron_model'] == 'IF_cond_alpha':
                self.params['cell_params_exc'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E': self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70}
                self.params['cell_params_inh'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E': self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70}
            elif self.params['neuron_model'] == 'EIF_cond_exp_isfa_ista':
                self.params['cell_params_exc'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E':self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70., \
                        'b' : 0.5, 'a':4.}
                self.params['cell_params_inh'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E':self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70., \
                        'b' : 0.5, 'a':4.}
            self.params['v_init'] = self.params['cell_params_exc']['v_rest'] + .5 * (self.params['cell_params_exc']['v_thresh'] - self.params['cell_params_exc']['v_rest'])
            self.params['v_init_sigma'] = .2 * (self.params['cell_params_exc']['v_thresh'] - self.params['cell_params_exc']['v_rest'])
            
        # #######################
        # CONNECTIVITY PARAMETERS
        # #######################

        # exc - exc: local
        self.params['p_ee_local'] = .25
        self.params['n_conn_ee_local_out_per_pyr'] = np.int(np.round(self.params['p_ee_local'] * self.params['n_exc_per_mc']))
        self.params['w_ee_local'] = 0.      # [nS]
        self.params['delay_ee_local'] = 1.  # [ms]

        # exc - exc: global
        self.params['p_ee_global'] = .3
        self.params['w_ee_global_max'] = 4.
        self.params['delay_ee_global'] = 2. # [ms]
        self.params['n_conn_ee_global_out_per_pyr'] = np.int(np.round(self.params['p_ee_global'] * self.params['n_exc_per_mc']))

        # exc - inh: spec
        self.params['delay_ei_spec'] = 2.   # [ms]

        # exc - inh: unspecific (targeting the basket cells within one hypercolumn)
        self.params['w_ei_unspec'] = 2.    # untrained, unspecific PYR -> Basket connections
        self.params['p_ei_unspec'] = .75     # probability for PYR -> Basket connections
        self.params['delay_ei_unspec'] = 1.
        self.params['n_conn_ei_unspec_per_mc'] = np.int(np.round(self.params['n_inh_unspec_per_hc'] * self.params['p_ei_unspec']))

        # inh - exc: unspecific inhibitory feedback within one hypercolumn
        self.params['w_ie_unspec'] = -10.  # untrained, unspecific Basket -> PYR connections
        self.params['p_ie_unspec'] = .75     # probability for Basket -> PYR Basket connections
        self.params['delay_ie_unspec'] = 1.
        self.params['n_conn_ie_unspec_per_mc'] = np.int(np.round(self.params['p_ie_unspec'] * self.params['n_exc_per_mc']))

        # ie_spec effective only after training
        self.params['w_ie_spec'] = -50.     # RSNP -> PYR, effective only after training
        self.params['p_ie_spec'] = 1.       # RSNP -> PYR
        self.params['delay_ie_spec'] = 1.

        # inh - inh
        self.params['w_ii_unspec'] = 1. # untrained, unspecific Basket -> PYR connections
        self.params['p_ii_unspec'] = .7 # probability for Basket -> PYR Basket connections
        self.params['delay_ii_unspec'] = 1.

        # approximately the same as in Mikael Lundqvist's work / copied from the olfaction project
#        self.params['p_rsnp_pyr'] = 0.7
#        self.params['p_pyr_pyr_local'] = 0.25
#        self.params['p_pyr_pyr_global'] = 0.3 # only relevant when 'orthogonal' patterns are studied 
#        self.params['p_pyr_basket'] = 0.7
#        self.params['p_basket_pyr'] = 0.7
#        self.params['p_pyr_rsnp'] = 0.3
#        self.params['p_basket_basket'] = 0.7




        # ###############
        # MOTION STIMULUS
        # ###############
        """
        x0 (y0) : start position on x-axis (y-axis)
        u0 (v0) : velocity in x-direction (y-direction)
        """
        self.params['anticipatory_mode'] = True # if True record selected cells to gids_to_record_fn
        self.params['motion_params'] = [.0, .5 , .5, 0, np.pi/6.0] # (x, y, v_x, v_y, orientation of bar)
        # the 'motion_params' are those that determine the stimulus
        # motion_params are used to create the test stimulus if training_run == False
        self.params['mp_select_cells'] = [.7, .5, .5, .0, np.pi / 6.0] # <-- those parameters determine from which cells v_mem should be recorded from
        self.params['motion_type'] = 'dot' # should be either 'bar' or 'dot'
        
        assert (self.params['motion_type'] == 'bar' or self.params['motion_type'] == 'dot'), 'Wrong motion type'

        self.params['blur_X'], self.params['blur_V'] = .10, .10
        self.params['blur_theta'] = 1.0
        self.params['torus_width'] = 1.
        self.params['torus_height'] = 1.
        # the blur parameter represents the input selectivity:
        # high blur means many cells respond to the stimulus
        # low blur means high input selectivity, few cells respond
        # the maximum number of spikes as response to the input alone is not much affected by the blur parameter

        # #####################
        # TRAINING PARAMETERS
        # #####################
        self.params['v_max_training'] = self.params['v_max_tp']# * .9
        self.params['v_min_training'] = self.params['v_min_tp']
        self.params['x_max_training'] = 0.9
        self.params['x_min_training'] = 0.1
        self.params['training_stim_noise_v'] = 0.05 # percentage of noise for each individual training speed
        self.params['training_stim_noise_x'] = 0.05 # percentage of noise for each individual training speed
        self.params['n_training_cycles'] = 1 # one cycle comprises training of all n_training_v
#        self.params['n_training_v'] = 3 # self.params['n_v'] # how many different speeds are trained per cycle

        self.params['n_training_v'] = 100 # how many different speeds are trained per cycle
        #self.params['n_training_v'] = self.params['n_v'] # how many different speeds are trained per cycle
        assert (self.params['n_training_v'] % 2 == 0), 'n_training_v should be an even number (for equal number of negative and positive speeds)'
        self.params['n_training_x'] = 1 # number of different starting positions per trained  speed

        self.params['n_theta_training'] = self.params['n_theta']
        self.params['n_training_stim_per_cycle'] = self.params['n_training_v'] * self.params['n_theta_training'] * self.params['n_training_x']

        # if one speed is trained, it is presented starting from this number on different locations
        # for 1-D this is irrelevant and can be set to 1
        self.params['n_stim_per_direction'] = 1 
#        self.params['n_stim_training'] = self.params['n_theta_training'] * self.params['n_training_cycles'] * self.params['n_training_v'] * self.params['n_stim_per_direction']
        self.params['n_stim_training'] = self.params['n_theta_training'] * self.params['n_training_cycles'] * self.params['n_training_v'] * self.params['n_stim_per_direction'] * self.params['n_training_x']
        self.params['random_training_order'] = True   # if true, stimuli within a cycle get shuffled
        self.params['sigma_theta_training'] = .05 # how much each stimulus belonging to one training direction is randomly rotated

#        self.params['test_stim_range'] = (0, self.params['n_stim_training'])
#        self.params['test_stim_range'] = (0, self.params['n_training_v'])
        self.params['test_stim_range'] = (0, 1)
        self.params['n_test_stim'] = self.params['test_stim_range'][1] - self.params['test_stim_range'][0]
        if self.params['training_run']:
            self.params['n_stim'] = self.params['n_stim_training']
        else:
            self.params['n_stim'] = self.params['n_test_stim']

        training_stim_offset = 0
        self.params['stim_range'] = [training_stim_offset, training_stim_offset + self.params['n_stim']] # naming the training folder, but params['stim_range'] will be overwritten 
        # stim_range indicates which stimuli have been presented to the network, i.e. the row index in the training_stimuli file
        self.params['trained_stimuli'] = None # contains only the motion parameters from those stimuli that actually have been presented
        self.params['frac_training_samples_from_grid'] = .8
        self.params['frac_training_samples_center'] = .0 # fraction of training samples drawn from the center
        self.params['center_stim_width'] = .0 # width from which the center training samples are drawn OR if reward_based_learning: stimuli positions are sampled from .5 +- center_stim_width
        assert (1.0 >= self.params['frac_training_samples_center'] + self.params['frac_training_samples_from_grid'])
        # to generate the training samples, three methods are used: 1) sampling from the tuning properties, 2) sampling from a grid  3) sampling nearby the center (as these stimuli occur more frequently)
        # then the frac_training_samples_from_grid determines how many training stimuli are taken from the grid sample

#        self.params['n_test_stim'] = self.params['n_training_v'] # number of training stimuli to be presented during testing
#        self.params['n_test_stim'] = 1


        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['seed'] = 12345 # the master seed
        # Master seeds for for independent experiments must differ by at least 2Nvp + 1. 
        # Otherwise, the same sequence(s) would enter in several experiments.
        self.params['visual_stim_seed'] = 123
        self.params['np_random_seed'] = 0
        self.params['tp_seed'] = 666
        self.params['t_training_max'] = 3000. # [ms]
        if self.params['training_run']:
            self.params['t_stim_pause'] = 500.
        else:
            self.params['t_stim_pause'] = 500.
        # a test stim is presented for t_test_stim - t_stim_pause
        self.params['t_test_stim'] = 2000. + self.params['t_stim_pause']

        # [ms] total simulation time -- will be overwritten depending on how long a stimulus will be presented 
        # if a stimulus leaves the visual field, the simulation is ended earlier for this stimulus, and takes maximally t_training_max per stimulus
        if self.params['training_run']:
            self.params['t_sim'] = self.params['n_stim_training'] * (self.params['t_training_max'] + self.params['t_stim_pause']) # will be overwritten
        else:
            self.params['t_sim'] = self.params['n_test_stim'] * self.params['t_test_stim']
        self.params['t_stimulus'] = 1000.       # [ms] time for a stimulus of speed 1.0 to cross the whole visual field from 0 to 1.
        self.params['t_start'] = 0.           # [ms] blank time before stimulus appears
        if self.params['training_run']:
            self.params['t_blank'] = 0.           # [ms] time for 'blanked' input
        else:
            self.params['t_blank'] = 200
        self.params['t_before_blank'] = self.params['t_start'] + 600.               # [ms] time when stimulus reappears, i.e. t_reappear = t_stimulus + t_blank
        self.params['tuning_prop_seed'] = 0     # seed for randomized tuning properties
        self.params['input_spikes_seed'] = 0
        self.params['delay_range'] = (0.1, 10.) # allowed range of delays
        self.params['dt_sim'] = self.params['delay_range'][0] * 1 # [ms] time step for simulation
        self.params['dt_rate'] = .1             # [ms] time step for the non-homogenous Poisson process
        self.params['n_gids_to_record'] = 20    # number to be sampled across some trajectory
        self.params['gids_to_record'] = []#181, 185]  # additional gids to be recorded 
        
        
        # ########################
        # BCPNN SYNAPSE PARAMETERS
        # ########################
        self.params['fmax_bcpnn'] = 200.0   # should be as the maximum output rate (with inhibitory feedback)
        self.params['taup_bcpnn'] = self.params['t_sim']# / 2.
        self.params['taui_bcpnn'] = 50.
        epsilon = 1 / (self.params['fmax_bcpnn'] * self.params['taup_bcpnn'])
        #self.params['bcpnn_init_val'] = epsilon
        self.params['bcpnn_init_val'] = 0.0001
        #self.params['bcpnn_init_val'] = 0.1

        if self.params['training_run']:
            self.params['gain'] = 0.
            self.params['kappa'] = 1.
        else:
            self.params['gain'] = 10.
            self.params['kappa'] = 0.

        self.params['bcpnn_params'] =  {
                'gain': 0.0, \
                'K': self.params['kappa'], \
                'fmax': self.params['fmax_bcpnn'],\
                'delay': 1.0, \
                'tau_i': self.params['taui_bcpnn'], \
                'tau_j': 10.,\
                'tau_e': 10.,\
                'tau_p': self.params['taup_bcpnn'],\
                'epsilon': epsilon, \
                'p_i': self.params['bcpnn_init_val'], \
                'p_j': self.params['bcpnn_init_val'], \
                'p_ij': self.params['bcpnn_init_val']**2, \
                'weight': 0.0,  \
                'receptor_type': 1
                }
        # gain is set to zero in order to have no plasiticity effects while training
        # K: learning rate (how strong the p-traces get updated)

        # transformation parameter for v_x --> tau_zi
        # for a linear transformation
        self.params['tau_zi_max'] = 500.
        self.params['tau_zi_min'] = 50.
        self.params['tau_vx_transformation_mode'] = 'linear'


        # ######
        # INPUT
        # ######
        self.params['f_max_stim'] = 200.       # [Hz]
        self.params['w_input_exc'] = 1. # [nS] mean value for input stimulus ---< exc_units (columns
        # needs to be changed if PyNN is used
        if not self.params['use_pynest']:
            self.params['w_input_exc'] /= 1000. # [uS] --> [nS] Nest expects nS
        self.params['w_trigger'] = 30.

        # ######
        # NOISE
        # ######
        #self.params['w_exc_noise'] = 2. # [nS] mean value for noise ---< columns
        #self.params['f_exc_noise'] = 1000# [Hz] 
        #self.params['w_inh_noise'] = 2. # [nS] mean value for noise ---< columns
        #self.params['f_inh_noise'] = 1000# [Hz]

        # no noise:
        self.params['w_exc_noise'] = 1e-5          # [uS] mean value for noise ---< columns
        self.params['f_exc_noise'] = 1# [Hz]
        self.params['w_inh_noise'] = 1e-5          # [uS] mean value for noise ---< columns
        self.params['f_inh_noise'] = 1# [Hz]



    def set_folder_name(self, folder_name=None):

        if folder_name == None:
            if self.params['training_run']:
#                folder_name = 'TrainingSim_tauzimin%d_max%d' % (self.params['tau_zi_min'], self.params['tau_zi_max'])
                folder_name = 'TrainingSim_%s_%dx%d_%d-%d_taui%d_nHC%d_nMC%d_blurXV_%.2f_%.2f_pi%.1e' % ( \
                        self.params['sim_id'], self.params['n_training_cycles'], self.params['n_training_v'], \
                        self.params['stim_range'][0], self.params['stim_range'][1], \
                        self.params['bcpnn_params']['tau_i'], \
                        self.params['n_hc'], self.params['n_mc_per_hc'], self.params['blur_X'], self.params['blur_V'], \
                        self.params['bcpnn_init_val'])
            else:
                folder_name = 'TestSim_%s_%d_taui%d_nHC%d_nMC%d_nExcPerMc%d_wee%.2f_wei%.2f' % ( \
                        self.params['sim_id'], self.params['n_test_stim'], 
                        self.params['bcpnn_params']['tau_i'], \
                        self.params['n_hc'], self.params['n_mc_per_hc'], self.params['n_exc_per_mc'], self.params['w_ee_global_max'], \
                        self.params['w_ei_global_max'])
            folder_name += '/'
            self.params['folder_name'] = folder_name
        else:
            self.params['folder_name'] = folder_name


    def set_filenames(self, folder_name=None):

        self.set_folder_name(folder_name)
        print 'Folder name:', self.params['folder_name']

        # in order to NOT re-compute the input spike trains when the stimulus parameters have not changed, do NOT store them in a subfolder of self.params['folder_name']
#        self.params['input_folder'] = "Debug_InputSpikeTrains_bX%.2e_bV%.2e_fstim%.1e_tsim%d_tblank%d_tbeforeblank%d_%dnrns/" % \
#                (self.params['blur_X'], self.params['blur_V'], self.params['f_max_stim'], self.params['t_sim'], self.params['t_blank'], self.params['t_before_blank'], self.params['n_cells'])
#        self.params['input_folder'] = "TwoCellInputSpikeTrains/" # folder containing the input spike trains for the network generated from a certain stimulus

#        self.params['input_folder'] = "%sInputFiles/" % self.params['folder_name'] # folder containing the input spike trains for the network generated from a certain stimulus
        if self.params['training_run']:
            self.params['input_folder'] = "InputFilesTraining_seed%d_nX%d_nV%d_stimRange%d-%d/" % (self.params['visual_stim_seed'], self.params['n_training_x'], self.params['n_training_v'], \
                    self.params['stim_range'][0], self.params['stim_range'][1])
        else:
            self.params['input_folder'] = "InputFilesTest_seed%d/" % (self.params['visual_stim_seed'])

        # if you want to store the input files in a subfolder of self.params['folder_name'], do this:
#        self.params['input_folder'] = "%sInputSpikeTrains/"   % self.params['folder_name']# folder containing the input spike trains for the network generated from a certain stimulus
        self.params['spiketimes_folder'] = "%sSpikes/" % self.params['folder_name']
        self.params['volt_folder'] = "%sVoltageTraces/" % self.params['folder_name']
        self.params['gsyn_folder'] = "%sCondTraces/" % self.params['folder_name']
        self.params['curr_folder'] = "%sCurrentTraces/" % self.params['folder_name']
        self.params['parameters_folder'] = "%sParameters/" % self.params['folder_name']
        self.params['connections_folder'] = "%sConnections/" % self.params['folder_name']
        self.params['figures_folder'] = "%sFigures/" % self.params['folder_name']
        self.params['movie_folder'] = "%sMovies/" % self.params['folder_name']
        self.params['tmp_folder'] = "%stmp/" % self.params['folder_name']
        self.params['data_folder'] = '%sData/' % (self.params['folder_name']) # for storage of analysis results etc
        # all folders to be created if not yet existing:
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['spiketimes_folder'], \
                            self.params['volt_folder'], \
                            self.params['gsyn_folder'], \
                            self.params['curr_folder'], \
                            self.params['parameters_folder'], \
                            self.params['connections_folder'], \
                            self.params['figures_folder'], \
                            self.params['movie_folder'], \
                            self.params['tmp_folder'], \
                            self.params['data_folder'], \
                            self.params['input_folder']] 

        self.params['params_fn_json'] = '%ssimulation_parameters.json' % (self.params['parameters_folder'])

        # input spiketrains
        self.params['merged_input_spiketrains_fn'] = "%sinput_spiketrain_merged.dat" % (self.params['input_folder'])
        self.params['input_st_fn_base'] = "%sstim_spike_train_" % self.params['input_folder']# input spike trains filename base
        self.params['input_rate_fn_base'] = "%srate_" % self.params['input_folder']# input spike trains filename base
        # for 'recorder neurons'
        self.params['recorder_neuron_input_fn_base'] = '%srecorder_neuron_input_spikes_' % (self.params['input_folder'])

        # output spiketrains
        self.params['exc_spiketimes_fn_base'] = '%sexc_spikes' % self.params['spiketimes_folder']
        self.params['exc_spiketimes_fn_merged'] = '%sexc_merged_spikes.dat' % self.params['spiketimes_folder']
        self.params['exc_nspikes_fn_merged'] = '%sexc_nspikes' % self.params['spiketimes_folder']
        self.params['exc_nspikes_nonzero_fn'] = '%sexc_nspikes_nonzero.dat' % self.params['spiketimes_folder']

        self.params['inh_spec_spiketimes_fn_base'] = '%sinh_spec_spikes' % self.params['spiketimes_folder']
        self.params['inh_spec_spiketimes_fn_merged'] = '%sinh_spec_merged_spikes.dat' % self.params['spiketimes_folder']
        self.params['inh_spec_nspikes_fn_merged'] = '%sinh_spec_nspikes' % self.params['spiketimes_folder']
        self.params['inh_spec_nspikes_nonzero_fn'] = '%sinh_spec_nspikes_nonzero.dat' % self.params['spiketimes_folder']

        self.params['inh_unspec_spiketimes_fn_base'] = '%sinh_unspec_spikes' % self.params['spiketimes_folder']
        self.params['inh_unspec_spiketimes_fn_merged'] = '%sinh_unspec_merged_spikes.dat' % self.params['spiketimes_folder']
        self.params['inh_unspec_nspikes_fn_merged'] = '%sinh_unspec_nspikes' % self.params['spiketimes_folder']
        self.params['inh_unspec_nspikes_nonzero_fn'] = '%sinh_unspec_nspikes_nonzero.dat' % self.params['spiketimes_folder']

        self.params['exc_volt_fn_base'] = '%sexc_volt' % self.params['volt_folder']
        self.params['exc_volt_anticipation'] = '%sexc_volt_anticipation.v' % self.params['volt_folder']
        self.params['exc_gsyn_anticipation'] = '%sexc_gsyn_anticipation.dat' % self.params['gsyn_folder']
        self.params['exc_curr_anticipation'] = '%sexc_curr_anticipation.dat' % self.params['curr_folder']
        self.params['inh_spec_volt_fn_base'] = '%sinh_spec_volt' % self.params['volt_folder']
        self.params['inh_spec_volt_anticipation'] = '%sinh_spec_volt_anticipation.v' % self.params['volt_folder']
        self.params['inh_spec_gsyn_anticipation'] = '%sinh_spec_gsyn_anticipation.dat' % self.params['gsyn_folder']
        self.params['inh_spec_curr_anticipation'] = '%sinh_spec_curr_anticipation.dat' % self.params['curr_folder']
        self.params['inh_unspec_volt_fn_base'] = '%sinh_unspec_volt' % self.params['volt_folder']
        self.params['inh_unspec_volt_anticipation'] = '%sinh_unspec_volt_anticipation.v' % self.params['volt_folder']
        self.params['inh_unspec_gsyn_anticipation'] = '%sinh_unspec_gsyn_anticipation.dat' % self.params['gsyn_folder']
        self.params['inh_unspec_curr_anticipation'] = '%sinh_unspec_curr_anticipation.dat' % self.params['curr_folder']
        self.params['rasterplot_exc_fig'] = '%srasterplot_exc.png' % (self.params['figures_folder'])
        self.params['rasterplot_inh_spec_fig'] = '%srasterplot_inh_spec.png' % (self.params['figures_folder'])
        self.params['rasterplot_inh_unspec_fig'] = '%srasterplot_inh_unspec.png' % (self.params['figures_folder'])

        # files for "recorder_neurons" recording the "free" membrane potential
        self.params['free_vmem_fn_base'] = 'free_vmem'
        self.params['recorder_neurons_gid_mapping'] = self.params['parameters_folder'] + 'recorder_neurons_gid_mapping.json'
        # file storing the gid and the PID 
        self.params['local_gids_fn_base'] = self.params['data_folder'] + 'local_gids_'
        self.params['local_gids_merged_fn'] = self.params['data_folder'] + 'merged_local_gids.json'
        # files storing the minicolumn and hypercolumn for the given GID
        self.params['gid_fn'] = self.params['parameters_folder'] + 'gids.json'

        # tuning properties and other cell parameter files
        self.params['tuning_prop_exc_fn'] = '%stuning_prop_exc.prm' % (self.params['parameters_folder']) # for excitatory cells
        self.params['tuning_prop_inh_fn'] = '%stuning_prop_inh.prm' % (self.params['parameters_folder']) # for inhibitory cells
        self.params['receptive_fields_exc_fn'] = self.params['parameters_folder'] + 'receptive_field_sizes_exc.txt'
        self.params['tuning_prop_fig_exc_fn'] = '%stuning_properties_exc.png' % (self.params['figures_folder'])
        self.params['tuning_prop_fig_inh_fn'] = '%stuning_properties_inh.png' % (self.params['figures_folder'])
        self.params['gids_to_record_fn'] = '%sgids_to_record.dat' % (self.params['parameters_folder'])
        self.params['all_predictor_params_fn'] = '%sall_predictor_params.dat' % (self.params['parameters_folder'])
        self.params['training_stimuli_fn'] = '%straining_stimuli.dat' % (self.params['parameters_folder']) # contains all training stimuli (not only those that have been trained in one simulation)
        self.params['training_stim_durations_fn'] = '%straining_stim_durations.dat' % (self.params['parameters_folder'])
        self.params['presented_stim_fn'] = '%spresented_stim_params.dat' % (self.params['data_folder']) # contains only those stimuli that have been presented
        self.params['test_sequence_fn'] = '%stest_sequence.dat' % (self.params['parameters_folder'])

        self.params['prediction_fig_fn_base'] = '%sprediction_' % (self.params['figures_folder'])

        # CONNECTION FILES
        self.params['weight_and_delay_fig'] = '%sweights_and_delays.png' % (self.params['figures_folder'])
        self.params['connection_matrix_fig'] = '%sconnection_matrix_' % (self.params['figures_folder'])

        # connection lists have the following format: src_gid  tgt_gid  weight  delay
        # E - E
        self.params['conn_list_ee_fn_base'] = '%sconn_list_ee_' % (self.params['connections_folder'])
        self.params['merged_conn_list_ee'] = '%smerged_conn_list_ee.dat' % (self.params['connections_folder'])
        # E - I
        self.params['conn_list_ei_fn_base'] = '%sconn_list_ei_' % (self.params['connections_folder'])
        self.params['merged_conn_list_ei'] = '%smerged_conn_list_ei.dat' % (self.params['connections_folder'])
        # I - E
        self.params['conn_list_ie_fn_base'] = '%sconn_list_ie_' % (self.params['connections_folder'])
        self.params['merged_conn_list_ie'] = '%smerged_conn_list_ie.dat' % (self.params['connections_folder'])
        # I - I
        self.params['conn_list_ii_fn_base'] = '%sconn_list_ii_' % (self.params['connections_folder'])
        self.params['merged_conn_list_ii'] = '%smerged_conn_list_ii.dat' % (self.params['connections_folder'])

        # used for different projections ['ee', 'ei', 'ie', 'ii'] for plotting
        # not needed anymore
#        self.params['conn_list_ee_global_fn_base'] = '%sconn_list_ee_' % (self.params['connections_folder']) 
        self.params['bias_ee_fn_base'] = '%sbias_ee_' % (self.params['connections_folder'])
        self.params['adj_list_tgt_fn_base'] = '%sadj_list_tgt_index_' % (self.params['connections_folder']) # key = target_gid
        self.params['adj_list_src_fn_base'] = '%sadj_list_src_index_' % (self.params['connections_folder']) # key = source_gid
        self.params['merged_adj_list_tgt_index'] = self.params['adj_list_tgt_fn_base'] + 'merged.json'
        self.params['merged_adj_list_src_index'] = self.params['adj_list_src_fn_base'] + 'merged.json'
        self.params['conn_mat_fn_base'] = '%sconn_mat_' % (self.params['connections_folder'])
        self.params['delay_mat_fn_base'] = '%sdelay_mat_' % (self.params['connections_folder'])

        # ANALYSIS RESULTS
        # these files receive the output folder when they are create / processed --> more suitable for parameter sweeps
        self.params['xdiff_vs_time_fn'] = 'xdiff_vs_time.dat'
        self.params['vdiff_vs_time_fn'] = 'vdiff_vs_time.dat'

#        self.create_folders()

    def check_folders(self):
        """
        Returns True if all folders exist, False otherwise
        """
        all_folders_exist = True
        for f in self.params['folder_names']:
            if not os.path.exists(f):
                all_folders_exist = False

        return all_folders_exist

    def create_folders(self):
        """
        Must be called from 'outside' this class before the simulation
        """

        for f in self.params['folder_names']:
            if not os.path.exists(f):
                print 'Creating folder:\t%s' % f
                os.system("mkdir -p %s" % (f))

    def load_params(self):
        """
        return the simulation parameters in a dictionary
        """
#        self.ParamSet = ntp.ParameterSet(self.params)
#        return self.ParamSet
        return self.params


    def update_values(self, kwargs):
        for key, value in kwargs.iteritems():
            self.params[key] = value
        # update the dependent parameters
        self.set_filenames()
#        self.ParamSet = ntp.ParameterSet(self.params)


    def write_parameters_to_file(self, fn=None, params_to_write=None):
        """
        This function must be called from 'outside' the class.
        Keyword arguments:
        fn -- (optional) target output filename for json dictionary
        params -- (optional) the modified parameter dictionary that is to write
        """
        if fn == None:
            fn = self.params['params_fn_json']
            print 'ParameterContainer.DEBUG Writing to the default params_fn_json:', fn
        if params_to_write == None:
            params_to_write = self.params
            print '\nDEBUG params_to_write is None\nParameterContainer.DEBUG params_to_write folder:', self.params['folder_name']
        self.create_folders()
        print 'Writing parameters to: %s' % (fn)
        output_file = file(fn, 'w')
        d = json.dump(params_to_write, output_file, indent=2)
        output_file.flush()
        output_file.close()



    def load_params_from_file(self, fn):
        """
        Loads the file via json from a filename
        Returns the simulation parameters stored in a file 
        Keyword arguments:
        fn -- file name
        """
        f = file(fn, 'r')
        print 'Loading parameters from', fn
        self.params = json.load(f)
        return self.params


