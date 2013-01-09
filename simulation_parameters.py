import numpy as np
import numpy.random as rnd
import os
from NeuroTools import parameters as ntp

class parameter_storage(object):
    """
    This class contains the simulation parameters in a dictionary called params.
    """

    def __init__(self):

        self.params = {}
        self.set_default_params()
        self.set_filenames()
        self.ParamSet = ntp.ParameterSet(self.params)

    def set_default_params(self):
        self.params['simulator'] = 'nest'
        self.params['abstract'] = False

        # ###################
        # HEXGRID PARAMETERS
        # ###################
        # Large-scale system
#        self.params['N_RF'] = 90# np.int(n_cells/N_V/N_theta)
#        self.params['N_RF_X'] = np.int(np.sqrt(self.params['N_RF']*np.sqrt(3)))
#        self.params['N_RF_Y'] = np.int(np.sqrt(self.params['N_RF']/np.sqrt(3))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"
#        self.params['N_V'], self.params['N_theta'] = 10, 10# resolution in velocity norm and direction

        # Medium-scale system
        self.params['N_RF'] = 70# np.int(n_cells/N_V/N_theta)
        self.params['N_RF_X'] = np.int(np.sqrt(self.params['N_RF']*np.sqrt(3)))
        self.params['N_RF_Y'] = np.int(np.sqrt(self.params['N_RF']/np.sqrt(3))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"
        self.params['N_V'], self.params['N_theta'] = 3, 8# resolution in velocity norm and direction

        # Minimum sized system
#        self.params['N_RF'] = 9# np.int(n_cells/N_V/N_theta)
#        self.params['N_RF_X'] = np.int(np.sqrt(self.params['N_RF']*np.sqrt(3)))
#        self.params['N_RF_Y'] = np.int(np.sqrt(self.params['N_RF']/np.sqrt(3))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"
#        self.params['N_V'], self.params['N_theta'] = 1, 1# resolution in velocity norm and direction

        # Single-speed
#        self.params['N_RF'] = 84# np.int(n_cells/N_V/N_theta)
#        self.params['N_RF_X'] = np.int(np.sqrt(self.params['N_RF']*np.sqrt(3)))
#        self.params['N_RF_Y'] = np.int(np.sqrt(self.params['N_RF']/np.sqrt(3))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"
#        self.params['N_V'], self.params['N_theta'] = 1, 16# resolution in velocity norm and direction

        # Tuning-properties spiking
#        self.params['N_RF'] = 20# np.int(n_cells/N_V/N_theta)
#        self.params['N_RF_X'] = np.int(np.sqrt(self.params['N_RF']*np.sqrt(3.)))
#        self.params['N_RF_Y'] = np.int(np.sqrt(self.params['N_RF']/np.sqrt(3.))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"
#        self.params['N_V'], self.params['N_theta'] = 2, 8# resolution in velocity norm and direction

        # Tuning-properties abstract
#        self.params['N_RF'] = 7# np.int(n_cells/N_V/N_theta)
#        self.params['N_RF_X'] = np.int(np.sqrt(self.params['N_RF']*np.sqrt(3.)))
#        self.params['N_RF_Y'] = np.int(np.sqrt(self.params['N_RF']/np.sqrt(3.))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"
#        self.params['N_V'], self.params['N_theta'] = 1, 8# resolution in velocity norm and direction


        print 'N_RF_X %d N_RF_Y %d' % (self.params['N_RF_X'], self.params['N_RF_Y'])
        print 'N_HC: %d   N_MC_PER_HC: %d' % (self.params['N_RF_X'] * self.params['N_RF_Y'], self.params['N_V'] * self.params['N_theta'])
        self.params['abstract_input_scaling_factor'] = 1.
        self.params['log_scale'] = 2. # bas4 of the logarithmic tiling of particle_grid; linear if equal to one
        self.params['sigma_RF_pos'] = .0 # some variability in the position of RFs
        self.params['sigma_RF_speed'] = .0 # some variability in the speed of RFs
        self.params['sigma_RF_direction'] = .0 * 2 * np.pi # some variability in the direction of RFs
        self.params['sigma_theta_training'] = 2 * np.pi * 0.00


        # ###################
        # NETWORK PARAMETERS
        # ###################
        self.params['n_mc'] = 1# number of minicolumns 
#        self.params['n_exc_per_mc' ] = 1024 # number of excitatory cells per minicolumn
        self.params['n_exc_per_mc'] = self.params['N_RF_X'] * self.params['N_RF_Y'] * self.params['N_V'] * self.params['N_theta'] # number of excitatory cells per minicolumn
        self.params['n_exc'] = self.params['n_mc'] * self.params['n_exc_per_mc']
        self.params['fraction_inh_cells'] = 0.20 # fraction of inhibitory cells in the network, only approximately!
        self.params['N_theta_inh'] = self.params['N_theta']
        self.params['N_V_INH'] = 2
        self.params['N_RF_INH'] = int(round(self.params['fraction_inh_cells'] * self.params['N_RF'] * float(self.params['N_V'] * self.params['N_theta']) / (self.params['N_V_INH'] * self.params['N_theta_inh'])))
        self.params['N_RF_X_INH'] = np.int(np.sqrt(self.params['N_RF_INH']*np.sqrt(3)))
        self.params['N_RF_Y_INH'] = np.int(np.sqrt(self.params['N_RF_INH']/np.sqrt(3))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"

        self.params['n_inh' ] = self.params['N_RF_X_INH'] * self.params['N_RF_Y_INH'] * self.params['N_theta_inh'] * self.params['N_V_INH']
#        self.params['n_inh' ] = int(round(self.params['n_exc'] * self.params['fraction_inh_cells']))
        self.params['n_cells'] = self.params['n_mc'] * self.params['n_exc_per_mc'] + self.params['n_inh']
        print 'n_cells: %d\tn_exc: %d\tn_inh: %d\nn_inh/n_exc = %.3f\tn_cells/n_inh = %.3f' % (self.params['n_cells'], self.params['n_exc'], self.params['n_inh'], \
                self.params['n_inh'] / float(self.params['n_exc']), self.params['n_inh'] / float(self.params['n_cells']))

        # #######################
        # CONNECTIVITY PARAMETERS
        # #######################
        self.params['connect_exc_exc'] = True# enable / disable exc - exc connections for test purpose only
        self.params['selective_inhibition'] = False# if True: inh cells have tuning prop and receive input from exc according to those
        # there are three different ways to set up the connections:
#        self.params['initial_connectivity'] = 'precomputed_linear_transform'
        self.params['initial_connectivity'] = 'precomputed_convergence_constrained'
#        self.params['initial_connectivity'] = 'random'
        self.params['p_ee'] = 0.03# fraction of network cells allowed to connect to each target cell, used in CreateConnections
        # when the initial connections are derived on the cell's tuning properties, these two values are used
        self.params['w_thresh_connection'] = 1e-5 # connections with a weight less then this value will be discarded
        self.params['delay_scale'] = 20.        # delays are computed based on the expected latency of the stimulus to reach to cells multiplied with this factor
        self.params['delay_range'] = (0.1, 200.)
        self.params['w_sigma_x'] = 0.40          # width of connectivity profile for pre-computed weights
        self.params['w_sigma_v'] = 0.40         # small w_sigma: tuning_properties get stronger weight when deciding on connection
                                                # large w_sigma: high connection probability (independent of tuning_properties)
                                                # small w_sigma_*: deviation from unaccelerated movements become less likely, straight line movements preferred
                                                # large w_sigma_*: broad (deviation from unaccelerated movements possible to predict)
        self.params['w_tgt_in'] = 0.25 # [uS]
        self.params['w_min'] = 5e-4             # When probabilities are transformed to weights, they are scaled so that the map into this range
        self.params['w_max'] = 4e-3
        self.params['n_src_cells_per_neuron'] = round(self.params['p_ee'] * self.params['n_exc'])

        # exc - inh
        self.params['p_ei'] = 0.05 #self.params['p_ee']
        self.params['w_ei_mean'] = 0.006
        self.params['w_ei_sigma'] = 0.001          

        # inh - exc
#        self.params['p_ie'] = 1.
        self.params['p_ie'] = 0.05 #self.params['p_ee']
        self.params['w_ie_mean'] = 0.006
        self.params['w_ie_sigma'] = 0.001          

        # inh - inh
        self.params['p_ii'] = self.params['p_ee']
        self.params['w_ii_mean'] = 0.003
        self.params['w_ii_sigma'] = 0.001          

        # ###################
        # CELL PARAMETERS   #
        # ###################
        # TODO: distribution of parameters (e.g. tau_m)
        self.params['cell_params_exc'] = {'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E':5.0, 'tau_syn_I':10.0, 'tau_m' : 10, 'v_reset' : -70, 'v_rest':-70}
        self.params['cell_params_inh'] = {'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E':5.0, 'tau_syn_I':10.0, 'tau_m' : 10, 'v_reset' : -70, 'v_rest':-70}
        self.params['tau_syn_exc'] = self.params['cell_params_exc']['tau_syn_E']
        self.params['tau_syn_inh'] = self.params['cell_params_inh']['tau_syn_I']
        # default parameters: /usr/local/lib/python2.6/dist-packages/pyNN/standardmodels/cells.py
        self.params['v_init'] = -65                 # [mV]
        self.params['v_init_sigma'] = 3             # [mV]

        # ######################
        # SIMULATION PARAMETERS 
        # ###################### 
        self.params['seed'] = 12345
        self.params['t_sim'] = 400.                 # [ms] total simulation time
        self.params['t_stimulus'] = 200.            # [ms] time when stimulus ends, 
        self.params['tuning_prop_seed'] = 0         # seed for randomized tuning properties
        self.params['input_spikes_seed'] = 0
        self.params['dt_sim'] = self.params['delay_range'][0] * 1 # [ms] time step for simulation
        if self.params['abstract']:
            self.params['dt_rate'] = 1.                # [ms] time step for the non-homogenous Poisson process 
        else:
            self.params['dt_rate'] = .1                # [ms] time step for the non-homogenous Poisson process 
        # 5.0 for abstract learning, 0.1 when used as envelope for poisson procees
        self.params['n_gids_to_record'] = 10

        # ###################
        # BCPNN PARAMS 
        # ###################
#        tau_p = self.params['t_sim'] * 0.33 # * self.n_stim# * self.n_cycles # tau_p should be in the order of t_stimulus * n_iterations * n_cycles
        tau_p = 2400 # * self.n_stim# * self.n_cycles # tau_p should be in the order of t_stimulus * n_iterations * n_cycles
        tau_pij = tau_p
        self.params['tau_dict'] = {'tau_zi' : 50.,    'tau_zj' : 5., 
                        'tau_ei' : 100.,   'tau_ej' : 100., 'tau_eij' : 100.,
                        'tau_pi' : tau_p,  'tau_pj' : tau_p, 'tau_pij' : tau_pij,
                        }

        # ######
        # INPUT 
        # ######
        self.params['f_max_stim'] = 5000. # [Hz]
        self.params['w_input_exc'] = 4.0e-3 # [uS] mean value for input stimulus ---< exc_units (columns

        # ###############
        # MOTION STIMULUS
        # ###############
        """
        x0 (y0) : start position on x-axis (y-axis)
        u0 (v0) : velocity in x-direction (y-direction)
        """
        self.params['motion_params'] = (0.1, 0.5, 0.2, 0) # x0, y0, u0, v0.5
        self.params['v_max_tp'] = 0.50  # [a.u.] maximal velocity in visual space for tuning_parameters (for each component), 1. means the whole visual field is traversed
        self.params['v_min_tp'] = 0.15  # [a.u.] minimal velocity in visual space for training
        self.params['v_max_training'] = 0.2
        self.params['v_min_training'] = 0.2
        self.params['blur_X'], self.params['blur_V'] = .10, .10

        # the blur parameter represents the input selectivity:
        # high blur means many cells respond to the stimulus
        # low blur means high input selectivity, few cells respond
        # the maximum number of spikes as response to the input alone is not much affected by the blur parameter

        # ###################
        # TRAINING PARAMETERS
        # ###################
        self.params['n_theta'] = 1 # number of different orientations to train with
        self.params['n_speeds'] = 1     
        self.params['n_cycles'] = 1
        self.params['n_stim_per_direction'] = 40 # each direction is trained this many times


        # ######
        # NOISE
        # ######
        self.params['w_exc_noise'] = 1e-3          # [uS] mean value for noise ---< columns
        self.params['f_exc_noise'] = 2000# [Hz] 
        self.params['w_inh_noise'] = 1e-3          # [uS] mean value for noise ---< columns
        self.params['f_inh_noise'] = 2000# [Hz]
#        self.params['w_exc_noise'] = 1e-8          # [uS] mean value for noise ---< columns
#        self.params['f_exc_noise'] = 1e-8# [Hz] 
#        self.params['w_inh_noise'] = 1e-8          # [uS] mean value for noise ---< columns
#        self.params['f_inh_noise'] = 1e-8# [Hz]

        rnd.seed(self.params['seed'])


    def set_filenames(self):
        # ######################
        # FILENAMES and FOLDERS
        # ######################
        # the main folder with all simulation specific content

#        folder_name = 'LargeScaleModel_'
        folder_name = 'SpikingModel_'
        if self.params['selective_inhibition']:
            folder_name += 'selectiveInh_'
        if self.params['connect_exc_exc']:
            if self.params['initial_connectivity'] == 'precomputed_linear_transform':
                folder_name += 'LT_'
            elif self.params['initial_connectivity'] == 'precomputed_convergence_constrained':
                folder_name += 'CC_'
            else:
                folder_name += 'rndConn_'
        else:
            folder_name += 'noRec_'
        folder_name += "delayScale%d_blurX%.2e_blurV%.2e_wsigmax%.2e_wsigmav%.2e_np8/" % \
                        (self.params['delay_scale'], self.params['blur_X'], self.params['blur_V'], self.params['w_sigma_x'], self.params['w_sigma_v'])

#        folder_name = 'LargeScaleModel_selectiveInh_LT_delayScale20_blurX1.50e-01_blurV3.50e-01_wsigmax3.00e-01_wsigmav3.00e-01/'
#        folder_name = 'SpikingModel/'
#        if self.params['abstract']:
#            folder_name = 'TuningCurvesAbstract/'
#            folder_name = 'Abstract_blurx%.2f_v%.2f/' % (self.params['blur_X'], self.params['blur_V'])
#            folder_name = 'Abstract_taupi%dms/' % (self.params['tau_dict']['tau_pi'])
#            folder_name = 'Abstract_c++/'
#            folder_name = 'Abstract_for_AndersCode_new/'
#            folder_name = 'AndersWij/'
#        else:
#            folder_name = 'SpikingModel/'
#            folder_name = 'InputAnalysis_SpikingModel_ScaledInput/'
		
        self.params['folder_name'] = folder_name 
        print 'Folder name:', self.params['folder_name']

        self.params['input_folder'] = "%sInputSpikeTrains/"   % self.params['folder_name']# folder containing the input spike trains for the network generated from a certain stimulus
#        self.params['input_folder'] = "InputSpikeTrains/"
        self.params['spiketimes_folder'] = "%sSpikes/" % self.params['folder_name']
        self.params['volt_folder'] = "%sVoltageTraces/" % self.params['folder_name']
        self.params['parameters_folder'] = "%sParameters/" % self.params['folder_name']
        self.params['connections_folder'] = "%sConnections/" % self.params['folder_name']
        self.params['activity_folder'] = "%sANNActivity/" % self.params['folder_name']
        self.params['weights_folder'] = "%sWeightsAndBias/" % self.params['folder_name']
        self.params['bias_folder'] = "%sBias/" % self.params['folder_name']
        self.params['bcpnntrace_folder'] = "%sBcpnnTraces/" % self.params['folder_name']
        self.params['figures_folder'] = "%sFigures/" % self.params['folder_name']
        self.params['movie_folder'] = "%sMovies/" % self.params['folder_name']
        self.params['tmp_folder'] = "%stmp/" % self.params['folder_name']
        self.params['training_input_folder'] = "%sTrainingInput/"   % self.params['folder_name']# folder containing the input spike trains for the network generated from a certain stimulus
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['spiketimes_folder'], \
                            self.params['volt_folder'], \
                            self.params['parameters_folder'], \
                            self.params['connections_folder'], \
                            self.params['activity_folder'], \
                            self.params['weights_folder'], \
                            self.params['bias_folder'], \
                            self.params['bcpnntrace_folder'], \
                            self.params['figures_folder'], \
                            self.params['movie_folder'], \
                            self.params['tmp_folder'], \
#                            self.params['training_input_folder'], \
                            self.params['input_folder']] # to be created if not yet existing

        self.params['params_fn'] = '%ssimulation_parameters.info' % (self.params['parameters_folder'])

        # input spiketrains
        self.params['input_st_fn_base'] = "%sstim_spike_train_" % self.params['input_folder']# input spike trains filename base
        self.params['input_rate_fn_base'] = "%srate_" % self.params['input_folder']# input spike trains filename base
        self.params['input_sum_fn'] = "%sinput_sum.dat" % (self.params['input_folder'])
        self.params['motion_fn'] = "%smotion_xy.dat" % self.params['input_folder']# input spike trains filename base
        self.params['input_fig_fn_base'] = "%sinputmap_" % self.params['figures_folder']# input spike trains filename base
        self.params['input_movie'] = "%sinputmap.mp4" % self.params['movie_folder']# input spike trains filename base

        # abstract input
        self.params['abstract_input_fn_base'] = 'abstract_input_'

        # output spiketrains
        self.params['exc_spiketimes_fn_base'] = '%sexc_spikes_' % self.params['spiketimes_folder']
        self.params['exc_spiketimes_fn_merged'] = '%sexc_spikes_merged_' % self.params['spiketimes_folder']
        self.params['inh_spiketimes_fn_base'] = '%sinh_spikes_' % self.params['spiketimes_folder']
        self.params['inh_spiketimes_fn_merged'] = '%sinh_spikes_merged_' % self.params['spiketimes_folder']
        self.params['exc_volt_fn_base'] = '%sexc_volt' % self.params['volt_folder']
        self.params['inh_volt_fn_base'] = '%sinh_volt' % self.params['volt_folder']
        self.params['ztrace_fn_base'] = '%sztrace_' % self.params['bcpnntrace_folder']
        self.params['etrace_fn_base'] = '%setrace_' % self.params['bcpnntrace_folder']
        self.params['ptrace_fn_base'] = '%sptrace_' % self.params['bcpnntrace_folder']
        self.params['rasterplot_exc_fig'] = '%srasterplot_exc.png' % (self.params['figures_folder'])
        self.params['rasterplot_inh_fig'] = '%srasterplot_inh.png' % (self.params['figures_folder'])

        # tuning properties and other cell parameter files
        self.params['tuning_prop_means_fn'] = '%stuning_prop_means.prm' % (self.params['parameters_folder']) # for excitatory cells
        self.params['tuning_prop_inh_fn'] = '%stuning_prop_inh.prm' % (self.params['parameters_folder']) # for inhibitory cells
        self.params['tuning_prop_sigmas_fn'] = '%stuning_prop_sigmas.prm' % (self.params['parameters_folder'])
        self.params['tuning_prop_fig_exc_fn'] = '%stuning_properties_exc.png' % (self.params['figures_folder'])
        self.params['tuning_prop_fig_inh_fn'] = '%stuning_properties_inh.png' % (self.params['figures_folder'])
        self.params['inh_cell_pos_fn'] = '%sinh_cell_positions.dat' % (self.params['parameters_folder'])
        self.params['gids_to_record_fn'] = '%sgids_to_record.dat' % (self.params['parameters_folder'])
        self.params['predicted_positions_fn'] = '%spredicted_positions.dat' % (self.params['parameters_folder'])
        self.params['x_distance_matrix_fn'] = '%sx_distance_matrix.dat' % (self.params['parameters_folder'])
        self.params['v_distance_matrix_fn'] = '%sv_distance_matrix.dat' % (self.params['parameters_folder'])
        self.params['tp_distance_matrix_fn'] = '%stp_distance_matrix.dat' % (self.params['parameters_folder'])
        self.params['input_params_fn'] = '%sinput_params.txt' % (self.params['parameters_folder'])

        self.params['bias_values_fn_base'] = '%sbias_values_' % (self.params['bias_folder'])

        # CONNECTION FILES
        self.params['weight_and_delay_fig'] = '%sweights_and_delays.png' % (self.params['figures_folder'])

        # connection lists have the following format: src_gid  tgt_gid  weight  delay
        # for models not based on minicolumns:
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

        # variations for different connectivity patterns 
        self.params['conn_list_ee_conv_constr_fn_base'] = '%sconn_list_ee_conv_constr_' % (self.params['connections_folder']) # convergence constrained, i.e. each cell gets limited input
        self.params['conn_list_ee_balanced_fn'] = '%sconn_list_ee_balanced.dat' % (self.params['connections_folder'])
        self.params['random_weight_list_fn']  = '%sconn_list_rnd_ee_' % (self.params['connections_folder'])

        # used for different projections ['ee', 'ei', 'ie', 'ii'] for plotting
        self.params['conn_mat_fn_base'] = '%sconn_mat_' % (self.params['connections_folder']) 
        self.params['delay_mat_fn_base'] = '%sdelay_mat_' % (self.params['connections_folder'])

        self.params['exc_inh_adjacency_list_fn'] = '%sexc_to_inh_indices.dat' % (self.params['connections_folder']) # row = target inh cell index, elements = exc source indices
        self.params['exc_inh_distances_fn'] = '%sexc_to_inh_distances.dat' % (self.params['connections_folder']) # file storing distances between the exc and inh cells, row = target inh cell index
        self.params['exc_inh_weights_fn'] = '%sexc_to_inh_weights.dat' % (self.params['connections_folder']) # same format as exc_inh_distances_fn, containing the exc - inh weights

        # for minicolumnar based units:
        self.params['conn_mat_ee_fn_base'] = '%sconn_mat_ee_' % (self.params['connections_folder'])
        # conn_mat_0_1 contains the cell-to-cell weights from minicolumn 0 to minicolumn 1

        # BCPNN TRACES
        self.params['weight_matrix_abstract'] = '%sweight_matrix_abstract.dat' % (self.params['weights_folder'])
        self.params['bias_matrix_abstract'] = '%sbias_matrix_abstract.dat' % (self.params['weights_folder'])
        self.params['weights_fn_base'] = '%sweight_trace_' % (self.params['weights_folder'])
        self.params['bias_fn_base'] = '%sbias_trace_' % (self.params['bias_folder'])


        # FIGURES
        self.params['spatial_readout_fn_base'] = '%sspatial_readout_' % (self.params['figures_folder'])
        self.params['spatial_readout_detailed_movie'] = '%sspatial_readout_detailed.mp4' % (self.params['movie_folder'])
        self.params['spatial_readout_movie'] = '%sspatial_readout.mp4' % (self.params['movie_folder'])
        self.params['prediction_fig_fn_base'] = '%sprediction_' % (self.params['figures_folder'])
        self.params['grouped_actitivty_fig_fn_base'] = '%sgrouped_activity_' % (self.params['figures_folder'])
        self.params['conductances_fig_fn_base'] = '%sconductance.png' % (self.params['figures_folder'])
        self.params['conductances_hist_fig_fn_base'] = '%sconductance_histogram_' % (self.params['figures_folder'])

        self.params['test'] = 'asdf'
        self.params['response_3d_fig'] = '%s3D_nspikes_winsum_dcellstim.png' % (self.params['figures_folder'])


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
                os.system("mkdir %s" % (f))

    def load_params(self):
        """
        return the simulation parameters in a dictionary
        """
        return self.ParamSet
#        return self.params


    def update_values(self, kwargs):
        for key, value in kwargs.iteritems():
#            print 'debug', key, value
            self.params[key] = value
        # update the dependent parameters
        self.set_filenames()
        self.ParamSet = ntp.ParameterSet(self.params)

    def write_parameters_to_file(self, fn=None):
        if fn == None:
            fn = self.params['params_fn']
        print 'Writing parameters to: %s' % (fn)

#        if not (os.path.isdir(self.params['folder_name'])):
#            print 'Creating folder:\n\t%s' % self.params['folder_name']
#            os.system('/bin/mkdir %s' % self.params['folder_name'])

        self.ParamSet.save(fn)
#        output_f = file(fn, 'w')
#        self.list_of_params = self.params.keys()
#        for p in self.params.keys():
#            if (type(p) == type([])):
#                string_to_write = ""
#                for i in p:
#                    string_to_write += str(p[i]) 
#                    string_to_write += '\t'
#                output_f.write('%s' % string_to_write)
#            else:
#                output_f.write('%s = %s\n' % (p, str(self.params.get(p))))
#        output_f.close()


class ParameterContainer(parameter_storage):

    def __init__(self, fn):
        super(ParameterContainer, self).__init__()
        self.root_dir = os.path.dirname(fn)
        # If the folder has been moved, all filenames need to be updated
        self.update_values({self.params['folder_name'] : self.root_dir})

    def load_params(self, fn):
        self.params = ntp.ParameterSet(fn)

    def update_values(self, kwargs):
        for key, value in kwargs.iteritems():
            self.params[key] = value
        # update the dependent parameters
        self.ParamSet = ntp.ParameterSet(self.params)

    def create_folders(self):
        """
        Must be called from 'outside' this class before the simulation
        """
        for f in self.params['folder_names']:
            if not os.path.exists(f):
                print 'Creating folder:\t%s' % f
                os.system("mkdir %s" % (f))

    def load_params(self):
        """
        return the simulation parameters in a dictionary
        """
        return self.ParamSet
#        return self.params


    def write_parameters_to_file(self, fn=None):
        if fn == None:
            fn = self.params['params_fn']
        print 'Writing parameters to: %s' % (fn)

#        if not (os.path.isdir(self.params['folder_name'])):
#            print 'Creating folder:\n\t%s' % self.params['folder_name']
#            os.system('/bin/mkdir %s' % self.params['folder_name'])

        self.ParamSet.save(fn)

