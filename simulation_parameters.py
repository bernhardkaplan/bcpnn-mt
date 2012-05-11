import numpy as np
import numpy.random as rnd
import os
"""

BK:
    TODO: use NeuroTools ParameterSets instead
"""

class parameter_storage(object):
    """
    This class contains the simulation parameters in a dictionary called params.
    """

    def __init__(self):

        self.params = {}

        self.params['simulator'] = 'nest'# number of minicolumns 

        # ###################
        # HEXGRID PARAMETERS
        # ###################

        self.params['N_RF'] = 25# np.int(n_cells/N_V/N_theta)
        # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"
        self.params['N_RF_X'] = np.int(np.sqrt(self.params['N_RF']*np.sqrt(3)))
        self.params['N_RF_Y'] = np.int(np.sqrt(self.params['N_RF']/np.sqrt(3)))
        self.params['N_V'], self.params['N_theta'] = 3, 4 # resolution in velocity norm and direction
        self.params['log_scale'] = 2. # base of the logarithmic tiling of particle_grid; linear if equal to one

        # ###################
        # NETWORK PARAMETERS
        # ###################
        self.params['n_mc'] = 1# number of minicolumns 
#        self.params['n_exc_per_mc' ] = 1024 # number of excitatory cells per minicolumn
        self.params['n_exc_per_mc'] = self.params['N_RF_X'] * self.params['N_RF_Y'] * self.params['N_V'] * self.params['N_theta'] # number of excitatory cells per minicolumn
        self.params['n_exc'] = self.params['n_mc'] * self.params['n_exc_per_mc']
        self.params['fraction_inh_cells'] = 0.25        # fraction of inhibitory cells in the network
        self.params['n_inh' ] = int(round(self.params['n_exc'] * self.params['fraction_inh_cells']))
        self.params['n_cells'] = self.params['n_mc'] * self.params['n_exc_per_mc'] + self.params['n_inh']

        # #######################
        # CONNECTIVITY PARAMETERS
        # #######################
        self.params['conn_mat_init_sparseness'] = 0.1   # sparseness of the initial connection matrix; 0.0 : no connections, 1.0 : full (all-to-all) connectivity
        # when the initial connections are derived on the cell's tuning properties, these two values are used
        self.params['w_init_thresh'] = 0.01     # if p <= this value -> no connection
        self.params['delay_scale'] = 1         # delays are computed based on the expected latency of the stimulus to reach to cells multiplied with this factor
        self.params['delay_min'] = 0.1         # delays are computed based on the expected latency of the stimulus to reach to cells multiplied with this factor
        self.params['delay_max'] = 20         # delays are computed based on the expected latency of the stimulus to reach to cells multiplied with this factor

        # exc - exc 
        self.params['p_ee'] = 0.5           # if two MCs are connected, cells within these MCs are connected with this probability (FixedProbabilityConnector)
        self.params['w_ee_mean'] = 0.001           # cells within two MCs are connected with this weight
        self.params['w_ee_sigma'] = 0.001           # cells within two MCs are connected with this weight

        # exc - inh
        self.params['p_ei'] = 0.25
        self.params['w_ei_mean'] = 0.001          
        self.params['w_ei_sigma'] = 0.001          

        # inh - exc
        self.params['p_ie'] = 0.25
        self.params['w_ie_mean'] = 0.001          
        self.params['w_ie_sigma'] = 0.001          

        # inh - inh
        self.params['p_ii'] = 1.0 
        self.params['w_ii_mean'] = 0.001          
        self.params['w_ii_sigma'] = 0.001          

        self.params['w_init_max'] = 0.001               # [nS] Maximal weight when creating the initial weight matrix
        self.params['w_thresh_bcpnn'] = 0.001           # [nS] Threshold under which a weight is set to zero in the cell-to-cell connectivity matrix
        self.params['dw_scale'] = 1e-4                 # [nS] Weight updates computed by BCPNN are scaled with this factor

        # ###################
        # CELL PARAMETERS   #
        # ###################
        # TODO: distribution of parameters (e.g. tau_m)
        self.params['cell_params_exc'] = {'tau_refrac':2.0, 'v_thresh':-50.0, 'tau_syn_E':2.0, 'tau_syn_I':10.0}
        self.params['cell_params_inh'] = {'tau_refrac':2.0, 'v_thresh':-50.0, 'tau_syn_E':2.0, 'tau_syn_I':10.0}
        # default parameters: /usr/local/lib/python2.6/dist-packages/pyNN/standardmodels/cells.py
        self.params['v_init'] = -65                 # [mV]
        self.params['v_init_sigma'] = 5             # [mV]

        # ######################
        # SIMULATION PARAMETERS 
        # ###################### 
        self.params['seed'] = 12345
        self.params['t_sim'] = 500.
        self.params['n_sim'] = 2                    # number of simulations (iterations) for learning

        # ######
        # INPUT 
        # ######
        self.params['f_max_stim'] = 50 * 100.       # [Hz]
        self.params['stim_dur_sigma'] = 300.        # [ms]
        self.params['w_input_exc'] = 0.005         # [nS] mean value for input stimulus ---< exc_units (columns
        self.params['w_input_exc_sigma'] = 0.1 * self.params['w_input_exc']  # [nS]


        # ###############
        # MOTION STIMULUS
        # ###############
        """
        x0 (y0) : start position on x-axis (y-axis)
        u0 (v0) : velocity in x-direction (y-direction)
        """
        self.params['motion_params'] = (0.5, 0.5, 1., 0) # x0, y0, u0, v0

        # ######
        # NOISE
        # ######
        self.params['w_exc_noise'] = 0.001          # [nS] mean value for noise ---< columns
        self.params['f_exc_noise'] = 400            # [Hz]
        self.params['w_inh_noise'] = 0.001          # [nS] mean value for noise ---< columns
        self.params['f_inh_noise'] = 400            # [Hz]



        # ######################
        # FILENAMES and FOLDERS
        # ######################
        self.params['folder_name'] = "NoColumns/"# the main folder with all simulation specific content
#        self.params['folder_name'] = "NoColumns_%d/" % (self.params['n_exc'])  # the main folder with all simulation specific content
        self.params['input_folder'] = "%sInputSpikeTrains/"   % self.params['folder_name']# folder containing the input spike trains for the network generated from a certain stimulus
        self.params['input_st_fn_base'] = "%sInputSpikeTrains/stim_spike_train_" % self.params['folder_name']# input spike trains filename base
        self.params['spiketimes_folder'] = "%sSpikes/" % self.params['folder_name']
        self.params['volt_folder'] = "%sVoltageTraces/" % self.params['folder_name']
        self.params['parameters_folder'] = "%sParameters/" % self.params['folder_name']
        self.params['connections_folder'] = "%sConnections/" % self.params['folder_name']
        self.params['weights_folder'] = "%sWeights/" % self.params['folder_name']
        self.params['bias_folder'] = "%sBias/" % self.params['folder_name']
        self.params['bcpnntrace_folder'] = "%sBcpnnTraces/" % self.params['folder_name']
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['spiketimes_folder'], \
                            self.params['volt_folder'], \
                            self.params['parameters_folder'], \
                            self.params['connections_folder'], \
                            self.params['weights_folder'], \
                            self.params['bias_folder'], \
                            self.params['bcpnntrace_folder'], \
                            self.params['input_folder']] # to be created if not yet existing

        self.params['exc_spiketimes_fn_merged'] = '%sexc_spikes_merged_' % self.params['spiketimes_folder']
        self.params['exc_spiketimes_fn_base'] = '%sexc_spikes_' % self.params['spiketimes_folder']
        self.params['merged_exc_spiketimes_fn_base'] = '%smerged_exc_spikes.ras' % self.params['spiketimes_folder']
        self.params['inh_spiketimes_fn_base'] = '%sinh_spikes_' % self.params['spiketimes_folder']
        self.params['merged_inh_spiketimes_fn_base'] = '%smerged_inh_spikes.ras' % self.params['spiketimes_folder']
        self.params['exc_volt_fn_base'] = '%sexc_volt' % self.params['volt_folder']
        self.params['inh_volt_fn_base'] = '%sinh_volt.npy' % self.params['volt_folder']
        self.params['ztrace_fn_base'] = '%sztrace_' % self.params['bcpnntrace_folder']
        self.params['etrace_fn_base'] = '%setrace_' % self.params['bcpnntrace_folder']
        self.params['ptrace_fn_base'] = '%sptrace_' % self.params['bcpnntrace_folder']

        self.params['tuning_prop_means_fn'] = '%stuning_prop_means.prm' % (self.params['parameters_folder'])
        self.params['tuning_prop_sigmas_fn'] = '%stuning_prop_sigmas.prm' % (self.params['parameters_folder'])

        self.params['bias_values_fn_base'] = '%sbias_values_' % (self.params['bias_folder'])

        # CONNECTION FILES
        # connection matrices are n x n matrices with weights as elements
        self.params['conn_mat_mc_mc_init'] = '%sconn_mat_init.npy' % (self.params['connections_folder'])

        self.params['conn_mat_mc_mc_fn_base'] = '%sconn_mat_mc_mc_' % (self.params['connections_folder'])
        self.params['conn_mat_ee_fn_base'] = '%sconn_mat_ee_' % (self.params['connections_folder'])
        # conn_mat_0_1 contains the cell-to-cell weights from minicolumn 0 to minicolumn 1

        # connection lists have the following format: src_gid  tgt_gid  weight  delay
        # for models not based on minicolumns:
        self.params['conn_list_ee_fn_base'] = '%sconn_list_ee_' % (self.params['connections_folder'])
        self.params['conn_list_ei_fn'] = '%sconn_list_ei.dat' % (self.params['connections_folder'])
        self.params['conn_list_ie_fn'] = '%sconn_list_ie.dat' % (self.params['connections_folder'])
        self.params['conn_list_ii_fn'] = '%sconn_list_ii.dat' % (self.params['connections_folder'])
        self.params['conn_list_input_fn'] = '%sconn_list_input.dat' % (self.params['connections_folder'])

        # BCPNN TRACES
        self.params['weights_fn_base'] = '%sweight_trace_' % (self.params['weights_folder'])
        self.params['bias_fn_base'] = '%sbias_trace_' % (self.params['bias_folder'])

        rnd.seed(self.params['seed'])
#        self.create_folders()


    def create_folders(self):
        # ---------------- WRITE ALL PARAMETERS TO FILE ----------- #
        for f in self.params['folder_names']:
            if not os.path.exists(f):
                print 'Creating folder:\t%s' % f
                os.system("mkdir %s" % (f))


    def load_params(self):
        """
        return the simulation parameters in a dictionary
        """
        return self.params


