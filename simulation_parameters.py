import numpy
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

        # ###################
        # NETWORK PARAMETERS
        # ###################
        self.params['n_mc' ] = 64       # number of minicolumns 
        self.params['n_exc_per_mc' ] = 1    # number of excitatory cells per minicolumn
        self.params['n_inh_per_mc' ] = 1           # number of excitatory cells per minicolumn
        self.params['n_cells'] = self.params['n_mc'] * (self.params['n_exc_per_mc'] + self.params['n_inh_per_mc'])
        self.params['conn_mat_init_sparseness'] = 0.1

        # ###################
        # CELL PARAMETERS   #
        # ###################
        self.params['cell_params'] = {'tau_refrac':2.0, 'v_thresh':-50.0, 'tau_syn_E':2.0, 'tau_syn_I':2.0}
        self.params['v_init'] = -65
        self.params['v_init_sigma'] = 5

        # ######################
        # SIMULATION PARAMETERS 
        # ###################### 
        self.params['seed'] = 12345
        self.params['t_sim'] = 1000.

        # ######
        # INPUT 
        # ######
        self.params['f_max_stim'] = 50 * 100.       # [Hz]
        self.params['stim_dur_sigma'] = 300.        # [ms]
        self.params['w_exc_input'] = 0.005           # [nS] mean value for input stimulus ---< columns
        self.params['w_exc_input_sigma'] = 0.1      # 


        # ###############
        # MOTION STIMULUS
        # ###############
        """
        x0 (y0) : start position on x-axis (y-axis)
        u0 (v0) : velocity in x-direction (y-direction)
        """
        self.params['motion_params'] = (0.5, 0, 1.5, 0) # x0, y0, u0, v0

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
        self.params['folder_name'] = "TestSim_n%d/" % self.params['n_mc']             # the main folder with all simulation specific content
        self.params['input_folder'] = "%sInputSpikeTrains/"   % self.params['folder_name']# folder containing the input spike trains for the network generated from a certain stimulus
        self.params['input_st_fn_base'] = "%sInputSpikeTrains/stim_spike_train_" % self.params['folder_name']# input spike trains filename base
        self.params['spiketimes_folder'] = "%sSpikes/" % self.params['folder_name']
        self.params['volt_folder'] = "%sVoltageTraces/" % self.params['folder_name']
        self.params['parameters_folder'] = "%sParameters/" % self.params['folder_name']
        self.params['connections_folder'] = "%sConnections/" % self.params['folder_name']
        self.params['weights_folder'] = "%sWeights/" % self.params['folder_name']
        self.params['bias_folder'] = "%sBias/" % self.params['folder_name']
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['spiketimes_folder'], \
                            self.params['volt_folder'], \
                            self.params['parameters_folder'], \
                            self.params['connections_folder'], \
                            self.params['weights_folder'], \
                            self.params['bias_folder'], \
                            self.params['input_folder']] # to be created if not yet existing

        self.params['exc_spiketimes_fn_base'] = '%sexc_spikes_' % self.params['spiketimes_folder']
        self.params['merged_exc_spiketimes_fn_base'] = '%smerged_exc_spikes.ras' % self.params['spiketimes_folder']
        self.params['inh_spiketimes_fn_base'] = '%sinh_spikes_' % self.params['spiketimes_folder']
        self.params['merged_inh_spiketimes_fn_base'] = '%smerged_inh_spikes.ras' % self.params['spiketimes_folder']
        self.params['exc_volt_fn_base'] = '%sexc_volt_' % self.params['volt_folder']
        self.params['inh_volt_fn_base'] = '%sinh_volt_' % self.params['volt_folder']

        self.params['tuning_prop_means_fn'] = '%stuning_prop_means.prm' % (self.params['parameters_folder'])
        self.params['tuning_prop_sigmas_fn'] = '%stuning_prop_sigmas.prm' % (self.params['parameters_folder'])

        # CONNECTION FILES
        self.params['conn_mat_init'] = '%sconn_mat_init.npy' % (self.params['connections_folder'])
        self.params['weights_fn_base'] = '%sweight_' % (self.params['weights_folder'])
        self.params['bias_fn_base'] = '%sbias_' % (self.params['bias_folder'])


        rnd.seed(self.params['seed'])
        self.create_folders()

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


