
import numpy as np
import numpy.random as nprnd
import sys
import os
import utils
import nest
import CreateStimuli
import json
import pylab
import re


class ToyExperiment(object):

    def __init__(self, params, output_folder, selected_gids):

        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.system('mkdir %s' % output_folder)
        nest.SetKernelStatus({'data_path': output_folder, 'resolution': .1, 'overwrite_files': True})
        self.selected_gids = selected_gids
        self.params = params

    def run_sim(self, t_sim):
        self.t_sim = t_sim
#        tp = np.loadtxt(self.params['tuning_prop_means_fn'])
#        tp_s = tp[self.selected_gids, :]
#        print 'tp_s', tp_s
        tp_s = np.array([[.3, .0, .5, .0, .0], \
                [.7, .0, .5, .0, .0]])


        # choose the motion - parameters for the test stimulus
        stim_id = 0
#        mp = np.loadtxt(self.params['training_sequence_fn'])[stim_id, :]  # (x, v)
        mp = [.2, .5]

        n_cells = len(self.selected_gids)
        time = np.arange(0, self.t_sim, self.params['dt_rate'])
        self.L_input = np.zeros((n_cells, time.shape[0]))  # the envelope of the Poisson process
        input_spiketrains = []
        
        load_input = False
        # get the input signal
        if not load_input:
            # create the input
            time = np.arange(0, self.t_sim, self.params['dt_rate'])
            idx_t_stop = int(self.t_sim / self.params['dt_rate'])
#            idx_t_stop = int(self.params['t_training_stim'] / self.params['dt_rate'])
            for i_time in xrange(0, idx_t_stop):
                time_ = (i_time * self.params['dt_rate']) / self.params['t_stimulus']
                x_stim = mp[0] + time_ * mp[1]
                self.L_input[:, i_time] = utils.get_input(tp_s, self.params, (x_stim, 0, mp[1], 0, 0))
                self.L_input[:, i_time] *= self.params['f_max_stim']
                if (i_time % 1000 == 0):
                    print "t: %.2f [ms]" % (time_ * self.params['t_stimulus'])

            for i_ in xrange(n_cells):
                rate_of_t = np.array(self.L_input[i_, :])
                n_steps = rate_of_t.size
                spike_times = []
                for i in xrange(n_steps):
                    r = nprnd.rand()
                    if (r <= ((rate_of_t[i]/1000.) * self.params['dt_rate'])): # rate is given in Hz -> 1/1000.
                        spike_times.append(i * self.params['dt_rate'])
                input_spiketrains.append(spike_times)
            

        else: 
            for i_, gid in enumerate(self.selected_gids):
                input_fn = self.params['input_st_fn_base'] + str(gid) + '.npy'
                input_spiketrains.append(np.load(input_fn))
                input_fn = self.params['input_rate_fn_base'] + str(gid) + '.npy'
                self.L_input[i_, :] = np.load(input_fn)


        
        # NEST - code 
        # Setup 
        self.params['w_input_exc'] = 1.  # [nS]
        nest.CopyModel('static_synapse', 'input_exc_0', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
        nest.CopyModel('static_synapse', 'input_exc_1', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 1})

        if (not 'bcpnn_synapse' in nest.Models('synapses')):
            nest.Install('pt_module')


        # Create cells
        cell_params = self.params['cell_params_exc'].copy()
        cells = nest.Create(self.params['neuron_model'], n_cells, params=cell_params)


        # Create stimulus
        stimulus = nest.Create('spike_generator', n_cells)
        for i_ in xrange(n_cells):
            spike_times = input_spiketrains[i_]
            nest.SetStatus([stimulus[i_]], {'spike_times' : spike_times})
            nest.Connect([stimulus[i_]], [cells[i_]], model='input_exc_0')
            nest.Connect([stimulus[i_]], [cells[i_]], model='input_exc_1')

        
        # Record
        voltmeter = nest.Create('voltmeter')
        nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'volt'}])
        exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
        nest.ConvergentConnect(cells, exc_spike_recorder)
        nest.ConvergentConnect(voltmeter, cells)

        
        # Connect cells in a chain:
        initial_weight = np.log(nest.GetDefaults('bcpnn_synapse')['p_ij']/(nest.GetDefaults('bcpnn_synapse')['p_i']*nest.GetDefaults('bcpnn_synapse')['p_j']))
        initial_bias = np.log(nest.GetDefaults('bcpnn_synapse')['p_j'])
        self.bcpnn_params = {'tau_i': 10., 'tau_j':10.0, 'tau_e': 100., 'tau_p': 1000.}
        self.syn_params = {'weight': initial_weight, 'bias': initial_bias, 'K': 1.0, 'delay': 1.0,\
                'tau_i': self.bcpnn_params['tau_i'], 'tau_j': self.bcpnn_params['tau_j'], 'tau_e': self.bcpnn_params['tau_e'], 'tau_p': self.bcpnn_params['tau_p']}
        for i_ in xrange(n_cells - 1):
            nest.Connect([cells[i_]], [cells[i_ + 1]], model='bcpnn_synapse')

            # modify the parameters
            nest.SetStatus(nest.GetConnections([cells[i_]], [cells[i_ + 1]]), self.syn_params)


        # Simulate
        nest.Simulate(self.t_sim)

        # Get weights
        for i_ in xrange(n_cells - 1):
            cp = nest.GetStatus(nest.GetConnections([cells[i_]], [cells[i_ + 1]]))[0]
            print 'cp:', cp
            w_ij = cp['weight']
            print 'Weight %d - %d after sim = %.5e'  % (i_, i_ + 1, w_ij)



    def plot_input(self, ax):

        time_axis = np.arange(0, self.t_sim, self.params['dt_rate'])
        plots = []
        for i_, gid in enumerate(self.selected_gids):
            p, = ax.plot(time_axis, self.L_input[i_, :], label='%d', lw=2)
            plots.append(p)
        ax.legend(plots, self.selected_gids)



    def plot_voltages(self, ax):
        to_match = 'volt-'
        for fn in os.listdir(self.output_folder):
            print 'fn:', fn
            m = re.match(to_match, fn)
            if m:
                volt_path = self.output_folder + fn

        d = np.loadtxt(volt_path)
        
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Voltage [mV]')
        gids = np.unique(d[:, 0])
        plots = []
        for gid in gids:
            time_axis, volt = utils.extract_trace(d, gid)
            p, = ax.plot(time_axis, volt, label='%d' % gid, lw=2)
            plots.append(p)
        ax.legend(plots, self.selected_gids)


    def get_bcpnn_traces(self, spike_train):

        # convert the spike train into an array of 0s and 1s
        n_bins = self.t_sim / self.params['dt_rate']
        s = utils.convert_spiketrain_to_trace(spike_train, n_bins)

        z = utils.low_pass_filter(s, tau=10, initial_value=0.001, dt=1., spike_height=1.)
        pass



if __name__ == '__main__':

    print '\nPlotting the default parameters give in simulation_parameters.py\n'
    import simulation_parameters
    ps = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = ps.load_params()                       # params stores cell numbers, etc as a dictionary


    if len(sys.argv) > 1:
        selected_gids = [int(sys.argv[i]) for i in xrange(1, 1 + len(sys.argv[1:]))]

    else:
        selected_gids = np.loadtxt(params['gids_to_record_fn'], dtype=int)[:2]

    print 'Selected gids:', selected_gids
    output_folder = 'ToyExperiment/'
    t_sim = 1500 
    TE = ToyExperiment(params, output_folder, selected_gids)

    TE.run_sim(t_sim)


    fig1 = pylab.figure()
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)
    TE.plot_input(ax1)
    TE.plot_voltages(ax2)


    pylab.show()
